"""Studio operations: every wizard step as a background job or a quick interactive call.

Frame indices coming from the UI are absolute (frames of the source video); the pipeline works
on the trimmed clip, so they are shifted by trim.start where masks and SCAIL-2 are involved.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import masks as mk
from ..config import load_config
from ..geometry import working_size
from ..video_io import read_frames
from .jobs import Job, JobManager
from .store import Project, ProjectStore, import_video
from .garment_bible import GarmentBibleStore, continuity_prompt
from .worker import ModelWorker
from .sam3_controls import effective, FIELDS, schema

log = logging.getLogger(__name__)

PRESETS = {
    # target_area = pixels of the SCAIL-2 working size; steps with the lightx2v LoRA
    "preview": {"target_area": 458752, "steps": 8},    # ~896x512
    "standard": {"target_area": 655360, "steps": 8},   # ~1088x608
    "high": {"target_area": 901120, "steps": 10, "segment_len": 33},      # ~1280x704, needs > 32 GB VRAM to be fast
}


def gen_output(p) -> str:
    """Delivered resolution of a project: original | 1080p | 4k."""
    return "original"


class ModelCache:
    """Keeps the small interactive models (click segmentation, person detection) loaded between
    clicks; heavy models live in the ModelWorker process."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.models = {}

    def get(self, kind: str):
        if kind not in self.models:
            if kind == "point":
                from ..segmentation import PointSegmenter
                self.models[kind] = PointSegmenter(self.cfg)
            elif kind == "detect":
                from ..segmentation import ImageSegmenter
                self.models[kind] = ImageSegmenter(self.cfg)
            else:
                raise KeyError(kind)
        return self.models[kind]

    def release(self, *kinds: str) -> None:
        for kind in kinds or list(self.models):
            model = self.models.pop(kind, None)
            if model is not None:
                model.close()


def mask_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    """Outlines (outer contours and holes) for SVG drawing with fill-rule evenodd."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.00005 * mask.size
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True) + 0.5, True)
        if len(contour) >= 3:
            polygons.append(contour.reshape(-1, 2).tolist())
    return polygons


class ScailProgress:
    """Turns SCAIL-2 runner output into job progress (0.02 .. 0.92 of the generate job)."""

    def __init__(self, job: Job):
        self.job = job
        self.segment, self.segments = 1, 1

    def __call__(self, line: str) -> None:
        job = self.job
        if "[animate]" in line:
            job.log(line)
            m = re.search(r"pose:(\d+)/(\d+)", line)
            if m:
                job.set(0.02 + 0.10 * int(m[1]) / max(1, int(m[2])), "preparing")
            elif "loading_pose" in line:
                job.set(0.02, "loading_pose")
            elif "loading_model" in line:
                job.set(0.25, "loading_model")
            elif "sampling:" in line:
                m = re.search(r"sampling:(\d+)/(\d+):(\d+)/(\d+)", line)
                if m:
                    fraction = (int(m[1])-1 + int(m[3])/max(1,int(m[4]))) / max(1,int(m[2]))
                    job.set(0.30 + 0.60 * min(1, fraction), line.split("] ",1)[-1])
            return
        if "[scail2]" in line:
            job.log(line.split("] ", 1)[-1][:300])
        m = re.search(r"Loaded (\d+)/(\d+) DiT", line)
        if m:
            job.set(0.15 + 0.2 * int(m.group(1)) / int(m.group(2)), "loading_model")
            return
        m = re.search(r"Processing segment (\d+)/(\d+)", line)
        if m:
            self.segment, self.segments = int(m.group(1)), int(m.group(2))
            return
        m = re.search(r"\|\s*(\d+)/(\d+) \[", line)
        if m and "it" in line:
            done, total = int(m.group(1)), int(m.group(2))
            frac = ((self.segment - 1) + done / max(total, 1)) / self.segments
            job.set(0.37 + 0.53 * frac, f"sampling:{self.segment}/{self.segments}:{done}/{total}")
            return
        if "umt5" in line and "loading" in line:
            job.set(0.04, "loading_text")
        elif "Wan2.1_VAE" in line or "models_clip" in line:
            job.set(0.12, "loading_vae")
        elif "Saved" in line:
            job.set(0.92, "decoding")
        elif "Traceback" in line or "Error" in line:
            job.log(line[:300])


class StudioService:
    def __init__(self, root: Path, config_path: str | Path | None):
        self.base_cfg = load_config(config_path)
        self.store = ProjectStore(root)
        self.bibles = GarmentBibleStore(root)
        self.models = ModelCache(self.base_cfg)
        self.worker = ModelWorker(self.base_cfg)
        self.jobs = JobManager()
        self._mask_cache: tuple | None = None
        self._packed_mask_cache: tuple | None = None

    def warmup(self) -> None:
        """Load the click-segmentation model in the background so the first click is instant."""
        try:
            with self.jobs.interactive():
                self.models.get("point")
            log.info("Click segmentation model ready")
        except Exception as exc:  # the UI still works; the model loads on first use
            log.warning("Warm-up skipped: %s", exc)

    # ------------------------------------------------------------------ helpers
    def cfg_for(self, p: Project) -> dict:
        s = p.state
        cfg = copy.deepcopy(self.base_cfg)
        cfg["sam3"].update(effective(cfg["sam3"], s.get("sam3_settings")))
        trim = s["trim"]
        cfg["video"]["start_frame"] = trim["start"]
        cfg["video"]["max_frames"] = trim["end"] - trim["start"]
        cfg["prompt"] = {**cfg["prompt"], "subject": s["prompt"].get("subject") or "A person",
                         "garment": s["garment"].get("description") or None,
                         "scene": s["prompt"].get("scene") or None}
        cfg["scail"]["prompt"] = s["prompt"].get("full") or None
        bible_id = s["garment"].get("bible_id")
        if bible_id:
            bible = self.bibles.get(bible_id)
            if s["garment"].get("bible_revision") != bible["revision"]:
                raise ValueError("Garment bible updated: reapply the latest revision and approve a new reference frame")
            bible_prompt = continuity_prompt(bible)
            cfg["garment_bible_views"] = [item["view"] for item in bible["images"]]
            cfg["prompt"]["garment"] = bible_prompt
            if cfg["scail"]["prompt"]:
                cfg["scail"]["prompt"] += " " + bible_prompt
        gen = s["generate"]
        preset = PRESETS.get(gen["preset"], PRESETS["preview"])
        cfg["scail"]["target_area"] = preset["target_area"]
        cfg["scail"]["segment_len"] = preset.get("segment_len", 81)
        cfg["scail"]["steps"] = int(gen.get("steps") or (cfg.get("animate", {}).get("steps", 20) if cfg.get("generation", {}).get("backend") == "wan_animate" else preset["steps"]))
        cfg["scail"]["seed"] = int(gen["seed"] if gen.get("seed") is not None else 42)
        relight = (cfg.get("studio") or {}).get("relight_lora")
        if cfg.get("generation", {}).get("backend") != "wan_animate" and gen.get("relight") and relight:
            cfg["scail"]["loras"] = list(cfg["scail"].get("loras") or []) + [{"path": relight, "alpha": 1.0}]
        cfg["upscale"]["backend"] = "resize"
        cfg["output"]["resolution"] = "original"
        cfg["mask_layers_enabled"] = bool(s["mask_layers"].get("ready") and s["mask_layers"].get("manual"))
        comp = s["composite"]
        cfg["composite"].update(mode=comp["mode"], feather=float(comp["feather"]), dilate=float(comp["dilate"]))
        return cfg

    def frame(self, p: Project, t: int) -> np.ndarray:
        path = p.dir / "frames" / f"{t + 1:06d}.jpg"
        # imdecode, not imread: OpenCV cannot open paths with non-ASCII (e.g. Persian) folder names
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path.exists() else None
        if image is None:
            raise KeyError(f"frame {t}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def exact_frame(self, p: Project, t: int) -> np.ndarray:
        return read_frames(p.source, t, 1)[0]

    def masks(self, p: Project, name: str = "actor") -> np.ndarray | None:
        path = p.dir / "01_masks" / f"{name}.npz"
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        cache = self._mask_cache if self._mask_cache and self._mask_cache[0] == p.id else (p.id, {})
        self._mask_cache = cache
        if name not in cache[1] or cache[1][name][0] != mtime:
            cache[1][name] = (mtime, mk.load_masks(path))
        return cache[1][name][1]

    def mask_frame(self, p: Project, name: str, rel: int) -> np.ndarray | None:
        """Unpack only one requested frame of a tracked mask for the UI.

        A 229-frame 4K boolean mask is ~1.8 GiB. Keep only its packed
        representation (~230 MiB) in the HTTP process while previewing.
        """
        if not hasattr(p, "dir"):
            # Minimal project stubs used by older callers and unit tests.
            masks = self.masks(p, name)
            return masks[rel] if masks is not None and 0 <= rel < len(masks) else None
        path = p.dir / "01_masks" / f"{name}.npz"
        if not path.is_file():
            return None
        mtime = path.stat().st_mtime_ns
        cache = (self._packed_mask_cache if self._packed_mask_cache
                 and self._packed_mask_cache[0] == p.id else (p.id, {}))
        self._packed_mask_cache = cache
        if name not in cache[1] or cache[1][name][0] != mtime:
            bits, width = mk.load_packed_masks(path)
            cache[1][name] = (mtime, bits, width)
        _, bits, width = cache[1][name]
        if not 0 <= rel < len(bits):
            return None
        return np.unpackbits(bits[rel], axis=-1, count=width).astype(bool)

    def content(self, p: Project) -> list[int]:
        """Picture box without letterbox bars (computed once; older projects get it on first use)."""
        box = p.state["video"].get("content_box")
        if not box:
            from .store import detect_content_box
            box = detect_content_box(p, p.state["video"]["width"])
            p.state["video"]["content_box"] = box
            p.save()
        return box

    def _rel(self, p: Project, t: int) -> int:
        trim = p.state["actor"].get("trim") or [p.state["trim"]["start"], p.state["trim"]["end"]]
        rel = int(t) - trim[0]
        if not 0 <= rel < trim[1] - trim[0]:
            raise ValueError("This frame is outside the trimmed clip")
        return rel

    # ------------------------------------------------------------------ projects
    def create_project(self, name: str, upload: Path) -> tuple[Project, Job]:
        p = self.store.create(name, upload)
        job = self.jobs.submit(p.id, "import", "import", lambda job: import_video(p), gpu=False)
        return p, job

    # ------------------------------------------------------------------ actor
    def detect_people(self, p: Project, t: int, text: str | None = None) -> list[dict]:
        """Instances matching a text prompt (default: every person). SAM3 only understands English."""
        text = (text or "").strip() or self.base_cfg["actor"]["prompt"]
        if re.search(r"[؀-ۿ]", text):
            raise ValueError("english_only")
        image = self.frame(p, t)
        with self.jobs.interactive():
            found = self.models.get("detect").segment(image, text)
        h, w = image.shape[:2]
        people = []
        for m in sorted(found, key=lambda m: -m.sum()):
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            people.append({"box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                           "area": float(m.sum()) / (h * w), "polygons": mask_polygons(m)})
        return people

    def isolated_frame(self, p: Project, t: int) -> np.ndarray:
        """One original-coordinate image with only the tracked person visible."""
        from ..person_parts import isolate_person
        if not p.state["actor"]["tracked"]:
            raise ValueError("Track the person first")
        actor = self.mask_frame(p, "actor", self._rel(p, t))
        if actor is None:
            raise ValueError("Person mask not found for this frame")
        frame = self.frame(p, t)
        return isolate_person(frame[None], actor[None], copy=False)[0]

    def segment_preview(self, p: Project, t: int, points, labels, box, candidate=None,
                        isolated: bool = False, text: str = "") -> dict:
        if candidate is not None:
            if p.state["keyframe"].get("chosen") != candidate:
                raise ValueError("Choose the reference image before segmenting")
            selected = next((c for c in p.state["keyframe"]["candidates"] if c["id"] == candidate), None)
            if selected is None or selected["frame"] != t:
                raise ValueError("Reference selection must use the selected keyframe")
            image = np.asarray(Image.open(p.path("02_keyframe", "candidates", candidate + ".jpg")).convert("RGB"))
        else:
            image = self.isolated_frame(p, t) if isolated else self.frame(p, t)
        text = text.strip()
        if text and (points or box is not None):
            raise ValueError("Choose text, points or box; do not mix text and spatial prompts")
        if not text and not points and box is None:
            raise ValueError("Enter a prompt, select a positive point or draw a box")
        if text and re.search(r"[؀-ۿ]", text):
            raise ValueError("english_only")
        with self.jobs.interactive():
            if text:
                found = self.models.get("detect").segment(image, text,
                    threshold=self.cfg_for(p)["sam3"]["score_threshold"],
                    mask_threshold=self.cfg_for(p)["sam3"]["image_mask_threshold"])
                mask = np.zeros(image.shape[:2], dtype=bool)
                for item in found:
                    mask |= item
            else:
                mask = self.models.get("point").segment(image, points or None, labels or None, box,
                    settings=self.cfg_for(p)["sam3"])
        if isolated:
            actor = self.mask_frame(p, "actor", self._rel(p, t))
            if actor is None:
                raise ValueError("Person mask not found for this frame")
            mask &= actor
        return {"polygons": mask_polygons(mask), "area": float(mask.mean())}

    def set_prompts(self, p: Project, prompts: list[dict]) -> None:
        clean = []
        for pr in prompts:
            points = [[float(x), float(y)] for x, y in pr.get("points") or []]
            labels = [int(v) for v in pr.get("labels") or [1] * len(points)]
            box = [float(v) for v in pr["box"]] if pr.get("box") else None
            if points or box:
                clean.append({"frame": int(pr["frame"]), "points": points, "labels": labels, "box": box})
        with p.lock:
            if clean != p.state["actor"]["prompts"]:
                p.state["actor"].update(prompts=clean, tracked=False, parts_ready=False, visible=[], trim=None)
                p.state["mask_layers"].update(ready=False, manual=False, tracked=False,
                                               object_signature="", formula_signature="")
                p.state["keyframe"].update(chosen=None, candidates=[], prompt="")
                if p.state.get("result"):
                    p.state["result"].update(stale=True, stale_reason="source")
                p.save()

    def get_sam3_settings(self, p: Project) -> dict:
        return {**schema(), "values": effective(self.base_cfg["sam3"], p.state.get("sam3_settings")),
                "overrides": dict(p.state.get("sam3_settings") or {})}

    def set_sam3_settings(self, p: Project, updates: dict) -> Project:
        """Update only approved fields; invalidate dependent masks, never silently reuse them."""
        from .sam3_controls import validate
        validate(updates, partial=True)
        before = effective(self.base_cfg["sam3"], p.state.get("sam3_settings"))
        after = {**before, **updates}
        validate(after)
        changed = {name for name in updates if before[name] != after[name]}
        if not changed:
            return p
        # Never allow model/control updates while the same project's worker is active.
        with p.lock:
            overrides = dict(p.state.get("sam3_settings") or {})
            overrides.update(updates)
            p.state["sam3_settings"] = overrides
            effects = {FIELDS[name][4] for name in changed}
            if "actor" in effects:
                p.state["actor"].update(tracked=False, parts_ready=False, visible=[], trim=None)
                effects.add("layers")
            if "layers" in effects or "compose" in effects:
                p.state["mask_layers"].update(ready=False, manual=False, tracked=False,
                    object_signature="", formula_signature="")
            if "actor" in effects or "layers" in effects or "compose" in effects:
                p.state["keyframe"].update(chosen=None, candidates=[], prompt="")
                if p.state.get("result"):
                    p.state["result"].update(stale=True, stale_reason="source")
            p.save()
        self._mask_cache = None
        self._packed_mask_cache = None
        return p

    def track(self, p: Project) -> Job:
        if not p.state["actor"]["prompts"]:
            raise ValueError("Select the person on at least one frame first")
        # Register the job BEFORE invalidating old results. If another task is
        # active, submit raises and leaves the current actor/masks untouched.
        # The gate prevents the new worker from seeing a half-updated project.
        state_ready = threading.Event()

        def run(job: Job):
            state_ready.wait()
            cfg = self.cfg_for(p)
            start, end = p.state["trim"]["start"], p.state["trim"]["end"]
            prompts = [{**pr, "frame": pr["frame"] - start} for pr in p.state["actor"]["prompts"]
                       if start <= pr["frame"] < end]
            if not prompts:
                raise ValueError("None of the selections is inside the trimmed range")
            paths = {"actor": str(p.path("01_masks", "actor.npz"))}
            res = self.worker.call("track", {"source": str(p.source), "start": start, "count": end - start,
                                             "prompts": prompts, "paths": paths, "sam3_settings": cfg["sam3"]}, job)
            n, h, w = res["shape"]
            ww, wh = working_size(w, h, cfg["scail"]["target_area"], cfg["scail"]["size_multiple"])
            meta = {"fps": p.state["video"]["fps"], "width": w, "height": h, "start_frame": start,
                    "num_frames": n, "work_width": ww, "work_height": wh, "content_box": self.content(p)}
            p.path("meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            visible = [start + i for i in res["visible"]]
            scores = np.asarray(res["scores"])
            suggestions, gap = [], max(1, n // 12)
            for i in np.argsort(-scores):
                if scores[i] <= 0 or len(suggestions) >= 6:
                    break
                if all(abs(int(i) + start - s["frame"]) >= gap for s in suggestions):
                    suggestions.append({"frame": int(i) + start, "score": round(float(scores[i]), 3)})
            p.update("actor", {"tracked": True, "parts_ready": False, "visible": visible, "trim": [start, end]})
            p.update("mask_layers", {"ready": False, "manual": False, "tracked": False,
                                     "object_signature": "", "formula_signature": ""})
            self._mask_cache = None
            self._packed_mask_cache = None
            kf = p.state["keyframe"]
            frame = kf["frame"] if kf["frame"] in visible else (suggestions[0]["frame"] if suggestions else None)
            p.update("keyframe", {"suggestions": suggestions, "frame": frame})
            return {"visible": len(visible), "frames": n}

        submitted = self.jobs.submit(p.id, "track", "track", run)
        try:
            p.update("actor", {"tracked": False, "parts_ready": False, "parts_quality": None})
            p.update("mask_layers", {"ready": False, "manual": False, "tracked": False,
                                     "object_signature": "", "formula_signature": ""})
            with getattr(p, "lock", nullcontext()):
                p.state.setdefault("keyframe", {}).update(chosen=None, candidates=[], prompt="")
                if p.state.get("result"):
                    p.state["result"].update(stale=True, stale_reason="source")
                if hasattr(p, "save"):
                    p.save()
        finally:
            state_ready.set()
        return submitted

    def separate_person_parts(self, p: Project) -> Job:
        """Legacy endpoint retained for old clients, but never auto-select head parts."""
        raise ValueError("Automatic face/hair selection was removed. Select any protected parts manually after tracking the actor.")

    def actor_outline(self, p: Project, t: int, part: str = "actor") -> dict:
        allowed = {"actor", "edit", "protect", "removed", "face", "hair"}
        if part.startswith("layer:"):
            layer = part.split(":", 1)[1]
            if layer not in {rule["name"] for rule in p.state["mask_layers"]["rules"]}:
                raise ValueError("Unknown manual mask layer")
            if not p.state["mask_layers"].get("ready"):
                return {"polygons": []}
            name = f"layers/{layer}"
        elif part in allowed:
            if part in ("edit", "protect", "removed") and not p.state["mask_layers"].get("ready"):
                return {"polygons": []}
            name = part
        else:
            raise ValueError("Unknown mask part")
        if not p.state["actor"].get("tracked"):
            return {"polygons": []}
        try:
            rel = self._rel(p, t)
        except ValueError:
            return {"polygons": []}
        if part == "removed":
            actor = self.mask_frame(p, "actor", rel)
            edit = self.mask_frame(p, "edit", rel)
            m = actor & ~edit if actor is not None and edit is not None else None
        else:
            m = self.mask_frame(p, name, rel)
        return {"polygons": mask_polygons(m) if m is not None else []}

    # ------------------------------------------------------------------ layered edit masks
    def _mask_object_signature(self, rules: list[dict]) -> str:
        items = [{"name": r["name"], "prompt": r.get("prompt", ""), "selection": r.get("selection")} for r in rules]
        return json.dumps(items, ensure_ascii=False, sort_keys=True)

    def _mask_formula_signature(self, rules: list[dict], base_actor: bool) -> str:
        items = {"base_actor": bool(base_actor),
                 "rules": [{"name": r["name"], "operation": r.get("operation", "ignore")} for r in rules]}
        return json.dumps(items, ensure_ascii=False, sort_keys=True)

    def set_mask_layers(self, p: Project, rules: list[dict], base_actor: bool = True) -> Project:
        if not p.state["actor"].get("tracked"):
            raise ValueError("Track the whole actor before selecting additional masks")
        clean = []
        for rule in rules:
            name = (rule.get("name") or "").strip().lower()
            prompt = (rule.get("prompt") or "").strip()
            op = rule.get("operation") or "ignore"
            selection = rule.get("selection")
            if not name or op not in ("add", "subtract", "ignore") or not selection:
                raise ValueError("Every mask object requires a manual selection on the tracked person")
            t = int(selection.get("frame", -1))
            start, end = p.state["trim"]["start"], p.state["trim"]["end"]
            if not start <= t < end:
                raise ValueError("Mask selection frame must be inside video trim")
            points = selection.get("points") or []
            labels = selection.get("labels") or []
            box = selection.get("box")
            text = str(selection.get("text") or prompt or "").strip()
            if text and (points or box is not None):
                raise ValueError("Choose text or spatial prompts for each mask")
            if not points and box is None and not text:
                raise ValueError("Image selection needs a prompt, positive point or box")
            if points and not any(v == 1 for v in labels) and box is None:
                raise ValueError("Select at least one positive point")
            prompt = text
            if len(points) != len(labels) or any(len(pt) != 2 for pt in points) or (box is not None and len(box) != 4):
                raise ValueError("Invalid image selection points or box")
            candidate = selection.get("candidate")
            if candidate:
                raise ValueError("Manual mask must be selected on the isolated original actor, not a generated reference")
            selection = {"frame": t, "points": [[float(v) for v in pt] for pt in points],
                         "labels": [int(v) for v in labels],
                         "box": [float(v) for v in box] if box else None,
                         "candidate": candidate, "text": text}
            if any(v not in (0, 1) for v in selection["labels"]):
                raise ValueError("Image selection labels must be 0 or 1")
            if re.search(r"[؀-ۿ]", prompt):
                raise ValueError("english_only")
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
                raise ValueError("Layer names must be lowercase letters, digits or underscores")
            if name in (r["name"] for r in clean):
                raise ValueError("Duplicate mask layer name")
            clean.append({"name": name, "prompt": prompt, "operation": op, "selection": selection})
        if len(clean) > 12:
            raise ValueError("Maximum 12 layers")

        mask_state = p.state.get("mask_layers") or {}
        old_object_sig = mask_state.get("object_signature", "")
        old_formula_sig = mask_state.get("formula_signature", "")
        object_sig = self._mask_object_signature(clean)
        formula_sig = self._mask_formula_signature(clean, base_actor)
        tracked = bool(mask_state.get("tracked") and old_object_sig == object_sig)
        ready = bool(mask_state.get("ready") and tracked and old_formula_sig == formula_sig)
        p.update("mask_layers", {"rules": clean, "ready": ready, "manual": False if not ready else bool(mask_state.get("manual")),
                                 "tracked": tracked, "base_actor": bool(base_actor),
                                 "object_signature": object_sig, "formula_signature": formula_sig})
        if not ready:
            with getattr(p, "lock", nullcontext()):
                p.state.setdefault("keyframe", {}).update(chosen=None, candidates=[], prompt="")
                if p.state.get("result"):
                    p.state["result"].update(stale=True, stale_reason="source")
                if hasattr(p, "save"):
                    p.save()
        self._mask_cache = None
        self._packed_mask_cache = None
        return p

    def build_mask_layers(self, p: Project) -> Job:
        if not p.state["actor"].get("tracked"):
            raise ValueError("Track the person first")
        rules = p.state["mask_layers"]["rules"]
        start, end = p.state["trim"]["start"], p.state["trim"]["end"]

        def run(job: Job):
            actor_file = p.path("01_masks", "actor.npz")
            if not actor_file.is_file() or p.state["actor"].get("trim") != [start, end]:
                raise ValueError("Actor tracking must match the current trim")
            self._mask_cache = None
            self._packed_mask_cache = None
            object_sig = self._mask_object_signature(rules)
            formula_sig = self._mask_formula_signature(rules, p.state["mask_layers"].get("base_actor", True))
            tracked_ok = bool(p.state["mask_layers"].get("tracked") and p.state["mask_layers"].get("object_signature") == object_sig)
            task = "compose_mask_layers" if tracked_ok else "mask_layers"
            result = self.worker.call(task, {
                "source": str(p.source), "start": start, "count": end - start,
                "rules": rules, "base_actor": bool(p.state["mask_layers"].get("base_actor", True)),
                "part_protection_padding_frac": self.cfg_for(p)["sam3"]["part_protection_padding_frac"],
                "actor_path": str(actor_file),
                "output_dir": str(p.path("01_masks", "layers", ".keep").parent),
                "edit_path": str(p.path("01_masks", "edit.npz")),
                "protect_path": str(p.path("01_masks", "protect.npz")), "sam3_settings": self.cfg_for(p)["sam3"]}, job)
            p.update("mask_layers", {"ready": True, "manual": True, "tracked": True,
                                     "base_actor": bool(p.state["mask_layers"].get("base_actor", True)),
                                     "object_signature": object_sig, "formula_signature": formula_sig})
            self._mask_cache = None
            self._packed_mask_cache = None
            return result

        return self.jobs.submit(p.id, "mask_layers", "mask_layers", run)

    def compose_mask_layers(self, p: Project) -> Job:
        if not p.state["actor"].get("tracked"):
            raise ValueError("Track the person first")
        if not p.state["mask_layers"].get("tracked"):
            raise ValueError("Track the mask objects at least once before recomposing")
        start, end = p.state["trim"]["start"], p.state["trim"]["end"]
        rules = p.state["mask_layers"]["rules"]

        def run(job: Job):
            actor_file = p.path("01_masks", "actor.npz")
            result = self.worker.call("compose_mask_layers", {
                "source": str(p.source), "start": start, "count": end - start,
                "rules": rules, "base_actor": bool(p.state["mask_layers"].get("base_actor", True)),
                "part_protection_padding_frac": self.cfg_for(p)["sam3"]["part_protection_padding_frac"],
                "actor_path": str(actor_file),
                "output_dir": str(p.path("01_masks", "layers", ".keep").parent),
                "edit_path": str(p.path("01_masks", "edit.npz")),
                "protect_path": str(p.path("01_masks", "protect.npz")), "sam3_settings": self.cfg_for(p)["sam3"]}, job)
            p.update("mask_layers", {"ready": True, "manual": True, "tracked": True,
                                     "base_actor": bool(p.state["mask_layers"].get("base_actor", True)),
                                     "object_signature": self._mask_object_signature(rules),
                                     "formula_signature": self._mask_formula_signature(rules, p.state["mask_layers"].get("base_actor", True))})
            self._mask_cache = None
            self._packed_mask_cache = None
            return result

        return self.jobs.submit(p.id, "mask_layers", "mask_layers", run)
    # ------------------------------------------------------------------ garment
    def add_garment(self, p: Project, upload: Path) -> str:
        name = f"{uuid.uuid4().hex[:8]}.png"
        from PIL import ImageOps, UnidentifiedImageError
        try:
            with Image.open(upload) as source:
                if source.width * source.height > 80_000_000:
                    raise ValueError("Garment image is too large")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                image.save(p.path("garments", name))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid garment image") from exc
        with p.lock:
            p.state["garment"]["images"].append(name)
            p.state["keyframe"].update(chosen=None, candidates=[], prompt="")
            if p.state.get("result"):
                p.state["result"].update(stale=True, stale_reason="source")
            p.save()
        return name

    def remove_garment(self, p: Project, name: str) -> None:
        if (name not in p.state["garment"]["images"] or
                not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}\.png", name) or name.startswith(".")):
            raise ValueError("Unknown garment image")
        with p.lock:
            p.state["garment"]["images"].remove(name)
            p.state["keyframe"].update(chosen=None, candidates=[], prompt="")
            if p.state.get("result"):
                p.state["result"].update(stale=True, stale_reason="source")
            p.save()
        (p.dir / "garments" / name).unlink(missing_ok=True)

    # ------------------------------------------------------------------ keyframe
    def _keyframe_inputs(self, p: Project, t: int):
        if not p.state["actor"]["tracked"]:
            raise ValueError("Track the actor first")
        rel = self._rel(p, t)
        actor_mask = self.mask_frame(p, "actor", rel)
        if actor_mask is None:
            raise ValueError("Tracked actor masks are missing; track the person again")
        actor = actor_mask.copy()
        if not actor.any():
            raise ValueError("The actor is not visible on this frame; pick another one")
        if not (p.state["mask_layers"].get("ready") and p.state["mask_layers"].get("manual")):
            raise ValueError("Confirm the manual positive/negative masks first")
        protect = self.mask_frame(p, "protect", rel)
        return rel, actor, (protect.copy() if protect is not None else np.zeros_like(actor))

    def _add_candidate(self, p: Project, t: int, image: np.ndarray, mask: np.ndarray, **info) -> dict:
        cid = uuid.uuid4().hex[:8]
        Image.fromarray(image).save(p.path("02_keyframe", "candidates", f"{cid}.jpg"), quality=95)
        mk.save_masks(p.path("02_keyframe", "candidates", f"{cid}_mask.npz"), mask[None])
        cand = {"id": cid, "frame": int(t), "created": time.time(),
                "bible_id": p.state["garment"].get("bible_id"),
                "bible_revision": p.state["garment"].get("bible_revision"), **info}
        with p.lock:
            p.state["keyframe"]["candidates"].append(cand)
            p.save()
        return cand

    def edit_instruction(self, p: Project, extra: str | None = None) -> str:
        """The built-in editor instruction for this project, with the user's extra text appended."""
        from ..keyframe import edit_prompt
        return edit_prompt(self.cfg_for(p), len(p.state["garment"]["images"][:2]),
                           p.state["garment"].get("description"), extra)

    def generate_keyframes(self, p: Project, t: int, count: int, extra: str | None, seed: int | None) -> Job:
        """Variants of frame t in the new garment. `extra` only adds to the built-in instruction."""
        from ..tracks import mask_iou

        rel, actor, face = self._keyframe_inputs(p, t)
        garments = [Image.open(p.dir / "garments" / n).convert("RGB") for n in p.state["garment"]["images"][:2]]
        if extra and re.search(r"[؀-ۿ]", extra):
            raise ValueError("english_only")
        text = self.edit_instruction(p, extra)
        count = max(1, min(int(count), 6))
        seed = int(seed) if seed is not None else int(time.time()) % 100000
        content = self.content(p)

        def run(job: Job):
            cfg = self.cfg_for(p)
            seeds = [seed + k for k in range(count)]
            results = self.worker.call("variants", {
                "frame": self.exact_frame(p, t), "actor": actor, "face": face,
                "garments": [np.asarray(g) for g in garments], "prompt": text, "content": content,
                "margin": cfg["keyframe"]["crop_margin"], "seeds": seeds}, job)
            # how well the edited person still covers the original silhouette (low = re-posed)
            made = [self._add_candidate(p, t, image, mask, seed=s, prompt=text, source="ai",
                                        match=round(mask_iou(mask, actor), 3))["id"]
                    for s, (image, mask) in zip(seeds, results)]
            p.update("keyframe", {"prompt": text, "extra": extra or ""})
            return {"candidates": made}

        return self.jobs.submit(p.id, "keyframe", "keyframe", run)

    def upload_keyframe(self, p: Project, t: int, upload: Path) -> Job:
        from ..tracks import mask_iou

        rel, actor, face = self._keyframe_inputs(p, t)
        w, h = p.state["video"]["width"], p.state["video"]["height"]
        image = np.asarray(Image.open(upload).convert("RGB").resize((w, h), Image.LANCZOS))
        content = self.content(p)

        def run(job: Job):
            key, mask = self.worker.call("finish_upload", {"image": image, "actor": actor, "face": face,
                                                           "content": content}, job)
            cand = self._add_candidate(p, t, key, mask, source="upload", match=round(mask_iou(mask, actor), 3))
            return {"candidates": [cand["id"]]}

        return self.jobs.submit(p.id, "keyframe", "keyframe", run)

    def choose_keyframe(self, p: Project, cid: str) -> None:
        cand = next((c for c in p.state["keyframe"]["candidates"] if c["id"] == cid), None)
        if cand is None:
            raise KeyError(cid)
        if p.state["garment"].get("bible_id") and (
            cand.get("bible_id") != p.state["garment"].get("bible_id") or
            cand.get("bible_revision") != p.state["garment"].get("bible_revision")
        ):
            raise ValueError("This keyframe belongs to an earlier garment bible revision; generate a new one")
        rel = self._rel(p, cand["frame"])
        folder = p.dir / "02_keyframe"
        Image.open(folder / "candidates" / f"{cid}.jpg").convert("RGB").save(folder / "keyframe_0.png")
        shutil.copy(folder / "candidates" / f"{cid}_mask.npz", folder / "ref_mask_0.npz")
        Image.fromarray(self.exact_frame(p, cand["frame"])).save(folder / "original_0.png")
        (folder / "keyframes.json").write_text(json.dumps({"indices": [rel]}), encoding="utf-8")
        # Mask layers were picked on the original tracked actor, not on an
        # edited reference candidate. Choosing a reference must not invalidate them.
        p.update("keyframe", {"chosen": cid, "frame": cand["frame"]})
        if p.state.get("result"):
            with p.lock:
                p.state["result"].update(stale=True, stale_reason="source")
                p.save()

    def delete_candidate(self, p: Project, cid: str) -> None:
        with p.lock:
            kf = p.state["keyframe"]
            kf["candidates"] = [c for c in kf["candidates"] if c["id"] != cid]
            if kf["chosen"] == cid:
                kf["chosen"] = None
                if p.state.get("result"):
                    p.state["result"].update(stale=True, stale_reason="source")
            p.save()
        for suffix in (".jpg", "_mask.npz"):
            (p.dir / "02_keyframe" / "candidates" / f"{cid}{suffix}").unlink(missing_ok=True)

    # ------------------------------------------------------------------ video
    def _pipeline(self, p: Project, cfg: dict):
        from ..pipeline import Pipeline
        # An interrupted render must not corrupt the last successful result.
        return Pipeline(cfg, p.source, p.dir / "result.pending.mp4", p.dir)

    def generate(self, p: Project) -> Job:
        if self.base_cfg.get("generation", {}).get("backend") == "wan_animate":
            from ..animate_stage import resolve_models
            try:
                resolve_models(self.base_cfg)
            except (FileNotFoundError, OSError) as exc:
                raise ValueError("Wan Animate: model download is not complete yet") from exc
        if not p.state["actor"]["tracked"]:
            raise ValueError("Track the actor first")
        if not (p.state["mask_layers"].get("ready") and p.state["mask_layers"].get("manual")):
            raise ValueError("Confirm the whole-person edit mask and any manually protected parts before generation")
        if not p.state["keyframe"]["chosen"]:
            raise ValueError("Choose a keyframe first")

        def run(job: Job):
            # SCAIL-2 needs nearly all RAM and VRAM: stop the model worker and drop the click models
            self.worker.stop()
            self.models.release()
            cfg = self.cfg_for(p)
            # Studio compares source/result interactively; encoding a separate side-by-side
            # preview repeats a full H.264 pass and is not used by the web UI.
            cfg["output"]["preview"] = False
            meta_path = p.dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["work_width"], meta["work_height"] = working_size(
                meta["width"], meta["height"], cfg["scail"]["target_area"], cfg["scail"]["size_multiple"])
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            pipe = self._pipeline(p, cfg)
            pipe.on_scail_line = ScailProgress(job)
            def started(proc):
                job.process = proc
                job.check()  # cancellation between phases must stop the newly spawned child too
            pipe.on_scail_start = started
            job.set(0.01, "preparing")
            pipe.stage_scail()
            big = (gen_output(p) != "original")  # the SeedVR2 pass takes a large share of the time
            job.set(0.80 if big else 0.93, "upscaling")
            pipe.stage_upscale()
            job.set(0.82 if big else 0.97, "upscaling_4k" if big else "compositing")
            pipe.stage_composite()
            (p.dir / "result.pending.mp4").replace(p.dir / "result.mp4")
            p.update("result", {"file": "result.mp4", "updated": time.time(),
                                "preset": p.state["generate"]["preset"],
                                "size": [meta["work_width"], meta["work_height"]], "stale": False, "stale_reason": None})
            return {"result": "result.mp4"}

        return self.jobs.submit(p.id, "generate", "generate", run)

    def recomposite(self, p: Project) -> Job:
        if not p.state.get("result") or not (p.dir / "04_upscale" / "generated_full.mp4").exists():
            raise ValueError("Generate the video first")
        if p.state["result"].get("stale") and p.state["result"].get("stale_reason") != "composite":
            raise ValueError("The actor, garment, reference frame or generation settings changed; generate the video again")

        big = gen_output(p) != "original"

        def run(job: Job):
            job.set(0.1, "upscaling_4k" if big else "compositing")
            if big:
                self.worker.stop()  # SeedVR2 needs the GPU to itself
                self.models.release()
            cfg = self.cfg_for(p)
            cfg["output"]["preview"] = False
            self._pipeline(p, cfg).stage_composite()
            (p.dir / "result.pending.mp4").replace(p.dir / "result.mp4")
            p.update("result", {"file": "result.mp4", "updated": time.time(), "stale": False, "stale_reason": None})
            return {"result": "result.mp4"}

        return self.jobs.submit(p.id, "composite", "composite", run, gpu=big)
