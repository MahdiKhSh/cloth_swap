"""Heavy model work (SAM3 video tracking, Qwen-Image-Edit) in a separate process.

Freeing a 20 GB model inside a long-running Python process does not reliably give the RAM back,
and SCAIL-2 needs nearly all of it. Running these models in a child process that is stopped
before SCAIL-2 starts guarantees the memory is returned. The child keeps its models loaded
between calls (e.g. several rounds of keyframe variants without reloading Qwen).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import traceback

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- child side
def _track(cache: dict, cfg: dict, a: dict, progress) -> dict:
    """Stage 1: track ONLY the chosen person, not independent face/hair concepts."""
    from .. import masks as mk
    from ..keyframe import frame_scores
    from ..segmentation import track_prompts
    from ..tracks import drop_occluded
    from ..video_io import read_frames

    progress(0.01, "reading_video")
    frames = read_frames(a["source"], a["start"], a["count"])
    progress(0.03, "tracking_person")
    actor = track_prompts(frames, a["prompts"], cfg,
                          progress=lambda d, n: progress(0.05 + 0.85 * d / n, f"tracking_person:{d}/{n}"),
                          model_cache=cache)
    actor = drop_occluded(actor,
                          min_iou=cfg["sam3"]["occlusion_min_iou"],
                          min_area=cfg["sam3"]["occlusion_min_area"])
    if not actor.any():
        raise ValueError("SAM3 did not track the selected person")
    mk.save_masks(a["paths"]["actor"], actor)
    progress(0.95, "person_ready")
    n = len(frames)
    return {"visible": np.flatnonzero(actor.reshape(n, -1).any(axis=1)).tolist(),
            "scores": frame_scores(frames, actor, np.zeros_like(actor)).tolist(),
            "shape": list(frames.shape[:3])}


def _person_parts(cache: dict, cfg: dict, a: dict, progress) -> dict:
    """Stage 2: run face/hair SAM3 on the ALREADY segmented person video."""
    from .. import masks as mk
    from ..person_parts import derive_edit_masks, isolate_person
    from ..segmentation import track_concepts
    from ..tracks import masks_within
    from ..video_io import read_frames

    progress(0.02, "reading_tracked_person")
    frames = read_frames(a["source"], a["start"], a["count"])
    actor = mk.load_masks(a["actor_path"])
    isolated = isolate_person(frames, actor, copy=False)  # avoid a second 4K clip in RAM
    sc = cfg["sam3"]
    prompts = [sc["face_prompt"], sc["hair_prompt"]]
    if prompts[0] == prompts[1]:
        raise ValueError("Face and hair SAM3 prompts must be different")
    progress(0.10, "tracking_face_hair_inside_person")
    tracks = track_concepts(isolated, prompts, cfg,
                            progress=lambda d, n: progress(0.10 + 0.75 * d / n, f"tracking_parts:{d}/{n}"))
    # All masks are in full-video coordinates but cannot belong to passers-by.
    face = masks_within(tracks, prompts[0], actor) & actor
    hair = masks_within(tracks, prompts[1], actor) & actor
    pad = max(2, round(min(actor.shape[1:]) * 0.006))
    face, hair, protect, edit, quality = derive_edit_masks(actor, face, hair, padding_px=pad)
    progress(0.9, "saving_protected_edit_masks")
    for name, data in (("face", face), ("hair", hair), ("auto_protect", protect), ("protect", protect), ("edit", edit)):
        mk.save_masks(a["paths"][name], data)
    progress(0.98, "face_hair_ready")
    return quality


def _detector(cache: dict, cfg: dict):
    if "detect" not in cache:
        from ..segmentation import ImageSegmenter
        cache["detect"] = ImageSegmenter(cfg)
    return cache["detect"]


def _variants(cache: dict, cfg: dict, a: dict, progress) -> list:
    from PIL import Image

    from ..keyframe import QwenEditor, edit_crop, finish_keyframe

    if "qwen" not in cache:
        _drop_video_tracker(cache)
        progress(0.03, "loading_editor")
        cache["qwen"] = QwenEditor(cfg)
    garments = [Image.fromarray(g) for g in a["garments"]] or None
    raw = []
    for k, seed in enumerate(a["seeds"]):
        progress(0.1 + 0.75 * k / len(a["seeds"]), f"variant:{k + 1}/{len(a['seeds'])}")
        raw.append(edit_crop(cache["qwen"], a["frame"], a["actor"], garments, a["prompt"], a["margin"], seed,
                             content=a.get("content")))
    progress(0.88, "finishing")
    detect = _detector(cache, cfg)
    return [finish_keyframe(detect, a["frame"], a["actor"], a["face"], c, cfg, paste=True, content=a.get("content"))
            for c in raw]


def _finish_upload(cache: dict, cfg: dict, a: dict, progress) -> tuple:
    from ..keyframe import finish_keyframe

    _drop_video_tracker(cache)
    progress(0.2, "finishing")
    return finish_keyframe(_detector(cache, cfg), a["image"], a["actor"], a["face"], a["image"], cfg,
                           paste=False, content=a.get("content"))


def _layer_meta_path(output_dir) -> str:
    from pathlib import Path
    return str(Path(output_dir) / "_meta.json")


def _save_layer_meta(output_dir, meta: dict) -> None:
    from pathlib import Path
    Path(_layer_meta_path(output_dir)).write_text(__import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_layer_meta(output_dir) -> dict:
    from pathlib import Path
    path = Path(_layer_meta_path(output_dir))
    if not path.is_file():
        return {"missing_frames": {}, "tracking_resolution": None}
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def _compose_from_saved_layers(a: dict, progress) -> dict:
    from pathlib import Path
    from .. import masks as mk
    from ..mask_layers import compose

    progress(0.01, "loading_saved_masks")
    actor_bits, width = mk.load_packed_masks(a["actor_path"])
    actor = np.unpackbits(actor_bits, axis=-1, count=width).astype(bool)
    if actor.ndim != 3 or actor.shape[0] != a["count"]:
        raise ValueError("Actor mask must match the tracked video length")
    n, height, width = actor.shape
    packed_shape = actor_bits.shape
    output = Path(a["output_dir"])
    meta = _load_layer_meta(output)
    missing_by_name = {k: set(v) for k, v in (meta.get("missing_frames") or {}).items()}
    pad = max(0, round(min(height, width) * float(a.get("part_protection_padding_frac", 0.006))))
    visible = actor.reshape(n, -1).any(axis=1)
    layer_data: dict[str, tuple[np.ndarray, int]] = {}
    used_rules = [r for r in a["rules"] if r.get("operation") != "ignore"]
    for rule in a["rules"]:
        name = rule["name"]
        path = output / f"{name}.npz"
        if not path.is_file():
            raise ValueError(f"Mask object '{name}' is not available; track objects first")
        layer_data[name] = mk.load_packed_masks(path)
    protect_bits = np.zeros(packed_shape, dtype=np.uint8)
    edit_bits = np.zeros(packed_shape, dtype=np.uint8)
    edited_pixels = protected_pixels = 0
    per_layer_pixels: dict[str, int] = {r["name"]: 0 for r in a["rules"]}
    base_actor = bool(a.get("base_actor", True))
    progress(0.10, "composing_inverse_union_inverse")
    for t in range(n):
        frame_layers: dict[str, np.ndarray] = {}
        for rule in a["rules"]:
            name = rule["name"]
            bits, layer_width = layer_data[name]
            if layer_width != width or bits.shape != packed_shape:
                raise ValueError(f"Mask object '{name}' has wrong dimensions")
            region = np.unpackbits(bits[t], axis=-1, count=width).astype(bool) & actor[t]
            per_layer_pixels[name] += int(region.sum())
            # A missing negative object must never silently expose those pixels.
            # Keep the full actor protected in this frame and let the user review it.
            if rule["operation"] == "subtract" and visible[t] and t in missing_by_name.get(name, set()):
                region = actor[t]
            frame_layers[name] = region[None]
        edit, protected = compose(actor[t:t+1], frame_layers, a["rules"],
                                  base_actor=base_actor, protection_px=pad)
        edit, protected = edit[0], protected[0]
        if np.any(edit & protected) or np.any(edit & ~actor[t]):
            raise RuntimeError(f"Invalid final mask at frame {t}: exclusion leaked into edit")
        protect_bits[t] = np.packbits(protected, axis=-1)
        edit_bits[t] = np.packbits(edit, axis=-1)
        edited_pixels += int(edit.sum())
        protected_pixels += int(protected.sum())
        if t % max(1, n // 100) == 0 or t == n - 1:
            progress(0.10 + 0.85 * (t + 1) / n, f"composing_final_mask:{t+1}/{n}")
    # A successfully tracked subtraction cannot be empty across the whole clip.
    # Catch wrong mask polarity, stale tracks and off-actor selections early.
    for rule in a["rules"]:
        if rule["operation"] == "subtract" and per_layer_pixels[rule["name"]] == 0:
            raise ValueError(f"Subtraction object '{rule['name']}' has no pixels inside the tracked actor; check its preview and re-track it")
    if not edited_pixels:
        raise ValueError("The final formula produced an empty editable mask; review + / - objects")
    mk.save_packed_masks(a["edit_path"], edit_bits, width)
    mk.save_packed_masks(a["protect_path"], protect_bits, width)
    actor_pixels = int(actor.sum())
    removed_pixels = actor_pixels - edited_pixels
    report = {"method": "invert_union_invert", "actor_pixels": actor_pixels,
              "edited_pixels": edited_pixels, "removed_pixels": removed_pixels,
              "protected_pixels": protected_pixels, "base_actor": base_actor,
              "operations": {rule["name"]: rule["operation"] for rule in a["rules"]},
              "layers": per_layer_pixels, "missing_protection_frames": {k: len(v) for k, v in missing_by_name.items()},
              "tracking_resolution": meta.get("tracking_resolution")}
    (output / "_composition_report.json").write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Mask composition: actor=%d edit=%d removed=%d protect=%d", actor_pixels, edited_pixels, removed_pixels, protected_pixels)
    progress(0.99, "manual_masks_ready")
    return {**report, "coverage": round(edited_pixels / (n * height * width), 5)}


def _mask_layers(cache: dict, cfg: dict, a: dict, progress) -> dict:
    """Track any number of user-defined mask objects on the isolated actor clip,
    save each object independently, then compose the final WAN mask from +/-/ignore.
    """
    import cv2
    from pathlib import Path
    from .. import masks as mk
    from ..person_parts import isolate_person
    from ..segmentation import track_prompts, track_concepts
    from ..tracks import masks_within
    from ..video_io import read_frames

    progress(0.01, "loading_actor_mask")
    actor = mk.load_masks(a["actor_path"])
    if actor.ndim != 3 or actor.shape[0] != a["count"]:
        raise ValueError("Actor mask must match the tracked video length")
    n, height, width = actor.shape
    rules = list(a["rules"])
    output = Path(a["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    # No extra objects: the result is just the tracked actor (or empty if base_actor is off).
    if not rules:
        _save_layer_meta(output, {"missing_frames": {}, "tracking_resolution": None})
        return _compose_from_saved_layers({**a, "rules": []}, progress)

    max_side = int(cfg["sam3"]["part_tracking_max_side"])
    if not 256 <= max_side <= 4096:
        raise ValueError("sam3.part_tracking_max_side must be between 256 and 4096")
    scale = min(1.0, max_side / max(height, width))
    small_w = max(1, round(width * scale))
    small_h = max(1, round(height * scale))
    small_size = (small_w, small_h)
    progress(0.02, f"decoding_isolated_actor:{small_w}x{small_h}")
    if small_size == (width, height):
        frames = read_frames(a["source"], a["start"], n)
    else:
        frames = read_frames(a["source"], a["start"], n, size=small_size)
    if len(frames) != n or frames.shape[1:3] != (small_h, small_w):
        raise ValueError("Decoded clip does not match tracked actor length and expected resolution")

    progress(0.055, "isolating_person_for_manual_segmentation")
    actor_small = np.stack([
        cv2.resize(a_frame.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST) > 0
        for a_frame in actor
    ]) if small_size != (width, height) else actor
    isolated = isolate_person(frames, actor_small, copy=False)
    visible = actor.reshape(n, -1).any(axis=1)
    missing_meta: dict[str, list[int]] = {}
    packed_shape = (n, height, (width + 7) // 8)

    for i, rule in enumerate(rules):
        selection = rule.get("selection")
        text = str((selection or {}).get("text") or rule.get("prompt") or "").strip()
        if not selection or (not text and not selection.get("points") and selection.get("box") is None):
            raise ValueError(f"Layer {rule['name']!r} needs a manual prompt, click or box")
        if selection.get("candidate"):
            raise ValueError("Select masks on the original tracked person, not an edited reference")
        prompt = dict(selection)
        prompt["frame"] = int(prompt["frame"]) - a["start"]
        if not 0 <= prompt["frame"] < n:
            raise ValueError(f"Selection frame for {rule['name']!r} is out of range")
        sx, sy = small_w / width, small_h / height
        prompt["points"] = [[float(x) * sx, float(y) * sy] for x, y in (prompt.get("points") or [])]
        if prompt.get("box") is not None:
            x1, y1, x2, y2 = prompt["box"]
            prompt["box"] = [float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy]

        def layer_progress(done, total):
            progress(0.08 + 0.74 * (i + done / max(1, total)) / len(rules),
                     f"tracking_mask_object:{rule['name']}:{done}/{total}")

        progress(0.08 + 0.74 * i / len(rules), f"starting_mask_object:{rule['name']}")
        if text:
            _drop_video_tracker(cache)
            tracks = track_concepts(isolated, [text], cfg, progress=layer_progress)
            small_mask = masks_within(tracks, text, actor_small,
                    min_overlap=cfg["sam3"]["concept_min_overlap"]) & actor_small
            del tracks
        else:
            small_mask = track_prompts(isolated, [prompt], cfg, progress=layer_progress, model_cache=cache) & actor_small
        if not small_mask.any():
            raise ValueError(f"SAM3 did not find selected object {rule['name']!r}; correct its prompt/clicks/box")
        has_mask = small_mask.reshape(n, -1).any(axis=1)
        missing = np.flatnonzero(visible & ~has_mask).tolist()
        missing_meta[rule["name"]] = missing
        layer_bits = np.zeros(packed_shape, dtype=np.uint8)
        for t in range(n):
            from ..mask_layers import refine_part_mask
            region = refine_part_mask(small_mask[t], (width, height), cfg["sam3"]) & actor[t]
            layer_bits[t] = np.packbits(region, axis=-1)
        mk.save_packed_masks(output / f"{rule['name']}.npz", layer_bits, width)
        del small_mask, layer_bits

    _save_layer_meta(output, {"missing_frames": missing_meta, "tracking_resolution": [small_w, small_h]})
    progress(0.87, "all_mask_objects_tracked")
    return _compose_from_saved_layers(a, progress)


def _compose_mask_layers(cache: dict, cfg: dict, a: dict, progress) -> dict:
    """Recompose the final edit/protect masks from already tracked object masks."""
    return _compose_from_saved_layers(a, progress)


def _drop_video_tracker(cache: dict) -> None:
    """Give the keyframe editor VRAM previously held by SAM3 without killing the worker."""
    cached = cache.pop("sam3_video_tracker", None)
    if cached is not None:
        cached = None
        from ..segmentation import free_cuda
        free_cuda()
        log.info("Released cached SAM3 video tracker for image editing")


TASKS = {"track": _track, "person_parts": _person_parts, "variants": _variants, "finish_upload": _finish_upload,
         "mask_layers": _mask_layers, "compose_mask_layers": _compose_mask_layers}


def _child_main(conn, cfg: dict, tasks_module: str = __name__) -> None:
    import importlib

    logging.basicConfig(level=logging.INFO, format="[worker] %(asctime)s %(message)s")
    tasks = importlib.import_module(tasks_module).TASKS
    cache: dict = {}
    while True:
        msg = conn.recv()
        if msg is None:
            break
        name, args = msg
        try:
            # A long-lived GPU worker must use the settings belonging to EACH job.
            # Never mutate its shared base config between projects.
            task_cfg = cfg
            if isinstance(args, dict) and "sam3_settings" in args:
                task_cfg = {**cfg, "sam3": {**cfg["sam3"], **args["sam3_settings"]}}
            result = tasks[name](cache, task_cfg, args, lambda f, m: conn.send(("progress", f, m)))
            conn.send(("ok", result))
        except Exception as exc:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------- parent side
class WorkerError(RuntimeError):
    pass


class ModelWorker:
    """tasks_module: importable module with a TASKS dict {name: fn(cache, cfg, args, progress)}."""

    def __init__(self, cfg: dict, tasks_module: str = __name__):
        self.cfg = cfg
        self.tasks_module = tasks_module
        self.proc = None
        self.conn = None
        self.lock = threading.Lock()

    def _ensure(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_child_main, args=(child, self.cfg, self.tasks_module), daemon=True,
                                name="model-worker")
        self.proc.start()
        self.conn = parent
        log.info("Model worker started (pid %s)", self.proc.pid)

    def call(self, task: str, args: dict, job=None):
        """Run a task in the worker; progress goes to job.set. Cancelling the job kills the worker."""
        with self.lock:
            self._ensure()
            self.conn.send((task, args))
            while True:
                if job is not None and job.cancel_requested:
                    self.stop()
                    job.check()
                if not self.conn.poll(0.5):
                    if not self.proc.is_alive():
                        self.proc, self.conn = None, None
                        raise WorkerError("The model worker stopped unexpectedly (out of memory?)")
                    continue
                kind, *payload = self.conn.recv()
                if kind == "progress":
                    if job is not None:
                        job.set(*payload)
                elif kind == "ok":
                    return payload[0]
                else:
                    raise WorkerError(payload[0].split("\n", 1)[0] if payload[0] else "worker error")

    def stop(self) -> None:
        """Shut the worker down so all its memory (RAM and VRAM) is returned."""
        if self.proc is None:
            return
        try:
            if self.proc.is_alive():
                try:
                    self.conn.send(None)
                except (BrokenPipeError, OSError):
                    pass
                self.proc.join(10)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(5)
        finally:
            log.info("Model worker stopped")
            self.proc, self.conn = None, None
