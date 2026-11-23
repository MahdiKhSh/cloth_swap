"""End-to-end clothing replacement: masks -> keyframe -> SCAIL-2 -> upscale -> composite.

Every stage writes its results into the work directory and a `<stage>.done` marker, so a run
can be resumed or restarted from any stage (e.g. retry SCAIL-2 with other settings without
recomputing masks and keyframes).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import masks as mk
from .config import save_config
from .geometry import content_box, working_size
from .video_io import iter_frames, mux_audio, probe, read_frames, write_video

# output.resolution -> short side of the delivered video (SeedVR2 upscale); "original" keeps the source size
OUTPUT_SHORT_SIDE = {"original": None, "1080p": 1080, "4k": 2160}

log = logging.getLogger(__name__)
STAGES = ("masks", "keyframe", "scail", "upscale", "composite")


class Pipeline:
    def __init__(self, cfg: dict, video: str | Path, out: str | Path, work_dir: str | Path,
                 garment: str | Path | None = None, keyframe_image: str | Path | None = None):
        self.cfg = cfg
        self.video = Path(video)
        self.out = Path(out)
        self.work = Path(work_dir)
        self.garment = Path(garment) if garment else None
        self.keyframe_image = Path(keyframe_image) if keyframe_image else None
        self.work.mkdir(parents=True, exist_ok=True)
        self.on_scail_line = None   # optional callbacks for live progress / cancelling (see scail2_stage.run)
        self.on_scail_start = None

    # ------------------------------------------------------------------ helpers
    def path(self, *parts: str) -> Path:
        p = self.work.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def subdir(self, name: str) -> Path:
        d = self.work / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def meta(self) -> dict:
        return json.loads(self.path("meta.json").read_text(encoding="utf-8"))

    def frames(self) -> np.ndarray:
        m = self.meta()
        return read_frames(self.video, m["start_frame"], m["num_frames"])

    def load_mask(self, name: str) -> np.ndarray | None:
        p = self.path("01_masks", f"{name}.npz")
        return mk.load_masks(p) if p.exists() else None

    # --------------------------------------------------------------------- run
    def run(self, from_stage: str | None = None, stop_after: str | None = None) -> None:
        save_config(self.cfg, self.path("config_used.yaml"))
        start = STAGES.index(from_stage) if from_stage else len(STAGES)
        old_inputs = self.work / "03_scail" / "inputs.json"
        if old_inputs.exists():
            previous_backend = json.loads(old_inputs.read_text(encoding="utf-8")).get("backend", "scail2")
            if previous_backend != self.cfg.get("generation", {}).get("backend", "scail2"):
                start = min(start, STAGES.index("scail"))
        stop = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
        force = False
        for i, name in enumerate(STAGES[: stop + 1]):
            marker = self.path(f"{name}.done")
            force = force or i >= start
            if marker.exists() and not force:
                log.info("Stage %s: already done, skipping", name)
                continue
            marker.unlink(missing_ok=True)
            log.info("Stage %s: start", name)
            began = time.time()
            getattr(self, f"stage_{name}")()
            marker.write_text(f"{time.time() - began:.1f}s\n", encoding="utf-8")
            log.info("Stage %s: done in %.1fs", name, time.time() - began)
            force = True  # everything after a re-run stage depends on its output

    # ------------------------------------------------------------------ stages
    def stage_masks(self) -> None:
        from . import segmentation, tracks as tr

        cfg, acfg, scfg = self.cfg, self.cfg["actor"], self.cfg["sam3"]
        info = probe(self.video)
        start = cfg["video"]["start_frame"]
        frames = read_frames(self.video, start, cfg["video"]["max_frames"])
        num_frames, height, width = frames.shape[:3]
        select_frame = min(acfg["select_frame"], num_frames - 1)

        concept_prompts = [scfg["face_prompt"], scfg["hair_prompt"]]  # the face picks the keyframe and is restored on it
        if cfg["composite"]["mode"] == "garment":
            concept_prompts.append(scfg["old_garment_prompt"])

        meta = {"fps": info.fps, "width": width, "height": height,
                "start_frame": start, "num_frames": num_frames}
        if acfg["mode"] == "text":
            tracks = segmentation.track_concepts(frames, [acfg["prompt"], *concept_prompts], cfg)
            actor_id, chosen_on = tr.select_actor(tracks, acfg["prompt"], select_frame, acfg["point"], acfg["box"])
            actor = tr.actor_masks(tracks, actor_id, acfg["prompt"], acfg["stitch_iou"], acfg["gap_fill"])
            meta.update(actor_id=actor_id, actor_chosen_on=chosen_on)
        elif acfg["mode"] == "track":
            actor = segmentation.track_prompted(frames, select_frame, acfg["point"], acfg["box"], cfg)
            tracks = segmentation.track_concepts(frames, concept_prompts, cfg)
        else:
            raise ValueError(f"Unknown actor.mode: {acfg['mode']}")
        actor = tr.drop_occluded(actor)

        visible = int(actor.reshape(num_frames, -1).any(axis=1).sum())
        if visible == 0:
            raise RuntimeError("The actor mask is empty in every frame")
        log.info("Actor visible in %d/%d frames", visible, num_frames)
        mk.save_masks(self.path("01_masks", "actor.npz"), actor)

        near_actor = mk.temporal_union(actor, 1)
        for name, prompt in (("face", scfg["face_prompt"]), ("hair", scfg["hair_prompt"]),
                             ("garment_old", scfg["old_garment_prompt"] if cfg["composite"]["mode"] == "garment" else None)):
            if prompt:
                mk.save_masks(self.path("01_masks", f"{name}.npz"), tr.masks_within(tracks, prompt, near_actor))

        w_work, h_work = working_size(width, height, cfg["scail"]["target_area"], cfg["scail"]["size_multiple"])
        meta.update(work_width=w_work, work_height=h_work, content_box=list(content_box(frames)))
        self.path("meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if cfg["output"]["preview"]:
            face = self.load_mask("face")
            def preview():
                for t in range(num_frames):
                    f = mk.overlay(frames[t], actor[t], (255, 0, 0))
                    if face is not None:
                        f = mk.overlay(f, face[t], (0, 255, 0))
                    yield f[::2, ::2]
            write_video(self.path("01_masks", "preview.mp4"), preview(), info.fps, crf=23)

    def stage_keyframe(self) -> None:
        from . import keyframe as kf

        cfg, kcfg = self.cfg, self.cfg["keyframe"]
        frames = self.frames()
        actor = self.load_mask("actor")
        face = self.load_mask("face")
        if face is None:
            face = np.zeros_like(actor)

        if self.keyframe_image is not None:
            if kcfg["frame"] is None:
                log.warning("--keyframe-frame not given; assuming the edited image is frame 0")
            index = kcfg["frame"] if kcfg["frame"] is not None else 0
            image = Image.open(self.keyframe_image).convert("RGB")
            image = image.resize((frames.shape[2], frames.shape[1]), Image.LANCZOS)
            indices, candidates, paste = [index], [np.asarray(image)], False
        else:
            scores = kf.frame_scores(frames, actor, face)
            indices = kf.choose_keyframes(scores, kcfg["num_extra"], kcfg["frame"])
            log.info("Keyframes: %s", indices)
            garment = Image.open(self.garment).convert("RGB") if self.garment else None
            candidates = kf.edit_frames(frames, actor, indices, garment, cfg["prompt"].get("garment"), cfg)
            paste = True
            for t, c in zip(indices, candidates):
                Image.fromarray(c).save(self.path("02_keyframe", f"edited_raw_{t:05d}.png"))

        results = kf.finish_keyframes(frames, actor, face, indices, candidates, cfg, paste)
        for k, (t, (image, mask)) in enumerate(zip(indices, results)):
            Image.fromarray(frames[t]).save(self.path("02_keyframe", f"original_{k}.png"))
            Image.fromarray(image).save(self.path("02_keyframe", f"keyframe_{k}.png"))
            mk.save_masks(self.path("02_keyframe", f"ref_mask_{k}.npz"), mask[None])
        self.path("02_keyframe", "keyframes.json").write_text(json.dumps({"indices": indices}), encoding="utf-8")

    def _keyframes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = json.loads(self.path("02_keyframe", "keyframes.json").read_text(encoding="utf-8"))["indices"]
        return [(np.asarray(Image.open(self.path("02_keyframe", f"keyframe_{k}.png")).convert("RGB")),
                 mk.load_masks(self.path("02_keyframe", f"ref_mask_{k}.npz"))[0])
                for k in range(len(indices))]

    def stage_scail(self) -> None:
        if self.cfg.get("generation", {}).get("backend") == "wan_animate":
            from . import animate_stage
            m = self.meta()
            animate_stage.run(self.subdir("03_scail"), self.frames(), self.load_mask("actor"),
                self._keyframes()[0][0], (m["work_width"], m["work_height"]), m["fps"], self.cfg,
                on_line=self.on_scail_line, on_start=self.on_scail_start)
            return
        from . import scail2_stage

        m = self.meta()
        inputs = scail2_stage.prepare_inputs(
            self.subdir("03_scail"), self.frames(),
            (self.load_mask("edit") if self.cfg.get("mask_layers_enabled") else self.load_mask("actor")), self._keyframes(),
            (m["work_width"], m["work_height"]), m["fps"], self.cfg)
        self.path("03_scail", "inputs.json").write_text(
            json.dumps({"padded_frames": inputs.padded_frames, "backend": "scail2"}), encoding="utf-8")
        scail2_stage.run(inputs, self.path("03_scail", "scail_out.mp4"), self.cfg,
                         on_line=self.on_scail_line, on_start=self.on_scail_start)

    def stage_upscale(self) -> None:
        from . import scail2_stage, upscale

        m = self.meta()
        padded = json.loads(self.path("03_scail", "inputs.json").read_text(encoding="utf-8"))["padded_frames"]
        work = scail2_stage.load_output(self.path("03_scail", "scail_out.mp4"), m["num_frames"],
                                        (m["work_width"], m["work_height"]), padded)
        if self.cfg["upscale"]["backend"] == "resize":
            # Resize individual frames as they are encoded. Do not allocate an
            # additional 4K T×H×W×3 array solely for this conversion.
            import cv2
            width, height = m["width"], m["height"]

            def resized():
                for frame in work:
                    if frame.shape[:2] == (height, width):
                        yield frame
                    else:
                        interp = cv2.INTER_AREA if width < frame.shape[1] else cv2.INTER_LANCZOS4
                        yield cv2.resize(frame, (width, height), interpolation=interp)

            write_video(self.path("04_upscale", "generated_full.mp4"), resized(), m["fps"], crf=12)
        else:
            full = upscale.to_original_size(work, (m["width"], m["height"]), m["fps"], self.cfg, self.subdir("04_upscale"))
            write_video(self.path("04_upscale", "generated_full.mp4"), full, m["fps"], crf=12)

    def stage_composite(self) -> None:
        from . import composite as cp

        cfg = self.cfg
        m = self.meta()
        generated_path = self.path("04_upscale", "generated_full.mp4")
        generated = None
        actor = self.load_mask("actor")
        edit = self.load_mask("edit") if self.cfg.get("mask_layers_enabled") else None
        protected = self.load_mask("protect") if self.cfg.get("mask_layers_enabled") else None

        new_garment = None
        if cfg["composite"]["mode"] == "garment":
            from . import segmentation, tracks as tr
            # SAM3 video tracking currently requires the whole clip. The other
            # composite modes can avoid this second uncompressed 4K allocation.
            generated = read_frames(generated_path, count=m["num_frames"])
            tracks = segmentation.track_concepts(generated, [cfg["sam3"]["new_garment_prompt"]], cfg)
            new_garment = tr.masks_within(tracks, cfg["sam3"]["new_garment_prompt"],
                                          mk.temporal_union(actor, 2))
            mk.save_masks(self.path("05_composite", "garment_new.npz"), new_garment)

        alpha = cp.AlphaBuilder(cfg, m["height"], actor, old_garment=self.load_mask("garment_old"), new_garment=new_garment,
                                content=m.get("content_box"), edit=edit, protected=protected)
        silent = self.path("05_composite", "composite_silent.mp4")
        def frames_to_composite():
            original_stream = iter_frames(self.video, m["start_frame"], m["num_frames"])
            generated_stream = iter(generated) if generated is not None else iter_frames(generated_path, count=m["num_frames"])
            try:
                for t in range(m["num_frames"]):
                    try:
                        source = next(original_stream)
                        altered = next(generated_stream)
                    except StopIteration as exc:
                        raise ValueError(f"Video stream ended before frame {t + 1}/{m['num_frames']}") from exc
                    yield mk.blend(altered, source, alpha(t))
            finally:
                for stream in (original_stream, generated_stream):
                    close = getattr(stream, "close", None)
                    if close:
                        close()

        write_video(silent, frames_to_composite(), m["fps"], crf=cfg["output"]["crf"])
        final = silent
        short_side = OUTPUT_SHORT_SIDE.get(cfg["output"].get("resolution") or "original")
        if short_side and short_side > min(m["width"], m["height"]):
            # the whole finished shot (actor and background alike) goes through SeedVR2 in one pass
            from .upscale import seedvr2_file
            final = seedvr2_file(silent, self.path("05_composite", f"composite_{short_side}p.mp4"), short_side, cfg)

        self.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            mux_audio(final, self.video, self.out, m["start_frame"] / m["fps"], m["num_frames"] / m["fps"])
        except Exception as exc:  # no audio stream or ffmpeg trouble: keep the silent result
            log.warning("Audio mux failed (%s); writing video without audio", exc)
            self.out.write_bytes(final.read_bytes())

        if cfg["output"]["preview"]:
            result = read_frames(silent)
            original = self.frames()
            write_video(self.path("05_composite", "side_by_side.mp4"), cp.side_by_side(original, result),
                        m["fps"], crf=20)
        log.info("Result: %s", self.out)
