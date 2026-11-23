"""SAM3 wrappers (Hugging Face transformers): concept tracking, prompted tracking, image masks."""
from __future__ import annotations

import gc
import logging

import numpy as np

from .tracks import ConceptTracks

log = logging.getLogger(__name__)


def torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def free_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_bool(mask) -> np.ndarray:
    mask = mask.detach().float().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    return mask > 0.5


def track_concepts(frames: np.ndarray, prompts: list[str], cfg: dict, progress=None) -> ConceptTracks:
    """Detect and track every instance of each text prompt through the clip."""
    import torch
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

    scfg = cfg["sam3"]
    device = cfg["device"]
    dtype = torch_dtype(scfg["dtype"])
    prompts = list(dict.fromkeys(p for p in prompts if p))
    num_frames, height, width = frames.shape[:3]
    if not prompts:
        return ConceptTracks(num_frames, height, width)

    model_cfg = Sam3VideoConfig.from_pretrained(scfg["model_id"])
    from .studio.sam3_controls import VIDEO_CONFIG_MAP
    for control, field in VIDEO_CONFIG_MAP.items():
        if field not in vars(model_cfg):
            raise ValueError(f"Installed transformers SAM3 VideoConfig does not support {field}")
        setattr(model_cfg, field, scfg[control])
    # The model and processor must use the SAME internal resolution.
    image_size = scfg["image_size"]
    model_cfg.image_size = image_size
    model = Sam3VideoModel.from_pretrained(scfg["model_id"], config=model_cfg, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(scfg["model_id"],
                size={"height": image_size, "width": image_size})

    session = processor.init_video_session(
        video=list(frames),
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    processor.add_text_prompt(inference_session=session, text=prompts)

    tracks = ConceptTracks(num_frames, height, width)
    log.info("SAM3 tracking %s over %d frames", prompts, num_frames)
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(inference_session=session, show_progress_bar=True):
            result = processor.postprocess_outputs(session, output)
            for prompt, obj_ids in (result.get("prompt_to_obj_ids") or {}).items():
                for obj_id in obj_ids:
                    tracks.prompt_of.setdefault(int(obj_id), prompt)
            obj_ids = [int(o) for o in result["object_ids"].tolist()]
            for i, obj_id in enumerate(obj_ids):
                if len(prompts) == 1:
                    tracks.prompt_of.setdefault(obj_id, prompts[0])
                tracks.add(obj_id, int(output.frame_idx), _to_bool(result["masks"][i]))
            if progress:
                progress(int(output.frame_idx) + 1, num_frames)

    del model, processor, session
    free_cuda()
    return tracks


def _prompt_inputs(points=None, labels=None, box=None) -> dict:
    """SAM2-style nested prompt arguments for one object on one image/frame."""
    inputs = {}
    if box is not None:
        inputs["input_boxes"] = [[[float(v) for v in box]]]
    if points:
        inputs["input_points"] = [[[[float(x), float(y)] for x, y in points]]]
        inputs["input_labels"] = [[[int(v) for v in (labels or [1] * len(points))]]]
    return inputs


def track_prompted(frames: np.ndarray, frame_idx: int, point, box, cfg: dict) -> np.ndarray:
    """Track one object from a click and/or box on `frame_idx`, forward and backward."""
    if point is None and box is None:
        raise ValueError("actor.mode=track needs actor.point or actor.box")
    prompt = {"frame": frame_idx, "points": [point] if point is not None else [], "box": box}
    return track_prompts(frames, [prompt], cfg)


def track_prompts(frames: np.ndarray, prompts: list[dict], cfg: dict, progress=None,
                  model_cache: dict | None = None) -> np.ndarray:
    """Track one object through the clip from prompts on one or more frames.

    Each prompt is {"frame": t, "points": [[x, y], ...], "labels": [1|0, ...], "box": [x1, y1, x2, y2] | None};
    label 1 adds the clicked area, 0 removes it. Extra frames act as corrections. Propagation runs
    forward and backward from the earliest prompted frame. progress(done, total) is optional.
    """
    import torch
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    prompts = [p for p in prompts if p.get("points") or p.get("box") is not None]
    if not prompts:
        raise ValueError("Select the person with a click or a box first")
    scfg = cfg["sam3"]
    device = cfg["device"]
    dtype = torch_dtype(scfg["dtype"])
    # A worker owns ONE video tracker for its entire segmentation session. Recreate
    # inference sessions, not the several-GB model weights, for each user-defined layer.
    key = (scfg["model_id"], device, scfg["dtype"])
    cached = model_cache.get("sam3_video_tracker") if model_cache is not None else None
    if cached is not None and cached[0] == key:
        model, processor = cached[1], cached[2]
        log.info("Reusing SAM3 video tracker (no weight reload)")
    else:
        if cached is not None:
            model_cache.pop("sam3_video_tracker", None)
            del cached
            free_cuda()
        model = Sam3TrackerVideoModel.from_pretrained(scfg["model_id"], dtype=dtype).to(device).eval()
        processor = Sam3TrackerVideoProcessor.from_pretrained(scfg["model_id"])
        if model_cache is not None:
            model_cache["sam3_video_tracker"] = (key, model, processor)
            log.info("Cached SAM3 video tracker for subsequent mask layers")

    num_frames, height, width = frames.shape[:3]
    session = processor.init_video_session(video=list(frames), inference_device=device, dtype=dtype)
    for p in sorted(prompts, key=lambda p: p["frame"]):
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=int(p["frame"]), obj_ids=1,
            **_prompt_inputs(p.get("points"), p.get("labels"), p.get("box")))

    start = min(int(p["frame"]) for p in prompts)
    out = np.zeros((num_frames, height, width), bool)
    done = 0
    with torch.inference_mode():
        for reverse in (False, True):
            if reverse and start == 0:
                break
            for output in model.propagate_in_video_iterator(
                    inference_session=session, start_frame_idx=start, reverse=reverse):
                masks = processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[height, width]], binarize=True,
                    mask_threshold=scfg["tracker_mask_threshold"],
                    max_hole_area=scfg["tracker_max_hole_area"],
                    max_sprinkle_area=scfg["tracker_max_sprinkle_area"],
                    apply_non_overlapping_constraints=scfg["tracker_non_overlap"])[0]
                out[int(output.frame_idx)] = _to_bool(masks[0])
                done += 1
                if progress:
                    progress(min(done, num_frames), num_frames)

    del session
    # Release tensors from the previous video inference session between
    # manually selected parts without discarding cached model weights.
    if model_cache is None:
        del model, processor
    free_cuda()
    return out


class PointSegmenter:
    """Interactive single-frame segmentation from clicks and/or a box (SAM3 tracker, image mode)."""

    def __init__(self, cfg: dict):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        scfg = cfg["sam3"]
        self.device = cfg["device"]
        self.model = Sam3TrackerModel.from_pretrained(scfg["model_id"], dtype=torch_dtype(scfg["dtype"])).to(self.device).eval()
        self.processor = Sam3TrackerProcessor.from_pretrained(scfg["model_id"])

    def segment(self, image: np.ndarray, points=None, labels=None, box=None,
                settings: dict | None = None) -> np.ndarray:
        import torch

        inputs = self.processor(images=image, return_tensors="pt", **_prompt_inputs(points, labels, box)).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        scfg = settings or {}
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"],
            mask_threshold=scfg.get("tracker_mask_threshold", 0.0),
            max_hole_area=scfg.get("tracker_max_hole_area", 0.0),
            max_sprinkle_area=scfg.get("tracker_max_sprinkle_area", 0.0),
            apply_non_overlapping_constraints=scfg.get("tracker_non_overlap", False))[0]
        return _to_bool(masks[0])

    def close(self) -> None:
        del self.model, self.processor
        free_cuda()


class ImageSegmenter:
    """SAM3 text-prompted instance masks on single images."""

    def __init__(self, cfg: dict):
        from transformers import Sam3Model, Sam3Processor

        scfg = cfg["sam3"]
        self.device = cfg["device"]
        self.threshold = scfg["score_threshold"]
        self.model = Sam3Model.from_pretrained(scfg["model_id"], dtype=torch_dtype(scfg["dtype"])).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(scfg["model_id"])

    def segment(self, image: np.ndarray, prompt: str, *, threshold: float | None = None,
                mask_threshold: float = 0.5) -> list[np.ndarray]:
        import torch

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold if threshold is None else threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        return [_to_bool(m) for m in result["masks"]]

    def close(self) -> None:
        del self.model, self.processor
        free_cuda()
