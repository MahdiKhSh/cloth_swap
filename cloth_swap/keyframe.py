"""Keyframe selection and garment editing with Qwen-Image-Edit-2511."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from . import masks as mk
from .geometry import bbox, box_mask, clamp_box, content_box, edit_size, expand_box
from .segmentation import ImageSegmenter, free_cuda
from .tracks import mask_iou

log = logging.getLogger(__name__)


def frame_scores(frames: np.ndarray, actor: np.ndarray, face: np.ndarray) -> np.ndarray:
    """How good each frame is as the appearance anchor.

    Large actor, not cut by the frame border, face visible and little motion blur.
    """
    num_frames, height, width = actor.shape
    areas = actor.reshape(num_frames, -1).mean(axis=1)
    sharpness = np.zeros(num_frames, np.float32)
    complete = np.ones(num_frames, np.float32)
    for t in range(num_frames):
        box = bbox(actor[t])
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1:
            complete[t] = 0.6
        gray = cv2.cvtColor(frames[t, y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        sharpness[t] = cv2.Laplacian(gray, cv2.CV_32F).var()
    if areas.max() > 0:
        areas = areas / areas.max()
    if sharpness.max() > 0:
        sharpness = sharpness / sharpness.max()
    has_face = np.where(face.reshape(num_frames, -1).any(axis=1), 1.15, 1.0)
    return areas * complete * has_face * (0.5 + 0.5 * sharpness)


def choose_keyframes(scores: np.ndarray, num_extra: int, fixed: int | None) -> list[int]:
    """Best frame (or `fixed`) first, then extra frames spread out in time."""
    num_frames = len(scores)
    chosen = [int(fixed) if fixed is not None else int(np.argmax(scores))]
    min_gap = max(1, num_frames // (num_extra + 2))
    for t in np.argsort(-scores):
        if len(chosen) > num_extra:
            break
        if scores[t] > 0 and all(abs(int(t) - c) >= min_gap for c in chosen):
            chosen.append(int(t))
    return chosen


class QwenEditor:
    def __init__(self, cfg: dict):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        qcfg = cfg["qwen"]
        model_id = qcfg["model_id"]
        components = {}
        if qcfg["quantization"] == "nf4":
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
            from diffusers import QwenImageTransformer2DModel
            from transformers import BitsAndBytesConfig as TransformersBnbConfig
            from transformers import Qwen2_5_VLForConditionalGeneration

            nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            components["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer",
                quantization_config=DiffusersBnbConfig(**nf4), torch_dtype=torch.bfloat16)
            components["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, subfolder="text_encoder",
                quantization_config=TransformersBnbConfig(**nf4), dtype=torch.bfloat16)
        elif qcfg["quantization"] not in ("prequantized", "none"):
            raise ValueError(f"Unknown qwen.quantization: {qcfg['quantization']}")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, **components)
        if qcfg.get("lightning_lora"):
            self.pipe.load_lora_weights(qcfg["lightning_lora"])
        if qcfg["cpu_offload"] == "model":
            self.pipe.enable_model_cpu_offload()
        elif qcfg["cpu_offload"] == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(cfg["device"])
        self.steps = qcfg["steps"]
        self.true_cfg = qcfg["true_cfg_scale"]
        self.seed = qcfg["seed"]

    def edit(self, images: list[Image.Image], prompt: str, seed: int | None = None) -> Image.Image:
        import torch

        # The pipeline sizes its output from the LAST image (the garment photo). The output must
        # have the person crop's shape, or the editor recomposes the scene and re-poses the person.
        width, height = edit_size(*images[0].size)
        with torch.inference_mode():
            result = self.pipe(
                image=images,
                prompt=prompt,
                width=width,
                height=height,
                negative_prompt=NEGATIVE_PROMPT,
                true_cfg_scale=self.true_cfg,
                guidance_scale=1.0,
                num_inference_steps=self.steps,
                generator=torch.Generator(device="cpu").manual_seed(self.seed if seed is None else seed),
                num_images_per_prompt=1,
            )
        return result.images[0].convert("RGB")

    def close(self) -> None:
        del self.pipe
        free_cuda()


NEGATIVE_PROMPT = ("different person, changed face, different facial expression, different pose, turned body, "
                   "frontal pose, extra people, extra limbs, deformed hands, cropped head, blurry, low quality, "
                   "studio background, white background, product photo")


def edit_prompt(cfg: dict, garment_images: int, garment_desc: str | None, extra: str | None = None) -> str:
    """Built-in instruction for the image editor, plus the user's optional additions.

    The built-in part keeps everything except the clothing fixed; the user's text only adds
    details (e.g. "tuck the shirt in", "sleeves rolled up").
    """
    kcfg = cfg["keyframe"]
    desc = (garment_desc or "").strip().rstrip(".")
    if garment_images:
        views = cfg.get("garment_bible_views") or []
        if views:
            refs = ", ".join(f"image {i + 2} ({view} view)" for i, view in enumerate(views[:garment_images]))
        else:
            refs = "image 2" if garment_images == 1 else "images 2 and 3 (front and back of the same garment)"
        garment = f"the garment shown in {refs}" + (f" ({desc})" if desc else "")
        text = kcfg["prompt"].format(garment=garment)
    elif desc:
        text = kcfg["text_only_prompt"].format(garment=desc)
    else:
        raise ValueError("Give a garment image or a description of the new garment")
    if extra and extra.strip():
        text += " Additional instructions: " + extra.strip()
    return text


def edit_crop(editor: QwenEditor, frame: np.ndarray, actor_mask: np.ndarray, garment,
              prompt: str, margin: float, seed: int | None = None, content=None) -> np.ndarray:
    """Edit a crop around the actor and return the full frame with the edited crop in place.

    garment: None, one image, or a list (e.g. front and back); the editor takes at most two besides the crop.
    content: picture box without letterbox bars; the crop never reaches into the bars.
    """
    height, width = frame.shape[:2]
    box = bbox(actor_mask)
    if box is None:
        raise RuntimeError("The actor is not visible on this frame")
    x0, y0, x1, y1 = clamp_box(expand_box(box, margin, width, height), content or (0, 0, width, height))
    garments = [] if garment is None else list(garment) if isinstance(garment, (list, tuple)) else [garment]
    images = [Image.fromarray(frame[y0:y1, x0:x1])] + garments[:2]
    log.info("Editing crop %dx%d (seed %s)", x1 - x0, y1 - y0, seed)
    result = editor.edit(images, prompt, seed).resize((x1 - x0, y1 - y0), Image.LANCZOS)
    candidate = frame.copy()
    candidate[y0:y1, x0:x1] = np.asarray(result)
    return candidate


def edit_frames(frames: np.ndarray, actor: np.ndarray, indices: list[int], garment: Image.Image | None,
                garment_desc: str | None, cfg: dict) -> list[np.ndarray]:
    """Run the image editor on a crop around the actor and put the crop back into the full frame."""
    prompt = edit_prompt(cfg, 1 if garment is not None else 0, garment_desc or cfg["prompt"].get("garment"))
    content = content_box(frames)
    editor = QwenEditor(cfg)
    try:
        return [edit_crop(editor, frames[t], actor[t], garment, prompt, cfg["keyframe"]["crop_margin"], content=content)
                for t in indices]
    finally:
        editor.close()


def face_aligned(frame: np.ndarray, candidate: np.ndarray, face: np.ndarray, threshold: float = 0.5) -> bool:
    """Whether the edited face sits where the original one is (edge correlation inside the face box)."""
    box = bbox(face)
    if box is None:
        return False
    x0, y0, x1, y1 = box

    def edges(img):
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), max(1.0, (x1 - x0) / 60))
        return cv2.Laplacian(g, cv2.CV_32F).ravel()
    a, b = edges(frame), edges(candidate)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return bool(denom > 0 and float(a @ b) / denom >= threshold)


def blend_keyframe(frame: np.ndarray, candidate: np.ndarray, actor: np.ndarray, new_actor: np.ndarray,
                   face: np.ndarray | None, restore_face: bool, content=None) -> np.ndarray:
    """Keep the edit only around the actor (old or new silhouette), never in letterbox bars, and restore
    the original face only when the edited face is still in the same place (otherwise it would ghost)."""
    height, width = frame.shape[:2]
    px = max(2, int(round(0.01 * height)))
    alpha = mk.feather(mk.dilate(actor | new_actor, 2 * px), 2 * px)
    if content is not None:
        alpha *= box_mask(content, height, width)
    key = mk.blend(candidate, frame, alpha)
    if restore_face and face is not None and face.any():
        if face_aligned(frame, candidate, face):
            fpx = max(1, int(round(0.004 * height)))
            key = mk.blend(frame, key, mk.feather(mk.erode(face, fpx), 2 * fpx))
        else:
            log.warning("The edited face moved; not restoring the original face (it would ghost)")
    return key


def finish_keyframes(frames: np.ndarray, actor: np.ndarray, face: np.ndarray, indices: list[int],
                     candidates: list[np.ndarray], cfg: dict, paste: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Find the actor in each edited frame and blend the edit into the original frame.

    Returns (keyframe RGB, actor mask in the keyframe) pairs. With paste=False the candidate is
    used as is (a keyframe the user edited elsewhere).
    """
    segmenter = ImageSegmenter(cfg)
    content = content_box(frames)
    try:
        return [finish_keyframe(segmenter, frames[t], actor[t], face[t], candidate, cfg, paste, content)
                for t, candidate in zip(indices, candidates)]
    finally:
        segmenter.close()


def find_actor(segmenter: ImageSegmenter, image: np.ndarray, actor: np.ndarray, prompt: str) -> np.ndarray:
    """The person in `image` that best overlaps the original actor mask."""
    found = segmenter.segment(image, prompt)
    if not found:
        return actor.copy()
    best = max(found, key=lambda m: mask_iou(m, actor))
    if mask_iou(best, actor) < 0.2:
        log.warning("The edited actor barely overlaps the original; check the edit")
    return best


def finish_keyframe(segmenter: ImageSegmenter, frame: np.ndarray, actor: np.ndarray, face: np.ndarray,
                    candidate: np.ndarray, cfg: dict, paste: bool, content=None) -> tuple[np.ndarray, np.ndarray]:
    new_actor = find_actor(segmenter, candidate, actor, cfg["actor"]["prompt"])
    if content is not None:
        new_actor &= box_mask(content, *new_actor.shape)
    if not paste:  # edited elsewhere: may not be pixel-aligned with the frame, use it as is
        return candidate, new_actor
    key = blend_keyframe(frame, candidate, actor, new_actor, face, cfg["keyframe"]["restore_face"], content)
    return key, new_actor
