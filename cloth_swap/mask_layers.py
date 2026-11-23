"""Boolean mask composition for independently tracked video objects.

All masks use True for selected/foreground pixels (including SAM3 outputs).
The ComfyUI Masks Add -> InvertMask workflow is represented explicitly:
    result = NOT(NOT(positive_union) OR dilated_negative_union).
This is equivalent to set subtraction, not an inversion of SAM3's object mask.
"""
from __future__ import annotations

import numpy as np
from . import masks as mk


def compose(actor: np.ndarray, layers: dict[str, np.ndarray], rules: list[dict],
            base_actor: bool = True, protection_px: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (editable, protected) bool masks for N frames.

    Start with actor if requested, OR in '+' objects, union all '-' objects,
    expand that union once, then apply ComfyUI's invert / add / invert.
    Objects are clipped to the actor, so 'ignore' does not affect the result.
    """
    actor = np.asarray(actor, dtype=bool)
    if actor.ndim != 3:
        raise ValueError("Actor mask must have shape [frames, height, width]")
    positive = actor.copy() if base_actor else np.zeros_like(actor)
    negative = np.zeros_like(actor)
    for rule in rules:
        name, operation = rule.get("name"), rule.get("operation")
        if operation not in ("add", "subtract", "ignore") or not isinstance(name, str) or not name.strip():
            raise ValueError("Mask layers require a name and add/subtract/ignore operation")
        if name not in layers:
            raise ValueError(f"Mask layer '{name}' is not available; run segmentation first")
        mask = np.asarray(layers[name], dtype=bool)
        if mask.shape != actor.shape:
            raise ValueError(f"Mask layer '{name}' has wrong dimensions")
        masked = mask & actor
        if operation == "add":
            positive |= masked
        elif operation == "subtract":
            negative |= masked

    if protection_px:
        negative = np.stack([mk.dilate(m, protection_px) for m in negative])
    # ComfyUI: (selected negatives) + Invert(positive), then Invert the sum.
    # This preserves the correct meaning of white == editable in edit.npz.
    combined_inverse = np.logical_or(np.logical_not(positive), negative)
    edit = np.logical_not(combined_inverse) & actor
    protected = negative & actor
    if np.any(edit & protected):
        raise RuntimeError("Mask subtraction failed: editable and protected pixels overlap")
    return edit, protected


def expand_edit(edit: np.ndarray, protection: np.ndarray, dilate_px: int = 0,
                temporal: int = 0, fill: bool = True) -> np.ndarray:
    """Expansion precedes the FINAL subtraction of protected pixels."""
    region = mk.temporal_union(edit, temporal) if temporal else edit.copy()
    expanded = []
    for m, excluded in zip(region, protection):
        candidate = mk.dilate(m, dilate_px)
        if fill:
            candidate = mk.fill_holes(candidate)
        expanded.append(candidate & ~excluded)
    return np.stack(expanded)


def refine_part_mask(mask: np.ndarray, original_size: tuple[int, int], settings: dict) -> np.ndarray:
    """Scale/clean one tracked garment mask. No color or RGB image upscaling.

    With all studio defaults this preserves the previous NEAREST behavior exactly.
    Mask operations run at original frame coordinates, not at a hidden small size.
    """
    import cv2

    w, h = original_size
    m = np.asarray(mask, dtype=bool)
    if m.shape != (h, w):
        algorithm = settings.get("part_mask_resample", "nearest")
        interpolation = cv2.INTER_NEAREST if algorithm == "nearest" else cv2.INTER_LINEAR
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=interpolation) >= 0.5
    else:
        m = m.copy()
    blur = int(settings.get("part_mask_blur_px", 0))
    if blur > 0:
        sigma = blur / 2.0
        soft = cv2.GaussianBlur(m.astype(np.float32), (2 * blur + 1, 2 * blur + 1), sigma)
        m = soft >= 0.5
    delta = int(settings.get("part_mask_dilate_px", 0))
    if delta > 0:
        m = mk.dilate(m, delta)
    elif delta < 0:
        m = mk.erode(m, -delta)
    if settings.get("part_fill_holes", False):
        m = mk.fill_holes(m)
    min_size = int(settings.get("part_min_component_px", 0))
    if min_size:
        count, cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_size
        m = keep[cc]
    return m
