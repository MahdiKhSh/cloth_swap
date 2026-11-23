"""Blend generated frames back over the original footage."""
from __future__ import annotations

import numpy as np

from . import masks as mk
from .geometry import box_mask


class AlphaBuilder:
    """Per-frame alpha: where the generated frame replaces the original."""

    def __init__(self, cfg: dict, height: int, actor: np.ndarray, old_garment: np.ndarray | None = None,
                 new_garment: np.ndarray | None = None, content=None,
                 edit: np.ndarray | None = None, protected: np.ndarray | None = None):
        c = cfg["composite"]
        # letterbox bars always stay original
        self.content = box_mask(content, *actor.shape[1:]).astype(np.float32) if content else None
        self.mode = c["mode"]
        self.radius = c["temporal"]
        self.dilate_px = max(1, int(round(c["dilate"] * height)))
        self.feather_px = max(1, int(round(c["feather"] * height)))
        self.actor = actor
        self.edit = edit
        self.protected = protected
        if self.mode == "garment":
            if old_garment is None or new_garment is None:
                raise ValueError("composite.mode=garment needs sam3.old_garment_prompt and sam3.new_garment_prompt")
            self.garment = old_garment | new_garment
        elif self.mode not in ("actor", "full"):
            raise ValueError(f"Unknown composite.mode: {self.mode}")

    def __call__(self, t: int) -> np.ndarray:
        alpha = self._alpha(t)
        if self.edit is not None:
            # For layered edit, union/feather must not bring back protected pixels.
            alpha = alpha * mk.dilate(self.edit[t], self.dilate_px)
        if self.protected is not None:
            alpha[mk.dilate(self.protected[t], 2)] = 0.0
        return alpha * self.content if self.content is not None else alpha

    def _alpha(self, t: int) -> np.ndarray:
        if self.edit is not None:
            region = mk.union_window(self.edit, t, self.radius)
            region = mk.fill_holes(mk.dilate(region, self.dilate_px))
            return mk.feather(region, self.feather_px)
        if self.mode == "full":
            return np.ones(self.actor.shape[1:], np.float32)
        if self.mode == "garment":
            region = mk.union_window(self.garment, t, self.radius)
            region &= mk.dilate(mk.union_window(self.actor, t, self.radius), 2 * self.dilate_px)
        else:
            region = mk.union_window(self.actor, t, self.radius)
        region = mk.fill_holes(mk.dilate(region, self.dilate_px))
        return mk.feather(region, self.feather_px)


def composite(original: np.ndarray, generated: np.ndarray, alpha_for):
    """Yield blended frames."""
    for t in range(len(original)):
        yield mk.blend(generated[t], original[t], alpha_for(t))


def side_by_side(left: np.ndarray, right: np.ndarray, scale: float = 0.5):
    import cv2

    for a, b in zip(left, right):
        frame = np.concatenate([a, b], axis=1)
        h, w = frame.shape[:2]
        size = (max(2, int(w * scale)) // 2 * 2, max(2, int(h * scale)) // 2 * 2)
        yield cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
