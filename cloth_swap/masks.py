"""Mask morphology, storage and the colour encoding SCAIL-2 expects."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

# SCAIL-Pose paints the first tracked character in palette colour 0 (pure blue). The model reads
# each RGB channel with a threshold of 225, so blue = character slot, white = untouched area.
SCAIL_CHARACTER_RGB = (0, 0, 255)


def _kernel(px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _kernel(px)) > 0


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    return cv2.erode(mask.astype(np.uint8), _kernel(px)) > 0


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill regions not connected to the image border."""
    padded = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(padded, None, (0, 0), 2)
    holes = padded[1:-1, 1:-1] == 0
    return mask | holes


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    """Soft float32 version of a boolean mask."""
    soft = mask.astype(np.float32)
    if px <= 0:
        return soft
    sigma = px / 2.0
    ksize = int(2 * math.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(soft, (ksize, ksize), sigma)


def union_window(masks: np.ndarray, t: int, radius: int) -> np.ndarray:
    """Union of masks[t-radius .. t+radius]."""
    if radius <= 0:
        return masks[t].copy()
    return masks[max(0, t - radius): t + radius + 1].any(axis=0)


def temporal_union(masks: np.ndarray, radius: int) -> np.ndarray:
    out = masks.copy()
    for r in range(1, radius + 1):
        out[r:] |= masks[:-r]
        out[:-r] |= masks[r:]
    return out


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to size=(width, height)."""
    width, height = size
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask.copy()
    interp = cv2.INTER_AREA if width < mask.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=interp) >= 0.5


def scail_mask_frame(mask: np.ndarray, background: str) -> np.ndarray:
    """RGB frame with the character in blue on a white (driving mask) or black (reference mask) background."""
    value = {"white": 255, "black": 0}[background]
    frame = np.full(mask.shape + (3,), value, np.uint8)
    frame[mask] = SCAIL_CHARACTER_RGB
    return frame


def blend(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    out = foreground.astype(np.float32) * a + background.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def overlay(frame: np.ndarray, mask: np.ndarray, color=(255, 0, 0), strength: float = 0.45) -> np.ndarray:
    out = frame.copy()
    tint = np.array(color, np.float32)
    out[mask] = (out[mask].astype(np.float32) * (1 - strength) + tint * strength).astype(np.uint8)
    return out


def save_masks(path: str | Path, masks: np.ndarray) -> None:
    np.savez_compressed(path, bits=np.packbits(masks, axis=-1), width=masks.shape[-1])


def save_packed_masks(path: str | Path, bits: np.ndarray, width: int) -> None:
    """Write pre-packed frame masks without a several-GB unpacked T×H×W array."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 3 or width < 1 or bits.shape[-1] != (width + 7) // 8:
        raise ValueError("Packed mask shape or width is invalid")
    np.savez_compressed(path, bits=bits, width=int(width))


def load_packed_masks(path: str | Path) -> tuple[np.ndarray, int]:
    """Read compact masks without expanding all frames to T×H×W booleans."""
    with np.load(path) as data:
        bits, width = data["bits"], int(data["width"])
    if bits.ndim != 3 or width <= 0 or bits.shape[-1] != (width + 7) // 8:
        raise ValueError("Invalid packed mask array")
    return bits, width


def load_masks(path: str | Path) -> np.ndarray:
    bits, width = load_packed_masks(path)
    return np.unpackbits(bits, axis=-1, count=width).astype(bool)
