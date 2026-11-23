"""Sizes, boxes and frame-index arithmetic."""
from __future__ import annotations

import math

import numpy as np


def working_size(width: int, height: int, target_area: int, multiple: int = 32) -> tuple[int, int]:
    """Size with about `target_area` pixels, the clip's aspect ratio, and sides divisible by `multiple`."""
    aspect = width / height
    w = math.sqrt(target_area * aspect)
    h = w / aspect
    w = max(multiple, int(round(w / multiple)) * multiple)
    h = max(multiple, int(round(h / multiple)) * multiple)
    return w, h


def round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple)) * multiple


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) with exclusive x1/y1, or None for an empty mask."""
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if ys.size == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def expand_box(box: tuple[int, int, int, int], margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    dx = int(round((x1 - x0) * margin))
    dy = int(round((y1 - y0) * margin))
    return max(0, x0 - dx), max(0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)


def box_mask(box, height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask = np.zeros((height, width), bool)
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return mask


def content_box(frames, threshold: int = 18) -> tuple[int, int, int, int]:
    """Picture area without letterbox/pillarbox bars: (x0, y0, x1, y1), exclusive end.

    A row/column counts as a bar when most of it stays darker than `threshold` in every sampled
    frame (a logo or subtitle inside the bar does not make it picture).
    """
    sample = np.asarray(frames)[:: max(1, len(frames) // 12)]
    gray = sample.max(axis=-1).max(axis=0)          # brightest value seen per pixel
    rows = np.flatnonzero(np.percentile(gray, 70, axis=1) > threshold)
    cols = np.flatnonzero(np.percentile(gray, 70, axis=0) > threshold)
    h, w = gray.shape
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def clamp_box(box, limit) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    lx0, ly0, lx1, ly1 = limit
    return max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)


def edit_size(width: int, height: int, area: int = 1024 * 1024, multiple: int = 32) -> tuple[int, int]:
    """Output size for the image editor: about `area` pixels with the crop's aspect ratio."""
    ratio = width / height
    w = math.sqrt(area * ratio)
    h = w / ratio
    return max(multiple, int(round(w / multiple)) * multiple), max(multiple, int(round(h / multiple)) * multiple)


def scail_padded_length(num_frames: int, segment_len: int, overlap: int) -> int:
    """Smallest length >= num_frames that SCAIL-2 generates completely.

    Clips up to `segment_len` are trimmed to 4k+1 frames by the VAE; longer clips are cut into
    windows of `segment_len` advancing by `segment_len - overlap`, and an incomplete tail window
    is dropped. Padding the input to this length means no real frame is lost.
    """
    if segment_len % 4 != 1:
        raise ValueError(f"segment_len must be 4k+1, got {segment_len}")
    if not 0 < overlap < segment_len:
        raise ValueError("segment_overlap must be between 0 and segment_len")
    if num_frames <= segment_len:
        return max(5, (num_frames - 1 + 3) // 4 * 4 + 1)
    stride = segment_len - overlap
    windows = math.ceil((num_frames - segment_len) / stride)
    return segment_len + windows * stride


def reflect_indices(num_frames: int, total: int) -> list[int]:
    """0..num_frames-1, then bounce back and forth so padded frames keep moving naturally."""
    if num_frames == 1:
        return [0] * total
    period = 2 * (num_frames - 1)
    indices = []
    for i in range(total):
        j = i % period
        indices.append(j if j < num_frames else period - j)
    return indices
