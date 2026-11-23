"""Bookkeeping for SAM3 concept tracks and choosing/following the actor (NumPy only)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import bbox, box_mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


@dataclass
class ConceptTracks:
    """Masks of every object SAM3 found, bit-packed per frame to keep memory low."""

    num_frames: int
    height: int
    width: int
    prompt_of: dict[int, str] = field(default_factory=dict)
    packed: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    areas: dict[int, dict[int, int]] = field(default_factory=dict)

    def add(self, obj_id: int, frame: int, mask: np.ndarray) -> None:
        area = int(mask.sum())
        if area:
            self.packed.setdefault(obj_id, {})[frame] = np.packbits(mask, axis=-1)
            self.areas.setdefault(obj_id, {})[frame] = area

    def presence(self, obj_id: int) -> float:
        """Screen presence: summed area (as a fraction of the frame) times the fraction of frames visible."""
        areas = self.areas.get(obj_id, {})
        return sum(areas.values()) / (self.height * self.width) * len(areas) / self.num_frames

    def has(self, obj_id: int, frame: int) -> bool:
        return frame in self.packed.get(obj_id, {})

    def get(self, obj_id: int, frame: int) -> np.ndarray:
        bits = self.packed.get(obj_id, {}).get(frame)
        if bits is None:
            return np.zeros((self.height, self.width), bool)
        return np.unpackbits(bits, axis=-1, count=self.width).astype(bool)

    def objects(self, prompt: str) -> list[int]:
        return sorted(o for o, p in self.prompt_of.items() if p == prompt and o in self.packed)

    def present(self, prompt: str, frame: int) -> list[int]:
        return [o for o in self.objects(prompt) if self.has(o, frame)]


def select_actor(tracks: ConceptTracks, prompt: str, frame: int,
                 point=None, box=None) -> tuple[int, int]:
    """Pick the actor's object id. Returns (obj_id, frame the choice was made on).

    point: the person whose mask contains it (smallest such mask), else the nearest box centre.
    box: the person with the highest IoU against the box.
    Neither: the person with the most screen presence over the whole clip, so passers-by close to
    the camera do not win over the actor who stays in the shot.
    """
    if point is None and box is None:
        people = tracks.objects(prompt)
        if not people:
            raise RuntimeError(f"SAM3 found no '{prompt}' in the clip. Lower sam3.score_threshold or use actor.mode=track.")
        best = max(people, key=tracks.presence)
        return best, min(tracks.packed[best])

    order = list(range(frame, tracks.num_frames)) + list(range(frame - 1, -1, -1))
    for t in order:
        candidates = tracks.present(prompt, t)
        if candidates:
            break
    else:
        raise RuntimeError(f"SAM3 found no '{prompt}' in the clip. Lower sam3.score_threshold or use actor.mode=track.")

    masks = {o: tracks.get(o, t) for o in candidates}
    if point is not None:
        x, y = (int(round(v)) for v in point)
        inside = [o for o, m in masks.items() if 0 <= y < tracks.height and 0 <= x < tracks.width and m[y, x]]
        if inside:
            return min(inside, key=lambda o: masks[o].sum()), t

        def centre_distance(o):
            x0, y0, x1, y1 = bbox(masks[o])
            return (x - (x0 + x1) / 2) ** 2 + (y - (y0 + y1) / 2) ** 2
        return min(masks, key=centre_distance), t
    target = box_mask(box, tracks.height, tracks.width)
    return max(masks, key=lambda o: mask_iou(masks[o], target)), t


def actor_masks(tracks: ConceptTracks, actor_id: int, prompt: str,
                stitch_iou: float, gap_fill: int) -> np.ndarray:
    """(T, H, W) masks of the actor, following ID switches after occlusions.

    When none of the actor's ids is present, a track that was not visible on the actor's last
    frame, overlaps the actor's last mask by >= stitch_iou and has a similar size (within 2x)
    is adopted as the actor.
    """
    out = np.zeros((tracks.num_frames, tracks.height, tracks.width), bool)
    ids = {actor_id}
    last_mask, last_t = None, None
    for t in range(tracks.num_frames):
        present = [o for o in ids if tracks.has(o, t)]
        if not present and last_mask is not None:
            best, best_iou = None, 0.0
            last_area = last_mask.sum()
            for o in tracks.present(prompt, t):
                if o in ids or tracks.has(o, last_t):
                    continue
                if not 0.5 <= tracks.areas[o][t] / last_area <= 2.0:
                    continue
                iou = mask_iou(tracks.get(o, t), last_mask)
                if iou > best_iou:
                    best, best_iou = o, iou
            if best is not None and best_iou >= stitch_iou:
                ids.add(best)
                present = [best]
        if present:
            mask = np.zeros((tracks.height, tracks.width), bool)
            for o in present:
                mask |= tracks.get(o, t)
            out[t] = mask
            last_mask, last_t = mask, t
        elif last_mask is not None and t - last_t <= gap_fill:
            out[t] = last_mask
    return out


def drop_occluded(masks: np.ndarray, min_iou: float = 0.05, min_area: float = 0.4) -> np.ndarray:
    """Empty the frames where the actor's mask jumps to something else.

    When an occluder passes right in front of the camera, SAM3 sometimes leaves a small blob on
    the occluder instead of an empty mask. A mask that is much smaller than the actor's median
    and barely overlaps the last trusted mask is treated as "actor not visible".
    """
    out = masks.copy()
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    visible = areas > 0
    if not visible.any():
        return out
    median = np.median(areas[visible])
    last = None
    for t in range(len(masks)):
        if not visible[t]:
            continue
        if last is not None and areas[t] < min_area * median and mask_iou(masks[t], last) < min_iou:
            out[t] = False
            continue
        last = masks[t]
    return out


def masks_within(tracks: ConceptTracks, prompt: str, region: np.ndarray,
                 min_overlap: float = 0.3) -> np.ndarray:
    """Union per frame of `prompt` objects with at least `min_overlap` of their area inside `region`."""
    out = np.zeros_like(region)
    for o in tracks.objects(prompt):
        for t in tracks.packed[o]:
            mask = tracks.get(o, t)
            area = mask.sum()
            if area and np.logical_and(mask, region[t]).sum() >= min_overlap * area:
                out[t] |= mask
    return out
