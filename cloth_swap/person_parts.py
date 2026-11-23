"""Second-stage SAM3 masks, restricted to the previously tracked person.

The actor mask remains intact for pose extraction.  The edit mask never contains
protected pixels, and uncertain frames fail closed rather than editing a head
that SAM3 could not identify. All arrays use [frame, height, width].
"""
from __future__ import annotations

import numpy as np

from . import masks as mk


def isolate_person(frames: np.ndarray, actor: np.ndarray, background: int = 127,
                   copy: bool = True) -> np.ndarray:
    """Keep person pixels at their ORIGINAL coordinates; hide other people/background."""
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.shape[:3] != actor.shape:
        raise ValueError("Actor mask and source frames must have matching [T,H,W] dimensions")
    isolated = frames.copy() if copy else frames
    isolated[~actor.astype(bool)] = background
    return isolated


def derive_edit_masks(actor: np.ndarray, face: np.ndarray, hair: np.ndarray,
                      padding_px: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (face, hair, protect, edit, quality) with fail-closed missing detections.

    If neither part is found anywhere, the run is unusable: report an error.
    For a visible frame missing *either* part, protect the entire actor on that
    frame. It is preferable to skip clothing replacement there than alter a
    face/hair that the model failed to detect. Users can re-prompt and re-run.
    """
    actor = np.asarray(actor, bool)
    face = np.asarray(face, bool)
    hair = np.asarray(hair, bool)
    if actor.ndim != 3 or face.shape != actor.shape or hair.shape != actor.shape:
        raise ValueError("Person, face and hair tracks must have identical [T,H,W] shapes")
    face &= actor
    hair &= actor
    if not actor.any():
        raise ValueError("No tracked person pixels available")
    if not face.any() or not hair.any():
        missing = 'face' if not face.any() else 'hair'
        raise ValueError(f"SAM3 did not find {missing} on the isolated person; review/retry this stage")
    visible = actor.reshape(len(actor), -1).any(axis=1)
    has_face = face.reshape(len(face), -1).any(axis=1)
    has_hair = hair.reshape(len(hair), -1).any(axis=1)
    uncertain = visible & ~(has_face & has_hair)
    protect = np.stack([mk.dilate(f | h, padding_px) for f, h in zip(face, hair)])
    protect[uncertain] |= actor[uncertain]  # never edit uncertain frames
    edit = actor & ~protect
    if not edit.any():
        raise ValueError("Every visible frame lacks reliable face/hair protection; edit mask would be empty")
    quality = {
        "visible_frames": int(visible.sum()),
        "face_frames": int((has_face & visible).sum()),
        "hair_frames": int((has_hair & visible).sum()),
        "skipped_frames": int(uncertain.sum()),
        "skipped_indices": np.flatnonzero(uncertain).tolist(),
        "edited_pixels": int(edit.sum()),
        "protected_pixels": int(protect.sum()),
    }
    return face, hair, protect, edit, quality
