"""Regression tests for ComfyUI-style invert-union-invert mask subtraction."""
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_swap import masks as mk
from cloth_swap.mask_layers import compose, expand_edit
from cloth_swap.studio.worker import _compose_mask_layers
from cloth_swap.studio.service import StudioService


def masks_fixture():
    actor = np.zeros((3, 64, 80), dtype=bool)
    actor[:, 5:59, 12:70] = True
    head = np.zeros_like(actor)
    head[:, 8:24, 32:51] = True
    shoes = np.zeros_like(actor)
    shoes[:, 52:58, 28:60] = True
    arm = np.zeros_like(actor)
    arm[:, 24:46, 12:23] = True
    return actor, head, shoes, arm


def test_invert_union_invert_excludes_multiple_objects_and_keeps_body():
    actor, head, shoes, arm = masks_fixture()
    rules = [
        {"name": "head", "operation": "subtract"},
        {"name": "shoes", "operation": "subtract"},
        {"name": "arm", "operation": "ignore"},
    ]
    edit, protect = compose(actor, {"head": head, "shoes": shoes, "arm": arm}, rules)
    expected = actor & ~(head | shoes)
    assert np.array_equal(edit, expected)
    assert np.array_equal(protect, head | shoes)
    assert not (edit & head).any()
    assert not (edit & shoes).any()
    assert edit[:, 30:42, 40:55].all()  # body retained
    # Even dilation and temporal union in the WAN mask must not reintroduce exclusion.
    expanded = expand_edit(edit, protect, dilate_px=4, temporal=2)
    assert not (expanded & protect).any()


def test_mask_can_use_independent_positive_objects_without_actor_base():
    actor, head, shoes, arm = masks_fixture()
    edit, protect = compose(actor, {"head": head, "shoes": shoes, "arm": arm}, [
        {"name": "arm", "operation": "add"},
        {"name": "head", "operation": "subtract"},
        {"name": "shoes", "operation": "ignore"},
    ], base_actor=False)
    assert np.array_equal(edit, arm)
    assert np.array_equal(protect, head)
    assert not edit[:, 10, 35].any()


def test_packed_worker_recomposition_produces_same_masks(tmp_path):
    actor, head, shoes, arm = masks_fixture()
    folder = tmp_path / "layers"
    folder.mkdir()
    for name, array in [("head", head), ("shoes", shoes), ("arm", arm)]:
        mk.save_masks(folder / f"{name}.npz", array)
    mk.save_masks(tmp_path / "actor.npz", actor)
    args = {"actor_path": str(tmp_path / "actor.npz"), "output_dir": str(folder),
            "count": len(actor), "base_actor": True,
            "edit_path": str(tmp_path / "edit.npz"),
            "protect_path": str(tmp_path / "protect.npz"),
            "rules": [
                {"name": "head", "operation": "subtract"},
                {"name": "shoes", "operation": "subtract"},
                {"name": "arm", "operation": "ignore"},
            ]}
    reported = _compose_mask_layers({}, {}, args, lambda *_: None)
    edit = mk.load_masks(args["edit_path"])
    protect = mk.load_masks(args["protect_path"])
    assert np.array_equal(edit, actor & ~protect)
    assert not (edit & head).any() and not (edit & shoes).any()
    assert reported["protected_pixels"] > 0
    assert reported["edited_pixels"] > 0
    assert not (protect & ~actor).any()
    # Flipping formula only should not retrack: recompose already saved masks.
    args["rules"][1]["operation"] = "ignore"
    _compose_mask_layers({}, {}, args, lambda *_: None)
    edit2 = mk.load_masks(args["edit_path"])
    assert (edit2 & shoes).any()
    assert not (edit2 & head).any()


def test_recomposition_refuses_empty_subtraction_not_silent_success(tmp_path):
    actor, head, _, _ = masks_fixture()
    bad = np.zeros_like(actor)
    folder = tmp_path / "layers"
    folder.mkdir()
    mk.save_masks(folder / "missing.npz", bad)
    mk.save_masks(tmp_path / "actor.npz", actor)
    args = {"actor_path": str(tmp_path / "actor.npz"), "output_dir": str(folder),
            "count": len(actor), "base_actor": True,
            "edit_path": str(tmp_path / "edit.npz"),
            "protect_path": str(tmp_path / "protect.npz"),
            "rules": [{"name": "missing", "operation": "subtract"}]}
    with pytest.raises(ValueError, match="no pixels inside"):
        _compose_mask_layers({}, {}, args, lambda *_: None)


def test_retracking_actor_invalidates_old_object_tracks(tmp_path):
    service = StudioService(tmp_path / "store", None)
    class FakeProject:
        id = "project1"
        state = {"actor": {"prompts": [{"frame": 0, "points": [[10, 10]], "labels": [1], "box": None}],
                           "tracked": True},
                 "mask_layers": {"rules": [{"name": "obj", "operation": "subtract"}], "ready": True,
                                 "manual": True, "tracked": True, "object_signature": "old", "formula_signature": "old"}}
        def update(self, section, values):
            self.state.setdefault(section, {}).update(values)
    p = FakeProject()
    service.jobs = SimpleNamespace(submit=lambda *args, **kwargs: "submitted")
    assert service.track(p) == "submitted"
    assert not p.state["mask_layers"]["tracked"]
    assert not p.state["mask_layers"]["ready"]
    assert p.state["mask_layers"]["object_signature"] == ""
