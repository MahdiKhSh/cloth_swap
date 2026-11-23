"""Tests for the model-free parts: geometry, masks, track bookkeeping, config, video I/O."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloth_swap import masks as mk  # noqa: E402
from cloth_swap.config import load_config  # noqa: E402
from cloth_swap.geometry import (bbox, expand_box, reflect_indices,  # noqa: E402
                                 scail_padded_length, working_size)
from cloth_swap.tracks import (ConceptTracks, actor_masks, masks_within,  # noqa: E402
                               select_actor)


def test_working_size_keeps_aspect_and_multiple():
    w, h = working_size(1920, 800, 704 * 1280, 32)
    assert (w, h) == (1472, 608)
    assert w % 32 == 0 and h % 32 == 0
    w, h = working_size(1080, 1920, 704 * 1280, 32)
    assert w % 32 == 0 and h % 32 == 0 and h > w


@pytest.mark.parametrize("n,expected", [(1, 5), (5, 5), (6, 9), (80, 81), (81, 81), (82, 157), (157, 157), (158, 233)])
def test_scail_padded_length(n, expected):
    assert scail_padded_length(n, 81, 5) == expected


def test_padded_length_covers_all_windows():
    for n in range(1, 400):
        total = scail_padded_length(n, 81, 5)
        assert total >= n
        if total > 81:
            assert (total - 81) % 76 == 0


def test_reflect_indices():
    assert reflect_indices(4, 9) == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert reflect_indices(1, 3) == [0, 0, 0]


def test_content_box_ignores_logo_in_letterbox():
    from cloth_swap.geometry import content_box

    frames = np.zeros((4, 90, 160, 3), np.uint8)
    frames[:, 15:75] = 120                    # picture band
    frames[:, 80:88, 4:14] = (220, 200, 40)   # logo inside the lower bar
    assert content_box(frames) == (0, 15, 160, 75)


def test_edit_size_keeps_crop_aspect():
    from cloth_swap.geometry import edit_size

    w, h = edit_size(1100, 1700)
    assert w % 32 == 0 and h % 32 == 0 and abs(w / h - 1100 / 1700) < 0.03
    assert 0.9e6 < w * h < 1.15e6


def test_edit_prompt_builtin_plus_user_extra():
    from cloth_swap.keyframe import edit_prompt

    cfg = load_config()
    one = edit_prompt(cfg, 1, "a brown shirt", "Tuck the shirt in.")
    assert "image 2 (a brown shirt)" in one and one.endswith("Additional instructions: Tuck the shirt in.")
    assert "pixel-identical" in one
    two = edit_prompt(cfg, 2, None)
    assert "images 2 and 3" in two
    assert "a red coat" in edit_prompt(cfg, 0, "a red coat")
    with pytest.raises(ValueError):
        edit_prompt(cfg, 0, "")


def test_bbox_and_expand():
    m = np.zeros((10, 20), bool)
    m[2:5, 3:9] = True
    assert bbox(m) == (3, 2, 9, 5)
    assert bbox(np.zeros((4, 4), bool)) is None
    assert expand_box((3, 2, 9, 5), 0.5, 20, 10) == (0, 0, 12, 7)


def test_fill_holes_and_morphology():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    m[8:12, 8:12] = False
    filled = mk.fill_holes(m)
    assert filled[10, 10] and not filled[0, 0]
    assert mk.dilate(m, 2).sum() > m.sum()
    assert mk.erode(filled, 2).sum() < filled.sum()


def test_scail_mask_colours():
    m = np.zeros((4, 4), bool)
    m[1, 1] = True
    white = mk.scail_mask_frame(m, "white")
    assert tuple(white[1, 1]) == (0, 0, 255) and tuple(white[0, 0]) == (255, 255, 255)
    black = mk.scail_mask_frame(m, "black")
    assert tuple(black[1, 1]) == (0, 0, 255) and tuple(black[0, 0]) == (0, 0, 0)


def test_mask_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    m = rng.random((3, 7, 13)) > 0.5
    mk.save_masks(tmp_path / "m.npz", m)
    assert np.array_equal(mk.load_masks(tmp_path / "m.npz"), m)


def _square(h, w, x, y, s):
    m = np.zeros((h, w), bool)
    m[y:y + s, x:x + s] = True
    return m


def test_actor_selection_and_id_stitching():
    tracks = ConceptTracks(num_frames=6, height=40, width=60)
    # object 1: the actor, lost after frame 2; object 7 re-appears at the same place on frame 4
    for t in range(3):
        tracks.add(1, t, _square(40, 60, 10 + t, 10, 10))
    for t in range(4, 6):
        tracks.add(7, t, _square(40, 60, 13, 10, 10))
    # object 2: another person, visible throughout
    for t in range(6):
        tracks.add(2, t, _square(40, 60, 40, 5, 15))
    tracks.prompt_of.update({1: "person", 2: "person", 7: "person"})

    assert select_actor(tracks, "person", 0)[0] == 2              # largest
    assert select_actor(tracks, "person", 0, point=(14, 14))[0] == 1
    assert select_actor(tracks, "person", 0, box=(10, 10, 20, 20))[0] == 1

    masks = actor_masks(tracks, 1, "person", stitch_iou=0.2, gap_fill=1)
    assert masks[3].any()                  # gap filled from frame 2
    assert masks[4].any() and masks[5].any()  # object 7 adopted
    assert not (masks & _square(40, 60, 40, 5, 15)[None]).any()  # never grabbed the other person


def test_default_choice_prefers_screen_presence_over_size():
    # a passer-by fills the frame for 3 frames; the actor is smaller but in all 20 frames
    tracks = ConceptTracks(num_frames=20, height=40, width=60)
    for t in range(3):
        tracks.add(5, t, _square(40, 60, 20, 0, 35))
    for t in range(20):
        tracks.add(9, t, _square(40, 60, 2, 20, 12))
    tracks.prompt_of.update({5: "person", 9: "person"})
    assert select_actor(tracks, "person", 0)[0] == 9


def test_stitching_ignores_tracks_of_very_different_size():
    tracks = ConceptTracks(num_frames=3, height=40, width=60)
    tracks.add(1, 0, _square(40, 60, 10, 10, 8))
    tracks.add(4, 2, _square(40, 60, 5, 5, 30))  # big new track covering the old position
    tracks.prompt_of.update({1: "person", 4: "person"})
    masks = actor_masks(tracks, 1, "person", stitch_iou=0.05, gap_fill=0)
    assert not masks[2].any()


def test_drop_occluded_removes_blob_on_occluder():
    from cloth_swap.tracks import drop_occluded

    masks = np.stack([_square(40, 60, 10, 10, 12)] * 4 + [_square(40, 60, 40, 30, 4)])
    out = drop_occluded(masks)
    assert out[:4].all(axis=(1, 2)).sum() == 0 and out[:4].any(axis=(1, 2)).all()
    assert not out[4].any()


def test_masks_within_region():
    tracks = ConceptTracks(num_frames=1, height=20, width=20)
    tracks.add(3, 0, _square(20, 20, 2, 2, 4))
    tracks.add(4, 0, _square(20, 20, 14, 14, 4))
    tracks.prompt_of.update({3: "face", 4: "face"})
    region = _square(20, 20, 0, 0, 8)[None]
    out = masks_within(tracks, "face", region)
    assert out[0, 3, 3] and not out[0, 15, 15]


def test_config_overrides():
    cfg = load_config(overrides=["scail.steps=40", "actor.point=[10, 20]"])
    assert cfg["scail"]["steps"] == 40
    assert cfg["actor"]["point"] == [10, 20]
    with pytest.raises(ValueError):
        load_config(overrides=["scail.steps"])


def test_prepare_scail_inputs_colours_survive_encoding(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from PIL import Image

    from cloth_swap.scail2_stage import prepare_inputs
    from cloth_swap.video_io import read_frames

    cfg = load_config(overrides=["prompt.garment=a red jacket"])
    frames = np.full((7, 96, 160, 3), 90, np.uint8)
    actor = np.zeros((7, 96, 160), bool)
    actor[:, 20:80, 50:100] = True
    inputs = prepare_inputs(tmp_path, frames, actor, [(frames[0], actor[0])], (128, 64), 24.0, cfg)

    assert inputs.padded_frames == 9
    mask_video = read_frames(inputs.replace_mask)
    assert mask_video.shape == (9, 64, 128, 3)
    on = mask_video >= 225  # SCAIL-2's per-channel threshold
    centre, corner = on[:, 32, 64], on[:, 2, 2]
    assert (centre == [False, False, True]).all()   # blue = character
    assert corner.all()                             # white = keep
    ref_mask = np.asarray(Image.open(inputs.ref_mask))
    assert tuple(ref_mask[32, 64]) == (0, 0, 255) and tuple(ref_mask[2, 2]) == (0, 0, 0)
    assert "a red jacket" in inputs.prompt_file.read_text(encoding="utf-8")


def test_alpha_keeps_background():
    from cloth_swap.composite import AlphaBuilder, composite

    cfg = load_config()
    actor = np.zeros((3, 100, 100), bool)
    actor[:, 20:90, 30:70] = True
    alpha = AlphaBuilder(cfg, 100, actor)
    a = alpha(1)
    assert a[60, 50] > 0.99   # body: generated
    assert a[31, 50] > 0.99   # face too: the model is told to keep it
    assert a[2, 2] < 0.01     # background: original
    original = np.zeros((3, 100, 100, 3), np.uint8)
    generated = np.full_like(original, 255)
    out = list(composite(original, generated, alpha))
    assert out[0][60, 50, 0] > 250 and out[0][2, 2, 0] < 5


def test_prompt_always_says_keep_face_and_hair():
    from cloth_swap.scail2_stage import build_prompt

    cfg = load_config()
    cfg["prompt"].update(subject="A woman", garment="a red hoodie")
    assert "same face" in build_prompt(cfg) and "hairstyle" in build_prompt(cfg)
    cfg["scail"]["prompt"] = "A woman in a red hoodie plays the violin."
    text = build_prompt(cfg)
    assert text.startswith("A woman in a red hoodie") and "same face" in text


def test_video_roundtrip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from cloth_swap.video_io import probe, read_frames, write_video

    frames = np.zeros((9, 32, 48, 3), np.uint8)
    for t in range(9):
        frames[t, :, : 5 * (t + 1)] = 200
    path = tmp_path / "v.mp4"
    write_video(path, frames, fps=24, crf=0)
    info = probe(path)
    assert (info.width, info.height, info.num_frames) == (48, 32, 9)
    back = read_frames(path, start=2, count=3)
    assert back.shape == (3, 32, 48, 3)
    assert np.abs(back.astype(int) - frames[2:5].astype(int)).mean() < 3
