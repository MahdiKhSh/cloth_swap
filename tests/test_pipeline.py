"""Runs the whole pipeline on a synthetic clip with the models replaced by fakes."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("imageio_ffmpeg")

from cloth_swap import keyframe, scail2_stage, segmentation  # noqa: E402
from cloth_swap.config import load_config  # noqa: E402
from cloth_swap.pipeline import Pipeline  # noqa: E402
from cloth_swap.tracks import ConceptTracks  # noqa: E402
from cloth_swap.video_io import probe, read_frames, write_video  # noqa: E402

H, W, T = 72, 128, 10


def _person(t):
    m = np.zeros((H, W), bool)
    m[10:70, 30 + t:60 + t] = True
    return m


def _face(t):
    m = np.zeros((H, W), bool)
    m[12:22, 40 + t:50 + t] = True
    return m


@pytest.fixture
def fakes(monkeypatch):
    calls = {"scail": 0}

    def track_concepts(frames, prompts, cfg):
        tracks = ConceptTracks(len(frames), *frames.shape[1:3])
        for t in range(len(frames)):
            tracks.add(1, t, _person(t))
            tracks.add(2, t, _face(t))
        tracks.prompt_of.update({1: "person", 2: "face"})
        return tracks

    class FakeSegmenter:
        def __init__(self, cfg):
            pass

        def segment(self, image, prompt):
            return [_person(0)]

        def close(self):
            pass

    def edit_frames(frames, actor, indices, garment, garment_desc, cfg):
        out = []
        for t in indices:
            f = frames[t].copy()
            f[actor[t]] = (200, 30, 30)  # "new red garment"
            out.append(f)
        return out

    def run(inputs, out_video, cfg, **callbacks):
        calls["scail"] += 1
        frames = read_frames(inputs.driving)
        frames[..., 0] = 255  # mark generated frames
        write_video(out_video, frames, 24)

    monkeypatch.setattr(segmentation, "track_concepts", track_concepts)
    monkeypatch.setattr(keyframe, "ImageSegmenter", FakeSegmenter)
    monkeypatch.setattr(keyframe, "edit_frames", edit_frames)
    monkeypatch.setattr(scail2_stage, "run", run)
    return calls


def test_full_run_and_resume(tmp_path, fakes):
    video = tmp_path / "clip.mp4"
    write_video(video, np.full((T, H, W, 3), 60, np.uint8), fps=24, crf=0)
    out = tmp_path / "result.mp4"
    work = tmp_path / "work"
    cfg = load_config(overrides=["generation.backend=scail2", "scail.target_area=8192", "prompt.garment=a red jacket"])

    Pipeline(cfg, video, out, work, garment=None).run()
    info = probe(out)
    assert (info.width, info.height, info.num_frames) == (W, H, T)
    for stage in ("masks", "keyframe", "scail", "upscale", "composite"):
        assert (work / f"{stage}.done").exists()
    assert (work / "02_keyframe" / "keyframe_0.png").exists()
    assert (work / "03_scail" / "replace_mask.mp4").exists()

    result = read_frames(out)
    assert result[5, 2, 2, 0] < 100          # background kept from the original
    assert result[5, 50, 45 + 5, 0] > 200    # body comes from the generated frames

    Pipeline(cfg, video, out, work).run()     # everything cached
    assert fakes["scail"] == 1
    Pipeline(cfg, video, out, work).run(from_stage="scail")
    assert fakes["scail"] == 2


def test_stop_after_keyframe(tmp_path, fakes):
    video = tmp_path / "clip.mp4"
    write_video(video, np.full((T, H, W, 3), 60, np.uint8), fps=24, crf=0)
    work = tmp_path / "work"
    cfg = load_config(overrides=["generation.backend=scail2", "scail.target_area=8192", "prompt.garment=a red jacket"])
    Pipeline(cfg, video, tmp_path / "o.mp4", work).run(stop_after="keyframe")
    assert (work / "keyframe.done").exists() and not (work / "scail.done").exists()
    shutil.rmtree(work)


def test_4k_output_upscales_the_finished_shot(tmp_path, fakes, monkeypatch):
    from cloth_swap import upscale

    seen = {}

    def fake_seedvr2(source, target, short_side, cfg):
        seen["short_side"] = short_side
        frames = read_frames(source)
        big = np.repeat(np.repeat(frames, 2, axis=1), 2, axis=2)  # stands in for the 2x restore
        write_video(target, big, 24)
        return target

    monkeypatch.setattr(upscale, "seedvr2_file", fake_seedvr2)
    video = tmp_path / "clip.mp4"
    write_video(video, np.full((T, H, W, 3), 60, np.uint8), fps=24, crf=0)
    out = tmp_path / "result.mp4"
    cfg = load_config(overrides=["generation.backend=scail2", "scail.target_area=8192", "prompt.garment=a red jacket", "output.resolution=4k"])
    Pipeline(cfg, video, out, tmp_path / "work").run()
    info = probe(out)
    assert seen["short_side"] == 2160
    assert (info.width, info.height, info.num_frames) == (2 * W, 2 * H, T)
