"""Bring SCAIL-2 frames from the working size to the original size."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from .video_io import read_frames, resize_frames, write_video

log = logging.getLogger(__name__)


def to_original_size(frames: np.ndarray, size: tuple[int, int], fps: float, cfg: dict, work_dir: Path) -> np.ndarray:
    backend = cfg["upscale"]["backend"]
    if backend == "seedvr2":
        frames = _seedvr2(frames, size, fps, cfg, work_dir)
    elif backend != "resize":
        raise ValueError(f"Unknown upscale.backend: {backend}")
    return resize_frames(frames, size)


def seedvr2_file(source: Path, target: Path, short_side: int, cfg: dict) -> Path:
    """Upscale a video file with SeedVR2 so its short side becomes `short_side` (it restores detail
    instead of only resizing). Runs inference_cli.py of ComfyUI-SeedVR2_VideoUpscaler in its own Python."""
    s = cfg["upscale"]["seedvr2"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    cmd = [s["python"], "inference_cli.py", str(Path(source).resolve()), "--output", str(Path(target).resolve()),
           "--model_dir", s["model_dir"], "--dit_model", s["dit_model"], "--resolution", str(int(short_side)),
           "--batch_size", str(s["batch_size"]), "--uniform_batch_size", "--temporal_overlap", str(s["temporal_overlap"]),
           "--prepend_frames", str(s.get("prepend_frames", 0)), "--color_correction", s["color_correction"],
           "--seed", str(s["seed"]), "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
           "--blocks_to_swap", str(s["blocks_to_swap"]), "--vae_encode_tiled", "--vae_decode_tiled",
           "--video_backend", "ffmpeg"]
    log.info("Running SeedVR2 (%s) to %dp", s["dit_model"], short_side)
    subprocess.run(cmd, cwd=s["cli_dir"], check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if not target.exists():
        raise FileNotFoundError(f"SeedVR2 did not write {target}")
    return target


def _seedvr2(frames: np.ndarray, size: tuple[int, int], fps: float, cfg: dict, work_dir: Path) -> np.ndarray:
    """Generated frames to the original size with SeedVR2 (sharper than resizing)."""
    source = work_dir / "seedvr2_in.mp4"
    write_video(source, frames, fps, crf=10)
    upscaled = read_frames(seedvr2_file(source, work_dir / "seedvr2_out.mp4", min(size), cfg))
    if len(upscaled) < len(frames):
        raise RuntimeError(f"SeedVR2 returned {len(upscaled)} frames for {len(frames)}")
    return upscaled[: len(frames)]
