"""Video reading/writing through the ffmpeg binary bundled with imageio-ffmpeg."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    num_frames: int


def probe(path: str | Path) -> VideoInfo:
    num_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    reader.close()
    width, height = meta["size"]
    return VideoInfo(str(path), int(width), int(height), float(meta["fps"]), int(num_frames))


def read_frames(path: str | Path, start: int = 0, count: int | None = None,
                size: tuple[int, int] | None = None) -> np.ndarray:
    """Read RGB frames (T,H,W,3), optionally scaling *inside FFmpeg*.

    For a known frame count, allocate the output array once rather than building a
    list and then making an equally large copy with np.stack. This is especially
    important for long 4K clips on Windows where RAM/commit is limited.
    """
    output_params = None
    if size is not None:
        w, h = map(int, size)
        if w <= 0 or h <= 0:
            raise ValueError("Read size must be positive")
        output_params = ["-vf", f"scale={w}:{h}:flags=fast_bilinear"]
    reader = imageio_ffmpeg.read_frames(str(path), output_params=output_params)
    meta = next(reader)
    width, height = meta["size"]
    frames = np.empty((count, height, width, 3), dtype=np.uint8) if count is not None else []
    seen = 0
    try:
        for index, buffer in enumerate(reader):
            if index < start:
                continue
            if count is not None and seen >= count:
                break
            frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
            if count is not None:
                frames[seen] = frame
            else:
                frames.append(frame.copy())
            seen += 1
    finally:
        reader.close()
    if not seen:
        raise ValueError(f"No frames read from {path} (start={start}, count={count})")
    if count is None:
        return np.stack(frames)
    return frames if seen == count else frames[:seen].copy()


def iter_frames(path: str | Path, start: int = 0, count: int | None = None):
    """Decode RGB frames one at a time without buffering the complete video.

    Useful for 4K compositing: a single 300-frame clip occupies ~2.5 GiB
    uncompressed, and the original + generated video used to be loaded at once.
    """
    if start < 0 or (count is not None and count < 1):
        raise ValueError("Invalid frame range")
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    delivered = 0
    try:
        for index, buffer in enumerate(reader):
            if index < start:
                continue
            if count is not None and delivered >= count:
                break
            delivered += 1
            yield np.frombuffer(buffer, np.uint8).reshape(height, width, 3).copy()
    finally:
        reader.close()


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: float, crf: int = 16) -> None:
    """Encode RGB uint8 frames to H.264. Odd sizes fall back to yuv444p."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out=pix_fmt,
        quality=None,
        macro_block_size=1,
        output_params=["-crf", str(crf), "-preset", "medium"],
    )
    writer.send(None)
    try:
        writer.send(np.ascontiguousarray(first))
        for frame in iterator:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize (T, H, W, 3) frames to size=(width, height)."""
    width, height = size
    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    interp = cv2.INTER_AREA if width < frames.shape[2] else cv2.INTER_LANCZOS4
    return np.stack([cv2.resize(f, (width, height), interpolation=interp) for f in frames])


def mux_audio(video: str | Path, audio_source: str | Path, out: str | Path,
              start_sec: float, duration_sec: float) -> None:
    """Copy the video stream and add the matching span of audio from the source (if it has any)."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video),
        "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}", "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True)
