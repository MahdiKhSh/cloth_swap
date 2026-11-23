"""Prepare SCAIL-2 replacement inputs and run the model in its own environment."""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import masks as mk
from .geometry import reflect_indices, scail_padded_length
from .video_io import read_frames, resize_frames, write_video

log = logging.getLogger(__name__)
RUNNER = Path(__file__).with_name("scail2_runner.py")


@dataclass
class ScailInputs:
    driving: Path
    replace_mask: Path
    ref_image: Path
    ref_mask: Path
    prompt_file: Path
    width: int
    height: int
    num_frames: int
    padded_frames: int
    extra_images: list[Path] = field(default_factory=list)
    extra_masks: list[Path] = field(default_factory=list)


def build_prompt(cfg: dict) -> str:
    p = cfg["prompt"]
    if cfg["scail"].get("prompt"):
        text = cfg["scail"]["prompt"].strip()
    else:
        garment = p.get("garment") or "the outfit shown in the reference image"
        text = f"{p.get('subject') or 'A person'}, wearing {garment}."
        if p.get("scene"):
            text += " " + p["scene"].strip()
    # only the clothes change: said in every prompt (with the distill LoRA there is no negative prompt)
    keep = (p.get("keep_identity") or "").strip()
    if keep and keep not in text:
        text += " " + keep
    return text


def prepare_inputs(out_dir: Path, frames: np.ndarray, actor: np.ndarray,
                   keyframes: list[tuple[np.ndarray, np.ndarray]], work_size: tuple[int, int],
                   fps: float, cfg: dict) -> ScailInputs:
    scfg = cfg["scail"]
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = work_size
    num_frames = len(frames)
    padded = scail_padded_length(num_frames, scfg["segment_len"], scfg["segment_overlap"])
    order = reflect_indices(num_frames, padded)
    log.info("SCAIL-2 input: %d frames padded to %d, %dx%d", num_frames, padded, width, height)

    driving = out_dir / "driving.mp4"
    write_video(driving, (cv2.resize(frames[i], (width, height), interpolation=cv2.INTER_AREA) for i in order),
                fps, crf=12)

    grow = max(1, int(round(scfg["replace_dilate"] * height)))
    region = mk.temporal_union(actor, scfg["replace_temporal"])
    protected = mk.load_masks(out_dir.parent / "01_masks" / "protect.npz") if cfg.get("mask_layers_enabled") else None

    def mask_frames():
        for i in order:
            small = mk.fill_holes(mk.dilate(mk.resize_mask(region[i], (width, height)), grow))
            if protected is not None:
                small &= ~mk.resize_mask(protected[i], (width, height))
            yield mk.scail_mask_frame(small, "white")

    replace_mask = out_dir / "replace_mask.mp4"
    write_video(replace_mask, mask_frames(), fps, crf=12)

    saved = []
    for k, (image, mask) in enumerate(keyframes):
        img_path = out_dir / f"ref_{k}.png"
        mask_path = out_dir / f"ref_mask_{k}.png"
        Image.fromarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)).save(img_path)
        Image.fromarray(mk.scail_mask_frame(mk.resize_mask(mask, (width, height)), "black")).save(mask_path)
        saved.append((img_path, mask_path))

    prompt_file = out_dir / "prompt.txt"
    prompt_file.write_text(build_prompt(cfg), encoding="utf-8")
    return ScailInputs(
        driving=driving, replace_mask=replace_mask,
        ref_image=saved[0][0], ref_mask=saved[0][1], prompt_file=prompt_file,
        width=width, height=height, num_frames=num_frames, padded_frames=padded,
        extra_images=[p for p, _ in saved[1:]], extra_masks=[m for _, m in saved[1:]],
    )


def enhance_prompt(inputs: ScailInputs, cfg: dict) -> None:
    """Optional: let the repository's Gemini prompt enhancer rewrite prompt.txt."""
    scfg = cfg["scail"]
    if not os.environ.get("GEMINI_API_KEY"):
        log.warning("scail.prompt_enhancer is on but GEMINI_API_KEY is not set; keeping the prompt as is")
        return
    enhanced = inputs.prompt_file.with_name("prompt_enhanced.txt")
    cmd = [
        scfg["python"], "prompt_enhancer.py",
        "--video", str(inputs.driving.resolve()),
        "--image", str(inputs.ref_image.resolve()),
        "--instruction", scfg["enhancer_instruction"],
        "--examples", "examples/prompt_examples.txt",
        "--num_frames", "8",
        "--output", str(enhanced.resolve()),
    ]
    try:
        subprocess.run(cmd, cwd=scfg["repo_dir"], check=True)
    except subprocess.CalledProcessError as exc:
        log.warning("Prompt enhancer failed (%s); keeping the original prompt", exc)
        return
    text = enhanced.read_text(encoding="utf-8").strip()
    if text:
        inputs.prompt_file.write_text(text, encoding="utf-8")


def _abs(path) -> str:
    # The runner starts inside the SCAIL-2 repository, so every path must be absolute.
    return str(Path(path).resolve())


def run(inputs: ScailInputs, out_video: Path, cfg: dict, on_line=None, on_start=None) -> None:
    """Run SCAIL-2. on_line(text) receives every output line (tqdm updates included);
    on_start(process) gets the Popen object, e.g. to terminate it."""
    scfg = cfg["scail"]
    for key in ("repo_dir", "ckpt_dir", "scail_path"):
        if not Path(scfg[key]).exists():
            raise FileNotFoundError(f"scail.{key} does not exist: {scfg[key]} (set it in your --config file)")
    for lora in scfg.get("loras") or []:
        if not Path(lora["path"]).exists():
            raise FileNotFoundError(f"LoRA not found: {lora['path']} (fix scail.loras or set it to [])")
    if scfg.get("prompt_enhancer"):
        enhance_prompt(inputs, cfg)
    cmd = [
        scfg["python"], str(RUNNER),
        "--repo", _abs(scfg["repo_dir"]),
        "--ckpt_dir", _abs(scfg["ckpt_dir"]),
        "--scail_path", _abs(scfg["scail_path"]),
        "--model", scfg["model"],
        "--image", _abs(inputs.ref_image),
        "--mask_image", _abs(inputs.ref_mask),
        "--pose", _abs(inputs.driving),
        "--mask_video", _abs(inputs.replace_mask),
        "--prompt_file", _abs(inputs.prompt_file),
        "--out", _abs(out_video),
        "--target_w", str(inputs.width),
        "--target_h", str(inputs.height),
        "--steps", str(scfg["steps"]),
        "--shift", str(scfg["shift"]),
        "--guide_scale", str(scfg["guide_scale"]),
        "--solver", scfg["solver"],
        "--seed", str(scfg["seed"]),
        "--segment_len", str(scfg["segment_len"]),
        "--segment_overlap", str(scfg["segment_overlap"]),
        "--offload_model", "true" if scfg["offload_model"] else "false",
        "--weights", scfg.get("weights", "fp8"),
        "--attention", scfg.get("attention", "auto"),
        "--text_cache", _abs(Path(__file__).resolve().parent.parent / ".cache" / "t5"),
    ]
    if scfg["t5_cpu"]:
        cmd.append("--t5_cpu")
    for lora in scfg.get("loras") or []:
        cmd += ["--lora", _abs(lora["path"]), str(lora.get("alpha", 1.0))]
    if inputs.extra_images:
        cmd += ["--additional_image", *map(_abs, inputs.extra_images)]
        cmd += ["--additional_mask", *map(_abs, inputs.extra_masks)]
    log.info("Running SCAIL-2: %s", " ".join(cmd))
    if on_line is None and on_start is None:
        subprocess.run(cmd, cwd=scfg["repo_dir"], check=True)
        return
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    # text mode turns tqdm's carriage returns into separate lines
    proc = subprocess.Popen(cmd, cwd=scfg["repo_dir"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    if on_start:
        on_start(proc)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-40:]
        if on_line:
            on_line(line)
    if proc.wait() != 0:
        raise RuntimeError(f"SCAIL-2 failed (exit {proc.returncode}):\n" + "\n".join(tail[-15:]))


def load_output(path: Path, num_frames: int, size: tuple[int, int], padded_frames: int | None = None) -> np.ndarray:
    """SCAIL-2 frames trimmed to the clip length and resized to `size` if needed."""
    frames = read_frames(path)
    if padded_frames is not None and len(frames) != padded_frames:
        log.warning("SCAIL-2 returned %d frames for a %d-frame driving video; check the alignment "
                    "in 05_composite/side_by_side.mp4", len(frames), padded_frames)
    if len(frames) < num_frames:
        log.warning("SCAIL-2 returned %d frames for %d; repeating the last frame", len(frames), num_frames)
        frames = np.concatenate([frames, np.repeat(frames[-1:], num_frames - len(frames), axis=0)])
    return resize_frames(frames[:num_frames], size)
