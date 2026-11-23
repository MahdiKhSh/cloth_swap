"""Wan Animate replacement adapter. Existing VAE and UMT5 weights are reused."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from . import masks as mk
from .scail2_stage import build_prompt
from .video_io import resize_frames, write_video


def resolve_models(cfg):
    from huggingface_hub import hf_hub_download
    a = cfg['animate']
    repo = Path(hf_hub_download(a['model_id'], 'model_index.json', local_files_only=True)).parent
    transformer_dir = Path(a.get('transformer_dir') or repo / 'transformer')
    index = json.loads((transformer_dir / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    needed = [transformer_dir / f for f in set(index['weight_map'].values())]
    needed += [repo / 'image_encoder/model.safetensors', Path(a['vae']),
               Path(a['repo_dir']) / 'wan/modules/animate/preprocess/pose2d.py']
    pose = Path(hf_hub_download(a['pose_model_id'],
        'process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx', local_files_only=True)).parent
    needed += [pose / 'end2end.onnx']
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(f'Wan Animate model is not ready: {path}')
    import onnx
    graph = onnx.load_model(str(pose / 'end2end.onnx'), load_external_data=False)
    for tensor in graph.graph.initializer:
        for item in tensor.external_data:
            if item.key == 'location' and not (pose / item.value).is_file():
                raise FileNotFoundError(f'Wan Animate pose weight is not ready: {item.value}')
    return repo, pose


def run_process(cmd, cwd, on_line=None, on_start=None):
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', env=env)
    tail = []
    try:
        if on_start:
            on_start(proc)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-25:]
                if on_line:
                    on_line(line)
        if proc.wait():
            raise RuntimeError('Wan Animate failed:\n' + '\n'.join(tail))
    finally:
        if proc.poll() is None:
            import psutil
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
        proc.stdout.close()


def run(folder, frames, actor, keyframe, size, fps, cfg, on_line=None, on_start=None):
    repo, pose = resolve_models(cfg)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    resized = resize_frames(frames, size)
    write_video(folder / 'driving.mp4', resized, fps, crf=10)
    mk.save_masks(folder / 'actor.npz', actor)
    if cfg.get('mask_layers_enabled'):
        root_masks = folder.parent / '01_masks'
        for name in ('edit', 'protect'):
            src = root_masks / f'{name}.npz'
            if not src.exists():
                raise ValueError(f'Missing layered mask: {src}')
            import shutil
            shutil.copyfile(src, folder / f'{name}.npz')
    else:
        for name in ('edit', 'protect'):
            (folder / f'{name}.npz').unlink(missing_ok=True)
    Image.fromarray(keyframe).resize(size, Image.Resampling.LANCZOS).save(folder / 'reference.png')
    settings = dict(cfg['animate'], model_dir=str(repo), pose_dir=str(pose),
                    scail=cfg['scail'], width=size[0], height=size[1], frames=len(frames), fps=fps,
                    steps=cfg['scail']['steps'], seed=cfg['scail']['seed'], prompt=build_prompt(cfg))
    manifest = folder / 'animate.json'
    manifest.write_text(json.dumps(settings, ensure_ascii=False), encoding='utf-8')
    root = Path(__file__).resolve().parent.parent
    for phase in ('prepare', 'text', 'generate'):
        python = cfg['scail']['python'] if phase == 'text' else sys.executable
        run_process([python, '-m', 'cloth_swap.animate_runner', str(manifest), phase], root, on_line, on_start)
    (folder / 'inputs.json').write_text(json.dumps({'padded_frames': len(frames), 'backend': 'wan_animate'}))
