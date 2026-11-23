"""Projects on disk: one folder per project, its state in project.json.

The folder doubles as the pipeline work directory (01_masks, 02_keyframe, ... meta.json), so
every studio step reuses the batch pipeline's stages and file layout.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import imageio_ffmpeg

from ..video_io import probe

DEFAULT_STATE = {
    "name": "",
    "created": 0.0,
    "updated": 0.0,
    "video": None,          # {"file", "fps", "width", "height", "num_frames"}
    "trim": None,           # {"start", "end"} absolute frame indices, end exclusive
    "actor": {"prompts": [], "tracked": False, "parts_ready": False, "parts_quality": None, "visible": [], "trim": None},
    "sam3_settings": {},             # per-project SAM3 overrides; never mutates global defaults
    "mask_layers": {"rules": [], "ready": False, "manual": False, "tracked": False, "base_actor": True,
                    "object_signature": "", "formula_signature": ""},
    "garment": {"images": [], "description": "", "bible_id": None, "bible_revision": None},
    "prompt": {"subject": "", "scene": ""},
    "keyframe": {"frame": None, "suggestions": [], "candidates": [], "chosen": None, "prompt": ""},
    "generate": {"preset": "preview", "steps": None, "seed": 42, "relight": False, "upscale": "resize",
                 "output": "original"},
    "composite": {"mode": "actor", "feather": 0.012, "dilate": 0.02},
    "result": None,         # {"file", "updated"}
}

# Validate IDs at the storage boundary, not only in API URL routing.  In
# particular, a malicious or malformed ID must never escape the data root on
# get/delete, while retaining compatibility with older alphanumeric IDs.
PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def valid_project_id(project_id: str) -> bool:
    return isinstance(project_id, str) and bool(PROJECT_ID_PATTERN.fullmatch(project_id))


class Project:
    def __init__(self, folder: Path):
        self.dir = folder
        self.id = folder.name
        self.lock = threading.RLock()
        with open(folder / "project.json", encoding="utf-8") as f:
            stored = json.load(f)
        self.state = _merge(copy.deepcopy(DEFAULT_STATE), stored)
        # An old auto-generated head mask must never count as a user-approved
        # manual selection after upgrading to the manual-only workflow.
        if not stored.get("mask_layers", {}).get("manual", False):
            self.state["mask_layers"]["ready"] = False
            self.state["mask_layers"]["manual"] = False

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self) -> None:
        with self.lock:
            self.state["updated"] = time.time()
            tmp = self.dir / "project.json.tmp"
            tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.dir / "project.json")

    def update(self, section: str, values: dict) -> None:
        with self.lock:
            self.state[section] = {**(self.state.get(section) or {}), **values}
            self.save()

    @property
    def source(self) -> Path:
        return self.dir / self.state["video"]["file"]

    def to_dict(self) -> dict:
        # Never return a mutable reference to state while a worker is updating
        # it. API responses and list views must be consistent snapshots.
        with self.lock:
            return {"id": self.id, **copy.deepcopy(self.state)}


def _merge(base: dict, stored: dict) -> dict:
    for key, value in stored.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Project] = {}
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        projects = []
        for folder in self.root.iterdir():
            if folder.is_dir() and valid_project_id(folder.name) and (folder / "project.json").is_file():
                projects.append(self.get(folder.name).to_dict())
        return sorted(projects, key=lambda p: p["updated"], reverse=True)

    def get(self, project_id: str) -> Project:
        if not valid_project_id(project_id):
            raise KeyError(project_id)
        with self._lock:
            if project_id not in self._cache:
                folder = self.root / project_id
                if not (folder / "project.json").exists():
                    raise KeyError(project_id)
                self._cache[project_id] = Project(folder)
            return self._cache[project_id]

    def create(self, name: str, video_file: Path) -> Project:
        project_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        folder = self.root / project_id
        folder.mkdir(parents=True)
        suffix = Path(video_file).suffix.lower() or ".mp4"
        shutil.move(str(video_file), folder / f"source{suffix}")
        state = copy.deepcopy(DEFAULT_STATE)
        state.update(name=name, created=time.time(), updated=time.time())
        state["video"] = {"file": f"source{suffix}"}
        (folder / "project.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.get(project_id)

    def delete(self, project_id: str) -> None:
        if not valid_project_id(project_id):
            raise KeyError(project_id)
        # Ensure the entry exists instead of silently claiming deletion.
        self.get(project_id)
        with self._lock:
            self._cache.pop(project_id, None)
        shutil.rmtree(self.root / project_id)


def detect_content_box(project: Project, width: int) -> list[int]:
    """Picture area without letterbox bars, measured on the thumbnails and scaled to full size."""
    import numpy as np
    from PIL import Image

    from ..geometry import content_box

    thumbs = sorted((project.dir / "thumbs").glob("*.jpg"))
    sample = np.stack([np.asarray(Image.open(t).convert("RGB")) for t in thumbs[:: max(1, len(thumbs) // 12)]])
    scale = width / sample.shape[2]
    height = round(sample.shape[1] * scale)
    x0, y0, x1, y1 = content_box(sample)
    # one thumbnail pixel of slack so the picture edge is never cut
    return [max(0, round((x0 - 1) * scale)), max(0, round((y0 - 1) * scale)),
            min(width, round((x1 + 1) * scale)), min(height, round((y1 + 1) * scale))]


def import_video(project: Project) -> None:
    """Probe the source and extract full-size frames and small timeline thumbnails."""
    info = probe(project.source)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames_dir = project.path("frames", "x").parent
    thumbs_dir = project.path("thumbs", "x").parent
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(project.source), "-vsync", "0",
                    "-q:v", "2", str(frames_dir / "%06d.jpg")], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(project.source), "-vsync", "0",
                    "-vf", "scale=192:-2", "-q:v", "5", str(thumbs_dir / "%06d.jpg")], check=True)
    count = len(list(frames_dir.glob("*.jpg")))
    with project.lock:
        project.state["video"].update(fps=info.fps, width=info.width, height=info.height, num_frames=count,
                                      content_box=detect_content_box(project, info.width))
        project.state["trim"] = {"start": 0, "end": count}
        project.save()
