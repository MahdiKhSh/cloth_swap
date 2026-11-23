"""Persistent, shared garment specifications for multi-shot visual continuity.

A garment bible is a *locked specification*, not a guarantee of identical generative
output. Project references are snapshotted on apply and invalidated on revision changes.
The store deliberately lives outside project directories so it survives project deletion.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

FIELDS = (
    "name", "description", "type", "color", "fabric", "pattern", "fit",
    "collar", "sleeves", "closures", "pockets", "logo", "seams", "notes",
)
VIEWS = ("front", "back", "side", "detail")
VALID_ID = re.compile(r"^[a-f0-9]{16}$")


def _clean(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    value = value.strip()
    if len(value) > (120 if field == "name" else 1200):
        raise ValueError(f"{field} is too long")
    return value


def continuity_prompt(bible: dict) -> str:
    """One deterministic English specification used in keyframes and all shots."""
    parts = [f"outfit ID/name: {bible['name']}", bible.get("description", "")]
    labels = {"type": "garment type", "color": "exact color", "fabric": "material",
              "pattern": "print/pattern", "fit": "fit", "collar": "neckline/collar",
              "sleeves": "sleeve design", "closures": "buttons/zippers",
              "pockets": "pocket placement", "logo": "logo/branding",
              "seams": "stitching", "notes": "additional fixed features"}
    parts += [f"{label}: {bible[k]}" for k, label in labels.items() if bible.get(k)]
    return ("CONTINUITY-LOCKED GARMENT, identical across all shots: " + "; ".join(p for p in parts if p) +
            ". Preserve exact garment design, color, placement, texture and construction across camera angles "
            "and lighting. Never invent or remove details.")


class GarmentBibleStore:
    def __init__(self, root: Path):
        self.root = Path(root) / "_garment_bibles"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _path(self, bible_id: str) -> Path:
        if not VALID_ID.fullmatch(bible_id):
            raise KeyError(bible_id)
        return self.root / bible_id

    def get(self, bible_id: str) -> dict:
        with self.lock:
            path = self._path(bible_id) / "bible.json"
            if not path.is_file():
                raise KeyError(bible_id)
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        with self.lock:
            all_bibles = []
            for path in self.root.iterdir():
                if path.is_dir() and VALID_ID.fullmatch(path.name) and (path / "bible.json").exists():
                    all_bibles.append(self.get(path.name))
            return sorted(all_bibles, key=lambda x: x["updated"], reverse=True)

    def _write(self, bible: dict) -> dict:
        folder = self._path(bible["id"])
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "bible.json"
        tmp = folder / "bible.json.tmp"
        tmp.write_text(json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        return bible

    def create(self, values: dict, from_project=None) -> dict:
        with self.lock:
            now = time.time()
            bible = {f: _clean(f, values.get(f, "")) for f in FIELDS}
            if not bible["name"]:
                raise ValueError("A garment bible needs a name")
            bible.update(id=uuid.uuid4().hex[:16], revision=1, created=now, updated=now, images=[])
            if from_project is not None:
                old = from_project.state["garment"]
                if not bible["description"]:
                    bible["description"] = old.get("description", "")
                for i, img in enumerate(old.get("images", [])[:len(VIEWS)]):
                    src = from_project.dir / "garments" / Path(img).name
                    if src.is_file():
                        filename = f"{VIEWS[i]}.png"
                        # Newly created folder does not exist yet.
                        target = self._path(bible["id"]) / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, target)
                        bible["images"].append({"view": VIEWS[i], "file": filename})
            return self._write(bible)

    def update(self, bible_id: str, changes: dict, expected_revision: int | None = None) -> dict:
        with self.lock:
            bible = self.get(bible_id)
            if expected_revision is not None and bible["revision"] != expected_revision:
                raise RuntimeError("Garment bible changed in another window; reload before saving")
            if not changes or set(changes) - set(FIELDS):
                raise ValueError("Only garment bible descriptive fields can be edited")
            parsed = {f: _clean(f, v) for f, v in changes.items()}
            if parsed.get("name", bible["name"]) == "":
                raise ValueError("A garment bible needs a name")
            if all(bible[k] == v for k, v in parsed.items()):
                return bible
            bible.update(parsed)
            bible["revision"] += 1
            bible["updated"] = time.time()
            return self._write(bible)

    def add_image(self, bible_id: str, upload: Path, view: str,
                  expected_revision: int | None = None) -> dict:
        if view not in VIEWS:
            raise ValueError("Image view must be front, back, side or detail")
        with self.lock:
            bible = self.get(bible_id)
            if expected_revision is not None and bible["revision"] != expected_revision:
                raise RuntimeError("Garment bible changed in another window; reload before uploading")
            dest = self._path(bible_id) / f"{view}.png"
            staged = self._path(bible_id) / f".{view}.{uuid.uuid4().hex}.tmp"
            backup = self._path(bible_id) / f".{view}.{uuid.uuid4().hex}.bak"
            try:
                with Image.open(upload) as im:
                    im.verify()
                with Image.open(upload) as im:
                    if im.width * im.height > 80_000_000:
                        raise ValueError("Reference image is too large")
                    image = ImageOps.exif_transpose(im).convert("RGB")
                    image.thumbnail((3072, 3072), Image.Resampling.LANCZOS)
                    image.save(staged, format="PNG")
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
                raise ValueError("Invalid garment reference image") from exc
            try:
                if dest.exists():
                    shutil.copyfile(dest, backup)
                os.replace(staged, dest)
                bible["images"] = [i for i in bible["images"] if i["view"] != view] + [{"view": view, "file": f"{view}.png"}]
                bible["images"].sort(key=lambda i: VIEWS.index(i["view"]))
                bible["revision"] += 1
                bible["updated"] = time.time()
                return self._write(bible)
            except Exception:
                if backup.exists():
                    os.replace(backup, dest)
                else:
                    dest.unlink(missing_ok=True)
                raise
            finally:
                staged.unlink(missing_ok=True)
                backup.unlink(missing_ok=True)

    def apply(self, bible_id: str, project) -> dict:
        with self.lock:
            bible = self.get(bible_id)
            # Validate all references and stage all image copies before touching
            # project state. An interrupted/missing copy must not mark a revision
            # as successfully applied.
            sources = []
            for image in bible["images"]:
                view = image.get("view")
                if view not in VIEWS or image.get("file") != f"{view}.png":
                    raise ValueError("Invalid garment bible image metadata")
                src = self._path(bible_id) / image["file"]
                if not src.is_file():
                    raise ValueError(f"Garment bible image is missing: {view}")
                sources.append((src, f"bible_{bible_id}_{view}.png"))
            folder = project.path("garments", "placeholder").parent
            staged = []
            try:
                for src, filename in sources:
                    tmp = folder / f".{filename}.{uuid.uuid4().hex}.tmp"
                    staged.append((tmp, folder / filename))
                    shutil.copyfile(src, tmp)
                with project.lock:
                    old = project.state["garment"]
                    old_images = list(old.get("images") or [])
                    filenames = [filename for _, filename in sources]
                    changed = (old.get("bible_id") != bible_id or
                               old.get("bible_revision") != bible["revision"] or
                               old_images != filenames)
                    for tmp, target in staged:
                        os.replace(tmp, target)
                    project.state["garment"].update(images=filenames, description=continuity_prompt(bible),
                                                    bible_id=bible_id, bible_revision=bible["revision"])
                    if changed:
                        project.state["keyframe"].update(chosen=None, candidates=[], prompt="")
                        if project.state.get("result"):
                            project.state["result"].update(stale=True, stale_reason="source")
                    project.save()
                    # Remove obsolete *generated bible snapshots* only. Never
                    # delete user's independently uploaded images.
                    for name in set(old_images) - set(filenames):
                        if re.fullmatch(r"bible_[a-f0-9]{16}_(front|back|side|detail)\.png", name):
                            (folder / name).unlink(missing_ok=True)
                    return project.to_dict()
            finally:
                for tmp, _ in staged:
                    tmp.unlink(missing_ok=True)
