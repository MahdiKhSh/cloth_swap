"""HTTP API of the studio and the built web UI."""
from __future__ import annotations

import shutil
import tempfile
import math
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import PROJECT_ROOT
from .jobs import GpuBusy
from .service import PRESETS, StudioService

UI_DIST = PROJECT_ROOT / "studio-ui" / "dist"


class Prompt(BaseModel):
    frame: int
    points: list[list[float]] = []
    labels: list[int] = []
    box: list[float] | None = None


class SegmentRequest(Prompt):
    candidate: str | None = None
    isolated: bool = False
    text: str = ""


class PromptsRequest(BaseModel):
    prompts: list[Prompt]


class DetectRequest(BaseModel):
    frame: int
    text: str | None = None   # English noun phrase, e.g. "man in a brown shirt"; default: every person


class MaskLayer(BaseModel):
    name: str
    prompt: str = ""
    operation: str
    selection: dict | None = None  # selected reference candidate + point/box hints


class MaskLayersRequest(BaseModel):
    layers: list[MaskLayer]
    base_actor: bool = True


class KeyframeRequest(BaseModel):
    frame: int
    count: int = 4
    extra: str | None = None  # added to the built-in instruction, never replaces it
    seed: int | None = None


class ChooseRequest(BaseModel):
    candidate: str


def create_app(service: StudioService) -> FastAPI:
    app = FastAPI(title="Cloth Swap Studio")
    store, jobs = service.store, service.jobs

    def project(pid: str):
        try:
            return store.get(pid)
        except KeyError:
            raise HTTPException(404, "Project not found")

    def editable(pid: str):
        p = project(pid)
        if jobs.active_for_project(p.id):
            raise HTTPException(409, "Wait for or cancel the running step before editing this project")
        return p

    def file(path: Path, media_type: str | None = None, cache: bool = True):
        if not path.exists():
            raise HTTPException(404, "File not found")
        headers = {"Cache-Control": "max-age=86400"} if cache else {"Cache-Control": "no-store"}
        return FileResponse(path, media_type=media_type, headers=headers)

    def save_upload(upload: UploadFile) -> Path:
        suffix = Path(upload.filename or "").suffix or ".bin"
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=store.root)
        with fd:
            shutil.copyfileobj(upload.file, fd)
        return Path(fd.name)

    @app.exception_handler(GpuBusy)
    async def gpu_busy(_: Request, __: GpuBusy):
        return JSONResponse({"detail": "gpu_busy"}, status_code=409)

    @app.exception_handler(ValueError)
    async def bad_value(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ---------------------------------------------------------------- system
    @app.get("/api/system")
    def system():
        import torch

        gpu = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {"name": props.name, "memory_gb": round(props.total_memory / 2**30, 1)}
        busy = jobs.busy()
        ready = True
        if service.base_cfg.get("generation", {}).get("backend") == "wan_animate":
            from ..animate_stage import resolve_models
            try:
                resolve_models(service.base_cfg)
            except (OSError, ImportError):
                ready = False
        return {"model_ready": ready, "gpu": gpu, "busy": busy.to_dict() if busy else None, "presets": {k: {**v, "steps": service.base_cfg["animate"]["steps"]} for k,v in PRESETS.items()} if service.base_cfg.get("generation", {}).get("backend") == "wan_animate" else PRESETS,
                "backend": service.base_cfg.get("generation", {}).get("backend", "scail2"),
                "relight_available": service.base_cfg.get("generation", {}).get("backend") != "wan_animate" and bool((service.base_cfg.get("studio") or {}).get("relight_lora"))}

    # ---------------------------------------------------------- garment bibles
    @app.get("/api/garment-bibles")
    def list_bibles():
        return service.bibles.list()

    @app.post("/api/garment-bibles")
    def create_bible(body: dict):
        from_pid = body.pop("from_project_id", None)
        source = project(from_pid) if from_pid else None
        try:
            return service.bibles.create(body, source)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/garment-bibles/{bid}")
    def get_bible(bid: str):
        try:
            return service.bibles.get(bid)
        except KeyError:
            raise HTTPException(404, "Garment bible not found")

    @app.patch("/api/garment-bibles/{bid}")
    def edit_bible(bid: str, body: dict):
        expected_revision = body.pop("_expected_revision", None)
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
            raise ValueError("Invalid garment bible revision")
        try:
            return service.bibles.update(bid, body, expected_revision)
        except KeyError:
            raise HTTPException(404, "Garment bible not found")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/garment-bibles/{bid}/images")
    def bible_image_upload(bid: str, file: UploadFile = File(...), view: str = Form(...),
                           expected_revision: int | None = Form(None)):
        tmp = save_upload(file)
        try:
            return service.bibles.add_image(bid, tmp, view, expected_revision)
        except KeyError:
            raise HTTPException(404, "Garment bible not found")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        finally:
            tmp.unlink(missing_ok=True)

    @app.get("/api/garment-bibles/{bid}/images/{name}")
    def bible_image(bid: str, name: str):
        try:
            bible = service.bibles.get(bid)
        except KeyError:
            raise HTTPException(404, "Garment bible not found")
        if name not in {item["file"] for item in bible["images"]}:
            raise HTTPException(404, "Image not found")
        return file(service.bibles._path(bid) / name, "image/png", cache=False)

    @app.post("/api/projects/{pid}/garment-bible/{bid}")
    def apply_bible(pid: str, bid: str):
        p = editable(pid)
        try:
            return service.bibles.apply(bid, p)
        except KeyError:
            raise HTTPException(404, "Garment bible not found")

    @app.post("/api/garment-bibles/{bid}/sync")
    def sync_bible(bid: str):
        try:
            bible = service.bibles.get(bid)
        except KeyError:
            raise HTTPException(404, "Garment bible not found")
        synced, skipped = [], []
        for entry in store.list():
            if entry["garment"].get("bible_id") != bid:
                continue
            pid = entry["id"]
            if jobs.active_for_project(pid):
                skipped.append(pid)
                continue
            service.bibles.apply(bid, project(pid))
            synced.append(pid)
        return {"revision": bible["revision"], "synced": synced, "skipped_running": skipped}

    # ---------------------------------------------------------------- projects
    @app.get("/api/projects")
    def list_projects():
        return store.list()

    @app.post("/api/projects")
    def create_project(file: UploadFile = File(...), name: str = Form("")):
        if Path(file.filename or "").suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mxf"}:
            raise HTTPException(400, "Unsupported video format")
        tmp = save_upload(file)
        try:
            p, job = service.create_project((name or Path(file.filename or "video").stem)[:160], tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return {"project": p.to_dict(), "job": job.to_dict()}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        p = project(pid)
        return {**p.to_dict(), "jobs": [j.to_dict() for j in jobs.for_project(pid)]}

    @app.patch("/api/projects/{pid}")
    def patch_project(pid: str, body: dict):
        p = editable(pid)
        allowed = {"name", "trim", "prompt", "garment", "generate", "composite"}
        if set(body) - allowed:
            raise HTTPException(400, f"Cannot change {sorted(set(body) - allowed)[0]}")

        # Validate the entire request before changing anything. Prior versions
        # could partially save a multi-section patch before a later field failed.
        changes = {}
        for key, value in body.items():
            if key == "name":
                if not isinstance(value, str) or len(value) > 160:
                    raise ValueError("Project name must be text of at most 160 characters")
                changes[key] = value.strip()
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            if key == "trim":
                n = (p.state.get("video") or {}).get("num_frames")
                if not n:
                    raise HTTPException(409, "Video import has not completed")
                start, end = value.get("start"), value.get("end")
                if any(type(x) is not int for x in (start, end)) or not (0 <= start < end <= n):
                    raise ValueError("Invalid trim frame range")
                if end - start < 5:
                    raise ValueError("The clip must be at least 5 frames long")
                changes[key] = {"start": start, "end": end}
                continue
            permitted = {
                "garment": {"description"},
                "prompt": {"subject", "scene", "full"},
                "generate": {"preset", "steps", "seed", "relight", "upscale", "output"},
                "composite": {"mode", "feather", "dilate"},
            }[key]
            if set(value) - permitted:
                raise ValueError(f"Unsupported {key} setting: {sorted(set(value) - permitted)[0]}")
            if key == "garment":
                if p.state["garment"].get("bible_id"):
                    raise HTTPException(409, "Garment linked to a bible: edit the bible and sync instead")
                if any(not isinstance(v, str) or len(v) > 4000 for v in value.values()):
                    raise ValueError("Garment description must be text of at most 4000 characters")
            elif key == "prompt":
                if any(not isinstance(v, str) or len(v) > 4000 for v in value.values()):
                    raise ValueError("Prompt fields must be text of at most 4000 characters")
            elif key == "generate":
                if "preset" in value and value["preset"] not in PRESETS:
                    raise ValueError("Unknown quality preset")
                if "steps" in value and value["steps"] is not None and (type(value["steps"]) is not int or not 1 <= value["steps"] <= 80):
                    raise ValueError("Steps must be between 1 and 80")
                if "seed" in value and (type(value["seed"]) is not int or not 0 <= value["seed"] <= 4294967295):
                    raise ValueError("Seed must be an unsigned 32-bit integer")
                if "relight" in value and type(value["relight"]) is not bool:
                    raise ValueError("Relight must be true or false")
                if "output" in value and value["output"] != "original":
                    raise ValueError("Only original-size output is supported without upscaling")
                if "upscale" in value and value["upscale"] != "resize":
                    raise ValueError("AI upscaling is disabled in Studio")
            elif key == "composite":
                if "mode" in value and value["mode"] not in ("actor", "garment", "full"):
                    raise ValueError("Unknown composite mode")
                for setting, maximum in (("feather", .2), ("dilate", .3)):
                    if setting in value and (type(value[setting]) not in (float, int)
                                             or not math.isfinite(value[setting]) or not 0 <= value[setting] <= maximum):
                        raise ValueError(f"{setting} is out of range")
            changes[key] = value

        with p.lock:
            invalid_keyframe = False
            changed = False
            render_changed = False
            composite_changed = False
            for key, value in changes.items():
                if key == "name":
                    changed |= p.state["name"] != value
                    p.state["name"] = value
                    continue
                previous = p.state.get(key) or {}
                merged = {**previous, **value}
                if previous == merged:
                    continue
                changed = True
                p.state[key] = merged
                if key == "composite":
                    composite_changed = True
                else:
                    render_changed = True
                if key == "trim":
                    p.state["actor"].update(tracked=False, parts_ready=False, parts_quality=None, visible=[], trim=None)
                    p.state["mask_layers"].update(ready=False, manual=False, tracked=False,
                                                   object_signature="", formula_signature="")
                    invalid_keyframe = True
                elif key == "garment" or (key == "prompt" and ("subject" in value or "scene" in value)):
                    invalid_keyframe = True
            if invalid_keyframe:
                p.state["keyframe"].update(chosen=None, candidates=[], prompt="")
            if (render_changed or composite_changed) and p.state.get("result"):
                old_reason = p.state["result"].get("stale_reason")
                reason = ("source" if render_changed or (p.state["result"].get("stale")
                          and old_reason != "composite") else "composite")
                p.state["result"].update(stale=True, stale_reason=reason)
            if changed:
                p.save()
        return p.to_dict()

    @app.get("/api/projects/{pid}/sam3-settings")
    def sam3_get(pid: str):
        return service.get_sam3_settings(project(pid))

    @app.patch("/api/projects/{pid}/sam3-settings")
    def sam3_patch(pid: str, body: dict):
        return service.set_sam3_settings(editable(pid), body).to_dict()

    @app.post("/api/projects/{pid}/sam3-settings/reset")
    def sam3_reset(pid: str):
        p = editable(pid)
        from .sam3_controls import defaults, effective
        return service.set_sam3_settings(p, {k: v for k, v in
            effective(service.base_cfg.get("sam3", {})).items() if k in defaults()}).to_dict()

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        editable(pid)
        store.delete(pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/frames/{t}.jpg")
    def frame(pid: str, t: int):
        return file(project(pid).dir / "frames" / f"{t + 1:06d}.jpg", "image/jpeg")

    @app.get("/api/projects/{pid}/thumbs/{t}.jpg")
    def thumb(pid: str, t: int):
        return file(project(pid).dir / "thumbs" / f"{t + 1:06d}.jpg", "image/jpeg")

    @app.get("/api/projects/{pid}/source")
    def source(pid: str):
        return file(project(pid).source)

    @app.get("/api/projects/{pid}/result")
    def result(pid: str):
        return file(project(pid).dir / "result.mp4", "video/mp4", cache=False)

    @app.get("/api/projects/{pid}/result/download")
    def download(pid: str):
        p = project(pid)
        path = p.dir / "result.mp4"
        if not path.exists():
            raise HTTPException(404, "No result yet")
        return FileResponse(path, media_type="video/mp4", filename=f"{p.state['name'] or 'result'}_clothswap.mp4")

    # ---------------------------------------------------------------- actor
    @app.post("/api/projects/{pid}/detect")
    def detect(pid: str, body: DetectRequest):
        return {"people": service.detect_people(project(pid), body.frame, body.text)}

    @app.post("/api/projects/{pid}/segment")
    def segment(pid: str, body: SegmentRequest):
        return service.segment_preview(project(pid), body.frame, body.points, body.labels, body.box,
                                       body.candidate, body.isolated, body.text)

    @app.put("/api/projects/{pid}/actor/prompts")
    def set_prompts(pid: str, body: PromptsRequest):
        p = editable(pid)
        service.set_prompts(p, [pr.model_dump() for pr in body.prompts])
        return p.to_dict()

    @app.post("/api/projects/{pid}/actor/track")
    def track(pid: str):
        return service.track(project(pid)).to_dict()

    @app.post("/api/projects/{pid}/actor/person-parts")
    def separate_parts(pid: str):
        return service.separate_person_parts(project(pid)).to_dict()

    @app.get("/api/projects/{pid}/actor/isolated/{t}.jpg")
    def isolated_person_frame(pid: str, t: int):
        import cv2
        image = service.isolated_frame(project(pid), t)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(500, "Could not encode isolated frame")
        return Response(content=encoded.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/projects/{pid}/actor/mask/{t}")
    def actor_mask(pid: str, t: int, part: str = "actor"):
        if part not in {"actor", "edit", "protect", "removed", "face", "hair"} and not part.startswith("layer:"):
            raise HTTPException(400, "Unknown mask part")
        return service.actor_outline(project(pid), t, part)

    @app.put("/api/projects/{pid}/mask-layers")
    def mask_layers(pid: str, body: MaskLayersRequest):
        return service.set_mask_layers(editable(pid), [v.model_dump() for v in body.layers], body.base_actor).to_dict()

    @app.post("/api/projects/{pid}/mask-layers/build")
    def build_mask_layers(pid: str):
        return service.build_mask_layers(project(pid)).to_dict()

    @app.post("/api/projects/{pid}/mask-layers/compose")
    def compose_mask_layers(pid: str):
        return service.compose_mask_layers(project(pid)).to_dict()

    # ---------------------------------------------------------------- garment
    @app.post("/api/projects/{pid}/garments")
    def add_garment(pid: str, file: UploadFile = File(...)):
        p = editable(pid)
        if p.state["garment"].get("bible_id"):
            raise HTTPException(409, "Upload the new view to the linked garment bible")
        tmp = save_upload(file)
        try:
            service.add_garment(p, tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return p.to_dict()

    @app.delete("/api/projects/{pid}/garments/{name}")
    def remove_garment(pid: str, name: str):
        p = editable(pid)
        if p.state["garment"].get("bible_id"):
            raise HTTPException(409, "Edit the linked garment bible instead")
        service.remove_garment(p, name)
        return p.to_dict()

    @app.get("/api/projects/{pid}/garments/{name}")
    def garment_image(pid: str, name: str):
        p = project(pid)
        if name not in p.state["garment"].get("images", []) or Path(name).name != name:
            raise HTTPException(404, "Image not found")
        return file(p.dir / "garments" / name)

    # ---------------------------------------------------------------- keyframe
    @app.get("/api/projects/{pid}/keyframe/instruction")
    def keyframe_instruction(pid: str, extra: str = ""):
        try:
            return {"text": service.edit_instruction(project(pid), extra)}
        except ValueError:
            return {"text": ""}

    @app.post("/api/projects/{pid}/keyframe/generate")
    def keyframe_generate(pid: str, body: KeyframeRequest):
        return service.generate_keyframes(project(pid), body.frame, body.count, body.extra, body.seed).to_dict()

    @app.post("/api/projects/{pid}/keyframe/upload")
    def keyframe_upload(pid: str, frame: int = Form(...), file: UploadFile = File(...)):
        tmp = save_upload(file)
        try:
            return service.upload_keyframe(project(pid), frame, tmp).to_dict()
        finally:
            tmp.unlink(missing_ok=True)

    @app.post("/api/projects/{pid}/keyframe/choose")
    def keyframe_choose(pid: str, body: ChooseRequest):
        p = editable(pid)
        try:
            service.choose_keyframe(p, body.candidate)
        except KeyError:
            raise HTTPException(404, "Candidate not found")
        return p.to_dict()

    @app.delete("/api/projects/{pid}/keyframe/candidates/{cid}")
    def keyframe_delete(pid: str, cid: str):
        p = editable(pid)
        service.delete_candidate(p, cid)
        return p.to_dict()

    @app.get("/api/projects/{pid}/keyframe/candidates/{cid}.jpg")
    def keyframe_image(pid: str, cid: str):
        return file(project(pid).dir / "02_keyframe" / "candidates" / f"{Path(cid).name}.jpg", "image/jpeg")

    # ---------------------------------------------------------------- video
    @app.post("/api/projects/{pid}/generate")
    def generate(pid: str):
        return service.generate(project(pid)).to_dict()

    @app.post("/api/projects/{pid}/composite")
    def composite(pid: str):
        return service.recomposite(project(pid)).to_dict()

    # ---------------------------------------------------------------- jobs
    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs.jobs:
            raise HTTPException(404, "Job not found")
        return jobs.jobs[job_id].to_dict(logs=True)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if job_id not in jobs.jobs:
            raise HTTPException(404, "Job not found")
        return jobs.cancel(job_id).to_dict()

    # ---------------------------------------------------------------- web UI
    if UI_DIST.exists():
        app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            # Never serve paths outside the UI build directory, even when a
            # client sends unusual encoded dot segments or a symlink.
            target = (UI_DIST / path).resolve()
            ui_root = UI_DIST.resolve()
            if path and target.is_file() and target.is_relative_to(ui_root):
                return FileResponse(target)
            return FileResponse(UI_DIST / "index.html", headers={"Cache-Control": "no-store"})

    return app
