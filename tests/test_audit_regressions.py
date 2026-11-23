"""Post-audit regression checks for concurrency, disk safety, stale renders and 4K RAM paths.

No ML checkpoints, CUDA or internet required; production model quality is *not* tested.
"""
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cloth_swap import masks as mk
from cloth_swap.config import load_config
from cloth_swap.pipeline import Pipeline
from cloth_swap.studio.api import create_app
from cloth_swap.studio.jobs import JobManager
from cloth_swap.studio.service import StudioService
from cloth_swap.studio.store import ProjectStore
from cloth_swap.video_io import iter_frames, read_frames, write_video


def new_project(service, tmp_path, name="shot"):
    video = tmp_path / (name + ".mp4")
    video.write_bytes(b"fake test video")
    project = service.store.create(name, video)
    project.update("video", {"num_frames": 12, "fps": 24, "width": 64, "height": 48})
    project.update("trim", {"start": 0, "end": 12})
    return project


def test_store_rejects_traversal_even_if_outside_contains_valid_project(tmp_path):
    store = ProjectStore(tmp_path / "data")
    outside = tmp_path / "victim"
    outside.mkdir()
    (outside / "project.json").write_text("{}")
    for bad in ["../victim", "..", "a/b", "../../", "\\..\\", "", ".", "x/..", "ab" * 100]:
        with pytest.raises(KeyError):
            store.get(bad)
        with pytest.raises(KeyError):
            store.delete(bad)
    assert (outside / "project.json").exists()


def test_project_responses_are_deep_copies(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    state = p.to_dict()
    state["garment"]["images"].append("fake.png")
    state["video"]["fps"] = 999
    assert p.state["garment"]["images"] == []
    assert p.state["video"]["fps"] == 24


def test_jobs_of_different_kinds_on_same_project_cannot_overlap():
    manager = JobManager()
    begun = threading.Event()
    release = threading.Event()

    def slow(job):
        begun.set()
        assert release.wait(4)
        return "ok"

    a = manager.submit("clip-a", "track", "track", slow, gpu=False)
    assert begun.wait(2)
    with pytest.raises(ValueError, match="Another step"):
        manager.submit("clip-a", "generate", "generate", lambda _: "wrong", gpu=True)
    b = manager.submit("clip-b", "track", "track", lambda _: "ok", gpu=False)
    release.set()
    for _ in range(200):
        if a.status == "done" and b.status == "done":
            break
        time.sleep(.01)
    assert a.status == "done" and b.status == "done"


def test_job_queued_cancellation_and_late_cancellation():
    manager = JobManager()
    release = threading.Event()
    started = threading.Event()

    def first(job):
        started.set()
        release.wait(4)

    first_job = manager.submit("first", "generate", "generate", first)
    assert started.wait(2)
    queued = manager.submit("second", "generate", "generate", lambda _: "must not run")
    cancelled = manager.cancel(queued.id)
    assert cancelled.status == "cancelled" and cancelled.finished is not None
    release.set()
    manager.queue.join()
    assert first_job.status == "done"
    assert queued.status == "cancelled" and queued.started is None

    def cancellable(job):
        job.cancel_requested = True
        return "completed but cancelled"

    late = manager.submit("third", "track", "track", cancellable, gpu=False)
    for _ in range(200):
        if late.finished is not None:
            break
        time.sleep(.01)
    assert late.status == "cancelled"


def test_atomic_patch_validation_does_not_change_partial_state(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    client = TestClient(create_app(service))
    original = p.to_dict()
    response = client.patch(f"/api/projects/{p.id}", json={"name": "changed", "trim": {"start": 11, "end": 2}})
    assert response.status_code == 400
    assert p.state["name"] == original["name"]
    assert p.state["trim"] == original["trim"]


@pytest.mark.parametrize("payload", [
    {"garment": {"images": ["../bad.png"]}},
    {"garment": {"bible_id": "fake"}},
    {"generate": {"preset": "nonsense"}},
    {"generate": {"steps": -1}},
    {"generate": {"seed": 2**32}},
    {"generate": {"relight": 1}},
    {"generate": {"upscale": "seedvr2"}},
    {"prompt": {"full": ["bad"]}},
    {"composite": {"feather": -1}},
    {"composite": {"mode": "invalid"}},
])
def test_api_refuses_invalid_or_unsafe_settings(tmp_path, payload):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    response = TestClient(create_app(service)).patch(f"/api/projects/{p.id}", json=payload)
    assert response.status_code == 400, response.text


def test_busy_job_blocks_mutations_even_when_step_is_different(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    client = TestClient(create_app(service))
    started, release = threading.Event(), threading.Event()

    def wait(job):
        started.set()
        release.wait(4)

    job = service.jobs.submit(p.id, "track", "track", wait, gpu=False)
    assert started.wait(2)
    assert client.patch(f"/api/projects/{p.id}", json={"name": "oops"}).status_code == 409
    assert client.put(f"/api/projects/{p.id}/actor/prompts", json={"prompts": []}).status_code == 409
    assert client.delete(f"/api/projects/{p.id}").status_code == 409
    release.set()
    for _ in range(200):
        if job.finished:
            break
        time.sleep(.01)
    assert p.state["name"] != "oops"


def test_changing_trim_invalidates_derived_data_and_render(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    p.update("actor", {"tracked": True, "trim": [0, 12]})
    p.update("mask_layers", {"ready": True, "manual": True, "tracked": True})
    p.update("keyframe", {"chosen": "old", "candidates": [{"id": "old"}]})
    p.update("result", {"file": "result.mp4", "updated": 1, "stale": False})
    response = TestClient(create_app(service)).patch(f"/api/projects/{p.id}", json={"trim": {"start": 1, "end": 11}})
    assert response.status_code == 200
    assert p.state["actor"]["tracked"] is False
    assert p.state["mask_layers"]["ready"] is False
    assert p.state["keyframe"]["chosen"] is None
    assert p.state["keyframe"]["candidates"] == []
    assert p.state["result"]["stale_reason"] == "source"


def test_only_compositing_change_allows_recomposite_without_regeneration(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    client = TestClient(create_app(service))
    p.update("result", {"file": "result.mp4", "updated": 1, "stale": False})
    # Previous render must exist as well as the intermediate generated video.
    (p.dir / "04_upscale").mkdir()
    (p.dir / "04_upscale" / "generated_full.mp4").write_bytes(b"fake")
    response = client.patch(f"/api/projects/{p.id}", json={"composite": {"feather": 0.02}})
    assert response.status_code == 200
    assert p.state["result"]["stale_reason"] == "composite"
    # Scheduling the job is enough here; avoid running a real encode.
    service.jobs.submit = lambda *args, **kwargs: "scheduled"
    assert service.recomposite(p) == "scheduled"
    response = client.patch(f"/api/projects/{p.id}", json={"garment": {"description": "another coat"}})
    assert response.status_code == 200
    assert p.state["result"]["stale_reason"] == "source"
    with pytest.raises(ValueError, match="generate the video again"):
        service.recomposite(p)


def test_bible_revision_conflict_does_not_overwrite_newer_changes(tmp_path):
    service = StudioService(tmp_path / "data", None)
    client = TestClient(create_app(service))
    bible = client.post("/api/garment-bibles", json={"name": "Blue coat"}).json()
    route = f"/api/garment-bibles/{bible['id']}"
    first = client.patch(route, json={"_expected_revision": 1, "collar": "stand"})
    assert first.status_code == 200 and first.json()["revision"] == 2
    conflict = client.patch(route, json={"_expected_revision": 1, "collar": "v-neck"})
    assert conflict.status_code == 409
    assert client.get(route).json()["collar"] == "stand"


def test_missing_bible_reference_is_safe_for_existing_project(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    bible = service.bibles.create({"name": "Coat"})
    image = tmp_path / "coat.png"
    Image.new("RGB", (18, 24), "red").save(image)
    service.bibles.add_image(bible["id"], image, "front")
    service.bibles.apply(bible["id"], p)
    p.update("keyframe", {"chosen": "approved"})
    old = p.to_dict()
    (service.bibles._path(bible["id"]) / "front.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        service.bibles.apply(bible["id"], p)
    assert p.to_dict() == old


def test_invalid_uploaded_image_is_clean_user_error(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    client = TestClient(create_app(service))
    response = client.post(f"/api/projects/{p.id}/garments",
                           files={"file": ("corrupt.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert p.state["garment"]["images"] == []


def test_iter_frames_matches_regular_decode_for_trimmed_video(tmp_path):
    frames = np.stack([np.full((24, 32, 3), i * 15, dtype=np.uint8) for i in range(8)])
    path = tmp_path / "test.mp4"
    write_video(path, frames, 8, crf=10)
    eager = read_frames(path, start=2, count=4)
    streamed = np.stack(list(iter_frames(path, start=2, count=4)))
    assert np.array_equal(streamed, eager)


def test_streaming_composite_preserves_untouched_pixels(tmp_path):
    count, height, width = 8, 24, 32
    cfg = load_config()
    cfg["composite"].update(mode="actor", temporal=0, feather=0.0, dilate=0.0)
    cfg["output"].update(preview=False, resolution="original", crf=10)
    video = tmp_path / "original.mp4"
    work = tmp_path / "work"
    result = tmp_path / "out.mp4"
    original = np.full((count, height, width, 3), 40, np.uint8)
    altered = np.full((count, height, width, 3), 180, np.uint8)
    write_video(video, original, 8, crf=10)
    generated = work / "04_upscale" / "generated_full.mp4"
    generated.parent.mkdir(parents=True)
    write_video(generated, altered, 8, crf=10)
    (work / "01_masks").mkdir(parents=True)
    mask = np.zeros((count, height, width), bool)
    mask[:, 7:18, 8:24] = True
    mk.save_masks(work / "01_masks" / "actor.npz", mask)
    work.mkdir(exist_ok=True)
    (work / "meta.json").write_text(json.dumps({"width": width, "height": height, "fps": 8,
                                               "start_frame": 0, "num_frames": count}))
    pipe = Pipeline(cfg, video, result, work)
    pipe.stage_composite()
    output = read_frames(result)
    assert len(output) == count
    assert abs(int(output[0, 0, 0, 0]) - 40) < 10
    assert int(output[0, 12, 16, 0]) > 140


def test_stage_upscale_resize_is_streamed(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["upscale"]["backend"] = "resize"
    work = tmp_path / "work"
    (work / "03_scail").mkdir(parents=True)
    (work / "03_scail" / "inputs.json").write_text(json.dumps({"padded_frames": 5}))
    (work / "meta.json").write_text(json.dumps({"width": 64, "height": 48, "fps": 8,
                                               "num_frames": 5, "work_width": 32, "work_height": 24}))
    from cloth_swap import scail2_stage
    fake = np.full((5, 24, 32, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(scail2_stage, "load_output", lambda *a, **kw: fake)
    pipe = Pipeline(cfg, tmp_path / "not-used.mp4", tmp_path / "out.mp4", work)
    pipe.stage_upscale()
    output = read_frames(work / "04_upscale" / "generated_full.mp4")
    assert output.shape == (5, 48, 64, 3)
    assert abs(int(output[0, 10, 10, 0]) - 128) < 10


def test_render_stage_writes_pending_file_not_last_good_result(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    pipe = service._pipeline(p, service.base_cfg)
    assert pipe.out.name == "result.pending.mp4"
    assert pipe.out != p.dir / "result.mp4"


def test_rejected_retrack_must_not_destroy_existing_valid_masks(tmp_path):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    p.update("actor", {"tracked": True, "prompts": [{"frame": 0, "points": [[2, 2]], "labels": [1], "box": None}]})
    p.update("mask_layers", {"ready": True, "manual": True, "tracked": True})
    p.update("keyframe", {"chosen": "approved"})
    existing = p.to_dict()
    service.jobs.submit = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("busy"))
    with pytest.raises(ValueError, match="busy"):
        service.track(p)
    assert p.to_dict() == existing


def test_bible_image_revision_conflict_is_reported_without_overwrite(tmp_path):
    service = StudioService(tmp_path / "data", None)
    client = TestClient(create_app(service))
    bible = service.bibles.create({"name": "Costume"})
    url = f"/api/garment-bibles/{bible['id']}/images"
    image = io.BytesIO()
    Image.new("RGB", (16, 24), "green").save(image, format="PNG")
    files = {"file": ("front.png", image.getvalue(), "image/png")}
    first = client.post(url, files=files, data={"view": "front", "expected_revision": "1"})
    assert first.status_code == 200
    stale = client.post(url, files=files, data={"view": "front", "expected_revision": "1"})
    assert stale.status_code == 409
    assert service.bibles.get(bible["id"])["revision"] == 2


def test_bible_image_rolls_back_if_metadata_save_fails(tmp_path, monkeypatch):
    service = StudioService(tmp_path / "data", None)
    bible = service.bibles.create({"name": "Costume"})
    old_photo = tmp_path / "before.png"
    new_photo = tmp_path / "after.png"
    Image.new("RGB", (16, 24), "red").save(old_photo)
    Image.new("RGB", (16, 24), "blue").save(new_photo)
    service.bibles.add_image(bible["id"], old_photo, "front")
    image_path = service.bibles._path(bible["id"]) / "front.png"
    original = image_path.read_bytes()
    original_revision = service.bibles.get(bible["id"])["revision"]
    def refuse_write(*a, **kw):
        raise OSError("simulated disk full")
    monkeypatch.setattr(service.bibles, "_write", refuse_write)
    with pytest.raises(OSError, match="simulated"):
        service.bibles.add_image(bible["id"], new_photo, "front")
    assert image_path.read_bytes() == original
    assert service.bibles.get(bible["id"])["revision"] == original_revision


def test_keyframe_mask_preparation_unpacks_only_selected_frame(tmp_path, monkeypatch):
    service = StudioService(tmp_path / "data", None)
    p = new_project(service, tmp_path)
    p.update("actor", {"tracked": True, "trim": [0, 12]})
    p.update("mask_layers", {"ready": True, "manual": True})
    actor = np.zeros((12, 48, 64), bool)
    actor[:, 5:40, 8:45] = True
    mk.save_masks(p.path("01_masks", "actor.npz"), actor)
    monkeypatch.setattr(service, "masks", lambda *a: (_ for _ in ()).throw(AssertionError("unpacks whole video")))
    index, selection, protection = service._keyframe_inputs(p, 6)
    assert index == 6
    assert selection.shape == protection.shape == (48, 64)
    assert selection.any() and not protection.any()


def test_failed_render_keeps_last_successful_video(tmp_path, monkeypatch):
    service = StudioService(tmp_path / "data", None)
    service.base_cfg["generation"]["backend"] = "scail2"
    p = new_project(service, tmp_path)
    p.update("actor", {"tracked": True, "trim": [0, 12]})
    p.update("mask_layers", {"ready": True, "manual": True})
    p.update("keyframe", {"chosen": "approved"})
    p.update("result", {"file": "result.mp4", "updated": 1, "stale": True})
    (p.dir / "result.mp4").write_bytes(b"old-good-video")
    p.path("meta.json").write_text(json.dumps({"width": 64, "height": 48,
                                               "num_frames": 12, "fps": 24, "start_frame": 0}))
    monkeypatch.setattr(service.worker, "stop", lambda: None)
    monkeypatch.setattr(service.models, "release", lambda: None)

    class SimulatedPipeline:
        on_scail_line = None
        on_scail_start = None
        def stage_scail(self):
            (p.dir / "result.pending.mp4").write_bytes(b"unfinished-video")
            raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(service, "_pipeline", lambda *a: SimulatedPipeline())
    job = service.generate(p)
    for _ in range(300):
        if job.finished is not None:
            break
        time.sleep(.01)
    assert job.status == "failed"
    assert (p.dir / "result.mp4").read_bytes() == b"old-good-video"
    assert p.state["result"]["stale"] is True


def test_completed_render_replaces_pending_atomically_and_marks_current(tmp_path, monkeypatch):
    service = StudioService(tmp_path / "data", None)
    service.base_cfg["generation"]["backend"] = "scail2"
    p = new_project(service, tmp_path)
    p.update("actor", {"tracked": True, "trim": [0, 12]})
    p.update("mask_layers", {"ready": True, "manual": True})
    p.update("keyframe", {"chosen": "approved"})
    p.update("result", {"file": "result.mp4", "updated": 1, "stale": True,
                        "stale_reason": "source"})
    (p.dir / "result.mp4").write_bytes(b"old-good-video")
    p.path("meta.json").write_text(json.dumps({"width": 64, "height": 48,
                                               "num_frames": 12, "fps": 24, "start_frame": 0}))
    monkeypatch.setattr(service.worker, "stop", lambda: None)
    monkeypatch.setattr(service.models, "release", lambda: None)

    class SimulatedPipeline:
        on_scail_line = None
        on_scail_start = None
        def stage_scail(self): pass
        def stage_upscale(self): pass
        def stage_composite(self):
            (p.dir / "result.pending.mp4").write_bytes(b"new-good-video")

    monkeypatch.setattr(service, "_pipeline", lambda *a: SimulatedPipeline())
    job = service.generate(p)
    for _ in range(300):
        if job.finished is not None:
            break
        time.sleep(.01)
    assert job.status == "done", job.error
    assert (p.dir / "result.mp4").read_bytes() == b"new-good-video"
    assert p.state["result"]["stale"] is False
    assert p.state["result"]["stale_reason"] is None


def test_legacy_safe_garment_filename_can_be_deleted(tmp_path):
    service = StudioService(tmp_path / 'data', None)
    p = new_project(service, tmp_path)
    image = p.path('garments', 'photo_01.png')
    image.write_bytes(b'legacy image')
    p.update('garment', {'images': ['photo_01.png']})
    service.remove_garment(p, 'photo_01.png')
    assert not image.exists() and p.state['garment']['images'] == []
    for bad in ['../../other.png', '..', '.secret.png']:
        with pytest.raises(ValueError, match='Unknown garment'):
            service.remove_garment(p, bad)


def test_api_never_serves_unlisted_project_garment_images(tmp_path):
    service = StudioService(tmp_path / 'data', None)
    p = new_project(service, tmp_path)
    p.path('garments', 'private.png').write_bytes(b'not a public garment')
    client = TestClient(create_app(service))
    assert client.get(f'/api/projects/{p.id}/garments/private.png').status_code == 404
