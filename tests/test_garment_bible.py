"""Shared garment continuity, API, stale references; runs without loading AI models."""
import io
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from cloth_swap.studio.api import create_app
from cloth_swap.studio.service import StudioService


def sample_project(service, tmp_path: Path, label):
    path = tmp_path / f"{label}.mp4"
    path.write_bytes(b"dummy-video-not-decoded")
    return service.store.create(label, path)


def sample_image(path: Path, color=(200, 20, 60)):
    Image.new("RGB", (40, 60), color).save(path)


def test_shared_bible_across_projects_and_versions(tmp_path):
    service = StudioService(tmp_path / "data", None)
    a, b = sample_project(service, tmp_path, "scene-a"), sample_project(service, tmp_path, "scene-b")
    a.update("garment", {"images": ["photo.png"], "description": "red denim jacket"})
    sample_image(a.path("garments", "photo.png"))
    bible = service.bibles.create({"name": "Hero jacket", "color": "scarlet", "collar": "stand collar"}, a)
    assert bible["revision"] == 1
    assert len(bible["images"]) == 1
    assert service.bibles.list()[0]["id"] == bible["id"]

    a.update("keyframe", {"chosen": "approved-old"})
    b.update("keyframe", {"chosen": "approved-old"})
    service.bibles.apply(bible["id"], a)
    service.bibles.apply(bible["id"], b)
    assert a.state["keyframe"]["chosen"] is None
    assert b.state["garment"]["images"] == a.state["garment"]["images"]
    assert "exact color: scarlet" in b.state["garment"]["description"]
    assert service.bibles._path(bible["id"]).joinpath("front.png").exists()
    assert (b.dir / "garments" / b.state["garment"]["images"][0]).is_file()

    a.update("keyframe", {"chosen": "now-approved"})
    a.update("result", {"file": "result.mp4", "updated": 1})
    same = service.bibles.apply(bible["id"], a)
    assert same["keyframe"]["chosen"] == "now-approved"  # no false invalidation
    updated = service.bibles.update(bible["id"], {"color": "dark burgundy"})
    assert updated["revision"] == 2
    with pytest.raises(ValueError, match="reapply"):
        a.update("trim", {"start": 0, "end": 10})
        service.cfg_for(a)
    service.bibles.apply(bible["id"], a)
    assert a.state["keyframe"]["chosen"] is None
    assert a.state["result"]["stale"] is True
    assert a.state["garment"]["bible_revision"] == 2
    assert "dark burgundy" in a.state["garment"]["description"]
    assert "dark burgundy" not in b.state["garment"]["description"]


def test_bible_image_view_replacement_and_validation(tmp_path):
    service = StudioService(tmp_path / "data", None)
    bible = service.bibles.create({"name": "Black coat", "description": "black coat"})
    image = tmp_path / "front.jpg"
    sample_image(image)
    bible = service.bibles.add_image(bible["id"], image, "front")
    assert bible["revision"] == 2 and bible["images"] == [{"view": "front", "file": "front.png"}]
    bible = service.bibles.add_image(bible["id"], image, "front")
    assert len(bible["images"]) == 1  # replace, not duplicate
    with pytest.raises(ValueError, match="view"):
        service.bibles.add_image(bible["id"], image, "random")
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("not an image")
    with pytest.raises(ValueError, match="Invalid"):
        service.bibles.add_image(bible["id"], invalid, "back")


def test_bible_endpoints_and_shot_sync(tmp_path):
    service = StudioService(tmp_path / "data", None)
    a, b = sample_project(service, tmp_path, "one"), sample_project(service, tmp_path, "two")
    client = TestClient(create_app(service))
    bible_resp = client.post("/api/garment-bibles", json={"name": "Navy suit", "description": "navy blazer", "collar": "notched"})
    assert bible_resp.status_code == 200
    bible_id = bible_resp.json()["id"]
    assert client.get("/api/garment-bibles").json()[0]["name"] == "Navy suit"
    assert client.post(f"/api/projects/{a.id}/garment-bible/{bible_id}").status_code == 200
    assert client.post(f"/api/projects/{b.id}/garment-bible/{bible_id}").status_code == 200
    assert client.patch(f"/api/projects/{a.id}", json={"garment": {"description": "different coat"}}).status_code == 409
    resp = client.patch(f"/api/garment-bibles/{bible_id}", json={"sleeves": "long with four buttons"})
    assert resp.status_code == 200 and resp.json()["revision"] == 2
    synced = client.post(f"/api/garment-bibles/{bible_id}/sync").json()
    assert set(synced["synced"]) == {a.id, b.id}
    assert a.state["garment"]["bible_revision"] == 2
    assert b.state["garment"]["bible_revision"] == 2
    assert "four buttons" in b.state["garment"]["description"]
    image = io.BytesIO()
    Image.new("RGB", (24, 24), "orange").save(image, format="PNG")
    response = client.post(f"/api/garment-bibles/{bible_id}/images", files={"file": ("front.png", image.getvalue(), "image/png")}, data={"view": "front"})
    assert response.status_code == 200
    assert client.get(f"/api/garment-bibles/{bible_id}/images/front.png").status_code == 200
    assert client.get(f"/api/garment-bibles/{bible_id}/images/missing.png").status_code == 404
    assert client.get(f"/api/garment-bibles/invalid").status_code == 404
    assert client.patch(f"/api/garment-bibles/{bible_id}", json={"id": "nonsense"}).status_code == 400
