"""Studio backend: progress parsing, mask outlines and the project API (no models needed)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from cloth_swap.studio.jobs import Job  # noqa: E402
from cloth_swap.studio.service import ScailProgress, StudioService, mask_polygons  # noqa: E402
from cloth_swap.video_io import write_video  # noqa: E402


def test_scail_progress_parsing():
    job = Job("p", "generate", "generate", None)
    parse = ScailProgress(job)
    parse("[scail2] 2026 loading D:/models/umt5-xxl/models_t5.pth")
    assert job.message == "loading_text"
    parse("[scail2] 2026 Loaded 650/1300 DiT tensors")
    assert job.message == "loading_model" and 0.24 < job.progress < 0.26
    parse("[scail2] 2026 Processing segment 2/2: frames [76, 157)")
    parse(" 50%|█████     | 4/8 [00:30<00:30, 7.5s/it]")
    assert job.message == "sampling:2/2:4/8"
    assert job.progress == pytest.approx(0.37 + 0.53 * 0.75)


def test_mask_polygons_with_hole():
    m = np.zeros((60, 80), bool)
    m[10:50, 10:70] = True
    m[25:35, 30:50] = False
    polys = mask_polygons(m)
    assert len(polys) == 2 and all(len(p) >= 3 for p in polys)


def test_project_api(tmp_path):
    from fastapi.testclient import TestClient
    from cloth_swap.studio.api import create_app

    video = tmp_path / "clip.mp4"
    write_video(video, np.full((12, 64, 96, 3), 80, np.uint8), fps=24, crf=10)
    service = StudioService(tmp_path / "data", None)
    client = TestClient(create_app(service))

    with open(video, "rb") as f:
        res = client.post("/api/projects", files={"file": ("clip.mp4", f, "video/mp4")}, data={"name": "test"})
    assert res.status_code == 200
    pid, job = res.json()["project"]["id"], res.json()["job"]["id"]
    for _ in range(100):
        if client.get(f"/api/jobs/{job}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    p = client.get(f"/api/projects/{pid}").json()
    assert p["video"]["num_frames"] == 12 and p["trim"] == {"start": 0, "end": 12}
    assert client.get(f"/api/projects/{pid}/frames/5.jpg").status_code == 200
    assert client.get(f"/api/projects/{pid}/thumbs/11.jpg").status_code == 200

    p = client.patch(f"/api/projects/{pid}", json={"trim": {"start": 2, "end": 10}, "garment": {"description": "a red coat"}}).json()
    assert p["trim"] == {"start": 2, "end": 10} and p["garment"]["description"] == "a red coat"
    assert client.patch(f"/api/projects/{pid}", json={"trim": {"start": 2, "end": 4}}).status_code == 400

    prompts = [{"frame": 3, "points": [[10, 20]], "labels": [1], "box": None}]
    assert client.put(f"/api/projects/{pid}/actor/prompts", json={"prompts": prompts}).json()["actor"]["prompts"][0]["frame"] == 3
    assert client.post(f"/api/projects/{pid}/generate").status_code == 400  # not tracked yet
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}").status_code == 404

def test_native_quality_ignores_legacy_upscaling(tmp_path):
    from cloth_swap.studio.store import DEFAULT_STATE
    from types import SimpleNamespace
    import copy
    service = StudioService(tmp_path / 'native', None)
    state = copy.deepcopy(DEFAULT_STATE)
    state['trim'] = {'start': 0, 'end': 81}
    state['generate'].update(preset='high', output='4k', upscale='seedvr2')
    cfg = service.cfg_for(SimpleNamespace(state=state))
    assert cfg['scail']['target_area'] == 901120
    assert cfg['scail']['segment_len'] == 33
    assert cfg['output']['resolution'] == 'original'
    assert cfg['upscale']['backend'] == 'resize'
    state['generate']['preset'] = 'preview'
    assert service.cfg_for(SimpleNamespace(state=state))['scail']['segment_len'] == 81
