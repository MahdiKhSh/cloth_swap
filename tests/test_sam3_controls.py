"""SAM3 studio settings regression checks (do not require CUDA or model download)."""
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_swap.mask_layers import refine_part_mask
from cloth_swap.studio.sam3_controls import defaults, effective, schema, validate
from cloth_swap.studio.service import StudioService
from cloth_swap.studio.store import DEFAULT_STATE


def test_schema_only_exposes_wired_controls():
    data = schema()
    keys = {item['key'] for item in data['fields']}
    assert len(keys) >= 30
    assert keys == set(defaults())
    assert {'part_tracking_max_side', 'part_mask_resample', 'tracker_mask_threshold', 'image_size'} <= keys
    assert data['groups']['processing']


@pytest.mark.parametrize('payload', [
    {'not_a_real_setting': 42},
    {'part_tracking_max_side': 100},
    {'part_tracking_max_side': True},
    {'part_mask_resample': 'cubic'},
    {'tracker_mask_threshold': float('nan')},
    {'tracker_non_overlap': 'false'},
    {'min_trk_keep_alive': 200, 'max_trk_keep_alive': 20},
])
def test_invalid_settings_rejected(payload):
    with pytest.raises(ValueError):
        validate(payload)


def test_effective_values_preserve_yaml_overrides():
    base = {'score_threshold': 0.62, 'part_tracking_max_side': 1280}
    value = effective(base, {'part_tracking_max_side': 1920})
    assert value['score_threshold'] == 0.62
    assert value['part_tracking_max_side'] == 1920
    assert defaults()['part_tracking_max_side'] == 960


def test_refine_mask_default_is_binary_nearest():
    small = np.array([[0, 1], [1, 0]], dtype=bool)
    enlarged = refine_part_mask(small, (4, 4), defaults())
    assert enlarged.dtype == bool and enlarged.shape == (4, 4)
    assert np.all(enlarged[:2, 2:]) and not enlarged[0, 0]


def test_refine_mask_component_filter_and_dilate():
    small = np.zeros((20, 20), dtype=bool)
    small[2, 2] = True
    small[7:14, 7:14] = True
    result = refine_part_mask(small, (20, 20), {**defaults(), 'part_min_component_px': 5})
    assert not result[2, 2] and result[10, 10]
    grown = refine_part_mask(small, (20, 20), {**defaults(), 'part_mask_dilate_px': 2})
    assert grown.sum() > small.sum()


class DummyProject:
    def __init__(self):
        self.state = copy.deepcopy(DEFAULT_STATE)
        self.lock = threading.RLock()
        self.saved = 0

    def save(self):
        self.saved += 1


def test_settings_update_invalidates_dependent_masks(tmp_path):
    service = StudioService(tmp_path / 'settings', None)
    project = DummyProject()
    project.state['actor']['tracked'] = True
    project.state['mask_layers']['ready'] = True
    project.state['keyframe']['chosen'] = 'old'
    project.state['result'] = {'stale': False}
    service.set_sam3_settings(project, {'part_tracking_max_side': 1440})
    assert project.state['actor']['tracked']  # layers-only setting does not force person re-tracking
    assert not project.state['mask_layers']['ready']
    assert project.state['keyframe']['chosen'] is None
    assert project.state['result']['stale']
    assert project.saved == 1


def test_actor_settings_require_new_actor_tracking(tmp_path):
    service = StudioService(tmp_path / 'actor', None)
    project = DummyProject()
    project.state['actor']['tracked'] = True
    service.set_sam3_settings(project, {'tracker_mask_threshold': 0.2})
    assert not project.state['actor']['tracked']
    assert not project.state['mask_layers']['ready']


def test_preview_settings_do_not_invalidate_tracks(tmp_path):
    service = StudioService(tmp_path / 'preview', None)
    project = DummyProject()
    project.state['actor']['tracked'] = True
    project.state['mask_layers']['ready'] = True
    service.set_sam3_settings(project, {'image_mask_threshold': 0.7})
    assert project.state['actor']['tracked']
    assert project.state['mask_layers']['ready']
    assert project.saved == 1
