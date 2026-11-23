"""Mask union, subtraction, and image-selection validation (without CUDA)."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from cloth_swap.mask_layers import compose, expand_edit
from cloth_swap.studio.service import StudioService


def test_positive_minus_negative_no_leaks():
    actor = np.ones((3, 32, 32), dtype=bool)
    clothes = np.zeros_like(actor); clothes[:, 5:20, 4:20] = True
    pants = np.zeros_like(actor); pants[:, 18:30, 4:20] = True
    face = np.zeros_like(actor); face[:, 4:9, 7:12] = True
    shoes = np.zeros_like(actor); shoes[:, 27:32, 8:14] = True
    edit, protect = compose(actor, {'clothes': clothes, 'pants': pants, 'face': face, 'shoes': shoes}, [
        {'name': 'clothes', 'operation': 'add'}, {'name': 'pants', 'operation': 'add'},
        {'name': 'face', 'operation': 'subtract'}, {'name': 'shoes', 'operation': 'subtract'}])
    assert edit.any() and not (edit & protect).any()
    expanded = expand_edit(edit, protect, dilate_px=3)
    assert not (expanded & protect).any()


def test_image_only_selections_validated(tmp_path):
    s = StudioService(tmp_path / 'store', None)
    class FakeProject:
        id = 'stub'
        state = {'actor': {'tracked': True}, 'trim': {'start': 2, 'end': 10}}
        def update(self, section, values):
            self.state[section] = values
    project = FakeProject()
    rules = [{'name': 'pants', 'prompt': '', 'operation': 'add',
              'selection': {'frame': 5, 'points': [[20, 30]], 'labels': [1], 'box': None}}]
    s.set_mask_layers(project, rules)
    assert project.state['mask_layers']['rules'][0]['selection']['frame'] == 5
    rules[0]['selection']['frame'] = 12
    with pytest.raises(ValueError, match='inside video trim'):
        s.set_mask_layers(project, rules)


def test_reference_selection_requires_current_chosen_image(tmp_path):
    s = StudioService(tmp_path / "store2", None)
    class FakeProject:
        id = "stub"
        state = {"actor": {"tracked": True}, "trim": {"start": 2, "end": 10},
                 "keyframe": {"chosen": "ref123", "candidates": [{"id": "ref123", "frame": 5}]}}
        def update(self, section, values):
            self.state[section] = values
    project = FakeProject()
    rule = {"name": "jacket", "prompt": "", "operation": "add",
            "selection": {"frame": 5, "candidate": "ref123", "points": [[10, 12]],
                          "labels": [1], "box": None}}
    with pytest.raises(ValueError, match="original actor"):
        s.set_mask_layers(project, [rule])


@pytest.mark.parametrize("selection", [
    {"text": "face", "points": [], "labels": [], "box": None},
    {"points": [[20, 30]], "labels": [1], "box": None},
    {"points": [], "labels": [], "box": [10, 10, 40, 40]},
])
def test_three_modes_persist(tmp_path, selection):
    s = StudioService(tmp_path / 'modes', None)
    class Project:
        state = {'actor': {'tracked': True}, 'trim': {'start': 2, 'end': 10}}
        def update(self, section, values):
            self.state[section] = values
    p = Project()
    s.set_mask_layers(p, [{'name': 'face', 'operation': 'subtract',
                          'selection': {'frame': 5, **selection}}])
    saved = p.state['mask_layers']['rules'][0]
    assert saved['selection']['text'] == selection.get('text', '')
    assert saved['selection']['points'] == selection['points']
    assert saved['selection']['box'] == selection['box']


def test_text_preview_unions_instances_and_clips_actor(tmp_path):
    from contextlib import nullcontext
    s = StudioService(tmp_path / 'preview', None)
    image = np.zeros((10, 10, 3), np.uint8)
    actor = np.zeros((10, 10), bool); actor[2:8, 2:8] = True
    found = np.ones((10, 10), bool)
    s.isolated_frame = lambda p, t: image
    s.mask_frame = lambda *a: actor
    s._rel = lambda *a: 0
    s.jobs = SimpleNamespace(interactive=nullcontext)
    def get(kind):
        assert kind == 'detect'
        return SimpleNamespace(segment=lambda img, text, **kw: [found])
    s.models = SimpleNamespace(get=get)
    s.cfg_for = lambda p: {'sam3': {'score_threshold': 0.5, 'image_mask_threshold': 0.5}}
    result = s.segment_preview(None, 0, [], [], None, isolated=True, text='face')
    assert result['area'] == pytest.approx(0.36)
    assert result['polygons']
    with pytest.raises(ValueError, match='do not mix'):
        s.segment_preview(None, 0, [[3, 3]], [1], None, isolated=True, text='face')


@pytest.mark.parametrize('mode', ['text', 'point', 'box'])
def test_worker_tracks_selected_mode_and_subtracts(tmp_path, monkeypatch, mode):
    from cloth_swap import masks as mk, segmentation, video_io
    from cloth_swap.studio.worker import _mask_layers
    from cloth_swap.tracks import ConceptTracks
    actor = np.ones((2, 32, 32), bool)
    region = np.zeros_like(actor); region[:, 10:20, 10:20] = True
    frames = np.zeros((2, 32, 32, 3), np.uint8)
    monkeypatch.setattr(video_io, 'read_frames', lambda *a, **kw: frames)
    selection = {'frame': 4, 'points': [], 'labels': [], 'box': None}
    if mode == 'text':
        selection['text'] = 'face'
        def track(frames, prompts, cfg, **kw):
            assert prompts == ['face']
            tracks = ConceptTracks(2, 32, 32, prompt_of={1: 'face'})
            for t in range(2): tracks.add(1, t, region[t])
            return tracks
        monkeypatch.setattr(segmentation, 'track_concepts', track)
    else:
        if mode == 'point': selection.update(points=[[15, 15]], labels=[1])
        else: selection['box'] = [10, 10, 20, 20]
        def track(frames, prompts, cfg, **kw):
            assert prompts[0]['frame'] == 0
            assert prompts[0]['box'] == selection['box']
            assert prompts[0]['points'] == selection['points']
            return region.copy()
        monkeypatch.setattr(segmentation, 'track_prompts', track)
    mk.save_masks(tmp_path / 'actor.npz', actor)
    args = {'actor_path': str(tmp_path / 'actor.npz'), 'source': 'stub', 'start': 4, 'count': 2,
            'rules': [{'name': 'face', 'operation': 'subtract', 'selection': selection}],
            'output_dir': str(tmp_path / 'layers'), 'edit_path': str(tmp_path / 'edit.npz'),
            'protect_path': str(tmp_path / 'protect.npz')}
    from cloth_swap.studio.sam3_controls import defaults
    result = _mask_layers({}, {'sam3': defaults()}, args, lambda *a: None)
    edit = mk.load_masks(args['edit_path']); protect = mk.load_masks(args['protect_path'])
    assert result['layers']['face'] == 200
    assert not (edit & protect).any()
    assert not edit[:, 10:20, 10:20].any()
    assert edit[:, 0, 0].all()
