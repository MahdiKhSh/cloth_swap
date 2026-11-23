"""Contract tests for Animate routing and independent actor-part masks."""
from types import SimpleNamespace
import json
import numpy as np

from cloth_swap import animate_stage, masks as mk
from cloth_swap.config import load_config
from cloth_swap.pipeline import Pipeline
from cloth_swap.studio.service import StudioService, ScailProgress
from cloth_swap.studio.jobs import Job
from cloth_swap.video_io import write_video, read_frames


def test_animate_pipeline_routes_to_new_model(tmp_path, monkeypatch):
    cfg = load_config(overrides=['generation.backend=wan_animate'])
    frames = np.zeros((5,32,64,3),np.uint8)
    actor = np.ones((5,32,64),bool)
    pipe = Pipeline(cfg,tmp_path/'source.mp4',tmp_path/'result.mp4',tmp_path/'work')
    monkeypatch.setattr(pipe,'meta',lambda: dict(work_width=64,work_height=32,fps=24))
    monkeypatch.setattr(pipe,'frames',lambda:frames)
    monkeypatch.setattr(pipe,'load_mask',lambda name:actor)
    monkeypatch.setattr(pipe,'_keyframes',lambda:[(frames[0],actor[0])])
    seen=[]
    monkeypatch.setattr(animate_stage,'run',lambda *args,**kw:seen.append((args,kw)))
    pipe.stage_scail()
    assert len(seen)==1 and seen[0][0][5]==24
    assert seen[0][0][2] is actor


def test_prepare_manifest_runs_isolated_phases(tmp_path,monkeypatch):
    cfg=load_config(overrides=['generation.backend=wan_animate'])
    frames=np.full((5,32,64,3),70,np.uint8)
    mask=np.zeros((5,32,64),bool);mask[:,:,10:30]=True
    monkeypatch.setattr(animate_stage,'resolve_models',lambda cfg:(tmp_path/'models',tmp_path/'pose'))
    calls=[]
    monkeypatch.setattr(animate_stage,'run_process',lambda *args:calls.append(args))
    animate_stage.run(tmp_path/'job',frames,mask,frames[0],(64,32),24,cfg)
    assert [c[0][-1] for c in calls]==['prepare','text','generate']
    assert calls[1][0][0]==cfg['scail']['python']
    assert np.array_equal(mk.load_masks(tmp_path/'job/actor.npz'),mask)
    manifest=json.loads((tmp_path/'job/animate.json').read_text(encoding='utf-8'))
    assert manifest['frames']==5 and manifest['width']==64
    assert 'same face' in manifest['prompt']


def test_mask_parts_are_independent_and_older_projects_safe(tmp_path):
    service=StudioService(tmp_path/'data',None)
    all_masks={name:np.zeros((2,32,64),bool) for name in ('actor','face','hair')}
    all_masks['actor'][:,4:30,10:40]=True
    all_masks['face'][:,8:15,15:25]=True
    all_masks['hair'][:,4:8,12:29]=True
    service.masks=lambda p,name='actor':all_masks.get(name)
    service._rel=lambda p,t:t
    project=SimpleNamespace(state={'actor':{'tracked':True}})
    face=service.actor_outline(project,0,'face')
    hair=service.actor_outline(project,0,'hair')
    assert face['polygons'] and hair['polygons'] and face!=hair
    del all_masks['hair']
    assert service.actor_outline(project,0,'hair')=={'polygons':[]}
    assert service.actor_outline(project,10,'face')=={'polygons':[]}


def test_animate_progress_tracks_windows():
    job=Job('p','generate','generate',None)
    parse=ScailProgress(job)
    parse('[animate] sampling:2/3:10/20')
    assert 0.5 < job.progress < 0.8

def test_backend_change_invalidates_generation_cache(tmp_path, monkeypatch):
    cfg = load_config(overrides=['generation.backend=wan_animate'])
    work = tmp_path / 'work'
    pipe = Pipeline(cfg, tmp_path/'source.mp4', tmp_path/'out.mp4', work)
    stages = ('masks','keyframe','scail','upscale','composite')
    for name in stages:
        (work/f'{name}.done').write_text('old')
    (work/'03_scail').mkdir()
    (work/'03_scail/inputs.json').write_text(json.dumps({'backend':'scail2'}))
    calls = []
    for name in stages:
        monkeypatch.setattr(pipe, f'stage_{name}', lambda name=name: calls.append(name))
    pipe.run()
    assert calls == ['scail','upscale','composite']
