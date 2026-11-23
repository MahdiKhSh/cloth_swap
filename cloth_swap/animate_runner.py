"""Isolated preparation, prompt encoding and inference for Wan2.2 Animate."""
from __future__ import annotations
import argparse
import hashlib
import gc
import json
import sys
from pathlib import Path
import torch  # load CUDA DLLs before OpenCV
import cv2
import numpy as np
from PIL import Image

from . import masks as mk
from .mask_layers import expand_edit
from .video_io import read_frames, write_video


def wan_animate_with_precomputed_embeddings(pipeline_class):
    """Adapt Diffusers input validation for its reference-image dual-use path.

    Wan Animate uses the PIL reference for VAE latents AND (if already computed)
    CLIP image embeddings for conditioning. Some Diffusers versions reject both
    in check_inputs(), despite consuming both later in __call__. Keep both for
    the real forward pass; validate the image normally and keep the supplied
    embeddings rather than forcing a second CLIP model into VRAM.

    Only the validation call is adapted; no installed package is patched.
    """
    import inspect

    class PrecomputedImageWanAnimatePipeline(pipeline_class):
        def check_inputs(self, *args, **kwargs):
            signature = inspect.signature(pipeline_class.check_inputs)
            bound = signature.bind(self, *args, **kwargs)
            image = bound.arguments.get("image")
            image_embeds = bound.arguments.get("image_embeds")
            if image is not None and image_embeds is not None:
                if not isinstance(image_embeds, torch.Tensor) or image_embeds.ndim != 3:
                    raise ValueError("Precomputed CLIP image_embeds must be a [batch, tokens, features] tensor")
                # Only bypass the contradictory mutual-exclusion test. The
                # actual __call__ keeps both inputs for VAE and CLIP paths.
                bound.arguments["image_embeds"] = None
            return pipeline_class.check_inputs(*bound.args, **bound.kwargs)

    return PrecomputedImageWanAnimatePipeline


def prepare(a, folder):
    prep = Path(a['repo_dir']) / 'wan/modules/animate/preprocess'
    sys.path.insert(0, str(prep))
    from pose2d import ViTPose
    from pose2d_utils import load_pose_metas_from_kp2ds_seq, AAPoseMeta
    from human_visualization import draw_aapose_by_meta_new
    from utils import get_face_bboxes
    import onnxruntime as ort
    print('[animate] loading_pose', flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    model = ViTPose.__new__(ViTPose)
    model.session = ort.InferenceSession(str(Path(a['pose_dir']) / 'end2end.onnx'),
        sess_options=options, providers=['CPUExecutionProvider'])
    print('[animate] pose_ready', flush=True)
    frames = read_frames(folder / 'driving.mp4')
    masks = mk.load_masks(folder / 'actor.npz')
    if (folder / 'edit.npz').exists():
        edit = mk.load_masks(folder / 'edit.npz')
        protect = mk.load_masks(folder / 'protect.npz')
        if edit.shape != masks.shape or protect.shape != masks.shape:
            raise ValueError('Layered mask dimensions do not match actor')
        # Pose uses original actor, but WAN replacement is restricted to edit pixels.
        edit = expand_edit(edit, protect, dilate_px=max(1, round(masks.shape[1]*0.015)))
    else:
        edit = None
    h, w = frames.shape[1:3]
    poses, faces, backgrounds, regions = [], [], [], []
    for i, frame in enumerate(frames):
        actor = cv2.resize(masks[i].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(actor)
        face = np.zeros((512,512,3), np.uint8)
        canvas = np.zeros_like(frame)
        if len(xs):
            # Explicit actor box: never let a detector switch to a larger passer-by.
            box = np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
            inp, center, scale = model.preprocess(frame[..., ::-1], box)
            keypoints = model(inp[None], center[None], scale[None])
            meta = load_pose_metas_from_kp2ds_seq(keypoints, width=w, height=h)[0]
            canvas = draw_aapose_by_meta_new(canvas, AAPoseMeta.from_humanapi_meta(meta))
            x1,x2,y1,y2 = get_face_bboxes(meta['keypoints_face'][:,:2], scale=1.3, image_shape=(h,w))
            x1,x2 = max(0,int(x1)), min(w,int(x2))
            y1,y2 = max(0,int(y1)), min(h,int(y2))
            if x2 > x1 and y2 > y1:
                face = cv2.resize(frame[y1:y2,x1:x2], (512,512))
        radius = max(1, round(h * 0.015))
        mask = (cv2.resize(edit[i].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
                if edit is not None else
                cv2.dilate(actor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))))
        if edit is not None:
            protect_small = cv2.resize(protect[i].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
            mask[protect_small > 0] = 0
        poses.append(canvas); faces.append(face)
        backgrounds.append(frame * (1-mask[...,None]))
        regions.append(np.repeat((mask*255)[...,None],3,axis=2))
        print(f'[animate] pose:{i+1}/{len(frames)}', flush=True)
    for name, data in [('pose',poses), ('face',faces), ('background',backgrounds), ('mask',regions)]:
        write_video(folder / f'{name}.mp4', data, a['fps'], crf=10)


def text(a, folder):
    from .scail2_runner import load_t5_fast
    sys.path.insert(0,a['scail']['repo_dir'])
    from wan.configs import SCAIL_CONFIGS
    config = SCAIL_CONFIGS[a['scail']['model']]
    key = hashlib.sha1(f"{config.t5_checkpoint}\n{a['prompt']}".encode('utf-8')).hexdigest()
    cache = Path(__file__).resolve().parent.parent / '.cache' / 't5' / f'{key}.pt'
    value = None
    if cache.is_file():
        value = torch.load(cache, weights_only=True).get(a['prompt'])
    if value is None:
        device = torch.device('cuda')
        encoder = load_t5_fast(config, a['scail']['ckpt_dir'], device)
        with torch.inference_mode():
            value = encoder([a['prompt']],device)[0].cpu()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({a['prompt']: value}, cache)
    value = torch.cat([value, value.new_zeros(512-value.shape[0],value.shape[1])])[None]
    torch.save(value, folder/'prompt.pt')


def generate(a, folder):
    from diffusers import (AutoencoderKLWan, WanAnimatePipeline, WanAnimateTransformer3DModel,
                           UniPCMultistepScheduler)
    from transformers import CLIPImageProcessor, CLIPVisionModel
    repo = a['model_dir']
    print('[animate] loading_image', flush=True)
    image = Image.open(folder/'reference.png').convert('RGB')
    encoder = CLIPVisionModel.from_pretrained(repo, subfolder='image_encoder',
                       torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    processor = CLIPImageProcessor.from_pretrained(repo,subfolder='image_processor',local_files_only=True)
    with torch.inference_mode():
        pixels = processor(images=image,return_tensors='pt').pixel_values.to('cuda',torch.bfloat16)
        embeds = encoder(pixels,output_hidden_states=True).hidden_states[-2].cpu()
    del encoder, processor, pixels
    gc.collect(); torch.cuda.empty_cache()
    print('[animate] loading_model', flush=True)
    transformer_dir = a.get('transformer_dir') or str(Path(repo) / 'transformer')
    transformer = WanAnimateTransformer3DModel.from_pretrained(transformer_dir,
         torch_dtype=torch.bfloat16, device_map='auto',
         max_memory={0: a.get('gpu_weight_memory', '22GiB'), 'cpu': '14GiB'},
         offload_folder=str(folder / 'offload'), local_files_only=True, low_cpu_mem_usage=True)
    print('[animate] device_map=' + str(transformer.hf_device_map), flush=True)
    vae = AutoencoderKLWan.from_single_file(a['vae'],config=repo,subfolder='vae',torch_dtype=torch.float32)
    vae.enable_tiling()
    scheduler = UniPCMultistepScheduler.from_pretrained(repo,subfolder='scheduler',local_files_only=True)
    # The pipeline needs image= for VAE reference latents and image_embeds=
    # for CLIP conditioning. Its stock validator incorrectly forbids the pair.
    # We keep the precomputed CLIP embedding to avoid loading the encoder again.
    CompatibleWanAnimatePipeline = wan_animate_with_precomputed_embeddings(WanAnimatePipeline)
    pipe = CompatibleWanAnimatePipeline(tokenizer=None,text_encoder=None,image_encoder=None,image_processor=None,
                                        transformer=transformer,vae=vae,scheduler=scheduler)
    vae.to('cuda')
    videos = {name+'_video': [Image.fromarray(f) for f in read_frames(folder/f'{name}.mp4')]
              for name in ['pose','face','background','mask']}
    window = a.get('segment_len',33)
    total = max(1, (a['frames'] - 1 + window - 2) // (window - 1))
    segment = 0
    def progress(pipe,step,timestep,kwargs):
        nonlocal segment
        if step == 0:
            segment += 1
        print(f'[animate] sampling:{segment}/{total}:{step+1}/{a["steps"]}',flush=True)
        return kwargs
    result = pipe(image=image, **videos, image_embeds=embeds.to('cuda'),
        prompt_embeds=torch.load(folder/'prompt.pt',weights_only=True).to('cuda'),
        height=a['height'],width=a['width'],mode='replace',
        segment_frame_length=a.get('segment_len',33),prev_segment_conditioning_frames=1,
        motion_encode_batch_size=1,num_inference_steps=a['steps'],guidance_scale=1.0,
        generator=torch.Generator('cuda').manual_seed(a['seed']),output_type='np',callback_on_step_end=progress).frames[0]
    if len(result) < a['frames']:
        raise RuntimeError(f'Animate returned {len(result)} frames, expected {a["frames"]}')
    write_video(folder/'scail_out.mp4', (np.clip(f*255+0.5,0,255).astype(np.uint8)
                 for f in result[:a['frames']]),a['fps'],crf=12)
    print('[animate] done',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('phase',choices=['prepare','text','generate'])
    args=p.parse_args(); path=Path(args.manifest)
    globals()[args.phase](json.loads(path.read_text(encoding='utf-8')),path.parent)

if __name__=='__main__':
    main()
