"""Resume only this Animate download, verify BF16 tensors, remove its FP32 duplicates.

The user approved this conversion/removal to fit disk space. Existing models are untouched.
"""
from pathlib import Path
import os
import json
import gc
import shutil
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
import requests

torch.set_num_threads(4)
repo = 'Wan-AI/Wan2.2-Animate-14B-Diffusers'
cache = Path('D:/hf_cache/hub')
hub = (cache / 'models--Wan-AI--Wan2.2-Animate-14B-Diffusers').resolve()
out = Path('D:/models/Wan2.2-Animate-14B-bf16/transformer')
out.mkdir(parents=True, exist_ok=True)
downloads = Path('D:/ai_cache/tmp/animate_download').resolve()
downloads.mkdir(parents=True, exist_ok=True)


def get_weight(name):
    url = hf_hub_url(repo, 'transformer/' + name)
    metadata = get_hf_file_metadata(url)
    download_url = metadata.location
    expected_hash = metadata.etag.strip('"')
    size = metadata.size
    # Reuse complete HF files or partial bytes from the same content-addressed revision.
    candidates = list((hub/'snapshots').glob('*/transformer/'+name))
    if candidates:
        return candidates[0]
    partials = [p for p in (hub/'blobs').glob(expected_hash+'*.incomplete') if p.stat().st_size <= size]
    own = downloads / (name + '.part')
    if own.exists():
        partials.append(own)
    path = max(partials, key=lambda p:p.stat().st_size) if partials else own
    offset = path.stat().st_size if path.exists() else 0
    chunk = 16 * 1024 * 1024
    workers = 8

    def fetch(bounds):
        start, end = bounds
        for attempt in range(6):
            try:
                with requests.get(download_url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15,40)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Server did not return the requested byte range')
                    data = response.content
                if len(data) != end-start+1:
                    raise RuntimeError('Truncated response')
                return data
            except (requests.RequestException, RuntimeError):
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt,10))

    with path.open('ab') as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        while offset < size:
            bounds = [(s,min(s+chunk-1,size-1)) for s in range(offset,min(size,offset+workers*chunk),chunk)]
            for data in pool.map(fetch,bounds):
                stream.write(data)
                offset += len(data)
            stream.flush()
            print(f'Download {name}: {offset*100/size:.1f}%', flush=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(16*1024*1024):
            digest.update(data)
    if digest.hexdigest() != expected_hash:
        raise RuntimeError('Downloaded SHA256 mismatch; file retained for inspection: '+str(path))
    return path

keep = ('time_embedder', 'scale_shift_table', 'norm1', 'norm2', 'norm3', 'motion_synthesis_weight', 'rope')
for name in ['config.json', 'diffusion_pytorch_model.safetensors.index.json']:
    source = hf_hub_download(repo, 'transformer/' + name, cache_dir=cache)
    shutil.copy2(source, out / name)
index = json.loads((out / 'diffusion_pytorch_model.safetensors.index.json').read_text())
for number, name in enumerate(sorted(set(index['weight_map'].values())), 1):
    target = out / name
    expected = {k for k,v in index['weight_map'].items() if v == name}
    if target.is_file():
        with safe_open(target, framework='pt', device='cpu') as handle:
            if set(handle.keys()) != expected:
                raise RuntimeError('Incomplete existing shard: ' + name)
        print('Already verified', number, flush=True)
        continue
    print('Downloading', number, name, flush=True)
    source = get_weight(name)
    print('Converting', number, flush=True)
    tensors = {}
    with safe_open(source, framework='pt', device='cpu') as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            tensors[key] = value.to(torch.bfloat16) if value.is_floating_point() and not any(x in key for x in keep) else value.clone()
    temporary = target.with_suffix('.tmp')
    save_file(tensors, str(temporary), metadata={'format': 'pt'})
    with safe_open(temporary, framework='pt', device='cpu') as check:
        if set(check.keys()) != expected:
            raise RuntimeError('Missing tensors: ' + name)
        for key, value in tensors.items():
            if not torch.equal(value, check.get_tensor(key)):
                raise RuntimeError('Tensor mismatch: ' + key)
    del tensors, value
    gc.collect()
    temporary.replace(target)
    # Resolve and verify the exact task-owned cache boundary before deleting a single file.
    if not (source.resolve().is_relative_to(hub) or source.resolve().is_relative_to(downloads)):
        raise RuntimeError('Unexpected cache path: ' + str(source))
    source.unlink()
    print('Verified BF16', number, flush=True)
for name in ['model_index.json','image_encoder/config.json','image_encoder/model.safetensors',
             'image_processor/preprocessor_config.json','scheduler/scheduler_config.json','vae/config.json']:
    hf_hub_download(repo, name, cache_dir=cache)
print('BF16_COMPLETE', out, flush=True)
