"""SCAIL-2 replacement runner, executed with the SCAIL-2 environment's Python.

It loads the inputs exactly like the repository's generate.py, but accepts several LoRAs
(e.g. lightx2v + relighting) and reads the prompt from a file. It must stay importable
without the cloth_swap package, because it runs in a different environment.

--weights fp8 (default) streams the DiT from a .safetensors file straight to the GPU and keeps
the 40 transformer blocks' linear layers in float8 with a per-row scale (~16 GB instead of
~66 GB in fp32), merging the LoRAs on the way. That is what lets the 14B model run on a
32 GB GPU with 32 GB of system RAM. --weights full uses the repository's own loader.
"""
import argparse
import logging
import os
import sys
import types


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="SCAIL-2 repository (branch wan-scail2)")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--scail_path", required=True, help="DiT .safetensors (convert.py output or Comfy-Org fp16/fp8_scaled)")
    p.add_argument("--model", default="SCAIL-14B")
    p.add_argument("--image", required=True)
    p.add_argument("--mask_image", required=True)
    p.add_argument("--pose", required=True, help="driving video")
    p.add_argument("--mask_video", required=True)
    p.add_argument("--additional_image", nargs="*", default=[])
    p.add_argument("--additional_mask", nargs="*", default=[])
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target_w", type=int, required=True)
    p.add_argument("--target_h", type=int, required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--solver", default="unipc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--segment_len", type=int, default=81)
    p.add_argument("--segment_overlap", type=int, default=5)
    p.add_argument("--lora", nargs=2, action="append", metavar=("PATH", "ALPHA"), default=[])
    p.add_argument("--t5_cpu", action="store_true")
    p.add_argument("--offload_model", choices=("true", "false"), default="true")
    p.add_argument("--weights", choices=("fp8", "full"), default="fp8")
    p.add_argument("--attention", choices=("auto", "sdpa", "flash"), default="auto")
    p.add_argument("--text_cache", default=None, help="folder caching T5 prompt encodings (skips loading T5 for a known prompt)")
    return p.parse_args()


# ----------------------------------------------------------------------------- environment shims
def install_decord_shim():
    """scail_utils imports decord at module level; fall back to imageio-ffmpeg where it is missing."""
    try:
        import decord  # noqa: F401
        return
    except ImportError:
        pass
    import imageio_ffmpeg
    import numpy as np
    import torch

    class VideoReader:
        def __init__(self, uri, height=-1, width=-1, **kwargs):
            reader = imageio_ffmpeg.read_frames(str(uri))
            meta = next(reader)
            w, h = meta["size"]
            self._frames = np.stack([np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in reader])

        def __len__(self):
            return len(self._frames)

        def get_batch(self, indices):
            return torch.from_numpy(self._frames[np.asarray(indices)].copy())

    shim = types.ModuleType("decord")
    shim.VideoReader = VideoReader
    shim.bridge = types.SimpleNamespace(set_bridge=lambda name: None)
    sys.modules["decord"] = shim
    logging.info("decord not installed; using an imageio-ffmpeg reader")


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                   causal=False, window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    """Drop-in for wan.modules.attention.flash_attention using torch SDPA ([B, L, N, C] layout)."""
    import torch
    import torch.nn.functional as F

    dtype = dtype or torch.bfloat16
    out_dtype = q.dtype
    lk = k.size(1)

    def half(t):
        return t if t.dtype in (torch.float16, torch.bfloat16) else t.to(dtype)

    q, k, v = (half(t).transpose(1, 2) for t in (q, k, v))
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None:
        k_lens = k_lens.to(q.device)
        if bool((k_lens < lk).any()):
            mask = (torch.arange(lk, device=q.device)[None, :] < k_lens[:, None])[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                         is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().type(out_dtype)


def patch_attention(mode):
    from wan.modules import attention

    if mode == "flash" or (mode == "auto" and (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE)):
        return
    original = attention.flash_attention
    for name, module in list(sys.modules.items()):
        if name.startswith("wan.") and getattr(module, "flash_attention", None) is original:
            module.flash_attention = sdpa_attention
    logging.info("Using torch scaled_dot_product_attention instead of flash-attn")


# ----------------------------------------------------------------------------- activation memory
ROPE_CHUNK = 8192


def _rope_rotate(x, freqs_i):
    """x: [L, N, D] real, freqs_i: [L, 1, D/2] complex -> rotated float32 [L, N, D], in chunks."""
    import torch

    seq_len, n = x.size(0), x.size(1)
    out = torch.empty(seq_len, n, x.size(2), dtype=torch.float32, device=x.device)
    for s in range(0, seq_len, ROPE_CHUNK):
        xc = torch.view_as_complex(x[s:s + ROPE_CHUNK].float().reshape(-1, n, x.size(2) // 2, 2))
        out[s:s + ROPE_CHUNK] = torch.view_as_real(xc * freqs_i[s:s + ROPE_CHUNK].to(torch.complex64)).flatten(2)
    return out


def _video_freqs(freqs, c, f, h, w, shift_f, shift_h, shift_w):
    import torch

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    return torch.cat([
        freqs[0][shift_f:shift_f + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][shift_h:shift_h + h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][shift_w:shift_w + w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1)


def rope_apply_ref(x, freqs, **kwargs):
    """Same result as wan.modules.model_scail2.rope_apply_ref, in complex64 chunks instead of complex128."""
    import torch

    key = kwargs.get("rope_key", "ref")
    f = kwargs.get("rope_ref_T", {}).get(key, 1)
    h, w = kwargs["rope_H"], kwargs["rope_W"]
    freqs_i = _video_freqs(freqs, x.size(3) // 2, f, h, w, kwargs["rope_T_shift"][key],
                           kwargs["rope_H_shift"][key], kwargs["rope_W_shift"][key]).reshape(f * h * w, 1, -1)
    assert f * h * w == x.size(1)
    return torch.stack([_rope_rotate(x[i], freqs_i) for i in range(x.size(0))])


def rope_apply_video(x, freqs, **kwargs):
    import torch

    f, h, w = kwargs["rope_T"], kwargs["rope_H"], kwargs["rope_W"]
    freqs_i = _video_freqs(freqs, x.size(3) // 2, f, h, w, kwargs["rope_T_shift"]["video"],
                           kwargs["rope_H_shift"]["video"], kwargs["rope_W_shift"]["video"]).reshape(f * h * w, 1, -1)
    assert f * h * w == x.size(1)
    return torch.stack([_rope_rotate(x[i], freqs_i) for i in range(x.size(0))])


def rope_apply_pose(x, freqs, **kwargs):
    import torch
    import torch.nn.functional as F

    f, h, w = kwargs["rope_T"], kwargs["rope_H"], kwargs["rope_W"]
    seq_len = f * (h // 2) * (w // 2)
    assert seq_len == x.size(1)
    full = _video_freqs(freqs, x.size(3) // 2, f, h, w, kwargs["rope_T_shift"]["pose"],
                        kwargs["rope_H_shift"]["pose"], kwargs["rope_W_shift"]["pose"])  # T H W D

    def pool(t):
        return F.avg_pool2d(t.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
    freqs_i = torch.complex(pool(full.real), pool(full.imag)).reshape(seq_len, 1, -1)
    return torch.stack([_rope_rotate(x[i], freqs_i) for i in range(x.size(0))])


def make_chunked_ffn_class():
    import torch

    class ChunkedFFN(torch.nn.Module):
        """Runs the block's FFN over the sequence in pieces (its 13824-wide hidden layer is the largest activation)."""

        def __init__(self, ffn, chunk):
            super().__init__()
            self.ffn = ffn
            self.chunk = chunk

        def forward(self, x):
            if x.size(1) <= self.chunk:
                return self.ffn(x)
            return torch.cat([self.ffn(part) for part in x.split(self.chunk, dim=1)], dim=1)

    return ChunkedFFN


def reduce_activation_memory(model, ffn_chunk=16384):
    import wan.modules.model_scail2 as m2

    m2.rope_apply_ref = rope_apply_ref
    m2.rope_apply_video = rope_apply_video
    m2.rope_apply_pose = rope_apply_pose
    ChunkedFFN = make_chunked_ffn_class()
    for block in model.blocks:
        block.ffn = ChunkedFFN(block.ffn, ffn_chunk)


# ----------------------------------------------------------------------------- fp8 DiT loading
def make_fp8_linear_class():
    import torch

    class FP8Linear(torch.nn.Module):
        """Linear layer with a float8_e4m3fn weight and a float32 scale per output row."""

        def __init__(self, in_features, out_features, bias, device):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device))
            self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32, device=device))
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32, device=device), requires_grad=False)
            else:
                self.register_parameter("bias", None)

        @torch.no_grad()
        def set_weight(self, w):
            scale = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12) / 448.0
            self.weight.copy_((w / scale).to(torch.float8_e4m3fn))
            self.scale.copy_(scale)

        def forward(self, x):
            if torch.is_autocast_enabled(x.device.type):
                dtype = torch.get_autocast_dtype(x.device.type)
            else:
                dtype = x.dtype
            w = self.weight.to(dtype) * self.scale.to(dtype)
            bias = self.bias.to(dtype) if self.bias is not None else None
            return torch.nn.functional.linear(x.to(dtype), w, bias)

    return FP8Linear


def collect_lora_deltas(loras):
    """{param name: [(alpha, up, down)]} and {bias name: [(alpha, diff_b)]}, as in wan.utils.lora."""
    from safetensors.torch import load_file

    weights, biases = {}, {}
    for path, alpha in loras:
        state = load_file(path)
        used = 0
        for key in state:
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key[: -len(".lora_down.weight")]
            up = state.get(prefix + ".lora_up.weight")
            if up is None:
                continue
            target = prefix[len("diffusion_model."):] if prefix.startswith("diffusion_model.") else prefix
            weights.setdefault(target + ".weight", []).append((alpha, up, state[key]))
            if prefix + ".diff_b" in state:
                biases.setdefault(target + ".bias", []).append((alpha, state[prefix + ".diff_b"]))
            used += 1
        logging.info("LoRA %s: %d layers (alpha %s)", os.path.basename(path), used, alpha)
    return weights, biases


def load_dit_fp8(model_cls, config_path, weights_path, loras, device):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    FP8Linear = make_fp8_linear_class()
    with init_empty_weights():
        model = model_cls.from_config(config_path)
    for name, module in list(model.named_modules()):
        if name.startswith("blocks.") and isinstance(module, torch.nn.Linear):
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child,
                    FP8Linear(module.in_features, module.out_features, module.bias is not None, device))

    lora_weights, lora_biases = collect_lora_deltas(loras)
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if not k.endswith(".scale_weight") and k != "scaled_fp8"]
        all_keys = set(f.keys())
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            value = tensor.to(device, torch.float32)
            if tensor.dtype == torch.float8_e4m3fn:  # Comfy "fp8_scaled": one scale per tensor
                scale_key = key[: -len("weight")] + "scale_weight"
                if scale_key in all_keys:
                    value = value * f.get_tensor(scale_key).to(device, torch.float32)
            for alpha, up, down in lora_weights.pop(key, []):
                value += alpha * (up.to(device, torch.float32) @ down.to(device, torch.float32))
            for alpha, diff in lora_biases.pop(key, []):
                value += alpha * diff.to(device, torch.float32)
            module_name, param_name = key.rsplit(".", 1)
            module = model.get_submodule(module_name)
            if isinstance(module, FP8Linear):
                if param_name == "weight":
                    module.set_weight(value)
                else:
                    module.bias.data.copy_(value)
            else:
                set_module_tensor_to_device(model, key, device, value=value)
            if (i + 1) % 200 == 0:
                logging.info("Loaded %d/%d DiT tensors", i + 1, len(keys))
    for name in list(lora_weights) + list(lora_biases):
        logging.warning("[Skip] LoRA target %s not in model", name)
    on_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if on_meta:
        raise RuntimeError(f"DiT tensors missing from {weights_path}: {on_meta[:5]} ...")
    model.eval().requires_grad_(False)
    used = torch.cuda.memory_allocated(device) / 2**30
    logging.info("DiT loaded in fp8 (%.1f GB on GPU)", used)
    return model


def load_t5_fast(cfg, ckpt_dir, device):
    """T5EncoderModel without building a random fp32 copy first (the repo's loader needs ~45 GB RAM).

    The encoder is created on the meta device and the loaded checkpoint tensors are assigned to it,
    so only one bf16 copy (11 GB) is ever in RAM. The file is read in one sequential pass: mmap
    page faults made the same load take ~6 minutes from a hard disk.
    """
    import torch
    from wan.modules.t5 import T5EncoderModel, umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

    path = os.path.join(ckpt_dir, cfg.t5_checkpoint)
    encoder = T5EncoderModel.__new__(T5EncoderModel)
    encoder.text_len = cfg.text_len
    encoder.dtype = cfg.t5_dtype
    encoder.device = device
    encoder.checkpoint_path = path
    encoder.tokenizer_path = os.path.join(ckpt_dir, cfg.t5_tokenizer)
    model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=cfg.t5_dtype,
                     device=torch.device("meta")).eval().requires_grad_(False)
    logging.info("loading %s", path)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), assign=True)
    encoder.model = model.to(device=device, dtype=cfg.t5_dtype)
    encoder.tokenizer = HuggingfaceTokenizer(name=encoder.tokenizer_path, seq_len=cfg.text_len, clean="whitespace")
    return encoder


class CachedTextEncoder:
    """Stands in for the T5 encoder after the prompts were encoded, so its 11 GB can be freed."""

    def __init__(self, contexts):
        self.contexts = contexts

    def __call__(self, texts, device):
        return [self.contexts[t].to(device) for t in texts]


def build_pipeline_fp8(args, cfg, config_path, device, prompt):
    """SCAIL2Pipeline.__init__ without its fp32 DiT loading (same attributes, same components).

    The prompt is encoded with T5 on the GPU before the DiT is loaded and T5 is then released,
    and CLIP goes back to the CPU after use, so system RAM and VRAM hold little besides the DiT.
    """
    import gc

    import torch
    import wan
    from wan.modules.clip import CLIPModel
    from wan.modules.model_scail2 import SCAIL2Model
    from wan.modules.vae import WanVAE

    pipe = wan.SCAIL2Pipeline.__new__(wan.SCAIL2Pipeline)
    pipe.device = torch.device(f"cuda:{device}")
    pipe.config = cfg
    pipe.rank = 0
    pipe.use_usp = False
    pipe.lora_path = None
    pipe.lora_alpha = None
    pipe.num_train_timesteps = cfg.num_train_timesteps
    pipe.param_dtype = cfg.param_dtype

    contexts = None
    cache_file = None
    if args.text_cache:
        import hashlib
        key = hashlib.sha1(f"{cfg.t5_checkpoint}\n{prompt}".encode("utf-8")).hexdigest()
        cache_file = os.path.join(args.text_cache, f"{key}.pt")
        if os.path.exists(cache_file):
            contexts = torch.load(cache_file, weights_only=True)
            logging.info("Prompt encoding loaded from cache (T5 not needed)")
    if contexts is None:
        t5_device = torch.device("cpu") if args.t5_cpu else pipe.device
        text_encoder = load_t5_fast(cfg, args.ckpt_dir, t5_device)
        with torch.no_grad():
            contexts = {text: text_encoder([text], t5_device)[0].cpu() for text in (prompt, "")}
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        if cache_file:
            os.makedirs(args.text_cache, exist_ok=True)
            torch.save(contexts, cache_file)
    pipe.text_encoder = CachedTextEncoder(contexts)
    pipe.t5_cpu = True  # generate() then asks the stub for CPU tensors and moves them to the GPU

    pipe.vae_stride = cfg.vae_stride
    pipe.patch_size = cfg.patch_size
    pipe.vae = WanVAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=pipe.device)
    pipe.clip = CLIPModel(
        dtype=cfg.clip_dtype, device=pipe.device,
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.clip_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.clip_tokenizer))
    pipe.clip.model.cpu()
    clip_visual = pipe.clip.visual

    def visual_then_offload(videos):
        out = clip_visual(videos)
        pipe.clip.model.cpu()
        torch.cuda.empty_cache()
        return out
    pipe.clip.visual = visual_then_offload

    loras = [(path, float(alpha)) for path, alpha in args.lora]
    pipe.model = load_dit_fp8(SCAIL2Model, config_path, args.scail_path, loras, pipe.device)
    reduce_activation_memory(pipe.model)

    # VAE decoding of a whole segment needs ~10 GB; with the DiT resident that spills into shared
    # system memory on a 32 GB card and becomes very slow, so the DiT waits in RAM meanwhile.
    vae_decode = pipe.vae.decode

    def decode_with_dit_offloaded(zs):
        pipe.model.to("cpu")
        torch.cuda.empty_cache()
        try:
            return vae_decode(zs)
        finally:
            pipe.model.to(pipe.device)
    pipe.vae.decode = decode_with_dit_offloaded
    pipe.sp_size = 1
    pipe.sample_neg_prompt = cfg.sample_neg_prompt
    return pipe


def build_pipeline_full(args, cfg, config_path, device, prompt):
    import torch
    import wan

    first = args.lora[0] if args.lora else (None, None)
    pipe = wan.SCAIL2Pipeline(
        config=cfg, checkpoint_dir=args.ckpt_dir, scail_safetensors_path=args.scail_path,
        scail_config_path=config_path, device_id=device, rank=0, t5_fsdp=False, dit_fsdp=False,
        use_usp=False, t5_cpu=args.t5_cpu, lora_path=first[0],
        lora_alpha=float(first[1]) if first[1] is not None else None)
    with torch.no_grad():
        for path, alpha in args.lora[1:]:
            pipe.fuse_lora(path, float(alpha))
    reduce_activation_memory(pipe.model)
    return pipe


# ----------------------------------------------------------------------------- main
def main():
    import faulthandler
    faulthandler.enable()  # print the Python stack if a native library crashes
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[scail2] %(asctime)s %(message)s", stream=sys.stdout)
    for name in ("image", "mask_image", "pose", "mask_video", "prompt_file", "out", "ckpt_dir", "scail_path"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    if args.text_cache:
        args.text_cache = os.path.abspath(args.text_cache)
    args.lora = [(os.path.abspath(p), a) for p, a in args.lora]
    args.additional_image = [os.path.abspath(p) for p in args.additional_image]
    args.additional_mask = [os.path.abspath(p) for p in args.additional_mask]
    if len(args.additional_image) != len(args.additional_mask):
        raise SystemExit("--additional_image and --additional_mask need the same number of files")

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)  # the SCAIL configs use paths relative to the repository

    # torch must be loaded before decord: on Windows decord's bundled DLLs make c10.dll fail to initialise.
    import torch
    install_decord_shim()
    from einops import rearrange
    from PIL import Image

    from wan.configs import SCAIL_CONFIG_PATHS, SCAIL_CONFIGS
    from wan.utils.scail_utils import (load_image_to_tensor_chw_normalized,
                                       load_video_for_pose_sample,
                                       resize_for_rectangle_crop)
    from wan.utils.utils import cache_video
    import wan  # noqa: F401  (imports every model module before patching attention)

    patch_attention(args.attention)
    device = 0
    torch.cuda.set_device(device)
    model = args.model.upper()
    cfg = SCAIL_CONFIGS[model]
    config_path = SCAIL_CONFIG_PATHS[model]
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read().strip()
    logging.info("Prompt: %s", prompt)
    build = build_pipeline_fp8 if args.weights == "fp8" else build_pipeline_full
    pipeline = build(args, cfg, config_path, device, prompt)

    target = (args.target_h, args.target_w)

    def load_image(path):
        tensor = load_image_to_tensor_chw_normalized(Image.open(path).convert("RGB")).to(device)
        return resize_for_rectangle_crop(tensor, target, reshape_mode="center").squeeze(0)  # c h w, -1..1

    def load_video(path):
        video = load_video_for_pose_sample(path).permute(0, 3, 1, 2)  # t c h w
        video = resize_for_rectangle_crop(video, target, reshape_mode="center")
        return (video - 127.5) / 127.5

    img = load_image(args.image)
    mask_img = load_image(args.mask_image)
    pose_video = load_video(args.pose)
    driving_mask_video = rearrange(load_video(args.mask_video), "t c h w -> c t h w")
    extra = {}
    if args.additional_image:
        extra["additional_ref_imgs"] = [load_image(p) for p in args.additional_image]
        extra["additional_ref_mask_imgs"] = [load_image(p) for p in args.additional_mask]

    # With fp8 weights the DiT stays on the GPU; moving it back and forth would only cost time.
    offload = args.offload_model == "true" and args.weights == "full"
    video = pipeline.generate(
        prompt,
        img,
        ref_mask_img=mask_img,
        pose_video=pose_video,
        driving_mask_video=driving_mask_video,
        replace_flag=True,
        shift=args.shift,
        sample_solver=args.solver,
        segment_len=args.segment_len,
        segment_overlap=args.segment_overlap,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=offload,
        **extra,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cache_video(tensor=video[None], save_file=args.out, fps=cfg.sample_fps,
                nrow=1, normalize=True, value_range=(-1, 1))
    logging.info("Saved %s (%d frames), peak GPU %.1f GB", args.out, video.shape[1],
                 torch.cuda.max_memory_allocated(device) / 2**30)


if __name__ == "__main__":
    main()
