#!/usr/bin/env bash
# Downloads every model the pipeline needs (about 55 GB), sized for a 32 GB RAM / 32 GB VRAM machine:
# SCAIL-2 as Comfy-Org's fp8_scaled DiT (same tensor names as the official convert.py output),
# the official VAE/T5/CLIP files, lightx2v, SAM3 and a pre-quantized NF4 Qwen-Image-Edit-2511.
# Usage: scripts/download_models.sh [MODELS_DIR]   (default D:/models)
set -euo pipefail
HF="${HF:-hf}"
MODELS="${1:-D:/models}"
mkdir -p "$MODELS"

"$HF" download zai-org/SCAIL-2 Wan2.1_VAE.pth \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth \
  umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth umt5-xxl/special_tokens_map.json \
  umt5-xxl/spiece.model umt5-xxl/tokenizer.json umt5-xxl/tokenizer_config.json \
  --local-dir "$MODELS/SCAIL-2"
# The CLIP tokenizer the Wan code expects next to the checkpoints.
"$HF" download FacebookAI/xlm-roberta-large config.json sentencepiece.bpe.model tokenizer.json tokenizer_config.json \
  --local-dir "$MODELS/SCAIL-2/xlm-roberta-large"
"$HF" download Comfy-Org/SCAIL-2 diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors \
  loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors --local-dir "$MODELS/SCAIL-2-comfy"
"$HF" download Kijai/WanVideo_comfy Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors \
  --local-dir "$MODELS/lightx2v"
# Into the Hugging Face cache, loaded by id. SAM3 is gated: accept its licence on the website first.
"$HF" download facebook/sam3 --exclude sam3.pt
"$HF" download ovedrive/Qwen-Image-Edit-2511-4bit
echo "All models downloaded to $MODELS and the Hugging Face cache."
