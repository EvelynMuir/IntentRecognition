#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# python src/train.py experiment=intentonomy_clip_vit_layer_cls_patch_mean \
# logger=tensorboard \
# logger.tensorboard.name="Intentonomy-CLIP-ViT-Layer24-ClsPatchMean" \
# model.net.clip_model_name="ViT-L/14" \
# model.layer_idx=24 \
# model.net.layer_idx=24 \
# data.batch_size=64 \
# model.optimizer.lr=1e-4

# Usage:
#   bash scripts/intentonomy_clip_vit_layer_cls_patch_mean.sh [CKPT_PATH] [LAYER_IDX] [CLIP_MODEL] [BATCH_SIZE]
#
# Example:
#   bash scripts/intentonomy_clip_vit_layer_cls_patch_mean.sh \
#     logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt 24 ViT-L/14 64
#
# Notes:
# - src/eval.py runs both validation and test.
# - Do NOT pass `experiment=...` to eval; eval config doesn't include that defaults group.

CKPT_PATH="${1:-logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt}"
LAYER_IDX="${2:-24}"
CLIP_MODEL="${3:-ViT-L/14}"
BATCH_SIZE="${4:-64}"

python src/eval.py \
  model=intentonomy_clip_vit_layer_cls_patch_mean \
  data=intentonomy \
  trainer=default \
  logger=tensorboard \
  logger.tensorboard.name="Intentonomy-CLIP-ViT-Layer${LAYER_IDX}-ClsPatchMean-Eval" \
  model.net.clip_model_name="${CLIP_MODEL}" \
  model.layer_idx="${LAYER_IDX}" \
  model.net.layer_idx="${LAYER_IDX}" \
  data.batch_size="${BATCH_SIZE}" \
  ckpt_path="${CKPT_PATH}"