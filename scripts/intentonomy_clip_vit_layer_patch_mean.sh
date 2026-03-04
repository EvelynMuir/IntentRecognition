#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Training (disabled by default):
# python src/train.py experiment=intentonomy_clip_vit_layer_patch_mean \
#   logger=tensorboard \
#   logger.tensorboard.name="Intentonomy-CLIP-ViT-Layer24-PatchMean" \
#   model.net.clip_model_name="ViT-L/14" \
#   model.layer_idx=24 \
#   model.net.layer_idx=24 \
#   data.batch_size=64 \
#   model.optimizer.lr=1e-4

# Evaluation (validation + test)
# Usage:
#   bash scripts/intentonomy_clip_vit_layer_patch_mean.sh [CKPT_PATH] [LAYER_IDX] [CLIP_MODEL] [BATCH_SIZE]
#
# Example:
#   bash scripts/intentonomy_clip_vit_layer_patch_mean.sh \
#     logs/train/runs/2026-02-26_14-28-27/checkpoints/last.ckpt 24 ViT-L/14 64
#
# Note:
# - src/eval.py runs both validation and test.
CKPT_PATH="${1:-logs/train/runs/2026-02-26_16-26-04/checkpoints/epoch_015.ckpt}"
LAYER_IDX="${2:-24}"
CLIP_MODEL="${3:-ViT-L/14}"
BATCH_SIZE="${4:-64}"

python src/eval.py \
  model=intentonomy_clip_vit_layer_patch_mean \
  data=intentonomy \
  trainer=default \
  logger=tensorboard \
  logger.tensorboard.name="Intentonomy-CLIP-ViT-Layer${LAYER_IDX}-PatchMean-Eval" \
  model.layer_idx="${LAYER_IDX}" \
  model.net.layer_idx="${LAYER_IDX}" \
  model.net.clip_model_name="${CLIP_MODEL}" \
  data.batch_size="${BATCH_SIZE}" \
  ckpt_path="${CKPT_PATH}"
