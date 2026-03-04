#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Training (disabled by default):
# python src/train.py experiment=intentonomy_clip_vit_layer_cls \
#   model.layer_idx=24 \
#   model.net.layer_idx=24 \
#   logger=tensorboard \
#   logger.tensorboard.name="Intentonomy-CLIP-ViT-cls" \
#   data.batch_size=64 \
#   model.optimizer.lr=1e-4

# Evaluation (validation + test)
# Usage:
#   bash scripts/intentonomy_clip_vit_cls.sh [CKPT_PATH] [LAYER_IDX] [CLIP_MODEL] [BATCH_SIZE]
#
# Example:
#   bash scripts/intentonomy_clip_vit_cls.sh \
#     logs/train/runs/2026-03-03_16-34-30/checkpoints/epoch_017.ckpt 24 ViT-L/14 64
#
# Note:
# - src/eval.py runs both validation and test.
CKPT_PATH="${1:-logs/train/runs/2026-03-03_09-56-07/checkpoints/epoch_015.ckpt}"
LAYER_IDX="${2:-24}"
CLIP_MODEL="${3:-ViT-L/14}"
BATCH_SIZE="${4:-64}"

python src/eval.py \
  model=intentonomy_clip_vit_layer_cls \
  data=intentonomy \
  trainer=default \
  logger=tensorboard \
  logger.tensorboard.name="Intentonomy-CLIP-ViT-cls-Layer${LAYER_IDX}-Eval" \
  model.layer_idx="${LAYER_IDX}" \
  model.net.layer_idx="${LAYER_IDX}" \
  model.net.clip_model_name="${CLIP_MODEL}" \
  data.batch_size="${BATCH_SIZE}" \
  ckpt_path="${CKPT_PATH}"
