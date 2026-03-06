#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Usage:
#   bash scripts/intentonomy_clip_vit_icrn.sh [CONCEPTS_FILE] [INTENT2CONCEPTS_FILE] [CLIP_MODEL] [BATCH_SIZE] [LR_CONCEPT] [CLS_MEAN_PATCH_CKPT]
#
# Example:
#   bash scripts/intentonomy_clip_vit_icrn.sh \
#     /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/concepts.json \
#     /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent2concepts.json \
#     ViT-L/14 64 5e-4 \
#     /home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/train/runs/2026-03-05_12-15-15/checkpoints/epoch_008.ckpt

CONCEPTS_FILE="${1:-/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/concepts.json}"
INTENT2CONCEPTS_FILE="${2:-/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent2concepts.json}"
CLIP_MODEL="${3:-ViT-L/14}"
BATCH_SIZE="${4:-64}"
# Concept branch LR (TODO target): 5e-4
LR_CONCEPT="${5:-5e-4}"
CLS_MEAN_PATCH_CKPT="${6:-/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/train/runs/2026-03-05_12-15-15/checkpoints/epoch_008.ckpt}"
# Stable defaults from TODO-2 iteration:
#   base_head lr = 0.0 (freeze baseline head)
#   alpha lr = 1e-4

python src/train.py \
  experiment=intentonomy_clip_vit_icrn \
  logger=tensorboard \
  logger.tensorboard.name="Intentonomy-CLIP-ViT-ICRN" \
  model.net.clip_model_name="${CLIP_MODEL}" \
  model.net.cls_mean_patch_ckpt_path="${CLS_MEAN_PATCH_CKPT}" \
  model.net.alpha_init=0.0 \
  model.net.init_with_llm_prior=true \
  model.use_prior_regularization=false \
  model.lambda_prior=0.0 \
  model.lr_base_head=0.0 \
  model.lr_concept_branch="${LR_CONCEPT}" \
  model.lr_alpha=1e-4 \
  model.net.concepts_file="${CONCEPTS_FILE}" \
  model.net.intent2concepts_file="${INTENT2CONCEPTS_FILE}" \
  data.batch_size="${BATCH_SIZE}" \
  model.optimizer.lr="${LR_CONCEPT}"
