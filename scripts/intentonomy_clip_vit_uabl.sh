#!/bin/bash

set -euo pipefail

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Usage:
#   bash scripts/intentonomy_clip_vit_uabl.sh \
#     [EXP_NAME] [DEVICES] [BATCH_SIZE] [CLIP_MODEL] [LAYER_IDX] [LR] \
#     [UNCERTAINTY_LR] [ADAPTATION_LR] [UNCERTAINTY_W] [IDENTITY_W]
#
# Example:
#   bash scripts/intentonomy_clip_vit_uabl.sh \
#     Intentonomy-CLIP-ViT-L14-L24-UABL 1 64 ViT-L/14 24 5e-3 1e-3 1e-3 0.1 1e-3

EXP_NAME="${1:-Intentonomy-CLIP-ViT-L14-L24-UABL}"
DEVICES="${2:-1}"
BATCH_SIZE="${3:-64}"
CLIP_MODEL="${4:-ViT-L/14}"
LAYER_IDX="${5:-24}"
LR="${6:-5e-3}"
UNCERTAINTY_LR="${7:-1e-3}"
ADAPTATION_LR="${8:-1e-3}"
UNCERTAINTY_W="${9:-0.1}"
IDENTITY_W="${10:-1e-3}"

# Keep the training recipe close to the current strongest baseline:
# - frozen CLIP backbone
# - CLS + patch mean head
# - ASL
# - ViT-L/14, layer 24
# UABL only adds:
# - uncertainty prediction from feature + logits
# - uncertainty-aware boundary adaptation z' = a(u) z + b(u)
# - identity regularization to keep corrections minimal

python src/train.py \
  experiment=intentonomy_clip_vit_uabl \
  trainer=gpu \
  trainer.devices="${DEVICES}" \
  logger=tensorboard \
  logger.tensorboard.name="${EXP_NAME}" \
  data.batch_size="${BATCH_SIZE}" \
  data.binarize_softprob=true \
  model.net.clip_model_name="${CLIP_MODEL}" \
  model.layer_idx="${LAYER_IDX}" \
  model.net.layer_idx="${LAYER_IDX}" \
  model.optimizer.lr="${LR}" \
  model.uncertainty_lr="${UNCERTAINTY_LR}" \
  model.adaptation_lr="${ADAPTATION_LR}" \
  model.uncertainty_loss_weight="${UNCERTAINTY_W}" \
  model.identity_regularization_weight="${IDENTITY_W}" \
  model.freeze_backbone=true \
  model.use_ema=false \
  model.uncertainty_target_mode="positive_inverse" \
  model.detach_logits_for_uncertainty=true \
  model.uncertainty_hidden_dim=512 \
  model.uncertainty_dropout=0.1 \
  model.adaptation_hidden_dim=16 \
  model.adaptation_dropout=0.0 \
  model.adaptation_scale_limit=1.0 \
  model.adaptation_bias_limit=1.0
