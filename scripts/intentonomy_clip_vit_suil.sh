#!/bin/bash

set -euo pipefail

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Usage:
#   bash scripts/intentonomy_clip_vit_suil.sh \
#     [EXP_NAME] [DEVICES] [BATCH_SIZE] [CLIP_MODEL] [LAYER_IDX] [LR] [CAL_LR] [HIER_W]
#
# Example:
#   bash scripts/intentonomy_clip_vit_suil.sh \
#     Intentonomy-CLIP-ViT-L14-L24-SUIL 1 64 ViT-L/14 24 5e-3 1e-3 0.1

EXP_NAME="${1:-Intentonomy-CLIP-ViT-L14-L24-SUIL}"
DEVICES="${2:-1}"
BATCH_SIZE="${3:-64}"
CLIP_MODEL="${4:-ViT-L/14}"
LAYER_IDX="${5:-24}"
LR="${6:-5e-3}"
CAL_LR="${7:-1e-3}"
HIER_W="${8:-0.1}"

# Keep the main training recipe close to the current strongest baseline:
# - frozen CLIP backbone
# - CLS + patch mean head
# - ASL
# - ViT-L/14, layer 24
# SUIL only adds:
# - confidence-aware supervision
# - hierarchy regularization
# - class-wise calibration

python src/train.py \
  experiment=intentonomy_clip_vit_suil \
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
  model.calibration_lr="${CAL_LR}" \
  model.hierarchy_loss_weight="${HIER_W}" \
  model.freeze_backbone=true \
  model.use_ema=false \
  model.use_confidence_aware_supervision=true \
  model.force_binarize_targets=true \
  model.confidence_mapping="discrete" \
  model.confidence_weight_low=1.0 \
  model.confidence_weight_mid=1.15 \
  model.confidence_weight_high=1.3 \
  model.use_hierarchy_regularization=true \
  model.hierarchy_aggregation="noisy_or" \
  model.hierarchy_margin=0.0 \
  model.use_coarse_auxiliary_loss=false \
  model.hierarchy_coarse_loss_weight=0.0 \
  model.use_classwise_calibration=true \
  model.calibration_mode="bias" \
  model.calibration_regularization_weight=1e-4
