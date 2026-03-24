#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?Usage: $0 <seed> [logger_name]}"
LOGGER_NAME="${2:-Intentonomy-CLIP-ViT-Layer24-ClsPatchMean-Seed${SEED}}"

python src/train.py \
  experiment=intentonomy_clip_vit_layer_cls_patch_mean \
  logger=tensorboard \
  logger.tensorboard.name="${LOGGER_NAME}" \
  model.net.clip_model_name="ViT-L/14" \
  model.layer_idx=24 \
  model.net.layer_idx=24 \
  data.batch_size=64 \
  model.optimizer.lr=1e-4 \
  seed="${SEED}"
