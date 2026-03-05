#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Usage:
#   bash scripts/intentonomy_clip_vit_icrn.sh [INTENT_CONCEPT_FILE] [CLIP_MODEL] [BATCH_SIZE] [LR]
#
# Example:
#   bash scripts/intentonomy_clip_vit_icrn.sh \
#     /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_concepts_gemini.json \
#     ViT-L/14 64 2e-4

INTENT_CONCEPT_FILE="${1:-/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_concepts_gemini.json}"
CLIP_MODEL="${2:-ViT-L/14}"
BATCH_SIZE="${3:-64}"
LR="${4:-2e-4}"

python src/train.py \
  experiment=intentonomy_clip_vit_icrn \
  logger=tensorboard \
  logger.tensorboard.name="Intentonomy-CLIP-ViT-ICRN" \
  model.net.clip_model_name="${CLIP_MODEL}" \
  model.net.init_with_llm_prior=true \
  model.net.intent_concepts_gemini_file="${INTENT_CONCEPT_FILE}" \
  data.batch_size="${BATCH_SIZE}" \
  model.optimizer.lr="${LR}"
