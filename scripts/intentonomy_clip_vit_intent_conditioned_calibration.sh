#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# Usage:
#   bash scripts/intentonomy_clip_vit_intent_conditioned_calibration.sh \
#     [GEMINI_FILE] [CLIP_MODEL] [BATCH_SIZE] [LR] [LAMBDA_INIT] [USE_VECTOR_LAMBDA] [USE_LOGIT_CORRELATION]
#
# Example:
#   bash scripts/intentonomy_clip_vit_intent_conditioned_calibration.sh \
#     /home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_description_gemini.json \
#     ViT-L/14 64 1e-4 0.1 true true

GEMINI_FILE="${1:-/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_description_gemini.json}"
CLIP_MODEL="${2:-ViT-L/14}"
BATCH_SIZE="${3:-64}"
LR="${4:-1e-4}"
LAMBDA_INIT="${5:-0.1}"
USE_VECTOR_LAMBDA="${6:-false}"
USE_LOGIT_CORRELATION="${7:-false}"

# python src/train.py experiment=intentonomy_clip_vit_intent_conditioned_calibration logger=tensorboard \
# logger.tensorboard.name="Intentonomy-CLIP-ViT-IntentConditionedCalibration-LLM-${USE_VECTOR_LAMBDA}-${USE_LOGIT_CORRELATION}" \
# model.net.clip_model_name="${CLIP_MODEL}" \
# model.net.intent_description_mode="llm" \
# model.net.intent_gemini_file="${GEMINI_FILE}" \
# model.net.lambda_init="${LAMBDA_INIT}" \
# model.net.use_vector_lambda="${USE_VECTOR_LAMBDA}" \
# model.net.use_logit_correlation="${USE_LOGIT_CORRELATION}" \
# data.batch_size="${BATCH_SIZE}" \
# model.optimizer.lr="${LR}"

python src/eval.py \
model=intentonomy_clip_vit_intent_conditioned_calibration \
data=intentonomy \
trainer=default \
logger=tensorboard  \
logger.tensorboard.name="Intentonomy-CLIP-ViT-IntentConditionedCalibration-LLM-${USE_VECTOR_LAMBDA}-${USE_LOGIT_CORRELATION}" \
model.net.clip_model_name="${CLIP_MODEL}" \
model.net.intent_description_mode="llm" \
model.net.intent_gemini_file="${GEMINI_FILE}" \
model.net.lambda_init="${LAMBDA_INIT}" \
model.net.use_vector_lambda="${USE_VECTOR_LAMBDA}" \
model.net.use_logit_correlation="${USE_LOGIT_CORRELATION}" \
data.batch_size="${BATCH_SIZE}" \
model.optimizer.lr="${LR}" \
ckpt_path="logs/train/runs/2026-03-04_17-16-33/checkpoints/epoch_025.ckpt"