#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_RUN_DIR="${CACHE_RUN_DIR:-logs/analysis/emotic_clip_dual_cache_full_20260323}"
CACHE_DIR="${CACHE_RUN_DIR}/_cache"
BASELINE_OUT="${BASELINE_OUT:-logs/analysis/emotic_clip_baseline_full_20260323}"
VLM_OUT="${VLM_OUT:-logs/analysis/emotic_vlm_20260323}"
DISTILL_OUT="${DISTILL_OUT:-logs/analysis/emotic_privileged_distillation_20260323}"
SLRC_OUT="${SLRC_OUT:-logs/analysis/emotic_distillation_slrc_20260323}"

ANNOTATION_FILE="${ANNOTATION_FILE:-../Emotic/Annotations/Annotations.mat}"
IMAGE_ROOT="${IMAGE_ROOT:-../Emotic/emotic}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-../Emotic/emotion_description_gemini.json}"

mkdir -p "$VLM_OUT" "$DISTILL_OUT" "$SLRC_OUT"

if [[ ! -f "${CACHE_DIR}/train_clip.npz" || ! -f "${CACHE_DIR}/val_clip.npz" || ! -f "${CACHE_DIR}/test_clip.npz" ]]; then
  python scripts/build_clip_distill_cache.py \
    --data emotic \
    --output-dir "$CACHE_RUN_DIR" \
    --batch-size 128 \
    --num-workers 0 \
    --device cuda
fi

python scripts/run_clip_feature_baseline.py \
  --reuse-cache-dir "$CACHE_DIR" \
  --output-dir "$BASELINE_OUT" \
  --device cuda \
  --batch-size 256 \
  --max-epochs 30 \
  --patience 6

python scripts/generate_emotic_vlm_rationales.py \
  --split train \
  --output-jsonl "${VLM_OUT}/rationale_full.jsonl" \
  --request-batch-size 32 \
  --tensor-parallel-size 1

python scripts/generate_emotic_vlm_rationales.py \
  --split val \
  --output-jsonl "${VLM_OUT}/val_rationale_baseline_pred.jsonl" \
  --request-batch-size 32 \
  --tensor-parallel-size 1

python scripts/generate_emotic_vlm_rationales.py \
  --split test \
  --output-jsonl "${VLM_OUT}/test_rationale_baseline_pred.jsonl" \
  --request-batch-size 32 \
  --tensor-parallel-size 1

python scripts/extract_vlm_rationale_features.py \
  --input-jsonl "${VLM_OUT}/rationale_full.jsonl" \
  --output-npz "${VLM_OUT}/rationale_full_bge_features.npz" \
  --text-encoder bge \
  --batch-size 64

python scripts/extract_vlm_rationale_features.py \
  --input-jsonl "${VLM_OUT}/val_rationale_baseline_pred.jsonl" \
  --output-npz "${VLM_OUT}/val_rationale_baseline_pred_bge_features.npz" \
  --text-encoder bge \
  --batch-size 64

python scripts/extract_vlm_rationale_features.py \
  --input-jsonl "${VLM_OUT}/test_rationale_baseline_pred.jsonl" \
  --output-npz "${VLM_OUT}/test_rationale_baseline_pred_bge_features.npz" \
  --text-encoder bge \
  --batch-size 64

python scripts/analyze_privileged_distillation.py \
  --reuse-cache-dir "$CACHE_DIR" \
  --train-text-npz "${VLM_OUT}/rationale_full_bge_features.npz" \
  --val-text-npz "${VLM_OUT}/val_rationale_baseline_pred_bge_features.npz" \
  --test-text-npz "${VLM_OUT}/test_rationale_baseline_pred_bge_features.npz" \
  --train-annotation-file "$ANNOTATION_FILE" \
  --image-dir "$IMAGE_ROOT" \
  --output-dir "$DISTILL_OUT" \
  --device cuda

python scripts/analyze_distillation_slrc.py \
  --reuse-cache-dir "$CACHE_DIR" \
  --slr-cache-dir "$CACHE_DIR" \
  --train-text-npz "${VLM_OUT}/rationale_full_bge_features.npz" \
  --val-text-npz "${VLM_OUT}/val_rationale_baseline_pred_bge_features.npz" \
  --test-text-npz "${VLM_OUT}/test_rationale_baseline_pred_bge_features.npz" \
  --teacher-run-dir "$DISTILL_OUT" \
  --annotation-file "$ANNOTATION_FILE" \
  --gemini-file "$DESCRIPTION_FILE" \
  --output-dir "$SLRC_OUT" \
  --device cuda
