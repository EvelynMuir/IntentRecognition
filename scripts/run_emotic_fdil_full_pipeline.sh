#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${CACHE_DIR:-logs/analysis/emotic_clip_dual_cache_full_20260323/_cache}"
VLM_DIR="${VLM_DIR:-logs/analysis/emotic_vlm_20260323}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/analysis/emotic_fdil_pool_kfold_20260703}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-../Emotic/emotion_description_gemini.json}"
SEEDS="${SEEDS:-20260625,20260626,20260627,20260628,20260629}"
PYTHON_BIN="${PYTHON_BIN:-../.conda/bin/python}"

for required in \
  "${CACHE_DIR}/val_clip.npz" \
  "${CACHE_DIR}/test_clip.npz" \
  "${VLM_DIR}/val_rationale_baseline_pred_bge_features.npz" \
  "${VLM_DIR}/test_rationale_baseline_pred_bge_features.npz" \
  "${DESCRIPTION_FILE}"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required upstream artifact: $required" >&2
    echo "Run scripts/run_emotic_full_pipeline.sh first." >&2
    exit 1
  fi
done

"$PYTHON_BIN" -u scripts/analyze_distillation_pool_kfold.py \
  --cache-dir "$CACHE_DIR" \
  --vlm-dir "$VLM_DIR" \
  --emotion-description-file "$DESCRIPTION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --seeds "$SEEDS" \
  --folds 5 \
  --oof-folds 3 \
  --slr-topk 10 \
  --slr-alpha 0.3 \
  --device cuda
