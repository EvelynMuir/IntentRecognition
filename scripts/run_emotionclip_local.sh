#!/usr/bin/env bash
set -eo pipefail

ROOT="${EMOTIONCLIP_ROOT:-/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra}"
ENV_NAME="${EMOTIONCLIP_ENV:-emotionclip-local}"
FEATURE_DIR="${EMOTIONCLIP_FEATURES:-${ROOT}/logs/analysis/emotionclip_matched_vitl14_features}"
OUTPUT_DIR="${EMOTIONCLIP_OUTPUT:-${ROOT}/logs/analysis/emotionclip_matched_vitl14_local}"

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
set -u
cd "${ROOT}"

if [[ ! -f "${FEATURE_DIR}/emotionclip_features.npy" ]]; then
  python -u scripts/prepare_emotic_matched_features.py \
    --method emotionclip \
    --annotation-file "${ROOT}/../Emotic/Annotations/Annotations.mat" \
    --image-root "${ROOT}/../Emotic/emotic" \
    --description-file "${ROOT}/../Emotic/emotion_description_gemini.json" \
    --fdil-cache "${ROOT}/logs/analysis/emotic_clip_dual_cache_full_20260323/_cache" \
    --output-dir "${FEATURE_DIR}" \
    --batch-size "${EMOTIONCLIP_BATCH_SIZE:-32}" \
    --workers "${EMOTIONCLIP_WORKERS:-8}"
fi

python -u scripts/run_emotic_matched_baseline_fold.py \
  --method emotionclip \
  --feature-dir "${FEATURE_DIR}" \
  --output-dir "${OUTPUT_DIR}"

python -u scripts/summarize_emotic_matched_baselines.py \
  --output-dir "${OUTPUT_DIR}" \
  --methods emotionclip \
  --require-complete
