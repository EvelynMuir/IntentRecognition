#!/usr/bin/env bash
#SBATCH --job-name=emotic-match-feat
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/emotic_match_feat_%j.out
#SBATCH --error=logs/slurm/emotic_match_feat_%j.err

set -eo pipefail

PROJECT_ROOT="${EMOTIC_MATCHED_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
CONDA_ENV="${EMOTIC_MATCHED_ENV:-emotic-matched}"
source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

python -u scripts/prepare_emotic_matched_features.py \
  --annotation-file "${EMOTIC_ANNOTATION:-${PROJECT_ROOT}/../Emotic/Annotations/Annotations.mat}" \
  --image-root "${EMOTIC_IMAGE_ROOT:-${PROJECT_ROOT}/../Emotic/emotic}" \
  --description-file "${EMOTIC_DESCRIPTION:-${PROJECT_ROOT}/../Emotic/emotion_description_gemini.json}" \
  --fdil-cache "${EMOTIC_FDIL_CACHE:-${PROJECT_ROOT}/logs/analysis/emotic_clip_dual_cache_full_20260323/_cache}" \
  --output-dir "${EMOTIC_MATCHED_FEATURES:-${PROJECT_ROOT}/logs/analysis/emotic_matched_vitl14_features}" \
  --batch-size "${EMOTIC_FEATURE_BATCH_SIZE:-16}" \
  --workers "${SLURM_CPUS_PER_TASK}"
