#!/usr/bin/env bash
#SBATCH --job-name=cocoer-emotic-5x5
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/cocoer_emotic_%A_%a.out
#SBATCH --error=logs/slurm/cocoer_emotic_%A_%a.err

set -eo pipefail

PROJECT_ROOT="${COCOER_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
CONDA_ENV="${COCOER_ENV:-emotic-matched}"
FEATURE_DIR="${COCOER_FEATURES:-${PROJECT_ROOT}/logs/analysis/cocoer_matched_vitl14_features}"
OUTPUT_DIR="${COCOER_OUTPUT:-${PROJECT_ROOT}/logs/analysis/cocoer_matched_vitl14_cluster}"

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

# Cluster policy permits at most two array elements. Split the five seeds
# deterministically across the two elements while each seed still runs 5 folds.
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
  SEED_INDICES=(0 2 4)
else
  SEED_INDICES=(1 3)
fi

for SEED_INDEX in "${SEED_INDICES[@]}"; do
  python -u scripts/run_emotic_matched_baseline_fold.py \
    --method cocoer \
    --feature-dir "${FEATURE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed-index "${SEED_INDEX}" \
    --batch-size "${COCOER_BATCH_SIZE:-64}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --epochs "${COCOER_EPOCHS:-20}" \
    --patience "${COCOER_PATIENCE:-5}"
done

