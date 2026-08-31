#!/usr/bin/env bash
#SBATCH --job-name=emotic-match-run
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/emotic_match_%A_%a.out
#SBATCH --error=logs/slurm/emotic_match_%A_%a.err

set -eo pipefail

PROJECT_ROOT="${EMOTIC_MATCHED_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
CONDA_ENV="${EMOTIC_MATCHED_ENV:-emotic-matched}"
METHODS=(cocoer emotionclip)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

python -u scripts/run_emotic_matched_baseline_fold.py \
  --method "${METHOD}" \
  --feature-dir "${EMOTIC_MATCHED_FEATURES:-${PROJECT_ROOT}/logs/analysis/emotic_matched_vitl14_features}" \
  --output-dir "${EMOTIC_MATCHED_OUTPUT:-${PROJECT_ROOT}/logs/analysis/emotic_matched_baselines_vitl14}" \
  --batch-size "${EMOTIC_COCOER_BATCH_SIZE:-64}" \
  --workers "${SLURM_CPUS_PER_TASK}"
