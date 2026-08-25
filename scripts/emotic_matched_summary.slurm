#!/usr/bin/env bash
#SBATCH --job-name=emotic-match-sum
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/emotic_match_summary_%j.out
#SBATCH --error=logs/slurm/emotic_match_summary_%j.err

set -eo pipefail
PROJECT_ROOT="${EMOTIC_MATCHED_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
CONDA_ENV="${EMOTIC_MATCHED_ENV:-emotic-matched}"
source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_ROOT}"
python -u scripts/summarize_emotic_matched_baselines.py \
  --output-dir "${EMOTIC_MATCHED_OUTPUT:-${PROJECT_ROOT}/logs/analysis/emotic_matched_baselines_vitl14}" \
  --require-complete
