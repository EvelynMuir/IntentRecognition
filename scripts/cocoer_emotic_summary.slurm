#!/usr/bin/env bash
#SBATCH --job-name=cocoer-emotic-sum
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/cocoer_emotic_summary_%j.out
#SBATCH --error=logs/slurm/cocoer_emotic_summary_%j.err

set -eo pipefail
PROJECT_ROOT="${COCOER_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
CONDA_ENV="${COCOER_ENV:-emotic-matched}"
source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_ROOT}"

python -u scripts/summarize_emotic_matched_baselines.py \
  --output-dir "${COCOER_OUTPUT:-${PROJECT_ROOT}/logs/analysis/cocoer_matched_vitl14_cluster}" \
  --methods cocoer \
  --require-complete

