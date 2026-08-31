#!/usr/bin/env bash
#SBATCH --job-name=e1b-vitl14
#SBATCH --partition=HGX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/e1b_vitl14_%A_%a.out
#SBATCH --error=logs/slurm/e1b_vitl14_%A_%a.err

set -euo pipefail

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh

ROOT="${INTENT_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition}"
export INTENT_ROOT="$ROOT"
export PYTHON="${PYTHON:-$ROOT/.conda/bin/python}"
mkdir -p "$ROOT/lightning-hydra/logs/slurm"

METHOD_GROUP="${METHOD_GROUP:-hleg_labcr}"
if [[ "$METHOD_GROUP" == "hleg_labcr" ]]; then
  METHODS=(hleg labcr)
elif [[ "$METHOD_GROUP" == "pip_intclip" ]]; then
  METHODS=(pip intclip)
else
  echo "Unknown METHOD_GROUP=$METHOD_GROUP; expected hleg_labcr or pip_intclip" >&2
  exit 2
fi

METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
bash "$ROOT/lightning-hydra/scripts/e1b_retrain_clip_vitl14_original.sh" "$METHOD"
