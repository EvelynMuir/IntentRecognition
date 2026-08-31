#!/usr/bin/env bash
set -euo pipefail

ROOT="${COCOER_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
cd "${ROOT}"
mkdir -p logs/slurm

FEATURE_JOB="$(sbatch --parsable scripts/cocoer_emotic_features.slurm)"
RUN_JOB="$(sbatch --parsable --dependency="afterok:${FEATURE_JOB}" scripts/cocoer_emotic_5x5.slurm)"
SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:${RUN_JOB}" scripts/cocoer_emotic_summary.slurm)"

echo "CocoER feature job: ${FEATURE_JOB}"
echo "CocoER 5x5 array (2 tasks): ${RUN_JOB}"
echo "CocoER summary job: ${SUMMARY_JOB}"
echo "Monitor: squeue -j ${FEATURE_JOB},${RUN_JOB},${SUMMARY_JOB}"

