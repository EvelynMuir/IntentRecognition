#!/usr/bin/env bash
set -euo pipefail

ROOT="${EMOTIC_MATCHED_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
cd "${ROOT}"
mkdir -p logs/slurm

FEATURE_JOB="$(sbatch --parsable scripts/emotic_matched_features.slurm)"
RUN_JOB="$(sbatch --parsable --dependency="afterok:${FEATURE_JOB}" scripts/emotic_matched_baselines.slurm)"
SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:${RUN_JOB}" scripts/emotic_matched_summary.slurm)"

echo "Feature job: ${FEATURE_JOB}"
echo "Baseline array (2 tasks): ${RUN_JOB}"
echo "Summary job: ${SUMMARY_JOB}"
echo "Monitor: squeue -j ${FEATURE_JOB},${RUN_JOB},${SUMMARY_JOB}"

