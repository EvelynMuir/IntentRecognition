#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <seed1> [seed2 ...]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
summary_paths=()

for seed in "$@"; do
  echo "=== Training seed ${seed} ==="
  bash scripts/train_intentonomy_layer_cls_patch_mean_seed.sh "${seed}"

  run_dir="$(find logs/train/runs -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  ckpt_path="$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'epoch_*.ckpt' | sort | tail -1)"
  if [[ -z "${ckpt_path}" ]]; then
    ckpt_path="${run_dir}/checkpoints/last.ckpt"
  fi

  analysis_dir="logs/analysis/calibrated_seed_${seed}_${timestamp}"
  echo "=== Calibrated decision analysis for seed ${seed} ==="
  python scripts/analyze_calibrated_decision_rule.py \
    --run-dir "${run_dir}" \
    --ckpt-path "${ckpt_path}" \
    --output-dir "${analysis_dir}"

  summary_paths+=("${analysis_dir}/summary.json")
done

summary_arg="$(IFS=,; echo "${summary_paths[*]}")"
aggregate_path="logs/analysis/multiseed_calibrated_stability_${timestamp}.json"

echo "=== Aggregating multi-seed stability ==="
python scripts/aggregate_multirun_stability.py \
  --summary-jsons "${summary_arg}" \
  --output-json "${aggregate_path}"

echo "Saved aggregate summary to ${aggregate_path}"
