#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-all}"
ROOT="${INTENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/.conda/bin/python}"
HLEG_BATCH_SIZE="${HLEG_BATCH_SIZE:-4}"
LABCR_BATCH_SIZE="${LABCR_BATCH_SIZE:-4}"
LABCR_LR="${LABCR_LR:-2e-5}"
LABCR_CLIP_GRAD_NORM="${LABCR_CLIP_GRAD_NORM:-1.0}"
PIP_BATCH_SIZE="${PIP_BATCH_SIZE:-16}"
INTCLIP_BATCH_SIZE="${INTCLIP_BATCH_SIZE:-16}"

append_optional_common_args() {
  local -n args_ref="$1"
  if [[ -n "${WORKERS:-}" ]]; then
    args_ref+=(--workers "$WORKERS")
  fi
  if [[ -n "${SEED:-}" ]]; then
    args_ref+=(--seed "$SEED")
  fi
}

run_hleg() {
  cd "$ROOT/HLEG"
  args=(
    --backbone clip_vit_l14 \
    --local-rank "${LOCAL_RANK:-0}" \
    --output "output/e1b_hleg_clip_vitl14"
  )
  if [[ -n "${HLEG_EPOCHS:-}" ]]; then
    args+=(--epochs "$HLEG_EPOCHS")
  fi
  args+=(--batch-size "$HLEG_BATCH_SIZE")
  append_optional_common_args args
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 train.py "${args[@]}"
}

run_labcr() {
  cd "$ROOT/LabCR"
  args=(
    --backbone clip_vit_l14 \
    --local_rank "${LOCAL_RANK:-0}" \
    --lr "$LABCR_LR" \
    --clip-grad-norm "$LABCR_CLIP_GRAD_NORM" \
    --no-amp \
    --output "output/e1b_labcr_clip_vitl14"
  )
  if [[ -n "${LABCR_EPOCHS:-}" ]]; then
    args+=(--epochs "$LABCR_EPOCHS")
  fi
  args+=(--batch-size "$LABCR_BATCH_SIZE")
  append_optional_common_args args
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 train.py "${args[@]}"
}

run_pip() {
  cd "$ROOT/PIP-Net"
  args=(--backbone clip_vit_l14)
  args+=(--batch_size "$PIP_BATCH_SIZE")
  PYTHONPATH="$ROOT/PIP-Net:${PYTHONPATH:-}" "$PYTHON" tools/main.py "${args[@]}"
}

run_intclip() {
  cd "$ROOT/IntCLIP"
  args=(
    --config_file configs/models/vit_l14_ep50.yaml \
    --dataset_config_file configs/datasets/intentonomy.yaml \
    --datadir "$ROOT/Intentonomy/data" \
    --output_dir "outputs/e1b_intclip_vitl14"
  )
  args+=(--train_batch_size "$INTCLIP_BATCH_SIZE")
  if [[ -n "${INTCLIP_EPOCHS:-}" ]]; then
    args+=(--max_epochs "$INTCLIP_EPOCHS" --stop_epochs "$INTCLIP_EPOCHS")
  fi
  if [[ -n "${INTCLIP_VAL_EVERY:-}" ]]; then
    args+=(--val_every_n_epochs "$INTCLIP_VAL_EVERY")
  fi
  "$PYTHON" train.py "${args[@]}"
}

case "$METHOD" in
  hleg) run_hleg ;;
  labcr) run_labcr ;;
  pip) run_pip ;;
  intclip) run_intclip ;;
  all)
    run_hleg
    run_labcr
    run_pip
    run_intclip
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    echo "Usage: $0 {all|hleg|labcr|pip|intclip}" >&2
    exit 2
    ;;
esac
