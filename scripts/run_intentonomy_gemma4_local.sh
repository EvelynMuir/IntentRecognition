#!/usr/bin/env bash
# Run Gemma 4 E4B IT Intentonomy zero-shot evaluation on the local RTX 4090.

set -euo pipefail

PROJECT_ROOT="/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra"
GEMMA_ENV="/home/evelynmuir/softwares/miniconda3/envs/gemma"
MODEL_SNAPSHOT="/home/evelynmuir/lambda/hf-models/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
LOCAL_MODEL="/tmp/gemma4-e4b-it"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/vllm_zeroshot/gemma4-e4b-it"
PORT="${INTENTONOMY_GEMMA_PORT:-29581}"

if [[ ! -f "${LOCAL_MODEL}/config.json" || ! -f "${LOCAL_MODEL}/model.safetensors" ]]; then
  mkdir -p "${LOCAL_MODEL}"
  cp -aL "${MODEL_SNAPSHOT}/." "${LOCAL_MODEL}/"
fi

VLLM_USE_V2_MODEL_RUNNER=0 "${GEMMA_ENV}/bin/vllm" serve "${LOCAL_MODEL}" \
  --served-model-name gemma4-e4b-it \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.76 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}"
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Gemma vLLM server exited before becoming ready." >&2
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "Timed out waiting for the local Gemma vLLM server." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
python -u scripts/run_intentonomy_vllm_zeroshot.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model gemma4-e4b-it \
  --output-dir "${OUTPUT_DIR}" \
  --max-tokens 768 \
  --workers 4 \
  --request-timeout 180 \
  --retries 3 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --disable-thinking \
  --json-response \
  --require-all-classes
