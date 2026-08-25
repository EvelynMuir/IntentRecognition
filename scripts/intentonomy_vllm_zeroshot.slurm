#!/usr/bin/env bash
#SBATCH --job-name=intent-zs
#SBATCH --partition=HGX,DGX
#SBATCH --qos=lv0b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

MODEL_KEY="${1:?Usage: sbatch intentonomy_vllm_zeroshot.slurm MODEL_KEY MODEL_PATH}"
MODEL_PATH="${2:?Usage: sbatch intentonomy_vllm_zeroshot.slurm MODEL_KEY MODEL_PATH}"

# A Hugging Face cache repository root (models--org--name) is not itself a
# loadable model directory. Resolve refs/main to the snapshot containing
# config.json while continuing to accept normal directories and Hub IDs.
if [[ -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  if [[ -f "${MODEL_PATH}/refs/main" ]]; then
    MODEL_REVISION="$(tr -d '[:space:]' < "${MODEL_PATH}/refs/main")"
    MODEL_SNAPSHOT="${MODEL_PATH%/}/snapshots/${MODEL_REVISION}"
    if [[ ! -f "${MODEL_SNAPSHOT}/config.json" ]]; then
      echo "ERROR: Hugging Face snapshot has no readable config.json: ${MODEL_SNAPSHOT}" >&2
      exit 2
    fi
    MODEL_PATH="${MODEL_SNAPSHOT}"
  else
    echo "ERROR: Local model directory has no config.json or refs/main: ${MODEL_PATH}" >&2
    exit 2
  fi
fi

echo "Resolved model path: ${MODEL_PATH}"

PROJECT_ROOT="${INTENTONOMY_PROJECT_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
CONDA_ENV_NAME="${INTENTONOMY_CONDA_ENV:-vllm}"
SERVER_PORT="${INTENTONOMY_VLLM_PORT:-29580}"
OUTPUT_ROOT="${INTENTONOMY_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/vllm_zeroshot}"
ANNOTATION_FILE="${INTENTONOMY_TEST_ANNOTATION:-${PROJECT_ROOT}/../Intentonomy/data/annotation/intentonomy_test2020.json}"
IMAGE_DIR="${INTENTONOMY_IMAGE_DIR:-${PROJECT_ROOT}/../Intentonomy/data/images/low}"

mkdir -p "${PROJECT_ROOT}/logs/slurm" "${OUTPUT_ROOT}/${MODEL_KEY}"
source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

VLLM_BIN="${INTENTONOMY_VLLM_BIN:-}"
if [[ -z "${VLLM_BIN}" ]]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
if [[ -z "${VLLM_BIN}" || ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vLLM is not installed in conda environment '${CONDA_ENV_NAME}'." >&2
  echo "Python: $(command -v python)" >&2
  echo "Install the CUDA 12.9 variant with uv before resubmitting:" >&2
  echo "  conda activate ${CONDA_ENV_NAME}" >&2
  echo "  python -m pip install -U uv" >&2
  echo "  uv pip install --python \"$(command -v python)\" -U vllm --torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/nightly/cu129" >&2
  echo "Step3-VL-10B requires vLLM >= 0.14.0rc2.dev143+gc0a350ca7." >&2
  exit 127
fi

echo "Using Python: $(command -v python)"
echo "Using vLLM: ${VLLM_BIN}"
"${VLLM_BIN}" --version

VLLM_VERSION="$(python -c 'from importlib.metadata import version; print(version("vllm"))')"
TRANSFORMERS_VERSION="$(python -c 'from importlib.metadata import version; print(version("transformers"))')"
TORCH_VERSION="$(python -c 'from importlib.metadata import version; print(version("torch"))')"
TORCH_CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)')"
echo "vLLM package version: ${VLLM_VERSION}"
echo "Transformers package version: ${TRANSFORMERS_VERSION}"
echo "PyTorch package version: ${TORCH_VERSION}"
echo "PyTorch CUDA build: ${TORCH_CUDA_VERSION}"
if ! python -c 'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is false"'; then
  echo "ERROR: This PyTorch CUDA build is incompatible with the allocated node or its NVIDIA driver." >&2
  echo "Use a CUDA 12.9 vLLM/PyTorch environment on this cluster." >&2
  exit 4
fi
if [[ "${MODEL_KEY}" == gemma4* \
      && "${VLLM_VERSION}" == 0.27.* \
      && "${TRANSFORMERS_VERSION}" == 5.15.* ]]; then
  echo "ERROR: vLLM ${VLLM_VERSION} is incompatible with Transformers ${TRANSFORMERS_VERSION} for Gemma 4." >&2
  echo "Pin the verified compatible Transformers release, then resubmit:" >&2
  echo "  python -m pip install 'transformers==5.14.1'" >&2
  exit 3
fi

SERVER_LOG="${OUTPUT_ROOT}/${MODEL_KEY}/vllm_server.log"
VLLM_ARGS=(
  serve "${MODEL_PATH}"
  --served-model-name "${MODEL_KEY}"
  --host 127.0.0.1
  --port "${SERVER_PORT}"
  --dtype bfloat16
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.90
  --max-model-len 8192
  --limit-mm-per-prompt '{"image":1}'
)

if [[ "${MODEL_KEY}" == "stepvl-10b" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

# Optional site-specific flags, for example:
# INTENTONOMY_VLLM_EXTRA_ARGS="--enforce-eager"
if [[ -n "${INTENTONOMY_VLLM_EXTRA_ARGS:-}" ]]; then
  # Intentional word splitting: this variable represents additional CLI arguments.
  read -r -a EXTRA_ARGS <<< "${INTENTONOMY_VLLM_EXTRA_ARGS}"
  VLLM_ARGS+=("${EXTRA_ARGS[@]}")
fi

"${VLLM_BIN}" "${VLLM_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
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
    echo "vLLM exited before becoming ready; see ${SERVER_LOG}" >&2
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; then
    READY=1
    break
  fi
  sleep 5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "Timed out waiting for vLLM; see ${SERVER_LOG}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
EVAL_ARGS=(
  --base-url "http://127.0.0.1:${SERVER_PORT}/v1"
  --model "${MODEL_KEY}"
  --annotation-file "${ANNOTATION_FILE}"
  --image-dir "${IMAGE_DIR}"
  --output-dir "${OUTPUT_ROOT}/${MODEL_KEY}"
  --max-image-size 768
  --max-tokens 768
  --workers 24
  --request-timeout 180
  --retries 3
  --disable-thinking
  --json-response
  --require-all-classes
)

if [[ "${MODEL_KEY}" == "qwen3.5-9b" ]]; then
  EVAL_ARGS+=(
    --temperature 1.0
    --top-p 0.95
    --top-k 20
    --min-p 0.0
    --presence-penalty 1.5
    --repetition-penalty 1.0
  )
else
  EVAL_ARGS+=(--temperature 0 --top-p 1.0)
fi

python -u scripts/run_intentonomy_vllm_zeroshot.py "${EVAL_ARGS[@]}"
