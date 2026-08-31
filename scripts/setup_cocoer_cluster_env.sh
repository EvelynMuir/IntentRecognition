#!/usr/bin/env bash
set -eo pipefail

ROOT="${COCOER_ROOT:-/share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra}"
ENV_NAME="${COCOER_ENV:-emotic-matched}"
source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd "${ROOT}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f environment.emotic-matched.yaml --prune
else
  conda env create -n "${ENV_NAME}" -f environment.emotic-matched.yaml
fi

conda activate "${ENV_NAME}"
set -u
python -m pip install -e . --no-deps
python - <<'PY'
import clip, cv2, insightface, numpy, onnxruntime, sklearn, torch
from insightface.app import FaceAnalysis
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
print("CUDA visible on login node:", torch.cuda.is_available())
print("insightface", insightface.__version__, "onnxruntime", onnxruntime.__version__)
FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
print("InsightFace buffalo_l is available in the shared home cache")
print("CocoER cluster environment check passed")
PY
