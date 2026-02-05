#!/bin/bash

# Streamlit Codebook 可视化启动脚本

# 设置默认路径（根据实际情况修改）
DEFAULT_CKPT_PATH="/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/train/runs/2026-02-02_14-29-58/checkpoints/epoch_044.ckpt"
DEFAULT_ANNOTATION_DIR="/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/annotation"
DEFAULT_IMAGE_DIR="/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/images/low"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_DIR="$SCRIPT_DIR"

# 切换到visualize目录
cd "$VISUALIZE_DIR"

# 运行Streamlit应用
streamlit run streamlit_codebook.py --server.port 8501 --server.address 0.0.0.0

