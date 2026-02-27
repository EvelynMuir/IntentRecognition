#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

python src/train.py experiment=intentonomy_clip_vit_layer_patch_mean logger=tensorboard \
logger.tensorboard.name="Intentonomy-CLIP-ViT-Layer24-PatchMean" \
model.net.clip_model_name="ViT-L/14" \
model.layer_idx=24 \
model.net.layer_idx=24 \
data.batch_size=64 \
model.optimizer.lr=1e-4 \
ckpt_path="logs/train/runs/2026-02-26_14-28-27/checkpoints/last.ckpt"