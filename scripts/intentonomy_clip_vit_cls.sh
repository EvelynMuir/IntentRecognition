#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

python src/train.py experiment=intentonomy_clip_vit_layer_cls \
    model.layer_idx=24 \
    model.net.layer_idx=24 \
    logger=tensorboard \
    logger.tensorboard.name="Intentonomy-CLIP-ViT-cls" \
    data.batch_size=64 \
    model.optimizer.lr=1e-4