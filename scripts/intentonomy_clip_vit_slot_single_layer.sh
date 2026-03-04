#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

python src/train.py experiment=intentonomy_clip_vit_slot_single_layer logger=tensorboard \
logger.tensorboard.name="Intentonomy-CLIP-ViT-IntentSlot-SingleLayer" \
model.net.clip_model_name="ViT-L/14" \
model.net.selected_layers=[24] \
data.batch_size=64 \
model.optimizer.lr=1e-4 \
model.use_slot_orthogonality=false
