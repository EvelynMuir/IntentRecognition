#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# python src/train.py experiment=intentonomy_clip_vit_slot logger=tensorboard \
# logger.tensorboard.name="Intentonomy-CLIP-ViT-IntentSlot-CLSFusion-DetailedDes" \
# model.net.clip_model_name="ViT-L/14" \
# model.net.selected_layers=[24] \
# model.net.num_slots=4 \
# model.net.slot_iters=3 \
# model.net.use_intent_conditioning=true \
# data.batch_size=64 \
# model.optimizer.lr=1e-4 \
# +model.use_cls_fusion=true \
# model.net.intent_description_mode="detailed" \
# +model.use_decoupled_classifier=false

python src/train.py experiment=intentonomy_clip_vit_slot logger=tensorboard \
logger.tensorboard.name="Intentonomy-CLIP-ViT-IntentSlot-CLSFusion-LLMDes-Separate-Concat-TextToVisual-NoOrthogonality" \
model.net.clip_model_name="ViT-L/14" \
model.net.selected_layers=[24] \
model.net.num_slots=4 \
model.net.slot_iters=3 \
model.net.use_intent_conditioning=true \
model.net.intent_description_mode="llm" \
model.net.intent_gemini_file="/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_description_gemini.json" \
data.batch_size=64 \
model.optimizer.lr=1e-4 \
model.use_cls_fusion=true \
model.use_decoupled_classifier=true \
model.use_slot_orthogonality=false
