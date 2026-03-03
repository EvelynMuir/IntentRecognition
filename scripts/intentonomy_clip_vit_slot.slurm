#!/bin/bash

#SBATCH -J Intentonomy-CLIP-ViT-IntentSlot-CLSFusion-LLMDes-Separate-Concat-EMA-TextToVisual-NoOrthogonality
#SBATCH -o output/%x.log
#SBATCH -e output/%x.err
#SBATCH -t 2-00:00:00
#SBATCH -p HGX,DGX
#SBATCH -G 1
#SBATCH -q lv0b
#SBATCH -x dgx-hyperplane15,hgx-hyperplane07

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
conda activate intent

python src/train.py experiment=intentonomy_clip_vit_slot logger=tensorboard \
logger.tensorboard.name="Intentonomy-CLIP-ViT-IntentSlot-CLSFusion-LLMDes-Separate-Concat-EMA-TextToVisual-NoOrthogonality" \
model.net.clip_model_name="ViT-L/14" \
model.net.selected_layers=[24] \
model.net.num_slots=4 \
model.net.slot_iters=3 \
model.net.use_intent_conditioning=true \
model.net.intent_description_mode="llm" \
model.net.intent_gemini_file="/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/intent_description_gemini.json" \
data.batch_size=64 \
model.optimizer.lr=1e-4 \
+model.use_cls_fusion=true \
+model.use_decoupled_classifier=true \
+model.use_proto_classifier=true \
model.use_slot_orthogonality=false
