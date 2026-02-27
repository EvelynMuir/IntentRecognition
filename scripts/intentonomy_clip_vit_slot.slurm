#!/bin/bash

#SBATCH -J Intentonomy-CLIP-ViT-IntentSlot
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
logger.tensorboard.name="${SLURM_JOB_NAME}" \
model.net.clip_model_name="ViT-L/14" \
model.net.selected_layers=[12,18,24] \
model.net.num_slots=4 \
model.net.slot_iters=3 \
model.net.use_intent_conditioning=true \
model.slot_orthogonality_weight=0.1 \
data.batch_size=1024
