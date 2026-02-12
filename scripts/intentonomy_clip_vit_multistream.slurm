#!/bin/bash

#SBATCH -J Intentonomy-Clip-ViT-MultiStream-Mixed-Token-Factor-Separation-Gated-Fusion-Learnable-Codebook
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

python src/train.py experiment=intentonomy_clip_vit_multistream logger=tensorboard \
logger.tensorboard.name="${SLURM_JOB_NAME}" \
+model.use_gated_fusion=true \
model.optimizer.lr=5e-4 \
model.lr_classifier=5e-4 \
model.lr_vq_coco=1e-4 \
model.lr_vq_places=1e-4 \
model.lr_vq_emotion=1e-4 \
model.lr_vq_ava=1e-4 \
model.lr_vq_actions=1e-4 \
model.ema_decay=0.999 \
+model.token_mode="mixed" \
model.freeze_codebook=false \
model.vq_commitment_cost=0.07 \
model.intent_loss_weight_warmup_epochs=2 \
model.intent_loss_weight_warmup=0.3 \
model.intent_loss_weight_normal=1.5 \
+model.use_factor_separation_loss=true \
data.batch_size=256
