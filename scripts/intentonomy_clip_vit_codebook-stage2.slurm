#!/bin/bash

#SBATCH -J Intentonomy-Clip-ViT-Codebook-256-Factor-Separation-Loss-Patch-MLP-Fusion-Stage2
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

python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
logger.tensorboard.name="${SLURM_JOB_NAME}" \
data.batch_size=256 \
model.intent_loss_weight_warmup_epochs=3 \
model.intent_loss_weight_warmup=0.3 \
model.intent_loss_weight_normal=1.5 \
model.vq_commitment_cost=0.07 \
+model.use_factor_separation_loss=true \
model.use_anchor_loss=true \
model.unfreeze_last_n_blocks=2 \
model.lr_vit=5e-6 \
model.lr_projector=5e-4 \
model.lr_vq=5e-4 \
model.lr_classifier=5e-4 \
model.optimizer.lr=5e-4 \
model.pretrained_ckpt_path=logs/train/runs/2026-02-05_13-47-02/checkpoints/epoch_046.ckpt