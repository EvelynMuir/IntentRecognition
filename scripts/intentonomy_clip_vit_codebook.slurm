#!/bin/bash

#SBATCH -J Intentonomy-Clip-ViT-Codebook-256-Factor-Separation-Loss-Patch-MLP-Fusion
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

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=1024 \
# model.intent_loss_weight_warmup_epochs=3 \
# model.intent_loss_weight_warmup=0.6 \
# model.intent_loss_weight_normal=3.0 \
# model.vq_commitment_cost=0.05

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=512 \
# model.intent_loss_weight_warmup_epochs=3 \
# model.intent_loss_weight_warmup=0.4 \
# model.intent_loss_weight_normal=2.0 \
# model.vq_commitment_cost=0.05

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=256 \
# model.intent_loss_weight_warmup_epochs=3 \
# model.intent_loss_weight_warmup=0.3 \
# model.intent_loss_weight_normal=1.5 \
# model.vq_commitment_cost=0.07

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=128 \
# model.intent_loss_weight_warmup_epochs=2 \
# model.intent_loss_weight_warmup=0.2 \
# model.intent_loss_weight_normal=1.0 \
# model.vq_commitment_cost=0.10

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=1024 \
# model.intent_loss_weight_warmup_epochs=3 \
# model.intent_loss_weight_warmup=0.6 \
# model.intent_loss_weight_normal=3.0 \
# model.vq_commitment_cost=0.05 \
# +model.use_factor_separation_loss=true

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=512 \
# model.intent_loss_weight_warmup_epochs=3 \
# model.intent_loss_weight_warmup=0.4 \
# model.intent_loss_weight_normal=2.0 \
# model.vq_commitment_cost=0.05 \
# +model.use_factor_separation_loss=true

python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
logger.tensorboard.name="${SLURM_JOB_NAME}" \
data.batch_size=256 \
model.intent_loss_weight_warmup_epochs=3 \
model.intent_loss_weight_warmup=0.3 \
model.intent_loss_weight_normal=1.5 \
model.vq_commitment_cost=0.07 \
+model.use_factor_separation_loss=true

# python src/train.py experiment=intentonomy_clip_vit_codebook logger=tensorboard \
# logger.tensorboard.name="${SLURM_JOB_NAME}" \
# data.batch_size=128 \
# model.intent_loss_weight_warmup_epochs=2 \
# model.intent_loss_weight_warmup=0.2 \
# model.intent_loss_weight_normal=1.0 \
# model.vq_commitment_cost=0.10 \
# +model.use_factor_separation_loss=true