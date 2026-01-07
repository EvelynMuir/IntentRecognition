#!/bin/bash

#SBATCH -J intentonomy_vit_mcc_adaptive
#SBATCH -o output/intentonomy_vit_mcc_adaptive.log
#SBATCH -e output/intentonomy_vit_mcc_adaptive.err
#SBATCH -t 2-00:00:00
#SBATCH -p HGX,DGX
#SBATCH -G 1
#SBATCH -q lv0b
#SBATCH -x dgx-hyperplane15,dgx-hyperplane17,hgx-hyperplane07

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
conda activate intent

# python src/train.py -m experiment=intentonomy_vit_mcc.yaml logger=wandb model.k_patches=32,64,128
python src/train.py experiment=intentonomy_vit_mcc_adaptive logger=tensorboard ckpt_path=logs/train/runs/2026-01-05_10-11-43/checkpoints/last.ckpt
