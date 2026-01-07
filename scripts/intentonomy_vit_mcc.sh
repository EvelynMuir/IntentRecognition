#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
cd /home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# python src/train.py -m experiment=intentonomy_vit_mcc logger=wandb model.k_patches=32,64,128
# python src/train.py experiment=intentonomy_vit_mcc logger=wandb model.k_patches=32
python src/train.py experiment=intentonomy_vit_mcc logger=tensorboard model.k_patches=32 ckpt_path=logs/train/multiruns/2026-01-04_11-10-13/0/checkpoints/last.ckpt
