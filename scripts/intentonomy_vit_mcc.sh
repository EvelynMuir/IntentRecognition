#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
cd /home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

# python src/train.py -m experiment=intentonomy_vit_mcc.yaml logger=wandb model.k_patches=32,64,128
# python src/train.py experiment=intentonomy_vit_mcc logger=wandb model.k_patches=32
python src/train.py -m experiment=intentonomy_vit_mcc.yaml logger=tensorboard model.k_patches=32,64,128
