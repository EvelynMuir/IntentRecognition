#!/bin/bash

#SBATCH -J intentonomy_vit_mcc_global
#SBATCH -o output/intentonomy_vit_mcc_global.log
#SBATCH -e output/intentonomy_vit_mcc_global.err
#SBATCH -t 2-00:00:00
#SBATCH -p HGX,DGX
#SBATCH -G 1
#SBATCH -q lv0b
#SBATCH -x dgx-hyperplane15,dgx-hyperplane17,hgx-hyperplane07

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
conda activate intent

# python src/train.py -m experiment=intentonomy_vit_mcc.yaml logger=wandb model.k_patches=32,64,128
python src/train.py experiment=intentonomy_vit_mcc_global logger=tensorboard model.k_patches=32
