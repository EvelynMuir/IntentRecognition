#!/bin/bash

#SBATCH -J intentonomy_resnet101
#SBATCH -o output/intentonomy_resnet101.log
#SBATCH -e output/intentonomy_resnet101.err
#SBATCH -t 2-00:00:00
#SBATCH -p HGX,DGX
#SBATCH -G 1
#SBATCH -q lv0b
#SBATCH -x dgx-hyperplane15,dgx-hyperplane17,hgx-hyperplane07

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
conda activate intent

python src/train.py experiment=intentonomy_resnet101 logger=tensorboard
