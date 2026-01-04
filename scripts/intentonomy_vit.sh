#!/bin/bash

source /home/evelynmuir/softwares/miniconda3/etc/profile.d/conda.sh
cd /home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra
conda activate /home/evelynmuir/lambda/projects/IntentRecognition/.conda

python src/train.py experiment=intentonomy_vit.yaml logger=wandb trainer.max_epochs=30 