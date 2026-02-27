#!/bin/bash

#SBATCH -J Intentonomy-Clip-ViT-Layer-CLS
#SBATCH -o output/%x-%A_%a.log
#SBATCH -e output/%x-%A_%a.err
#SBATCH -t 2-00:00:00          # 最长2天
#SBATCH -p HGX,DGX
#SBATCH -G 1
#SBATCH -q lv0b
#SBATCH -x dgx-hyperplane15,hgx-hyperplane07

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
conda activate intent

# Get layer index from SLURM array task ID
LAYER=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Running experiment for layer ${LAYER}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

python src/train.py experiment=intentonomy_clip_vit_layer_cls \
    model.layer_idx=${LAYER} \
    model.net.layer_idx=${LAYER} \
    logger=tensorboard \
    logger.tensorboard.name="Intentonomy-CLIP-ViT-Layer${LAYER}"

echo "Completed experiment for layer ${LAYER}"
