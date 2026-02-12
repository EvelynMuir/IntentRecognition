#!/bin/bash

#SBATCH -J Intentonomy-ResNet101-MCC-global-16-ema-glove-binarize-softprob-similarity_temperature-inference_strategy-sparsity_loss_smart
#SBATCH -o output/%x.log
#SBATCH -e output/%x.err
#SBATCH -t 2-00:00:00
#SBATCH -p HGX,DGX
#SBATCH -G 1
#SBATCH -q lv0b
#SBATCH -x dgx-hyperplane15,dgx-hyperplane17,hgx-hyperplane07

source /share/lmcp/tangyin/softwares/miniconda3/etc/profile.d/conda.sh
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
conda activate intent

python src/train.py experiment=intentonomy_resnet101_mcc logger=tensorboard \
logger.tensorboard.name="${SLURM_JOB_NAME}" model.k_patches=16 \
+model.use_global_feature=true \
+data.binarize_softprob=true \
+model.use_cosine_similarity_temperature=true \
+model.use_inference_strategy=true \
model.sparsity_loss_type=smart