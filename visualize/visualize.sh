# python src/visualize_patches.py \
#   --ckpt_path /home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/train/runs/2026-01-27_14-33-29/checkpoints/epoch_008.ckpt \
#   --num_samples 20 \
#   --image_size 224 \
#   --annotation_dir ../Intentonomy/data/annotation \
#   --image_dir ../Intentonomy/data/images/low

python visualize/visualize_codebook.py \
    --ckpt_path /home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/train/runs/2026-02-02_14-29-58/checkpoints/epoch_044.ckpt \
    --factor_id 0 \
    --code_id 5 \
    --topk 20