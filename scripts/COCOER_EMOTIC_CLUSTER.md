# CocoER matched ViT-L/14 on EMOTIC

This workflow runs `CocoER (CLIP ViT-L/14 matched adaptation)` on the same
10,613-person EMOTIC val+test pool and the same five-seed/five-fold split used
by the FDIL transfer experiment. It does not use CocoER's native ResNet-50 and
CLIP-RN50 weights; one frozen OpenAI CLIP ViT-L/14 supplies context, body and
head tokens, while CocoER's alignment, competition and coordination head is
trained inside each outer fold.

## 1. Create the cluster environment once

```bash
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
bash scripts/setup_cocoer_cluster_env.sh
```

This creates `emotic-matched` from `environment.emotic-matched.yaml` and
downloads InsightFace `buffalo_l` into the shared home cache. The formal feature
job runs in strict mode and fails instead of silently using a head-crop fallback
when the detector cannot initialize.

## 2. Submit the complete workflow

```bash
bash scripts/submit_cocoer_emotic.sh
```

The submission chain is:

1. Detect/match faces and cache context/body/head ViT-L/14 tokens.
2. Run a two-element Slurm array. Element 0 handles seed indices 0,2,4 and
   element 1 handles 1,3; every seed runs all five folds.
3. Require all 25 folds and write the final JSON/CSV summary.

No job is submitted merely by creating or inspecting these scripts.

## Paths and resuming

Default outputs:

```text
logs/analysis/cocoer_matched_vitl14_features/
logs/analysis/cocoer_matched_vitl14_cluster/
logs/slurm/cocoer_emotic_*.out
```

Completed fold JSON/NPZ/checkpoints are skipped automatically when a job is
resubmitted. Feature extraction needs roughly 2.5 GB for the three FP16 token
caches. Common overrides are `COCOER_ENV`, `COCOER_FEATURES`, `COCOER_OUTPUT`,
`COCOER_FEATURE_BATCH_SIZE`, `COCOER_BATCH_SIZE`, and `COCOER_EPOCHS`.

For a single-fold smoke test after the feature job:

```bash
conda activate emotic-matched
python scripts/run_emotic_matched_baseline_fold.py \
  --method cocoer \
  --feature-dir logs/analysis/cocoer_matched_vitl14_features \
  --output-dir logs/analysis/cocoer_matched_vitl14_cluster \
  --seed-index 0 --fold 0
```

