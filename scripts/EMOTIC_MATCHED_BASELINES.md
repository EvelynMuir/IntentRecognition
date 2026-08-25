# EMOTIC matched ViT-L/14 baselines

This workflow runs two explicitly named matched-backbone adaptations on the
same 10,613-person EMOTIC val+test pool and the same five-seed/five-fold split
used by the FDIL transfer experiment:

- `CocoER (CLIP ViT-L/14 matched adaptation)`
- `EmotionCLIP (CLIP ViT-L/14 mask-aware linear-probe adaptation)`

These are controlled backbone adaptations, not the native-backbone official
checkpoints. The OpenAI CLIP ViT-L/14 backbone is frozen in both methods.

## One-time environment setup

Run on a cluster login node:

```bash
cd /share/lmcp/tangyin/projects/IntentRecognition/lightning-hydra
bash scripts/setup_emotic_matched_env.sh
```

The separate `emotic-matched` environment is intentional: InsightFace and its
ONNX runtime are not required by the existing FDIL environment and should not
change that environment's resolved packages.

## Smoke test

First submit feature extraction. It reuses FDIL labels/IDs and only adds the
mask-aware and spatial-token caches:

```bash
FEATURE_JOB=$(sbatch --parsable scripts/emotic_matched_features.slurm)
```

After it succeeds, test one fold of each method interactively on an allocated
GPU or by adding the same command to a temporary one-task Slurm job:

```bash
conda activate emotic-matched
python scripts/run_emotic_matched_baseline_fold.py --method emotionclip --seed-index 0 --fold 0
python scripts/run_emotic_matched_baseline_fold.py --method cocoer --seed-index 0 --fold 0
```

Completed folds are resumable. Add `--overwrite` only when an existing fold
must deliberately be replaced.

## Full run

The following submits feature extraction, a two-task baseline array, and the
dependent summary job:

```bash
bash scripts/submit_emotic_matched_baselines.sh
```

The array has exactly two tasks, satisfying the cluster array-size limit. Each
task runs all 25 folds for one method. Outputs are written under:

```text
logs/analysis/emotic_matched_vitl14_features/
logs/analysis/emotic_matched_baselines_vitl14/
logs/slurm/emotic_match_*.out
```

Useful overrides include `EMOTIC_MATCHED_ENV`, `EMOTIC_MATCHED_FEATURES`,
`EMOTIC_MATCHED_OUTPUT`, `EMOTIC_FEATURE_BATCH_SIZE`, and
`EMOTIC_COCOER_BATCH_SIZE`.

