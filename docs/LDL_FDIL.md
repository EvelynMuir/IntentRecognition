# Flickr-LDL / Twitter-LDL preparation for FDIL

## Prepared layout

Run:

```bash
python scripts/prepare_ldl.py
```

The command reads the four extracted split-1 LMDBs and the MATLAB v7.3
configuration files under `../LDL`. It writes lightweight metadata to:

```text
../LDL/processed/flickrldl/
../LDL/processed/twitterldl/
```

Images remain in the original LMDBs. This avoids duplicating roughly 4 GB of
raw image data as thousands of small files on the shared filesystem. The data
module opens each LMDB lazily in its DataLoader worker. To explicitly export
JPEG files instead, pass `--extract-images`.

The class order is:

```text
amusement, anger, awe, contentment, disgust,
excitement, fear, sadness
```

Official split 1 is retained as the test set. Official fold 2 is used as the
validation set, and folds 3-5 are used for training. This yields:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| Flickr-LDL | 6,690 | 2,231 | 2,229 |
| Twitter-LDL | 6,026 | 2,010 | 2,009 |

Use `--val-fold {2,3,4,5}` to select a different validation fold. The generated
metadata preserves `official_split` and `fold_index`, so a five-fold protocol
can be built without reconstructing the source data.

## Build FDIL-compatible CLIP caches

The existing cache builder works with the two new Hydra data configs:

```bash
python scripts/build_clip_distill_cache.py \
  --data flickrldl \
  --output-dir logs/analysis/flickrldl_clip_cache

python scripts/build_clip_distill_cache.py \
  --data twitterldl \
  --output-dir logs/analysis/twitterldl_clip_cache
```

Each cache contains `features`, `crop_features`, `full_features`, `labels`,
`soft_labels`, and `image_ids`, matching the current FDIL feature-cache schema.
Here `soft_labels` is the actual LDL target distribution. `labels` is an argmax
one-hot compatibility field only.

## Important protocol difference

The current Intentonomy/EMOTIC FDIL trainer uses independent sigmoid outputs,
asymmetric binary loss, class-wise thresholds, and multi-label metrics. Those
choices are not valid for label-distribution learning. An LDL experiment should:

- normalize predictions with softmax;
- supervise against `soft_labels` using distribution cross-entropy/KL (or an
  explicitly selected LDL objective);
- use the distribution itself for the UTD agreement/uncertainty gate (the data
  module also exposes maximum mass as `agreement` and entropy separately);
- report Chebyshev, Clark, KLD, cosine similarity, Spearman rank correlation,
  DeltaLDL's parameter-free `mu` percentage, and DVSE with the Flickr/Twitter
  polarity partition specified by the divisiveness-consistent LDL protocol;
- select hyperparameters on the validation fold and report the untouched
  official split-1 test set.

Do not report results obtained by feeding the one-hot `labels` field into the
existing multi-label FDIL loss; that would discard the property this benchmark
is designed to measure.

## LDL-FDIL adaptation

The implementation is `scripts/run_ldl_fdil.py`. It preserves the two original
FDIL functions while changing only task-dependent probability semantics.

For a target distribution `y` and student logits `z`, direct supervision is
categorical distribution cross-entropy:

```text
L_sup = -sum_c y_c log softmax(z)_c
```

SLR-C retains the original lexical/canonical/scenario CLIP prior, independent
per-source sample-wise z-score, source averaging, Top-5 local reranking with
`alpha=0.3`, and image-conditioned residual MLP:

```text
z_slrc,c = z_base,c + alpha * prior_c  if c is in TopK(z_base)
z_full   = z_slrc + h_res(image_feature)
```

UTD rationales are generated only for training rows by Qwen3-VL using the
ground-truth distribution and a baseline confusion candidate, then encoded by
the original BGE-large-en-v1.5 encoder. A three-fold cross-fitted teacher makes
every train-row distillation target out-of-fold. For LDL, annotator agreement is
the dominant vote mass `omega=max_c(y_c)`, so `g=1-omega`; categorical KL
replaces the original multi-label Bernoulli KL:

```text
L_FDIL = (1-g) * L_sup + lambda * g * KL(q_teacher,T || p_student,T) * T^2
```

The teacher and rationale encoder are discarded at validation/test inference.
There is no threshold search because each prediction is a simplex distribution.

## Shared Gemini SLR-C prior

Both datasets use the same authoritative prior file:

```text
../LDL/processed/semantic_priors/ldl_gemini_slrc_prior.json
```

The experiment runner automatically prefers this file over the small fallback
descriptions generated during preprocessing. Its SHA-256 is written into every
run's `summary.json`. A different prior must be selected explicitly with
`--description-file`, which keeps prior changes auditable.
