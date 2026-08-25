# Baseline, FDIL, and LDL-DVS on Emotion6

> Historical 3-seed report. Superseded by `record_0821_AllLDL_5Seeds.md`.

## Protocol

- Dataset: Emotion6, 1,980 images and seven distribution labels.
- Class order: anger, disgust, fear, joy, sadness, surprise, neutral.
- Official train/test: 1,386/594; a fixed dominant-class-stratified 20% holdout
  of official train is validation, yielding 1,108/278/594 train/val/test.
- Frozen CLIP ViT-L/14 full-image features, 768 dimensions.
- Results: mean ± population standard deviation over seeds 2026/2027/2028.
- Best epoch selected exclusively by validation KLD.
- FDIL: shared Gemini lexical/canonical/6-scenario prior, Top-5, alpha=0.3;
  Qwen3-VL rationale teacher encoded with BGE-large and 3-fold OOF distillation.
- LDL-DVS: paper defaults `k=10`, `alpha=0.1`, matched MLP initialization and
  fully enumerated pairwise divisiveness surrogate.
- DVSE polarity: positive={joy, surprise}, negative={anger, disgust, fear,
  sadness}, excluded={neutral}.

## Main comparison

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP baseline | 0.1806 ± 0.0007 | 1.5990 ± 0.0022 | 0.2549 ± 0.0005 | 0.8938 ± 0.0004 | 0.7229 ± 0.0019 | 63.95 ± 0.04 | 0.07846 ± 0.00096 |
| LDL-DVS + CLIP | 0.1807 ± 0.0007 | 1.5991 ± 0.0022 | 0.2549 ± 0.0005 | 0.8938 ± 0.0004 | 0.7228 ± 0.0019 | 63.94 ± 0.04 | **0.07842 ± 0.00096** |
| Full FDIL | **0.1801 ± 0.0011** | **1.5897 ± 0.0013** | **0.2514 ± 0.0007** | **0.8955 ± 0.0008** | **0.7236 ± 0.0007** | **64.41 ± 0.10** | 0.07869 ± 0.00039 |

Full FDIL improves 6/7 metrics over the matched baseline:

- Cheby: -0.000515
- Clark: -0.009319
- KLD: -0.003531
- Cosine: +0.001674
- Spearman: +0.000684
- µ: +0.462 percentage points
- DVSE: +0.000228 (worse)

LDL-DVS at the paper-default alpha is again nearly identical to baseline. It
improves only DVSE by 0.0000368 and is marginally worse on the other six
metrics. An alpha=0 paired audit reproduces the baseline with maximum metric
difference 1.51e-7, confirming that this is not an initialization artifact.

## FDIL branch ablation

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP baseline | 0.1806 | 1.5990 | 0.2549 | 0.8938 | 0.7229 | 63.95 | 0.07846 |
| UTD only | 0.1802 | 1.5930 | 0.2527 | 0.8946 | **0.7272** | 64.25 | 0.07902 |
| SLR-C only | **0.1801** | 1.5948 | **0.2510** | **0.8959** | 0.7248 | **64.56** | **0.07786** |
| Full FDIL | 0.1801 | **1.5897** | 0.2514 | 0.8955 | 0.7236 | 64.41 | 0.07869 |

SLR-C is the strongest single branch for distribution-shape metrics, whereas
UTD gives the best rank correlation. Their full combination gives the best
Clark score but does not dominate the best branch on every metric.

## Comparison with the paper's 168-D setting

The LDL-DVS paper reports the following Emotion6 result using PCA-reduced
168-D LBP/HOG/Color-Moment features and 10-fold × 10-repeat evaluation:

| Setting | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper LDL-DVS, 168-D handcrafted | 0.3030 | 1.6579 | 0.5624 | 0.7288 | 0.4061 | 31.26 | 0.1074 |
| Matched LDL-DVS, CLIP | 0.1807 | 1.5991 | 0.2549 | 0.8938 | 0.7228 | 63.94 | 0.0784 |

The absolute improvement again comes overwhelmingly from the CLIP
representation. Once the representation and initialization are matched,
paper-default LDL-DVS contributes almost no additional change.

## Artifact provenance

- Gemini prior: `../LDL/processed/semantic_priors/emotion6_gemini_slrc_prior.json`
- Gemini prior SHA-256: `45efc53cfaa8d16189a5b1fe19023aeab845500638868e2da1c179ba81f10957`
- Rationale verification: `logs/analysis/emotion6_rationales/verification.json`
- FDIL summary: `logs/analysis/emotion6_fdil_20260821/summary.json`
- LDL-DVS summary: `logs/analysis/emotion6_ldl_dvs_clip_20260821/summary.json`
- Saved test predictions reproduce all summary metrics with zero discrepancy.

## Five-seed rerun used by the paper (revision_2)

Extended to seeds 2026-2030 to match the Flickr-LDL and Intentonomy standard.
Seeds 2026-2028 reproduce the 3-seed run to within 1.9e-3 with identical
best-epoch selection (GPU nondeterminism; Emotion6's 594-image test split
amplifies small numerical differences relative to Flickr).

```bash
python scripts/run_ldl_fdil.py --dataset emotion6 \
  --cache-dir logs/analysis/emotion6_clip_cache/_cache \
  --rationale-features logs/analysis/emotion6_rationales/train_rationale_bge_features.npz \
  --output-dir logs/analysis/emotion6_fdil_5seed_20260821 \
  --seeds 2026,2027,2028,2029,2030

python scripts/run_ldl_dvs_clip.py --dataset emotion6 \
  --cache-dir logs/analysis/emotion6_clip_cache/_cache \
  --output-dir logs/analysis/emotion6_ldl_dvs_clip_5seed_20260821 \
  --seeds 2026,2027,2028,2029,2030
```

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---|---|---|---|---|---|---|
| CLIP baseline | 0.1794 ± 0.0019 | 1.5979 ± 0.0025 | 0.2532 ± 0.0026 | 0.8947 ± 0.0014 | 0.7223 ± 0.0025 | 64.20 ± 0.37 | 0.0784 ± 0.0009 |
| LDL-DVS (matched) | 0.1794 ± 0.0019 | 1.5981 ± 0.0026 | 0.2532 ± 0.0026 | 0.8946 ± 0.0014 | 0.7222 ± 0.0024 | 64.20 ± 0.37 | 0.0783 ± 0.0009 |
| SLR-C only | 0.1797 ± 0.0016 | 1.5976 ± 0.0053 | 0.2515 ± 0.0019 | 0.8955 ± 0.0013 | 0.7231 ± 0.0026 | 64.52 ± 0.28 | 0.0781 ± 0.0006 |
| UTD only | 0.1804 ± 0.0018 | 1.5949 ± 0.0047 | 0.2537 ± 0.0033 | 0.8941 ± 0.0018 | **0.7256 ± 0.0026** | 64.12 ± 0.43 | 0.0791 ± 0.0014 |
| FDIL | 0.1802 ± 0.0011 | **1.5910 ± 0.0023** | 0.2513 ± 0.0006 | 0.8953 ± 0.0008 | 0.7219 ± 0.0026 | 64.41 ± 0.12 | 0.0781 ± 0.0009 |

Paired t-tests vs baseline (n=5): the ONLY significant contrasts are
FDIL Clark −0.0069 (p=0.013) and UTD-only Spearman +0.0034 (p=0.042).
Nothing else reaches p<0.05, and SLR-C reaches nothing at all.

### The awkward finding, stated for the record

The 5-seed Emotion6 result does NOT replicate Flickr-LDL at the metric level,
and the paper says so rather than aggregating:

- Clark: FDIL is significantly WORSE on Flickr (+0.0079, p=0.008) and
  significantly BETTER on Emotion6 (−0.0069, p=0.013). Opposite signs, both
  significant.
- DVSE: on Flickr the gain is UTD's (−0.0043, p=0.010); on Emotion6 UTD makes
  DVSE worse and SLR-C is the better branch.

What DOES replicate in direction on both datasets, and is the only claim the
manuscript makes from these two datasets:

- SLR-C moves pointwise fit only: KLD (−0.0027, −0.0016), cosine (+0.0007,
  +0.0008), µ (+0.18, +0.32); never significant on either.
- UTD moves rank correlation: Spearman (+0.0028 p=0.098, +0.0034 p=0.042)
  while leaving KLD no better (+0.0021, +0.0005).

Emotion6's 594-image test split has little power at this effect size; the
manuscript states this as a caveat rather than as an excuse.

LDL-DVS on Emotion6 is significant on 6/7 metrics but numerically negligible
(|delta| <= 2e-4). It does improve its own target metric DVSE reliably
(−4e-5, p=0.0004) — directionally correct, magnitude irrelevant on strong
frozen features.
