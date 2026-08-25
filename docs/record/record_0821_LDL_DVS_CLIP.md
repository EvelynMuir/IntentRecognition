# LDL-DVS on matched CLIP features

> Historical 3-seed report. Superseded by `record_0821_AllLDL_5Seeds.md`.

## Question

The original LDL-DVS paper uses PCA-reduced 168-D LBP/HOG/Color-Moment
features, while our LDL-FDIL study uses frozen CLIP ViT-L/14 features. This
experiment isolates the method-level effect by running LDL-DVS on exactly the
same CLIP features, MLP architecture, train/validation/test split, optimizer,
early stopping rule, and seeds as the FDIL study.

## Protocol

- Frozen CLIP ViT-L/14 full-image feature: 768 dimensions.
- Prediction head: the matched 768-hidden `DistributionMLP`.
- Official split 1 test; official fold 2 validation.
- Best epoch selected by validation KLD only.
- Seeds: 2026, 2027, 2028.
- LDL-DVS paper defaults: `k=10`, `alpha=0.1`.
- Loss: `KLD + alpha * fully-enumerated pairwise divisiveness surrogate`.
- Positive labels: amusement, contentment, excitement.
- Negative labels: anger, disgust, fear, sadness; awe is excluded.

## Flickr-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched CLIP baseline | 0.2035 ± 0.0006 | **2.3043 ± 0.0042** | 0.3441 ± 0.0010 | **0.8969 ± 0.0003** | 0.7109 ± 0.0026 | 71.35 ± 0.13 | 0.0773 ± 0.0022 |
| LDL-DVS (matched CLIP) | 0.2035 ± 0.0006 | 2.3044 ± 0.0042 | 0.3442 ± 0.0010 | 0.8969 ± 0.0003 | 0.7108 ± 0.0026 | 71.35 ± 0.13 | 0.0772 ± 0.0022 |
| Full FDIL | **0.2025 ± 0.0005** | 2.3123 ± 0.0024 | **0.3438 ± 0.0002** | 0.8968 ± 0.0002 | **0.7137 ± 0.0012** | **71.41 ± 0.04** | **0.0733 ± 0.0007** |

With paired initialization, LDL-DVS is almost identical to the matched baseline.
It improves only Cheby (`-0.0000077`) and DVSE (`-0.0000354`); the other changes
are similarly tiny and unfavorable. Full FDIL is better than LDL-DVS on Cheby,
KLD, Spearman, µ, and DVSE, while LDL-DVS retains slightly better Clark and
cosine.

## Twitter-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched CLIP baseline | 0.2047 ± 0.0008 | **2.3859 ± 0.0038** | **0.3704 ± 0.0008** | **0.9054 ± 0.0003** | 0.6687 ± 0.0005 | **72.22 ± 0.02** | 0.0718 ± 0.0021 |
| LDL-DVS (matched CLIP) | **0.2047 ± 0.0008** | 2.3860 ± 0.0038 | 0.3704 ± 0.0008 | 0.9054 ± 0.0003 | 0.6687 ± 0.0005 | 72.22 ± 0.02 | 0.0718 ± 0.0021 |
| Full FDIL | 0.2053 ± 0.0007 | 2.3904 ± 0.0045 | 0.3732 ± 0.0030 | 0.9043 ± 0.0012 | **0.6695 ± 0.0007** | 72.09 ± 0.19 | **0.0691 ± 0.0021** |

LDL-DVS improves Cheby (`-0.0000109`), Spearman (`+0.0000135`), and DVSE
(`-0.0000515`) over the matched baseline, but all effects are extremely small.
It remains better than full FDIL on Cheby, Clark, KLD, cosine, and µ; full FDIL
is better on Spearman and DVSE.

## Loss-scale diagnostic

On the seed-2026 selected checkpoints, the train-set loss terms are:

| Dataset | KLD | Pairwise DVS | `0.1 × DVS` | Auxiliary / KLD |
|---|---:|---:|---:|---:|
| Flickr-LDL | 0.235231 | 0.004776 | 0.000478 | 0.203% |
| Twitter-LDL | 0.268356 | 0.005108 | 0.000511 | 0.190% |

Thus, with the paper-default `alpha=0.1`, the auxiliary contributes only about
0.2% of the supervised KLD scale once the CLIP representation is strong. This
explains why matched LDL-DVS is nearly indistinguishable from the baseline. An
alpha sweep could test a re-tuned CLIP regime, but it would no longer be a
strict run of the paper's fixed default.

## Why the paper's absolute scores are lower

For context, the original paper reports the following under its 168-D
handcrafted-feature, 10-fold × 10-repeat protocol:

| Dataset | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flickr, paper LDL-DVS | 0.2934 ± 0.005 | 2.2026 ± 0.007 | 0.6127 ± 0.012 | 0.7867 ± 0.006 | 0.5258 ± 0.009 | 46.50 ± 0.8 | 0.1185 ± 0.003 |
| Twitter, paper LDL-DVS | 0.2948 ± 0.005 | 2.4056 ± 0.006 | 0.6238 ± 0.013 | 0.8259 ± 0.006 | 0.5512 ± 0.009 | 55.56 ± 0.9 | 0.1131 ± 0.004 |

Those rows are not directly comparable to our split because both representation
and evaluation protocol differ. The matched experiment shows that once LDL-DVS
receives CLIP features, most of the apparent absolute gap disappears. The
strong representation, rather than FDIL alone, is therefore the dominant source
of our high absolute scores.

## Artifacts

- `logs/analysis/flickrldl_ldl_dvs_clip_20260821/summary.json`
- `logs/analysis/twitterldl_ldl_dvs_clip_20260821/summary.json`
- Each directory contains three checkpoints and three test-prediction NPZ files.
- Saved predictions independently reproduce every summary metric with zero
  numerical discrepancy.
