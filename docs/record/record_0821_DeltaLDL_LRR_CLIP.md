# δ-LDL and LDL-LRR on matched CLIP features

> Historical 3-seed report. Superseded by `record_0821_AllLDL_5Seeds.md`.

## Protocol

- Datasets: Flickr-LDL, Twitter-LDL, Emotion6.
- Frozen CLIP ViT-L/14 768-D full-image features.
- Same 768-hidden DistributionMLP, initialization seeds, AdamW, splits, and
  validation-KLD early stopping as the matched baseline and FDIL runs.
- Seeds: 2026, 2027, 2028.
- δ-LDL: hyperparameter-free 33-node Simpson-integral objective from the
  published PyLDL implementation.
- LDL-LRR: `KLD + 1e-3 × pairwise ranking-relation loss`, using the published
  default alpha.

## Flickr-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP baseline | 0.2035 ± 0.0006 | 2.3043 ± 0.0042 | 0.3441 ± 0.0010 | 0.8969 ± 0.0003 | 0.7109 ± 0.0026 | 71.35 ± 0.13 | 0.0773 ± 0.0022 |
| δ-LDL | 0.2036 ± 0.0008 | **2.3042 ± 0.0044** | 0.3443 ± 0.0013 | **0.8969 ± 0.0004** | 0.7110 ± 0.0028 | 71.34 ± 0.15 | 0.0772 ± 0.0023 |
| LDL-LRR | 0.2035 ± 0.0006 | 2.3044 ± 0.0042 | 0.3442 ± 0.0010 | 0.8969 ± 0.0003 | 0.7109 ± 0.0026 | 71.35 ± 0.13 | 0.0773 ± 0.0022 |
| LDL-DVS | 0.2035 ± 0.0006 | 2.3044 ± 0.0042 | 0.3442 ± 0.0010 | 0.8969 ± 0.0003 | 0.7108 ± 0.0026 | 71.35 ± 0.13 | 0.0772 ± 0.0022 |
| Full FDIL | **0.2025 ± 0.0005** | 2.3123 ± 0.0024 | **0.3438 ± 0.0002** | 0.8968 ± 0.0002 | **0.7137 ± 0.0012** | **71.41 ± 0.04** | **0.0733 ± 0.0007** |

δ-LDL and LDL-LRR are effectively tied with the baseline. δ-LDL slightly
improves Clark, cosine, Spearman, and DVSE but slightly worsens Cheby, KLD, and
µ; all changes are very small. Full FDIL remains strongest on the larger
method-sensitive changes, particularly Spearman and DVSE.

## Twitter-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP baseline | 0.2047 ± 0.0008 | 2.3859 ± 0.0038 | 0.3704 ± 0.0008 | 0.9054 ± 0.0003 | 0.6687 ± 0.0005 | 72.22 ± 0.02 | 0.0718 ± 0.0021 |
| δ-LDL | **0.2045 ± 0.0009** | **2.3857 ± 0.0036** | **0.3701 ± 0.0007** | **0.9055 ± 0.0003** | 0.6688 ± 0.0002 | **72.25 ± 0.01** | 0.0720 ± 0.0022 |
| LDL-LRR | 0.2047 ± 0.0008 | 2.3859 ± 0.0038 | 0.3704 ± 0.0008 | 0.9054 ± 0.0003 | 0.6687 ± 0.0005 | 72.22 ± 0.02 | 0.0718 ± 0.0021 |
| LDL-DVS | 0.2047 ± 0.0008 | 2.3860 ± 0.0038 | 0.3704 ± 0.0008 | 0.9054 ± 0.0003 | 0.6687 ± 0.0005 | 72.22 ± 0.02 | 0.0718 ± 0.0021 |
| Full FDIL | 0.2053 ± 0.0007 | 2.3904 ± 0.0045 | 0.3732 ± 0.0030 | 0.9043 ± 0.0012 | **0.6695 ± 0.0007** | 72.09 ± 0.19 | **0.0691 ± 0.0021** |

Twitter is the only dataset where δ-LDL gives a consistent conventional-metric
gain: it improves Cheby, Clark, KLD, cosine, Spearman, and µ over baseline, but
DVSE becomes slightly worse. Full FDIL still has the best Spearman and DVSE,
while δ-LDL is strongest on the remaining global distribution metrics.

## Emotion6

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP baseline | 0.1806 ± 0.0007 | 1.5990 ± 0.0022 | 0.2549 ± 0.0005 | 0.8938 ± 0.0004 | 0.7229 ± 0.0019 | 63.95 ± 0.04 | 0.07846 ± 0.00096 |
| δ-LDL | 0.1812 ± 0.0007 | 1.5989 ± 0.0021 | 0.2557 ± 0.0002 | 0.8934 ± 0.0003 | 0.7215 ± 0.0027 | 63.84 ± 0.02 | 0.07863 ± 0.00099 |
| LDL-LRR | 0.1806 ± 0.0007 | 1.5990 ± 0.0022 | 0.2549 ± 0.0005 | 0.8938 ± 0.0004 | 0.7229 ± 0.0019 | 63.95 ± 0.04 | 0.07846 ± 0.00096 |
| LDL-DVS | 0.1807 ± 0.0007 | 1.5991 ± 0.0022 | 0.2549 ± 0.0005 | 0.8938 ± 0.0004 | 0.7228 ± 0.0019 | 63.94 ± 0.04 | **0.07842 ± 0.00096** |
| Full FDIL | **0.1801 ± 0.0011** | **1.5897 ± 0.0013** | **0.2514 ± 0.0007** | **0.8955 ± 0.0008** | **0.7236 ± 0.0007** | **64.41 ± 0.10** | 0.07869 ± 0.00039 |

δ-LDL improves only Clark by a negligible amount and is worse on the other six
metrics. LDL-LRR is numerically indistinguishable from baseline. Full FDIL is
clearly strongest on six metrics, while LDL-DVS retains the best DVSE.

## LDL-LRR loss-scale diagnostic

| Dataset | Train KLD | LRR term | `1e-3 × LRR` | Auxiliary / KLD | Active `diff>0.5` pairs/sample |
|---|---:|---:|---:|---:|---:|
| Flickr-LDL | 0.235248 | 0.906983 | 0.000907 | 0.386% | 4.16 / 64 |
| Twitter-LDL | 0.268384 | 1.044773 | 0.001045 | 0.389% | 4.18 / 64 |
| Emotion6 | 0.167481 | 0.489030 | 0.000489 | 0.292% | 1.86 / 49 |

The published default alpha makes the LRR contribution only 0.29%-0.39% of
the KLD scale. This explains why LDL-LRR nearly reproduces baseline, especially
on Emotion6 where very few class pairs exceed the method's 0.5 target-gap
threshold.

An `alpha=0` paired audit reproduces the corresponding seed-2026 baseline with
maximum metric discrepancies of `6.05e-8` (Flickr), `4.75e-8` (Twitter), and
`1.50e-7` (Emotion6), confirming matched initialization and data order.

## Conclusion

1. δ-LDL is worth retaining as a strong, simple Twitter-LDL baseline, but its
   benefit does not transfer consistently to Flickr or Emotion6.
2. Default LDL-LRR adds effectively no value on strong CLIP features; a larger
   alpha or softer pair-selection rule would be a new tuned variant rather than
   a faithful default reproduction.
3. Across the three datasets, full FDIL produces larger changes than all three
   conventional LDL objectives, although its gains concentrate on different
   metrics and it is not universally best.

## Artifacts

- `logs/analysis/flickrldl_delta_lrr_clip_20260821/summary.json`
- `logs/analysis/twitterldl_delta_lrr_clip_20260821/summary.json`
- `logs/analysis/emotion6_delta_lrr_clip_20260821/summary.json`
- Each directory contains six checkpoints and six test-prediction files.
- Every summary metric was independently recomputed from saved predictions with
  zero numerical discrepancy.
