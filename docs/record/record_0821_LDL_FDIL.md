# FDIL on Flickr-LDL and Twitter-LDL

## Protocol

- Frozen CLIP ViT-L/14 image features.
- Official split 1 is test-only; official fold 2 from the remaining data is validation.
- Best epochs are selected exclusively by validation KLD.
- Results are mean ± population standard deviation over seeds [2026, 2027, 2028, 2029, 2030].
- SLR-C uses the shared Gemini lexical/canonical/6-scenario prior, Top-5, alpha=0.3.
- UTD uses a training-only Qwen3-VL rationale teacher encoded by BGE-large and cross-fitted into OOF train predictions.
- `µ` is reported as a percentage. DVSE uses positive={amusement, contentment, excitement}, negative={anger, disgust, fear, sadness}, with awe excluded.

## Flickr-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---|---|---|---|---|---|---|
| CLIP baseline | 0.2039 ± 0.0011 | 2.3033 ± 0.0040 | 0.3433 ± 0.0014 | 0.8966 ± 0.0006 | 0.7109 ± 0.0023 | 71.40 ± 0.12 | 0.0772 ± 0.0018 |
| UTD only | **0.2026 ± 0.0010** | 2.3123 ± 0.0022 | 0.3454 ± 0.0026 | 0.8961 ± 0.0011 | 0.7137 ± 0.0010 | 71.31 ± 0.16 | **0.0729 ± 0.0010** |
| SLR-C only | 0.2031 ± 0.0005 | **2.3031 ± 0.0022** | **0.3406 ± 0.0009** | **0.8973 ± 0.0006** | 0.7129 ± 0.0009 | **71.58 ± 0.08** | 0.0764 ± 0.0013 |
| FDIL | 0.2027 ± 0.0006 | 2.3113 ± 0.0024 | 0.3433 ± 0.0014 | 0.8967 ± 0.0005 | **0.7140 ± 0.0011** | 71.45 ± 0.10 | 0.0738 ± 0.0008 |

## Twitter-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---|---|---|---|---|---|---|
| CLIP baseline | 0.2046 ± 0.0006 | **2.3853 ± 0.0031** | **0.3699 ± 0.0009** | **0.9057 ± 0.0004** | 0.6682 ± 0.0008 | **72.26 ± 0.05** | 0.0724 ± 0.0018 |
| UTD only | **0.2037 ± 0.0015** | 2.3914 ± 0.0018 | 0.3736 ± 0.0027 | 0.9049 ± 0.0017 | 0.6686 ± 0.0015 | 72.15 ± 0.19 | **0.0676 ± 0.0005** |
| SLR-C only | 0.2057 ± 0.0009 | 2.3858 ± 0.0032 | 0.3710 ± 0.0012 | 0.9048 ± 0.0003 | **0.6691 ± 0.0005** | 72.18 ± 0.05 | 0.0723 ± 0.0017 |
| FDIL | 0.2050 ± 0.0011 | 2.3921 ± 0.0041 | 0.3740 ± 0.0026 | 0.9042 ± 0.0013 | 0.6688 ± 0.0011 | 72.06 ± 0.16 | 0.0682 ± 0.0019 |

## Result interpretation

- Flickr-LDL: Full FDIL improves 5/7 metrics over the matched baseline; deltas (FDIL-baseline): Cheby. -0.0011, Clark +0.0079, KLD +0.0000, Cosine +0.0001, Spear. +0.0032, µ +0.05, DVSE -0.0034. SLR-C is strongest on Clark, KLD, cosine and µ, while full FDIL is strongest on Spearman and DVSE.
- Twitter-LDL: Full FDIL improves 2/7 metrics over the matched baseline; deltas (FDIL-baseline): Cheby. +0.0004, Clark +0.0068, KLD +0.0041, Cosine -0.0015, Spear. +0.0006, µ -0.19, DVSE -0.0041. UTD-only is strongest on Cheby and DVSE; full FDIL is strongest only on Spearman. The transfer therefore does not establish uniform dominance on Twitter-LDL.

## Artifact provenance

- Flickr summary: `/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/analysis/flickrldl_fdil_20260821/summary.json`
- Twitter summary: `/home/evelynmuir/lambda/projects/IntentRecognition/lightning-hydra/logs/analysis/twitterldl_fdil_20260821/summary.json`
- Shared SLR-C prior SHA-256: `a611835f47463492f2c4b457413006125d04c949ce8f1a4848af0dc24b746e22`
- Test predictions and per-seed checkpoints are stored next to each summary.
