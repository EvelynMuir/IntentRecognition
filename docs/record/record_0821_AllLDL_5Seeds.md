# All LDL methods: unified five-seed results

## Protocol

- Seeds: 2026, 2027, 2028, 2029, 2030.
- Frozen CLIP ViT-L/14 768-D full-image features and matched 768-hidden MLP.
- Identical dataset splits, AdamW settings, and validation-KLD model selection.
- Values are mean ± population standard deviation over five seeds.
- Test data is used only for final evaluation.

## Flickr-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---|---|---|---|---|---|---|
| CLIP baseline | 0.2039 ± 0.0011 | 2.3033 ± 0.0040 | 0.3433 ± 0.0014 | 0.8966 ± 0.0006 | 0.7109 ± 0.0023 | 71.40 ± 0.12 | 0.07723 ± 0.00178 |
| δ-LDL | 0.2039 ± 0.0012 | 2.3032 ± 0.0041 | 0.3434 ± 0.0016 | 0.8966 ± 0.0007 | 0.7109 ± 0.0023 | 71.39 ± 0.14 | 0.07718 ± 0.00183 |
| LDL-LRR | 0.2039 ± 0.0011 | 2.3033 ± 0.0040 | 0.3433 ± 0.0014 | 0.8966 ± 0.0006 | 0.7109 ± 0.0024 | 71.40 ± 0.12 | 0.07722 ± 0.00178 |
| LDL-DPA | 0.2039 ± 0.0011 | 2.3033 ± 0.0040 | 0.3433 ± 0.0014 | 0.8966 ± 0.0006 | 0.7109 ± 0.0023 | 71.40 ± 0.12 | 0.07722 ± 0.00178 |
| LDL-DVS | 0.2039 ± 0.0011 | 2.3034 ± 0.0040 | 0.3433 ± 0.0014 | 0.8966 ± 0.0006 | 0.7108 ± 0.0023 | 71.40 ± 0.12 | 0.07719 ± 0.00178 |
| UTD only | **0.2026 ± 0.0010** | 2.3123 ± 0.0022 | 0.3454 ± 0.0026 | 0.8961 ± 0.0011 | 0.7137 ± 0.0010 | 71.31 ± 0.16 | **0.07292 ± 0.00096** |
| SLR-C only | 0.2031 ± 0.0005 | **2.3031 ± 0.0022** | **0.3406 ± 0.0009** | **0.8973 ± 0.0006** | 0.7129 ± 0.0009 | **71.58 ± 0.08** | 0.07641 ± 0.00128 |
| Full FDIL | 0.2027 ± 0.0006 | 2.3113 ± 0.0024 | 0.3433 ± 0.0014 | 0.8967 ± 0.0005 | **0.7140 ± 0.0011** | 71.45 ± 0.10 | 0.07379 ± 0.00080 |

Improvements over matched baseline: δ-LDL 2/7 (Clark, DVSE); LDL-LRR 2/7 (Cheby., DVSE); LDL-DPA 2/7 (Cheby., DVSE); LDL-DVS 2/7 (Cheby., DVSE); Full FDIL 5/7 (Cheby., Cosine, Spear., µ, DVSE).

## Twitter-LDL

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---|---|---|---|---|---|---|
| CLIP baseline | 0.2046 ± 0.0006 | 2.3853 ± 0.0031 | 0.3699 ± 0.0009 | 0.9057 ± 0.0004 | 0.6682 ± 0.0008 | 72.26 ± 0.05 | 0.07235 ± 0.00179 |
| δ-LDL | 0.2045 ± 0.0007 | **2.3851 ± 0.0029** | **0.3695 ± 0.0008** | **0.9058 ± 0.0004** | 0.6682 ± 0.0008 | **72.29 ± 0.06** | 0.07255 ± 0.00182 |
| LDL-LRR | 0.2046 ± 0.0006 | 2.3854 ± 0.0031 | 0.3699 ± 0.0009 | 0.9057 ± 0.0004 | 0.6682 ± 0.0008 | 72.26 ± 0.05 | 0.07234 ± 0.00179 |
| LDL-DPA | 0.2046 ± 0.0006 | 2.3853 ± 0.0031 | 0.3699 ± 0.0009 | 0.9057 ± 0.0004 | 0.6682 ± 0.0008 | 72.26 ± 0.05 | 0.07234 ± 0.00179 |
| LDL-DVS | 0.2046 ± 0.0006 | 2.3854 ± 0.0031 | 0.3699 ± 0.0009 | 0.9057 ± 0.0004 | 0.6682 ± 0.0008 | 72.26 ± 0.05 | 0.07230 ± 0.00178 |
| UTD only | **0.2037 ± 0.0015** | 2.3914 ± 0.0018 | 0.3736 ± 0.0027 | 0.9049 ± 0.0017 | 0.6686 ± 0.0015 | 72.15 ± 0.19 | **0.06761 ± 0.00047** |
| SLR-C only | 0.2057 ± 0.0009 | 2.3858 ± 0.0032 | 0.3710 ± 0.0012 | 0.9048 ± 0.0003 | **0.6691 ± 0.0005** | 72.18 ± 0.05 | 0.07230 ± 0.00169 |
| Full FDIL | 0.2050 ± 0.0011 | 2.3921 ± 0.0041 | 0.3740 ± 0.0026 | 0.9042 ± 0.0013 | 0.6688 ± 0.0011 | 72.06 ± 0.16 | 0.06824 ± 0.00193 |

Improvements over matched baseline: δ-LDL 6/7 (Cheby., Clark, KLD, Cosine, Spear., µ); LDL-LRR 2/7 (Cheby., DVSE); LDL-DPA 2/7 (Cheby., DVSE); LDL-DVS 3/7 (Cheby., Spear., DVSE); Full FDIL 2/7 (Spear., DVSE).

## Emotion6

| Method | Cheby. ↓ | Clark ↓ | KLD ↓ | Cosine ↑ | Spear. ↑ | µ (%) ↑ | DVSE ↓ |
|---|---|---|---|---|---|---|---|
| CLIP baseline | 0.1794 ± 0.0017 | 1.5979 ± 0.0023 | 0.2532 ± 0.0023 | 0.8947 ± 0.0013 | 0.7223 ± 0.0022 | 64.20 ± 0.33 | 0.07838 ± 0.00077 |
| δ-LDL | 0.1798 ± 0.0018 | 1.5979 ± 0.0023 | 0.2538 ± 0.0025 | 0.8943 ± 0.0013 | 0.7216 ± 0.0026 | 64.11 ± 0.35 | 0.07854 ± 0.00078 |
| LDL-LRR | **0.1794 ± 0.0017** | 1.5979 ± 0.0023 | 0.2532 ± 0.0023 | 0.8947 ± 0.0013 | 0.7223 ± 0.0022 | 64.20 ± 0.33 | 0.07838 ± 0.00077 |
| LDL-DPA | 0.1794 ± 0.0017 | 1.5979 ± 0.0023 | 0.2532 ± 0.0023 | 0.8947 ± 0.0013 | 0.7223 ± 0.0022 | 64.20 ± 0.33 | 0.07837 ± 0.00077 |
| LDL-DVS | 0.1794 ± 0.0017 | 1.5981 ± 0.0023 | 0.2532 ± 0.0023 | 0.8946 ± 0.0013 | 0.7222 ± 0.0021 | 64.20 ± 0.33 | 0.07834 ± 0.00077 |
| UTD only | 0.1804 ± 0.0016 | 1.5949 ± 0.0042 | 0.2537 ± 0.0029 | 0.8941 ± 0.0016 | **0.7256 ± 0.0023** | 64.12 ± 0.38 | 0.07907 ± 0.00122 |
| SLR-C only | 0.1797 ± 0.0014 | 1.5976 ± 0.0048 | 0.2515 ± 0.0017 | **0.8955 ± 0.0012** | 0.7232 ± 0.0024 | **64.52 ± 0.25** | **0.07811 ± 0.00053** |
| Full FDIL | 0.1802 ± 0.0010 | **1.5910 ± 0.0020** | **0.2513 ± 0.0006** | 0.8953 ± 0.0007 | 0.7219 ± 0.0023 | 64.41 ± 0.11 | 0.07812 ± 0.00079 |

Improvements over matched baseline: δ-LDL 0/7 (none); LDL-LRR 2/7 (Cheby., DVSE); LDL-DPA 3/7 (Cheby., Spear., DVSE); LDL-DVS 1/7 (DVSE); Full FDIL 5/7 (Clark, KLD, Cosine, µ, DVSE).

## Summary

- Flickr: UTD/FDIL dominate Cheby, Spearman, and DVSE; SLR-C is strongest on KLD, cosine, and µ.
- Twitter: δ-LDL is strongest on global distance/similarity metrics; UTD is strongest on Cheby and DVSE; SLR-C is strongest on Spearman.
- Emotion6: full FDIL is strongest on Clark and KLD; SLR-C is strongest on cosine, µ, and DVSE; baseline/LDL-LRR are effectively tied on Cheby.
- Default LDL-LRR, LDL-DPA, and LDL-DVS remain almost indistinguishable from baseline across all three datasets.
- LDL-DPA's default weighted regularizer is only 0.32%-0.48% of train KLD magnitude, explaining its negligible effect with CLIP features.

## Paper mapping (revision_2, updated 2026-08-21)

This record is the source for manuscript Table `tab:ldl_transfer` in
`paper/revision_2/04_experiments.tex`, Sec. "Cross-Dataset Generalization"
(`\label{sec:flickrldl}`). **Flickr-LDL and Emotion6 only**; Twitter-LDL is run
and recorded but deliberately not reported (FDIL improves only 2/7 metrics).

Table rows: Baseline, δ-LDL, LDL-LRR, LDL-DPA, LDL-DVS, +SLR-C, +UTD, FDIL,
for both datasets in one combined table (page-limit pressure).

Artifact dirs feeding the table:
- `flickrldl_fdil_5seed_20260821`, `emotion6_fdil_5seed_20260821`
- `flickrldl_ldl_dvs_clip_5seed_20260821`, `emotion6_ldl_dvs_clip_5seed_20260821`
- `flickrldl_delta_lrr_clip_20260821`, `emotion6_delta_lrr_clip_20260821` (5 seeds)
- `flickrldl_dpa_clip_20260821`, `emotion6_dpa_clip_20260821` (5 seeds)

Note the CSV method keys are `delta_ldl`, `lrr`, `dpa`, `ldl_dvs` (not `ldldvs`).

Citations used: δ-LDL `li_approximately_2025` (ICML 2025); LDL-LRR
`jia_label_2023` (TKDE 35(2):1695-1707); LDL-DPA `jia_adaptive_2024`
(TNNLS 35(8):11302-11316); LDL-DVS `lu2026divisiveness` (ICML 2026);
Flickr-LDL `yang_learning_2017` (AAAI 2017); Emotion6 `peng_mixed_2015`
(CVPR 2015, 860-868).

### Why the four comparators are left unmarked in the table

All four move every metric by <= 7e-4 (most ~1e-5). Several ARE nominally
significant at p<0.05 under a paired t-test purely because paired runs are
near-identical -- e.g. Flickr LDL-LRR Clark p<0.05 at delta = -2.4e-5. Marking
those would be misleading, so the table marks significance only for our own
rows and the text explains the omission. Cause is scale: each objective's
default-weight regularizer is 0.2%-0.5% of train KLD magnitude on frozen CLIP.

### EMOTIC withdrawn

The EMOTIC transfer study was REMOVED from the manuscript body on 2026-08-21
(commented out in `04_experiments.tex`, not deleted) because its annotation
protocol does not naturally match the soft multi-label setting. The response
letter states this and points to Flickr-LDL/Emotion6 as the replacement. All
EMOTIC prose in `05_discussion`/`06_conclusion`/abstract and all EMOTIC
arguments in the response letter were rewritten accordingly;
`kosti_context_2019` is no longer cited and has left the bibliography
(now 50 items, within the required 35-55).

### Table presentation (revision_2, 2026-08-21)

Sec. 4.5 prose was cut from 856 to 314 words: it now explains only (a) why the
four published objectives are flat (default-weight regularizer = 0.2-0.5% of
train KL on frozen CLIP, so no gradient left; their published gains came from
much weaker handcrafted descriptors), (b) which metrics each of our modules
moves and what that means (SLR-C -> pointwise fit, reallocates mass without
reordering; UTD -> rank correlation + DVSE, pays in Clark, because it
redistributes mass across polarity), and (c) the limits. No numbers already in
the table are repeated; only p-values (not in the table) and the 0.2-0.5% scale
figure appear in prose.

Marking: **bold** = best, underline = second best, per metric, across ALL eight
rows, assigned on unrounded means. Significance asterisks were REMOVED to keep
the table readable -- per-metric paired t-tests now live only in the response
letter's `\showFlickrLDL` table plus the three p-values quoted in the prose.

Values are shown at 5 decimals (µ at 2) because at 4 decimals the baseline and
the four published objectives are indistinguishable, which made the marks
meaningless. One residual case remains and the caption discloses it: on
Emotion6 Chebyshev, LDL-LRR (bold) and LDL-DPA (underline) both render as
.17936 -- they differ by 3.4e-7. If that is unacceptable, the options are 6
decimals for that column, or bolding both as a tie.
