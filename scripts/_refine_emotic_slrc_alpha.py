#!/usr/bin/env python3
"""Fine alpha refinement around the SLR-C optimum found by sweep_emotic_slrc.py.

The coarse grid peaked at the low edge (alpha ~0.05-0.1, K = all classes), so this
re-scores a fine alpha ladder from the same cached baseline logits.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import scripts.analyze_privileged_distillation as P  # noqa: E402
import scripts.slrc_prior_norm as N  # noqa: E402

ROOT = Path("logs/analysis/emotic_slrc_sweep_20260819")
SEEDS = [20260625, 20260626, 20260627, 20260628, 20260629]
NORMS = ["zscore", "class_then_sample_zscore"]
TOPKS = [15, 26]
ALPHAS = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.15, 0.17]

prior = np.load(ROOT / "semantic_prior.npy")
labels = np.load(ROOT / "pool_labels.npz")["labels"]

folds = []
for seed in SEEDS:
    for k in range(5):
        z = np.load(ROOT / "logits" / f"seed_{seed}_fold_{k}.npz")
        folds.append({"seed": seed, "fold": k, "train_idx": z["train_idx"],
                      "val_idx": z["val_idx"], "test_idx": z["test_idx"],
                      "val_logits": z["val_logits"], "test_logits": z["test_logits"]})

pn = {}
for m in NORMS:
    if m in N.FOLD_FITTED:
        for i, f in enumerate(folds):
            pn[(m, i)] = N.norm_prior(prior, m, fit_idx=f["train_idx"])
    else:
        pn[m] = N.norm_prior(prior, m)
get = lambda m, i: pn[(m, i)] if m in N.FOLD_FITTED else pn[m]

bv = np.asarray([P.compute_mAP(P._sigmoid_np(f["val_logits"]), labels[f["val_idx"]]) for f in folds])
bt = np.asarray([P.compute_mAP(P._sigmoid_np(f["test_logits"]), labels[f["test_idx"]]) for f in folds])
print(f"baseline val {bv.mean():.4f} test {bt.mean():.4f}", flush=True)

rows = []
for m in NORMS:
    for K in TOPKS:
        for a in ALPHAS:
            v, t = [], []
            for i, f in enumerate(folds):
                p = get(m, i)
                v.append(P.compute_mAP(P._sigmoid_np(N.apply_slr(f["val_logits"], p[f["val_idx"]], K, a)), labels[f["val_idx"]]))
                t.append(P.compute_mAP(P._sigmoid_np(N.apply_slr(f["test_logits"], p[f["test_idx"]], K, a)), labels[f["test_idx"]]))
            v, t = np.asarray(v), np.asarray(t)
            rows.append({"prior_norm": m, "topk": K, "alpha": a,
                         "val_mAP_mean": round(float(v.mean()), 4),
                         "val_delta_vs_baseline": round(float((v - bv).mean()), 4),
                         "test_mAP_mean": round(float(t.mean()), 4),
                         "test_delta_vs_baseline": round(float((t - bt).mean()), 4),
                         "val_win_folds": int((v > bv).sum()),
                         "test_win_folds": int((t > bt).sum())})
            print(f"{m} K={K} a={a}: val d{rows[-1]['val_delta_vs_baseline']:+.4f} "
                  f"test d{rows[-1]['test_delta_vs_baseline']:+.4f} win {rows[-1]['val_win_folds']}/25", flush=True)

rows.sort(key=lambda r: -r["val_mAP_mean"])
out = ROOT / "slrc_grid_alpha_refine.csv"
with out.open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {out}")
