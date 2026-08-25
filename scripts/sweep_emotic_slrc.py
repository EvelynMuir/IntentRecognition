#!/usr/bin/env python3
"""Hyper-parameter sweep for SLR-C on the EMOTIC val+test k-fold pool.

Two stages:

  Stage 1 (`--stage cache`, GPU): reproduces the exact fold splits and the exact
  baseline students of `analyze_distillation_pool_kfold.py` (same RNG offsets), and
  caches the per-fold train/val/test baseline *logits* to disk.

  Stage 2 (`--stage sweep`, CPU): SLR-C is a post-hoc transform of those logits, so
  the whole (prior-normalisation x top-k x alpha) grid can be scored for free from
  the cache.  Configs are ranked by mean VALIDATION mAP (never test), and only the
  finalists get the expensive threshold-fitted F1 bundle + paired tests on test.

Usage:
  python scripts/sweep_emotic_slrc.py --stage cache --device cuda
  python scripts/sweep_emotic_slrc.py --stage sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.analyze_privileged_distillation as P  # noqa: E402
import scripts.analyze_distillation_slrc as S  # noqa: E402
import scripts.analyze_distillation_pool_kfold as K  # noqa: E402

DEFAULT_CACHE = "logs/analysis/emotic_clip_dual_cache_full_20260323/_cache"
DEFAULT_VLM = "logs/analysis/emotic_vlm_20260323"
DEFAULT_DESC = "../Emotic/emotion_description_gemini.json"
DEFAULT_OUT = "logs/analysis/emotic_slrc_sweep_20260819"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["cache", "sweep"], required=True)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE)
    p.add_argument("--vlm-dir", type=str, default=DEFAULT_VLM)
    p.add_argument("--emotion-description-file", type=str, default=DEFAULT_DESC)
    p.add_argument("--semantic-prior-cache", type=str, default=None)
    p.add_argument("--text-suffix", type=str, default="_rationale_baseline_pred_bge_features.npz")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--seeds", type=str, default="20260625,20260626,20260627,20260628,20260629")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # student/trainer knobs -- must match analyze_distillation_pool_kfold defaults
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--student-hidden-dim", type=int, default=768)
    p.add_argument("--teacher-hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--feature-proj-dim", type=int, default=256)
    p.add_argument("--student-agreement-pool", type=str, default="mean")
    p.add_argument("--n-finalists", type=int, default=4)
    return p.parse_args()


def _targs(a: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=int(a.batch_size), max_epochs=int(a.max_epochs), patience=int(a.patience),
        lr=float(a.lr), weight_decay=float(a.weight_decay), dropout=float(a.dropout),
        temperature=float(a.temperature), teacher_hidden_dim=int(a.teacher_hidden_dim),
        teacher_input_mode="text_only", oof_folds=3, seed=int(seed),
        standard_kd_weight=1.0, dynamic_kd_weight=1.0, dynamic_kd_variant="sample_inverse",
        dynamic_gate_alpha=0.3, dynamic_gate_beta=0.7, entropy_gate_lambda=1.0,
        feature_distill_mode="none", feature_distill_weight=0.0, feature_distill_temperature=0.1,
    )


def _folds_of(seed: int, n: int, folds: int) -> List[np.ndarray]:
    rng = np.random.RandomState(int(seed))
    return np.array_split(rng.permutation(n), int(folds))


# --------------------------------------------------------------------------- #
# Stage 1: cache baseline logits
# --------------------------------------------------------------------------- #
def stage_cache(args: argparse.Namespace) -> None:
    device = P._resolve_device(args.device)
    out = Path(args.output_dir) / "logits"
    out.mkdir(parents=True, exist_ok=True)

    pool = K._load_pool(args)
    prior = K._build_semantic_prior(pool, args, device)
    np.save(Path(args.output_dir) / "semantic_prior.npy", prior.astype(np.float32))
    np.savez_compressed(
        Path(args.output_dir) / "pool_labels.npz",
        labels=pool["labels"].astype(np.float32),
        soft_labels=pool["soft_labels"].astype(np.float32),
    )

    n = int(pool["labels"].shape[0])
    num_classes = int(pool["labels"].shape[1])
    image_dim = int(pool["image_features"].shape[1])
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    for seed in seeds:
        assign = _folds_of(seed, n, args.folds)
        targs = _targs(args, seed)
        for k in range(int(args.folds)):
            path = out / f"seed_{seed}_fold_{k}.npz"
            if path.exists():
                print(f"[cache] skip existing {path.name}")
                continue
            test_idx = np.asarray(assign[k], dtype=np.int64)
            val_idx = np.asarray(assign[(k + 1) % int(args.folds)], dtype=np.int64)
            train_idx = np.setdiff1d(
                np.arange(n, dtype=np.int64), np.concatenate([test_idx, val_idx]), assume_unique=False
            )
            tr_img, tr_lab, tr_soft = pool["image_features"][train_idx], pool["labels"][train_idx], pool["soft_labels"][train_idx]
            va_img, va_lab = pool["image_features"][val_idx], pool["labels"][val_idx]
            te_img, te_lab = pool["image_features"][test_idx], pool["labels"][test_idx]

            # identical seeding to analyze_distillation_pool_kfold (baseline == offset 1)
            P._set_component_seed(int(seed), offset=10 * k + 1)
            model = P.StudentMLP(image_dim, int(args.student_hidden_dim), num_classes,
                                 float(args.dropout), int(args.feature_proj_dim)).to(device)
            agreement = P._compute_sample_agreement(tr_lab, tr_soft, mode=str(args.student_agreement_pool))
            ds = P.StudentDataset(tr_img, tr_lab, np.ones_like(agreement), tr_soft, np.zeros_like(tr_lab))
            res = P._train_student("baseline", model, ds, va_img, va_lab, te_img, te_lab, device, targs)
            base_test_mAP = float(res["bundle"]["classwise"]["test"]["mAP"])

            lg = lambda x: P._logit_np(P._predict_student(model, x, device, int(args.batch_size)))
            np.savez_compressed(
                path,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                train_logits=lg(tr_img).astype(np.float32),
                val_logits=lg(va_img).astype(np.float32),
                test_logits=lg(te_img).astype(np.float32),
                baseline_test_mAP=np.asarray(base_test_mAP, dtype=np.float64),
            )
            print(f"[cache] seed {seed} fold {k}: baseline test mAP {base_test_mAP:.4f} -> {path.name}")


# --------------------------------------------------------------------------- #
# Stage 2: sweep
# --------------------------------------------------------------------------- #
def _norm_prior(prior: np.ndarray, mode: str, fit_idx: np.ndarray | None = None) -> np.ndarray:
    """Normalise the semantic prior.  `fit_idx` restricts the *class-wise* statistics
    to the fold's train rows so nothing is fitted on val/test rows."""
    prior = np.asarray(prior, dtype=np.float32)
    fit = prior if fit_idx is None else prior[np.asarray(fit_idx, dtype=np.int64)]
    if mode == "zscore":                      # current SLR-C behaviour
        return S._normalize_scores_per_sample(prior)
    if mode == "minmax":
        lo = prior.min(axis=1, keepdims=True)
        hi = prior.max(axis=1, keepdims=True)
        return (prior - lo) / np.maximum(hi - lo, 1e-6)
    if mode == "rank":                        # scale-free: 0..1 by within-sample rank
        order = np.argsort(np.argsort(prior, axis=1), axis=1).astype(np.float32)
        return order / max(prior.shape[1] - 1, 1)
    if mode == "softmax":
        z = prior - prior.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)) * prior.shape[1]
    if mode == "centered_softmax":
        z = prior - prior.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
        return (p - p.mean(axis=1, keepdims=True)) * prior.shape[1]
    if mode == "class_zscore":
        # de-bias each class over the sample axis (CLIP text similarities carry a
        # strong per-class offset that per-sample z-scoring cannot remove)
        mu = fit.mean(axis=0, keepdims=True)
        sd = np.maximum(fit.std(axis=0, keepdims=True), 1e-6)
        return (prior - mu) / sd
    if mode == "class_rank":
        # empirical CDF per class, fitted on train rows, applied to all rows
        out = np.empty_like(prior)
        for c in range(prior.shape[1]):
            ref = np.sort(fit[:, c])
            out[:, c] = np.searchsorted(ref, prior[:, c], side="left") / max(ref.size, 1)
        return (out - 0.5) * 2.0
    if mode == "class_then_sample_zscore":
        mu = fit.mean(axis=0, keepdims=True)
        sd = np.maximum(fit.std(axis=0, keepdims=True), 1e-6)
        return S._normalize_scores_per_sample((prior - mu) / sd)
    raise ValueError(mode)


def _apply(base: np.ndarray, prior_n: np.ndarray, topk: int, alpha: float) -> np.ndarray:
    """SLR-C additive reconstruction on the top-k baseline classes (prior pre-normalised)."""
    base = np.asarray(base, dtype=np.float32)
    out = base.copy()
    c = base.shape[1]
    topk = max(1, min(int(topk), c))
    if topk >= c:
        return base + float(alpha) * prior_n
    idx = np.argpartition(-base, kth=topk - 1, axis=1)[:, :topk]
    rows = np.arange(base.shape[0])[:, None]
    out[rows, idx] = base[rows, idx] + float(alpha) * prior_n[rows, idx]
    return out


NORMS = ["zscore", "rank", "minmax", "centered_softmax",
         "class_zscore", "class_rank", "class_then_sample_zscore"]
# modes whose statistics are fitted per fold on the train rows
FOLD_FITTED = {"class_zscore", "class_rank", "class_then_sample_zscore"}
TOPKS = [1, 2, 3, 5, 8, 10, 15, 26]
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]


def stage_sweep(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    prior = np.load(root / "semantic_prior.npy")
    labels = np.load(root / "pool_labels.npz")["labels"]
    soft = np.load(root / "pool_labels.npz")["soft_labels"]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    folds: List[Dict[str, Any]] = []
    for seed in seeds:
        for k in range(int(args.folds)):
            path = root / "logits" / f"seed_{seed}_fold_{k}.npz"
            if not path.exists():
                raise SystemExit(f"missing cache {path}; run --stage cache first")
            z = np.load(path)
            folds.append({
                "seed": seed, "fold": k,
                "train_idx": z["train_idx"], "val_idx": z["val_idx"], "test_idx": z["test_idx"],
                "val_logits": z["val_logits"], "test_logits": z["test_logits"],
                "baseline_test_mAP": float(z["baseline_test_mAP"]),
            })
    print(f"[sweep] loaded {len(folds)} (seed,fold) baseline caches")

    # pre-normalise the prior: sample-wise modes once, fold-fitted modes per fold
    prior_n = {m: _norm_prior(prior, m) for m in NORMS if m not in FOLD_FITTED}
    for m in FOLD_FITTED:
        for i, f in enumerate(folds):
            prior_n[(m, i)] = _norm_prior(prior, m, fit_idx=f["train_idx"])

    def pn_of(mode: str, i: int) -> np.ndarray:
        return prior_n[(mode, i)] if mode in FOLD_FITTED else prior_n[mode]

    # --- baseline reference (alpha = 0) ---
    base_val, base_test = [], []
    for f in folds:
        base_val.append(P.compute_mAP(P._sigmoid_np(f["val_logits"]), labels[f["val_idx"]]))
        base_test.append(P.compute_mAP(P._sigmoid_np(f["test_logits"]), labels[f["test_idx"]]))
    base_val = np.asarray(base_val, dtype=np.float64)
    base_test = np.asarray(base_test, dtype=np.float64)
    print(f"[sweep] baseline: val mAP {base_val.mean():.4f} | test mAP {base_test.mean():.4f}")

    # diagnostic: how informative is the semantic prior on its own?
    for m in NORMS:
        pv = [P.compute_mAP(pn_of(m, i)[f["test_idx"]], labels[f["test_idx"]]) for i, f in enumerate(folds)]
        print(f"[sweep] prior-alone ({m}) test mAP {np.mean(pv):.4f}")

    rows: List[Dict[str, Any]] = []
    for norm in NORMS:
        for topk in TOPKS:
            for alpha in ALPHAS:
                v, t = [], []
                for i, f in enumerate(folds):
                    pn = pn_of(norm, i)
                    vs = P._sigmoid_np(_apply(f["val_logits"], pn[f["val_idx"]], topk, alpha))
                    ts = P._sigmoid_np(_apply(f["test_logits"], pn[f["test_idx"]], topk, alpha))
                    v.append(P.compute_mAP(vs, labels[f["val_idx"]]))
                    t.append(P.compute_mAP(ts, labels[f["test_idx"]]))
                v = np.asarray(v, dtype=np.float64)
                t = np.asarray(t, dtype=np.float64)
                rows.append({
                    "prior_norm": norm, "topk": topk, "alpha": alpha,
                    "val_mAP_mean": round(float(v.mean()), 4),
                    "val_delta_vs_baseline": round(float((v - base_val).mean()), 4),
                    "test_mAP_mean": round(float(t.mean()), 4),
                    "test_delta_vs_baseline": round(float((t - base_test).mean()), 4),
                    "test_mAP_std": round(float(t.std(ddof=0)), 4),
                    "val_win_folds": int((v > base_val).sum()),
                })
            print(f"[sweep] {norm} topk={topk} done")

    rows.sort(key=lambda r: -r["val_mAP_mean"])
    grid_path = root / "slrc_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {grid_path} ({len(rows)} configs)")

    print("\n[sweep] top-15 configs by VAL mAP")
    print(f"{'norm':>17} {'k':>3} {'alpha':>5} {'val':>8} {'dVal':>7} {'test':>8} {'dTest':>7} {'winF':>5}")
    print(f"{'baseline':>17} {'-':>3} {'-':>5} {base_val.mean():8.4f} {0.0:7.4f} {base_test.mean():8.4f} {0.0:7.4f} {'-':>5}")
    for r in rows[:15]:
        print(f"{r['prior_norm']:>17} {r['topk']:>3} {r['alpha']:>5} {r['val_mAP_mean']:8.4f} "
              f"{r['val_delta_vs_baseline']:7.4f} {r['test_mAP_mean']:8.4f} "
              f"{r['test_delta_vs_baseline']:7.4f} {r['val_win_folds']:>5}")

    # ------------- finalists: full threshold-fitted bundle + paired tests ------------- #
    finalists = [{"prior_norm": "zscore", "topk": 10, "alpha": 0.3, "tag": "published_default"}]
    for r in rows[: int(args.n_finalists)]:
        cfg = {"prior_norm": r["prior_norm"], "topk": r["topk"], "alpha": r["alpha"], "tag": "val_selected"}
        if not any(c["prior_norm"] == cfg["prior_norm"] and c["topk"] == cfg["topk"]
                   and abs(c["alpha"] - cfg["alpha"]) < 1e-9 for c in finalists):
            finalists.append(cfg)

    final_rows: List[Dict[str, Any]] = []
    per_fold_store: Dict[str, List[Dict[str, float]]] = {}
    for cfg in [{"prior_norm": "zscore", "topk": 0, "alpha": 0.0, "tag": "baseline"}] + finalists:
        name = ("baseline" if cfg["tag"] == "baseline"
                else f"slrc_{cfg['prior_norm']}_k{cfg['topk']}_a{cfg['alpha']}")
        per_fold: List[Dict[str, float]] = []
        for i, f in enumerate(folds):
            pn = pn_of(cfg["prior_norm"], i)
            if cfg["tag"] == "baseline":
                vl, tl = f["val_logits"], f["test_logits"]
            else:
                vl = _apply(f["val_logits"], pn[f["val_idx"]], cfg["topk"], cfg["alpha"])
                tl = _apply(f["test_logits"], pn[f["test_idx"]], cfg["topk"], cfg["alpha"])
            bundle = P._evaluate_score_bundle(
                val_scores=P._sigmoid_np(vl), val_targets=labels[f["val_idx"]],
                test_scores=P._sigmoid_np(tl), test_targets=labels[f["test_idx"]],
            )
            m = K._metrics_of(bundle)
            m.update({"seed": f["seed"], "fold": f["fold"]})
            per_fold.append(m)
        per_fold_store[name] = per_fold
        agg: Dict[str, Any] = {"method": name, "tag": cfg["tag"], "prior_norm": cfg["prior_norm"],
                               "topk": cfg["topk"], "alpha": cfg["alpha"], "observations": len(per_fold)}
        for key in K.METRIC_KEYS:
            arr = np.asarray([r[key] for r in per_fold], dtype=np.float64)
            agg[f"{key}_mean"] = round(float(arr.mean()), 4)
            agg[f"{key}_std"] = round(float(arr.std(ddof=0)), 4)
        final_rows.append(agg)
        print(f"[final] {name}: mAP {agg['mAP_mean']:.4f} macro {agg['macro_mean']:.4f}")

    sig_rows: List[Dict[str, Any]] = []
    for name, per_fold in per_fold_store.items():
        if name == "baseline":
            continue
        for key in K.METRIC_KEYS:
            d = np.asarray([per_fold[i][key] - per_fold_store["baseline"][i][key]
                            for i in range(len(per_fold))], dtype=np.float64)
            st = K._paired_test(d)
            st.update({"contrast": f"{name}-vs-baseline", "metric": key})
            sig_rows.append(st)

    with (root / "slrc_finalists.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        w.writerows(final_rows)
    with (root / "slrc_finalists_significance.csv").open("w", newline="", encoding="utf-8") as h:
        keys = sorted({k for r in sig_rows for k in r.keys()})
        keys = ["contrast", "metric"] + [k for k in keys if k not in ("contrast", "metric")]
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        w.writerows(sig_rows)
    (root / "slrc_per_fold.json").write_text(json.dumps(per_fold_store, indent=2), encoding="utf-8")
    print(f"[sweep] wrote finalists + significance under {root}")


def main() -> None:
    args = _parse_args()
    if args.stage == "cache":
        stage_cache(args)
    else:
        stage_sweep(args)


if __name__ == "__main__":
    main()
