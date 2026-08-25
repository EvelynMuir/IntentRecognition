#!/usr/bin/env python3
"""Nested cross-validation for the EMOTIC SLR-C hyper-parameters.

`sweep_emotic_slrc.py` picked (prior-norm, top-K, alpha) by the mean VALIDATION mAP
pooled over all 25 (seed, fold) runs.  That is optimistic: in this pool protocol fold
k's validation split is fold k+1's TEST split, so the pooled criterion is informed by
rows that later serve as test data.

This script removes that optimism.  Inside every outer fold the selection uses only
that fold's TRAIN rows, split again into `--inner-folds` inner folds:

    inner-train -> train a baseline student
    inner-val   -> score the whole config grid, average over inner folds -> pick config
    outer test  -> apply the picked config to the *outer* baseline student (cached)

Nothing from the outer val/test rows enters the selection.  Two references are
reported alongside:

    oracle_pooled : the single config chosen by the pooled sweep (optimistic)
    outer_val     : per-fold selection on the outer val split (isolates how much of
                    the optimism came from pooling across folds)

Usage:
  python scripts/nested_cv_emotic_slrc.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.analyze_privileged_distillation as P  # noqa: E402
import scripts.analyze_distillation_pool_kfold as K  # noqa: E402
import scripts.slrc_prior_norm as N  # noqa: E402

SWEEP_DIR = "logs/analysis/emotic_slrc_sweep_20260819"
OUT_DIR = "logs/analysis/emotic_slrc_nested_cv_20260819"

# the full grid the pooled sweep searched, so nested selection is not handed a
# pre-narrowed space discovered by the leaky criterion
NORMS = N.NORMS
TOPKS = [1, 2, 3, 5, 8, 10, 15, 26]
ALPHAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
          0.13, 0.15, 0.17, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
ORACLE = ("zscore", 26, 0.09)  # config the pooled sweep selected


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default="logs/analysis/emotic_clip_dual_cache_full_20260323/_cache")
    p.add_argument("--vlm-dir", type=str, default="logs/analysis/emotic_vlm_20260323")
    p.add_argument("--text-suffix", type=str, default="_rationale_baseline_pred_bge_features.npz")
    p.add_argument("--sweep-dir", type=str, default=SWEEP_DIR)
    p.add_argument("--output-dir", type=str, default=OUT_DIR)
    p.add_argument("--seeds", type=str, default="20260625,20260626,20260627,20260628,20260629")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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
    return p.parse_args()


# --------------------------------------------------------------------------- #
# fast exact mAP (vectorised equivalent of analyze_privileged_distillation.compute_mAP)
# --------------------------------------------------------------------------- #
def fast_mAP(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    aps = np.zeros(scores.shape[1], dtype=np.float64)
    for c in range(scores.shape[1]):
        order = np.argsort(-scores[:, c])
        lab = targets[order, c] > 0
        true_num = float(lab.sum())
        if true_num == 0:
            continue
        tp = np.cumsum(lab, dtype=np.float64)
        fp = np.cumsum(~lab, dtype=np.float64)
        rec = tp / true_num
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        ch = np.where(mrec[1:] != mrec[:-1])[0]
        aps[c] = np.sum((mrec[ch + 1] - mrec[ch]) * mpre[ch + 1]) * 100.0
    return float(aps.mean())


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


def _select(logits_by_inner: List[Tuple[np.ndarray, np.ndarray]],
            prior: np.ndarray,
            inner_fit_idx: List[np.ndarray],
            inner_val_idx: List[np.ndarray],
            labels: np.ndarray) -> Tuple[Tuple[str, int, float], float]:
    """Pick the grid config with the best mean inner-val mAP."""
    # pre-normalise per inner fold (class-wise modes fit on that inner fold's train rows)
    pn: Dict[Tuple[str, int], np.ndarray] = {}
    for m in NORMS:
        for j, fit in enumerate(inner_fit_idx):
            pn[(m, j)] = N.norm_prior(prior, m, fit_idx=fit if m in N.FOLD_FITTED else None)

    best_cfg, best_score = None, -np.inf
    for norm in NORMS:
        for topk in TOPKS:
            for alpha in ALPHAS:
                acc = 0.0
                for j, (_, val_logits) in enumerate(logits_by_inner):
                    vi = inner_val_idx[j]
                    s = N.apply_slr(val_logits, pn[(norm, j)][vi], topk, alpha)
                    acc += fast_mAP(P._sigmoid_np(s), labels[vi])
                acc /= len(logits_by_inner)
                if acc > best_score + 1e-12:
                    best_score, best_cfg = acc, (norm, topk, alpha)
    return best_cfg, float(best_score)


def main() -> None:
    args = _parse_args()
    device = P._resolve_device(args.device)
    out = Path(args.output_dir)
    (out / "inner").mkdir(parents=True, exist_ok=True)
    sweep = Path(args.sweep_dir)

    pool = K._load_pool(args)
    labels = pool["labels"]
    prior = np.load(sweep / "semantic_prior.npy")
    image_dim = int(pool["image_features"].shape[1])
    num_classes = int(labels.shape[1])
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    # verify the fast mAP against the reference implementation
    z0 = np.load(sweep / "logits" / f"seed_{seeds[0]}_fold_0.npz")
    ref = P.compute_mAP(P._sigmoid_np(z0["test_logits"]), labels[z0["test_idx"]])
    got = fast_mAP(P._sigmoid_np(z0["test_logits"]), labels[z0["test_idx"]])
    # tolerance covers float accumulation order only (observed max 8e-6 mAP points)
    assert abs(ref - got) < 1e-3, f"fast_mAP mismatch: {ref} vs {got}"
    print(f"[nested] fast_mAP verified against compute_mAP ({ref:.6f})", flush=True)

    rows: List[Dict[str, Any]] = []
    per_fold: Dict[str, List[Dict[str, float]]] = {"baseline": [], "nested": [], "oracle_pooled": [], "outer_val": []}
    picks: List[Dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        fold_assign = np.array_split(rng.permutation(int(labels.shape[0])), int(args.folds))
        targs = _targs(args, seed)

        for k in range(int(args.folds)):
            cache = np.load(sweep / "logits" / f"seed_{seed}_fold_{k}.npz")
            train_idx, val_idx, test_idx = cache["train_idx"], cache["val_idx"], cache["test_idx"]
            out_val_logits, out_test_logits = cache["val_logits"], cache["test_logits"]

            inner_path = out / "inner" / f"seed_{seed}_fold_{k}.json"
            if inner_path.exists():
                saved = json.loads(inner_path.read_text(encoding="utf-8"))
                cfg = (saved["norm"], int(saved["topk"]), float(saved["alpha"]))
                inner_score = float(saved["inner_val_mAP"])
                print(f"[nested] seed {seed} fold {k}: resumed pick {cfg}", flush=True)
            else:
                # ---- inner CV inside the outer TRAIN rows only ----
                irng = np.random.RandomState(int(seed) + 7919 * (k + 1))
                inner_parts = np.array_split(irng.permutation(train_idx.size), int(args.inner_folds))
                logits_by_inner, fit_idxs, val_idxs = [], [], []
                for j in range(int(args.inner_folds)):
                    iv = train_idx[inner_parts[j]]
                    it = train_idx[np.concatenate([inner_parts[x] for x in range(int(args.inner_folds)) if x != j])]
                    P._set_component_seed(int(seed), offset=100 * k + 10 + j)
                    model = P.StudentMLP(image_dim, int(args.student_hidden_dim), num_classes,
                                         float(args.dropout), int(args.feature_proj_dim)).to(device)
                    agree = np.ones(it.size, dtype=np.float32)
                    ds = P.StudentDataset(pool["image_features"][it], labels[it], agree,
                                          pool["soft_labels"][it], np.zeros_like(labels[it]))
                    P._train_student("baseline", model, ds,
                                     pool["image_features"][iv], labels[iv],
                                     pool["image_features"][iv], labels[iv], device, targs)
                    lv = P._logit_np(P._predict_student(model, pool["image_features"][iv], device, int(args.batch_size)))
                    logits_by_inner.append((it, lv))
                    fit_idxs.append(it)
                    val_idxs.append(iv)
                cfg, inner_score = _select(logits_by_inner, prior, fit_idxs, val_idxs, labels)
                inner_path.write_text(json.dumps(
                    {"seed": int(seed), "fold": int(k), "norm": cfg[0], "topk": int(cfg[1]),
                     "alpha": float(cfg[2]), "inner_val_mAP": inner_score}, indent=2), encoding="utf-8")
                print(f"[nested] seed {seed} fold {k}: picked {cfg} (inner val mAP {inner_score:.4f})", flush=True)

            # ---- per-fold selection on the OUTER val (reference) ----
            best_ov, best_ov_score = None, -np.inf
            for norm in NORMS:
                pn_v = N.norm_prior(prior, norm, fit_idx=train_idx if norm in N.FOLD_FITTED else None)
                for topk in TOPKS:
                    for alpha in ALPHAS:
                        s = N.apply_slr(out_val_logits, pn_v[val_idx], topk, alpha)
                        sc = fast_mAP(P._sigmoid_np(s), labels[val_idx])
                        if sc > best_ov_score + 1e-12:
                            best_ov_score, best_ov = sc, (norm, topk, alpha)

            picks.append({"seed": int(seed), "fold": int(k),
                          "nested_norm": cfg[0], "nested_topk": cfg[1], "nested_alpha": cfg[2],
                          "outer_val_norm": best_ov[0], "outer_val_topk": best_ov[1], "outer_val_alpha": best_ov[2]})

            # ---- evaluate every variant on the outer test rows ----
            def bundle_of(config) -> Dict[str, float]:
                if config is None:
                    vl, tl = out_val_logits, out_test_logits
                else:
                    norm, topk, alpha = config
                    pnn = N.norm_prior(prior, norm, fit_idx=train_idx if norm in N.FOLD_FITTED else None)
                    vl = N.apply_slr(out_val_logits, pnn[val_idx], topk, alpha)
                    tl = N.apply_slr(out_test_logits, pnn[test_idx], topk, alpha)
                b = P._evaluate_score_bundle(
                    val_scores=P._sigmoid_np(vl), val_targets=labels[val_idx],
                    test_scores=P._sigmoid_np(tl), test_targets=labels[test_idx])
                return K._metrics_of(b)

            for name, config in [("baseline", None), ("nested", cfg),
                                 ("oracle_pooled", ORACLE), ("outer_val", best_ov)]:
                m = bundle_of(config)
                m.update({"seed": int(seed), "fold": int(k)})
                per_fold[name].append(m)
            print(f"[nested] seed {seed} fold {k}: base {per_fold['baseline'][-1]['mAP']:.4f} | "
                  f"nested {per_fold['nested'][-1]['mAP']:.4f} | oracle {per_fold['oracle_pooled'][-1]['mAP']:.4f} | "
                  f"outerval {per_fold['outer_val'][-1]['mAP']:.4f}", flush=True)

    # ---------------- aggregate ---------------- #
    for name, pf in per_fold.items():
        row: Dict[str, Any] = {"variant": name, "observations": len(pf)}
        for key in K.METRIC_KEYS:
            arr = np.asarray([r[key] for r in pf], dtype=np.float64)
            row[f"{key}_mean"] = round(float(arr.mean()), 4)
            row[f"{key}_std"] = round(float(arr.std(ddof=0)), 4)
        rows.append(row)
    with (out / "nested_cv_comparison.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    sig_rows = []
    for name in ("nested", "oracle_pooled", "outer_val"):
        for key in K.METRIC_KEYS:
            d = np.asarray([per_fold[name][i][key] - per_fold["baseline"][i][key]
                            for i in range(len(per_fold[name]))], dtype=np.float64)
            st = K._paired_test(d)
            st.update({"contrast": f"{name}-vs-baseline", "metric": key})
            sig_rows.append(st)
    keys = ["contrast", "metric"] + sorted({k for r in sig_rows for k in r} - {"contrast", "metric"})
    with (out / "nested_cv_significance.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(sig_rows)
    with (out / "nested_cv_picks.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(picks[0].keys())); w.writeheader(); w.writerows(picks)
    (out / "nested_cv_per_fold.json").write_text(json.dumps(per_fold, indent=2), encoding="utf-8")

    print("\n=== nested CV summary (25 outer folds) ===")
    for r in rows:
        print(f"  {r['variant']:>14}: mAP {r['mAP_mean']:.4f} macro {r['macro_mean']:.4f} "
              f"micro {r['micro_mean']:.4f} hard {r['hard_mean']:.4f}")
    print("\n=== configs picked by the inner CV ===")
    for c, n in Counter((p["nested_norm"], p["nested_topk"], p["nested_alpha"]) for p in picks).most_common():
        print(f"  {c}: {n}/25")
    print("\n=== configs picked on the outer val ===")
    for c, n in Counter((p["outer_val_norm"], p["outer_val_topk"], p["outer_val_alpha"]) for p in picks).most_common():
        print(f"  {c}: {n}/25")
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
