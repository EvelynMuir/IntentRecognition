#!/usr/bin/env python3
"""Controlled k-fold distillation study on the EMOTIC val+test pool.

The standard EMOTIC train split is single-annotator, so it carries no per-sample
annotator agreement and the agreement-gated KD term is inert there (see
analyze_privileged_distillation.py / emotic_privileged_distillation runs, where
dynamic_gated_kd is byte-identical to the plain baseline).

This script instead pools the val+test splits -- the only EMOTIC partitions that
were multi-annotated -- where every sample DOES expose a fractional agreement
soft label. We run k-fold cross-validation entirely inside this pool, so label
construction is consistent across train/val/test folds and the gate can actually
engage at train time. This is a controlled ablation ("does gated KD help when
agreement is available?"), NOT a benchmark number -- the pool is ~10.6k persons
with union-style (denser) labels and is not comparable to published EMOTIC
results on the standard split.

It reuses the model/trainer/metric code from analyze_privileged_distillation.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.analyze_privileged_distillation as P  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="logs/analysis/emotic_clip_dual_cache_full_20260323/_cache",
    )
    parser.add_argument("--vlm-dir", type=str, default="logs/analysis/emotic_vlm_20260323")
    parser.add_argument(
        "--text-suffix",
        type=str,
        default="_rationale_baseline_pred_bge_features.npz",
        help="val/test text-feature npz filename suffix (prefixed by split name).",
    )
    parser.add_argument("--output-dir", type=str, default="logs/analysis/emotic_pool_kfold_20260625")
    parser.add_argument("--folds", type=int, default=5, help="Number of outer CV folds.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for multi-seed significance. Overrides --seed.",
    )
    # method hyper-parameters (mirror analyze_privileged_distillation.py defaults)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-hidden-dim", type=int, default=1024)
    parser.add_argument("--student-hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-target-mode", type=str, default="oof", choices=["in_sample", "oof"])
    parser.add_argument("--oof-folds", type=int, default=3, help="Inner folds for OOF KD targets.")
    parser.add_argument("--standard-kd-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-kd-variant", type=str, default="sample_inverse")
    parser.add_argument("--dynamic-gate-alpha", type=float, default=0.3)
    parser.add_argument("--dynamic-gate-beta", type=float, default=0.7)
    parser.add_argument("--entropy-gate-lambda", type=float, default=1.0)
    parser.add_argument("--feature-distill-mode", type=str, default="none")
    parser.add_argument("--feature-distill-weight", type=float, default=0.0)
    parser.add_argument("--feature-distill-temperature", type=float, default=0.1)
    parser.add_argument("--feature-proj-dim", type=int, default=256)
    parser.add_argument("--student-agreement-pool", type=str, default="mean", choices=["mean", "min"])
    parser.add_argument("--teacher-input-mode", type=str, default="text_only", choices=["text_only", "image_text"])
    return parser.parse_args()


def _trainer_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    """Namespace consumed by the reused _train_* / _compute_oof_* helpers."""
    return SimpleNamespace(
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        temperature=float(args.temperature),
        teacher_hidden_dim=int(args.teacher_hidden_dim),
        teacher_input_mode=str(args.teacher_input_mode),
        oof_folds=int(args.oof_folds),
        seed=int(seed),
        standard_kd_weight=float(args.standard_kd_weight),
        dynamic_kd_weight=float(args.dynamic_kd_weight),
        dynamic_kd_variant=str(args.dynamic_kd_variant),
        dynamic_gate_alpha=float(args.dynamic_gate_alpha),
        dynamic_gate_beta=float(args.dynamic_gate_beta),
        entropy_gate_lambda=float(args.entropy_gate_lambda),
        feature_distill_mode=str(args.feature_distill_mode),
        feature_distill_weight=float(args.feature_distill_weight),
        feature_distill_temperature=float(args.feature_distill_temperature),
    )


def _load_pool(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    cache_dir = Path(args.cache_dir)
    vlm_dir = Path(args.vlm_dir)
    feats, texts, labels, soft, ids = [], [], [], [], []
    for split in ("val", "test"):
        clip = P._load_cache_bundle(cache_dir / f"{split}_clip.npz")
        text = P._align_text_bundle_to_clip(
            P._load_text_bundle(vlm_dir / f"{split}{args.text_suffix}", required_keys=["image_ids", "features"]),
            clip,
        )
        if clip["image_ids"] != text["image_ids"]:
            raise RuntimeError(f"{split}: image order mismatch after text alignment.")
        feats.append(np.asarray(clip["features"], dtype=np.float32))
        texts.append(np.asarray(text["features"], dtype=np.float32))
        labels.append(np.asarray(clip["labels"], dtype=np.float32))
        soft.append(np.asarray(clip["soft_labels"], dtype=np.float32))
        ids.extend([f"{split}|{i}" for i in clip["image_ids"]])
    pool = {
        "image_features": np.concatenate(feats, axis=0),
        "text_features": np.concatenate(texts, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "soft_labels": np.concatenate(soft, axis=0),
        "image_ids": np.asarray(ids),
    }
    return pool


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


METHODS = ["oracle_teacher", "baseline", "standard_kd", "dynamic_gated_kd"]
METRIC_KEYS = ["mAP", "macro", "micro", "samples", "hard"]


def run_cv(
    pool: Dict[str, np.ndarray],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """One full k-fold CV pass at a given seed. Returns per-fold metrics, pooled
    OOF predictions (each sample tested once), and per-fold dynamic-gate means."""
    targs = _trainer_args(args, seed)
    n = int(pool["labels"].shape[0])
    num_classes = int(pool["labels"].shape[1])
    text_dim = int(pool["text_features"].shape[1])
    image_dim = int(pool["image_features"].shape[1])

    rng = np.random.RandomState(int(seed))
    fold_assign = np.array_split(rng.permutation(n), int(args.folds))

    per_fold: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}
    gate_means: List[float] = []
    slice_rows: List[Dict[str, Any]] = []  # per (seed, fold, method, agreement) F1 breakdown
    oof_pred = {m: np.zeros((n, num_classes), dtype=np.float32) for m in METHODS if m != "oracle_teacher"}

    for k in range(int(args.folds)):
        test_idx = np.asarray(fold_assign[k], dtype=np.int64)
        val_idx = np.asarray(fold_assign[(k + 1) % int(args.folds)], dtype=np.int64)
        train_idx = np.setdiff1d(
            np.arange(n, dtype=np.int64), np.concatenate([test_idx, val_idx]), assume_unique=False
        )
        print(f"\n===== seed {seed} fold {k + 1}/{args.folds} | train={train_idx.size} val={val_idx.size} test={test_idx.size} =====")

        tr_text, tr_img = pool["text_features"][train_idx], pool["image_features"][train_idx]
        tr_lab, tr_soft = pool["labels"][train_idx], pool["soft_labels"][train_idx]
        va_text, va_img, va_lab = pool["text_features"][val_idx], pool["image_features"][val_idx], pool["labels"][val_idx]
        te_text, te_img, te_lab = pool["text_features"][test_idx], pool["image_features"][test_idx], pool["labels"][test_idx]
        te_soft = pool["soft_labels"][test_idx]
        te_groups = P._assign_agreement_groups(te_lab, te_soft, mode="min")

        # ---- teacher (privileged text) ----
        P._set_component_seed(int(seed), offset=10 * k + 0)
        teacher = P.TeacherMLP(text_dim, int(args.teacher_hidden_dim), num_classes, float(args.dropout), "text_only").to(device)
        teacher_res = P._train_teacher(
            teacher, P.TeacherDataset(tr_text, tr_lab), va_text, va_lab, te_text, te_lab, device, targs
        )
        per_fold["oracle_teacher"].append(_metrics_of(teacher_res["bundle"]))

        # ---- KD targets on train ----
        if str(args.kd_target_mode) == "oof":
            kd_scores = P._compute_oof_teacher_scores(tr_text, tr_lab, va_text, va_lab, te_text, te_lab, device, targs)
        else:
            kd_scores = P._predict_teacher(teacher, tr_text, device, int(args.batch_size))
        teacher_probs = P._sigmoid_np(P._logit_np(kd_scores) / float(args.temperature))
        agreement = P._compute_sample_agreement(tr_lab, tr_soft, mode=str(args.student_agreement_pool))

        student_specs = {
            "baseline": dict(agree=np.ones_like(agreement), tprobs=np.zeros_like(tr_lab), mode="baseline"),
            "standard_kd": dict(agree=np.ones_like(agreement), tprobs=teacher_probs, mode="standard_kd"),
            "dynamic_gated_kd": dict(agree=agreement, tprobs=teacher_probs, mode="dynamic_kd"),
        }
        for offset, (name, spec) in enumerate(student_specs.items(), start=1):
            P._set_component_seed(int(seed), offset=10 * k + offset)
            model = P.StudentMLP(image_dim, int(args.student_hidden_dim), num_classes, float(args.dropout), int(args.feature_proj_dim)).to(device)
            ds = P.StudentDataset(tr_img, tr_lab, spec["agree"], tr_soft, spec["tprobs"])
            res = P._train_student(spec["mode"], model, ds, va_img, va_lab, te_img, te_lab, device, targs)
            per_fold[name].append(_metrics_of(res["bundle"]))
            te_scores = P._predict_student(model, te_img, device, int(args.batch_size))
            oof_pred[name][test_idx] = te_scores
            # per-fold agreement-stratified F1/mAP using this fold's val-fit thresholds
            thresholds = np.asarray(res["bundle"]["classwise"]["val"]["class_thresholds"], dtype=np.float32)
            for srow in P._slice_metrics(te_scores, te_lab, te_groups, thresholds, method=name, split="test"):
                srow.update({"seed": int(seed), "fold": int(k)})
                slice_rows.append(srow)
            if name == "dynamic_gated_kd":
                best = next((h for h in res["history"] if h["epoch"] == res["best_epoch"]), res["history"][-1])
                gate_means.append(float(best.get("mean_gate", float("nan"))))

    return {"per_fold": per_fold, "oof_pred": oof_pred, "gate_means": gate_means, "slice_rows": slice_rows}


def _paired_test(deltas: np.ndarray) -> Dict[str, Any]:
    """Two-sided paired test on per-(seed,fold) deltas (method - baseline)."""
    from scipy import stats

    deltas = np.asarray(deltas, dtype=np.float64)
    out: Dict[str, Any] = {
        "n": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1)) if deltas.size > 1 else float("nan"),
    }
    if deltas.size > 1 and np.any(deltas != 0):
        t_stat, t_p = stats.ttest_1samp(deltas, 0.0)
        out["ttest_t"] = float(t_stat)
        out["ttest_p"] = float(t_p)
        try:
            w_stat, w_p = stats.wilcoxon(deltas)
            out["wilcoxon_p"] = float(w_p)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = P._resolve_device(args.device)

    seeds = (
        [int(s) for s in str(args.seeds).split(",") if s.strip()]
        if args.seeds
        else [int(args.seed)]
    )
    P._set_seed(seeds[0])

    pool = _load_pool(args)
    n = int(pool["labels"].shape[0])
    pool_agreement = P._compute_sample_agreement(
        labels=pool["labels"], soft_labels=pool["soft_labels"], mode=str(args.student_agreement_pool)
    )
    frac_gated = float((pool_agreement < 0.999).mean())
    print(
        f"[Pool] persons={n} seeds={seeds} folds={args.folds}\n"
        f"[Pool] mean positive-label agreement={pool_agreement.mean():.3f} "
        f"frac samples gate active={frac_gated:.3f}"
    )

    # accumulate per-(seed,fold) metrics
    all_fold: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}
    per_seed_means: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}
    gate_all: List[float] = []
    all_slice: List[Dict[str, Any]] = []

    for seed in seeds:
        cv = run_cv(pool, seed, args, device)
        gate_all.extend(cv["gate_means"])
        all_slice.extend(cv["slice_rows"])
        for m in METHODS:
            all_fold[m].extend(cv["per_fold"][m])
            per_seed_means[m].append({mk: float(np.mean([f[mk] for f in cv["per_fold"][m]])) for mk in METRIC_KEYS})

    # ---- comparison table (mean±std over all seed×fold observations) ----
    summary_rows: List[Dict[str, Any]] = []
    for m in METHODS:
        row: Dict[str, Any] = {"method": m, "observations": len(all_fold[m])}
        for mk in METRIC_KEYS:
            agg = _mean_std([f[mk] for f in all_fold[m]])
            row[f"{mk}_mean"] = round(agg["mean"], 4)
            row[f"{mk}_std"] = round(agg["std"], 4)
        summary_rows.append(row)
    _write_csv(out_dir / "pool_kfold_comparison.csv", summary_rows)

    # ---- overall paired significance: each method vs a reference, paired by split ----
    sig_metrics = ("mAP", "macro", "micro", "samples")
    contrasts = [
        ("standard_kd", "baseline"),
        ("dynamic_gated_kd", "baseline"),
        ("dynamic_gated_kd", "standard_kd"),  # isolates the gate's marginal effect
    ]
    sig_rows: List[Dict[str, Any]] = []
    for a_name, b_name in contrasts:
        for mk in sig_metrics:
            d_fold = np.array([a[mk] - b[mk] for a, b in zip(all_fold[a_name], all_fold[b_name])])
            d_seed = np.array([a[mk] - b[mk] for a, b in zip(per_seed_means[a_name], per_seed_means[b_name])])
            ft, st = _paired_test(d_fold), _paired_test(d_seed)
            sig_rows.append({
                "contrast": f"{a_name}-vs-{b_name}", "metric": mk,
                "per_fold_mean_delta": round(ft["mean_delta"], 4),
                "per_fold_ttest_p": round(ft.get("ttest_p", float("nan")), 4),
                "per_fold_wilcoxon_p": round(ft.get("wilcoxon_p", float("nan")), 4),
                "per_seed_mean_delta": round(st["mean_delta"], 4),
                "per_seed_ttest_p": round(st.get("ttest_p", float("nan")), 4),
            })
    _write_csv(out_dir / "pool_kfold_significance.csv", sig_rows)

    # ---- agreement-slice: full F1 breakdown (mean±std) + paired gate effect ----
    slice_metric_keys = ["macro", "micro", "samples", "mAP", "hard"]
    lookup: Dict[tuple, Dict[str, float]] = {
        (r["method"], r["seed"], r["fold"], r["agreement"]): r for r in all_slice
    }
    agreements_seen = sorted({r["agreement"] for r in all_slice}, key=lambda a: {"1/3": 0, "2/3": 1, "1": 2}.get(a, 9))
    slice_rows: List[Dict[str, Any]] = []
    for ag in agreements_seen:
        keys = sorted({(r["seed"], r["fold"]) for r in all_slice if r["agreement"] == ag})
        for m in ("baseline", "standard_kd", "dynamic_gated_kd"):
            row: Dict[str, Any] = {"agreement": ag, "method": m,
                                   "obs": sum(1 for (s, f) in keys if (m, s, f, ag) in lookup)}
            for mk in slice_metric_keys:
                vals = [lookup[(m, s, f, ag)][mk] for (s, f) in keys if (m, s, f, ag) in lookup]
                agg = _mean_std(vals) if vals else {"mean": float("nan"), "std": float("nan")}
                row[f"{mk}_mean"] = round(agg["mean"], 3)
                row[f"{mk}_std"] = round(agg["std"], 3)
            slice_rows.append(row)
    _write_csv(out_dir / "pool_kfold_agreement_slice.csv", slice_rows)

    # paired gate effect within each agreement bin (gated - standard_kd, gated - baseline)
    slice_sig_rows: List[Dict[str, Any]] = []
    for ag in agreements_seen:
        keys = sorted({(r["seed"], r["fold"]) for r in all_slice if r["agreement"] == ag})
        for a_name, b_name in [("dynamic_gated_kd", "standard_kd"), ("dynamic_gated_kd", "baseline")]:
            paired = [(lookup[(a_name, s, f, ag)], lookup[(b_name, s, f, ag)])
                      for (s, f) in keys if (a_name, s, f, ag) in lookup and (b_name, s, f, ag) in lookup]
            for mk in ("macro", "micro", "samples", "mAP"):
                d = np.array([a[mk] - b[mk] for a, b in paired])
                t = _paired_test(d)
                slice_sig_rows.append({
                    "agreement": ag, "contrast": f"{a_name}-vs-{b_name}", "metric": mk, "n": t["n"],
                    "mean_delta": round(t["mean_delta"], 3), "ttest_p": round(t.get("ttest_p", float("nan")), 4),
                    "wilcoxon_p": round(t.get("wilcoxon_p", float("nan")), 4),
                })
    _write_csv(out_dir / "pool_kfold_agreement_slice_significance.csv", slice_sig_rows)

    gate_summary = _mean_std([g for g in gate_all if g == g]) if gate_all else {"mean": float("nan"), "std": float("nan")}
    (out_dir / "summary.json").write_text(json.dumps({
        "note": "Multi-seed controlled k-fold on EMOTIC val+test pool (agreement available at train time); NOT a benchmark.",
        "pool_persons": n, "seeds": seeds, "folds": int(args.folds),
        "pool_mean_agreement": float(pool_agreement.mean()),
        "frac_samples_gate_active": frac_gated,
        "dynamic_gate_mean": gate_summary,
        "comparison": summary_rows,
        "significance": sig_rows,
        "agreement_slice": slice_rows,
        "agreement_slice_significance": slice_sig_rows,
        "config": vars(args),
        "per_seed_means": per_seed_means,
    }, indent=2), encoding="utf-8")

    print(f"\n=== POOL K-FOLD (mean±std over {len(all_fold['baseline'])} seed×fold obs) ===")
    for r in summary_rows:
        print(f"  {r['method']:18s} mAP={r['mAP_mean']:.2f}±{r['mAP_std']:.2f}  macro={r['macro_mean']*100:.2f}  micro={r['micro_mean']*100:.2f}  samples={r['samples_mean']*100:.2f}")
    print(f"  gate mean weight = {gate_summary['mean']:.3f}±{gate_summary['std']:.3f}")
    print("\n=== OVERALL PAIRED CONTRASTS (per-fold n=25) ===")
    for r in sig_rows:
        print(f"  {r['contrast']:34s} {r['metric']:7s} Δ={r['per_fold_mean_delta']:+.4f} p_t={r['per_fold_ttest_p']:.4f} p_wil={r['per_fold_wilcoxon_p']:.4f}")
    print("\n=== AGREEMENT-SLICE: gate effect (gated - standard_kd) ===")
    for r in slice_sig_rows:
        if r["contrast"].endswith("standard_kd"):
            print(f"  agree={r['agreement']:3s} {r['metric']:7s} n={r['n']:2d} Δ={r['mean_delta']:+.3f} p_t={r['ttest_p']:.4f} p_wil={r['wilcoxon_p']:.4f}")
    print(f"[Pool] artifacts -> {out_dir}")


def _metrics_of(bundle: Dict[str, Any]) -> Dict[str, float]:
    t = bundle["classwise"]["test"]
    return {
        "mAP": float(t["mAP"]),
        "macro": float(t["macro"]),
        "micro": float(t["micro"]),
        "samples": float(t["samples"]),
        "hard": float(t["hard"]),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
