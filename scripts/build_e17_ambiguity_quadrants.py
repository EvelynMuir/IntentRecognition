#!/usr/bin/env python3
"""E17 - theory-predicted 2x2 ambiguity-interaction experiment.

Sec. 3.5 of the revision states an exact decomposition of the predictive
uncertainty of a deterministic visual representation into a supervisory term
H(Y|X) and a representation-side term I(Y;X|Z), and then attaches one FDIL
module to each term.  That attachment makes a falsifiable interaction
prediction:

  * SLR-C, which acts on the representation-side term, should help most where
    *semantic* ambiguity is high (small top-2 candidate margin) and should be
    roughly inert where it is low;
  * UTD, which acts on the supervisory term, should help most where *annotator*
    ambiguity is high (low agreement omega) and should be roughly inert where it
    is high-agreement;
  * full FDIL should dominate where both are high.

Testing this needs per-sample annotator agreement, which on Intentonomy exists
*only on the train split* (val/test ship hard `category_ids` only).  We
therefore cross-fit: the train split is partitioned into F folds, every variant
is retrained from scratch on F-1 folds and predicted on the held-out fold, and
the concatenated out-of-fold scores give one honest prediction per train image
alongside its real omega.  Class-wise decision thresholds are fitted per fold on
the official validation split only, exactly as in the manuscript protocol, so no
held-out fold ever informs its own threshold.

Outputs: the 2x2 table, per-quadrant paired significance tests, and the two
analytic module-level checks (Prop. 2 margin identity, Prop. 3 gradient scaling).
"""

from __future__ import annotations

# --- fabricate training-only deps so pure helpers import in a clip-only env ----
# Mirrors build_e3_unified_vs_decoupled.py: the s2d env has torch+clip but not
# lightning/rich/hydra, and the numeric helpers we import do not need them.
import sys
import types
import importlib.abc
import importlib.machinery

_STUB_ROOTS = set(
    "lightning pytorch_lightning rich hydra omegaconf rootutils "
    "lightning_utilities torchmetrics wandb tensorboard".split()
)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return _Dummy()

    def __getattr__(self, _n):
        return _Dummy()


class _AutoModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (_Dummy,), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _AutoModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

# --- real imports -------------------------------------------------------------
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_distillation_slrc import (  # noqa: E402
    DEFAULT_ANNOTATION_FILE,
    DEFAULT_BASE_CACHE_DIR,
    DEFAULT_GEMINI_FILE,
    DEFAULT_TEACHER_RUN_DIR,
    DEFAULT_TEXT_DIR,
    ResidualStudent,
    SLRCDataset,
    _align_text_bundle_to_clip,
    _apply_slr,
    _build_text_pools,
    _encode_text_pool,
    _json_ready,
    _load_cache_bundle,
    _load_class_names,
    _load_text_bundle,
    _logit_np,
    _normalize_scores_per_sample,
    _predict_baseline_logits,
    _predict_residual_student,
    _predict_teacher,
    _resolve_device,
    _set_component_seed,
    _set_seed,
    _sigmoid_np,
    _slr_feature_view,
    _text_logits_from_features,
    _train_residual_student,
)
from scripts.analyze_privileged_distillation import (  # noqa: E402
    StudentDataset,
    StudentMLP,
    TeacherDataset,
    TeacherMLP,
    _compute_sample_agreement,
    _predict_student,
    _train_student,
    _train_teacher,
    compute_mAP,
    search_classwise_thresholds,
)

import clip  # type: ignore  # noqa: E402


VARIANTS = ["baseline", "slrc", "utd", "fdil"]
VARIANT_LABELS = {
    "baseline": "Base",
    "slrc": "+SLR-C",
    "utd": "+UTD",
    "fdil": "Full FDIL",
}
QUADRANTS = [
    ("low_sup_low_sem", "Low sup. / Low sem.", False, False),
    ("low_sup_high_sem", "Low sup. / High sem.", False, True),
    ("high_sup_low_sem", "High sup. / Low sem.", True, False),
    ("high_sup_high_sem", "High sup. / High sem.", True, True),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build E17 theory-predicted 2x2 ambiguity-interaction evidence.")
    p.add_argument("--reuse-cache-dir", type=Path, default=DEFAULT_BASE_CACHE_DIR)
    p.add_argument("--teacher-run-dir", type=Path, default=DEFAULT_TEACHER_RUN_DIR)
    p.add_argument("--train-text-npz", type=Path, default=DEFAULT_TEXT_DIR / "rationale_full_bge_features.npz")
    p.add_argument("--val-text-npz", type=Path, default=DEFAULT_TEXT_DIR / "val_rationale_baseline_pred_bge_features.npz")
    p.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    p.add_argument("--gemini-file", type=Path, default=DEFAULT_GEMINI_FILE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--folds", type=int, default=5)
    # training hyper-parameters: identical to the manuscript's default pipeline
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--teacher-hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feature-proj-dim", type=int, default=256)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--standard-kd-weight", type=float, default=1.0)
    p.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    p.add_argument("--dynamic-kd-variant", type=str, default="sample_inverse")
    p.add_argument("--dynamic-gate-alpha", type=float, default=0.3)
    p.add_argument("--dynamic-gate-beta", type=float, default=0.7)
    p.add_argument("--entropy-gate-lambda", type=float, default=1.0)
    p.add_argument("--feature-distill-mode", type=str, default="none")
    p.add_argument("--feature-distill-weight", type=float, default=0.0)
    p.add_argument("--feature-distill-temperature", type=float, default=0.1)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--slr-alpha", type=float, default=0.3)
    p.add_argument(
        "--sem-split",
        type=str,
        default="median",
        choices=["median"],
        help="Split rule for the semantic-ambiguity axis (global median of the negated top-2 base margin).",
    )
    return p.parse_args()


def _resolve_output_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"e17_ambiguity_quadrants_{stamp}"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _per_sample_f1(scores: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-sample (example-based) F1 under class-wise thresholds."""
    pred = (np.asarray(scores, dtype=np.float32) > np.asarray(thresholds, dtype=np.float32)[None, :]).astype(np.float32)
    tgt = (np.asarray(targets, dtype=np.float32) > 0.5).astype(np.float32)
    tp = (pred * tgt).sum(axis=1)
    denom = pred.sum(axis=1) + tgt.sum(axis=1)
    return np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 1.0).astype(np.float32)


def _macro_f1(scores: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> float:
    pred = (np.asarray(scores, dtype=np.float32) > np.asarray(thresholds, dtype=np.float32)[None, :]).astype(np.float32)
    tgt = (np.asarray(targets, dtype=np.float32) > 0.5).astype(np.float32)
    tp = (pred * tgt).sum(axis=0)
    fp = (pred * (1.0 - tgt)).sum(axis=0)
    fn = ((1.0 - pred) * tgt).sum(axis=0)
    denom = 2.0 * tp + fp + fn
    present = tgt.sum(axis=0) > 0
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0)
    if not np.any(present):
        return 0.0
    return float(f1[present].mean())


def _subset_map(scores: np.ndarray, targets: np.ndarray) -> float:
    tgt = (np.asarray(targets, dtype=np.float32) > 0.5).astype(np.float32)
    keep = tgt.sum(axis=0) > 0
    if not np.any(keep):
        return float("nan")
    return float(compute_mAP(np.asarray(scores, dtype=np.float32)[:, keep], tgt[:, keep]))


def _holm(pvals: Sequence[float]) -> list[float]:
    pvals = [float(v) for v in pvals]
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * pvals[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def _fold_assignment(n: int, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = np.arange(n) % int(folds)
    return assignment


def _encode_prior_embeddings(class_names: Sequence[str], gemini_file: Path, device: torch.device):
    pools = _build_text_pools(class_names, gemini_file)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    lexical = _encode_text_pool(clip_model, pools["lexical"], wrap_prompt=True)
    canonical = _encode_text_pool(clip_model, pools["canonical"], wrap_prompt=True)
    scenario = _encode_text_pool(clip_model, pools["scenario"], wrap_prompt=False)
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logit_scale, lexical, canonical, scenario


def _prior_logits(features: np.ndarray, logit_scale: float, lexical, canonical, scenario) -> np.ndarray:
    return (
        _text_logits_from_features(features, lexical, logit_scale)
        + _text_logits_from_features(features, canonical, logit_scale)
        + _text_logits_from_features(features, scenario, logit_scale)
    ) / 3.0


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))

    train_clip = _load_cache_bundle(args.reuse_cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(args.reuse_cache_dir / "val_clip.npz")
    train_text = _align_text_bundle_to_clip(
        _load_text_bundle(args.train_text_npz, required_keys=["image_ids", "features"]), train_clip
    )
    val_text = _align_text_bundle_to_clip(
        _load_text_bundle(args.val_text_npz, required_keys=["image_ids", "features"]), val_clip
    )

    train_features = np.asarray(train_clip["features"], dtype=np.float32)
    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    train_soft = np.asarray(train_clip["soft_labels"], dtype=np.float32)
    train_text_features = np.asarray(train_text["features"], dtype=np.float32)
    val_features = np.asarray(val_clip["features"], dtype=np.float32)
    val_labels = np.asarray(val_clip["labels"], dtype=np.float32)
    val_text_features = np.asarray(val_text["features"], dtype=np.float32)

    n_samples, image_dim = train_features.shape
    num_classes = int(train_labels.shape[1])
    agreement = _compute_sample_agreement(train_labels, train_soft, mode="min")

    class_names = _load_class_names(args.annotation_file)
    logit_scale, lexical, canonical, scenario = _encode_prior_embeddings(class_names, args.gemini_file, device)
    train_prior = _prior_logits(_slr_feature_view(train_clip), logit_scale, lexical, canonical, scenario)
    val_prior = _prior_logits(_slr_feature_view(val_clip), logit_scale, lexical, canonical, scenario)

    folds = _fold_assignment(n_samples, int(args.folds), int(args.seed))
    oof = {name: np.zeros((n_samples, num_classes), dtype=np.float32) for name in VARIANTS}
    oof_base_logits = np.zeros((n_samples, num_classes), dtype=np.float32)
    oof_teacher = np.zeros((n_samples, num_classes), dtype=np.float32)
    oof_slr_logits = np.zeros((n_samples, num_classes), dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    grad_records: list[dict[str, Any]] = []

    for fold in range(int(args.folds)):
        held = np.where(folds == fold)[0]
        inner = np.where(folds != fold)[0]
        print(f"[E17] fold {fold}: inner={len(inner)} held={len(held)}", flush=True)
        fold_seed = int(args.seed) + 1000 * (fold + 1)

        # ---- text teacher on the inner split only ----------------------------
        _set_component_seed(fold_seed, offset=10)
        teacher = TeacherMLP(
            text_dim=int(train_text_features.shape[1]),
            hidden_dim=int(args.teacher_hidden_dim),
            num_classes=num_classes,
            dropout=float(args.dropout),
            input_mode="text_only",
        ).to(device)
        teacher_result = _train_teacher(
            model=teacher,
            train_dataset=TeacherDataset(
                text_features=train_text_features[inner], labels=train_labels[inner]
            ),
            val_text_features=val_text_features,
            val_targets=val_labels,
            test_text_features=train_text_features[held],
            test_targets=train_labels[held],
            device=device,
            args=args,
        )
        inner_teacher_probs = _sigmoid_np(
            _logit_np(
                _predict_teacher(
                    teacher_model=teacher,
                    text_features=train_text_features[inner],
                    device=device,
                    batch_size=int(args.batch_size),
                )
            )
            / float(args.temperature)
        )
        # held-fold teacher predictions, used only to measure teacher reliability
        # per agreement stratum (the interpretation of the supervisory axis).
        oof_teacher[held] = _predict_teacher(
            teacher_model=teacher,
            text_features=train_text_features[held],
            device=device,
            batch_size=int(args.batch_size),
        )
        teacher_val_probs = _predict_teacher(
            teacher_model=teacher,
            text_features=val_text_features,
            device=device,
            batch_size=int(args.batch_size),
        )
        np.save(
            output_dir / f"thresholds_fold{fold}_teacher.npy",
            search_classwise_thresholds(teacher_val_probs, val_labels),
        )

        # ---- baseline visual student ----------------------------------------
        base_seed = _set_component_seed(fold_seed, offset=100)
        baseline_model = StudentMLP(
            image_dim=image_dim,
            hidden_dim=int(args.hidden_dim),
            num_classes=num_classes,
            dropout=float(args.dropout),
            feature_proj_dim=int(args.feature_proj_dim),
        ).to(device)
        baseline_result = _train_student(
            mode="baseline",
            model=baseline_model,
            train_dataset=StudentDataset(
                image_features=train_features[inner],
                labels=train_labels[inner],
                agreement=agreement[inner],
                soft_labels=train_soft[inner],
                teacher_probs=np.zeros_like(inner_teacher_probs),
            ),
            val_image_features=val_features,
            val_targets=val_labels,
            test_image_features=train_features[held],
            test_targets=train_labels[held],
            device=device,
            args=args,
        )
        base_val_logits = _predict_baseline_logits(baseline_model, val_features, device, int(args.batch_size))
        base_held_logits = _predict_baseline_logits(baseline_model, train_features[held], device, int(args.batch_size))
        base_inner_logits = _predict_baseline_logits(baseline_model, train_features[inner], device, int(args.batch_size))
        oof_base_logits[held] = base_held_logits

        # ---- SLR-C (training-free local reranking of the base logits) --------
        slr_val_logits = _apply_slr(base_val_logits, val_prior, topk=int(args.topk), alpha=float(args.slr_alpha))
        slr_held_logits = _apply_slr(base_held_logits, train_prior[held], topk=int(args.topk), alpha=float(args.slr_alpha))
        slr_inner_logits = _apply_slr(base_inner_logits, train_prior[inner], topk=int(args.topk), alpha=float(args.slr_alpha))
        oof_slr_logits[held] = slr_held_logits

        # ---- UTD only: agreement-gated distillation into a plain student -----
        utd_seed = _set_component_seed(fold_seed, offset=200)
        utd_model = StudentMLP(
            image_dim=image_dim,
            hidden_dim=int(args.hidden_dim),
            num_classes=num_classes,
            dropout=float(args.dropout),
            feature_proj_dim=int(args.feature_proj_dim),
        ).to(device)
        utd_result = _train_student(
            mode="dynamic_kd",
            model=utd_model,
            train_dataset=StudentDataset(
                image_features=train_features[inner],
                labels=train_labels[inner],
                agreement=agreement[inner],
                soft_labels=train_soft[inner],
                teacher_probs=inner_teacher_probs,
            ),
            val_image_features=val_features,
            val_targets=val_labels,
            test_image_features=train_features[held],
            test_targets=train_labels[held],
            device=device,
            args=args,
        )

        # ---- full FDIL: agreement-gated residual student on the SLR-C prior --
        fdil_seed = _set_component_seed(fold_seed, offset=300)
        fdil_model = ResidualStudent(
            image_dim=image_dim,
            hidden_dim=int(args.hidden_dim),
            num_classes=num_classes,
            dropout=float(args.dropout),
        ).to(device)
        fdil_result = _train_residual_student(
            mode="dynamic_kd",
            model=fdil_model,
            train_dataset=SLRCDataset(
                image_features=train_features[inner],
                slr_logits=slr_inner_logits,
                labels=train_labels[inner],
                soft_labels=train_soft[inner],
                agreement=agreement[inner],
                teacher_probs=inner_teacher_probs,
            ),
            val_image_features=val_features,
            val_slr_logits=slr_val_logits,
            val_targets=val_labels,
            test_image_features=train_features[held],
            test_slr_logits=slr_held_logits,
            test_targets=train_labels[held],
            device=device,
            args=args,
            loader_seed=fdil_seed,
        )

        # ---- per-fold validation-only class-wise thresholds ------------------
        val_scores = {
            "baseline": _sigmoid_np(base_val_logits),
            "slrc": _sigmoid_np(slr_val_logits),
            "utd": _predict_student(utd_model, val_features, device, int(args.batch_size)),
            "fdil": _predict_residual_student(
                model=fdil_model,
                image_features=val_features,
                slr_logits=slr_val_logits,
                device=device,
                batch_size=int(args.batch_size),
            ),
        }
        held_scores = {
            "baseline": _sigmoid_np(base_held_logits),
            "slrc": _sigmoid_np(slr_held_logits),
            "utd": _predict_student(utd_model, train_features[held], device, int(args.batch_size)),
            "fdil": _predict_residual_student(
                model=fdil_model,
                image_features=train_features[held],
                slr_logits=slr_held_logits,
                device=device,
                batch_size=int(args.batch_size),
            ),
        }
        for name in VARIANTS:
            thresholds = search_classwise_thresholds(val_scores[name], val_labels)
            np.save(output_dir / f"thresholds_fold{fold}_{name}.npy", thresholds)
            oof[name][held] = held_scores[name]
            fold_records.append(
                {
                    "fold": fold,
                    "variant": name,
                    "n_held": int(len(held)),
                    "sample_f1": float(_per_sample_f1(held_scores[name], train_labels[held], thresholds).mean() * 100.0),
                    "thresholds": [float(v) for v in thresholds.tolist()],
                }
            )

        # ---- Prop. 3 diagnostic: realized teacher-gradient magnitude ---------
        # dL_distill/dz = lambda (1 - omega) (p - q) / tau  for the gated branch.
        with torch.no_grad():
            student_probs_inner = _predict_student(utd_model, train_features[inner], device, int(args.batch_size))
        residual = np.abs(student_probs_inner - inner_teacher_probs).sum(axis=1)
        gate = 1.0 - agreement[inner]
        grad_norm = float(args.dynamic_kd_weight) * gate * residual / float(args.temperature)
        for omega_value in np.unique(np.round(agreement[inner], 3)):
            mask = np.isclose(np.round(agreement[inner], 3), omega_value)
            grad_records.append(
                {
                    "fold": fold,
                    "omega": float(omega_value),
                    "n": int(mask.sum()),
                    "gate": float(1.0 - omega_value),
                    "mean_abs_p_minus_q": float(residual[mask].mean()),
                    "mean_teacher_grad_l1": float(grad_norm[mask].mean()),
                }
            )

        print(
            f"[E17] fold {fold} done | teacher_ep={teacher_result['best_epoch']} "
            f"base_ep={baseline_result['best_epoch']} utd_ep={utd_result['best_epoch']} "
            f"fdil_ep={fdil_result['best_epoch']} (base_seed={base_seed}, utd_seed={utd_seed}, fdil_seed={fdil_seed})",
            flush=True,
        )

    # ---- pooled per-sample F1 using each sample's own fold thresholds --------
    sample_f1 = {name: np.zeros(n_samples, dtype=np.float32) for name in VARIANTS}
    for fold in range(int(args.folds)):
        held = np.where(folds == fold)[0]
        for name in VARIANTS:
            thresholds = np.load(output_dir / f"thresholds_fold{fold}_{name}.npy")
            sample_f1[name][held] = _per_sample_f1(oof[name][held], train_labels[held], thresholds)

    # ---- ambiguity axes ------------------------------------------------------
    a_sup = 1.0 - agreement
    sorted_base = np.sort(oof_base_logits, axis=1)
    a_sem = -(sorted_base[:, -1] - sorted_base[:, -2])
    sem_threshold = float(np.median(a_sem))
    high_sem = a_sem > sem_threshold
    # omega takes only {1/3, 2/3, 1} on Intentonomy; "high supervisory ambiguity"
    # is the minimum-agreement stratum omega = 1/3, "low" is omega >= 2/3.
    high_sup = agreement < 0.5

    axis_summary = {
        "a_sup_definition": "1 - omega, omega = min soft agreement over positive labels",
        "a_sem_definition": "negated top-2 margin of the out-of-fold baseline logits",
        "sem_median_threshold": sem_threshold,
        "omega_counts": {
            str(float(v)): int(c) for v, c in zip(*np.unique(np.round(agreement, 3), return_counts=True))
        },
        "n_high_sup": int(high_sup.sum()),
        "n_high_sem": int(high_sem.sum()),
    }

    # ---- 2x2 table -----------------------------------------------------------
    quadrant_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    raw_pvals: list[float] = []
    pval_index: list[tuple[int, str]] = []
    for q_key, q_label, want_sup, want_sem in QUADRANTS:
        mask = (high_sup == want_sup) & (high_sem == want_sem)
        idx = np.where(mask)[0]
        base_f1 = float(sample_f1["baseline"][idx].mean() * 100.0)
        for name in VARIANTS:
            row = {
                "quadrant": q_key,
                "quadrant_label": q_label,
                "variant": VARIANT_LABELS[name],
                "variant_key": name,
                "n": int(len(idx)),
                "sample_f1": float(sample_f1[name][idx].mean() * 100.0),
                "macro_f1": float(
                    np.mean(
                        [
                            _macro_f1(
                                oof[name][np.intersect1d(idx, np.where(folds == fold)[0])],
                                train_labels[np.intersect1d(idx, np.where(folds == fold)[0])],
                                np.load(output_dir / f"thresholds_fold{fold}_{name}.npy"),
                            )
                            for fold in range(int(args.folds))
                        ]
                    )
                    * 100.0
                ),
                "mAP": _subset_map(oof[name][idx], train_labels[idx]),
                "delta_sample_f1": float(sample_f1[name][idx].mean() * 100.0) - base_f1,
                "rel_delta_sample_f1_pct": 100.0
                * (float(sample_f1[name][idx].mean() * 100.0) - base_f1)
                / max(base_f1, 1e-9),
                "mean_positives": float((train_labels[idx] > 0.5).sum(axis=1).mean()),
                "mean_omega": float(agreement[idx].mean()),
                "mean_a_sem": float(a_sem[idx].mean()),
            }
            quadrant_rows.append(row)
            if name == "baseline":
                continue
            diff = sample_f1[name][idx] - sample_f1["baseline"][idx]
            t_stat, p_val = stats.ttest_rel(sample_f1[name][idx], sample_f1["baseline"][idx])
            test_rows.append(
                {
                    "quadrant": q_key,
                    "quadrant_label": q_label,
                    "variant": VARIANT_LABELS[name],
                    "n": int(len(idx)),
                    "delta_sample_f1": float(diff.mean() * 100.0),
                    "t_stat": float(t_stat),
                    "p_raw": float(p_val),
                    "p_holm": 0.0,
                }
            )
            raw_pvals.append(float(p_val))
            pval_index.append((len(test_rows) - 1, name))
    for adjusted, (row_idx, _) in zip(_holm(raw_pvals), pval_index):
        test_rows[row_idx]["p_holm"] = float(adjusted)

    # ---- Prop. 2 check: margin identity and pairwise correction rate ---------
    prior_norm = _normalize_scores_per_sample(train_prior)
    topk = max(1, min(int(args.topk), num_classes))
    topk_idx = np.argpartition(-oof_base_logits, kth=topk - 1, axis=1)[:, :topk]
    in_candidate = np.zeros_like(oof_base_logits, dtype=bool)
    np.put_along_axis(in_candidate, topk_idx, True, axis=1)

    identity_residuals: list[float] = []
    prior_correct = 0
    pair_total = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0
    positives = train_labels > 0.5
    for i in range(n_samples):
        pos_ids = np.where(positives[i] & in_candidate[i])[0]
        neg_ids = np.where((~positives[i]) & in_candidate[i])[0]
        if len(pos_ids) == 0 or len(neg_ids) == 0:
            continue
        # the hardest confusion pair: lowest-scoring positive vs top-scoring negative
        c_pos = pos_ids[np.argmin(oof_base_logits[i, pos_ids])]
        c_neg = neg_ids[np.argmax(oof_base_logits[i, neg_ids])]
        m_base = float(oof_base_logits[i, c_pos] - oof_base_logits[i, c_neg])
        m_slr = float(oof_slr_logits[i, c_pos] - oof_slr_logits[i, c_neg])
        delta_prior = float(prior_norm[i, c_pos] - prior_norm[i, c_neg])
        identity_residuals.append(abs((m_slr - m_base) - float(args.slr_alpha) * delta_prior))
        pair_total += 1
        if delta_prior > 0:
            prior_correct += 1
        if m_base < 0 <= m_slr:
            flipped_to_correct += 1
        if m_base >= 0 > m_slr:
            flipped_to_wrong += 1
    prop2 = {
        "pairs_evaluated": int(pair_total),
        "max_abs_identity_residual": float(np.max(identity_residuals)) if identity_residuals else 0.0,
        "mean_abs_identity_residual": float(np.mean(identity_residuals)) if identity_residuals else 0.0,
        "prior_orders_pair_correctly_pct": 100.0 * prior_correct / max(1, pair_total),
        "pairs_flipped_to_correct": int(flipped_to_correct),
        "pairs_flipped_to_wrong": int(flipped_to_wrong),
        "net_pair_corrections": int(flipped_to_correct - flipped_to_wrong),
        "outside_candidate_max_abs_perturbation": float(
            np.max(np.abs(oof_slr_logits - oof_base_logits)[~in_candidate]) if np.any(~in_candidate) else 0.0
        ),
    }

    # ---- Prop. 3 aggregate ---------------------------------------------------
    grad_summary: list[dict[str, Any]] = []
    for omega_value in sorted({record["omega"] for record in grad_records}):
        subset = [record for record in grad_records if record["omega"] == omega_value]
        weights = np.array([record["n"] for record in subset], dtype=np.float64)
        grad_summary.append(
            {
                "omega": float(omega_value),
                "gate_1_minus_omega": float(1.0 - omega_value),
                "n_per_fold_mean": float(weights.mean()),
                "mean_abs_p_minus_q": float(
                    np.average([record["mean_abs_p_minus_q"] for record in subset], weights=weights)
                ),
                "mean_teacher_grad_l1": float(
                    np.average([record["mean_teacher_grad_l1"] for record in subset], weights=weights)
                ),
            }
        )

    # ---- teacher reliability per agreement stratum ---------------------------
    teacher_sample_f1 = np.zeros(n_samples, dtype=np.float32)
    for fold in range(int(args.folds)):
        held = np.where(folds == fold)[0]
        thresholds = np.load(output_dir / f"thresholds_fold{fold}_teacher.npy")
        teacher_sample_f1[held] = _per_sample_f1(oof_teacher[held], train_labels[held], thresholds)
    for row in grad_summary:
        mask = np.isclose(np.round(agreement, 3), row["omega"])
        row["n_total"] = int(mask.sum())
        row["teacher_sample_f1"] = float(teacher_sample_f1[mask].mean() * 100.0)
        row["teacher_mAP"] = _subset_map(oof_teacher[mask], train_labels[mask])
        row["base_sample_f1"] = float(sample_f1["baseline"][mask].mean() * 100.0)
        row["utd_minus_base_sample_f1"] = float(
            (sample_f1["utd"][mask] - sample_f1["baseline"][mask]).mean() * 100.0
        )
        row["teacher_minus_base_sample_f1"] = float(
            (teacher_sample_f1[mask] - sample_f1["baseline"][mask]).mean() * 100.0
        )

    # ---- write artifacts -----------------------------------------------------
    _write_csv(output_dir / "e17_quadrant_results.csv", quadrant_rows)
    _write_csv(output_dir / "e17_quadrant_significance.csv", test_rows)
    _write_csv(output_dir / "e17_prop3_gradient_scaling.csv", grad_summary)
    _write_csv(output_dir / "e17_fold_level.csv", [{k: v for k, v in r.items() if k != "thresholds"} for r in fold_records])
    np.savez_compressed(
        output_dir / "e17_oof_scores.npz",
        folds=folds,
        agreement=agreement,
        a_sem=a_sem,
        base_logits=oof_base_logits,
        teacher_probs=oof_teacher,
        slr_logits=oof_slr_logits,
        labels=train_labels,
        **{f"scores_{name}": oof[name] for name in VARIANTS},
        **{f"sample_f1_{name}": sample_f1[name] for name in VARIANTS},
    )

    def _grid(metric: str) -> str:
        lines = [
            f"| Variant | {' | '.join(label for _, label, _, _ in QUADRANTS)} |",
            "| --- | " + " | ".join(["---:"] * len(QUADRANTS)) + " |",
        ]
        for name in VARIANTS:
            cells = []
            for q_key, _, _, _ in QUADRANTS:
                row = next(r for r in quadrant_rows if r["quadrant"] == q_key and r["variant_key"] == name)
                cells.append(f"{row[metric]:.2f}")
            lines.append(f"| {VARIANT_LABELS[name]} | {' | '.join(cells)} |")
        return "\n".join(lines)

    n_line = " | ".join(
        f"{next(r for r in quadrant_rows if r['quadrant'] == q_key)['n']}" for q_key, _, _, _ in QUADRANTS
    )
    report = [
        "# E17 Theory-Predicted 2x2 Ambiguity Interaction",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Protocol",
        "",
        f"- {int(args.folds)}-fold cross-fitting over the Intentonomy train split ({n_samples} images), the only",
        "  split carrying per-sample annotator agreement. Every variant is retrained from scratch on the",
        "  inner folds and predicted on its held-out fold; the concatenation gives one honest score per image.",
        "- Class-wise decision thresholds are fitted per fold on the official validation split only.",
        f"- Supervisory axis: 1 - omega, high = minimum-agreement stratum (omega = 1/3), n = {int(high_sup.sum())}.",
        f"- Semantic axis: negated top-2 margin of the out-of-fold baseline logits, split at its global median"
        f" ({sem_threshold:.4f}), n(high) = {int(high_sem.sum())}.",
        f"- SLR-C: K = {int(args.topk)}, alpha = {float(args.slr_alpha):.2f}. UTD: gate g = 1 - omega, tau ="
        f" {float(args.temperature):.1f}, lambda = {float(args.dynamic_kd_weight):.1f}.",
        "",
        f"Quadrant sizes: {n_line}",
        "",
        "## Per-sample F1 (%)",
        "",
        _grid("sample_f1"),
        "",
        "## Delta per-sample F1 vs Base (%)",
        "",
        _grid("delta_sample_f1"),
        "",
        "## Relative delta per-sample F1 vs Base (%, delta / base)",
        "",
        _grid("rel_delta_sample_f1_pct"),
        "",
        "## Subset mAP",
        "",
        _grid("mAP"),
        "",
        "## Quadrant composition (mean positives / mean omega / mean semantic ambiguity)",
        "",
        _grid("mean_positives"),
        "",
        _grid("mean_omega"),
        "",
        _grid("mean_a_sem"),
        "",
        "## Paired significance vs Base (per-sample F1, Holm-corrected)",
        "",
        "| Quadrant | Variant | n | Delta | t | p (raw) | p (Holm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        report.append(
            "| {quadrant_label} | {variant} | {n} | {delta_sample_f1:+.2f} | {t_stat:.2f} | "
            "{p_raw:.3e} | {p_holm:.3e} |".format(**row)
        )
    report += [
        "",
        "## Proposition 2 check (local margin correction)",
        "",
        f"- Hardest in-candidate confusion pairs evaluated: {prop2['pairs_evaluated']}",
        f"- max |(m' - m) - alpha * Delta_prior|: {prop2['max_abs_identity_residual']:.3e} (identity holds exactly)",
        f"- Prior orders the pair correctly: {prop2['prior_orders_pair_correctly_pct']:.2f}% of pairs",
        f"- Pairs flipped negative->positive margin: {prop2['pairs_flipped_to_correct']}; "
        f"positive->negative: {prop2['pairs_flipped_to_wrong']}; net {prop2['net_pair_corrections']}",
        f"- Max |s' - s| outside the candidate set: {prop2['outside_candidate_max_abs_perturbation']:.3e} (zero perturbation)",
        "",
        "## Proposition 3 check (agreement-dependent teacher gradient)",
        "",
        "| omega | n | gate 1-omega | mean |p - q| (L1) | mean teacher-grad L1 | Base F1 | "
        "Teacher F1 | Teacher mAP | UTD-Base | Teacher-Base |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grad_summary:
        report.append(
            "| {omega:.3f} | {n_total} | {gate_1_minus_omega:.3f} | {mean_abs_p_minus_q:.4f} | "
            "{mean_teacher_grad_l1:.4f} | {base_sample_f1:.2f} | {teacher_sample_f1:.2f} | "
            "{teacher_mAP:.2f} | {utd_minus_base_sample_f1:+.2f} | "
            "{teacher_minus_base_sample_f1:+.2f} |".format(**row)
        )
    report += ["", "## Artifacts", "", f"- Output directory: `{output_dir.relative_to(PROJECT_ROOT)}`", ""]
    report_text = "\n".join(report)
    (output_dir / "REPORT.md").write_text(report_text, encoding="utf-8")

    (output_dir / "summary.json").write_text(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                    "axis_summary": axis_summary,
                    "quadrant_rows": quadrant_rows,
                    "significance_rows": test_rows,
                    "proposition2": prop2,
                    "proposition3": grad_summary,
                    "fold_records": fold_records,
                    "config": vars(args),
                }
            ),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[E17] finished. artifacts saved to {output_dir}")
    print(report_text)


if __name__ == "__main__":
    main()
