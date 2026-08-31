#!/usr/bin/env python3
"""Retrain-based branch-contribution + E14 negative-control analysis for FDIL.

Everything is retrained self-consistently on the *current* frozen CLIP ViT-L/14
cache (avoiding the stale-checkpoint drift of pure inference-time ablation), then
scored per class subset to test whether the SLR-C prior branch and the UTD
distillation/gating carry different functions.

Conditions (each = one retrained ResidualStudent, FDIL hyper-params):
  branch contribution
    full            : SLR(base, prior) + dynamic gated KD (real teacher)        [reference]
    no_prior        : base (no SLR)    + dynamic gated KD
    no_utd          : SLR(base, prior) + supervised only (no distillation)
  E14 negative controls (vs full)
    shuffled_prior  : SLR(base, class-permuted prior) + dynamic gated KD
    shuffled_rationale: SLR(base, prior) + dynamic gated KD, teacher rows permuted
    ungated         : SLR(base, prior) + standard (ungated) KD
    uniform_gate    : SLR(base, prior) + dynamic KD with constant agreement

Subsets (class-level proxies; test agreement is unavailable):
  semantic-ambiguous  = top-tertile inter-class CLIP-anchor cosine similarity
  supervisory-ambiguous = bottom-tertile mean training annotator agreement
We report the Jaccard overlap of the two subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

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
    _predict_baseline_logits,
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
    StudentMLP,
    TeacherMLP,
    _compute_sample_agreement,
)

import clip  # type: ignore  # noqa: E402

DEFAULT_TEACHER_TEXT = DEFAULT_TEXT_DIR / "rationale_full_bge_features.npz"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain-based FDIL branch / E14 dissociation analysis.")
    p.add_argument("--reuse-cache-dir", type=Path, default=DEFAULT_BASE_CACHE_DIR)
    p.add_argument("--teacher-run-dir", type=Path, default=DEFAULT_TEACHER_RUN_DIR)
    p.add_argument("--train-text-npz", type=Path, default=DEFAULT_TEACHER_TEXT)
    p.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    p.add_argument("--gemini-file", type=Path, default=DEFAULT_GEMINI_FILE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--standard-kd-weight", type=float, default=1.0)
    p.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--slr-alpha", type=float, default=0.3)
    p.add_argument("--subset-fraction", type=float, default=1.0 / 3.0)
    return p.parse_args()


def _resolve_output_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"e14_retrain_dissociation_{stamp}"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def _per_class_train_agreement(labels: np.ndarray, soft_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    soft = np.asarray(soft_labels, dtype=np.float32)
    out = np.full(labels.shape[1], np.nan, dtype=np.float32)
    for c in range(labels.shape[1]):
        pos = labels[:, c] > 0
        if pos.any():
            out[c] = float(soft[pos, c].mean())
    return out


def _per_class_max_similarity(anchor: np.ndarray) -> np.ndarray:
    a = _l2(np.asarray(anchor, dtype=np.float32))
    sim = a @ a.T
    np.fill_diagonal(sim, -np.inf)
    return sim.max(axis=1).astype(np.float32)


def _subset_mean(per_class_f1: np.ndarray, ids: Sequence[int]) -> float:
    ids = [i for i in ids if i < len(per_class_f1)]
    if not ids:
        return float("nan")
    return float(np.mean(np.asarray(per_class_f1, dtype=np.float32)[ids]) * 100.0)


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))

    train_clip = _load_cache_bundle(args.reuse_cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(args.reuse_cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(args.reuse_cache_dir / "test_clip.npz")
    train_text = _align_text_bundle_to_clip(
        _load_text_bundle(args.train_text_npz, required_keys=["image_ids", "features"]), train_clip
    )

    train_feats = np.asarray(train_clip["features"], dtype=np.float32)
    val_feats = np.asarray(val_clip["features"], dtype=np.float32)
    test_feats = np.asarray(test_clip["features"], dtype=np.float32)
    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    val_labels = np.asarray(val_clip["labels"], dtype=np.float32)
    test_labels = np.asarray(test_clip["labels"], dtype=np.float32)
    train_soft = np.asarray(train_clip["soft_labels"], dtype=np.float32)
    num_classes = int(train_labels.shape[1])
    image_dim = int(train_feats.shape[1])

    class_names = _load_class_names(args.annotation_file)

    # ---- semantic prior (lexical + canonical + scenario) ----
    pools = _build_text_pools(class_names, args.gemini_file)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    lexical = _encode_text_pool(clip_model, pools["lexical"], wrap_prompt=True)
    canonical = _encode_text_pool(clip_model, pools["canonical"], wrap_prompt=True)
    scenario = _encode_text_pool(clip_model, pools["scenario"], wrap_prompt=False)

    def prior_logits(bundle: Mapping[str, Any]) -> np.ndarray:
        f = _slr_feature_view(bundle)
        return (
            _text_logits_from_features(f, lexical, logit_scale)
            + _text_logits_from_features(f, canonical, logit_scale)
            + _text_logits_from_features(f, scenario, logit_scale)
        ) / 3.0

    train_prior, val_prior, test_prior = prior_logits(train_clip), prior_logits(val_clip), prior_logits(test_clip)

    # ---- base logits ----
    baseline_model = StudentMLP(image_dim=image_dim, hidden_dim=768, num_classes=num_classes, dropout=0.1, feature_proj_dim=256).to(device)
    baseline_model.load_state_dict(
        torch.load(args.teacher_run_dir / "baseline_best.pt", map_location="cpu", weights_only=True), strict=False
    )
    train_base = _predict_baseline_logits(baseline_model, train_feats, device, int(args.batch_size))
    val_base = _predict_baseline_logits(baseline_model, val_feats, device, int(args.batch_size))
    test_base = _predict_baseline_logits(baseline_model, test_feats, device, int(args.batch_size))

    def slr(base, prior):
        return _apply_slr(base, prior, topk=int(args.topk), alpha=float(args.slr_alpha))

    full_slr = (slr(train_base, train_prior), slr(val_base, val_prior), slr(test_base, test_prior))
    perm = np.random.RandomState(int(args.seed)).permutation(num_classes)
    shuf_slr = (slr(train_base, train_prior[:, perm]), slr(val_base, val_prior[:, perm]), slr(test_base, test_prior[:, perm]))
    base_only = (train_base.copy(), val_base.copy(), test_base.copy())

    # ---- teacher (real rationale) ----
    teacher = TeacherMLP(
        text_dim=int(np.asarray(train_text["features"], dtype=np.float32).shape[1]),
        hidden_dim=1024, num_classes=num_classes, dropout=0.1, input_mode="text_only",
    )
    teacher.load_state_dict(torch.load(args.teacher_run_dir / "teacher_best.pt", map_location="cpu", weights_only=True), strict=False)
    teacher_probs = _predict_teacher(teacher, np.asarray(train_text["features"], dtype=np.float32), device, int(args.batch_size))
    teacher_probs = _sigmoid_np(_logit_np(teacher_probs) / float(args.temperature)).astype(np.float32)
    rng = np.random.RandomState(int(args.seed) + 7)
    teacher_probs_shuffled = teacher_probs[rng.permutation(teacher_probs.shape[0])].copy()

    agreement = _compute_sample_agreement(train_labels, train_soft, mode="min").astype(np.float32)
    agreement_uniform = np.full_like(agreement, float(agreement.mean()))

    # ---- class subsets ----
    train_agreement_pc = _per_class_train_agreement(train_labels, train_soft)
    anchor = _l2(lexical) + _l2(canonical) + _l2(scenario)
    max_sim = _per_class_max_similarity(anchor)
    k = max(1, int(round(num_classes * float(args.subset_fraction))))
    supervisory_ids = sorted(np.argsort(train_agreement_pc)[:k].tolist())
    semantic_ids = sorted(np.argsort(-max_sim)[:k].tolist())
    sup_set, sem_set = set(supervisory_ids), set(semantic_ids)
    jaccard = len(sup_set & sem_set) / len(sup_set | sem_set) if (sup_set | sem_set) else 0.0

    # ---- conditions ----
    # (mode, slr_triplet, teacher_probs, agreement)
    conditions = {
        "full":               ("dynamic_kd", full_slr, teacher_probs, agreement),
        "no_prior":           ("dynamic_kd", base_only, teacher_probs, agreement),
        "no_utd":             ("supervised", full_slr, teacher_probs, agreement),
        "shuffled_prior":     ("dynamic_kd", shuf_slr, teacher_probs, agreement),
        "shuffled_rationale": ("dynamic_kd", full_slr, teacher_probs_shuffled, agreement),
        "ungated":            ("standard_kd", full_slr, teacher_probs, agreement),
        "uniform_gate":       ("dynamic_kd", full_slr, teacher_probs, agreement_uniform),
    }

    rows: list[dict[str, Any]] = []
    per_class_f1: dict[str, list[float]] = {}
    for offset, (name, (mode, slr_triplet, tprobs, agree)) in enumerate(conditions.items()):
        tr_slr, va_slr, te_slr = slr_triplet
        seed = _set_component_seed(int(args.seed), offset=100 + 50 * offset)
        dataset = SLRCDataset(
            image_features=train_feats, slr_logits=tr_slr, labels=train_labels,
            soft_labels=train_soft, agreement=agree, teacher_probs=tprobs,
        )
        model = ResidualStudent(image_dim=image_dim, hidden_dim=int(args.hidden_dim), num_classes=num_classes, dropout=float(args.dropout)).to(device)
        print(f"[E14] training '{name}' (mode={mode}) seed={seed}")
        result = _train_residual_student(
            mode=mode, model=model, train_dataset=dataset,
            val_image_features=val_feats, val_slr_logits=va_slr, val_targets=val_labels,
            test_image_features=test_feats, test_slr_logits=te_slr, test_targets=test_labels,
            device=device, args=args, loader_seed=seed,
        )
        m = result["bundle"]["classwise"]["test"]
        pcf1 = np.asarray(m["per_class_f1"], dtype=np.float32)
        per_class_f1[name] = pcf1.tolist()
        rows.append({
            "condition": name,
            "mode": mode,
            "macro": round(float(m["macro"]) * 100.0, 2),
            "micro": round(float(m["micro"]) * 100.0, 2),
            "samples": round(float(m["samples"]) * 100.0, 2),
            "avg_f1": round((float(m["macro"]) + float(m["micro"]) + float(m["samples"])) / 3.0 * 100.0, 2),
            "mAP": round(float(m["mAP"]), 2),
            "hard": round(float(m["hard"]) * 100.0, 2),
            "semantic_amb_f1": round(_subset_mean(pcf1, semantic_ids), 2),
            "supervisory_amb_f1": round(_subset_mean(pcf1, supervisory_ids), 2),
            "best_epoch": int(result["best_epoch"]),
        })

    full = next(r for r in rows if r["condition"] == "full")
    drops = []
    for r in rows:
        if r["condition"] == "full":
            continue
        drops.append({
            "condition": r["condition"],
            "overall_macro_drop": round(full["macro"] - r["macro"], 2),
            "semantic_amb_drop": round(full["semantic_amb_f1"] - r["semantic_amb_f1"], 2),
            "supervisory_amb_drop": round(full["supervisory_amb_f1"] - r["supervisory_amb_f1"], 2),
        })

    _write_csv(output_dir / "e14_metrics.csv", rows)
    _write_csv(output_dir / "e14_drops.csv", drops)

    subset_info = {
        "subset_size": k, "jaccard_overlap": round(jaccard, 3),
        "semantic_ambiguous": [str(class_names[i]) for i in semantic_ids],
        "supervisory_ambiguous": [str(class_names[i]) for i in supervisory_ids],
        "overlap": [str(class_names[i]) for i in sorted(sup_set & sem_set)],
    }

    lines = [
        "# FDIL Retrain-Based Branch / E14 Dissociation",
        "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
        "## Scope", "",
        "- Every condition is retrained self-consistently on the current frozen CLIP ViT-L/14 cache; "
        "FDIL hyper-params; validation-only class-wise thresholds searched per condition.",
        "- Subsets are class-level proxies (test agreement unavailable).", "",
        "## Class subsets", "",
        f"- Semantic-ambiguous (top-tertile anchor cosine sim), {k} classes: " + ", ".join(subset_info["semantic_ambiguous"]),
        f"- Supervisory-ambiguous (bottom-tertile train agreement), {k} classes: " + ", ".join(subset_info["supervisory_ambiguous"]),
        f"- **Jaccard overlap: {jaccard:.3f}** (shared: {subset_info['overlap']})", "",
        "## Per-condition metrics", "",
        "| Condition | Mode | Macro | Micro | Samples | AvgF1 | mAP | Hard | Semantic-amb F1 | Supervisory-amb F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append("| {condition} | {mode} | {macro:.2f} | {micro:.2f} | {samples:.2f} | {avg_f1:.2f} | {mAP:.2f} | {hard:.2f} | {semantic_amb_f1:.2f} | {supervisory_amb_f1:.2f} |".format(**r))
    lines += [
        "", "## Drop relative to full FDIL (positive = worse without it)", "",
        "| Condition | Overall Macro drop | Semantic-amb drop | Supervisory-amb drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in drops:
        lines.append("| {condition} | {overall_macro_drop:.2f} | {semantic_amb_drop:.2f} | {supervisory_amb_drop:.2f} |".format(**r))
    lines += [
        "", "## Reading", "",
        "- Branch contribution: `no_prior` vs `no_utd` drops, split by subset, indicate whether the "
        "prior and the UTD distillation specialize on semantic- vs supervisory-ambiguous classes.",
        "- E14 controls (`shuffled_prior`, `shuffled_rationale`, `ungated`, `uniform_gate`) test whether "
        "the gains require the *structured* prior / rationale / agreement gate rather than any text or any gate.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready({
            "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            "config": {k2: (str(v) if isinstance(v, Path) else v) for k2, v in vars(args).items()},
            "subsets": subset_info,
            "semantic_ambiguous_ids": semantic_ids,
            "supervisory_ambiguous_ids": supervisory_ids,
            "metrics": rows, "drops": drops, "per_class_f1": per_class_f1,
            "per_class_train_agreement": train_agreement_pc.tolist(),
            "per_class_max_similarity": max_sim.tolist(),
        }), indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n[E14] artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
