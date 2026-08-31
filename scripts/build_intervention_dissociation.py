#!/usr/bin/env python3
"""Branch-contribution / intervention analysis for FDIL (revision E3/R2 support).

We probe whether the SLR-C semantic-prior branch and the UTD-trained residual
branch carry *different* functions by intervening on the trained FDIL model at
inference time and measuring where the score drops, split by class subset:

  - semantic-ambiguous classes  : top-tertile inter-class CLIP-anchor cosine sim
  - supervisory-ambiguous classes: bottom-tertile mean training annotator agreement

Inference-time interventions on the trained ``slr_c_residual_dynamic_kd`` model
(``s_full = SLR(base, prior) + net(image)``):

  - full            : SLR(base, prior) + net(image)
  - remove_prior    : base + net(image)              (alpha -> 0)
  - shuffle_prior   : SLR(base, prior[:, perm]) + net(image)
  - remove_residual : SLR(base, prior)               (drop net)

The double dissociation we expect for R2: removing/shuffling the prior should
hurt the semantic-ambiguous subset most, while removing the residual should hurt
the supervisory-ambiguous subset most.

Test-set annotator agreement is unavailable (test soft labels are binary), so
both subsets are class-level proxies; we report their Jaccard overlap so the
reader can judge how cleanly the two axes separate.
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
    ResidualStudent,
    _apply_slr,
    _build_text_pools,
    _encode_text_pool,
    _json_ready,
    _load_cache_bundle,
    _load_class_names,
    _predict_baseline_logits,
    _resolve_device,
    _set_seed,
    _sigmoid_np,
    _slr_feature_view,
    _text_logits_from_features,
)
from scripts.analyze_privileged_distillation import (  # noqa: E402
    SUBSET2IDS,
    StudentMLP,
    _evaluate_score_bundle,
)

import clip  # type: ignore  # noqa: E402


DEFAULT_FDIL_RUN_DIR = PROJECT_ROOT / "logs" / "analysis" / "distillation_slrc_lcs_topk5_20260327"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDIL branch-contribution / intervention analysis.")
    parser.add_argument("--reuse-cache-dir", type=Path, default=DEFAULT_BASE_CACHE_DIR)
    parser.add_argument("--teacher-run-dir", type=Path, default=DEFAULT_TEACHER_RUN_DIR)
    parser.add_argument("--fdil-run-dir", type=Path, default=DEFAULT_FDIL_RUN_DIR)
    parser.add_argument("--residual-ckpt", type=str, default="slr_c_residual_dynamic_kd_best.pt")
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--gemini-file", type=Path, default=DEFAULT_GEMINI_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--slr-alpha", type=float, default=0.3)
    parser.add_argument("--subset-fraction", type=float, default=1.0 / 3.0,
                        help="Fraction of classes assigned to each ambiguity subset (tertile by default).")
    return parser.parse_args()


def _resolve_output_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"intervention_dissociation_{stamp}"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def _per_class_train_agreement(labels: np.ndarray, soft_labels: np.ndarray) -> np.ndarray:
    """Mean soft-agreement over positive training samples, per class."""
    labels = np.asarray(labels, dtype=np.float32)
    soft = np.asarray(soft_labels, dtype=np.float32)
    out = np.full(labels.shape[1], np.nan, dtype=np.float32)
    for c in range(labels.shape[1]):
        pos = labels[:, c] > 0
        if pos.any():
            out[c] = float(soft[pos, c].mean())
    return out


def _per_class_max_similarity(anchor: np.ndarray) -> np.ndarray:
    """Max off-diagonal cosine similarity of each class anchor to any other class."""
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

    val_feats = np.asarray(val_clip["features"], dtype=np.float32)
    test_feats = np.asarray(test_clip["features"], dtype=np.float32)
    val_labels = np.asarray(val_clip["labels"], dtype=np.float32)
    test_labels = np.asarray(test_clip["labels"], dtype=np.float32)
    num_classes = int(test_labels.shape[1])

    class_names = _load_class_names(args.annotation_file)

    # ---- semantic prior (lexical + canonical + scenario), per E3/FDIL ----
    text_pools = _build_text_pools(class_names, args.gemini_file)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    lexical = _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True)
    canonical = _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True)
    scenario = _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False)

    def prior_logits(bundle: Mapping[str, Any]) -> np.ndarray:
        feats = _slr_feature_view(bundle)
        return (
            _text_logits_from_features(feats, lexical, logit_scale)
            + _text_logits_from_features(feats, canonical, logit_scale)
            + _text_logits_from_features(feats, scenario, logit_scale)
        ) / 3.0

    val_prior = prior_logits(val_clip)
    test_prior = prior_logits(test_clip)

    # ---- baseline visual logits ----
    baseline_state = torch.load(args.teacher_run_dir / "baseline_best.pt", map_location="cpu", weights_only=True)
    baseline_model = StudentMLP(
        image_dim=val_feats.shape[1], hidden_dim=768, num_classes=num_classes, dropout=0.1, feature_proj_dim=256,
    ).to(device)
    baseline_model.load_state_dict(baseline_state, strict=False)
    val_base = _predict_baseline_logits(baseline_model, val_feats, device, int(args.batch_size))
    test_base = _predict_baseline_logits(baseline_model, test_feats, device, int(args.batch_size))

    # ---- trained FDIL residual student (UTD-distilled) ----
    residual = ResidualStudent(
        image_dim=val_feats.shape[1], hidden_dim=int(args.hidden_dim), num_classes=num_classes, dropout=float(args.dropout),
    ).to(device)
    residual.load_state_dict(
        torch.load(args.fdil_run_dir / args.residual_ckpt, map_location="cpu", weights_only=True), strict=True
    )
    residual.eval()

    @torch.inference_mode()
    def residual_term(feats: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, feats.shape[0], int(args.batch_size)):
            t = torch.as_tensor(feats[s:s + int(args.batch_size)], dtype=torch.float32, device=device)
            out.append(residual.net(t).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    val_res = residual_term(val_feats)
    test_res = residual_term(test_feats)

    # ---- class subsets ----
    train_agreement = _per_class_train_agreement(train_clip["labels"], train_clip["soft_labels"])
    anchor = _l2(lexical) + _l2(canonical) + _l2(scenario)
    max_sim = _per_class_max_similarity(anchor)
    k = max(1, int(round(num_classes * float(args.subset_fraction))))
    supervisory_ids = sorted(np.argsort(train_agreement)[:k].tolist())            # lowest agreement
    semantic_ids = sorted(np.argsort(-max_sim)[:k].tolist())                      # highest similarity
    sup_set, sem_set = set(supervisory_ids), set(semantic_ids)
    jaccard = len(sup_set & sem_set) / len(sup_set | sem_set) if (sup_set | sem_set) else 0.0

    # ---- interventions (inference-time) ----
    perm = np.random.RandomState(int(args.seed)).permutation(num_classes)

    def slr(base: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
        return _apply_slr(base, prior, topk=int(args.topk), alpha=float(alpha))

    interventions = {
        "full": (
            slr(val_base, val_prior, args.slr_alpha) + val_res,
            slr(test_base, test_prior, args.slr_alpha) + test_res,
        ),
        "remove_prior": (
            val_base + val_res,
            test_base + test_res,
        ),
        "shuffle_prior": (
            slr(val_base, val_prior[:, perm], args.slr_alpha) + val_res,
            slr(test_base, test_prior[:, perm], args.slr_alpha) + test_res,
        ),
        "remove_residual": (
            slr(val_base, val_prior, args.slr_alpha),
            slr(test_base, test_prior, args.slr_alpha),
        ),
    }

    rows: list[dict[str, Any]] = []
    bundles: dict[str, Any] = {}
    for name, (vlog, tlog) in interventions.items():
        bundle = _evaluate_score_bundle(_sigmoid_np(vlog), val_labels, _sigmoid_np(tlog), test_labels)
        m = bundle["classwise"]["test"]
        pcf1 = np.asarray(m["per_class_f1"], dtype=np.float32)
        rows.append({
            "intervention": name,
            "macro": round(float(m["macro"]) * 100.0, 2),
            "avg_f1": round((float(m["macro"]) + float(m["micro"]) + float(m["samples"])) / 3.0 * 100.0, 2),
            "mAP": round(float(m["mAP"]), 2),
            "hard": round(float(m["hard"]) * 100.0, 2),
            "semantic_amb_f1": round(_subset_mean(pcf1, semantic_ids), 2),
            "supervisory_amb_f1": round(_subset_mean(pcf1, supervisory_ids), 2),
        })
        bundles[name] = pcf1.tolist()

    # ---- drop relative to full ----
    full = next(r for r in rows if r["intervention"] == "full")
    drop_rows = []
    for r in rows:
        if r["intervention"] == "full":
            continue
        drop_rows.append({
            "intervention": r["intervention"],
            "overall_macro_drop": round(full["macro"] - r["macro"], 2),
            "semantic_amb_drop": round(full["semantic_amb_f1"] - r["semantic_amb_f1"], 2),
            "supervisory_amb_drop": round(full["supervisory_amb_f1"] - r["supervisory_amb_f1"], 2),
        })

    _write_csv(output_dir / "intervention_metrics.csv", rows)
    _write_csv(output_dir / "intervention_drops.csv", drop_rows)

    subset_info = {
        "num_classes": num_classes,
        "subset_size": k,
        "semantic_ambiguous_ids": semantic_ids,
        "semantic_ambiguous_names": [str(class_names[i]) for i in semantic_ids],
        "supervisory_ambiguous_ids": supervisory_ids,
        "supervisory_ambiguous_names": [str(class_names[i]) for i in supervisory_ids],
        "jaccard_overlap": round(jaccard, 3),
        "overlap_ids": sorted(sup_set & sem_set),
    }

    # ---- markdown report ----
    lines = [
        "# FDIL Branch-Contribution / Intervention Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Inference-time interventions on the trained `slr_c_residual_dynamic_kd` FDIL model "
        "(`s_full = SLR(base, prior) + net(image)`); frozen CLIP ViT-L/14 cache, validation-only "
        "class-wise thresholds searched independently per intervention.",
        "- Test-set annotator agreement is unavailable (test soft labels are binary), so both "
        "ambiguity subsets are **class-level proxies**.",
        "",
        "## Class subsets",
        "",
        f"- Semantic-ambiguous (top-tertile inter-class anchor cosine sim), {k} classes: "
        + ", ".join(subset_info["semantic_ambiguous_names"]),
        f"- Supervisory-ambiguous (bottom-tertile mean train agreement), {k} classes: "
        + ", ".join(subset_info["supervisory_ambiguous_names"]),
        f"- **Jaccard overlap between the two subsets: {jaccard:.3f}** "
        f"(shared classes: {[str(class_names[i]) for i in sorted(sup_set & sem_set)]})",
        "",
        "## Per-intervention metrics",
        "",
        "| Intervention | Macro | AvgF1 | mAP | Hard | Semantic-amb F1 | Supervisory-amb F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {intervention} | {macro:.2f} | {avg_f1:.2f} | {mAP:.2f} | {hard:.2f} | "
            "{semantic_amb_f1:.2f} | {supervisory_amb_f1:.2f} |".format(**r)
        )
    lines += [
        "",
        "## Drop relative to full FDIL (positive = worse without the branch)",
        "",
        "| Intervention | Overall Macro drop | Semantic-amb drop | Supervisory-amb drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in drop_rows:
        lines.append(
            "| {intervention} | {overall_macro_drop:.2f} | {semantic_amb_drop:.2f} | "
            "{supervisory_amb_drop:.2f} |".format(**r)
        )
    lines += [
        "",
        "## Expected double-dissociation",
        "",
        "- remove_prior / shuffle_prior -> large **semantic-amb** drop, small supervisory-amb drop.",
        "- remove_residual -> large **supervisory-amb** drop, small semantic-amb drop.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready({
            "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            "config": {
                "reuse_cache_dir": str(args.reuse_cache_dir),
                "fdil_run_dir": str(args.fdil_run_dir),
                "residual_ckpt": args.residual_ckpt,
                "topk": int(args.topk),
                "slr_alpha": float(args.slr_alpha),
                "seed": int(args.seed),
                "subset_fraction": float(args.subset_fraction),
            },
            "subsets": subset_info,
            "metrics": rows,
            "drops": drop_rows,
            "per_class_f1": bundles,
            "per_class_train_agreement": train_agreement.tolist(),
            "per_class_max_similarity": max_sim.tolist(),
        }), indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n[intervention] artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
