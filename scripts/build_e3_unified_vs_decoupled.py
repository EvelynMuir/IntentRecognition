#!/usr/bin/env python3
"""Build E3 unified-vs-decoupled evidence for the FDIL revision.

The E3 control asks whether a single undifferentiated mechanism can handle both
semantic ambiguity and annotator-disagreement ambiguity as well as the FDIL
decomposition.  This script keeps the backbone cache, teacher, semantic prior,
validation-only thresholds, and training budget aligned with the existing
revision runs, then compares:

1. UTD only: one uncertainty-oriented mechanism.
2. SLR-C only: one semantic-prior mechanism.
3. Unified joint-target KD: one student distilled from an averaged SLR-C/teacher
   target without agreement gating or residual decomposition.
4. Decoupled FDIL: SLR-C handles semantic candidate construction while gated
   residual distillation handles disagreement-sensitive correction.
"""

from __future__ import annotations

# --- fabricate training-only deps so pure helpers import in a clip-only env ----
# Mirrors build_e13_threshold_free_significance.py: the s2d env has torch+clip but
# not lightning/rich/hydra. The score-reconstruction helpers we import below only
# need the numeric path, so stub the training-only roots with no-op modules.
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
            return a[0]  # behave as a no-op decorator
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_distillation_slrc import (  # noqa: E402
    DEFAULT_ANNOTATION_FILE,
    DEFAULT_BASE_CACHE_DIR,
    DEFAULT_GEMINI_FILE,
    DEFAULT_TEACHER_RUN_DIR,
    DEFAULT_TEXT_DIR,
    _align_text_bundle_to_clip,
    _apply_slr,
    _build_text_pools,
    _comparison_row,
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
)
from scripts.analyze_privileged_distillation import (  # noqa: E402
    StudentDataset,
    StudentMLP,
    TeacherMLP,
    _compute_sample_agreement,
    _train_student,
)

import clip  # type: ignore  # noqa: E402


DEFAULT_FDIL_SUMMARY = PROJECT_ROOT / "logs" / "analysis" / "distillation_slrc_lcs_topk5_20260327" / "summary.json"
DEFAULT_UTD_SUMMARY = (
    PROJECT_ROOT / "logs" / "analysis" / "privileged_distillation_text_teacher_seedfix_20260316" / "summary.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E3 unified-vs-decoupled comparison artifacts.")
    parser.add_argument("--reuse-cache-dir", type=Path, default=DEFAULT_BASE_CACHE_DIR)
    parser.add_argument("--teacher-run-dir", type=Path, default=DEFAULT_TEACHER_RUN_DIR)
    parser.add_argument("--fdil-summary", type=Path, default=DEFAULT_FDIL_SUMMARY)
    parser.add_argument("--utd-summary", type=Path, default=DEFAULT_UTD_SUMMARY)
    parser.add_argument("--train-text-npz", type=Path, default=DEFAULT_TEXT_DIR / "rationale_full_bge_features.npz")
    parser.add_argument("--val-text-npz", type=Path, default=DEFAULT_TEXT_DIR / "val_rationale_baseline_pred_bge_features.npz")
    parser.add_argument("--test-text-npz", type=Path, default=DEFAULT_TEXT_DIR / "test_rationale_baseline_pred_bge_features.npz")
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--gemini-file", type=Path, default=DEFAULT_GEMINI_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-proj-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--standard-kd-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-kd-variant", type=str, default="sample_inverse")
    parser.add_argument("--dynamic-gate-alpha", type=float, default=0.3)
    parser.add_argument("--dynamic-gate-beta", type=float, default=0.7)
    parser.add_argument("--entropy-gate-lambda", type=float, default=1.0)
    parser.add_argument("--feature-distill-mode", type=str, default="none")
    parser.add_argument("--feature-distill-weight", type=float, default=0.0)
    parser.add_argument("--feature-distill-temperature", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--slr-alpha", type=float, default=0.3)
    parser.add_argument(
        "--unified-target-weight",
        type=float,
        default=0.5,
        help="Weight on teacher probabilities in the unified KD target. The rest comes from SLR-C probabilities.",
    )
    return parser.parse_args()


def _resolve_output_dir(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"e3_unified_vs_decoupled_{stamp}"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def _avg_f1(row: Mapping[str, Any]) -> float:
    return (float(row["macro"]) + float(row["micro"]) + float(row["samples"])) / 3.0


def _row_from_bundle(
    method: str,
    mechanism: str,
    bundle: Mapping[str, Any],
    note: str,
) -> dict[str, Any]:
    metrics = bundle["classwise"]["test"]
    row = {
        "method": method,
        "mechanism": mechanism,
        "macro": float(metrics["macro"]) * 100.0,
        "micro": float(metrics["micro"]) * 100.0,
        "samples": float(metrics["samples"]) * 100.0,
        "avg_f1": 0.0,
        "mAP": float(metrics["mAP"]),
        "hard": float(metrics["hard"]) * 100.0,
        "note": note,
    }
    row["avg_f1"] = _avg_f1(row)
    return row


def _summary_row(summary: Mapping[str, Any], method_key: str, method: str, mechanism: str, note: str) -> dict[str, Any]:
    payload = summary[method_key]
    bundle = payload["bundle"] if isinstance(payload, Mapping) and "bundle" in payload else payload
    return _row_from_bundle(method, mechanism, bundle, note)


def _encode_lcs_prior(
    class_names: Sequence[str],
    gemini_file: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, float, np.ndarray, np.ndarray, np.ndarray]:
    text_pools = _build_text_pools(class_names, gemini_file)
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    lexical_embeddings = _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True)
    canonical_embeddings = _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True)
    scenario_embeddings = _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False)
    return clip_model, logit_scale, lexical_embeddings, canonical_embeddings, scenario_embeddings


def _lcs_prior_logits(
    image_features: np.ndarray,
    logit_scale: float,
    lexical_embeddings: np.ndarray,
    canonical_embeddings: np.ndarray,
    scenario_embeddings: np.ndarray,
) -> np.ndarray:
    return (
        _text_logits_from_features(image_features, lexical_embeddings, logit_scale)
        + _text_logits_from_features(image_features, canonical_embeddings, logit_scale)
        + _text_logits_from_features(image_features, scenario_embeddings, logit_scale)
    ) / 3.0


def _make_markdown_report(rows: Sequence[Mapping[str, Any]], output_dir: Path, args: argparse.Namespace) -> str:
    decoupled = next(row for row in rows if row["method"] == "Decoupled FDIL")
    unified = next(row for row in rows if row["method"] == "Unified joint-target KD")
    delta_macro = float(decoupled["macro"]) - float(unified["macro"])
    delta_avg = float(decoupled["avg_f1"]) - float(unified["avg_f1"])
    delta_hard = float(decoupled["hard"]) - float(unified["hard"])

    lines = [
        "# E3 Unified vs Decoupled",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- All rows use frozen CLIP ViT-L/14 cache, the Intentonomy train/val/test split, and validation-only class-wise threshold selection.",
        "- The unified joint-target control averages the semantic SLR-C probabilities and rationale-teacher probabilities into one KD target, then trains a single image-feature student without residual SLR-C decomposition or agreement-aware gating.",
        "- Decoupled FDIL keeps the existing functional split: SLR-C constructs semantic candidates, and agreement-aware residual distillation corrects uncertainty-sensitive predictions.",
        "",
        "## Main Table",
        "",
        "| Method | Mechanism | Macro | Micro | Samples | AvgF1 | mAP | Hard | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {mechanism} | {macro:.2f} | {micro:.2f} | {samples:.2f} | "
            "{avg_f1:.2f} | {mAP:.2f} | {hard:.2f} | {note} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## E3 Contrast",
            "",
            "| Comparison | Macro Delta | AvgF1 Delta | Hard Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Decoupled FDIL - unified joint-target KD | {delta_macro:.2f} | {delta_avg:.2f} | {delta_hard:.2f} |",
            "",
            "## Artifacts",
            "",
            f"- Output directory: `{output_dir.relative_to(PROJECT_ROOT)}`",
            f"- Unified target teacher weight: `{float(args.unified_target_weight):.2f}`",
            f"- SLR-C prior: lexical + canonical + scenario, `K={int(args.topk)}`, `alpha={float(args.slr_alpha):.2f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))

    fdil_summary = _load_json(args.fdil_summary)
    utd_summary = _load_json(args.utd_summary)

    train_clip = _load_cache_bundle(args.reuse_cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(args.reuse_cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(args.reuse_cache_dir / "test_clip.npz")

    train_text = _align_text_bundle_to_clip(
        _load_text_bundle(args.train_text_npz, required_keys=["image_ids", "features"]),
        train_clip,
    )
    val_text = _align_text_bundle_to_clip(
        _load_text_bundle(args.val_text_npz, required_keys=["image_ids", "features"]),
        val_clip,
    )
    test_text = _align_text_bundle_to_clip(
        _load_text_bundle(args.test_text_npz, required_keys=["image_ids", "features"]),
        test_clip,
    )

    class_names = _load_class_names(args.annotation_file)
    _, logit_scale, lexical_embeddings, canonical_embeddings, scenario_embeddings = _encode_lcs_prior(
        class_names=class_names,
        gemini_file=args.gemini_file,
        device=device,
    )

    baseline_state = torch.load(args.teacher_run_dir / "baseline_best.pt", map_location="cpu", weights_only=True)
    baseline_model = StudentMLP(
        image_dim=int(np.asarray(train_clip["features"], dtype=np.float32).shape[1]),
        hidden_dim=768,
        num_classes=int(np.asarray(train_clip["labels"], dtype=np.float32).shape[1]),
        dropout=0.1,
        feature_proj_dim=256,
    ).to(device)
    baseline_model.load_state_dict(baseline_state, strict=False)

    train_base_logits = _predict_baseline_logits(
        baseline_model, np.asarray(train_clip["features"], dtype=np.float32), device, int(args.batch_size)
    )
    val_base_logits = _predict_baseline_logits(
        baseline_model, np.asarray(val_clip["features"], dtype=np.float32), device, int(args.batch_size)
    )
    test_base_logits = _predict_baseline_logits(
        baseline_model, np.asarray(test_clip["features"], dtype=np.float32), device, int(args.batch_size)
    )

    train_prior_logits = _lcs_prior_logits(
        _slr_feature_view(train_clip), logit_scale, lexical_embeddings, canonical_embeddings, scenario_embeddings
    )
    val_prior_logits = _lcs_prior_logits(
        _slr_feature_view(val_clip), logit_scale, lexical_embeddings, canonical_embeddings, scenario_embeddings
    )
    test_prior_logits = _lcs_prior_logits(
        _slr_feature_view(test_clip), logit_scale, lexical_embeddings, canonical_embeddings, scenario_embeddings
    )
    train_slr_logits = _apply_slr(train_base_logits, train_prior_logits, topk=int(args.topk), alpha=float(args.slr_alpha))
    val_slr_logits = _apply_slr(val_base_logits, val_prior_logits, topk=int(args.topk), alpha=float(args.slr_alpha))
    test_slr_logits = _apply_slr(test_base_logits, test_prior_logits, topk=int(args.topk), alpha=float(args.slr_alpha))

    teacher_state = torch.load(args.teacher_run_dir / "teacher_best.pt", map_location="cpu", weights_only=True)
    teacher_model = TeacherMLP(
        text_dim=int(np.asarray(train_text["features"], dtype=np.float32).shape[1]),
        hidden_dim=1024,
        num_classes=int(np.asarray(train_clip["labels"], dtype=np.float32).shape[1]),
        dropout=0.1,
        input_mode="text_only",
    ).to(device)
    teacher_model.load_state_dict(teacher_state, strict=False)

    train_teacher_probs = _predict_teacher(
        teacher_model=teacher_model,
        text_features=np.asarray(train_text["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )
    train_teacher_probs = _sigmoid_np(_logit_np(train_teacher_probs) / float(args.temperature))
    train_slr_probs = _sigmoid_np(train_slr_logits)
    teacher_weight = float(args.unified_target_weight)
    unified_teacher_probs = (
        teacher_weight * train_teacher_probs + (1.0 - teacher_weight) * train_slr_probs
    ).astype(np.float32)

    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    train_soft_labels = np.asarray(train_clip["soft_labels"], dtype=np.float32)
    train_agreement = _compute_sample_agreement(train_labels, train_soft_labels, mode="min")
    unified_dataset = StudentDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        labels=train_labels,
        agreement=train_agreement,
        soft_labels=train_soft_labels,
        teacher_probs=unified_teacher_probs,
    )

    unified_seed = _set_component_seed(int(args.seed), offset=500)
    unified_model = StudentMLP(
        image_dim=int(np.asarray(train_clip["features"], dtype=np.float32).shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_classes=int(train_labels.shape[1]),
        dropout=float(args.dropout),
        feature_proj_dim=int(args.feature_proj_dim),
    ).to(device)
    print(f"[E3] training unified joint-target KD seed={unified_seed}")
    unified_result = _train_student(
        mode="standard_kd",
        model=unified_model,
        train_dataset=unified_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_targets=np.asarray(val_clip["labels"], dtype=np.float32),
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_targets=np.asarray(test_clip["labels"], dtype=np.float32),
        device=device,
        args=args,
    )

    torch.save(unified_result["state_dict"], output_dir / "unified_joint_target_kd_best.pt")

    rows = [
        _summary_row(
            utd_summary,
            method_key="dynamic_gated_kd",
            method="Unified UTD only",
            mechanism="single uncertainty distillation",
            note="existing E2 seedfix run",
        ),
        _summary_row(
            fdil_summary,
            method_key="slr_c_fixed",
            method="Unified SLR-C only",
            mechanism="single semantic prior reranking",
            note="existing FDIL LCS K=5 run",
        ),
        _row_from_bundle(
            "Unified joint-target KD",
            "single mixed target, no gate/residual split",
            unified_result["bundle"],
            f"best_epoch={unified_result['best_epoch']}",
        ),
        _summary_row(
            fdil_summary,
            method_key="slr_c_residual_dynamic_kd",
            method="Decoupled FDIL",
            mechanism="SLR-C + agreement-gated residual UTD",
            note="existing FDIL LCS K=5 run",
        ),
    ]

    _write_csv(output_dir / "e3_unified_vs_decoupled.csv", rows)
    report = _make_markdown_report(rows, output_dir, args)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "comparison_rows": rows,
                "unified_joint_target_kd": _json_ready({k: v for k, v in unified_result.items() if k != "state_dict"}),
                "source_summaries": {
                    "fdil_summary": str(args.fdil_summary),
                    "utd_summary": str(args.utd_summary),
                },
                "config": {
                    "seed": int(args.seed),
                    "unified_seed": unified_seed,
                    "reuse_cache_dir": str(args.reuse_cache_dir),
                    "teacher_run_dir": str(args.teacher_run_dir),
                    "topk": int(args.topk),
                    "slr_alpha": float(args.slr_alpha),
                    "temperature": float(args.temperature),
                    "unified_target_weight": float(args.unified_target_weight),
                    "max_epochs": int(args.max_epochs),
                    "patience": int(args.patience),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[E3] finished. artifacts saved to {output_dir}")
    print(report)


if __name__ == "__main__":
    main()
