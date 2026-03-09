#!/usr/bin/env python3
"""Analyze baseline confusion patterns under the Intentonomy hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.components.intentonomy_hierarchy import (
    FINE_TO_LEVEL_1,
    FINE_TO_LEVEL_2,
    FINE_TO_LEVEL_3,
    HIERARCHY_LEVEL_1,
    HIERARCHY_LEVEL_2,
    HIERARCHY_LEVEL_3,
)
from src.utils.metrics import compute_f1, eval_validation_set


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze baseline same-parent vs cross-parent confusions on Intentonomy."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training or eval run directory containing .hydra/config.yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional checkpoint path override. Default follows the run config/checkpoints.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split to analyze.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers used during analysis.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable dataloader pin_memory.",
    )
    parser.add_argument(
        "--use-inference-strategy",
        action="store_true",
        help="Enable fallback-to-argmax when thresholding yields all-zero predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading the checkpoint state_dict.",
    )
    return parser.parse_args()


def _resolve_ckpt_path(run_dir: Path, ckpt_path: str | None) -> Path:
    if ckpt_path is not None:
        return Path(ckpt_path)

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        cfg_ckpt_path = cfg.get("ckpt_path")
        if cfg_ckpt_path:
            candidate = Path(str(cfg_ckpt_path))
            if candidate.exists():
                return candidate

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        epoch_ckpts = sorted(checkpoints_dir.glob("epoch_*.ckpt"))
        if epoch_ckpts:
            return epoch_ckpts[-1]
        candidate = checkpoints_dir / "last.ckpt"
        if candidate.exists():
            return candidate

    return run_dir / "checkpoints" / "last.ckpt"


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("ema_model."):
            continue
        new_key = key
        if ".net._orig_mod." in new_key:
            new_key = new_key.replace(".net._orig_mod.", ".net.")
        elif new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod.") :]
        normalized[new_key] = value
    return normalized


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_class_names(annotation_file: Path) -> List[str]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: item.get("id", item.get("category_id")),
    )
    return [str(category["name"]) for category in categories]


def _collect_scores_targets(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    scores_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    image_ids_all: List[str] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            targets = batch["labels"].detach().cpu().numpy()

            image_ids = batch["image_id"]
            if torch.is_tensor(image_ids):
                image_ids_all.extend(str(x) for x in image_ids.detach().cpu().tolist())
            else:
                image_ids_all.extend(str(x) for x in image_ids)

            scores_all.append(scores)
            targets_all.append(targets)

    return (
        np.concatenate(scores_all, axis=0),
        np.concatenate(targets_all, axis=0),
        image_ids_all,
    )


def _apply_threshold(
    scores: np.ndarray,
    threshold: float,
    use_inference_strategy: bool,
) -> np.ndarray:
    pred = scores > threshold
    if use_inference_strategy:
        zero_rows = np.where(pred.sum(axis=1) == 0)[0]
        if len(zero_rows) > 0:
            pred[zero_rows, np.argmax(scores[zero_rows], axis=1)] = True
    return pred.astype(np.int32)


def _expand_groups_to_fine(
    groups: Sequence[Sequence[int]],
    child_groups: Sequence[Sequence[int]] | None = None,
) -> List[List[int]]:
    expanded: List[List[int]] = []
    for child_indices in groups:
        fine_ids: List[int] = []
        for child_idx in child_indices:
            if child_groups is None:
                fine_ids.append(int(child_idx))
            else:
                fine_ids.extend(int(fine_idx) for fine_idx in child_groups[child_idx])
        expanded.append(sorted(fine_ids))
    return expanded


def _join_label_names(label_ids: Iterable[int], class_names: Sequence[str]) -> str:
    return "|".join(class_names[int(idx)] for idx in label_ids)


def _join_ints(values: Iterable[int]) -> str:
    return "|".join(str(int(v)) for v in values)


def _pair_record(
    *,
    split: str,
    image_id: str,
    pred_idx: int,
    gt_idx: int,
    scores_row: np.ndarray,
    pred_positive: Sequence[int],
    gt_positive: Sequence[int],
    fp_labels: Sequence[int],
    fn_labels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    pred_idx = int(pred_idx)
    gt_idx = int(gt_idx)

    pred_direct_parent = int(FINE_TO_LEVEL_1[pred_idx])
    gt_direct_parent = int(FINE_TO_LEVEL_1[gt_idx])
    pred_middle_parent = int(FINE_TO_LEVEL_2[pred_idx])
    gt_middle_parent = int(FINE_TO_LEVEL_2[gt_idx])
    pred_coarse_parent = int(FINE_TO_LEVEL_3[pred_idx])
    gt_coarse_parent = int(FINE_TO_LEVEL_3[gt_idx])

    return {
        "split": split,
        "image_id": str(image_id),
        "pred_fine_id": pred_idx,
        "pred_fine_name": class_names[pred_idx],
        "gt_fine_id": gt_idx,
        "gt_fine_name": class_names[gt_idx],
        "pred_score": float(scores_row[pred_idx]),
        "gt_score": float(scores_row[gt_idx]),
        "pred_direct_parent_id": pred_direct_parent,
        "gt_direct_parent_id": gt_direct_parent,
        "same_direct_parent": int(pred_direct_parent == gt_direct_parent),
        "pred_middle_parent_id": pred_middle_parent,
        "gt_middle_parent_id": gt_middle_parent,
        "same_middle_parent": int(pred_middle_parent == gt_middle_parent),
        "pred_coarse_parent_id": pred_coarse_parent,
        "gt_coarse_parent_id": gt_coarse_parent,
        "same_coarse_parent": int(pred_coarse_parent == gt_coarse_parent),
        "pred_positive_ids": _join_ints(pred_positive),
        "pred_positive_names": _join_label_names(pred_positive, class_names),
        "gt_positive_ids": _join_ints(gt_positive),
        "gt_positive_names": _join_label_names(gt_positive, class_names),
        "fp_ids": _join_ints(fp_labels),
        "fp_names": _join_label_names(fp_labels, class_names),
        "fn_ids": _join_ints(fn_labels),
        "fn_names": _join_label_names(fn_labels, class_names),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _top_pair_counts(
    counter: Counter[tuple[int, int]],
    class_names: Sequence[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for (gt_idx, pred_idx), count in counter.most_common(limit):
        items.append(
            {
                "gt_fine_id": int(gt_idx),
                "gt_fine_name": class_names[int(gt_idx)],
                "pred_fine_id": int(pred_idx),
                "pred_fine_name": class_names[int(pred_idx)],
                "count": int(count),
            }
        )
    return items


def _write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _analyze_split(
    *,
    split: str,
    image_ids: Sequence[str],
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    class_names: Sequence[str],
    use_inference_strategy: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pred_binary = _apply_threshold(scores, threshold, use_inference_strategy=use_inference_strategy)
    micro_f1, samples_f1, macro_f1, _ = compute_f1(
        targets,
        scores,
        threshold=threshold,
        use_inference_strategy=use_inference_strategy,
    )

    summary_counts: Counter[str] = Counter()
    dominant_pair_rows: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []
    dominant_pair_counter: Counter[tuple[int, int]] = Counter()
    all_pair_counter: Counter[tuple[int, int]] = Counter()

    for row_idx, image_id in enumerate(image_ids):
        gt_positive = np.flatnonzero(targets[row_idx] > 0.5)
        pred_positive = np.flatnonzero(pred_binary[row_idx] > 0)
        false_positive = np.setdiff1d(pred_positive, gt_positive, assume_unique=True)
        false_negative = np.setdiff1d(gt_positive, pred_positive, assume_unique=True)

        summary_counts["num_samples"] += 1

        if len(false_positive) == 0 and len(false_negative) == 0:
            summary_counts["num_correct_samples"] += 1
            continue

        summary_counts["num_error_samples"] += 1

        if len(false_positive) > 0 and len(false_negative) > 0:
            summary_counts["num_confusion_samples"] += 1

            for pred_idx in false_positive:
                for gt_idx in false_negative:
                    pair_row = _pair_record(
                        split=split,
                        image_id=str(image_id),
                        pred_idx=int(pred_idx),
                        gt_idx=int(gt_idx),
                        scores_row=scores[row_idx],
                        pred_positive=pred_positive,
                        gt_positive=gt_positive,
                        fp_labels=false_positive,
                        fn_labels=false_negative,
                        class_names=class_names,
                    )
                    all_pair_rows.append(pair_row)
                    all_pair_counter[(int(gt_idx), int(pred_idx))] += 1
                    summary_counts["all_pairs_total"] += 1
                    summary_counts["all_pairs_same_direct_parent"] += int(pair_row["same_direct_parent"])
                    summary_counts["all_pairs_same_middle_parent"] += int(pair_row["same_middle_parent"])
                    summary_counts["all_pairs_same_coarse_parent"] += int(pair_row["same_coarse_parent"])

            dominant_pred = int(false_positive[np.argmax(scores[row_idx, false_positive])])
            dominant_gt = int(false_negative[np.argmax(scores[row_idx, false_negative])])
            dominant_row = _pair_record(
                split=split,
                image_id=str(image_id),
                pred_idx=dominant_pred,
                gt_idx=dominant_gt,
                scores_row=scores[row_idx],
                pred_positive=pred_positive,
                gt_positive=gt_positive,
                fp_labels=false_positive,
                fn_labels=false_negative,
                class_names=class_names,
            )
            dominant_pair_rows.append(dominant_row)
            dominant_pair_counter[(dominant_gt, dominant_pred)] += 1
            summary_counts["dominant_pairs_total"] += 1
            summary_counts["dominant_pairs_same_direct_parent"] += int(dominant_row["same_direct_parent"])
            summary_counts["dominant_pairs_same_middle_parent"] += int(dominant_row["same_middle_parent"])
            summary_counts["dominant_pairs_same_coarse_parent"] += int(dominant_row["same_coarse_parent"])
        elif len(false_positive) > 0:
            summary_counts["num_pure_fp_samples"] += 1
        else:
            summary_counts["num_pure_fn_samples"] += 1

    error_samples = int(summary_counts["num_error_samples"])
    confusion_samples = int(summary_counts["num_confusion_samples"])
    dominant_total = int(summary_counts["dominant_pairs_total"])
    all_pairs_total = int(summary_counts["all_pairs_total"])

    result = {
        "split": split,
        "threshold": float(threshold),
        "use_inference_strategy": bool(use_inference_strategy),
        "micro_f1_at_threshold": float(micro_f1),
        "samples_f1_at_threshold": float(samples_f1),
        "macro_f1_at_threshold": float(macro_f1),
        "num_samples": int(summary_counts["num_samples"]),
        "num_correct_samples": int(summary_counts["num_correct_samples"]),
        "num_error_samples": error_samples,
        "error_sample_ratio": _safe_ratio(error_samples, int(summary_counts["num_samples"])),
        "num_confusion_samples": confusion_samples,
        "confusion_sample_ratio_among_errors": _safe_ratio(confusion_samples, error_samples),
        "num_pure_fp_samples": int(summary_counts["num_pure_fp_samples"]),
        "num_pure_fn_samples": int(summary_counts["num_pure_fn_samples"]),
        "dominant_pair": {
            "num_pairs": dominant_total,
            "same_direct_parent_confusion_ratio": _safe_ratio(
                int(summary_counts["dominant_pairs_same_direct_parent"]), dominant_total
            ),
            "cross_direct_parent_confusion_ratio": _safe_ratio(
                dominant_total - int(summary_counts["dominant_pairs_same_direct_parent"]),
                dominant_total,
            ),
            "same_middle_parent_confusion_ratio": _safe_ratio(
                int(summary_counts["dominant_pairs_same_middle_parent"]), dominant_total
            ),
            "cross_middle_parent_confusion_ratio": _safe_ratio(
                dominant_total - int(summary_counts["dominant_pairs_same_middle_parent"]),
                dominant_total,
            ),
            "same_coarse_parent_confusion_ratio": _safe_ratio(
                int(summary_counts["dominant_pairs_same_coarse_parent"]), dominant_total
            ),
            "cross_coarse_parent_confusion_ratio": _safe_ratio(
                dominant_total - int(summary_counts["dominant_pairs_same_coarse_parent"]),
                dominant_total,
            ),
            "same_middle_parent_among_all_errors_ratio": _safe_ratio(
                int(summary_counts["dominant_pairs_same_middle_parent"]), error_samples
            ),
            "same_coarse_parent_among_all_errors_ratio": _safe_ratio(
                int(summary_counts["dominant_pairs_same_coarse_parent"]), error_samples
            ),
            "top_confusion_pairs": _top_pair_counts(dominant_pair_counter, class_names),
        },
        "all_pairs": {
            "num_pairs": all_pairs_total,
            "same_direct_parent_confusion_ratio": _safe_ratio(
                int(summary_counts["all_pairs_same_direct_parent"]), all_pairs_total
            ),
            "cross_direct_parent_confusion_ratio": _safe_ratio(
                all_pairs_total - int(summary_counts["all_pairs_same_direct_parent"]),
                all_pairs_total,
            ),
            "same_middle_parent_confusion_ratio": _safe_ratio(
                int(summary_counts["all_pairs_same_middle_parent"]), all_pairs_total
            ),
            "cross_middle_parent_confusion_ratio": _safe_ratio(
                all_pairs_total - int(summary_counts["all_pairs_same_middle_parent"]),
                all_pairs_total,
            ),
            "same_coarse_parent_confusion_ratio": _safe_ratio(
                int(summary_counts["all_pairs_same_coarse_parent"]), all_pairs_total
            ),
            "cross_coarse_parent_confusion_ratio": _safe_ratio(
                all_pairs_total - int(summary_counts["all_pairs_same_coarse_parent"]),
                all_pairs_total,
            ),
            "top_confusion_pairs": _top_pair_counts(all_pair_counter, class_names),
        },
    }
    return result, dominant_pair_rows, all_pair_rows


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.pin_memory = bool(args.pin_memory)
    if "compile" in cfg.model:
        cfg.model.compile = False

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=bool(args.strict_load))

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = model.eval().to(device)

    val_annotation = Path(str(datamodule.hparams.annotation_dir)) / str(datamodule.hparams.val_annotation)
    class_names = _load_class_names(val_annotation)

    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    val_scores, val_targets, val_image_ids = _collect_scores_targets(model, val_loader, device)
    val_metric_dict = eval_validation_set(
        val_scores,
        val_targets,
        use_inference_strategy=bool(args.use_inference_strategy),
    )
    threshold = float(val_metric_dict["threshold"])

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "logs" / "analysis" / f"{run_dir.name}_hierarchy_confusion"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_groups = _expand_groups_to_fine(HIERARCHY_LEVEL_1)
    middle_groups = _expand_groups_to_fine(HIERARCHY_LEVEL_2, HIERARCHY_LEVEL_1)
    coarse_groups = _expand_groups_to_fine(HIERARCHY_LEVEL_3, middle_groups)

    mapping_rows = []
    for fine_idx, fine_name in enumerate(class_names):
        mapping_rows.append(
            {
                "fine_id": int(fine_idx),
                "fine_name": fine_name,
                "direct_parent_id_18": int(FINE_TO_LEVEL_1[fine_idx]),
                "middle_parent_id_15": int(FINE_TO_LEVEL_2[fine_idx]),
                "coarse_parent_id_9": int(FINE_TO_LEVEL_3[fine_idx]),
            }
        )
    _write_csv(mapping_rows, output_dir / "fine_to_hierarchy_mapping.csv")

    hierarchy_groups = {
        "direct_parent_18": [
            {
                "group_id": int(group_idx),
                "fine_ids": list(map(int, fine_ids)),
                "fine_names": [class_names[fine_idx] for fine_idx in fine_ids],
            }
            for group_idx, fine_ids in enumerate(direct_groups)
        ],
        "middle_parent_15": [
            {
                "group_id": int(group_idx),
                "fine_ids": list(map(int, fine_ids)),
                "fine_names": [class_names[fine_idx] for fine_idx in fine_ids],
            }
            for group_idx, fine_ids in enumerate(middle_groups)
        ],
        "coarse_parent_9": [
            {
                "group_id": int(group_idx),
                "fine_ids": list(map(int, fine_ids)),
                "fine_names": [class_names[fine_idx] for fine_idx in fine_ids],
            }
            for group_idx, fine_ids in enumerate(coarse_groups)
        ],
    }
    (output_dir / "hierarchy_groups.json").write_text(
        json.dumps(hierarchy_groups, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "device": str(device),
        "threshold_from_val": threshold,
        "use_inference_strategy": bool(args.use_inference_strategy),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "val_threshold_search": {
            "macro_f1": float(val_metric_dict["val_macro"]),
            "micro_f1": float(val_metric_dict["val_micro"]),
            "samples_f1": float(val_metric_dict["val_samples"]),
            "mAP": float(val_metric_dict["val_mAP"]),
            "threshold": threshold,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    split_results: Dict[str, Any] = {}

    if args.split in {"val", "both"}:
        val_result, val_dominant_rows, val_all_pair_rows = _analyze_split(
            split="val",
            image_ids=val_image_ids,
            scores=val_scores,
            targets=val_targets,
            threshold=threshold,
            class_names=class_names,
            use_inference_strategy=bool(args.use_inference_strategy),
        )
        split_results["val"] = val_result
        _write_csv(val_dominant_rows, output_dir / "dominant_confusions_val.csv")
        _write_csv(val_all_pair_rows, output_dir / "all_confusion_pairs_val.csv")
        (output_dir / "summary_val.json").write_text(
            json.dumps(val_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.split in {"test", "both"}:
        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()
        test_scores, test_targets, test_image_ids = _collect_scores_targets(model, test_loader, device)
        test_result, test_dominant_rows, test_all_pair_rows = _analyze_split(
            split="test",
            image_ids=test_image_ids,
            scores=test_scores,
            targets=test_targets,
            threshold=threshold,
            class_names=class_names,
            use_inference_strategy=bool(args.use_inference_strategy),
        )
        split_results["test"] = test_result
        _write_csv(test_dominant_rows, output_dir / "dominant_confusions_test.csv")
        _write_csv(test_all_pair_rows, output_dir / "all_confusion_pairs_test.csv")
        (output_dir / "summary_test.json").write_text(
            json.dumps(test_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "metadata": metadata,
        "splits": split_results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
