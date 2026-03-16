#!/usr/bin/env python3
"""Analyze VLM late fusion with text-guided logit residuals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_data_driven_agent_evidence_verification import (
    _evaluate_with_class_thresholds,
    _json_ready,
)
from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.decision_rule_calibration import search_classwise_thresholds
from src.utils.text_prior_analysis import evaluate_with_validation_threshold

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM late-fusion experiments with train/val/test text features."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--train-text-npz", type=str, required=True)
    parser.add_argument("--val-text-npz", type=str, required=True)
    parser.add_argument("--test-text-npz", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260316)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neg-alpha", type=float, default=1.0)
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_vlm_late_fusion"
    return Path(output_dir_arg)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_cache_bundle(path: Path) -> Dict[str, Any]:
    bundle = np.load(path, allow_pickle=False)
    return {
        key: np.asarray(bundle[key]) if key != "image_ids" else [str(item) for item in bundle[key].tolist()]
        for key in bundle.files
    }


def _load_text_bundle(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    return {
        key: np.asarray(arr[key]) if key != "image_ids" else [str(item) for item in arr[key].tolist()]
        for key in arr.files
    }


def _load_class_name_to_idx(annotation_file: Path) -> Dict[str, int]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    return {str(category["name"]): idx for idx, category in enumerate(categories)}


def _align_text_bundle_to_cache(text_bundle: Dict[str, Any], cache_bundle: Dict[str, Any]) -> Dict[str, Any]:
    id_to_idx = {str(image_id): idx for idx, image_id in enumerate(text_bundle["image_ids"])}
    row_ids = [int(id_to_idx[str(image_id)]) for image_id in cache_bundle["image_ids"]]
    row_ids_np = np.asarray(row_ids, dtype=np.int64)
    aligned: Dict[str, Any] = {}
    for key, value in text_bundle.items():
        if key == "image_ids":
            aligned[key] = [value[idx] for idx in row_ids_np.tolist()]
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(text_bundle["image_ids"]):
            aligned[key] = value[row_ids_np]
            continue
        aligned[key] = value
    return aligned


def _evaluate_score_bundle(
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_scores: np.ndarray,
    test_targets: np.ndarray,
) -> Dict[str, Any]:
    global_metrics = evaluate_with_validation_threshold(
        val_scores,
        val_targets,
        test_scores,
        test_targets,
        use_inference_strategy=False,
    )
    class_thresholds = search_classwise_thresholds(val_scores, val_targets)
    classwise_metrics = _evaluate_with_class_thresholds(
        val_scores,
        val_targets,
        test_scores,
        test_targets,
        class_thresholds,
    )
    return {"global": global_metrics, "classwise": classwise_metrics}


def _comparison_row(method: str, bundle: Mapping[str, Any], note: str) -> Dict[str, Any]:
    metrics = bundle["classwise"]["test"]
    return {
        "method": method,
        "macro": float(metrics["macro"]) * 100.0,
        "micro": float(metrics["micro"]) * 100.0,
        "samples": float(metrics["samples"]) * 100.0,
        "mAP": float(metrics["mAP"]),
        "hard": float(metrics["hard"]) * 100.0,
        "note": note,
    }


class LateFusionDataset(Dataset):
    def __init__(
        self,
        image_features: np.ndarray,
        base_logits: np.ndarray,
        labels: np.ndarray,
        step1_features: np.ndarray,
        step2_features: np.ndarray,
        step3_features: np.ndarray,
        confuse_ids: np.ndarray,
    ) -> None:
        self.image_features = np.asarray(image_features, dtype=np.float32)
        self.base_logits = np.asarray(base_logits, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.step1_features = np.asarray(step1_features, dtype=np.float32)
        self.step2_features = np.asarray(step2_features, dtype=np.float32)
        self.step3_features = np.asarray(step3_features, dtype=np.float32)
        self.confuse_ids = np.asarray(confuse_ids, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image_features": torch.from_numpy(self.image_features[idx]),
            "base_logits": torch.from_numpy(self.base_logits[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "step1_features": torch.from_numpy(self.step1_features[idx]),
            "step2_features": torch.from_numpy(self.step2_features[idx]),
            "step3_features": torch.from_numpy(self.step3_features[idx]),
            "confuse_id": torch.tensor(int(self.confuse_ids[idx]), dtype=torch.long),
        }


class LateFusionModel(nn.Module):
    def __init__(self, input_dim: int, text_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, text_dim, bias=False)
        self.alpha_pos = nn.Parameter(torch.tensor(0.1))
        self.alpha_ctx = nn.Parameter(torch.tensor(0.1))
        self.alpha_neg = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        image_features: torch.Tensor,
        base_logits: torch.Tensor,
        step1_features: torch.Tensor,
        step2_features: torch.Tensor,
        step3_features: torch.Tensor,
        confuse_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        proj = self.proj(image_features)
        proj_norm = F.normalize(proj, dim=-1)
        step1_norm = F.normalize(step1_features, dim=-1)
        step2_norm = F.normalize(step2_features, dim=-1)
        step3_norm = F.normalize(step3_features, dim=-1)

        s_pos = F.cosine_similarity(proj_norm, step1_norm, dim=-1)
        s_ctx = F.cosine_similarity(proj_norm, step2_norm, dim=-1)
        s_neg = F.cosine_similarity(proj_norm, step3_norm, dim=-1)

        logits = base_logits.clone()
        logits = logits + self.alpha_pos * s_pos.unsqueeze(1) + self.alpha_ctx * s_ctx.unsqueeze(1)
        row_ids = torch.arange(logits.shape[0], device=logits.device)
        logits[row_ids, confuse_ids] = logits[row_ids, confuse_ids] - self.alpha_neg * F.relu(s_neg)
        return {
            "logits": logits,
            "s_pos": s_pos,
            "s_ctx": s_ctx,
            "s_neg": s_neg,
        }


@torch.inference_mode()
def _predict_scores(
    model: LateFusionModel,
    image_features: np.ndarray,
    base_logits: np.ndarray,
    step1_features: np.ndarray,
    step2_features: np.ndarray,
    step3_features: np.ndarray,
    confuse_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    for start in range(0, image_features.shape[0], int(batch_size)):
        image_batch = torch.as_tensor(image_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logit_batch = torch.as_tensor(base_logits[start : start + int(batch_size)], dtype=torch.float32, device=device)
        s1 = torch.as_tensor(step1_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        s2 = torch.as_tensor(step2_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        s3 = torch.as_tensor(step3_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        conf = torch.as_tensor(confuse_ids[start : start + int(batch_size)], dtype=torch.long, device=device)
        batch_out = model(image_batch, logit_batch, s1, s2, s3, conf)["logits"]
        outputs.append(torch.sigmoid(batch_out).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = _parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = _resolve_device(args.device)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.reuse_cache_dir)
    class_name_to_idx = _load_class_name_to_idx(Path(args.annotation_file))

    train_base = _load_cache_bundle(cache_dir / "train_base.npz")
    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_base = _load_cache_bundle(cache_dir / "val_base.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_base = _load_cache_bundle(cache_dir / "test_base.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")

    train_text = _align_text_bundle_to_cache(_load_text_bundle(Path(args.train_text_npz)), train_base)
    val_text = _align_text_bundle_to_cache(_load_text_bundle(Path(args.val_text_npz)), val_base)
    test_text = _align_text_bundle_to_cache(_load_text_bundle(Path(args.test_text_npz)), test_base)

    train_confuse = np.asarray([class_name_to_idx[str(name)] for name in train_text["confuse_class_names"].tolist()], dtype=np.int64)
    val_confuse = np.asarray([class_name_to_idx[str(name)] for name in val_text["confuse_class_names"].tolist()], dtype=np.int64)
    test_confuse = np.asarray([class_name_to_idx[str(name)] for name in test_text["confuse_class_names"].tolist()], dtype=np.int64)

    dataset = LateFusionDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        base_logits=np.asarray(train_base["logits"], dtype=np.float32),
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        step1_features=np.asarray(train_text["step1_features"], dtype=np.float32),
        step2_features=np.asarray(train_text["step2_features"], dtype=np.float32),
        step3_features=np.asarray(train_text["step3_features"], dtype=np.float32),
        confuse_ids=train_confuse,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    model = LateFusionModel(
        input_dim=int(dataset.image_features.shape[1]),
        text_dim=int(dataset.step1_features.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = AsymmetricLossOptimized(
        gamma_neg=2,
        gamma_pos=0,
        clip=0.05,
        eps=1e-5,
        disable_torch_grad_focal_loss=False,
    )

    best_bundle: Dict[str, Any] | None = None
    best_epoch = -1
    best_val_macro = float("-inf")
    stale_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            image_features = batch["image_features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            step1 = batch["step1_features"].to(device, non_blocking=True)
            step2 = batch["step2_features"].to(device, non_blocking=True)
            step3 = batch["step3_features"].to(device, non_blocking=True)
            confuse = batch["confuse_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(image_features, base_logits, step1, step2, step3, confuse)
            loss = criterion(outputs["logits"], labels, reduction="mean")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1

        val_scores = _predict_scores(
            model,
            np.asarray(val_clip["features"], dtype=np.float32),
            np.asarray(val_base["logits"], dtype=np.float32),
            np.asarray(val_text["step1_features"], dtype=np.float32),
            np.asarray(val_text["step2_features"], dtype=np.float32),
            np.asarray(val_text["step3_features"], dtype=np.float32),
            val_confuse,
            device,
            int(args.batch_size),
        )
        test_scores = _predict_scores(
            model,
            np.asarray(test_clip["features"], dtype=np.float32),
            np.asarray(test_base["logits"], dtype=np.float32),
            np.asarray(test_text["step1_features"], dtype=np.float32),
            np.asarray(test_text["step2_features"], dtype=np.float32),
            np.asarray(test_text["step3_features"], dtype=np.float32),
            test_confuse,
            device,
            int(args.batch_size),
        )
        bundle = _evaluate_score_bundle(
            val_scores=val_scores,
            val_targets=np.asarray(val_base["labels"], dtype=np.float32),
            test_scores=test_scores,
            test_targets=np.asarray(test_base["labels"], dtype=np.float32),
        )
        val_macro = float(bundle["classwise"]["val"]["macro"])
        history.append(
            {
                "epoch": int(epoch),
                "loss": total_loss / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
                "alpha_pos": float(model.alpha_pos.detach().cpu()),
                "alpha_ctx": float(model.alpha_ctx.detach().cpu()),
                "alpha_neg": float(model.alpha_neg.detach().cpu()),
            }
        )
        print(
            "[VLM-LateFusion] epoch "
            f"{epoch:02d} | val_macro={val_macro*100.0:.2f} | "
            f"test_macro={float(bundle['classwise']['test']['macro'])*100.0:.2f} | "
            f"test_hard={float(bundle['classwise']['test']['hard'])*100.0:.2f}"
        )
        if val_macro > best_val_macro + 1e-9:
            best_val_macro = val_macro
            best_bundle = bundle
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    if best_bundle is None:
        raise RuntimeError("Late-fusion training did not produce a valid best checkpoint.")

    comparison_rows = [
        _comparison_row("baseline", baseline_bundle, note="cached baseline reference"),
        _comparison_row(
            "VLM late fusion",
            best_bundle,
            note=f"train_n={len(dataset)} best_epoch={best_epoch}",
        ),
    ]
    _write_csv(output_dir / "main_comparison.csv", comparison_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_samples": int(len(dataset)),
                "baseline": _json_ready(baseline_bundle),
                "late_fusion": {
                    "bundle": _json_ready(best_bundle),
                    "history": _json_ready(history),
                    "best_epoch": int(best_epoch),
                },
                "comparison_rows": comparison_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
