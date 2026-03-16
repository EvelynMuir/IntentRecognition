#!/usr/bin/env python3
"""Analyze cascaded student alignment with Step1/Step2/Step3 supervision."""

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
DEFAULT_TEXT_FEATURE_NPZ = (
    PROJECT_ROOT / "logs" / "analysis" / "vlm_batch256_20260316" / "rationale_batch256_step12_bge_features.npz"
)
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cascaded student alignment with Step1/Step2/Step3 supervision."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--text-feature-npz", type=str, default=str(DEFAULT_TEXT_FEATURE_NPZ))
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION_FILE))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260316)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-vis", type=float, default=0.5)
    parser.add_argument("--lambda-ctx", type=float, default=0.5)
    parser.add_argument("--lambda-neg", type=float, default=0.5)
    parser.add_argument("--neg-alpha", type=float, default=1.0)
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_vlm_cascaded_alignment"
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


def _load_class_name_to_idx(annotation_file: Path) -> Dict[str, int]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    return {str(category["name"]): idx for idx, category in enumerate(categories)}


def _subset_train_bundle(
    train_base: Dict[str, Any],
    train_clip: Dict[str, Any],
    target_image_ids: Sequence[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    id_to_idx = {str(image_id): idx for idx, image_id in enumerate(train_base["image_ids"])}
    row_ids = [int(id_to_idx[str(image_id)]) for image_id in target_image_ids]
    row_ids_np = np.asarray(row_ids, dtype=np.int64)
    subset_base = {
        key: value[row_ids_np] if isinstance(value, np.ndarray) else [value[idx] for idx in row_ids_np.tolist()]
        for key, value in train_base.items()
    }
    subset_clip = {
        key: value[row_ids_np] if isinstance(value, np.ndarray) else [value[idx] for idx in row_ids_np.tolist()]
        for key, value in train_clip.items()
    }
    return subset_base, subset_clip


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


class CascadedDataset(Dataset):
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


class CascadedStudent(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.vis_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, text_dim),
        )
        self.ctx_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, text_dim),
        )
        self.classifier = nn.Linear(input_dim * 2, num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        image_features: torch.Tensor,
        base_logits: torch.Tensor,
        step3_features: torch.Tensor,
        confuse_ids: torch.Tensor,
        neg_alpha: float,
    ) -> Dict[str, torch.Tensor]:
        v_vis = self.mlp1(image_features)
        v_ctx = self.mlp2(v_vis)
        t_vis = self.vis_projector(v_vis)
        t_ctx = self.ctx_projector(v_ctx)
        concat = torch.cat([v_vis, v_ctx], dim=-1)
        logits = base_logits + self.classifier(concat)
        step3_norm = F.normalize(step3_features, dim=-1)
        v_ctx_norm = F.normalize(t_ctx, dim=-1)
        neg_sim = F.cosine_similarity(v_ctx_norm, step3_norm, dim=-1)
        logits_corr = logits.clone()
        row_ids = torch.arange(logits.shape[0], device=logits.device)
        logits_corr[row_ids, confuse_ids] = logits[row_ids, confuse_ids] - float(neg_alpha) * F.relu(neg_sim)
        return {
            "v_vis_text": t_vis,
            "v_ctx_text": t_ctx,
            "logits": logits,
            "logits_corr": logits_corr,
            "neg_sim": neg_sim,
        }


def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(F.normalize(a, dim=-1), F.normalize(b, dim=-1), dim=-1)).mean()


@torch.inference_mode()
def _predict_scores(
    model: CascadedStudent,
    image_features: np.ndarray,
    base_logits: np.ndarray,
    text_dim: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    zeros = torch.zeros((int(batch_size), int(text_dim)), dtype=torch.float32, device=device)
    for start in range(0, image_features.shape[0], int(batch_size)):
        image_batch = torch.as_tensor(image_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logit_batch = torch.as_tensor(base_logits[start : start + int(batch_size)], dtype=torch.float32, device=device)
        step3_batch = zeros[: image_batch.shape[0]]
        confuse_ids = torch.zeros((image_batch.shape[0],), dtype=torch.long, device=device)
        logits = model(image_batch, logit_batch, step3_batch, confuse_ids, neg_alpha=0.0)["logits"]
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = _parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = _resolve_device(args.device)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.reuse_cache_dir)
    text_npz = np.load(Path(args.text_feature_npz), allow_pickle=False)
    train_image_ids = [str(item) for item in text_npz["image_ids"].tolist()]
    class_name_to_idx = _load_class_name_to_idx(Path(args.annotation_file))
    confuse_ids = np.asarray([class_name_to_idx[str(name)] for name in text_npz["confuse_class_names"].tolist()], dtype=np.int64)

    train_base = _load_cache_bundle(cache_dir / "train_base.npz")
    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_base = _load_cache_bundle(cache_dir / "val_base.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_base = _load_cache_bundle(cache_dir / "test_base.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")
    train_base, train_clip = _subset_train_bundle(train_base, train_clip, train_image_ids)

    dataset = CascadedDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        base_logits=np.asarray(train_base["logits"], dtype=np.float32),
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        step1_features=np.asarray(text_npz["step1_features"], dtype=np.float32),
        step2_features=np.asarray(text_npz["step2_features"], dtype=np.float32),
        step3_features=np.asarray(text_npz["step3_features"], dtype=np.float32),
        confuse_ids=confuse_ids,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    model = CascadedStudent(
        input_dim=int(dataset.image_features.shape[1]),
        text_dim=int(dataset.step1_features.shape[1]),
        num_classes=int(dataset.labels.shape[1]),
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
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_macro = float("-inf")
    stale_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_vis = 0.0
        total_ctx = 0.0
        total_neg = 0.0
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
            outputs = model(image_features, base_logits, step3, confuse, neg_alpha=float(args.neg_alpha))
            cls_loss = criterion(outputs["logits"], labels, reduction="mean")
            vis_loss = _cosine_loss(outputs["v_vis_text"], step1)
            ctx_loss = _cosine_loss(outputs["v_ctx_text"], step2)
            row_ids = torch.arange(outputs["logits"].shape[0], device=outputs["logits"].device)
            neg_loss = F.relu(
                outputs["logits_corr"][row_ids, confuse] - outputs["logits"][row_ids, confuse]
            ).mean()
            loss = (
                cls_loss
                + float(args.lambda_vis) * vis_loss
                + float(args.lambda_ctx) * ctx_loss
                + float(args.lambda_neg) * neg_loss
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_cls += float(cls_loss.detach().cpu())
            total_vis += float(vis_loss.detach().cpu())
            total_ctx += float(ctx_loss.detach().cpu())
            total_neg += float(neg_loss.detach().cpu())
            batch_count += 1

        val_scores = _predict_scores(
            model=model,
            image_features=np.asarray(val_clip["features"], dtype=np.float32),
            base_logits=np.asarray(val_base["logits"], dtype=np.float32),
            text_dim=int(dataset.step1_features.shape[1]),
            device=device,
            batch_size=int(args.batch_size),
        )
        test_scores = _predict_scores(
            model=model,
            image_features=np.asarray(test_clip["features"], dtype=np.float32),
            base_logits=np.asarray(test_base["logits"], dtype=np.float32),
            text_dim=int(dataset.step1_features.shape[1]),
            device=device,
            batch_size=int(args.batch_size),
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
                "cls_loss": total_cls / max(1, batch_count),
                "vis_loss": total_vis / max(1, batch_count),
                "ctx_loss": total_ctx / max(1, batch_count),
                "neg_loss": total_neg / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
            }
        )
        print(
            "[VLM-Cascade] epoch "
            f"{epoch:02d} | val_macro={val_macro*100.0:.2f} | "
            f"test_macro={float(bundle['classwise']['test']['macro'])*100.0:.2f} | "
            f"test_hard={float(bundle['classwise']['test']['hard'])*100.0:.2f}"
        )
        if val_macro > best_val_macro + 1e-9:
            best_val_macro = val_macro
            best_bundle = bundle
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(args.patience):
            break

    if best_bundle is None or best_state is None:
        raise RuntimeError("Cascaded alignment did not produce a valid best checkpoint.")

    comparison_rows = [
        _comparison_row("baseline", baseline_bundle, note="cached baseline reference"),
        _comparison_row(
            "cascaded Step1/Step2/Step3",
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
                "cascaded": {
                    "bundle": _json_ready(best_bundle),
                    "history": _json_ready(history),
                    "best_epoch": int(best_epoch),
                },
                "comparison_rows": comparison_rows,
                "loss_config": {
                    "lambda_vis": float(args.lambda_vis),
                    "lambda_ctx": float(args.lambda_ctx),
                    "lambda_neg": float(args.lambda_neg),
                    "neg_alpha": float(args.neg_alpha),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
