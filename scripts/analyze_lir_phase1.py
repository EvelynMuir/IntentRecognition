#!/usr/bin/env python3
"""Run LIR Phase 1 full-data MVP with latent basis residual over frozen baseline logits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clip  # type: ignore

from scripts.analyze_data_driven_agent_evidence_verification import (
    _evaluate_with_class_thresholds,
)
from src.data.intentonomy_datamodule import IntentonomyDataModule, IntentonomyDataset
from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.decision_rule_calibration import search_classwise_thresholds
from src.utils.text_prior_analysis import evaluate_with_validation_threshold

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_REGION_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "smoke_region_grounded_agent_20260312" / "_region_cache"
)
REFERENCE_ROWS = [
    {
        "method": "scenario SLR-C",
        "macro": 51.28005127390507,
        "micro": 59.13214990138067,
        "samples": 58.46663296498823,
        "mAP": 53.66037803129969,
        "hard": 33.97660825373071,
        "note": "repo analysis reference",
    },
    {
        "method": "fixed benchmark-bank best",
        "macro": 51.91781808986489,
        "micro": float("nan"),
        "samples": float("nan"),
        "mAP": float("nan"),
        "hard": 35.63157801259155,
        "note": "hard_negative_diff relation-family best from full_data_driven_agent_evidence_verification_20260311",
    },
    {
        "method": "comparative verifier best",
        "macro": 51.73106846619861,
        "micro": 59.21717171717171,
        "samples": 58.190332602339176,
        "mAP": 53.77725013192094,
        "hard": 35.834013555258295,
        "note": "repo analysis reference",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LIR Phase 1 full-data latent basis MVP on frozen CLIP/baseline caches."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--region-cache-dir", type=str, default=str(DEFAULT_REGION_CACHE_DIR))
    parser.add_argument("--train-patch-cache", type=str, default=None)
    parser.add_argument("--train-patch-meta", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--extract-num-workers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparse-weight", type=float, default=5e-4)
    parser.add_argument("--diversity-weight", type=float, default=1e-2)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-basis", type=int, default=32)
    parser.add_argument("--routing-topk", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-patch-only", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_lir_phase1_full"
    return Path(output_dir_arg)


def _resolve_runtime_device(device_arg: str) -> torch.device:
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


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _load_cache_bundle(path: Path) -> Dict[str, Any]:
    bundle = np.load(path, allow_pickle=False)
    return {
        key: np.asarray(bundle[key]) if key != "image_ids" else [str(item) for item in bundle[key].tolist()]
        for key in bundle.files
    }


def _load_region_cache(path: Path) -> Dict[str, Any]:
    bundle = np.load(path, allow_pickle=False)
    return {
        "features": np.asarray(bundle["features"], dtype=np.float32),
        "patch_tokens": np.asarray(bundle["patch_tokens"]),
        "image_ids": [str(item) for item in bundle["image_ids"].tolist()],
    }


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
    return {
        "global": global_metrics,
        "classwise": classwise_metrics,
    }


def _comparison_row_from_metrics(method: str, metrics: Mapping[str, Any], note: str) -> Dict[str, Any]:
    test_metrics = metrics["classwise"]["test"]
    return {
        "method": method,
        "macro": float(test_metrics["macro"]) * 100.0,
        "micro": float(test_metrics["micro"]) * 100.0,
        "samples": float(test_metrics["samples"]) * 100.0,
        "mAP": float(test_metrics["mAP"]),
        "hard": float(test_metrics["hard"]) * 100.0,
        "note": note,
    }


def _align_row_ids(source_ids: Sequence[str], target_ids: Sequence[str]) -> np.ndarray:
    id_to_idx = {str(image_id): idx for idx, image_id in enumerate(source_ids)}
    row_ids: List[int] = []
    missing: List[str] = []
    for image_id in target_ids:
        row_idx = id_to_idx.get(str(image_id))
        if row_idx is None:
            missing.append(str(image_id))
        else:
            row_ids.append(int(row_idx))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing {len(missing)} image ids while aligning caches: {preview}")
    return np.asarray(row_ids, dtype=np.int64)


def _resolve_data_paths() -> Dict[str, Any]:
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "data" / "intentonomy.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw_cfg, dict):
        raise TypeError("Expected intentonomy data config to deserialize into a dict.")

    def _resolve(value: Any) -> str:
        return str(value).replace("${paths.root_dir}", str(PROJECT_ROOT))

    return {
        "data_dir": _resolve(raw_cfg["data_dir"]),
        "annotation_dir": _resolve(raw_cfg["annotation_dir"]),
        "image_dir": _resolve(raw_cfg["image_dir"]),
        "train_annotation": str(raw_cfg["train_annotation"]),
        "val_annotation": str(raw_cfg["val_annotation"]),
        "test_annotation": str(raw_cfg["test_annotation"]),
        "binarize_softprob": bool(raw_cfg.get("binarize_softprob", False)),
        "image_size": int(raw_cfg["image_size"]),
    }


def _extract_projected_features_and_tokens(
    clip_model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = clip_model.visual
    images = images.to(dtype=backbone.conv1.weight.dtype)

    x = backbone.conv1(images)
    batch_size, channels, height, width = x.shape
    x = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)

    class_token = backbone.class_embedding.to(dtype=x.dtype, device=x.device)
    class_token = class_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
    x = torch.cat([class_token, x], dim=1)
    x = x + backbone.positional_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(0)
    x = backbone.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = backbone.transformer(x)
    x = x.permute(1, 0, 2)
    if hasattr(backbone, "ln_post"):
        x = backbone.ln_post(x)

    cls_token = x[:, 0:1, :]
    patch_tokens = x[:, 1:, :]
    if getattr(backbone, "proj", None) is not None:
        cls_token = cls_token @ backbone.proj
        patch_tokens = patch_tokens @ backbone.proj

    global_features = F.normalize(cls_token[:, 0, :].float(), dim=-1)
    return global_features, patch_tokens.float()


def _build_train_patch_dataloader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[IntentonomyDataset, DataLoader]:
    paths = _resolve_data_paths()
    datamodule = IntentonomyDataModule(
        data_dir=paths["data_dir"],
        annotation_dir=paths["annotation_dir"],
        image_dir=paths["image_dir"],
        train_annotation=paths["train_annotation"],
        val_annotation=paths["val_annotation"],
        test_annotation=paths["test_annotation"],
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=paths["image_size"],
        binarize_softprob=paths["binarize_softprob"],
    )
    dataset = IntentonomyDataset(
        annotation_file=str(Path(paths["annotation_dir"]) / paths["train_annotation"]),
        image_dir=str(paths["image_dir"]),
        transform=datamodule.val_test_transform,
        binarize_softprob=paths["binarize_softprob"],
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def _load_patch_cache_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_cache_matches_expected(
    cache_path: Path,
    meta_path: Path,
    expected_image_ids: Sequence[str],
) -> bool:
    if not cache_path.exists() or not meta_path.exists():
        return False
    meta = _load_patch_cache_meta(meta_path)
    meta_ids = [str(item) for item in meta.get("image_ids", [])]
    return len(meta_ids) == len(expected_image_ids) and meta_ids == [str(item) for item in expected_image_ids]


def _ensure_train_patch_cache(
    cache_path: Path,
    meta_path: Path,
    expected_image_ids: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[np.memmap, Dict[str, Any], str]:
    if _patch_cache_matches_expected(cache_path, meta_path, expected_image_ids):
        meta = _load_patch_cache_meta(meta_path)
        patches = np.load(cache_path, mmap_mode="r")
        return patches, meta, "reused"

    if cache_path.exists():
        cache_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    dataset, dataloader = _build_train_patch_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    memmap_array: np.memmap | None = None
    image_ids_all: List[str] = []
    feature_dim = -1
    num_patches = -1
    offset = 0

    print(f"[LIR] extracting full train patch tokens to {cache_path}")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            _, patch_tokens = _extract_projected_features_and_tokens(clip_model, images)
            patch_cpu = patch_tokens.detach().cpu().numpy().astype(np.float16, copy=False)

            if memmap_array is None:
                num_patches = int(patch_cpu.shape[1])
                feature_dim = int(patch_cpu.shape[2])
                memmap_array = np.lib.format.open_memmap(
                    cache_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(dataset), num_patches, feature_dim),
                )

            batch_size_actual = int(patch_cpu.shape[0])
            memmap_array[offset : offset + batch_size_actual] = patch_cpu
            image_ids_all.extend(str(item) for item in batch["image_id"])
            offset += batch_size_actual

            if batch_idx % 20 == 0 or offset == len(dataset):
                print(f"[LIR] extracted {offset}/{len(dataset)} train samples")

    if memmap_array is None:
        raise RuntimeError("Failed to extract any train patch tokens.")
    memmap_array.flush()

    meta = {
        "shape": [len(dataset), num_patches, feature_dim],
        "dtype": "float16",
        "image_ids": image_ids_all,
    }
    meta_path.write_text(json.dumps(_json_ready(meta), indent=2), encoding="utf-8")

    if image_ids_all != [str(item) for item in expected_image_ids]:
        raise RuntimeError(
            "Extracted train patch token order does not match cached train bundles. "
            "Please inspect train cache alignment before continuing."
        )

    patches = np.load(cache_path, mmap_mode="r")
    return patches, meta, "extracted"


class CachedLatentBasisDataset(Dataset):
    def __init__(
        self,
        patch_tokens: np.ndarray,
        global_features: np.ndarray,
        base_logits: np.ndarray,
        labels: np.ndarray,
        patch_row_ids: np.ndarray | None = None,
    ) -> None:
        self.patch_tokens = patch_tokens
        self.global_features = np.asarray(global_features, dtype=np.float32)
        self.base_logits = np.asarray(base_logits, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.patch_row_ids = None if patch_row_ids is None else np.asarray(patch_row_ids, dtype=np.int64)

        if not (
            self.global_features.shape[0] == self.base_logits.shape[0] == self.labels.shape[0]
        ):
            raise ValueError("Feature/logit/label sample counts do not match.")
        if self.patch_row_ids is not None and self.patch_row_ids.shape[0] != self.labels.shape[0]:
            raise ValueError("patch_row_ids length does not match dataset length.")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch_idx = idx if self.patch_row_ids is None else int(self.patch_row_ids[idx])
        patch_tokens = np.asarray(self.patch_tokens[patch_idx], dtype=np.float32)
        return {
            "patch_tokens": torch.from_numpy(patch_tokens),
            "global_features": torch.from_numpy(self.global_features[idx]),
            "base_logits": torch.from_numpy(self.base_logits[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
        }


def _topk_normalize(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if scores.dim() != 2:
        raise ValueError(f"Expected [batch, num_basis] scores, got shape {tuple(scores.shape)}")
    num_basis = scores.shape[1]
    if topk <= 0 or topk >= num_basis:
        weights = scores
    else:
        values, indices = torch.topk(scores, k=int(topk), dim=1)
        weights = torch.zeros_like(scores)
        weights.scatter_(1, indices, values)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return weights / denom


def _basis_diversity_loss(basis_queries: torch.Tensor) -> torch.Tensor:
    basis_norm = F.normalize(basis_queries, dim=-1)
    gram = basis_norm @ basis_norm.t()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return ((gram - eye) ** 2).mean()


class LatentBasisResidualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_basis: int,
        routing_topk: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.routing_topk = int(routing_topk)
        self.num_classes = int(num_classes)

        self.basis_queries = nn.Parameter(torch.empty(self.num_basis, self.input_dim))
        nn.init.normal_(self.basis_queries, mean=0.0, std=0.02)

        self.query_norm = nn.LayerNorm(self.input_dim)
        self.token_norm = nn.LayerNorm(self.input_dim)
        self.basis_norm = nn.LayerNorm(self.input_dim)
        self.global_norm = nn.LayerNorm(self.input_dim)
        self.routing_head = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.scale = 1.0 / math.sqrt(float(self.input_dim))

    def forward(
        self,
        patch_tokens: torch.Tensor,
        global_features: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tokens_key = F.normalize(self.token_norm(patch_tokens), dim=-1)
        tokens_value = patch_tokens
        queries = F.normalize(self.query_norm(self.basis_queries), dim=-1)

        attn_logits = torch.einsum("kd,bnd->bkn", queries, tokens_key) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        basis_states = torch.einsum("bkn,bnd->bkd", attn_weights, tokens_value)

        routing_scores = F.softplus(self.routing_head(basis_states).squeeze(-1))
        routing_weights = _topk_normalize(routing_scores, topk=self.routing_topk)

        basis_summary = torch.einsum("bk,bkd->bd", routing_weights, basis_states)
        basis_summary = self.basis_norm(basis_summary)
        global_summary = self.global_norm(global_features)
        fused = torch.cat([global_summary, basis_summary], dim=-1)

        residual_logits = self.residual_head(fused)
        final_logits = base_logits + residual_logits
        return {
            "final_logits": final_logits,
            "residual_logits": residual_logits,
            "routing_weights": routing_weights,
            "sparse_loss": routing_scores.mean(),
            "diversity_loss": _basis_diversity_loss(self.basis_queries),
        }


@torch.inference_mode()
def _predict_model_outputs(
    model: LatentBasisResidualModel,
    dataloader: DataLoader,
    device: torch.device,
    return_routing: bool = False,
) -> Dict[str, np.ndarray]:
    logits_all: List[np.ndarray] = []
    routing_all: List[np.ndarray] = []
    for batch in dataloader:
        patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
        global_features = batch["global_features"].to(device, non_blocking=True)
        base_logits = batch["base_logits"].to(device, non_blocking=True)
        outputs = model(
            patch_tokens=patch_tokens,
            global_features=global_features,
            base_logits=base_logits,
        )
        logits_all.append(outputs["final_logits"].detach().float().cpu().numpy())
        if return_routing:
            routing_all.append(outputs["routing_weights"].detach().float().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0).astype(np.float32)
    result = {
        "logits": logits,
        "scores": _sigmoid(logits).astype(np.float32),
    }
    if return_routing:
        result["routing_weights"] = np.concatenate(routing_all, axis=0).astype(np.float32)
    return result


def _build_basis_usage(
    routing_weights: np.ndarray,
    labels: np.ndarray,
    topk: int,
) -> Dict[str, Any]:
    weights = np.asarray(routing_weights, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    per_class_rows = []
    top_basis_count = min(int(topk), weights.shape[1])
    for class_idx in range(targets.shape[1]):
        positive_mask = targets[:, class_idx] > 0
        if not np.any(positive_mask):
            continue
        class_mass = weights[positive_mask].mean(axis=0)
        top_ids = np.argsort(-class_mass)[:top_basis_count]
        per_class_rows.append(
            {
                "class_idx": int(class_idx),
                "top_basis_ids": [int(idx) for idx in top_ids.tolist()],
                "top_basis_mass": [float(class_mass[idx]) for idx in top_ids.tolist()],
            }
        )
    mean_mass = weights.mean(axis=0)
    overall_top_ids = np.argsort(-mean_mass)[: min(10, weights.shape[1])]
    return {
        "avg_active_basis": float((weights > 0).sum(axis=1).mean()),
        "basis_mean_mass": mean_mass.astype(np.float32),
        "top_basis_ids": [int(idx) for idx in overall_top_ids.tolist()],
        "top_basis_mass": [float(mean_mass[idx]) for idx in overall_top_ids.tolist()],
        "per_class_top_basis": per_class_rows,
    }


def _train_lir_phase1(
    model: LatentBasisResidualModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.max_epochs)),
        eta_min=float(args.lr) * 0.05,
    )
    criterion = AsymmetricLossOptimized(
        gamma_neg=2,
        gamma_pos=0,
        clip=0.05,
        eps=1e-5,
        disable_torch_grad_focal_loss=False,
    )

    best_val_macro = float("-inf")
    best_epoch = -1
    best_bundle: Dict[str, Any] | None = None
    best_state_dict: Dict[str, torch.Tensor] | None = None
    best_routing_weights: np.ndarray | None = None
    history_rows: List[Dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        train_total_loss = 0.0
        train_cls_loss = 0.0
        train_sparse_loss = 0.0
        train_diversity_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            patch_tokens = batch["patch_tokens"].to(device, non_blocking=True)
            global_features = batch["global_features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                patch_tokens=patch_tokens,
                global_features=global_features,
                base_logits=base_logits,
            )
            cls_loss = criterion(outputs["final_logits"], labels)
            loss = (
                cls_loss
                + float(args.sparse_weight) * outputs["sparse_loss"]
                + float(args.diversity_weight) * outputs["diversity_loss"]
            )
            loss.backward()
            if float(args.clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            optimizer.step()

            train_total_loss += float(loss.detach().cpu())
            train_cls_loss += float(cls_loss.detach().cpu())
            train_sparse_loss += float(outputs["sparse_loss"].detach().cpu())
            train_diversity_loss += float(outputs["diversity_loss"].detach().cpu())
            batch_count += 1

        scheduler.step()

        val_outputs = _predict_model_outputs(model, val_loader, device=device, return_routing=False)
        test_outputs = _predict_model_outputs(model, test_loader, device=device, return_routing=True)
        bundle = _evaluate_score_bundle(
            val_scores=val_outputs["scores"],
            val_targets=val_targets,
            test_scores=test_outputs["scores"],
            test_targets=test_targets,
        )
        val_macro = float(bundle["classwise"]["val"]["macro"])

        history_rows.append(
            {
                "epoch": int(epoch),
                "train": {
                    "loss": train_total_loss / max(1, batch_count),
                    "cls_loss": train_cls_loss / max(1, batch_count),
                    "sparse_loss": train_sparse_loss / max(1, batch_count),
                    "diversity_loss": train_diversity_loss / max(1, batch_count),
                },
                "global": {
                    "val_macro": float(bundle["global"]["val"]["macro"]),
                    "test_macro": float(bundle["global"]["test"]["macro"]),
                    "test_hard": float(bundle["global"]["test"]["hard"]),
                },
                "classwise": {
                    "val_macro": float(bundle["classwise"]["val"]["macro"]),
                    "test_macro": float(bundle["classwise"]["test"]["macro"]),
                    "test_hard": float(bundle["classwise"]["test"]["hard"]),
                },
            }
        )

        print(
            "[LIR] epoch "
            f"{epoch:02d} | train_loss={history_rows[-1]['train']['loss']:.4f} "
            f"| val_macro(classwise)={val_macro * 100.0:.2f} "
            f"| test_macro(classwise)={float(bundle['classwise']['test']['macro']) * 100.0:.2f} "
            f"| test_hard(classwise)={float(bundle['classwise']['test']['hard']) * 100.0:.2f}"
        )

        if val_macro > best_val_macro + 1e-9:
            best_val_macro = val_macro
            best_epoch = int(epoch)
            best_bundle = bundle
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_routing_weights = np.asarray(test_outputs["routing_weights"], dtype=np.float32)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(args.patience):
            print(f"[LIR] early stop at epoch {epoch} after {epochs_without_improvement} stale epochs")
            break

    if best_bundle is None or best_state_dict is None or best_routing_weights is None:
        raise RuntimeError("Training did not produce a valid best checkpoint.")

    return {
        "best_epoch": best_epoch,
        "best_bundle": best_bundle,
        "best_state_dict": best_state_dict,
        "best_routing_weights": best_routing_weights,
        "history": history_rows,
    }


def _maybe_slice_train_arrays(
    max_train_samples: int | None,
    train_base: Dict[str, Any],
    train_clip: Dict[str, Any],
    train_patch_row_ids: np.ndarray,
) -> tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    if max_train_samples is None:
        return train_base, train_clip, train_patch_row_ids
    keep = max(1, min(int(max_train_samples), int(train_base["labels"].shape[0])))
    sliced_base = {
        key: value[:keep] if isinstance(value, np.ndarray) else value[:keep]
        for key, value in train_base.items()
    }
    sliced_clip = {
        key: value[:keep] if isinstance(value, np.ndarray) else value[:keep]
        for key, value in train_clip.items()
    }
    return sliced_base, sliced_clip, train_patch_row_ids[:keep]


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    device = _resolve_runtime_device(args.device)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_output_dir = output_dir / "_cache"
    cache_output_dir.mkdir(parents=True, exist_ok=True)

    reuse_cache_dir = Path(args.reuse_cache_dir)
    region_cache_dir = Path(args.region_cache_dir)
    train_patch_cache = (
        Path(args.train_patch_cache)
        if args.train_patch_cache is not None
        else cache_output_dir / "train_patch_tokens.f16.npy"
    )
    train_patch_meta = (
        Path(args.train_patch_meta)
        if args.train_patch_meta is not None
        else cache_output_dir / "train_patch_tokens_meta.json"
    )

    print(f"[LIR] device={device}")
    print(f"[LIR] output_dir={output_dir}")

    train_base = _load_cache_bundle(reuse_cache_dir / "train_base.npz")
    val_base = _load_cache_bundle(reuse_cache_dir / "val_base.npz")
    test_base = _load_cache_bundle(reuse_cache_dir / "test_base.npz")
    train_clip = _load_cache_bundle(reuse_cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(reuse_cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(reuse_cache_dir / "test_clip.npz")

    val_region = _load_region_cache(region_cache_dir / "val_clip_region.npz")
    test_region = _load_region_cache(region_cache_dir / "test_clip_region.npz")

    val_region_row_ids = _align_row_ids(val_region["image_ids"], val_base["image_ids"])
    test_region_row_ids = _align_row_ids(test_region["image_ids"], test_base["image_ids"])
    val_patch_tokens = np.asarray(val_region["patch_tokens"][val_region_row_ids], dtype=np.float16)
    test_patch_tokens = np.asarray(test_region["patch_tokens"][test_region_row_ids], dtype=np.float16)

    train_patch_memmap, train_patch_meta_data, train_patch_cache_mode = _ensure_train_patch_cache(
        cache_path=train_patch_cache,
        meta_path=train_patch_meta,
        expected_image_ids=train_base["image_ids"],
        device=device,
        batch_size=int(args.extract_batch_size),
        num_workers=int(args.extract_num_workers),
        pin_memory=(device.type == "cuda"),
    )
    train_patch_row_ids = _align_row_ids(
        [str(item) for item in train_patch_meta_data["image_ids"]],
        train_base["image_ids"],
    )

    train_base, train_clip, train_patch_row_ids = _maybe_slice_train_arrays(
        max_train_samples=args.max_train_samples,
        train_base=train_base,
        train_clip=train_clip,
        train_patch_row_ids=train_patch_row_ids,
    )

    if args.train_patch_only:
        print("[LIR] train patch cache ready; exiting because --train-patch-only was set")
        return

    input_dim = int(train_clip["features"].shape[1])
    num_classes = int(train_base["labels"].shape[1])

    train_dataset = CachedLatentBasisDataset(
        patch_tokens=train_patch_memmap,
        global_features=train_clip["features"],
        base_logits=train_base["logits"],
        labels=train_base["labels"],
        patch_row_ids=train_patch_row_ids,
    )
    val_dataset = CachedLatentBasisDataset(
        patch_tokens=val_patch_tokens,
        global_features=val_clip["features"],
        base_logits=val_base["logits"],
        labels=val_base["labels"],
    )
    test_dataset = CachedLatentBasisDataset(
        patch_tokens=test_patch_tokens,
        global_features=test_clip["features"],
        base_logits=test_base["logits"],
        labels=test_base["labels"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        num_workers=int(args.train_num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    model = LatentBasisResidualModel(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        num_basis=int(args.num_basis),
        routing_topk=int(args.routing_topk),
        num_classes=num_classes,
        dropout=float(args.dropout),
    ).to(device)

    train_result = _train_lir_phase1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
        device=device,
        args=args,
    )

    basis_usage = _build_basis_usage(
        routing_weights=np.asarray(train_result["best_routing_weights"], dtype=np.float32),
        labels=np.asarray(test_base["labels"], dtype=np.float32),
        topk=int(args.routing_topk),
    )

    comparison_rows = [
        _comparison_row_from_metrics(
            method="baseline",
            metrics=baseline_bundle,
            note="recomputed from cached baseline outputs",
        )
    ]
    comparison_rows.extend(REFERENCE_ROWS)
    comparison_rows.append(
        _comparison_row_from_metrics(
            method="latent basis MVP",
            metrics=train_result["best_bundle"],
            note=(
                "full train patch tokens + frozen baseline logits residual "
                f"(train_n={len(train_dataset)} lr={args.lr} wd={args.weight_decay} "
                f"sw={args.sparse_weight} dw={args.diversity_weight} bs={args.train_batch_size})"
            ),
        )
    )

    training_history = {
        "search": [
            {
                "config": (
                    f"full_lr{args.lr}_wd{args.weight_decay}_sw{args.sparse_weight}_"
                    f"dw{args.diversity_weight}_bs{args.train_batch_size}"
                ),
                "best_epoch": int(train_result["best_epoch"]),
                "val_macro_classwise": float(train_result["best_bundle"]["classwise"]["val"]["macro"]),
                "test_macro_classwise": float(train_result["best_bundle"]["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(train_result["best_bundle"]["classwise"]["test"]["hard"]),
                "test_macro_global": float(train_result["best_bundle"]["global"]["test"]["macro"]),
                "test_hard_global": float(train_result["best_bundle"]["global"]["test"]["hard"]),
            }
        ],
        "best_history": train_result["history"],
    }

    model_path = output_dir / "latent_basis_best.pt"
    torch.save(
        {
            "state_dict": train_result["best_state_dict"],
            "config": {
                "input_dim": input_dim,
                "hidden_dim": int(args.hidden_dim),
                "num_basis": int(args.num_basis),
                "routing_topk": int(args.routing_topk),
                "num_classes": num_classes,
                "dropout": float(args.dropout),
            },
            "best_epoch": int(train_result["best_epoch"]),
            "seed": int(args.seed),
        },
        model_path,
    )

    basis_usage_path = output_dir / "basis_usage.json"
    history_path = output_dir / "training_history.json"
    summary_path = output_dir / "summary.json"
    comparison_path = output_dir / "phase1_comparison.csv"

    basis_usage_path.write_text(json.dumps(_json_ready(basis_usage), indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(_json_ready(training_history), indent=2), encoding="utf-8")
    _write_csv(comparison_path, comparison_rows)

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "cache_paths": {
            "base_cache_dir": str(reuse_cache_dir),
            "region_cache_dir": str(region_cache_dir),
            "train_patch_cache": str(train_patch_cache),
            "train_patch_meta": str(train_patch_meta),
            "train_patch_cache_mode": train_patch_cache_mode,
        },
        "train_data": {
            "train_samples": int(len(train_dataset)),
            "val_samples": int(len(val_dataset)),
            "test_samples": int(len(test_dataset)),
        },
        "model": {
            "input_dim": input_dim,
            "hidden_dim": int(args.hidden_dim),
            "num_basis": int(args.num_basis),
            "routing_topk": int(args.routing_topk),
            "num_classes": num_classes,
            "dropout": float(args.dropout),
        },
        "train_config": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "sparse_weight": float(args.sparse_weight),
            "diversity_weight": float(args.diversity_weight),
            "batch_size": int(args.train_batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "clip_grad_norm": float(args.clip_grad_norm),
        },
        "baseline": baseline_bundle,
        "latent_basis_mvp": train_result["best_bundle"],
        "comparison_rows": comparison_rows,
        "basis_usage": basis_usage,
        "files": {
            "summary": str(summary_path),
            "comparison_csv": str(comparison_path),
            "training_history": str(history_path),
            "basis_usage": str(basis_usage_path),
            "best_model": str(model_path),
        },
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    latent_row = comparison_rows[-1]
    print(
        "[LIR] done | "
        f"train_n={len(train_dataset)} | "
        f"test_macro={latent_row['macro']:.2f} | "
        f"test_hard={latent_row['hard']:.2f} | "
        f"best_epoch={train_result['best_epoch']}"
    )


if __name__ == "__main__":
    main()
