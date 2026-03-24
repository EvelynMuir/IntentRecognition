#!/usr/bin/env python3
"""Analyze VLM positive alignment plus negative repulsion on cached CLIP features."""

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
DEFAULT_RATIONALE_FEATURE_NPZ = (
    PROJECT_ROOT / "logs" / "analysis" / "vlm_batch32_20260315" / "rationale_batch32_features.npz"
)
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T_pos alignment and T_neg repulsion experiments on cached CLIP image features."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--text-feature-npz", type=str, default=str(DEFAULT_RATIONALE_FEATURE_NPZ))
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION_FILE))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-pos", type=float, default=0.5)
    parser.add_argument("--lambda-neg", type=float, default=0.5)
    parser.add_argument("--lambda-keep", type=float, default=0.1)
    parser.add_argument("--lambda-delta", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--neg-margin", type=float, default=0.2)
    parser.add_argument("--keep-margin", type=float, default=0.05)
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_vlm_pos_neg_repulsion"
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


def _zero_bundle() -> Dict[str, Any]:
    return {
        "global": {"val": {}, "test": {}},
        "classwise": {"val": {}, "test": {}},
    }


def _subset_train_bundle(
    train_base: Dict[str, Any],
    train_clip: Dict[str, Any],
    target_image_ids: Sequence[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    id_to_idx = {str(image_id): idx for idx, image_id in enumerate(train_base["image_ids"])}
    row_ids = []
    for image_id in target_image_ids:
        if image_id not in id_to_idx:
            raise KeyError(f"Missing image_id in train cache: {image_id}")
        row_ids.append(int(id_to_idx[image_id]))
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


class PosNegDataset(Dataset):
    def __init__(
        self,
        image_features: np.ndarray,
        base_logits: np.ndarray,
        labels: np.ndarray,
        pos_features: np.ndarray,
        neg_features: np.ndarray,
        confuse_ids: np.ndarray,
    ) -> None:
        self.image_features = np.asarray(image_features, dtype=np.float32)
        self.base_logits = np.asarray(base_logits, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.pos_features = np.asarray(pos_features, dtype=np.float32)
        self.neg_features = np.asarray(neg_features, dtype=np.float32)
        self.confuse_ids = np.asarray(confuse_ids, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image_features": torch.from_numpy(self.image_features[idx]),
            "base_logits": torch.from_numpy(self.base_logits[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "pos_features": torch.from_numpy(self.pos_features[idx]),
            "neg_features": torch.from_numpy(self.neg_features[idx]),
            "confuse_id": torch.tensor(int(self.confuse_ids[idx]), dtype=torch.long),
        }


class PosNegRepulsionModel(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, num_classes: int) -> None:
        super().__init__()
        self.shared_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.projector_pos = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, text_dim),
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )
        nn.init.zeros_(self.classifier_head[-1].weight)
        nn.init.zeros_(self.classifier_head[-1].bias)

    def forward(
        self,
        image_features: torch.Tensor,
        base_logits: torch.Tensor,
        neg_features: torch.Tensor,
        confuse_ids: torch.Tensor,
        alpha: float,
        use_neg_repulsion: bool,
    ) -> Dict[str, torch.Tensor]:
        adapted = self.shared_adapter(image_features)
        projected_pos = self.projector_pos(adapted)
        logits_base = base_logits + self.classifier_head(adapted)
        logits_corr = logits_base.clone()
        neg_sim = torch.zeros((logits_base.shape[0],), dtype=logits_base.dtype, device=logits_base.device)
        if use_neg_repulsion:
            adapted_norm = F.normalize(adapted, dim=-1)
            neg_norm = F.normalize(neg_features, dim=-1)
            neg_sim = F.cosine_similarity(adapted_norm, neg_norm, dim=-1)
            row_ids = torch.arange(logits_corr.shape[0], device=logits_corr.device)
            logits_corr[row_ids, confuse_ids] = (
                logits_base[row_ids, confuse_ids]
                - float(alpha) * F.relu(neg_sim)
            )
        return {
            "adapted": adapted,
            "projected_pos": projected_pos,
            "logits_base": logits_base,
            "logits_corr": logits_corr,
            "neg_sim": neg_sim,
        }


@torch.inference_mode()
def _predict_scores(
    model: PosNegRepulsionModel,
    image_features: np.ndarray,
    base_logits: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    for start in range(0, image_features.shape[0], int(batch_size)):
        image_batch = torch.as_tensor(image_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logit_batch = torch.as_tensor(base_logits[start : start + int(batch_size)], dtype=torch.float32, device=device)
        neg_batch = torch.as_tensor(
            np.zeros((image_batch.shape[0], image_batch.shape[1]), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        confuse_ids = torch.zeros((image_batch.shape[0],), dtype=torch.long, device=device)
        batch_out = model(
            image_batch,
            logit_batch,
            neg_batch,
            confuse_ids=confuse_ids,
            alpha=0.0,
            use_neg_repulsion=False,
        )["logits_base"]
        outputs.append(torch.sigmoid(batch_out).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _cosine_align_loss(adapted: torch.Tensor, pos_features: torch.Tensor) -> torch.Tensor:
    a = F.normalize(adapted, dim=-1)
    t = F.normalize(pos_features, dim=-1)
    return (1.0 - F.cosine_similarity(a, t, dim=-1)).mean()


def _positive_keep_loss(
    logits_base: torch.Tensor,
    logits_corr: torch.Tensor,
    labels: torch.Tensor,
    keep_margin: float,
) -> torch.Tensor:
    pos_mask = labels > 0
    if pos_mask.sum().item() == 0:
        return torch.zeros((), dtype=logits_base.dtype, device=logits_base.device)
    diff = logits_base - float(keep_margin) - logits_corr
    return F.relu(diff[pos_mask]).mean()


def _negative_repulsion_loss(
    logits_base: torch.Tensor,
    logits_corr: torch.Tensor,
    confuse_ids: torch.Tensor,
    neg_margin: float,
) -> torch.Tensor:
    row_ids = torch.arange(logits_base.shape[0], device=logits_base.device)
    base_conf = logits_base[row_ids, confuse_ids]
    rep_conf = logits_corr[row_ids, confuse_ids]
    return F.relu(float(neg_margin) + rep_conf - base_conf).mean()


def _bundle_summary(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "global": {
            "val": _json_ready(bundle["global"]["val"]),
            "test": _json_ready(bundle["global"]["test"]),
        },
        "classwise": {
            "val": _json_ready(bundle["classwise"]["val"]),
            "test": _json_ready(bundle["classwise"]["test"]),
        },
    }


def _run_variant(
    variant_name: str,
    use_neg_repulsion: bool,
    train_dataset: PosNegDataset,
    val_clip: Dict[str, Any],
    val_base: Dict[str, Any],
    test_clip: Dict[str, Any],
    test_base: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    model = PosNegRepulsionModel(
        input_dim=int(train_dataset.image_features.shape[1]),
        text_dim=int(train_dataset.pos_features.shape[1]),
        num_classes=int(train_dataset.labels.shape[1]),
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
        total_pos = 0.0
        total_neg = 0.0
        total_keep = 0.0
        total_delta = 0.0
        batch_count = 0

        for batch in loader:
            image_features = batch["image_features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pos_features = batch["pos_features"].to(device, non_blocking=True)
            neg_features = batch["neg_features"].to(device, non_blocking=True)
            confuse_ids = batch["confuse_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                image_features=image_features,
                base_logits=base_logits,
                neg_features=neg_features if use_neg_repulsion else torch.zeros_like(neg_features),
                confuse_ids=confuse_ids,
                alpha=float(args.beta),
                use_neg_repulsion=use_neg_repulsion,
            )
            cls_loss = criterion(outputs["logits_base"], labels, reduction="mean")
            pos_loss = _cosine_align_loss(outputs["projected_pos"], pos_features)
            neg_loss = (
                _negative_repulsion_loss(outputs["logits_base"], outputs["logits_corr"], confuse_ids, neg_margin=float(args.neg_margin))
                if use_neg_repulsion
                else torch.zeros((), dtype=cls_loss.dtype, device=cls_loss.device)
            )
            keep_loss = (
                _positive_keep_loss(outputs["logits_base"], outputs["logits_corr"], labels, keep_margin=float(args.keep_margin))
                if use_neg_repulsion
                else torch.zeros((), dtype=cls_loss.dtype, device=cls_loss.device)
            )
            delta_loss = (
                F.relu(outputs["neg_sim"]).pow(2).mean()
                if use_neg_repulsion
                else torch.zeros((), dtype=cls_loss.dtype, device=cls_loss.device)
            )
            loss = (
                cls_loss
                + float(args.lambda_pos) * pos_loss
                + float(args.lambda_neg) * neg_loss
                + float(args.lambda_keep) * keep_loss
                + float(args.lambda_delta) * delta_loss
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_cls += float(cls_loss.detach().cpu())
            total_pos += float(pos_loss.detach().cpu())
            total_neg += float(neg_loss.detach().cpu())
            total_keep += float(keep_loss.detach().cpu())
            total_delta += float(delta_loss.detach().cpu())
            batch_count += 1

        val_scores = _predict_scores(
            model=model,
            image_features=np.asarray(val_clip["features"], dtype=np.float32),
            base_logits=np.asarray(val_base["logits"], dtype=np.float32),
            device=device,
            batch_size=int(args.batch_size),
        )
        test_scores = _predict_scores(
            model=model,
            image_features=np.asarray(test_clip["features"], dtype=np.float32),
            base_logits=np.asarray(test_base["logits"], dtype=np.float32),
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
                "pos_loss": total_pos / max(1, batch_count),
                "neg_loss": total_neg / max(1, batch_count),
                "keep_loss": total_keep / max(1, batch_count),
                "delta_loss": total_delta / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
            }
        )
        print(
            "[VLM-Repulse] "
            f"{variant_name} epoch {epoch:02d} | "
            f"val_macro={val_macro*100.0:.2f} | "
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
        raise RuntimeError(f"Variant {variant_name} failed to produce a valid best checkpoint.")

    return {
        "bundle": best_bundle,
        "history": history,
        "best_epoch": int(best_epoch),
        "state_dict": best_state,
    }


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

    train_base = _load_cache_bundle(cache_dir / "train_base.npz")
    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_base = _load_cache_bundle(cache_dir / "val_base.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_base = _load_cache_bundle(cache_dir / "test_base.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")

    train_base, train_clip = _subset_train_bundle(train_base, train_clip, train_image_ids)
    class_name_to_idx = _load_class_name_to_idx(Path(args.annotation_file))
    confuse_ids = np.asarray(
        [class_name_to_idx[str(name)] for name in text_npz["confuse_class_names"].tolist()],
        dtype=np.int64,
    )
    train_dataset = PosNegDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        base_logits=np.asarray(train_base["logits"], dtype=np.float32),
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        pos_features=np.asarray(text_npz["pos_features"], dtype=np.float32),
        neg_features=np.asarray(text_npz["neg_features"], dtype=np.float32),
        confuse_ids=confuse_ids,
    )

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    pos_only = _run_variant(
        variant_name="pos_only",
        use_neg_repulsion=False,
        train_dataset=train_dataset,
        val_clip=val_clip,
        val_base=val_base,
        test_clip=test_clip,
        test_base=test_base,
        args=args,
        device=device,
    )
    run_neg_variant = any(
        float(value) > 0.0
        for value in [args.lambda_neg, args.lambda_keep, args.lambda_delta]
    )
    if run_neg_variant:
        pos_neg = _run_variant(
            variant_name="pos_plus_neg_repulsion",
            use_neg_repulsion=True,
            train_dataset=train_dataset,
            val_clip=val_clip,
            val_base=val_base,
            test_clip=test_clip,
            test_base=test_base,
            args=args,
            device=device,
        )
    else:
        pos_neg = {
            "bundle": _zero_bundle(),
            "history": [],
            "best_epoch": -1,
            "state_dict": {},
        }

    comparison_rows = [
        _comparison_row("baseline", baseline_bundle, note="cached baseline reference"),
        _comparison_row(
            "T_pos only",
            pos_only["bundle"],
            note=f"train_n={len(train_dataset)} best_epoch={pos_only['best_epoch']}",
        ),
    ]
    if run_neg_variant:
        comparison_rows.append(
            _comparison_row(
                "T_pos + T_neg repulsion",
                pos_neg["bundle"],
                note=f"train_n={len(train_dataset)} best_epoch={pos_neg['best_epoch']}",
            )
        )
    _write_csv(output_dir / "main_comparison.csv", comparison_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "seed": int(args.seed),
                "device": str(device),
                "train_samples": int(len(train_dataset)),
                "baseline": _bundle_summary(baseline_bundle),
                "pos_only": {
                    "bundle": _bundle_summary(pos_only["bundle"]),
                    "history": _json_ready(pos_only["history"]),
                    "best_epoch": int(pos_only["best_epoch"]),
                },
                "pos_plus_neg_repulsion": None
                if not run_neg_variant
                else {
                    "bundle": _bundle_summary(pos_neg["bundle"]),
                    "history": _json_ready(pos_neg["history"]),
                    "best_epoch": int(pos_neg["best_epoch"]),
                },
                "comparison_rows": comparison_rows,
                "loss_config": {
                    "lambda_pos": float(args.lambda_pos),
                    "lambda_neg": float(args.lambda_neg),
                    "lambda_keep": float(args.lambda_keep),
                    "lambda_delta": float(args.lambda_delta),
                    "beta": float(args.beta),
                    "neg_margin": float(args.neg_margin),
                    "keep_margin": float(args.keep_margin),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
