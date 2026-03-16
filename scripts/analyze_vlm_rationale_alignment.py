#!/usr/bin/env python3
"""Analyze VLM rationale alignment on top of cached CLIP image features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is required for analyze_vlm_rationale_alignment.py. "
        "Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_data_driven_agent_evidence_verification import (
    _evaluate_with_class_thresholds,
    _json_ready,
)
from src.models.components.aslloss import AsymmetricLossOptimized
from src.models.intentonomy_clip_vit_slot_module import INTENTONOMY_DESCRIPTIONS
from src.utils.decision_rule_calibration import search_classwise_thresholds
from src.utils.text_prior_analysis import evaluate_with_validation_threshold

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rationale alignment experiments on cached CLIP features."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--text-feature-npz", type=str, default=None)
    parser.add_argument("--train-id-source-npz", type=str, default=None)
    parser.add_argument("--text-source", type=str, default="label_only", choices=["label_only", "external_npz"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--align-weight", type=float, default=0.5)
    parser.add_argument("--align-temperature", type=float, default=0.1)
    parser.add_argument("--align-mode", type=str, default="infonce", choices=["infonce", "mse", "cosine"])
    return parser.parse_args()


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_vlm_rationale_alignment"
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


def _assert_same_ids(name: str, left_ids: Sequence[str], right_ids: Sequence[str]) -> None:
    if list(left_ids) != list(right_ids):
        raise RuntimeError(f"{name} image order mismatch between cached bundles.")


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


def _encode_label_only_targets(
    image_ids: Sequence[str],
    labels: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    texts = []
    for row in np.asarray(labels, dtype=np.float32):
        positive_ids = np.where(row > 0.0)[0].tolist()
        if positive_ids:
            parts = [INTENTONOMY_DESCRIPTIONS[idx] for idx in positive_ids]
            text = "Concurrent human intents: " + " | ".join(parts)
        else:
            text = "Concurrent human intents: none"
        texts.append(text)

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    features: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            tokens = clip.tokenize(texts[start : start + 64], truncate=True).to(device)
            batch_features = clip_model.encode_text(tokens).float()
            batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


def _load_external_text_features(path: Path, train_image_ids: Sequence[str]) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    image_ids = [str(item) for item in arr["image_ids"].tolist()]
    features = np.asarray(arr["features"], dtype=np.float32)
    id_to_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    row_ids = []
    for image_id in train_image_ids:
        if image_id not in id_to_idx:
            raise KeyError(f"Missing text feature for image_id={image_id}")
        row_ids.append(int(id_to_idx[image_id]))
    return features[np.asarray(row_ids, dtype=np.int64)]


def _load_external_feature_bundle(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    return {
        "image_ids": [str(item) for item in arr["image_ids"].tolist()],
        "features": np.asarray(arr["features"], dtype=np.float32),
    }


def _subset_train_bundle(
    train_base: Dict[str, Any],
    train_clip: Dict[str, Any],
    target_image_ids: Sequence[str],
) -> tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    id_to_idx = {str(image_id): idx for idx, image_id in enumerate(train_base["image_ids"])}
    row_ids: List[int] = []
    missing: List[str] = []
    for image_id in target_image_ids:
        row_idx = id_to_idx.get(str(image_id))
        if row_idx is None:
            missing.append(str(image_id))
        else:
            row_ids.append(int(row_idx))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing {len(missing)} training ids in cache: {preview}")
    row_ids_np = np.asarray(row_ids, dtype=np.int64)
    subset_base = {
        key: value[row_ids_np] if isinstance(value, np.ndarray) else [value[idx] for idx in row_ids_np.tolist()]
        for key, value in train_base.items()
    }
    subset_clip = {
        key: value[row_ids_np] if isinstance(value, np.ndarray) else [value[idx] for idx in row_ids_np.tolist()]
        for key, value in train_clip.items()
    }
    return subset_base, subset_clip, row_ids_np


class AlignmentDataset(Dataset):
    def __init__(
        self,
        image_features: np.ndarray,
        base_logits: np.ndarray,
        labels: np.ndarray,
        text_features: np.ndarray,
    ) -> None:
        self.image_features = np.asarray(image_features, dtype=np.float32)
        self.base_logits = np.asarray(base_logits, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.text_features = np.asarray(text_features, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image_features": torch.from_numpy(self.image_features[idx]),
            "base_logits": torch.from_numpy(self.base_logits[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "text_features": torch.from_numpy(self.text_features[idx]),
        }


class VLMAlignmentModel(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, num_classes: int) -> None:
        super().__init__()
        self.shared_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, text_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, image_features: torch.Tensor, base_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        adapted = self.shared_adapter(image_features)
        projected = self.projector(adapted)
        residual = self.residual_head(adapted)
        final_logits = base_logits + residual
        return {
            "adapted": adapted,
            "projected": projected,
            "final_logits": final_logits,
        }


def _alignment_loss(
    projected: torch.Tensor,
    text_features: torch.Tensor,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    proj = F.normalize(projected, dim=-1)
    text = F.normalize(text_features, dim=-1)
    if mode == "mse":
        return F.mse_loss(proj, text)
    if mode == "cosine":
        return (1.0 - F.cosine_similarity(proj, text, dim=-1)).mean()
    logits = proj @ text.t() / max(float(temperature), 1e-6)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


@torch.inference_mode()
def _predict_scores(
    model: VLMAlignmentModel,
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
        batch_out = model(image_batch, logit_batch)["final_logits"]
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

    train_base = _load_cache_bundle(cache_dir / "train_base.npz")
    val_base = _load_cache_bundle(cache_dir / "val_base.npz")
    test_base = _load_cache_bundle(cache_dir / "test_base.npz")
    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")
    _assert_same_ids("train", train_base["image_ids"], train_clip["image_ids"])
    _assert_same_ids("val", val_base["image_ids"], val_clip["image_ids"])
    _assert_same_ids("test", test_base["image_ids"], test_clip["image_ids"])

    target_train_image_ids: Sequence[str] | None = None
    external_bundle: Dict[str, Any] | None = None
    if args.text_feature_npz is not None:
        external_bundle = _load_external_feature_bundle(Path(args.text_feature_npz))
        target_train_image_ids = external_bundle["image_ids"]
    elif args.train_id_source_npz is not None:
        target_train_image_ids = _load_external_feature_bundle(Path(args.train_id_source_npz))["image_ids"]

    if target_train_image_ids is not None:
        train_base, train_clip, _ = _subset_train_bundle(
            train_base=train_base,
            train_clip=train_clip,
            target_image_ids=target_train_image_ids,
        )

    if args.text_source == "label_only":
        train_text_features = _encode_label_only_targets(
            image_ids=train_base["image_ids"],
            labels=np.asarray(train_base["labels"], dtype=np.float32),
            device=device,
        )
    else:
        if args.text_feature_npz is None:
            raise ValueError("--text-feature-npz is required when --text-source=external_npz")
        if external_bundle is None:
            external_bundle = _load_external_feature_bundle(Path(args.text_feature_npz))
        train_text_features = _load_external_text_features(
            path=Path(args.text_feature_npz),
            train_image_ids=train_base["image_ids"],
        )

    dataset = AlignmentDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        base_logits=np.asarray(train_base["logits"], dtype=np.float32),
        labels=np.asarray(train_base["labels"], dtype=np.float32),
        text_features=train_text_features,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    baseline_bundle = _evaluate_score_bundle(
        val_scores=np.asarray(val_base["scores"], dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=np.asarray(test_base["scores"], dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    model = VLMAlignmentModel(
        input_dim=int(train_clip["features"].shape[1]),
        text_dim=int(train_text_features.shape[1]),
        num_classes=int(train_base["labels"].shape[1]),
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
        total_align = 0.0
        batch_count = 0
        for batch in loader:
            image_features = batch["image_features"].to(device, non_blocking=True)
            base_logits = batch["base_logits"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            text_features = batch["text_features"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(image_features, base_logits)
            cls_loss = criterion(outputs["final_logits"], labels, reduction="mean")
            align_loss = _alignment_loss(
                outputs["projected"],
                text_features,
                mode=str(args.align_mode),
                temperature=float(args.align_temperature),
            )
            loss = cls_loss + float(args.align_weight) * align_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_cls += float(cls_loss.detach().cpu())
            total_align += float(align_loss.detach().cpu())
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
                "align_loss": total_align / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
            }
        )
        print(
            "[VLM-Align] epoch "
            f"{epoch:02d} | loss={history[-1]['loss']:.4f} "
            f"| val_macro={val_macro * 100.0:.2f} "
            f"| test_macro={float(bundle['classwise']['test']['macro']) * 100.0:.2f} "
            f"| test_hard={float(bundle['classwise']['test']['hard']) * 100.0:.2f}"
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
        raise RuntimeError("Alignment training did not produce a valid best checkpoint.")

    comparison_rows = [
        _comparison_row("baseline", baseline_bundle, note="cached baseline reference"),
        _comparison_row(
            "baseline + rationale alignment",
            best_bundle,
            note=(
                f"text_source={args.text_source} align_mode={args.align_mode} "
                f"align_weight={args.align_weight} train_n={len(dataset)} best_epoch={best_epoch}"
            ),
        ),
    ]
    _write_csv(output_dir / "main_comparison.csv", comparison_rows)
    (output_dir / "training_history.json").write_text(
        json.dumps(_json_ready(history), indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": int(best_epoch),
            "args": vars(args),
        },
        output_dir / "alignment_best.pt",
    )
    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "text_source": str(args.text_source),
        "train_samples": int(len(dataset)),
        "align_mode": str(args.align_mode),
        "align_weight": float(args.align_weight),
        "align_temperature": float(args.align_temperature),
        "baseline": _json_ready(baseline_bundle),
        "alignment": _json_ready(best_bundle),
        "comparison_rows": comparison_rows,
        "best_epoch": int(best_epoch),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
