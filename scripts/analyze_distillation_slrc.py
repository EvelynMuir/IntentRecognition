#!/usr/bin/env python3
"""Add fixed SLR-C prior into the current distillation framework."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clip  # type: ignore

from scripts.analyze_privileged_distillation import (
    StudentMLP,
    TeacherMLP,
    _align_text_bundle_to_clip,
    _bernoulli_kl_per_class,
    _compute_sample_agreement,
    _evaluate_score_bundle,
    _load_cache_bundle,
    _load_text_bundle,
    _logit_np,
    _resolve_device,
    _resolve_output_dir,
    _set_component_seed,
    _set_seed,
    _sigmoid_np,
)
from src.models.components.aslloss import AsymmetricLossOptimized

DEFAULT_BASE_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_TEXT_DIR = PROJECT_ROOT / "logs" / "analysis" / "vlm_full_20260316"
DEFAULT_TEACHER_RUN_DIR = PROJECT_ROOT / "logs" / "analysis" / "privileged_distillation_text_teacher_seedfix_20260316"
DEFAULT_GEMINI_FILE = PROJECT_ROOT.parent / "Intentonomy" / "data" / "intent_description_gemini.json"
DEFAULT_ANNOTATION_FILE = PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SLR-C + residual distillation experiments on cached CLIP features."
    )
    parser.add_argument("--reuse-cache-dir", type=str, default=str(DEFAULT_BASE_CACHE_DIR))
    parser.add_argument("--slr-cache-dir", type=str, default=None)
    parser.add_argument(
        "--train-text-npz",
        type=str,
        default=str(DEFAULT_TEXT_DIR / "rationale_full_bge_features.npz"),
    )
    parser.add_argument(
        "--val-text-npz",
        type=str,
        default=str(DEFAULT_TEXT_DIR / "val_rationale_baseline_pred_bge_features.npz"),
    )
    parser.add_argument(
        "--test-text-npz",
        type=str,
        default=str(DEFAULT_TEXT_DIR / "test_rationale_baseline_pred_bge_features.npz"),
    )
    parser.add_argument("--teacher-run-dir", type=str, default=str(DEFAULT_TEACHER_RUN_DIR))
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION_FILE))
    parser.add_argument("--gemini-file", type=str, default=str(DEFAULT_GEMINI_FILE))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--standard-kd-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-kd-weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--slr-alpha", type=float, default=0.3)
    return parser.parse_args()


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
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _load_class_names(annotation_file: Path) -> List[str]:
    if annotation_file.suffix == ".json":
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = sorted(
            data["categories"],
            key=lambda item: int(item.get("id", item.get("category_id"))),
        )
        return [str(category["name"]) for category in categories]
    if annotation_file.suffix == ".mat":
        gemini_file = annotation_file.parent.parent / "emotion_description_gemini.json"
        if gemini_file.exists():
            payload = json.loads(gemini_file.read_text(encoding="utf-8"))
            return [str(item["emotion_name"]) for item in payload.get("emotions", [])]
    raise ValueError(f"Unsupported annotation file for class names: {annotation_file}")


def _ordered_unique(strings: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in strings:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _build_scenario_text_pool(class_names: Sequence[str], gemini_file: Path) -> List[List[str]]:
    data = json.loads(gemini_file.read_text(encoding="utf-8"))
    if "emotions" in data:
        emotions = data.get("emotions", [])
        scenario_pools = []
        for index, item in enumerate(emotions[: len(class_names)]):
            scenario_texts = [str(arch.get("text_query", "")) for arch in item.get("archetypes", [])]
            pool = _ordered_unique(scenario_texts)
            if not pool:
                pool = [str(class_names[index])]
            scenario_pools.append(pool)
        while len(scenario_pools) < len(class_names):
            scenario_pools.append([str(class_names[len(scenario_pools)])])
        return scenario_pools[: len(class_names)]
    scenario_pools: List[List[str]] = []
    for index, item in enumerate(data[: len(class_names)]):
        scenario_texts = [str(desc.get("Text Query", "")) for desc in item.get("description", [])]
        pool = _ordered_unique(scenario_texts)
        if not pool:
            pool = [str(class_names[index])]
        scenario_pools.append(pool)
    while len(scenario_pools) < len(class_names):
        scenario_pools.append([str(class_names[len(scenario_pools)])])
    return scenario_pools[: len(class_names)]


def _encode_text_pool(
    clip_model: torch.nn.Module,
    texts_per_class: Sequence[Sequence[str]],
) -> np.ndarray:
    device = next(clip_model.parameters()).device
    embeddings: List[torch.Tensor] = []
    clip_model.eval()
    with torch.inference_mode():
        for text_group in texts_per_class:
            tokens = clip.tokenize([str(text) for text in text_group], truncate=True).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            mean_feature = text_features.mean(dim=0)
            mean_feature = F.normalize(mean_feature, dim=0)
            embeddings.append(mean_feature.detach().cpu())
    return torch.stack(embeddings, dim=0).numpy()


def _normalize_scores_per_sample(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (scores - mean) / std


def _apply_slr(
    baseline_logits: np.ndarray,
    prior_logits: np.ndarray,
    topk: int,
    alpha: float,
) -> np.ndarray:
    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    prior_logits = np.asarray(prior_logits, dtype=np.float32)
    fused_prior = _normalize_scores_per_sample(prior_logits)
    output = baseline_logits.copy()
    num_classes = baseline_logits.shape[1]
    topk = max(1, min(int(topk), num_classes))
    topk_idx = np.argpartition(-baseline_logits, kth=topk - 1, axis=1)[:, :topk]
    row_idx = np.arange(baseline_logits.shape[0])[:, None]
    output[row_idx, topk_idx] = baseline_logits[row_idx, topk_idx] + float(alpha) * fused_prior[row_idx, topk_idx]
    return output


def _text_logits_from_features(
    image_features: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    similarity = np.asarray(image_features, dtype=np.float32) @ np.asarray(text_embeddings, dtype=np.float32).T
    return similarity * float(logit_scale)


def _slr_feature_view(bundle: Mapping[str, Any]) -> np.ndarray:
    if "full_features" in bundle:
        return np.asarray(bundle["full_features"], dtype=np.float32)
    if "crop_features" in bundle:
        return np.asarray(bundle["crop_features"], dtype=np.float32)
    return np.asarray(bundle["features"], dtype=np.float32)


class SLRCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_features: np.ndarray,
        slr_logits: np.ndarray,
        labels: np.ndarray,
        soft_labels: np.ndarray,
        agreement: np.ndarray,
        teacher_probs: np.ndarray,
    ) -> None:
        self.image_features = np.asarray(image_features, dtype=np.float32)
        self.slr_logits = np.asarray(slr_logits, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.soft_labels = np.asarray(soft_labels, dtype=np.float32)
        self.agreement = np.asarray(agreement, dtype=np.float32)
        self.teacher_probs = np.asarray(teacher_probs, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image_features": torch.from_numpy(self.image_features[idx]),
            "slr_logits": torch.from_numpy(self.slr_logits[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "soft_labels": torch.from_numpy(self.soft_labels[idx]),
            "agreement": torch.tensor(float(self.agreement[idx]), dtype=torch.float32),
            "teacher_probs": torch.from_numpy(self.teacher_probs[idx]),
        }


class ResidualStudent(nn.Module):
    def __init__(self, image_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features: torch.Tensor, slr_logits: torch.Tensor) -> torch.Tensor:
        return slr_logits + self.net(image_features)


@torch.inference_mode()
def _predict_baseline_logits(
    model: StudentMLP,
    image_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    for start in range(0, image_features.shape[0], int(batch_size)):
        feats = torch.as_tensor(image_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits = model(feats)
        outputs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _per_class_ap_rows(method: str, class_names: Sequence[str], bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    metrics = bundle["classwise"]["test"]
    per_class_ap = np.asarray(metrics.get("per_class_ap", []), dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for idx, ap in enumerate(per_class_ap.tolist()):
        rows.append(
            {
                "method": method,
                "class_id": idx,
                "class_name": str(class_names[idx]) if idx < len(class_names) else f"class_{idx}",
                "AP": float(ap),
            }
        )
    return rows


@torch.inference_mode()
def _predict_teacher(
    teacher_model: TeacherMLP,
    text_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    teacher_model = teacher_model.to(device).eval()
    for start in range(0, text_features.shape[0], int(batch_size)):
        batch = torch.as_tensor(text_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits = teacher_model(text_features=batch)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


@torch.inference_mode()
def _predict_residual_student(
    model: ResidualStudent,
    image_features: np.ndarray,
    slr_logits: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model = model.to(device).eval()
    for start in range(0, image_features.shape[0], int(batch_size)):
        feats = torch.as_tensor(image_features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        base = torch.as_tensor(slr_logits[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits = model(feats, base)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _train_residual_student(
    mode: str,
    model: ResidualStudent,
    train_dataset: SLRCDataset,
    val_image_features: np.ndarray,
    val_slr_logits: np.ndarray,
    val_targets: np.ndarray,
    test_image_features: np.ndarray,
    test_slr_logits: np.ndarray,
    test_targets: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
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
        total_supervised = 0.0
        total_kd = 0.0
        batch_count = 0

        for batch in loader:
            image_features = batch["image_features"].to(device, non_blocking=True)
            slr_logits = batch["slr_logits"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            agreement = batch["agreement"].to(device, non_blocking=True)
            teacher_probs = batch["teacher_probs"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image_features, slr_logits)
            supervised_per_class = criterion(logits, labels, reduction="none")
            supervised_per_sample = supervised_per_class.sum(dim=1)

            kd_per_class = _bernoulli_kl_per_class(
                student_logits=logits,
                teacher_probs=teacher_probs,
                temperature=float(args.temperature),
            )
            kd_per_sample = kd_per_class.sum(dim=1)

            if mode == "supervised":
                loss_per_sample = supervised_per_sample
            elif mode == "standard_kd":
                loss_per_sample = supervised_per_sample + float(args.standard_kd_weight) * kd_per_sample
            elif mode == "dynamic_kd":
                teacher_weight = (1.0 - agreement).unsqueeze(1)
                supervised_weight = 1.0 - teacher_weight
                loss_per_class = (
                    supervised_weight * supervised_per_class
                    + teacher_weight * float(args.dynamic_kd_weight) * kd_per_class
                )
                loss_per_sample = loss_per_class.sum(dim=1)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_supervised += float(supervised_per_sample.mean().detach().cpu())
            total_kd += float(kd_per_sample.mean().detach().cpu())
            batch_count += 1

        val_scores = _predict_residual_student(
            model=model,
            image_features=val_image_features,
            slr_logits=val_slr_logits,
            device=device,
            batch_size=int(args.batch_size),
        )
        test_scores = _predict_residual_student(
            model=model,
            image_features=test_image_features,
            slr_logits=test_slr_logits,
            device=device,
            batch_size=int(args.batch_size),
        )
        bundle = _evaluate_score_bundle(
            val_scores=val_scores,
            val_targets=val_targets,
            test_scores=test_scores,
            test_targets=test_targets,
        )
        val_macro = float(bundle["classwise"]["val"]["macro"])
        history.append(
            {
                "epoch": int(epoch),
                "loss": total_loss / max(1, batch_count),
                "supervised_loss": total_supervised / max(1, batch_count),
                "kd_loss": total_kd / max(1, batch_count),
                "val_macro_classwise": val_macro,
                "test_macro_classwise": float(bundle["classwise"]["test"]["macro"]),
                "test_hard_classwise": float(bundle["classwise"]["test"]["hard"]),
            }
        )
        print(
            f"[SLRC-Distill][{mode}] epoch "
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
        raise RuntimeError(f"Training failed for mode={mode}")

    model.load_state_dict(best_state)
    return {
        "bundle": best_bundle,
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": best_state,
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


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))

    cache_dir = Path(args.reuse_cache_dir)
    slr_cache_dir = Path(args.slr_cache_dir) if args.slr_cache_dir is not None else cache_dir

    train_clip = _load_cache_bundle(cache_dir / "train_clip.npz")
    val_clip = _load_cache_bundle(cache_dir / "val_clip.npz")
    test_clip = _load_cache_bundle(cache_dir / "test_clip.npz")
    train_slr_clip = _load_cache_bundle(slr_cache_dir / "train_clip.npz")
    val_slr_clip = _load_cache_bundle(slr_cache_dir / "val_clip.npz")
    test_slr_clip = _load_cache_bundle(slr_cache_dir / "test_clip.npz")

    for name, left_ids, right_ids in [
        ("train", train_clip["image_ids"], train_slr_clip["image_ids"]),
        ("val", val_clip["image_ids"], val_slr_clip["image_ids"]),
        ("test", test_clip["image_ids"], test_slr_clip["image_ids"]),
    ]:
        if [str(x) for x in left_ids] != [str(x) for x in right_ids]:
            raise RuntimeError(f"{name} image_ids mismatch between student cache and SLR cache.")

    train_text = _align_text_bundle_to_clip(
        _load_text_bundle(Path(args.train_text_npz), required_keys=["image_ids", "features"]),
        train_clip,
    )
    val_text = _align_text_bundle_to_clip(
        _load_text_bundle(Path(args.val_text_npz), required_keys=["image_ids", "features"]),
        val_clip,
    )
    test_text = _align_text_bundle_to_clip(
        _load_text_bundle(Path(args.test_text_npz), required_keys=["image_ids", "features"]),
        test_clip,
    )

    class_names = _load_class_names(Path(args.annotation_file))
    scenario_pools = _build_scenario_text_pool(class_names, Path(args.gemini_file))
    teacher_run_dir = Path(args.teacher_run_dir)
    teacher_summary = json.loads((teacher_run_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_state = torch.load(teacher_run_dir / "baseline_best.pt", map_location="cpu", weights_only=True)
    baseline_model = StudentMLP(
        image_dim=int(np.asarray(train_clip["features"], dtype=np.float32).shape[1]),
        hidden_dim=768,
        num_classes=int(np.asarray(train_clip["labels"], dtype=np.float32).shape[1]),
        dropout=0.1,
        feature_proj_dim=256,
    ).to(device)
    baseline_model.load_state_dict(baseline_state, strict=False)

    train_base = {
        "logits": _predict_baseline_logits(baseline_model, np.asarray(train_clip["features"], dtype=np.float32), device, int(args.batch_size)),
        "labels": np.asarray(train_clip["labels"], dtype=np.float32),
    }
    val_base = {
        "logits": _predict_baseline_logits(baseline_model, np.asarray(val_clip["features"], dtype=np.float32), device, int(args.batch_size)),
        "labels": np.asarray(val_clip["labels"], dtype=np.float32),
    }
    test_base = {
        "logits": _predict_baseline_logits(baseline_model, np.asarray(test_clip["features"], dtype=np.float32), device, int(args.batch_size)),
        "labels": np.asarray(test_clip["labels"], dtype=np.float32),
    }

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model = clip_model.eval().to(device)
    logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    scenario_text_embeddings = _encode_text_pool(clip_model, scenario_pools)

    train_prior_logits = _text_logits_from_features(_slr_feature_view(train_slr_clip), scenario_text_embeddings, logit_scale)
    val_prior_logits = _text_logits_from_features(_slr_feature_view(val_slr_clip), scenario_text_embeddings, logit_scale)
    test_prior_logits = _text_logits_from_features(_slr_feature_view(test_slr_clip), scenario_text_embeddings, logit_scale)

    train_slr_logits = _apply_slr(train_base["logits"], train_prior_logits, topk=int(args.topk), alpha=float(args.slr_alpha))
    val_slr_logits = _apply_slr(val_base["logits"], val_prior_logits, topk=int(args.topk), alpha=float(args.slr_alpha))
    test_slr_logits = _apply_slr(test_base["logits"], test_prior_logits, topk=int(args.topk), alpha=float(args.slr_alpha))

    slr_bundle = _evaluate_score_bundle(
        val_scores=_sigmoid_np(val_slr_logits),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_scores=_sigmoid_np(test_slr_logits),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
    )

    teacher_state = torch.load(Path(args.teacher_run_dir) / "teacher_best.pt", map_location="cpu", weights_only=True)
    teacher_model = TeacherMLP(
        text_dim=int(np.asarray(train_text["features"], dtype=np.float32).shape[1]),
        hidden_dim=1024,
        num_classes=int(np.asarray(train_clip["labels"], dtype=np.float32).shape[1]),
        dropout=0.1,
        input_mode="text_only",
    )
    teacher_model.load_state_dict(teacher_state, strict=False)

    teacher_probs_train = _predict_teacher(
        teacher_model=teacher_model,
        text_features=np.asarray(train_text["features"], dtype=np.float32),
        device=device,
        batch_size=int(args.batch_size),
    )
    teacher_probs_train = _sigmoid_np(_logit_np(teacher_probs_train) / float(args.temperature))

    train_labels = np.asarray(train_clip["labels"], dtype=np.float32)
    train_soft_labels = np.asarray(train_clip["soft_labels"], dtype=np.float32)
    train_agreement = _compute_sample_agreement(train_labels, train_soft_labels, mode="min")

    train_dataset = SLRCDataset(
        image_features=np.asarray(train_clip["features"], dtype=np.float32),
        slr_logits=np.asarray(train_slr_logits, dtype=np.float32),
        labels=train_labels,
        soft_labels=train_soft_labels,
        agreement=train_agreement,
        teacher_probs=teacher_probs_train,
    )

    image_dim = int(np.asarray(train_clip["features"], dtype=np.float32).shape[1])
    num_classes = int(train_labels.shape[1])

    supervised_seed = _set_component_seed(int(args.seed), offset=100)
    supervised_model = ResidualStudent(image_dim=image_dim, hidden_dim=int(args.hidden_dim), num_classes=num_classes, dropout=float(args.dropout)).to(device)
    print(f"[SLRC-Distill] training supervised residual seed={supervised_seed}")
    supervised_result = _train_residual_student(
        mode="supervised",
        model=supervised_model,
        train_dataset=train_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_slr_logits=np.asarray(val_slr_logits, dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_slr_logits=np.asarray(test_slr_logits, dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
        device=device,
        args=args,
    )

    standard_seed = _set_component_seed(int(args.seed), offset=200)
    standard_model = ResidualStudent(image_dim=image_dim, hidden_dim=int(args.hidden_dim), num_classes=num_classes, dropout=float(args.dropout)).to(device)
    print(f"[SLRC-Distill] training standard KD residual seed={standard_seed}")
    standard_result = _train_residual_student(
        mode="standard_kd",
        model=standard_model,
        train_dataset=train_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_slr_logits=np.asarray(val_slr_logits, dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_slr_logits=np.asarray(test_slr_logits, dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
        device=device,
        args=args,
    )

    dynamic_seed = _set_component_seed(int(args.seed), offset=300)
    dynamic_model = ResidualStudent(image_dim=image_dim, hidden_dim=int(args.hidden_dim), num_classes=num_classes, dropout=float(args.dropout)).to(device)
    print(f"[SLRC-Distill] training dynamic KD residual seed={dynamic_seed}")
    dynamic_result = _train_residual_student(
        mode="dynamic_kd",
        model=dynamic_model,
        train_dataset=train_dataset,
        val_image_features=np.asarray(val_clip["features"], dtype=np.float32),
        val_slr_logits=np.asarray(val_slr_logits, dtype=np.float32),
        val_targets=np.asarray(val_base["labels"], dtype=np.float32),
        test_image_features=np.asarray(test_clip["features"], dtype=np.float32),
        test_slr_logits=np.asarray(test_slr_logits, dtype=np.float32),
        test_targets=np.asarray(test_base["labels"], dtype=np.float32),
        device=device,
        args=args,
    )

    comparison_rows = [
        _comparison_row("teacher_text_only", teacher_summary["teacher"]["bundle"], note="reference teacher"),
        _comparison_row("slr_c_fixed", slr_bundle, note="fixed scenario SLR-C"),
        _comparison_row("slr_c_residual_sup", supervised_result["bundle"], note=f"best_epoch={supervised_result['best_epoch']}"),
        _comparison_row("slr_c_residual_standard_kd", standard_result["bundle"], note=f"best_epoch={standard_result['best_epoch']}"),
        _comparison_row("slr_c_residual_dynamic_kd", dynamic_result["bundle"], note=f"best_epoch={dynamic_result['best_epoch']}"),
    ]
    _write_csv(output_dir / "main_comparison.csv", comparison_rows)
    if class_names:
        per_class_ap_rows: List[Dict[str, Any]] = []
        per_class_ap_rows.extend(_per_class_ap_rows("teacher_text_only", class_names, teacher_summary["teacher"]["bundle"]))
        per_class_ap_rows.extend(_per_class_ap_rows("slr_c_fixed", class_names, slr_bundle))
        per_class_ap_rows.extend(_per_class_ap_rows("slr_c_residual_sup", class_names, supervised_result["bundle"]))
        per_class_ap_rows.extend(_per_class_ap_rows("slr_c_residual_standard_kd", class_names, standard_result["bundle"]))
        per_class_ap_rows.extend(_per_class_ap_rows("slr_c_residual_dynamic_kd", class_names, dynamic_result["bundle"]))
        _write_csv(output_dir / "per_class_ap.csv", per_class_ap_rows)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "comparison_rows": comparison_rows,
                "slr_c_fixed": _json_ready(slr_bundle),
                "teacher_reference": _json_ready(teacher_summary["teacher"]),
                "slr_c_residual_sup": _json_ready({k: v for k, v in supervised_result.items() if k != "state_dict"}),
                "slr_c_residual_standard_kd": _json_ready({k: v for k, v in standard_result.items() if k != "state_dict"}),
                "slr_c_residual_dynamic_kd": _json_ready({k: v for k, v in dynamic_result.items() if k != "state_dict"}),
                "config": {
                    "teacher_run_dir": str(args.teacher_run_dir),
                    "reuse_cache_dir": str(args.reuse_cache_dir),
                    "slr_cache_dir": str(slr_cache_dir),
                    "topk": int(args.topk),
                    "slr_alpha": float(args.slr_alpha),
                    "temperature": float(args.temperature),
                    "standard_kd_weight": float(args.standard_kd_weight),
                    "dynamic_kd_weight": float(args.dynamic_kd_weight),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SLRC-Distill] finished. artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
