#!/usr/bin/env python3
"""Train and evaluate an FDIL adaptation for Flickr-LDL or Twitter-LDL.

The adaptation preserves FDIL's functional split while respecting the label
distribution simplex:

* SLR-C: heterogeneous CLIP text priors, Top-K local reranking, and a residual
  image student.
* UTD: a training-only semantic text teacher, cross-fitted on the train split,
  with a gate given by normalized annotation-distribution entropy.

Model selection uses validation KL only. The official test split is evaluated
once after restoring each method's validation-selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flickrldl", "twitterldl", "emotion6"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--description-file", type=Path, default=None,
        help="Shared Gemini SLR-C prior JSON. Defaults to ../LDL/processed/semantic_priors/ldl_gemini_slrc_prior.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", type=str, default="2026,2027,2028")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--teacher-max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--teacher-hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--gate-mode", choices=["max_mass", "normalized_entropy"], default="max_mass",
        help="UTD uncertainty gate. max_mass preserves FDIL's annotator-agreement interpretation: g=1-max(y).",
    )
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--slr-topk", type=int, default=5)
    parser.add_argument("--slr-alpha", type=float, default=0.3)
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument(
        "--rationale-features",
        type=Path,
        default=None,
        help="Optional train rationale feature NPZ with image_ids/features. When provided, UTD uses the VLM rationale teacher.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    feature_key = "full_features" if "full_features" in payload.files else "features"
    targets = np.asarray(payload["soft_labels"], dtype=np.float32)
    targets /= np.maximum(targets.sum(axis=1, keepdims=True), 1e-12)
    return {
        "features": np.asarray(payload[feature_key], dtype=np.float32),
        "targets": targets,
        "image_ids": np.asarray(payload["image_ids"]).astype(str),
    }


def align_feature_bundle(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    ids = np.asarray(payload["image_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)
    if len(ids) != len(features):
        raise ValueError(f"{path}: image_ids/features length mismatch")
    positions = {image_id: index for index, image_id in enumerate(ids.tolist())}
    if len(positions) != len(ids):
        raise ValueError(f"{path}: duplicate image IDs")
    missing = [image_id for image_id in expected_ids.tolist() if image_id not in positions]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} expected train IDs; first={missing[0]}")
    return features[[positions[image_id] for image_id in expected_ids.tolist()]]


def normalized_entropy(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    entropy = -(targets * np.log(np.clip(targets, 1e-12, 1.0))).sum(axis=1)
    return np.clip(entropy / math.log(targets.shape[1]), 0.0, 1.0).astype(np.float32)


def uncertainty_gate(targets: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normalized_entropy":
        return normalized_entropy(targets)
    if mode == "max_mass":
        return (1.0 - np.max(np.asarray(targets, dtype=np.float32), axis=1)).astype(np.float32)
    raise ValueError(f"Unsupported gate mode: {mode}")


def distribution_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    y /= np.maximum(y.sum(axis=1, keepdims=True), eps)
    p /= np.maximum(p.sum(axis=1, keepdims=True), eps)
    denominator = np.maximum(y + p, eps)
    cosine = (
        np.sum(y * p, axis=1)
        / np.maximum(np.linalg.norm(y, axis=1) * np.linalg.norm(p, axis=1), eps)
    )
    target_ranks = rankdata(y, axis=1, method="average")
    prediction_ranks = rankdata(p, axis=1, method="average")
    target_ranks -= target_ranks.mean(axis=1, keepdims=True)
    prediction_ranks -= prediction_ranks.mean(axis=1, keepdims=True)
    rank_denominator = np.linalg.norm(target_ranks, axis=1) * np.linalg.norm(prediction_ranks, axis=1)
    spearman = np.divide(
        np.sum(target_ranks * prediction_ranks, axis=1),
        rank_denominator,
        out=np.zeros(len(y), dtype=np.float64),
        where=rank_denominator > eps,
    )

    # DeltaLDL µ (Li et al., ICML 2025), matching its published PyLDL
    # implementation: normalized area under the approximately-correct curve.
    per_sample_kl = np.sum(
        y * (np.log(np.clip(y, eps, 1.0)) - np.log(np.clip(p, eps, 1.0))), axis=1
    )
    uniform = np.full_like(y, 1.0 / y.shape[1])
    delta_0 = float(
        np.sum(y * (np.log(np.clip(y, eps, 1.0)) - np.log(uniform)), axis=1).mean()
    )
    mu = 0.0 if delta_0 <= eps else float(
        (1.0 - np.minimum(per_sample_kl, delta_0).mean() / delta_0) * 100.0
    )

    positive_tuple, negative_tuple = polarity_ids(y.shape[1])
    positive_ids = np.asarray(positive_tuple)
    negative_ids = np.asarray(negative_tuple)
    target_divisiveness = np.minimum(y[:, positive_ids].sum(1), y[:, negative_ids].sum(1))
    prediction_divisiveness = np.minimum(p[:, positive_ids].sum(1), p[:, negative_ids].sum(1))
    dvse = np.abs(prediction_divisiveness - target_divisiveness).mean()
    return {
        "chebyshev": float(np.max(np.abs(y - p), axis=1).mean()),
        "clark": float(np.sqrt(np.sum(((y - p) / denominator) ** 2, axis=1)).mean()),
        "kl": float(np.sum(y * (np.log(np.clip(y, eps, 1.0)) - np.log(np.clip(p, eps, 1.0))), axis=1).mean()),
        "cosine": float(cosine.mean()),
        "spearman": float(spearman.mean()),
        "mu": mu,
        "dvse": float(dvse),
    }


def polarity_ids(num_classes: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if num_classes == 8:
        return (0, 3, 5), (1, 4, 6, 7)  # Flickr/Twitter; awe excluded.
    if num_classes == 7:
        return (3, 5), (0, 1, 2, 4)  # Emotion6 joy/surprise vs negative; neutral excluded.
    raise ValueError(f"No DVSE polarity configured for {num_classes} classes")


class DistributionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor | None = None) -> torch.Tensor:
        output = self.net(features)
        return output if base_logits is None else base_logits + output


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    base_logits: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        base = None
        if base_logits is not None:
            base = torch.as_tensor(base_logits[start : start + batch_size], dtype=torch.float32, device=device)
        output.append(model(x, base).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def train_model(
    *,
    model: DistributionMLP,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_base: np.ndarray | None = None,
    val_base: np.ndarray | None = None,
    teacher_logits: np.ndarray | None = None,
    entropy_gate: np.ndarray | None = None,
    temperature: float = 2.0,
    kd_weight: float = 1.0,
) -> Dict[str, Any]:
    set_seed(seed)
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(train_features))
    loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=True, generator=generator)
    x_all = torch.as_tensor(train_features, dtype=torch.float32)
    y_all = torch.as_tensor(train_targets, dtype=torch.float32)
    base_all = torch.as_tensor(train_base, dtype=torch.float32) if train_base is not None else None
    teacher_all = torch.as_tensor(teacher_logits, dtype=torch.float32) if teacher_logits is not None else None
    gate_all = torch.as_tensor(entropy_gate, dtype=torch.float32) if entropy_gate is not None else None
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_kl = float("inf")
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for idx in loader:
            x = x_all[idx].to(device)
            y = y_all[idx].to(device)
            base = base_all[idx].to(device) if base_all is not None else None
            logits = model(x, base)
            supervised = -(y * F.log_softmax(logits, dim=1)).sum(dim=1)
            if teacher_all is not None and gate_all is not None:
                teacher = teacher_all[idx].to(device)
                gate = gate_all[idx].to(device)
                q = F.softmax(teacher / temperature, dim=1)
                log_q = F.log_softmax(teacher / temperature, dim=1)
                log_p = F.log_softmax(logits / temperature, dim=1)
                kd = (q * (log_q - log_p)).sum(dim=1) * (temperature**2)
                per_sample = (1.0 - gate) * supervised + gate * kd_weight * kd
            else:
                per_sample = supervised
            loss = per_sample.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_logits = predict_logits(model, val_features, device, batch_size, val_base)
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        val_metrics = distribution_metrics(val_targets, val_probs)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_kl": val_metrics["kl"]})
        if val_metrics["kl"] < best_val_kl - 1e-8:
            best_val_kl = val_metrics["kl"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_kl": best_val_kl,
        "history": history,
    }


def crossfit_teacher(
    semantic_features: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    folds: int,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(targets))
    fold_ids = np.empty(len(targets), dtype=np.int64)
    for fold, heldout in enumerate(np.array_split(order, folds)):
        fold_ids[heldout] = fold
    output = np.zeros_like(targets, dtype=np.float32)
    # A deterministic validation subset is taken only from each teacher's fit rows.
    for fold in range(folds):
        heldout = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        fit = fit[rng.permutation(len(fit))]
        val_count = max(1, int(round(0.1 * len(fit))))
        teacher_val, teacher_train = fit[:val_count], fit[val_count:]
        model = DistributionMLP(semantic_features.shape[1], hidden_dim, targets.shape[1], dropout)
        result = train_model(
            model=model,
            train_features=semantic_features[teacher_train],
            train_targets=targets[teacher_train],
            val_features=semantic_features[teacher_val],
            val_targets=targets[teacher_val],
            device=device,
            batch_size=args.batch_size,
            max_epochs=args.teacher_max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=seed + 1000 + fold,
        )
        output[heldout] = predict_logits(result["model"], semantic_features[heldout], device, args.batch_size)
    return output


def encode_semantic_sources(
    description_file: Path, class_names: Sequence[str], clip_model_name: str, device: torch.device
) -> tuple[Dict[str, np.ndarray], float]:
    import clip

    payload = json.loads(description_file.read_text(encoding="utf-8"))
    by_name = {str(item["emotion_name"]).lower(): item for item in payload["emotions"]}
    prompts: Dict[str, list[list[str]]] = {"lexical": [], "canonical": [], "scenario": []}
    for name in class_names:
        item = by_name[name.lower()]
        prompts["lexical"].append([f"a photo expressing {name}"])
        prompts["canonical"].append([str(item.get("definition", name))])
        scenarios = [str(entry.get("text_query", "")) for entry in item.get("archetypes", [])]
        prompts["scenario"].append([value for value in scenarios if value] or [f"a scene expressing {name}"])

    model, _ = clip.load(clip_model_name, device=device)
    model = model.eval().to(device)
    sources: Dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for source, groups in prompts.items():
            embeddings = []
            for group in groups:
                tokens = clip.tokenize(group, truncate=True).to(device)
                encoded = F.normalize(model.encode_text(tokens).float(), dim=1)
                embeddings.append(F.normalize(encoded.mean(dim=0), dim=0).cpu().numpy())
            sources[source] = np.stack(embeddings).astype(np.float32)
        logit_scale = float(model.logit_scale.exp().item())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sources, logit_scale


def semantic_prior(features: np.ndarray, sources: Mapping[str, np.ndarray], scale: float) -> np.ndarray:
    normalized = []
    for embeddings in sources.values():
        scores = features @ embeddings.T * scale
        mean = scores.mean(axis=1, keepdims=True)
        std = np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
        normalized.append((scores - mean) / std)
    return np.mean(normalized, axis=0).astype(np.float32)


def apply_local_reranking(base_logits: np.ndarray, prior: np.ndarray, topk: int, alpha: float) -> np.ndarray:
    output = np.asarray(base_logits, dtype=np.float32).copy()
    k = max(1, min(int(topk), output.shape[1]))
    candidates = np.argpartition(-output, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(len(output))[:, None]
    output[rows, candidates] += float(alpha) * prior[rows, candidates]
    return output


def evaluate_model(
    result: Mapping[str, Any], features: np.ndarray, targets: np.ndarray, device: torch.device,
    batch_size: int, base: np.ndarray | None = None
) -> tuple[Dict[str, float], np.ndarray]:
    logits = predict_logits(result["model"], features, device, batch_size, base)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return distribution_metrics(targets, probs), probs


def mean_std(rows: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows]))}
        for key in rows[0]
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    data_dir = args.data_dir or PROJECT_ROOT.parent / "LDL" / "processed" / args.dataset
    metadata = np.load(data_dir / "metadata.npz", allow_pickle=False)
    class_names = np.asarray(metadata["class_names"]).astype(str).tolist()
    prior_filename = "emotion6_gemini_slrc_prior.json" if args.dataset == "emotion6" else "ldl_gemini_slrc_prior.json"
    shared_description = data_dir.parent / "semantic_priors" / prior_filename
    description_file = args.description_file or (
        shared_description if shared_description.exists() else data_dir / "emotion_descriptions.json"
    )
    description_sha256 = hashlib.sha256(description_file.read_bytes()).hexdigest()
    splits = {name: load_cache(args.cache_dir / f"{name}_clip.npz") for name in ("train", "val", "test")}
    for name, bundle in splits.items():
        if bundle["targets"].shape[1] != len(class_names):
            raise ValueError(f"{name}: target/class-name mismatch")

    semantic_sources, logit_scale = encode_semantic_sources(description_file, class_names, args.clip_model, device)
    np.savez_compressed(args.output_dir / "semantic_embeddings.npz", logit_scale=logit_scale, **semantic_sources)
    priors = {name: semantic_prior(bundle["features"], semantic_sources, logit_scale) for name, bundle in splits.items()}
    if args.rationale_features is not None:
        privileged_train = align_feature_bundle(args.rationale_features, splits["train"]["image_ids"])
        teacher_description = "3-fold OOF VLM-rationale text teacher, training only"
    else:
        # Fallback used when the rationale service is unavailable. The expected
        # semantic embedding is privileged text-side input at training time only.
        class_semantics = np.mean(np.stack(list(semantic_sources.values()), axis=0), axis=0)
        class_semantics /= np.maximum(np.linalg.norm(class_semantics, axis=1, keepdims=True), 1e-12)
        privileged_train = splits["train"]["targets"] @ class_semantics
        privileged_train /= np.maximum(np.linalg.norm(privileged_train, axis=1, keepdims=True), 1e-12)
        teacher_description = "3-fold OOF distribution-weighted semantic text teacher, training only"
    gate = uncertainty_gate(splits["train"]["targets"], args.gate_mode)

    per_seed_rows = []
    metrics_by_method: Dict[str, list[Dict[str, float]]] = {
        "baseline": [], "utd": [], "slrc": [], "fdil": []
    }
    for seed in seeds:
        print(f"[{args.dataset}] seed={seed}: cross-fitting semantic teacher", flush=True)
        teacher_logits = crossfit_teacher(
            privileged_train,
            splits["train"]["targets"],
            hidden_dim=args.teacher_hidden_dim,
            dropout=args.dropout,
            folds=args.oof_folds,
            device=device,
            args=args,
            seed=seed,
        )
        teacher_metrics = distribution_metrics(
            splits["train"]["targets"], torch.softmax(torch.from_numpy(teacher_logits), dim=1).numpy()
        )

        common = dict(
            train_features=splits["train"]["features"], train_targets=splits["train"]["targets"],
            val_features=splits["val"]["features"], val_targets=splits["val"]["targets"],
            device=device, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, lr=args.lr, weight_decay=args.weight_decay,
        )
        print(f"[{args.dataset}] seed={seed}: baseline + UTD", flush=True)
        baseline = train_model(
            model=DistributionMLP(splits["train"]["features"].shape[1], args.hidden_dim, len(class_names), args.dropout),
            seed=seed + 10, **common,
        )
        utd = train_model(
            model=DistributionMLP(splits["train"]["features"].shape[1], args.hidden_dim, len(class_names), args.dropout),
            seed=seed + 20, teacher_logits=teacher_logits, entropy_gate=gate,
            temperature=args.temperature, kd_weight=args.kd_weight, **common,
        )

        base_logits = {}
        for name in splits:
            base_logits[name] = predict_logits(
                baseline["model"], splits[name]["features"], device, args.batch_size
            )
        reranked = {
            name: apply_local_reranking(base_logits[name], priors[name], args.slr_topk, args.slr_alpha)
            for name in splits
        }
        residual_common = common | {
            "train_base": reranked["train"], "val_base": reranked["val"]
        }
        print(f"[{args.dataset}] seed={seed}: SLR-C residual + full FDIL", flush=True)
        slrc = train_model(
            model=DistributionMLP(splits["train"]["features"].shape[1], args.hidden_dim, len(class_names), args.dropout),
            seed=seed + 30, **residual_common,
        )
        fdil = train_model(
            model=DistributionMLP(splits["train"]["features"].shape[1], args.hidden_dim, len(class_names), args.dropout),
            seed=seed + 40, teacher_logits=teacher_logits, entropy_gate=gate,
            temperature=args.temperature, kd_weight=args.kd_weight, **residual_common,
        )
        results = {"baseline": baseline, "utd": utd, "slrc": slrc, "fdil": fdil}
        predictions: Dict[str, np.ndarray] = {}
        for method, result in results.items():
            test_base = reranked["test"] if method in {"slrc", "fdil"} else None
            val_base = reranked["val"] if method in {"slrc", "fdil"} else None
            val_metrics, _ = evaluate_model(
                result, splits["val"]["features"], splits["val"]["targets"], device, args.batch_size, val_base
            )
            test_metrics, test_probs = evaluate_model(
                result, splits["test"]["features"], splits["test"]["targets"], device, args.batch_size, test_base
            )
            predictions[method] = test_probs
            metrics_by_method[method].append(test_metrics)
            row: Dict[str, Any] = {
                "dataset": args.dataset, "seed": seed, "method": method,
                "best_epoch": result["best_epoch"], "val_kl": val_metrics["kl"],
                "teacher_oof_train_kl": teacher_metrics["kl"],
            }
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
            per_seed_rows.append(row)
            torch.save(result["state_dict"], args.output_dir / f"{method}_seed{seed}.pt")
            print(
                f"[{args.dataset}] seed={seed} {method}: val_KL={val_metrics['kl']:.5f} "
                f"test_KL={test_metrics['kl']:.5f} cosine={test_metrics['cosine']:.5f}", flush=True
            )
        np.savez_compressed(
            args.output_dir / f"test_predictions_seed{seed}.npz",
            image_ids=splits["test"]["image_ids"], targets=splits["test"]["targets"], **predictions,
        )

    aggregate = {method: mean_std(rows) for method, rows in metrics_by_method.items()}
    positive_ids, negative_ids = polarity_ids(len(class_names))
    excluded_ids = sorted(set(range(len(class_names))) - set(positive_ids) - set(negative_ids))
    summary = {
        "dataset": args.dataset,
        "protocol": {
            "selection_metric": "validation KL (lower is better)",
            "test_split": "official split 1; never used for model selection",
            "seeds": seeds,
            "slr_topk": args.slr_topk,
            "slr_alpha": args.slr_alpha,
            "slrc_description_file": str(description_file.resolve()),
            "slrc_description_sha256": description_sha256,
            "utd_gate": (
                "1-max(y), where max(y) is sample-level annotator agreement; training only"
                if args.gate_mode == "max_mass"
                else "normalized label-distribution entropy H(y)/log(C), training only"
            ),
            "teacher": teacher_description,
            "reported_metrics": {
                "lower_is_better": ["chebyshev", "clark", "kl", "dvse"],
                "higher_is_better": ["cosine", "spearman", "mu"],
                "mu_unit": "percent",
                "dvse_polarity": {
                    "positive": [class_names[index] for index in positive_ids],
                    "negative": [class_names[index] for index in negative_ids],
                    "excluded": [class_names[index] for index in excluded_ids],
                },
            },
        },
        "aggregate_test": aggregate,
        "per_seed": per_seed_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    aggregate_rows = []
    for method, metric_values in aggregate.items():
        row: Dict[str, Any] = {"dataset": args.dataset, "method": method}
        for metric, values in metric_values.items():
            row[f"{metric}_mean"] = values["mean"]
            row[f"{metric}_std"] = values["std"]
        aggregate_rows.append(row)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
