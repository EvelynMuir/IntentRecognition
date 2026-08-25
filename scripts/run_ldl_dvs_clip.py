#!/usr/bin/env python3
"""Run LDL-DVS on the same frozen CLIP features and protocol as LDL-FDIL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ldl_fdil import (
    DistributionMLP,
    distribution_metrics,
    load_cache,
    predict_logits,
    polarity_ids,
    resolve_device,
    set_seed,
)


def smooth_min(left: torch.Tensor, right: torch.Tensor, k: float) -> torch.Tensor:
    left_weight = torch.exp(-float(k) * left)
    right_weight = torch.exp(-float(k) * right)
    return (left * left_weight + right * right_weight) / (left_weight + right_weight)


def pairwise_divisiveness_loss(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    k: float = 10.0,
    eps: float = float(np.finfo(np.float32).eps),
) -> torch.Tensor:
    """Equation (6): fully enumerated pairwise divisiveness surrogate.

    The paper averages over all C(C-1) ordered label pairs. Pairs outside the
    positive×negative polarity support contribute zero; retaining the original
    denominator is important for matching its loss scale.
    """
    classes = int(targets.shape[1])
    positive_ids, negative_ids = polarity_ids(classes)
    per_sample = torch.zeros(targets.shape[0], dtype=targets.dtype, device=targets.device)
    for positive_id in positive_ids:
        for negative_id in negative_ids:
            target_psi = smooth_min(targets[:, positive_id], targets[:, negative_id], k)
            prediction_psi = smooth_min(predictions[:, positive_id], predictions[:, negative_id], k)
            difference = target_psi - prediction_psi
            per_sample += torch.sqrt(difference * difference + float(eps) ** 2)
    return per_sample / float(classes * (classes - 1))


def train_ldl_dvs(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    k: float,
    alpha: float,
) -> Dict[str, Any]:
    model = DistributionMLP(train_features.shape[1], hidden_dim, train_targets.shape[1], dropout)
    set_seed(seed)
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.arange(len(train_features)), batch_size=batch_size, shuffle=True, generator=generator
    )
    features = torch.as_tensor(train_features, dtype=torch.float32)
    targets = torch.as_tensor(train_targets, dtype=torch.float32)
    best_state = None
    best_epoch = 0
    best_val_kl = float("inf")
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_kld, epoch_dvs = [], []
        for indices in loader:
            x = features[indices].to(device)
            y = targets[indices].to(device)
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            kld = torch.sum(
                y * (torch.log(torch.clamp(y, min=1e-12)) - torch.log(torch.clamp(probabilities, min=1e-12))),
                dim=1,
            )
            dvs = pairwise_divisiveness_loss(y, probabilities, k=k)
            loss = kld.mean() + float(alpha) * dvs.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_kld.append(float(kld.mean().detach().cpu()))
            epoch_dvs.append(float(dvs.mean().detach().cpu()))
        val_logits = predict_logits(model, val_features, device, batch_size)
        val_probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        val_metrics = distribution_metrics(val_targets, val_probabilities)
        history.append(
            {"epoch": epoch, "train_kld": float(np.mean(epoch_kld)),
             "train_pairwise_dvs": float(np.mean(epoch_dvs)), "val_kl": val_metrics["kl"]}
        )
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
        raise RuntimeError("LDL-DVS produced no checkpoint")
    model.load_state_dict(best_state)
    return {"model": model, "state_dict": best_state, "best_epoch": best_epoch,
            "best_val_kl": best_val_kl, "history": history}


def mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "std": float(np.std([row[key] for row in rows]))}
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flickrldl", "twitterldl", "emotion6"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seeds = [int(value) for value in args.seeds.split(",")]
    splits = {name: load_cache(args.cache_dir / f"{name}_clip.npz") for name in ("train", "val", "test")}
    rows = []
    metric_rows = []
    for seed in seeds:
        result = train_ldl_dvs(
            train_features=splits["train"]["features"], train_targets=splits["train"]["targets"],
            val_features=splits["val"]["features"], val_targets=splits["val"]["targets"],
            device=device, seed=seed + 10, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, lr=args.lr, weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim, dropout=args.dropout, k=args.k, alpha=args.alpha,
        )
        val_logits = predict_logits(result["model"], splits["val"]["features"], device, args.batch_size)
        test_logits = predict_logits(result["model"], splits["test"]["features"], device, args.batch_size)
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        val_metrics = distribution_metrics(splits["val"]["targets"], val_probs)
        test_metrics = distribution_metrics(splits["test"]["targets"], test_probs)
        metric_rows.append(test_metrics)
        row: dict[str, Any] = {"dataset": args.dataset, "seed": seed, "method": "ldl_dvs",
                               "best_epoch": result["best_epoch"], "val_kl": val_metrics["kl"]}
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        rows.append(row)
        torch.save(result["state_dict"], args.output_dir / f"ldl_dvs_seed{seed}.pt")
        np.savez_compressed(
            args.output_dir / f"test_predictions_seed{seed}.npz",
            image_ids=splits["test"]["image_ids"], targets=splits["test"]["targets"], predictions=test_probs,
        )
        print(f"[{args.dataset}] seed={seed} epoch={result['best_epoch']} val_KLD={val_metrics['kl']:.5f} "
              f"test_KLD={test_metrics['kl']:.5f} DVSE={test_metrics['dvse']:.5f}", flush=True)
    aggregate = mean_std(metric_rows)
    summary = {
        "dataset": args.dataset,
        "method": "LDL-DVS",
        "protocol": {
            "features": "frozen CLIP ViT-L/14 full-image features (matched to FDIL)",
            "architecture": f"DistributionMLP hidden_dim={args.hidden_dim} (matched to baseline)",
            "selection_metric": "validation KLD",
            "test_split": "official split 1",
            "seeds": seeds,
            "k": args.k,
            "alpha": args.alpha,
            "loss": "KLD + alpha * fully-enumerated pairwise divisiveness surrogate",
        },
        "aggregate_test": aggregate,
        "per_seed": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
