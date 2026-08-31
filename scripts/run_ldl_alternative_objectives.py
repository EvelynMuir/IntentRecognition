#!/usr/bin/env python3
"""Run matched δ-LDL and LDL-LRR objectives on frozen CLIP features."""

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
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ldl_fdil import (  # noqa: E402
    DistributionMLP,
    distribution_metrics,
    load_cache,
    predict_logits,
    resolve_device,
    set_seed,
)


def delta_ldl_loss(per_sample_kld: torch.Tensor, delta_0: float) -> torch.Tensor:
    """Published δ-LDL objective using 33-node Simpson integration."""
    nodes = torch.linspace(0.0, float(delta_0), 33, dtype=per_sample_kld.dtype, device=per_sample_kld.device)
    weights = torch.ones(33, dtype=per_sample_kld.dtype, device=per_sample_kld.device)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    values = torch.sigmoid(per_sample_kld[:, None] - nodes[None, :]).mean(dim=0)
    return float(delta_0) / 32.0 / 3.0 * torch.sum(weights * values)


def lrr_ranking_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Published LDL-LRR pairwise label-ranking relation loss."""
    target_difference = targets[:, :, None] - targets[:, None, :]
    relation = (target_difference > 0.5).to(targets.dtype)
    relation_weight = torch.square(target_difference)
    prediction_difference = predictions[:, :, None] - predictions[:, None, :]
    log_likelihood = (
        (1.0 - relation) * F.logsigmoid(1.0 - prediction_difference)
        + relation * F.logsigmoid(prediction_difference)
    ) * relation_weight
    return -torch.sum(log_likelihood) / (2.0 * targets.shape[0])


def dpa_regularizers(
    target_ranks: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Published LDL-DPA rank-aware and distribution-variance terms."""
    rank_term = -torch.mean(torch.mean(target_ranks * predictions, dim=1))
    variance_term = torch.mean(
        torch.square(
            torch.var(targets, dim=1, correction=0)
            - torch.var(predictions, dim=1, correction=0)
        )
    )
    return rank_term, variance_term


def train_objective(
    *,
    method: str,
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
    lrr_alpha: float,
    dpa_alpha: float,
    dpa_beta: float,
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
    target_ranks = torch.as_tensor(rankdata(train_targets, axis=1), dtype=torch.float32)
    uniform = np.full_like(train_targets, 1.0 / train_targets.shape[1])
    delta_0 = float(
        np.sum(
            train_targets
            * (np.log(np.clip(train_targets, 1e-12, 1.0)) - np.log(uniform)),
            axis=1,
        ).mean()
    )
    best_state = None
    best_epoch = 0
    best_val_kl = float("inf")
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses, klds, auxiliaries = [], [], []
        for indices in loader:
            x = features[indices].to(device)
            y = targets[indices].to(device)
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            per_sample_kld = torch.sum(
                y
                * (
                    torch.log(torch.clamp(y, min=1e-12))
                    - torch.log(torch.clamp(probabilities, min=1e-12))
                ),
                dim=1,
            )
            if method == "delta_ldl":
                auxiliary = delta_ldl_loss(per_sample_kld, delta_0)
                loss = auxiliary
            elif method == "lrr":
                auxiliary = lrr_ranking_loss(y, probabilities)
                loss = per_sample_kld.mean() + float(lrr_alpha) * auxiliary
            elif method == "dpa":
                rank_term, variance_term = dpa_regularizers(
                    target_ranks[indices].to(device), y, probabilities
                )
                auxiliary = float(dpa_alpha) * rank_term + float(dpa_beta) * variance_term
                loss = per_sample_kld.mean() + auxiliary
            else:
                raise ValueError(f"Unknown objective: {method}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            klds.append(float(per_sample_kld.mean().detach().cpu()))
            auxiliaries.append(float(auxiliary.detach().cpu()))
        val_logits = predict_logits(model, val_features, device, batch_size)
        val_probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        val_metrics = distribution_metrics(val_targets, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_kld": float(np.mean(klds)),
                "train_objective": float(np.mean(auxiliaries)),
                "val_kl": val_metrics["kl"],
            }
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
        raise RuntimeError(f"{method} produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_kl": best_val_kl,
        "delta_0": delta_0,
        "history": history,
    }


def mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flickrldl", "twitterldl", "emotion6"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="delta_ldl,lrr")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lrr-alpha", type=float, default=1e-3)
    parser.add_argument("--dpa-alpha", type=float, default=1e-3)
    parser.add_argument("--dpa-beta", type=float, default=1e-3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]
    device = resolve_device(args.device)
    splits = {
        name: load_cache(args.cache_dir / f"{name}_clip.npz")
        for name in ("train", "val", "test")
    }
    per_seed_rows = []
    metric_rows: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
    for method in methods:
        for seed in seeds:
            result = train_objective(
                method=method,
                train_features=splits["train"]["features"],
                train_targets=splits["train"]["targets"],
                val_features=splits["val"]["features"],
                val_targets=splits["val"]["targets"],
                device=device,
                seed=seed + 10,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                lrr_alpha=args.lrr_alpha,
                dpa_alpha=args.dpa_alpha,
                dpa_beta=args.dpa_beta,
            )
            val_logits = predict_logits(result["model"], splits["val"]["features"], device, args.batch_size)
            test_logits = predict_logits(result["model"], splits["test"]["features"], device, args.batch_size)
            val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
            test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
            val_metrics = distribution_metrics(splits["val"]["targets"], val_probs)
            test_metrics = distribution_metrics(splits["test"]["targets"], test_probs)
            metric_rows[method].append(test_metrics)
            row: dict[str, Any] = {
                "dataset": args.dataset,
                "seed": seed,
                "method": method,
                "best_epoch": result["best_epoch"],
                "val_kl": val_metrics["kl"],
                "delta_0": result["delta_0"],
            }
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
            per_seed_rows.append(row)
            torch.save(result["state_dict"], args.output_dir / f"{method}_seed{seed}.pt")
            np.savez_compressed(
                args.output_dir / f"{method}_test_predictions_seed{seed}.npz",
                image_ids=splits["test"]["image_ids"],
                targets=splits["test"]["targets"],
                predictions=test_probs,
            )
            print(
                f"[{args.dataset}] {method} seed={seed} epoch={result['best_epoch']} "
                f"val_KLD={val_metrics['kl']:.5f} test_KLD={test_metrics['kl']:.5f} "
                f"Spear={test_metrics['spearman']:.5f}",
                flush=True,
            )
    aggregate = {method: mean_std(rows) for method, rows in metric_rows.items()}
    summary = {
        "dataset": args.dataset,
        "methods": methods,
        "protocol": {
            "features": "frozen CLIP ViT-L/14 full-image features",
            "architecture": f"DistributionMLP hidden_dim={args.hidden_dim} (matched baseline)",
            "selection_metric": "validation KLD",
            "test_split": "official split 1",
            "seeds": seeds,
            "delta_ldl": "parameter-free 33-node Simpson integral objective",
            "lrr_alpha": args.lrr_alpha,
            "dpa_alpha": args.dpa_alpha,
            "dpa_beta": args.dpa_beta,
        },
        "aggregate_test": aggregate,
        "per_seed": per_seed_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
