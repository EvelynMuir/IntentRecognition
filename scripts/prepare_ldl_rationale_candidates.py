#!/usr/bin/env python3
"""Train a validation-selected LDL baseline and export train confusion scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ldl_fdil import (
    DistributionMLP,
    distribution_metrics,
    load_cache,
    predict_logits,
    resolve_device,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    splits = {name: load_cache(args.cache_dir / f"{name}_clip.npz") for name in ("train", "val")}
    device = resolve_device(args.device)
    model = DistributionMLP(splits["train"]["features"].shape[1], 512, splits["train"]["targets"].shape[1], 0.1)
    result = train_model(
        model=model,
        train_features=splits["train"]["features"],
        train_targets=splits["train"]["targets"],
        val_features=splits["val"]["features"],
        val_targets=splits["val"]["targets"],
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=5e-4,
        weight_decay=1e-4,
        seed=args.seed,
    )
    train_logits = predict_logits(result["model"], splits["train"]["features"], device, args.batch_size)
    val_logits = predict_logits(result["model"], splits["val"]["features"], device, args.batch_size)
    train_probs = torch.softmax(torch.from_numpy(train_logits), dim=1).numpy()
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        image_ids=splits["train"]["image_ids"],
        predictions=train_probs,
        logits=train_logits,
        targets=splits["train"]["targets"],
    )
    summary = {
        "seed": args.seed,
        "best_epoch": result["best_epoch"],
        "validation": distribution_metrics(splits["val"]["targets"], val_probs),
        "train": distribution_metrics(splits["train"]["targets"], train_probs),
    }
    args.output_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
