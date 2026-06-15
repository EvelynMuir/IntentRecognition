#!/usr/bin/env python3
"""Aggregate E2 multi-seed stability for FDIL revision experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["macro", "micro", "samples", "avg_f1", "mAP", "hard"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mean/std tables for E2 multi-seed runs.")
    parser.add_argument(
        "--privileged-summaries",
        type=str,
        required=True,
        help="Comma-separated analyze_privileged_distillation summary.json files.",
    )
    parser.add_argument(
        "--fdil-summaries",
        type=str,
        required=True,
        help="Comma-separated analyze_distillation_slrc summary.json files for final FDIL.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _paths(raw: str) -> list[Path]:
    paths = [Path(item.strip()) for item in raw.split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one summary path is required.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(split_metrics: Mapping[str, Any]) -> dict[str, float]:
    macro = float(split_metrics["macro"]) * 100.0
    micro = float(split_metrics["micro"]) * 100.0
    samples = float(split_metrics["samples"]) * 100.0
    return {
        "macro": macro,
        "micro": micro,
        "samples": samples,
        "avg_f1": (macro + micro + samples) / 3.0,
        "mAP": float(split_metrics["mAP"]),
        "hard": float(split_metrics["hard"]) * 100.0,
    }


def _extract_seed(summary: Mapping[str, Any], source_path: Path, fallback: int) -> int:
    config = summary.get("config", {})
    for key in ["base_seed", "dynamic_kd_seed"]:
        if key in config:
            value = int(config[key])
            if key == "dynamic_kd_seed":
                return value - 300
            return value
    match = re.search(r"seed(\d+)", str(source_path))
    if match:
        return int(match.group(1))
    if source_path.name == "summary.json" and "distillation_slrc_lcs_topk5_20260327" in str(source_path):
        return 20260317
    return fallback


def _row(
    *,
    method: str,
    threshold: str,
    seed: int,
    split_metrics: Mapping[str, Any],
    source_run: Path,
) -> dict[str, Any]:
    row = {
        "method": method,
        "threshold": threshold,
        "seed": seed,
        "source_run": str(source_run.relative_to(PROJECT_ROOT) if source_run.is_absolute() else source_run),
    }
    row.update(_metrics(split_metrics))
    return row


def _bundle_rows(
    *,
    method: str,
    seed: int,
    bundle: Mapping[str, Any],
    source_run: Path,
) -> list[dict[str, Any]]:
    return [
        _row(
            method=method,
            threshold="global",
            seed=seed,
            split_metrics=bundle["global"]["test"],
            source_run=source_run,
        ),
        _row(
            method=method,
            threshold="classwise",
            seed=seed,
            split_metrics=bundle["classwise"]["test"],
            source_run=source_run,
        ),
    ]


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round(row.get(key, "")) for key in fieldnames})


def _aggregate(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["threshold"])), []).append(row)

    output = []
    for (method, threshold), group_rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "method": method,
            "threshold": threshold,
            "num_seeds": len(group_rows),
            "seeds": ",".join(str(row["seed"]) for row in group_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float32)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
            summary[f"{metric}_values"] = ",".join(f"{float(value):.4f}" for value in values.tolist())
        output.append(summary)
    return output


def _markdown_table(rows: list[Mapping[str, Any]]) -> str:
    headers = ["Method", "Threshold", "N", "Macro", "Micro", "Samples", "AvgF1", "mAP", "Hard"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {threshold} | {num_seeds} | {macro_mean:.2f}±{macro_std:.2f} | "
            "{micro_mean:.2f}±{micro_std:.2f} | {samples_mean:.2f}±{samples_std:.2f} | "
            "{avg_f1_mean:.2f}±{avg_f1_std:.2f} | {mAP_mean:.2f}±{mAP_std:.2f} | "
            "{hard_mean:.2f}±{hard_std:.2f} |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    privileged_paths = _paths(args.privileged_summaries)
    fdil_paths = _paths(args.fdil_summaries)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "logs" / "analysis" / f"e2_multiseed_stability_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(privileged_paths):
        summary = _load(path)
        seed = _extract_seed(summary, path, fallback=idx)
        rows.extend(
            _bundle_rows(
                method="CLIP baseline",
                seed=seed,
                bundle=summary["baseline"]["bundle"],
                source_run=path,
            )
        )
        rows.extend(
            _bundle_rows(
                method="UTD only",
                seed=seed,
                bundle=summary["dynamic_gated_kd"]["bundle"],
                source_run=path,
            )
        )

    for idx, path in enumerate(fdil_paths):
        summary = _load(path)
        seed = _extract_seed(summary, path, fallback=idx)
        rows.extend(
            _bundle_rows(
                method="FDIL final LCS K=5",
                seed=seed,
                bundle=summary["slr_c_residual_dynamic_kd"]["bundle"],
                source_run=path,
            )
        )

    aggregate_rows = _aggregate(rows)

    raw_fields = ["method", "threshold", "seed", *METRICS, "source_run"]
    aggregate_fields = ["method", "threshold", "num_seeds", "seeds"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_values"])

    _write_csv(output_dir / "e2_seed_level_metrics.csv", rows, raw_fields)
    _write_csv(output_dir / "e2_mean_std.csv", aggregate_rows, aggregate_fields)

    summary = {
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "privileged_summaries": [
            str(path.relative_to(PROJECT_ROOT) if path.is_absolute() else path) for path in privileged_paths
        ],
        "fdil_summaries": [str(path.relative_to(PROJECT_ROOT) if path.is_absolute() else path) for path in fdil_paths],
        "seed_level_metrics": [{key: _round(value) for key, value in row.items()} for row in rows],
        "mean_std": [{key: _round(value) for key, value in row.items()} for row in aggregate_rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# E2 Multi-Seed Stability

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

- Aggregates cached-feature training runs for the revision-critical methods: CLIP baseline, UTD only, and final FDIL (`lexical_canonical_scenario`, `K=5`).
- Metrics are test-split percentages. `AvgF1` is the mean of Macro/Micro/Samples F1. Standard deviation uses population std over the completed seeds.
- Thresholds are selected on validation data inside each seed run, separately for global and class-wise settings.

## Mean ± Std

{_markdown_table(aggregate_rows)}
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
