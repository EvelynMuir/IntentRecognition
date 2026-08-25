#!/usr/bin/env python3
"""Aggregate Intentonomy VLM zero-shot metrics into a Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Model display name and directory containing metrics.json (repeatable).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_runs(specs: list[str]) -> list[tuple[str, Path, dict[str, Any]]]:
    runs = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run {spec!r}; expected NAME=DIR")
        name, directory = spec.split("=", 1)
        path = Path(directory) / "metrics.json"
        runs.append((name, path.parent, json.loads(path.read_text(encoding="utf-8"))))
    return runs


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run)
    sample_counts = {int(metrics["num_samples"]) for _, _, metrics in runs}
    if sample_counts != {1216}:
        raise RuntimeError(f"Expected 1,216 samples in every run, got {sorted(sample_counts)}")

    lines = [
        "# Intentonomy zero-shot prediction report",
        "",
        "## Protocol",
        "",
        "Official Intentonomy test split (1,216 images, 28 labels); one image per request; "
        "all classes scored from 0 to 1; thinking disabled; 0.5 threshold for F1; mAP "
        "computed from continuous class scores. Qwen3.5 uses temperature=1.0, top_p=0.95, "
        "top_k=20, min_p=0.0, presence_penalty=1.5, and repetition_penalty=1.0. "
        "Gemma 4 E4B IT uses its recommended temperature=1.0, top_p=0.95, and top_k=64. "
        "Step3-VL-10B uses greedy decoding (temperature=0).",
        "",
        "## Results",
        "",
        "| Model | Macro-F1 (%) | Micro-F1 (%) | Samples-F1 (%) | mAP (%) | Errors | Zero-fill repairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, metrics in runs:
        bad = int(metrics.get("request_error_count", 0)) + int(metrics.get("empty_prediction_count", 0))
        repairs = sum(
            int(count)
            for status, count in metrics.get("parse_status_counts", {}).items()
            if "missing_filled_zero" in status
        )
        lines.append(
            f"| {name} | {pct(metrics['macro_f1'])} | {pct(metrics['micro_f1'])} | "
            f"{pct(metrics['samples_f1'])} | {float(metrics['mAP']):.2f} | {bad} | {repairs} |"
        )

    lines += ["", "## Per-model class analysis", ""]
    for name, directory, metrics in runs:
        per_class = list(metrics["per_class"])
        best = sorted(per_class, key=lambda row: row["f1"], reverse=True)[:5]
        worst = sorted(per_class, key=lambda row: row["f1"])[:5]
        lines += [
            f"### {name}",
            "",
            "- Strongest F1 classes: "
            + ", ".join(f"{r['class_name']} ({pct(r['f1'])})" for r in best),
            "- Weakest F1 classes: "
            + ", ".join(f"{r['class_name']} ({pct(r['f1'])})" for r in worst),
            f"- Artifacts: `{directory / 'metrics.json'}`, `{directory / 'predictions.jsonl'}`",
            "",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
