#!/usr/bin/env python3
"""Aggregate multi-run stability from calibrated decision rule summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


VARIANT_PATHS = {
    "baseline_global": ("baseline", "global", "test"),
    "slr_global": ("slr_v0", "retuned_global", "test"),
    "slr_classwise": ("slr_v0", "classwise", "test"),
    "short_detailed_classwise": ("source_ensemble", "short_plus_detailed", "classwise", "test"),
}

METRIC_KEYS = ["macro", "micro", "samples", "hard", "easy", "medium"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate stability across multiple calibrated decision summaries.")
    parser.add_argument(
        "--summary-jsons",
        type=str,
        required=True,
        help="Comma-separated list of calibrated decision summary.json paths.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write aggregated JSON.",
    )
    return parser.parse_args()


def _parse_path_list(raw: str) -> List[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _extract(result: Dict, path: tuple[str, ...]) -> Dict:
    current = result
    for key in path:
        current = current[key]
    return current


def main() -> None:
    args = _parse_args()
    summary_paths = _parse_path_list(args.summary_jsons)
    results = [json.loads(path.read_text()) for path in summary_paths]

    aggregated = {
        "runs": [str(path) for path in summary_paths],
        "num_runs": len(summary_paths),
        "variants": {},
    }

    for variant_name, path in VARIANT_PATHS.items():
        rows = [_extract(result, path) for result in results]
        variant_summary = {}
        for metric_key in METRIC_KEYS:
            values = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float32)
            variant_summary[metric_key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "values": [float(v) for v in values.tolist()],
            }
        aggregated["variants"][variant_name] = variant_summary

    text = json.dumps(aggregated, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
