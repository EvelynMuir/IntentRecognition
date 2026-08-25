#!/usr/bin/env python3
"""Aggregate completed 5x5 matched EMOTIC baseline folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

METHODS = ("cocoer", "emotionclip")
METRICS = ("macro", "micro", "samples", "mAP", "hard")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="logs/analysis/emotic_matched_baselines_vitl14")
    p.add_argument("--methods", default=",".join(METHODS), help="Comma-separated methods to aggregate")
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    root = Path(args.output_dir)
    rows, summary = [], {}
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    for method in methods:
        files = sorted((root / method / "folds").glob("seed_*_fold_*.json"))
        if args.require_complete and len(files) != 25:
            raise RuntimeError(f"{method}: expected 25 folds, found {len(files)}")
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        method_summary = {"display_name": records[0]["display_name"] if records else method, "folds": len(records)}
        for metric in METRICS:
            values = np.asarray([r["bundle"]["classwise"]["test"][metric] for r in records], dtype=np.float64)
            if metric != "mAP":
                values *= 100.0
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            ci95 = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
            method_summary[metric] = {"mean": mean, "std": std, "ci95_half_width": ci95}
            rows.append({"method": method, "metric": metric, "n": n, "mean": mean, "std": std, "ci95_half_width": ci95})
        summary[method] = method_summary
    root.mkdir(parents=True, exist_ok=True)
    with (root / "matched_baselines_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "metric", "n", "mean", "std", "ci95_half_width"])
        writer.writeheader()
        writer.writerows(rows)
    (root / "matched_baselines_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for method, record in summary.items():
        print(f"{record['display_name']} ({record['folds']} folds)")
        print("  " + "  ".join(f"{m}={record[m]['mean']:.2f}±{record[m]['ci95_half_width']:.2f}" for m in METRICS[:4]))


if __name__ == "__main__":
    main()
