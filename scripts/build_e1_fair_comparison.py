#!/usr/bin/env python3
"""Build the E1 fair-comparison artifacts for the FDIL revision.

This script intentionally aggregates existing validation-selected experiment
summaries. It does not retrain models; E1 is a protocol audit/comparison table
over runs that already saved both global and class-wise validation thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_UTD = (
    PROJECT_ROOT / "logs" / "analysis" / "privileged_distillation_text_teacher_seedfix_20260316" / "summary.json"
)
DEFAULT_SCENARIO_FDIL = PROJECT_ROOT / "logs" / "analysis" / "distillation_slrc_20260317" / "summary.json"
DEFAULT_FINAL_FDIL = PROJECT_ROOT / "logs" / "analysis" / "distillation_slrc_lcs_topk5_20260327" / "summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate E1 fair-comparison tables.")
    parser.add_argument("--baseline-utd-summary", type=Path, default=DEFAULT_BASELINE_UTD)
    parser.add_argument("--scenario-fdil-summary", type=Path, default=DEFAULT_SCENARIO_FDIL)
    parser.add_argument("--final-fdil-summary", type=Path, default=DEFAULT_FINAL_FDIL)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_percent(metrics: Mapping[str, Any], key: str) -> float:
    value = float(metrics[key])
    if key == "mAP":
        return value
    return value * 100.0


def _avg_f1(row: Mapping[str, float]) -> float:
    return (float(row["macro"]) + float(row["micro"]) + float(row["samples"])) / 3.0


def _row(
    *,
    method: str,
    threshold: str,
    split_metrics: Mapping[str, Any],
    source_run: str,
    note: str,
) -> dict[str, Any]:
    row = {
        "method": method,
        "threshold": threshold,
        "macro": _metric_percent(split_metrics, "macro"),
        "micro": _metric_percent(split_metrics, "micro"),
        "samples": _metric_percent(split_metrics, "samples"),
        "avg_f1": 0.0,
        "mAP": _metric_percent(split_metrics, "mAP"),
        "hard": _metric_percent(split_metrics, "hard"),
        "selected_threshold": float(split_metrics["threshold"]),
        "source_run": source_run,
        "note": note,
    }
    row["avg_f1"] = _avg_f1(row)
    return row


def _bundle_rows(
    *,
    method: str,
    bundle: Mapping[str, Any],
    source_run: str,
    note: str,
) -> list[dict[str, Any]]:
    return [
        _row(
            method=method,
            threshold="global",
            split_metrics=bundle["global"]["test"],
            source_run=source_run,
            note=note,
        ),
        _row(
            method=method,
            threshold="classwise",
            split_metrics=bundle["classwise"]["test"],
            source_run=source_run,
            note=note,
        ),
    ]


def _round_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for key in ["macro", "micro", "samples", "avg_f1", "mAP", "hard", "selected_threshold"]:
        output[key] = round(float(output[key]), 4)
    return output


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = [
        "method",
        "threshold",
        "macro",
        "micro",
        "samples",
        "avg_f1",
        "mAP",
        "hard",
        "selected_threshold",
        "source_run",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_round_row(row))


def _markdown_table(rows: list[Mapping[str, Any]]) -> str:
    headers = ["Method", "Threshold", "Macro", "Micro", "Samples", "AvgF1", "mAP", "Hard"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {threshold} | {macro:.2f} | {micro:.2f} | {samples:.2f} | "
            "{avg_f1:.2f} | {mAP:.2f} | {hard:.2f} |".format(**row)
        )
    return "\n".join(lines)


def _method_row(rows: list[Mapping[str, Any]], method: str, threshold: str) -> Mapping[str, Any]:
    for row in rows:
        if row["method"] == method and row["threshold"] == threshold:
            return row
    raise KeyError(f"No row for {method}/{threshold}")


def main() -> None:
    args = _parse_args()
    baseline_utd = _load_json(args.baseline_utd_summary)
    scenario_fdil = _load_json(args.scenario_fdil_summary)
    final_fdil = _load_json(args.final_fdil_summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "logs" / "analysis" / f"e1_fair_comparison_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(
        _bundle_rows(
            method="CLIP baseline",
            bundle=baseline_utd["baseline"]["bundle"],
            source_run=str(args.baseline_utd_summary.relative_to(PROJECT_ROOT)),
            note="Frozen CLIP ViT-L/14 feature baseline; validation-selected thresholds.",
        )
    )
    rows.extend(
        _bundle_rows(
            method="UTD only",
            bundle=baseline_utd["dynamic_gated_kd"]["bundle"],
            source_run=str(args.baseline_utd_summary.relative_to(PROJECT_ROOT)),
            note="Text-teacher dynamic gated KD without SLR-C prior.",
        )
    )
    scenario_fixed = scenario_fdil["slr_c_fixed"]
    rows.extend(
        _bundle_rows(
            method="SLR-C scenario fixed",
            bundle=scenario_fixed,
            source_run=str(args.scenario_fdil_summary.relative_to(PROJECT_ROOT)),
            note="Scenario-only fixed SLR-C prior, no residual student.",
        )
    )
    rows.extend(
        _bundle_rows(
            method="FDIL scenario",
            bundle=scenario_fdil["slr_c_residual_dynamic_kd"]["bundle"],
            source_run=str(args.scenario_fdil_summary.relative_to(PROJECT_ROOT)),
            note="Scenario SLR-C plus residual dynamic KD.",
        )
    )
    rows.extend(
        _bundle_rows(
            method="FDIL final LCS K=5",
            bundle=final_fdil["slr_c_residual_dynamic_kd"]["bundle"],
            source_run=str(args.final_fdil_summary.relative_to(PROJECT_ROOT)),
            note="Final lexical+canonical+scenario FDIL, K=5.",
        )
    )

    core_rows = [
        _method_row(rows, "CLIP baseline", "global"),
        _method_row(rows, "CLIP baseline", "classwise"),
        _method_row(rows, "FDIL final LCS K=5", "global"),
        _method_row(rows, "FDIL final LCS K=5", "classwise"),
    ]

    baseline_global = _method_row(rows, "CLIP baseline", "global")
    baseline_classwise = _method_row(rows, "CLIP baseline", "classwise")
    fdil_global = _method_row(rows, "FDIL final LCS K=5", "global")
    fdil_classwise = _method_row(rows, "FDIL final LCS K=5", "classwise")

    decomposition = [
        {
            "comparison": "CLIP baseline: classwise - global",
            "macro_delta": baseline_classwise["macro"] - baseline_global["macro"],
            "avg_f1_delta": baseline_classwise["avg_f1"] - baseline_global["avg_f1"],
            "hard_delta": baseline_classwise["hard"] - baseline_global["hard"],
            "interpretation": "threshold effect on the visual baseline",
        },
        {
            "comparison": "FDIL final: classwise - global",
            "macro_delta": fdil_classwise["macro"] - fdil_global["macro"],
            "avg_f1_delta": fdil_classwise["avg_f1"] - fdil_global["avg_f1"],
            "hard_delta": fdil_classwise["hard"] - fdil_global["hard"],
            "interpretation": "threshold effect on the final method",
        },
        {
            "comparison": "FDIL final - CLIP baseline under global threshold",
            "macro_delta": fdil_global["macro"] - baseline_global["macro"],
            "avg_f1_delta": fdil_global["avg_f1"] - baseline_global["avg_f1"],
            "hard_delta": fdil_global["hard"] - baseline_global["hard"],
            "interpretation": "method effect when both use validation-tuned global thresholds",
        },
        {
            "comparison": "FDIL final - CLIP baseline under classwise threshold",
            "macro_delta": fdil_classwise["macro"] - baseline_classwise["macro"],
            "avg_f1_delta": fdil_classwise["avg_f1"] - baseline_classwise["avg_f1"],
            "hard_delta": fdil_classwise["hard"] - baseline_classwise["hard"],
            "interpretation": "method effect when both use validation-tuned class-wise thresholds",
        },
    ]

    _write_csv(output_dir / "e1_core_four_grid.csv", core_rows)
    _write_csv(output_dir / "e1_inhouse_controlled_baselines.csv", rows)

    with (output_dir / "e1_threshold_decomposition.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["comparison", "macro_delta", "avg_f1_delta", "hard_delta", "interpretation"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in decomposition:
            rounded = dict(item)
            for key in ["macro_delta", "avg_f1_delta", "hard_delta"]:
                rounded[key] = round(float(rounded[key]), 4)
            writer.writerow(rounded)

    summary = {
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "inputs": {
            "baseline_utd_summary": str(args.baseline_utd_summary.relative_to(PROJECT_ROOT)),
            "scenario_fdil_summary": str(args.scenario_fdil_summary.relative_to(PROJECT_ROOT)),
            "final_fdil_summary": str(args.final_fdil_summary.relative_to(PROJECT_ROOT)),
        },
        "status": {
            "E1a": "complete",
            "E1b": "not complete; external SOTA code/backbone swaps were not rerun in this pass",
        },
        "core_four_grid": [_round_row(row) for row in core_rows],
        "inhouse_controlled_baselines": [_round_row(row) for row in rows],
        "threshold_decomposition": [
            {
                **item,
                "macro_delta": round(float(item["macro_delta"]), 4),
                "avg_f1_delta": round(float(item["avg_f1_delta"]), 4),
                "hard_delta": round(float(item["hard_delta"]), 4),
            }
            for item in decomposition
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# E1 Fair Comparison Grid

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

- E1a is complete for controllable in-house methods: all rows below use CLIP ViT-L/14 cached features, the same train/val/test split, and validation-only threshold selection.
- E1b is not complete in this pass: HLEG/LabCR/PIP-Net/IntCLIP were not rerun with swapped CLIP ViT-L/14 features. The response letter should state this protocol limitation if those rows remain from the original papers.
- `AvgF1` is the mean of Macro/Micro/Samples F1. This is the source of the paper's `53.41` CLIP-baseline number; it is not mAP.

## Core Four-Grid

{_markdown_table(core_rows)}

## Threshold/Method Decomposition

| Comparison | Macro Delta | AvgF1 Delta | Hard Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
"""
    for item in decomposition:
        report += (
            f"| {item['comparison']} | {item['macro_delta']:.2f} | {item['avg_f1_delta']:.2f} | "
            f"{item['hard_delta']:.2f} | {item['interpretation']} |\n"
        )
    report += f"""
## In-House Controlled Baselines

{_markdown_table(rows)}

## Immediate Wording Implications

- `CLIP baseline 53.41` should be described as the class-wise-threshold AvgF1 of the frozen CLIP ViT-L/14 feature baseline.
- Under the same class-wise-threshold protocol, final FDIL improves over CLIP baseline by {fdil_classwise['macro'] - baseline_classwise['macro']:.2f} Macro F1, {fdil_classwise['avg_f1'] - baseline_classwise['avg_f1']:.2f} AvgF1, and {fdil_classwise['hard'] - baseline_classwise['hard']:.2f} Hard F1.
- Under the stricter global-threshold comparison, final FDIL has almost identical Macro F1 but still improves AvgF1 by {fdil_global['avg_f1'] - baseline_global['avg_f1']:.2f}; Hard F1 is essentially unchanged. The revised claim should therefore separate method gains from calibration gains.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
