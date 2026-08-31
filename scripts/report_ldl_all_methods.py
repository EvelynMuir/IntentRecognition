#!/usr/bin/env python3
"""Build unified five-seed report for all LDL methods and datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("chebyshev", "Cheby. ↓", 4, "min"),
    ("clark", "Clark ↓", 4, "min"),
    ("kl", "KLD ↓", 4, "min"),
    ("cosine", "Cosine ↑", 4, "max"),
    ("spearman", "Spear. ↑", 4, "max"),
    ("mu", "µ (%) ↑", 2, "max"),
    ("dvse", "DVSE ↓", 5, "min"),
]


def load_dataset(root: Path, dataset: str) -> dict[str, Any]:
    fdil = json.loads((root / f"{dataset}_fdil_20260821" / "summary.json").read_text())
    dvs = json.loads((root / f"{dataset}_ldl_dvs_clip_20260821" / "summary.json").read_text())
    alternatives = json.loads((root / f"{dataset}_delta_lrr_clip_20260821" / "summary.json").read_text())
    dpa = json.loads((root / f"{dataset}_dpa_clip_20260821" / "summary.json").read_text())
    for payload in (fdil, dvs, alternatives, dpa):
        assert payload["protocol"]["seeds"] == [2026, 2027, 2028, 2029, 2030]
    return {
        "baseline": fdil["aggregate_test"]["baseline"],
        "utd": fdil["aggregate_test"]["utd"],
        "slrc": fdil["aggregate_test"]["slrc"],
        "fdil": fdil["aggregate_test"]["fdil"],
        "ldl_dvs": dvs["aggregate_test"],
        "delta_ldl": alternatives["aggregate_test"]["delta_ldl"],
        "lrr": alternatives["aggregate_test"]["lrr"],
        "dpa": dpa["aggregate_test"]["dpa"],
    }


METHODS = [
    ("baseline", "CLIP baseline"),
    ("delta_ldl", "δ-LDL"),
    ("lrr", "LDL-LRR"),
    ("dpa", "LDL-DPA"),
    ("ldl_dvs", "LDL-DVS"),
    ("utd", "UTD only"),
    ("slrc", "SLR-C only"),
    ("fdil", "Full FDIL"),
]


def make_table(data: dict[str, Any]) -> str:
    best = {}
    for metric, _, _, direction in METRICS:
        values = {key: data[key][metric]["mean"] for key, _ in METHODS}
        best[metric] = min(values, key=values.get) if direction == "min" else max(values, key=values.get)
    headers = ["Method"] + [label for _, label, _, _ in METRICS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for key, label in METHODS:
        cells = [label]
        for metric, _, digits, _ in METRICS:
            values = data[key][metric]
            text = f"{values['mean']:.{digits}f} ± {values['std']:.{digits}f}"
            cells.append(f"**{text}**" if best[metric] == key else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def improvements(data: dict[str, Any], method: str) -> str:
    baseline = data["baseline"]
    improved = []
    for metric, label, _, direction in METRICS:
        current = data[method][metric]["mean"]
        reference = baseline[metric]["mean"]
        if (direction == "min" and current < reference) or (direction == "max" and current > reference):
            improved.append(label.split()[0])
    return f"{len(improved)}/7 ({', '.join(improved) if improved else 'none'})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("logs/analysis"))
    parser.add_argument("--output", type=Path, default=Path("docs/record/record_0821_AllLDL_5Seeds.md"))
    args = parser.parse_args()
    datasets = {name: load_dataset(args.logs_root, name) for name in ("flickrldl", "twitterldl", "emotion6")}
    sections = []
    for dataset, title in (("flickrldl", "Flickr-LDL"), ("twitterldl", "Twitter-LDL"), ("emotion6", "Emotion6")):
        sections.append(
            f"## {title}\n\n{make_table(datasets[dataset])}\n\n"
            f"Improvements over matched baseline: δ-LDL {improvements(datasets[dataset], 'delta_ldl')}; "
            f"LDL-LRR {improvements(datasets[dataset], 'lrr')}; LDL-DPA {improvements(datasets[dataset], 'dpa')}; "
            f"LDL-DVS {improvements(datasets[dataset], 'ldl_dvs')}; "
            f"Full FDIL {improvements(datasets[dataset], 'fdil')}."
        )
    text = """# All LDL methods: unified five-seed results

## Protocol

- Seeds: 2026, 2027, 2028, 2029, 2030.
- Frozen CLIP ViT-L/14 768-D full-image features and matched 768-hidden MLP.
- Identical dataset splits, AdamW settings, and validation-KLD model selection.
- Values are mean ± population standard deviation over five seeds.
- Test data is used only for final evaluation.

""" + "\n\n".join(sections) + """

## Summary

- Flickr: UTD/FDIL dominate Cheby, Spearman, and DVSE; SLR-C is strongest on KLD, cosine, and µ.
- Twitter: δ-LDL is strongest on global distance/similarity metrics; UTD is strongest on Cheby and DVSE; SLR-C is strongest on Spearman.
- Emotion6: full FDIL is strongest on Clark and KLD; SLR-C is strongest on cosine, µ, and DVSE; baseline/LDL-LRR are effectively tied on Cheby.
- Default LDL-LRR, LDL-DPA, and LDL-DVS remain almost indistinguishable from baseline across all three datasets.
- LDL-DPA's default weighted regularizer is only 0.32%-0.48% of train KLD magnitude, explaining its negligible effect with CLIP features.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
