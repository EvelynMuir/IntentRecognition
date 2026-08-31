#!/usr/bin/env python3
"""Build the two-dataset LDL-FDIL result report from run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("chebyshev", "Cheby. ↓", 4),
    ("clark", "Clark ↓", 4),
    ("kl", "KLD ↓", 4),
    ("cosine", "Cosine ↑", 4),
    ("spearman", "Spear. ↑", 4),
    ("mu", "µ (%) ↑", 2),
    ("dvse", "DVSE ↓", 4),
]
METHODS = [("baseline", "CLIP baseline"), ("utd", "UTD only"), ("slrc", "SLR-C only"), ("fdil", "FDIL")]


def format_metric(values: dict[str, float], digits: int) -> str:
    return f"{values['mean']:.{digits}f} ± {values['std']:.{digits}f}"


def table_for(summary: dict[str, Any]) -> str:
    headers = ["Method"] + [title for _, title, _ in METRICS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    aggregate = summary["aggregate_test"]
    best_method = {}
    for metric, _, _ in METRICS:
        values = {method: aggregate[method][metric]["mean"] for method, _ in METHODS}
        best_method[metric] = (
            min(values, key=values.get) if metric in {"chebyshev", "clark", "kl", "dvse"}
            else max(values, key=values.get)
        )
    for key, label in METHODS:
        cells = [label]
        for metric, _, digits in METRICS:
            value = format_metric(aggregate[key][metric], digits)
            cells.append(f"**{value}**" if key == best_method[metric] else value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fdil_delta_text(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate_test"]
    baseline = aggregate["baseline"]
    fdil = aggregate["fdil"]
    pieces = []
    for metric, title, digits in METRICS:
        delta = fdil[metric]["mean"] - baseline[metric]["mean"]
        pieces.append(f"{title.split()[0]} {delta:+.{digits}f}")
    improved = sum(
        (fdil[m]["mean"] < baseline[m]["mean"]) if m in {"chebyshev", "clark", "kl", "dvse"}
        else (fdil[m]["mean"] > baseline[m]["mean"])
        for m, _, _ in METRICS
    )
    return f"Full FDIL improves {improved}/7 metrics over the matched baseline; deltas (FDIL-baseline): " + ", ".join(pieces) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flickr-summary", type=Path, required=True)
    parser.add_argument("--twitter-summary", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    flickr = json.loads(args.flickr_summary.read_text(encoding="utf-8"))
    twitter = json.loads(args.twitter_summary.read_text(encoding="utf-8"))
    for expected, summary in (("flickrldl", flickr), ("twitterldl", twitter)):
        if summary.get("dataset") != expected:
            raise ValueError(f"Expected {expected}, got {summary.get('dataset')}")
        missing = [metric for metric, _, _ in METRICS if metric not in summary["aggregate_test"]["fdil"]]
        if missing:
            raise ValueError(f"{expected} summary misses metrics: {missing}")

    text = f"""# FDIL on Flickr-LDL and Twitter-LDL

## Protocol

- Frozen CLIP ViT-L/14 image features.
- Official split 1 is test-only; official fold 2 from the remaining data is validation.
- Best epochs are selected exclusively by validation KLD.
- Results are mean ± population standard deviation over seeds {flickr['protocol']['seeds']}.
- SLR-C uses the shared Gemini lexical/canonical/6-scenario prior, Top-{flickr['protocol']['slr_topk']}, alpha={flickr['protocol']['slr_alpha']}.
- UTD uses a training-only Qwen3-VL rationale teacher encoded by BGE-large and cross-fitted into OOF train predictions.
- `µ` is reported as a percentage. DVSE uses positive={{amusement, contentment, excitement}}, negative={{anger, disgust, fear, sadness}}, with awe excluded.

## Flickr-LDL

{table_for(flickr)}

## Twitter-LDL

{table_for(twitter)}

## Result interpretation

- Flickr-LDL: {fdil_delta_text(flickr)} SLR-C is strongest on Clark, KLD, cosine and µ, while full FDIL is strongest on Spearman and DVSE.
- Twitter-LDL: {fdil_delta_text(twitter)} UTD-only is strongest on Cheby and DVSE; full FDIL is strongest only on Spearman. The transfer therefore does not establish uniform dominance on Twitter-LDL.

## Artifact provenance

- Flickr summary: `{args.flickr_summary.resolve()}`
- Twitter summary: `{args.twitter_summary.resolve()}`
- Shared SLR-C prior SHA-256: `{flickr['protocol']['slrc_description_sha256']}`
- Test predictions and per-seed checkpoints are stored next to each summary.
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(text, encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
