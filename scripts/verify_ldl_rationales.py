#!/usr/bin/env python3
"""Verify completeness and structure of an LDL rationale JSONL artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flickrldl", "twitterldl", "emotion6"], required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or root.parent / "LDL" / "processed" / args.dataset
    metadata = np.load(data_dir / "metadata.npz", allow_pickle=False)
    mask = np.asarray(metadata["fdil_split"]).astype(str) == "train"
    expected = set(np.asarray(metadata["image_ids"]).astype(str)[mask].tolist())

    rows = []
    invalid_lines = []
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
    ids = [str(row.get("image_id")) for row in rows]
    unique = set(ids)
    duplicates = sorted(image_id for image_id, count in Counter(ids).items() if count > 1)
    errors = [row for row in rows if row.get("error")]
    missing_steps = [
        str(row.get("image_id"))
        for row in rows
        if not all(f"Step {index}:" in str(row.get("response_text", "")) for index in (1, 2, 3))
    ]
    missing = sorted(expected - unique)
    unexpected = sorted(unique - expected)
    word_counts = [len(str(row.get("response_text", "")).split()) for row in rows if not row.get("error")]
    summary = {
        "dataset": args.dataset,
        "expected": len(expected),
        "rows": len(rows),
        "unique_ids": len(unique),
        "invalid_json_lines": invalid_lines,
        "duplicate_ids": duplicates,
        "error_rows": len(errors),
        "missing_step_rows": len(missing_steps),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "word_count": {
            "min": int(np.min(word_counts)) if word_counts else 0,
            "median": float(np.median(word_counts)) if word_counts else 0.0,
            "max": int(np.max(word_counts)) if word_counts else 0,
        },
        "valid": not any([invalid_lines, duplicates, errors, missing_steps, missing, unexpected]),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
