#!/usr/bin/env python3
"""Normalize VLM rationale headings without changing generated body text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize(text: str) -> tuple[str, bool]:
    original = text
    replacements = [
        (r"(?im)^\*{0,2}Step\s*1\s*:\s*Visual Evidence\*{0,2}\s*$", "Step 1: Visual Evidence"),
        (r"(?im)^\*{0,2}Visual Evidence\*{0,2}\s*$", "Step 1: Visual Evidence"),
        (r"(?im)^\*{0,2}Step\s*2\s*:\s*Distributional Interpretation\*{0,2}\s*$", "Step 2: Distributional Interpretation"),
        (r"(?im)^\*{0,2}Distributional Interpretation\*{0,2}\s*$", "Step 2: Distributional Interpretation"),
        (r"(?im)^\*{0,2}Step\s*3\s*:\s*Counterfactual Disambiguation\*{0,2}\s*$", "Step 3: Counterfactual Disambiguation"),
        (r"(?im)^\*{0,2}Counterfactual Disambiguation\*{0,2}\s*$", "Step 3: Counterfactual Disambiguation"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    text = re.sub(
        r"(?im)^Distributional Interpretation:\s+",
        "Step 2: Distributional Interpretation\n",
        text,
    )
    text = re.sub(
        r"(?im)^Counterfactual Disambiguation:\s+",
        "Step 3: Counterfactual Disambiguation\n",
        text,
    )

    # Rare prompt-injection-like cases repeat an image-visible web heading for
    # all three generated sections. Only rewrite those standalone headings.
    if "Step 2:" not in text or "Step 3:" not in text:
        for repeated in ("RECOMMENDED BLOGS", "GROUP B"):
            matches = list(re.finditer(rf"(?im)^{re.escape(repeated)}\s*$", text))
            if len(matches) >= 3:
                section_names = [
                    "Step 1: Visual Evidence",
                    "Step 2: Distributional Interpretation",
                    "Step 3: Counterfactual Disambiguation",
                ]
                count = 0

                def replace_repeated(_: re.Match[str]) -> str:
                    nonlocal count
                    value = section_names[min(count, 2)]
                    count += 1
                    return value

                text = re.sub(rf"(?im)^{re.escape(repeated)}\s*$", replace_repeated, text)
                break

    # If the first section received an image-derived title (e.g. Establishment
    # or Storyboarding), retain it as body context and add the canonical header.
    if "Step 1:" not in text:
        text = "Step 1: Visual Evidence\n" + text
    return text, text != original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        row["response_text"], was_changed = normalize(str(row.get("response_text", "")))
        changed += int(was_changed)
    temporary = args.input_jsonl.with_suffix(args.input_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.input_jsonl)
    print(f"normalized={changed} rows={len(rows)} file={args.input_jsonl}")


if __name__ == "__main__":
    main()
