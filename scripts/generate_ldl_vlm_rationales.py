#!/usr/bin/env python3
"""Generate resumable training-only VLM rationales for Flickr/Twitter-LDL."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ldl_datamodule import LDLImageDataset


SYSTEM_PROMPT = (
    "You are an expert visual emotion analyst. Produce concise, visually grounded "
    "reasoning for label-distribution learning. Follow the three requested steps exactly. "
    "Write exactly two sentences per step and no more than 220 words in total."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flickrldl", "twitterldl", "emotion6"], required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--max-image-size", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(project_root=root)
    return parser.parse_args()


def load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if not row.get("error"):
                    output.add(str(row["image_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return output


def load_baseline(path: Path | None, expected_ids: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    payload = np.load(path, allow_pickle=False)
    ids = np.asarray(payload["image_ids"]).astype(str)
    key = "train" if "train" in payload.files else "predictions"
    scores = np.asarray(payload[key], dtype=np.float32)
    positions = {image_id: index for index, image_id in enumerate(ids.tolist())}
    missing = [image_id for image_id in expected_ids.tolist() if image_id not in positions]
    if missing:
        raise ValueError(f"Baseline predictions miss {len(missing)} train samples")
    return scores[[positions[image_id] for image_id in expected_ids.tolist()]]


def image_data_url(image: Any, max_size: int) -> str:
    image = image.copy().convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_prompt(class_names: list[str], target: np.ndarray, baseline: np.ndarray | None) -> tuple[str, list[str], str]:
    order = np.argsort(-target)
    top = np.asarray([index for index in order if target[index] >= 0.05][:3], dtype=np.int64)
    if top.size == 0:
        top = order[:1]
    target_text = ", ".join(f"{class_names[index]} ({target[index]:.3f})" for index in top)
    selected = set(top.tolist())
    if baseline is None:
        confuse_idx = int(order[3])
    else:
        confuse_idx = next(int(index) for index in np.argsort(-baseline) if int(index) not in selected)
    confuse = class_names[confuse_idx]
    prompt = f"""
The human annotation distribution is dominated by: {target_text}.
The current visual model also considers {confuse} plausible. Use the probabilities only
to understand relative annotator agreement; do not merely repeat them in the report.

Step 1: Visual Evidence
Describe concrete subjects, actions, expressions, objects, colors, and composition that
support the dominant emotions. Do not invent invisible facts.

Step 2: Distributional Interpretation
Explain why several emotions can coexist and why annotators may disagree about their
relative strength in this image.

Step 3: Counterfactual Disambiguation
Contrast the dominant interpretation with {confuse}. Identify visible evidence that
weakens {confuse}, or explicitly state when the image remains genuinely ambiguous.

Formatting constraint: retain the three headings verbatim, write exactly two sentences
under each heading, stay below 220 words total, and do not add an introduction or conclusion.
""".strip()
    return prompt, [class_names[index] for index in top], confuse


def request_one(
    index: int, dataset: LDLImageDataset, class_names: list[str], baseline: np.ndarray | None,
    args: argparse.Namespace
) -> dict[str, Any]:
    sample = dataset[index]
    target = sample["soft_labels"].numpy()
    prompt, dominant, confuse = build_prompt(class_names, target, None if baseline is None else baseline[index])
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(sample["image"], args.max_image_size)}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(
                args.base_url.rstrip("/") + "/chat/completions", json=payload, timeout=args.timeout
            )
            response.raise_for_status()
            text = str(response.json()["choices"][0]["message"]["content"]).strip()
            if not text:
                raise ValueError("empty response")
            return {
                "index": index, "image_id": sample["image_id"], "dominant_emotions": dominant,
                "confuse_emotion": confuse, "target_distribution": target.tolist(),
                "response_text": text, "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
    return {
        "index": index, "image_id": sample["image_id"], "dominant_emotions": dominant,
        "confuse_emotion": confuse, "target_distribution": target.tolist(),
        "response_text": "", "error": last_error,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or args.project_root.parent / "LDL" / "processed" / args.dataset
    dataset = LDLImageDataset(str(data_dir), "train", transform=None)
    dataset._get_lmdb("train")  # Open once before worker threads.
    class_names = dataset.class_names
    baseline = load_baseline(args.baseline_predictions, dataset.image_ids)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    existing = load_existing(args.output_jsonl)
    indices = [index for index, image_id in enumerate(dataset.image_ids.tolist()) if image_id not in existing]
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    mode = "a" if args.output_jsonl.exists() else "w"
    completed = 0
    with args.output_jsonl.open(mode, encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(request_one, index, dataset, class_names, baseline, args): index for index in indices
        }
        for future in as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if completed % 100 == 0 or row.get("error"):
                print(f"[{args.dataset}] completed={completed}/{len(indices)} error={row.get('error')}", flush=True)


if __name__ == "__main__":
    main()
