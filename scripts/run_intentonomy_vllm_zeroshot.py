#!/usr/bin/env python3
"""Run OpenAI-compatible vLLM zero-shot evaluation on Intentonomy."""

from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_SPEC = importlib.util.spec_from_file_location(
    "intentonomy_metrics",
    PROJECT_ROOT / "src" / "utils" / "metrics.py",
)
if METRICS_SPEC is None or METRICS_SPEC.loader is None:
    raise RuntimeError("Failed to load src/utils/metrics.py")
intentonomy_metrics = importlib.util.module_from_spec(METRICS_SPEC)
METRICS_SPEC.loader.exec_module(intentonomy_metrics)
compute_f1 = intentonomy_metrics.compute_f1
compute_mAP = intentonomy_metrics.compute_mAP
compute_difficulty_scores = intentonomy_metrics.compute_difficulty_scores

DEFAULT_ANNOTATION = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_test2020.json"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT.parent / "Intentonomy" / "data" / "images" / "low"

SYSTEM_PROMPT = (
    "You are an expert visual intent recognition evaluator. "
    "You must infer human intents from the visible image evidence only."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=str, required=True, help="OpenAI-compatible base URL, e.g. http://host:port/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_ANNOTATION))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_split(annotation_file: Path, image_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda item: int(item.get("id", item.get("category_id"))))
    class_names = [str(item["name"]) for item in categories]

    labels_by_image: dict[str, list[int]] = {}
    for ann in data["annotations"]:
        image_id = str(ann.get("image_id", ann.get("image_category_id")))
        labels = ann.get("category_ids", ann.get("category_category_ids", []))
        labels_by_image[image_id] = [int(x) for x in labels]

    records: list[dict[str, Any]] = []
    for image in data["images"]:
        image_id = str(image["id"])
        if image_id not in labels_by_image:
            continue
        filename = str(image["filename"])
        if filename.startswith("low/"):
            filename = filename[4:]
        image_path = image_dir / filename
        if not image_path.exists():
            continue
        records.append(
            {
                "index": len(records),
                "image_id": image_id,
                "image_path": str(image_path),
                "label_ids": labels_by_image[image_id],
                "label_names": [class_names[idx] for idx in labels_by_image[image_id]],
            }
        )
    return class_names, records


def _encode_image_data_url(image_path: str, max_image_size: int) -> str:
    with Image.open(image_path).convert("RGB") as image:
        max_side = max(image.size)
        if max_side > int(max_image_size):
            scale = float(max_image_size) / float(max_side)
            image = image.resize(
                (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
                resample=Image.BICUBIC,
            )
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_prompt(class_names: list[str]) -> str:
    categories = ", ".join(f"'{name}'" for name in class_names)
    return f"""
Please analyze the given <image> and determine which of the following categories it may belong to. Provide your answer in the format: {{'Category 1': probability, 'Category 2': probability, ...}}. Ensure that the predicted probabilities for each selected category are within the range of 0 to 1. The categories are: {categories}.
""".strip()


def _request_prediction(
    record: dict[str, Any],
    *,
    base_url: str,
    model: str,
    class_names: list[str],
    max_image_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(record["image_path"], max_image_size)}},
                    {"type": "text", "text": _build_prompt(class_names)},
                ],
            },
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    last_error = None
    for attempt in range(int(retries) + 1):
        try:
            response = requests.post(url, json=payload, timeout=float(timeout))
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            pred_ids, pred_probs, parse_status = _parse_prediction(text, class_names)
            return {
                **record,
                "response_text": text,
                "pred_ids": pred_ids,
                "pred_names": [class_names[idx] for idx in pred_ids],
                "pred_probs": pred_probs,
                "parse_status": parse_status,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - preserve remote error text in output.
            last_error = str(exc)
            if attempt < int(retries):
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    return {
        **record,
        "response_text": "",
        "pred_ids": [],
        "pred_names": [],
        "pred_probs": {},
        "parse_status": "error",
        "error": last_error,
    }


def _parse_prediction(text: str, class_names: list[str]) -> tuple[list[int], dict[str, float], str]:
    normalized_to_idx = {_normalize_label(name): idx for idx, name in enumerate(class_names)}
    parsed: Any = None
    status = "literal"
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    for loader_name, loader in (("json", json.loads), ("literal", ast.literal_eval)):
        try:
            parsed = loader(candidate)
            status = loader_name if match else f"{loader_name}_full"
            break
        except Exception:
            parsed = None

    probs_by_idx: dict[int, float] = {}
    if isinstance(parsed, dict):
        for label, prob in parsed.items():
            key = _normalize_label(str(label))
            if key in normalized_to_idx:
                try:
                    value = float(prob)
                except (TypeError, ValueError):
                    value = 1.0
                probs_by_idx[normalized_to_idx[key]] = min(1.0, max(0.0, value))

    if not probs_by_idx:
        for label, prob in re.findall(r"""['"]([^'"]+)['"]\s*:\s*([0-9]*\.?[0-9]+)""", candidate):
            key = _normalize_label(str(label))
            if key not in normalized_to_idx:
                continue
            value = min(1.0, max(0.0, float(prob)))
            probs_by_idx[normalized_to_idx[key]] = value
        if probs_by_idx:
            status = "regex_pairs"

    if not probs_by_idx:
        status = "text_match" if parsed is None else "fallback_text_match"
        lowered = _normalize_label(text)
        for key, idx in normalized_to_idx.items():
            if key and key in lowered:
                probs_by_idx[idx] = 1.0

    pred_ids = sorted(probs_by_idx)
    pred_probs = {class_names[idx]: float(probs_by_idx[idx]) for idx in pred_ids}
    return pred_ids, pred_probs, status


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") is None:
                rows[str(row["image_id"])] = row
    return rows


def _write_metrics(output_dir: Path, class_names: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = np.zeros((len(rows), len(class_names)), dtype=np.int32)
    scores = np.zeros((len(rows), len(class_names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        targets[row_idx, [int(idx) for idx in row["label_ids"]]] = 1
        for pred_idx in row["pred_ids"]:
            class_name = class_names[int(pred_idx)]
            scores[row_idx, int(pred_idx)] = float(row.get("pred_probs", {}).get(class_name, 1.0))

    micro, samples, macro, per_class_f1 = compute_f1(targets, scores, threshold=0.5, use_inference_strategy=False)
    map_score, per_class_ap = compute_mAP(scores, targets, return_each=True)
    difficulty = compute_difficulty_scores(per_class_f1)
    metrics = {
        "num_samples": int(len(rows)),
        "num_classes": int(len(class_names)),
        "micro_f1": float(micro),
        "samples_f1": float(samples),
        "macro_f1": float(macro),
        "mAP": float(map_score),
        "easy_f1": float(difficulty["easy"]),
        "medium_f1": float(difficulty["medium"]),
        "hard_f1": float(difficulty["hard"]),
        "parse_status_counts": _count_values(row["parse_status"] for row in rows),
        "empty_prediction_count": int(sum(1 for row in rows if not row["pred_ids"])),
        "per_class": [
            {
                "class_id": int(idx),
                "class_name": name,
                "f1": float(per_class_f1[idx]),
                "ap": float(per_class_ap[idx]),
            }
            for idx, name in enumerate(class_names)
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "predictions.npz",
        scores=scores,
        labels=targets,
        image_ids=np.asarray([row["image_id"] for row in rows]),
        class_names=np.asarray(class_names),
    )
    return metrics


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "predictions.jsonl"
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()

    class_names, records = _load_split(Path(args.annotation_file), Path(args.image_dir))
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]

    existing = _load_existing(output_jsonl)
    pending = [record for record in records if record["image_id"] not in existing]
    print(f"[zeroshot] samples={len(records)} existing={len(existing)} pending={len(pending)}")

    with output_jsonl.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [
                executor.submit(
                    _request_prediction,
                    record,
                    base_url=args.base_url,
                    model=args.model,
                    class_names=class_names,
                    max_image_size=int(args.max_image_size),
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    timeout=float(args.request_timeout),
                    retries=int(args.retries),
                )
                for record in pending
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing[str(row["image_id"])] = row
                if done % 25 == 0 or done == len(futures):
                    print(f"[zeroshot] completed={done}/{len(futures)}")

    rows = [existing[record["image_id"]] for record in records if record["image_id"] in existing]
    metrics = _write_metrics(output_dir, class_names, rows)
    print(
        "[zeroshot] "
        f"macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} "
        f"samples_f1={metrics['samples_f1']:.4f} mAP={metrics['mAP']:.2f}"
    )


if __name__ == "__main__":
    main()
