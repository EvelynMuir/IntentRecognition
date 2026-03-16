#!/usr/bin/env python3
"""Generate offline VLM rationales or captions for Intentonomy samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image
import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.intentonomy_clip_vit_slot_module import INTENTONOMY_DESCRIPTIONS
from src.utils.decision_rule_calibration import search_classwise_thresholds

DEFAULT_BASE_CACHE = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_TRAIN_ANNOTATION = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT.parent / "Intentonomy" / "data" / "images" / "low"
DEFAULT_MODEL_PATH = (
    Path("/home/evelynmuir/lambda/hf-models")
    / "models--Qwen--Qwen2.5-VL-7B-Instruct"
    / "snapshots"
    / "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
SYSTEM_PROMPT = (
    "You are an expert human behavioral analyst and computer vision specialist. "
    "Your task is to analyze an image and provide a highly logical, structured reasoning "
    "chain that explains the underlying human intent. Strictly follow the requested format "
    "and avoid conversational filler."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline VLM rationales or captions aligned to Intentonomy train samples."
    )
    parser.add_argument("--base-cache", type=str, default=str(DEFAULT_BASE_CACHE / "train_base.npz"))
    parser.add_argument("--annotation-file", type=str, default=str(DEFAULT_TRAIN_ANNOTATION))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rationale", choices=["rationale", "caption"])
    parser.add_argument("--label-source", type=str, default="gt", choices=["gt", "baseline_pred"])
    parser.add_argument(
        "--threshold-source-cache",
        type=str,
        default=None,
        help="Optional cache npz used to derive class-wise thresholds when label-source=baseline_pred.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-image-size", type=int, default=448)
    parser.add_argument("--request-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_annotation(annotation_file: Path, image_dir: Path) -> tuple[list[str], dict[str, Path]]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    class_names = [str(category["name"]) for category in categories]

    image_paths: dict[str, Path] = {}
    for item in data["images"]:
        image_id = str(item["id"])
        filename = str(item["filename"])
        if filename.startswith("low/"):
            filename = filename[4:]
        image_paths[image_id] = image_dir / filename
    return class_names, image_paths


def _load_base_cache(path: Path) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    return {
        "logits": np.asarray(arr["logits"], dtype=np.float32),
        "labels": np.asarray(arr["labels"], dtype=np.float32),
        "image_ids": [str(item) for item in arr["image_ids"].tolist()],
    }


def _join_label_names(names: Sequence[str]) -> str:
    items = [str(name).strip() for name in names if str(name).strip()]
    if not items:
        return "None"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _positive_label_names(label_row: np.ndarray, class_names: Sequence[str]) -> list[str]:
    return [class_names[idx] for idx in np.where(np.asarray(label_row) > 0.0)[0].tolist()]


def _predicted_label_names(
    score_row: np.ndarray,
    class_names: Sequence[str],
    class_thresholds: np.ndarray | None,
) -> list[str]:
    scores = np.asarray(score_row, dtype=np.float32)
    if class_thresholds is None:
        predicted_ids = np.where(scores > 0.5)[0].tolist()
    else:
        predicted_ids = np.where(scores > np.asarray(class_thresholds, dtype=np.float32))[0].tolist()
    if not predicted_ids:
        predicted_ids = [int(np.argmax(scores))]
    return [class_names[idx] for idx in predicted_ids]


def _top_false_positive_name(
    logit_row: np.ndarray,
    label_row: np.ndarray,
    class_names: Sequence[str],
) -> str:
    negatives = np.where(np.asarray(label_row) <= 0.0)[0]
    if negatives.size == 0:
        return class_names[int(np.argmax(logit_row))]
    negative_logits = np.asarray(logit_row)[negatives]
    best_neg = int(negatives[int(np.argmax(negative_logits))])
    return class_names[best_neg]


def _top_nonselected_name(
    logit_row: np.ndarray,
    selected_names: Sequence[str],
    class_names: Sequence[str],
) -> str:
    selected = {str(name) for name in selected_names}
    order = np.argsort(-np.asarray(logit_row, dtype=np.float32))
    for idx in order.tolist():
        candidate = class_names[int(idx)]
        if candidate not in selected:
            return candidate
    return class_names[int(order[0])]


def _build_user_prompt(mode: str, y_true: str, y_confuse: str) -> str:
    if mode == "caption":
        return (
            "Describe this image for visual intent recognition in 2-3 concise sentences. "
            "Focus on visible actions, salient objects, and scene context. Avoid speculation "
            "beyond what is visually grounded."
        )

    return f"""
Given this image, the ground-truth human intents are strictly defined as the concurrent occurrence of: [{y_true}].

Please carefully observe the image and generate a structured reasoning report strictly following these 3 steps:

Step 1: Visual Evidence
Describe the explicit physical actions, key objects, body language, or facial expressions in the image that strongly support the presence of these concurrent intents ([{y_true}]). Be specific about what you see and how the visual cues correspond to these multiple intents.

Step 2: Contextual Bridging
Explain how the background, environment, or the relationship between the subjects logically connects with the visual cues from Step 1 to reveal why these multiple psychological motivations ([{y_true}]) naturally co-exist in this specific scene.

Step 3: Counterfactual Disambiguation
In a purely visual context, a machine learning model might easily misclassify this image as also containing the intent of [{y_confuse}]. Point out the specific visual clues that are definitively missing, or the contradictory details that are present, which prove the intent of [{y_confuse}] is not happening here.
""".strip()


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ids.add(str(record.get("image_id")))
    return ids


def _iter_requests(
    image_ids: Sequence[str],
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    image_paths: Mapping[str, Path],
    mode: str,
    label_source: str,
    class_thresholds: np.ndarray | None,
    start_index: int,
    max_samples: int | None,
) -> Iterable[dict[str, Any]]:
    count = 0
    for idx, image_id in enumerate(image_ids):
        if idx < int(start_index):
            continue
        if max_samples is not None and count >= int(max_samples):
            break
        image_path = image_paths.get(str(image_id))
        if image_path is None or not image_path.exists():
            continue
        if label_source == "baseline_pred":
            positive_names = _predicted_label_names(
                score_row=logits[idx],
                class_names=class_names,
                class_thresholds=class_thresholds,
            )
            y_true = _join_label_names(positive_names)
            y_confuse = _top_nonselected_name(logits[idx], positive_names, class_names)
        else:
            positive_names = _positive_label_names(labels[idx], class_names)
            y_true = _join_label_names(positive_names)
            y_confuse = _top_false_positive_name(logits[idx], labels[idx], class_names)
        prompt_text = _build_user_prompt(mode=mode, y_true=y_true, y_confuse=y_confuse)
        yield {
            "index": int(idx),
            "image_id": str(image_id),
            "image_path": str(image_path),
            "positive_class_names": positive_names,
            "confuse_class_name": str(y_confuse),
            "user_prompt": prompt_text,
        }
        count += 1


def _load_resized_image(image_path: str, max_image_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    max_side = max(image.size)
    if max_side <= int(max_image_size):
        return image
    scale = float(max_image_size) / float(max_side)
    new_size = (
        max(1, int(round(image.size[0] * scale))),
        max(1, int(round(image.size[1] * scale))),
    )
    return image.resize(new_size, resample=Image.BICUBIC)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and bool(args.overwrite):
        output_path.unlink()

    class_names, image_paths = _load_annotation(
        annotation_file=Path(args.annotation_file),
        image_dir=Path(args.image_dir),
    )
    cache = _load_base_cache(Path(args.base_cache))
    class_thresholds = None
    if args.label_source == "baseline_pred" and args.threshold_source_cache is not None:
        threshold_cache = _load_base_cache(Path(args.threshold_source_cache))
        class_thresholds = search_classwise_thresholds(
            scores=np.asarray(threshold_cache["logits"], dtype=np.float32),
            targets=np.asarray(threshold_cache["labels"], dtype=np.float32),
        )

    existing_ids = _load_existing_ids(output_path)
    metadata_records = []
    for record in _iter_requests(
        image_ids=cache["image_ids"],
        labels=cache["labels"],
        logits=cache["logits"],
        class_names=class_names,
        image_paths=image_paths,
        mode=str(args.mode),
        label_source=str(args.label_source),
        class_thresholds=class_thresholds,
        start_index=int(args.start_index),
        max_samples=args.max_samples,
    ):
        if record["image_id"] in existing_ids:
            continue
        metadata_records.append(record)

    print(
        f"[VLM] pending_samples={len(metadata_records)} mode={args.mode} "
        f"request_batch_size={int(args.request_batch_size)}"
    )
    if not metadata_records:
        return

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )

    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(metadata_records), int(args.request_batch_size)):
            batch_meta = metadata_records[start : start + int(args.request_batch_size)]
            llm_inputs = []
            for record in batch_meta:
                image = _load_resized_image(record["image_path"], max_image_size=int(args.max_image_size))
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": record["image_path"]},
                            {"type": "text", "text": record["user_prompt"]},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                llm_inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                )

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            for meta, output in zip(batch_meta, outputs, strict=True):
                text = output.outputs[0].text.strip() if output.outputs else ""
                finish_reason = output.outputs[0].finish_reason if output.outputs else None
                generated_token_count = len(output.outputs[0].token_ids) if output.outputs else 0
                row = {
                    "index": int(meta["index"]),
                    "image_id": str(meta["image_id"]),
                    "image_path": str(meta["image_path"]),
                    "mode": str(args.mode),
                    "positive_class_names": meta["positive_class_names"],
                    "confuse_class_name": str(meta["confuse_class_name"]),
                    "user_prompt": str(meta["user_prompt"]),
                    "response_text": text,
                    "finish_reason": None if finish_reason is None else str(finish_reason),
                    "generated_token_count": int(generated_token_count),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            print(f"[VLM] wrote_batch={len(batch_meta)} total_written={written}/{len(metadata_records)}")

    print(f"[VLM] wrote={written} records to {output_path}")


if __name__ == "__main__":
    main()
