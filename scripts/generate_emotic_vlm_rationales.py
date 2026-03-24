#!/usr/bin/env python3
"""Generate Emotic person-level rationales with a red-box prompt for Qwen-2.5-VL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emotic_datamodule import EmoticDataModule

DEFAULT_MODEL_PATH = (
    Path("/home/evelynmuir/lambda/hf-models")
    / "models--Qwen--Qwen2.5-VL-7B-Instruct"
    / "snapshots"
    / "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
SYSTEM_PROMPT = (
    "You are an expert affective computing researcher. "
    "Analyze the person in the red box only and explain the visible emotional evidence "
    "with precise, grounded reasoning."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Emotic person-level Qwen rationales.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--annotation-file", type=str, default=str(PROJECT_ROOT.parent / "Emotic" / "Annotations" / "Annotations.mat"))
    parser.add_argument("--image-root", type=str, default=str(PROJECT_ROOT.parent / "Emotic" / "emotic"))
    parser.add_argument("--description-file", type=str, default=str(PROJECT_ROOT.parent / "Emotic" / "emotion_description_gemini.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace):
    dm = EmoticDataModule(
        annotation_file=args.annotation_file,
        image_root=args.image_root,
        description_file=args.description_file,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.prepare_data()
    dm.setup()
    return {"train": dm.data_train, "val": dm.data_val, "test": dm.data_test}[args.split]


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                ids.add(str(json.loads(line).get("image_id")))
            except json.JSONDecodeError:
                continue
    return ids


def _draw_red_box(image_path: str, bbox: List[int], max_image_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=max(3, max(image.size) // 200))
    max_side = max(image.size)
    if max_side <= int(max_image_size):
        return image
    scale = float(max_image_size) / float(max_side)
    new_size = (
        max(1, int(round(image.size[0] * scale))),
        max(1, int(round(image.size[1] * scale))),
    )
    return image.resize(new_size, resample=Image.BICUBIC)


def _join(names: List[str]) -> str:
    if not names:
        return "unknown emotion"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _build_prompt(label_names: List[str]) -> str:
    gt = _join(label_names)
    return f"""
Analyze the person in the red box only.

The ground-truth emotions for the person in the red box are: [{gt}].

Please write a structured report with exactly these three sections:

Step 1: Visual Evidence
Describe the visible facial expression, posture, gesture, interaction, clothing cues, and nearby objects that support these emotions for the person in the red box.

Step 2: Contextual Bridging
Explain how the surrounding people, scene, and situation help justify why these emotions co-occur for the person in the red box.

Step 3: Counterfactual Disambiguation
Name one other emotion that could be confused with this case, and explain what visual evidence is missing or contradictory for that alternative emotion.
""".strip()


def _iter_requests(dataset, start_index: int, max_samples: int | None, existing_ids: set[str]) -> Iterable[Dict[str, Any]]:
    count = 0
    for idx, sample in enumerate(dataset.samples):
        if idx < int(start_index):
            continue
        if max_samples is not None and count >= int(max_samples):
            break
        image_id = str(sample["image_id"])
        if image_id in existing_ids:
            continue
        yield {
            "index": idx,
            "image_id": image_id,
            "image_path": str(sample["image_path"]),
            "bbox": list(sample["bbox"]),
            "label_names": list(sample["label_names"]),
            "prompt": _build_prompt(list(sample["label_names"])),
        }
        count += 1


def main() -> None:
    args = _parse_args()
    dataset = _load_dataset(args)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        existing_ids = _load_existing_ids(output_path)
    else:
        existing_ids = set()
        output_path.write_text("", encoding="utf-8")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=int(args.tensor_parallel_size),
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )

    requests = list(
        _iter_requests(
            dataset=dataset,
            start_index=int(args.start_index),
            max_samples=args.max_samples,
            existing_ids=existing_ids,
        )
    )
    print(f"[EmoticRationale] split={args.split} pending_requests={len(requests)} output={output_path}")
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(requests), int(args.request_batch_size)):
            batch = requests[start : start + int(args.request_batch_size)]
            mm_requests = []
            for item in batch:
                image = _draw_red_box(
                    image_path=item["image_path"],
                    bbox=item["bbox"],
                    max_image_size=int(args.max_image_size),
                )
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": item["prompt"]},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                mm_requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})

            outputs = llm.generate(mm_requests, sampling_params)
            for item, output in zip(batch, outputs, strict=True):
                response_text = output.outputs[0].text.strip() if output.outputs else ""
                record = {
                    "split": args.split,
                    "index": int(item["index"]),
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "bbox": item["bbox"],
                    "label_names": item["label_names"],
                    "user_prompt": item["prompt"],
                    "response_text": response_text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[EmoticRationale] generated {min(start + len(batch), len(requests))}/{len(requests)}")


if __name__ == "__main__":
    main()
