#!/usr/bin/env python3
"""Encode generated VLM texts into text features using CLIP or BGE."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is required for extract_vlm_rationale_features.py. "
        "Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode generated VLM rationale/caption texts into CLIP text features."
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="response_text")
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="clip",
        choices=["clip", "bge"],
        help="Which text encoder to use for feature extraction.",
    )
    parser.add_argument(
        "--pos-source",
        type=str,
        default="step1_step2",
        choices=["step1_step2", "step1_only"],
        help="Which rationale segments to merge into pos_features.",
    )
    parser.add_argument("--clip-model-name", type=str, default="ViT-L/14")
    parser.add_argument("--hf-model-name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--hf-cache-dir", type=str, default="/home/evelynmuir/lambda/hf-models")
    parser.add_argument("--hf-max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


STEP1_RE = re.compile(
    r"(?:^|\n)(?:#+\s*)?Step\s*1\s*:\s*.*?\n(.*?)(?=(?:\n(?:#+\s*)?Step\s*2\s*:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
STEP2_RE = re.compile(
    r"(?:^|\n)(?:#+\s*)?Step\s*2\s*:\s*.*?\n(.*?)(?=(?:\n(?:#+\s*)?Step\s*3\s*:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
STEP3_RE = re.compile(
    r"(?:^|\n)(?:#+\s*)?Step\s*3\s*:\s*.*?\n(.*?)(?=\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_step_text(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    if match is None:
        return ""
    return str(match.group(1)).strip()


def _encode_texts(
    model,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    text_encoder: str,
    tokenizer=None,
    hf_max_length: int = 512,
) -> np.ndarray:
    features: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = texts[start : start + int(batch_size)]
            if text_encoder == "clip":
                tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                text_features = model.encode_text(tokens).float()
            else:
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(hf_max_length),
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**encoded)
                text_features = outputs.last_hidden_state[:, 0].float()
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            features.append(text_features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    rows = _load_rows(Path(args.input_jsonl))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_jsonl}")

    tokenizer = None
    if args.text_encoder == "clip":
        model, _ = clip.load(args.clip_model_name, device=device)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model_name,
            cache_dir=args.hf_cache_dir,
        )
        model = AutoModel.from_pretrained(
            args.hf_model_name,
            cache_dir=args.hf_cache_dir,
        ).to(device)
        model.eval()

    texts = [str(row.get(args.text_field, "")).strip() for row in rows]
    image_ids = [str(row["image_id"]) for row in rows]
    step1_texts = [_extract_step_text(text, STEP1_RE) for text in texts]
    step2_texts = [_extract_step_text(text, STEP2_RE) for text in texts]
    step3_texts = [_extract_step_text(text, STEP3_RE) for text in texts]
    pos_texts = []
    for step1, step2, full in zip(step1_texts, step2_texts, texts, strict=True):
        if args.pos_source == "step1_only":
            pos_text = step1 if step1 else full
        else:
            joined = "\n\n".join([part for part in [step1, step2] if part])
            pos_text = joined if joined else full
        pos_texts.append(pos_text)
    neg_texts = [step3 if step3 else text for step3, text in zip(step3_texts, texts, strict=True)]

    features = _encode_texts(
        model,
        texts,
        device=device,
        batch_size=int(args.batch_size),
        text_encoder=str(args.text_encoder),
        tokenizer=tokenizer,
        hf_max_length=int(args.hf_max_length),
    )
    step1_features = _encode_texts(
        model,
        [step if step else full for step, full in zip(step1_texts, texts, strict=True)],
        device=device,
        batch_size=int(args.batch_size),
        text_encoder=str(args.text_encoder),
        tokenizer=tokenizer,
        hf_max_length=int(args.hf_max_length),
    )
    step2_features = _encode_texts(
        model,
        [step if step else full for step, full in zip(step2_texts, texts, strict=True)],
        device=device,
        batch_size=int(args.batch_size),
        text_encoder=str(args.text_encoder),
        tokenizer=tokenizer,
        hf_max_length=int(args.hf_max_length),
    )
    step3_features = _encode_texts(
        model,
        [step if step else full for step, full in zip(step3_texts, texts, strict=True)],
        device=device,
        batch_size=int(args.batch_size),
        text_encoder=str(args.text_encoder),
        tokenizer=tokenizer,
        hf_max_length=int(args.hf_max_length),
    )
    pos_features = _encode_texts(
        model,
        pos_texts,
        device=device,
        batch_size=int(args.batch_size),
        text_encoder=str(args.text_encoder),
        tokenizer=tokenizer,
        hf_max_length=int(args.hf_max_length),
    )
    neg_features = _encode_texts(
        model,
        neg_texts,
        device=device,
        batch_size=int(args.batch_size),
        text_encoder=str(args.text_encoder),
        tokenizer=tokenizer,
        hf_max_length=int(args.hf_max_length),
    )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        image_ids=np.asarray(image_ids),
        texts=np.asarray(texts),
        features=features,
        step1_texts=np.asarray(step1_texts),
        step2_texts=np.asarray(step2_texts),
        step3_texts=np.asarray(step3_texts),
        step1_features=step1_features,
        step2_features=step2_features,
        step3_features=step3_features,
        pos_texts=np.asarray(pos_texts),
        neg_texts=np.asarray(neg_texts),
        pos_features=pos_features,
        neg_features=neg_features,
        confuse_class_names=np.asarray([str(row.get("confuse_class_name", "")) for row in rows]),
        text_encoder=np.asarray(str(args.text_encoder)),
    )
    print(f"[VLM] encoded_rows={len(rows)} output={output_path}")


if __name__ == "__main__":
    main()
