#!/usr/bin/env python3
"""Encode LDL VLM rationales with CLIP for the UTD text teacher."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--encoder", choices=["bge", "clip"], default="bge")
    parser.add_argument("--bge-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--hf-cache-dir", default="/home/evelynmuir/lambda/hf-models")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def clean_rationale(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(text.split()).strip()


def main() -> None:
    args = parse_args()
    rows = []
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("error") and str(row.get("response_text", "")).strip():
                rows.append(row)
    if not rows:
        raise RuntimeError("No successful rationale rows")
    rows.sort(key=lambda row: int(row["index"]))
    ids = np.asarray([str(row["image_id"]) for row in rows])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate rationale image IDs")
    texts = [clean_rationale(str(row["response_text"])) for row in rows]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    tokenizer = None
    if args.encoder == "clip":
        import clip

        model, _ = clip.load(args.clip_model, device=device)
        model = model.eval().to(device)
    else:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.bge_model, cache_dir=args.hf_cache_dir)
        model = AutoModel.from_pretrained(args.bge_model, cache_dir=args.hf_cache_dir).eval().to(device)
    features = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            if args.encoder == "clip":
                tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                encoded = model.encode_text(tokens).float()
            else:
                tokens = tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                encoded = model(**tokens).last_hidden_state[:, 0].float()
            features.append(F.normalize(encoded, dim=1).cpu().numpy())
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz, image_ids=ids, features=np.concatenate(features).astype(np.float32),
        texts=np.asarray(texts), encoder=np.asarray(args.encoder)
    )
    print(f"saved {len(ids)} rationale features to {args.output_npz}")


if __name__ == "__main__":
    main()
