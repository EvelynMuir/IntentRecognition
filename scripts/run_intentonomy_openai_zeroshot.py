#!/usr/bin/env python3
"""Run Intentonomy zero-shot evaluation through the OpenAI Chat Completions API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = importlib.util.spec_from_file_location(
    "intentonomy_vllm_zeroshot", PROJECT_ROOT / "scripts" / "run_intentonomy_vllm_zeroshot.py"
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("Failed to load the shared Intentonomy zero-shot evaluator")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

PROTOCOL_VERSION = "intentonomy-openai-zeroshot-v1-all28-json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--annotation-file", default=str(base.DEFAULT_ANNOTATION))
    parser.add_argument("--image-dir", default=str(base.DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-image-size", type=int, default=768)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def request_prediction(
    record: dict[str, Any],
    *,
    url: str,
    api_key: str,
    model: str,
    class_names: list[str],
    max_image_size: int,
    max_completion_tokens: int,
    reasoning_effort: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": base.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base._encode_image_data_url(record["image_path"], max_image_size)
                        },
                    },
                    {"type": "text", "text": base._build_prompt(class_names)},
                ],
            },
        ],
        "reasoning_effort": reasoning_effort,
        "temperature": 0.0,
        "max_completion_tokens": int(max_completion_tokens),
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            pred_ids, pred_probs, parse_status = base._parse_prediction(text, class_names)
            if len(pred_probs) != len(class_names):
                raise ValueError(f"Expected {len(class_names)} probabilities, parsed {len(pred_probs)}")
            return {
                **record,
                "response_text": text,
                "pred_ids": pred_ids,
                "pred_names": [class_names[idx] for idx in pred_ids],
                "pred_probs": pred_probs,
                "parse_status": parse_status,
                "usage": body.get("usage", {}),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            detail = ""
            if "response" in locals() and getattr(response, "text", None):
                detail = response.text[:500]
            last_error = f"{type(exc).__name__}: {exc}; {detail}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))
    return {
        **record,
        "response_text": "",
        "pred_ids": [],
        "pred_names": [],
        "pred_probs": {},
        "parse_status": "error",
        "usage": {},
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    if not args.base_url:
        raise RuntimeError("OPENAI_BASE_URL is unset and --base-url was not provided")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is unset")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "predictions.jsonl"
    config_path = output_dir / "run_config.json"
    run_config = {
        "protocol_version": PROTOCOL_VERSION,
        **{key: value for key, value in vars(args).items() if key != "api_key_env"},
    }
    if args.overwrite and output_jsonl.exists():
        output_jsonl.unlink()
    if config_path.exists() and output_jsonl.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in run_config.items() if key not in {"workers", "retries", "request_timeout"}}
        if any(previous.get(key) != value for key, value in comparable.items()):
            raise RuntimeError("Existing predictions use a different evaluation configuration")
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    class_names, records = base._load_split(Path(args.annotation_file), Path(args.image_dir))
    if args.max_samples is not None:
        records = records[: args.max_samples]
    existing = base._load_existing(output_jsonl)
    pending = [row for row in records if row["image_id"] not in existing]
    print(f"[openai-zeroshot] samples={len(records)} existing={len(existing)} pending={len(pending)}", flush=True)

    url = args.base_url.rstrip("/") + "/chat/completions"
    with output_jsonl.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    request_prediction,
                    record,
                    url=url,
                    api_key=api_key,
                    model=args.model,
                    class_names=class_names,
                    max_image_size=args.max_image_size,
                    max_completion_tokens=args.max_completion_tokens,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.request_timeout,
                    retries=args.retries,
                )
                for record in pending
            ]
            for done, future in enumerate(as_completed(futures), 1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if row["error"] is None:
                    existing[str(row["image_id"])] = row
                else:
                    print(f"[openai-zeroshot] error image_id={row['image_id']}: {row['error']}", flush=True)
                if done % 25 == 0 or done == len(futures):
                    print(f"[openai-zeroshot] completed={done}/{len(futures)}", flush=True)

    rows = [existing[row["image_id"]] for row in records if row["image_id"] in existing]
    base._compact_predictions(output_jsonl, rows)
    metrics = base._write_metrics(output_dir, class_names, rows)
    usage = {
        key: sum(int(row.get("usage", {}).get(key, 0) or 0) for row in rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    metrics["usage"] = usage
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[openai-zeroshot] successful={len(rows)}/{len(records)} "
        f"macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} "
        f"samples_f1={metrics['samples_f1']:.4f} mAP={metrics['mAP']:.2f} usage={usage}",
        flush=True,
    )


if __name__ == "__main__":
    main()
