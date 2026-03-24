#!/usr/bin/env python3
"""Analyze CLIP text priors, top-k reranking, and retrieval priors for Intentonomy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is required for analyze_text_prior_boundary.py. "
        "Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.intentonomy_datamodule import IntentonomyDataset
from src.models.intentonomy_clip_vit_slot_module import INTENTONOMY_DESCRIPTIONS
from src.utils.metrics import SUBSET2IDS
from src.utils.text_prior_analysis import (
    aggregate_prompt_scores,
    apply_topk_rerank_fusion,
    apply_selective_topk_rerank,
    build_topk_comparative_prior,
    build_classwise_gate,
    build_confusion_pairs,
    build_uncertainty_gate,
    class_gain_rows,
    compute_class_gains,
    compute_sample_f1_scores,
    evaluate_with_validation_threshold,
    normalize_scores_per_sample,
    threshold_predictions,
)


DEFAULT_BASELINE_CKPT = (
    PROJECT_ROOT / "logs" / "train" / "runs" / "2026-03-03_16-34-30" / "checkpoints" / "epoch_017.ckpt"
)
DEFAULT_PROMPT_TEMPLATE = "A photo that expresses the intent of {}."
INTENTONOMY_LEXICAL_PHRASES = [
    "being attractive",
    "beating others in competition",
    "communicating and expressing myself",
    "being creative and unique",
    "exploration and adventure",
    "having an easy and comfortable life",
    "enjoying life",
    "appreciating fine architecture",
    "appreciating artwork",
    "appreciating other cultures",
    "being a good parent and emotionally close to my children",
    "being happy and content",
    "being ambitious and hard-working",
    "achieving harmony and oneness",
    "being physically active, fit, and healthy",
    "being in love",
    "being in love with animals",
    "inspiring and influencing others",
    "keeping things manageable and making plans",
    "experiencing natural beauty",
    "being passionate about something",
    "being playful and lighthearted",
    "sharing my feelings with others",
    "having close friends and social belonging",
    "being successful in my occupation",
    "teaching others",
    "keeping things in order",
    "having work I really like",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze zero-shot text priors, baseline top-k rerank, retrieval prior upper bound, "
            "and hard-case corrections for the Intentonomy baseline."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Training run directory containing .hydra/config.yaml. Inferred from --ckpt-path when omitted.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=str(DEFAULT_BASELINE_CKPT) if DEFAULT_BASELINE_CKPT.exists() else None,
        help="Checkpoint path for the baseline model. Defaults to the known strongest baseline ckpt when present.",
    )
    parser.add_argument(
        "--gemini-file",
        type=str,
        default=None,
        help="Optional override for intent_description_gemini.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers used during analysis.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable dataloader pin_memory.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading the checkpoint state_dict.",
    )
    parser.add_argument(
        "--use-inference-strategy",
        action="store_true",
        help="Apply fallback-to-argmax when thresholding yields all-zero predictions.",
    )
    parser.add_argument(
        "--text-sources",
        type=str,
        default="lexical,canonical,scenario,discriminative",
        help=(
            "Comma-separated text sources to evaluate. Supported: "
            "lexical,canonical,scenario,discriminative."
        ),
    )
    parser.add_argument(
        "--rerank-source",
        type=str,
        default="scenario",
        choices=["lexical", "canonical", "scenario", "discriminative"],
        help="Which text prior to use for TODO-2 reranking and TODO-4 hard-case export.",
    )
    parser.add_argument(
        "--topk-list",
        type=str,
        default="5,10",
        help="Comma-separated candidate top-k list for reranking.",
    )
    parser.add_argument(
        "--rerank-alpha-list",
        type=str,
        default="0.05,0.1,0.2,0.3,0.5,0.7,1.0",
        help="Comma-separated alpha list for logit-space text reranking.",
    )
    parser.add_argument(
        "--rerank-modes",
        type=str,
        default="add,mix,add_norm",
        help="Comma-separated rerank fusion modes. Supported: add,mix,add_norm.",
    )
    parser.add_argument(
        "--retrieval-k-list",
        type=str,
        default="5,10,20",
        help="Comma-separated k list for retrieval priors.",
    )
    parser.add_argument(
        "--retrieval-beta-list",
        type=str,
        default="0.05,0.1,0.2,0.3",
        help="Comma-separated beta list for baseline + retrieval logit fusion.",
    )
    parser.add_argument(
        "--retrieval-priors",
        type=str,
        default="binary_vote,soft_distribution",
        help="Comma-separated retrieval prior types. Supported: binary_vote,soft_distribution.",
    )
    parser.add_argument(
        "--selective-base",
        type=str,
        default="best_rerank_macro",
        choices=["best_rerank_macro", "best_rerank_hard"],
        help="Which plain rerank config to treat as SLR-v0 when building selective variants.",
    )
    parser.add_argument(
        "--selective-prior-mode",
        type=str,
        default="auto",
        choices=["auto", "add", "add_norm"],
        help="Prior transformation used by selective rerank. 'auto' follows the selected plain rerank mode when possible.",
    )
    parser.add_argument(
        "--class-gate-modes",
        type=str,
        default="binary,continuous",
        help="Comma-separated class gate modes. Supported: binary,continuous.",
    )
    parser.add_argument(
        "--class-gate-gammas",
        type=str,
        default="4,8,12",
        help="Comma-separated gamma list for continuous class gates.",
    )
    parser.add_argument(
        "--uncertainty-modes",
        type=str,
        default="soft,binary,rank_decay",
        help="Comma-separated uncertainty gate modes. Supported: soft,binary,rank_decay.",
    )
    parser.add_argument(
        "--uncertainty-delta-list",
        type=str,
        default="0.2,0.3,0.4",
        help="Comma-separated delta list for binary uncertainty gates.",
    )
    parser.add_argument(
        "--uncertainty-tau-list",
        type=str,
        default="0.5,1.0",
        help="Comma-separated tau list for rank-decay uncertainty gates.",
    )
    parser.add_argument(
        "--positive-only-options",
        type=str,
        default="false,true",
        help="Comma-separated options for positive-only rerank. Supported values: true,false.",
    )
    parser.add_argument(
        "--semantic-sources",
        type=str,
        default="lexical,canonical,scenario,discriminative,lexical_plus_canonical",
        help="Comma-separated prompt source variants for semantic local reranking.",
    )
    parser.add_argument(
        "--semantic-aggregation-modes",
        type=str,
        default="average,max,top2_avg,logsumexp",
        help="Comma-separated prompt aggregation modes for semantic local reranking.",
    )
    parser.add_argument(
        "--comparative-modes",
        type=str,
        default="none,topk_center,topk_margin",
        help="Comma-separated top-k comparative rerank modes.",
    )
    parser.add_argument(
        "--confusion-top-n",
        type=int,
        default=12,
        help="How many confusion pairs to export per analysis block.",
    )
    parser.add_argument(
        "--hard-case-limit",
        type=int,
        default=40,
        help="How many improved/degraded hard cases to export.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-split sample cap for smoke testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV artifacts. Default: logs/analysis/<timestamp>_text_prior_boundary",
    )
    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_bool_list(raw: str) -> List[bool]:
    mapping = {"true": True, "false": False}
    values: List[bool] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise ValueError(f"Unsupported boolean token: {item}")
        values.append(mapping[token])
    return values


def _normalize_source_name(name: str) -> str:
    token = str(name).strip().lower()
    aliases = {
        "short": "lexical",
        "detailed": "canonical",
        "llm": "scenario",
        "mixed": "lexical_plus_canonical",
    }
    return aliases.get(token, token)


def _resolve_run_dir(run_dir_arg: str | None, ckpt_path_arg: str | None) -> Path:
    if run_dir_arg is not None:
        return Path(run_dir_arg)
    if ckpt_path_arg is None:
        raise ValueError("Either --run-dir or --ckpt-path must be provided.")
    ckpt_path = Path(ckpt_path_arg)
    if ckpt_path.parent.name != "checkpoints":
        raise ValueError(
            f"Cannot infer run_dir from ckpt path {ckpt_path}. Expected parent directory named checkpoints."
        )
    return ckpt_path.parent.parent


def _resolve_ckpt_path(run_dir: Path, ckpt_path_arg: str | None) -> Path:
    if ckpt_path_arg is not None:
        return Path(ckpt_path_arg)
    candidate = run_dir / "checkpoints" / "epoch_017.ckpt"
    if candidate.exists():
        return candidate
    candidate = run_dir / "checkpoints" / "last.ckpt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("ema_model."):
            continue
        new_key = key
        if ".net._orig_mod." in new_key:
            new_key = new_key.replace(".net._orig_mod.", ".net.")
        elif new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod.") :]
        normalized[new_key] = value
    return normalized


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _annotation_filename(cfg, split: str) -> str:
    key = {
        "train": "train_annotation",
        "val": "val_annotation",
        "test": "test_annotation",
    }[split]
    return str(getattr(cfg.data, key))


def _annotation_path(cfg, split: str) -> Path:
    return Path(str(cfg.data.annotation_dir)) / _annotation_filename(cfg, split)


def _load_class_names(annotation_file: Path) -> List[str]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    return [str(category["name"]) for category in categories]


def _ordered_unique(strings: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in strings:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _resolve_gemini_file(cfg, gemini_arg: str | None) -> Path:
    if gemini_arg is not None:
        path = Path(gemini_arg)
        if not path.exists():
            raise FileNotFoundError(f"Gemini description file not found: {path}")
        return path

    model_gemini = OmegaConf.select(cfg, "model.intent_gemini_file", default=None)
    if model_gemini is None:
        model_gemini = OmegaConf.select(cfg, "model.net.intent_gemini_file", default=None)
    if model_gemini:
        path = Path(str(model_gemini))
        if path.exists():
            return path

    candidate = Path(str(cfg.data.data_dir)).parent / "intent_description_gemini.json"
    if candidate.exists():
        return candidate

    fallback = PROJECT_ROOT.parent / "Intentonomy" / "data" / "intent_description_gemini.json"
    if fallback.exists():
        return fallback

    raise FileNotFoundError("Could not resolve intent_description_gemini.json.")


def _build_dataset(
    cfg,
    datamodule,
    split: str,
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[IntentonomyDataset, DataLoader, Dict[str, str]]:
    use_fixed_random_slot_perm = bool(
        OmegaConf.select(cfg, "data.use_fixed_random_slot_perm", default=False)
    )
    fixed_random_slot_perm_tokens = OmegaConf.select(
        cfg, "data.fixed_random_slot_perm_tokens", default=None
    )
    fixed_random_slot_perm_seed = int(
        OmegaConf.select(cfg, "data.fixed_random_slot_perm_seed", default=42)
    )

    dataset = IntentonomyDataset(
        annotation_file=str(_annotation_path(cfg, split)),
        image_dir=str(cfg.data.image_dir),
        transform=transform,
        binarize_softprob=bool(OmegaConf.select(cfg, "data.binarize_softprob", default=False)),
        use_fixed_random_slot_perm=use_fixed_random_slot_perm,
        fixed_random_slot_perm_tokens=fixed_random_slot_perm_tokens,
        fixed_random_slot_perm_seed=fixed_random_slot_perm_seed,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    image_id_to_path = {
        str(image_id): str(image_path) for image_id, image_path in dataset.images
    }
    return dataset, loader, image_id_to_path


def _collect_model_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    logits_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    soft_all: List[np.ndarray] = []
    image_ids_all: List[str] = []

    collected = 0
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]

            logits_cpu = logits.detach().float().cpu()
            scores_cpu = torch.sigmoid(logits_cpu)
            labels_cpu = batch["labels"].detach().float().cpu()
            soft_cpu = batch["soft_labels"].detach().float().cpu()

            image_ids = batch["image_id"]
            if torch.is_tensor(image_ids):
                image_ids_batch = [str(x) for x in image_ids.detach().cpu().tolist()]
            else:
                image_ids_batch = [str(x) for x in image_ids]

            if max_samples is not None:
                remaining = max_samples - collected
                if remaining <= 0:
                    break
                if logits_cpu.shape[0] > remaining:
                    logits_cpu = logits_cpu[:remaining]
                    scores_cpu = scores_cpu[:remaining]
                    labels_cpu = labels_cpu[:remaining]
                    soft_cpu = soft_cpu[:remaining]
                    image_ids_batch = image_ids_batch[:remaining]

            logits_all.append(logits_cpu.numpy())
            scores_all.append(scores_cpu.numpy())
            labels_all.append(labels_cpu.numpy())
            soft_all.append(soft_cpu.numpy())
            image_ids_all.extend(image_ids_batch)

            collected += len(image_ids_batch)
            if max_samples is not None and collected >= max_samples:
                break

    return {
        "logits": np.concatenate(logits_all, axis=0),
        "scores": np.concatenate(scores_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "soft_labels": np.concatenate(soft_all, axis=0),
        "image_ids": image_ids_all,
    }


def _collect_clip_features(
    clip_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    features_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    soft_all: List[np.ndarray] = []
    image_ids_all: List[str] = []

    collected = 0
    clip_model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            image_features = clip_model.encode_image(images).float()
            image_features = F.normalize(image_features, dim=-1)

            labels_cpu = batch["labels"].detach().float().cpu()
            soft_cpu = batch["soft_labels"].detach().float().cpu()
            feats_cpu = image_features.detach().cpu()

            image_ids = batch["image_id"]
            if torch.is_tensor(image_ids):
                image_ids_batch = [str(x) for x in image_ids.detach().cpu().tolist()]
            else:
                image_ids_batch = [str(x) for x in image_ids]

            if max_samples is not None:
                remaining = max_samples - collected
                if remaining <= 0:
                    break
                if feats_cpu.shape[0] > remaining:
                    feats_cpu = feats_cpu[:remaining]
                    labels_cpu = labels_cpu[:remaining]
                    soft_cpu = soft_cpu[:remaining]
                    image_ids_batch = image_ids_batch[:remaining]

            features_all.append(feats_cpu.numpy())
            labels_all.append(labels_cpu.numpy())
            soft_all.append(soft_cpu.numpy())
            image_ids_all.extend(image_ids_batch)

            collected += len(image_ids_batch)
            if max_samples is not None and collected >= max_samples:
                break

    return {
        "features": np.concatenate(features_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "soft_labels": np.concatenate(soft_all, axis=0),
        "image_ids": image_ids_all,
    }


def _assert_same_ids(name: str, left_ids: Sequence[str], right_ids: Sequence[str]) -> None:
    if list(left_ids) != list(right_ids):
        raise RuntimeError(f"{name} image order mismatch between baseline loader and CLIP loader.")


def _build_text_pools(
    class_names: Sequence[str],
    gemini_file: Path,
) -> Dict[str, List[List[str]]]:
    lexical_pools = [[phrase] for phrase in INTENTONOMY_LEXICAL_PHRASES[: len(class_names)]]
    canonical_pools = [[desc] for desc in INTENTONOMY_DESCRIPTIONS[: len(class_names)]]

    data = json.loads(gemini_file.read_text(encoding="utf-8"))
    scenario_pools: List[List[str]] = []
    discriminative_pools: List[List[str]] = []
    for index, item in enumerate(data[: len(class_names)]):
        scenario_texts: List[str] = []
        discriminative_texts: List[str] = []
        for desc in item.get("description", []):
            scenario_texts.append(str(desc.get("Text Query", "")))
            discriminative_texts.append(str(desc.get("Core Difference", "")))
        scenario_pools.append(_ordered_unique(scenario_texts))
        discriminative_pools.append(_ordered_unique(discriminative_texts))

    while len(scenario_pools) < len(class_names):
        scenario_pools.append([canonical_pools[len(scenario_pools)][0]])
    while len(discriminative_pools) < len(class_names):
        discriminative_pools.append([canonical_pools[len(discriminative_pools)][0]])

    lexical_plus_canonical_pools = []
    for idx in range(len(class_names)):
        lexical_plus_canonical_pools.append(
            _ordered_unique(
                [
                    DEFAULT_PROMPT_TEMPLATE.format(text)
                    for text in lexical_pools[idx] + canonical_pools[idx]
                ]
            )
        )

    return {
        "lexical": lexical_pools,
        "canonical": canonical_pools,
        "scenario": scenario_pools[: len(class_names)],
        "discriminative": discriminative_pools[: len(class_names)],
        "lexical_plus_canonical": lexical_plus_canonical_pools[: len(class_names)],
    }


def _encode_text_pool(
    clip_model: torch.nn.Module,
    texts_per_class: Sequence[Sequence[str]],
    wrap_prompt: bool,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> np.ndarray:
    device = next(clip_model.parameters()).device
    embeddings: List[torch.Tensor] = []
    clip_model.eval()

    with torch.inference_mode():
        for text_group in texts_per_class:
            prompts = [
                prompt_template.format(text) if wrap_prompt else str(text)
                for text in text_group
            ]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            mean_feature = text_features.mean(dim=0)
            mean_feature = F.normalize(mean_feature, dim=0)
            embeddings.append(mean_feature.detach().cpu())

    return torch.stack(embeddings, dim=0).numpy()


def _encode_text_pool_per_class(
    clip_model: torch.nn.Module,
    texts_per_class: Sequence[Sequence[str]],
    wrap_prompt: bool,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> List[np.ndarray]:
    device = next(clip_model.parameters()).device
    embeddings: List[np.ndarray] = []
    clip_model.eval()

    with torch.inference_mode():
        for text_group in texts_per_class:
            prompts = [
                prompt_template.format(text) if wrap_prompt else str(text)
                for text in text_group
            ]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            embeddings.append(text_features.detach().cpu().numpy())

    return embeddings


def _text_logits_from_features(
    image_features: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    similarity = np.asarray(image_features, dtype=np.float32) @ np.asarray(
        text_embeddings, dtype=np.float32
    ).T
    return similarity * float(logit_scale)


def _text_logits_from_prompt_embeddings(
    image_features: np.ndarray,
    prompt_embeddings_per_class: Sequence[np.ndarray],
    logit_scale: float,
    aggregation_mode: str,
) -> np.ndarray:
    image_features = np.asarray(image_features, dtype=np.float32)
    class_scores: List[np.ndarray] = []
    for prompt_embeddings in prompt_embeddings_per_class:
        prompt_embeddings = np.asarray(prompt_embeddings, dtype=np.float32)
        prompt_scores = image_features @ prompt_embeddings.T
        prompt_scores = prompt_scores * float(logit_scale)
        aggregated = aggregate_prompt_scores(prompt_scores, mode=aggregation_mode)
        class_scores.append(aggregated)
    return np.stack(class_scores, axis=1)


def _safe_logit(probabilities: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float32), eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def _metrics_for_json(metrics: Dict[str, Any], include_per_class: bool = False) -> Dict[str, Any]:
    output = {
        "macro": float(metrics["macro"]),
        "micro": float(metrics["micro"]),
        "samples": float(metrics["samples"]),
        "mAP": float(metrics["mAP"]),
        "threshold": float(metrics["threshold"]),
        "easy": float(metrics["easy"]),
        "medium": float(metrics["medium"]),
        "hard": float(metrics["hard"]),
    }
    if include_per_class:
        output["per_class_f1"] = [float(x) for x in metrics["per_class_f1"]]
    return output


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_ready(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _top_labels(scores_row: np.ndarray, class_names: Sequence[str], topk: int = 5) -> List[str]:
    top_idx = np.argsort(-scores_row)[:topk]
    return [class_names[int(idx)] for idx in top_idx]


def _label_names_from_multihot(row: np.ndarray, class_names: Sequence[str]) -> List[str]:
    return [class_names[int(idx)] for idx in np.where(row > 0)[0].tolist()]


def _build_hard_case_rows(
    *,
    image_ids: Sequence[str],
    image_id_to_path: Dict[str, str],
    targets: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    variant_scores: np.ndarray,
    variant_predictions: np.ndarray,
    text_only_scores: np.ndarray,
    class_names: Sequence[str],
    case_limit: int,
    variant_label: str = "llm_rerank",
    text_only_label: str = "llm_text_only",
) -> Dict[str, List[Dict[str, Any]]]:
    hard_ids = set(SUBSET2IDS["hard"])
    baseline_sample_f1 = compute_sample_f1_scores(targets, baseline_predictions)
    variant_sample_f1 = compute_sample_f1_scores(targets, variant_predictions)

    improved_cases: List[Dict[str, Any]] = []
    degraded_cases: List[Dict[str, Any]] = []

    for idx, image_id in enumerate(image_ids):
        gt_ids = np.where(targets[idx] > 0)[0]
        baseline_ids = np.where(baseline_predictions[idx] > 0)[0]
        variant_ids = np.where(variant_predictions[idx] > 0)[0]

        recovered_ids = [label for label in variant_ids.tolist() if targets[idx, label] > 0 and baseline_predictions[idx, label] == 0]
        dropped_true_ids = [label for label in baseline_ids.tolist() if targets[idx, label] > 0 and variant_predictions[idx, label] == 0]
        new_false_positive_ids = [label for label in variant_ids.tolist() if targets[idx, label] == 0 and baseline_predictions[idx, label] == 0]

        involves_hard = any(int(label) in hard_ids for label in gt_ids.tolist() + recovered_ids + dropped_true_ids + new_false_positive_ids)
        if not involves_hard:
            continue

        base_f1 = float(baseline_sample_f1[idx])
        variant_f1 = float(variant_sample_f1[idx])
        if abs(variant_f1 - base_f1) < 1e-8:
            continue

        record = {
            "image_id": str(image_id),
            "image_path": image_id_to_path.get(str(image_id), ""),
            "sample_f1_baseline": base_f1,
            f"sample_f1_{variant_label}": variant_f1,
            "sample_f1_delta": float(variant_f1 - base_f1),
            "ground_truth_labels": _label_names_from_multihot(targets[idx], class_names),
            "baseline_pred_labels": _label_names_from_multihot(baseline_predictions[idx], class_names),
            f"{variant_label}_pred_labels": _label_names_from_multihot(variant_predictions[idx], class_names),
            "recovered_labels": [class_names[label] for label in recovered_ids],
            "dropped_true_labels": [class_names[label] for label in dropped_true_ids],
            "new_false_positive_labels": [class_names[label] for label in new_false_positive_ids],
            "baseline_top5": _top_labels(baseline_scores[idx], class_names, topk=5),
            f"{variant_label}_top5": _top_labels(variant_scores[idx], class_names, topk=5),
            f"{text_only_label}_top5": _top_labels(text_only_scores[idx], class_names, topk=5),
        }

        if variant_f1 > base_f1 and recovered_ids:
            improved_cases.append(record)
        if variant_f1 < base_f1 and (dropped_true_ids or new_false_positive_ids):
            degraded_cases.append(record)

    improved_cases.sort(
        key=lambda item: (
            -float(item["sample_f1_delta"]),
            -len(item["recovered_labels"]),
            len(item["new_false_positive_labels"]),
        )
    )
    degraded_cases.sort(
        key=lambda item: (
            float(item["sample_f1_delta"]),
            -len(item["dropped_true_labels"]),
            -len(item["new_false_positive_labels"]),
        )
    )

    return {
        "baseline_wrong_llm_right": improved_cases[:case_limit],
        "baseline_right_llm_wrong": degraded_cases[:case_limit],
    }


def _prediction_shift_summary(
    *,
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    variant_predictions: np.ndarray,
    class_names: Sequence[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    recovered_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()
    new_fp_counter: Counter[str] = Counter()

    for idx in range(targets.shape[0]):
        recovered_ids = np.where(
            (targets[idx] == 1) & (baseline_predictions[idx] == 0) & (variant_predictions[idx] == 1)
        )[0]
        dropped_ids = np.where(
            (targets[idx] == 1) & (baseline_predictions[idx] == 1) & (variant_predictions[idx] == 0)
        )[0]
        new_fp_ids = np.where(
            (targets[idx] == 0) & (baseline_predictions[idx] == 0) & (variant_predictions[idx] == 1)
        )[0]

        recovered_counter.update(class_names[int(i)] for i in recovered_ids.tolist())
        dropped_counter.update(class_names[int(i)] for i in dropped_ids.tolist())
        new_fp_counter.update(class_names[int(i)] for i in new_fp_ids.tolist())

    baseline_empty = int(np.sum(baseline_predictions.sum(axis=1) == 0))
    variant_empty = int(np.sum(variant_predictions.sum(axis=1) == 0))

    return {
        "baseline_empty_predictions": baseline_empty,
        "variant_empty_predictions": variant_empty,
        "empty_prediction_delta": int(variant_empty - baseline_empty),
        "top_recovered_labels": [
            {"class_name": name, "count": int(count)}
            for name, count in recovered_counter.most_common(top_n)
        ],
        "top_dropped_true_labels": [
            {"class_name": name, "count": int(count)}
            for name, count in dropped_counter.most_common(top_n)
        ],
        "top_new_false_positive_labels": [
            {"class_name": name, "count": int(count)}
            for name, count in new_fp_counter.most_common(top_n)
        ],
    }


def _resolve_selective_prior_mode(selected_plain_mode: str, selective_prior_mode: str) -> str:
    if selective_prior_mode != "auto":
        return selective_prior_mode
    if selected_plain_mode in {"add", "add_norm"}:
        return selected_plain_mode
    return "add_norm"


def _build_class_gate_specs(
    class_gains: np.ndarray,
    gate_modes: Sequence[str],
    gate_gammas: Sequence[float],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for mode in gate_modes:
        if mode == "binary":
            specs.append(
                {
                    "name": "binary",
                    "mode": mode,
                    "gamma": None,
                    "gate": build_classwise_gate(class_gains, mode=mode),
                }
            )
        elif mode == "continuous":
            for gamma in gate_gammas:
                specs.append(
                    {
                        "name": f"continuous_g{gamma:g}",
                        "mode": mode,
                        "gamma": float(gamma),
                        "gate": build_classwise_gate(class_gains, mode=mode, gamma=float(gamma)),
                    }
                )
        else:
            raise ValueError(f"Unsupported class gate mode: {mode}")
    return specs


def _build_uncertainty_gate_specs(
    baseline_logits: np.ndarray,
    uncertainty_modes: Sequence[str],
    delta_list: Sequence[float],
    tau_list: Sequence[float],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for mode in uncertainty_modes:
        if mode == "soft":
            specs.append(
                {
                    "name": "soft",
                    "mode": mode,
                    "delta": None,
                    "tau": None,
                    "gate": build_uncertainty_gate(baseline_logits, mode=mode),
                }
            )
        elif mode == "binary":
            for delta in delta_list:
                specs.append(
                    {
                        "name": f"binary_d{delta:g}",
                        "mode": mode,
                        "delta": float(delta),
                        "tau": None,
                        "gate": build_uncertainty_gate(
                            baseline_logits,
                            mode=mode,
                            delta=float(delta),
                        ),
                    }
                )
        elif mode == "rank_decay":
            for tau in tau_list:
                specs.append(
                    {
                        "name": f"rank_decay_t{tau:g}",
                        "mode": mode,
                        "delta": None,
                        "tau": float(tau),
                        "gate": build_uncertainty_gate(
                            baseline_logits,
                            mode=mode,
                            tau=float(tau),
                        ),
                    }
                )
        else:
            raise ValueError(f"Unsupported uncertainty gate mode: {mode}")
    return specs


def _compute_knn_prior(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    query_features: np.ndarray,
    k: int,
    device: torch.device,
    chunk_size: int = 512,
) -> np.ndarray:
    train_features_t = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    train_targets_t = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    priors: List[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, query_features.shape[0], chunk_size):
            end = min(query_features.shape[0], start + chunk_size)
            query_t = torch.as_tensor(
                query_features[start:end], dtype=torch.float32, device=device
            )
            similarity = torch.matmul(query_t, train_features_t.T)
            topk_idx = torch.topk(
                similarity,
                k=min(int(k), train_features_t.shape[0]),
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            neighbor_targets = train_targets_t[topk_idx]
            priors.append(neighbor_targets.mean(dim=1).cpu())

    return torch.cat(priors, dim=0).numpy()


def main() -> None:
    args = _parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.ckpt_path)
    ckpt_path = _resolve_ckpt_path(run_dir, args.ckpt_path)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    cfg = OmegaConf.load(cfg_path)
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.pin_memory = bool(args.pin_memory)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_state_dict_keys(checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)

    device = _resolve_device(args.device)
    model = model.eval().to(device)

    clip_model = getattr(getattr(model, "net", model), "clip_model", None)
    clip_preprocess = getattr(getattr(model, "net", model), "clip_preprocess", None)
    if clip_model is None or clip_preprocess is None:
        clip_model_name = OmegaConf.select(cfg, "model.net.clip_model_name", default="ViT-L/14")
        clip_model, clip_preprocess = clip.load(str(clip_model_name), device=device)
    else:
        clip_model = clip_model.eval().to(device)

    clip_logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
    gemini_file = _resolve_gemini_file(cfg, args.gemini_file)
    class_names = _load_class_names(_annotation_path(cfg, "val"))

    batch_size = int(getattr(cfg.data, "batch_size", datamodule.batch_size_per_device))
    _, val_loader_base, val_id_to_path = _build_dataset(
        cfg, datamodule, "val", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, test_loader_base, test_id_to_path = _build_dataset(
        cfg, datamodule, "test", datamodule.val_test_transform, batch_size, args.num_workers, args.pin_memory
    )
    _, train_loader_clip, _ = _build_dataset(
        cfg, datamodule, "train", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )
    _, val_loader_clip, _ = _build_dataset(
        cfg, datamodule, "val", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )
    _, test_loader_clip, _ = _build_dataset(
        cfg, datamodule, "test", clip_preprocess, batch_size, args.num_workers, args.pin_memory
    )

    val_base = _collect_model_outputs(model, val_loader_base, device, max_samples=args.max_samples)
    test_base = _collect_model_outputs(model, test_loader_base, device, max_samples=args.max_samples)
    train_clip = _collect_clip_features(clip_model, train_loader_clip, device, max_samples=args.max_samples)
    val_clip = _collect_clip_features(clip_model, val_loader_clip, device, max_samples=args.max_samples)
    test_clip = _collect_clip_features(clip_model, test_loader_clip, device, max_samples=args.max_samples)

    _assert_same_ids("val", val_base["image_ids"], val_clip["image_ids"])
    _assert_same_ids("test", test_base["image_ids"], test_clip["image_ids"])

    zero_shot_sources = [_normalize_source_name(x) for x in _parse_str_list(args.text_sources)]
    rerank_modes = _parse_str_list(args.rerank_modes)
    rerank_topk_list = _parse_int_list(args.topk_list)
    rerank_alpha_list = _parse_float_list(args.rerank_alpha_list)
    retrieval_k_list = _parse_int_list(args.retrieval_k_list)
    retrieval_beta_list = _parse_float_list(args.retrieval_beta_list)
    retrieval_prior_names = _parse_str_list(args.retrieval_priors)
    class_gate_modes = _parse_str_list(args.class_gate_modes)
    class_gate_gammas = _parse_float_list(args.class_gate_gammas)
    uncertainty_modes = _parse_str_list(args.uncertainty_modes)
    uncertainty_delta_list = _parse_float_list(args.uncertainty_delta_list)
    uncertainty_tau_list = _parse_float_list(args.uncertainty_tau_list)
    positive_only_options = _parse_bool_list(args.positive_only_options)
    semantic_sources = [_normalize_source_name(x) for x in _parse_str_list(args.semantic_sources)]
    semantic_aggregation_modes = _parse_str_list(args.semantic_aggregation_modes)
    comparative_modes = _parse_str_list(args.comparative_modes)

    baseline_metrics = evaluate_with_validation_threshold(
        val_base["scores"],
        val_base["labels"],
        test_base["scores"],
        test_base["labels"],
        use_inference_strategy=args.use_inference_strategy,
    )
    baseline_predictions_test = threshold_predictions(
        test_base["scores"],
        threshold=float(baseline_metrics["val"]["threshold"]),
        use_inference_strategy=args.use_inference_strategy,
    )

    text_pools = _build_text_pools(class_names, gemini_file)
    text_embeddings = {
        "lexical": _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True),
        "canonical": _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True),
        "scenario": _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False),
        "discriminative": _encode_text_pool(clip_model, text_pools["discriminative"], wrap_prompt=False),
        "lexical_plus_canonical": _encode_text_pool(
            clip_model, text_pools["lexical_plus_canonical"], wrap_prompt=False
        ),
    }
    prompt_embeddings_per_class = {
        "lexical": _encode_text_pool_per_class(clip_model, text_pools["lexical"], wrap_prompt=True),
        "canonical": _encode_text_pool_per_class(clip_model, text_pools["canonical"], wrap_prompt=True),
        "scenario": _encode_text_pool_per_class(clip_model, text_pools["scenario"], wrap_prompt=False),
        "discriminative": _encode_text_pool_per_class(
            clip_model, text_pools["discriminative"], wrap_prompt=False
        ),
        "lexical_plus_canonical": _encode_text_pool_per_class(
            clip_model, text_pools["lexical_plus_canonical"], wrap_prompt=False
        ),
    }

    text_logits = {}
    text_scores = {}
    zero_shot_results: Dict[str, Any] = {}
    for source_name in zero_shot_sources:
        source_embeddings = text_embeddings[source_name]
        val_text_logits = _text_logits_from_features(
            val_clip["features"], source_embeddings, clip_logit_scale
        )
        test_text_logits = _text_logits_from_features(
            test_clip["features"], source_embeddings, clip_logit_scale
        )
        val_text_scores = 1.0 / (1.0 + np.exp(-val_text_logits))
        test_text_scores = 1.0 / (1.0 + np.exp(-test_text_logits))

        source_metrics = evaluate_with_validation_threshold(
            val_text_scores,
            val_clip["labels"],
            test_text_scores,
            test_clip["labels"],
            use_inference_strategy=args.use_inference_strategy,
        )
        text_logits[source_name] = {"val": val_text_logits, "test": test_text_logits}
        text_scores[source_name] = {"val": val_text_scores, "test": test_text_scores}

        test_predictions = threshold_predictions(
            test_text_scores,
            threshold=source_metrics["val"]["threshold"],
            use_inference_strategy=args.use_inference_strategy,
        )
        zero_shot_results[source_name] = {
            "val": _metrics_for_json(source_metrics["val"], include_per_class=True),
            "test": _metrics_for_json(source_metrics["test"], include_per_class=True),
            "test_top_confusions": build_confusion_pairs(
                test_clip["labels"],
                test_predictions,
                class_names,
                top_n=args.confusion_top_n,
                focus_class_ids=None,
            ),
            "test_top_hard_confusions": build_confusion_pairs(
                test_clip["labels"],
                test_predictions,
                class_names,
                top_n=args.confusion_top_n,
                focus_class_ids=SUBSET2IDS["hard"],
            ),
        }

    zero_shot_comparisons = {}
    if "lexical" in zero_shot_sources:
        for source_name in zero_shot_sources:
            if source_name == "lexical":
                continue
            zero_shot_comparisons[f"{source_name}_minus_lexical"] = {
                "val_macro_gain": float(
                    zero_shot_results[source_name]["val"]["macro"] - zero_shot_results["lexical"]["val"]["macro"]
                ),
                "val_hard_gain": float(
                    zero_shot_results[source_name]["val"]["hard"] - zero_shot_results["lexical"]["val"]["hard"]
                ),
                "test_macro_gain": float(
                    zero_shot_results[source_name]["test"]["macro"] - zero_shot_results["lexical"]["test"]["macro"]
                ),
                "test_hard_gain": float(
                    zero_shot_results[source_name]["test"]["hard"] - zero_shot_results["lexical"]["test"]["hard"]
                ),
                "test_top_gain_classes": class_gain_rows(
                    np.asarray(zero_shot_results["lexical"]["test"]["per_class_f1"], dtype=np.float32),
                    np.asarray(zero_shot_results[source_name]["test"]["per_class_f1"], dtype=np.float32),
                    class_names,
                    top_n=8,
                ),
            }

    rerank_source = _normalize_source_name(args.rerank_source)
    rerank_records_internal: List[Dict[str, Any]] = []
    baseline_test_per_class = baseline_metrics["test"]["per_class_f1"]
    for topk in rerank_topk_list:
        for mode in rerank_modes:
            for alpha in rerank_alpha_list:
                if mode == "mix" and not (0.0 <= alpha <= 1.0):
                    continue

                val_rerank_logits = apply_topk_rerank_fusion(
                    val_base["logits"],
                    text_logits[rerank_source]["val"],
                    topk=topk,
                    alpha=alpha,
                    mode=mode,
                )
                test_rerank_logits = apply_topk_rerank_fusion(
                    test_base["logits"],
                    text_logits[rerank_source]["test"],
                    topk=topk,
                    alpha=alpha,
                    mode=mode,
                )
                val_rerank_scores = 1.0 / (1.0 + np.exp(-val_rerank_logits))
                test_rerank_scores = 1.0 / (1.0 + np.exp(-test_rerank_logits))

                rerank_metrics = evaluate_with_validation_threshold(
                    val_rerank_scores,
                    val_base["labels"],
                    test_rerank_scores,
                    test_base["labels"],
                    use_inference_strategy=args.use_inference_strategy,
                )
                rerank_records_internal.append(
                    {
                        "topk": int(topk),
                        "mode": mode,
                        "alpha": float(alpha),
                        "val": rerank_metrics["val"],
                        "test": rerank_metrics["test"],
                        "test_scores": test_rerank_scores,
                    }
                )

    rerank_records_internal.sort(
        key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
        reverse=True,
    )
    best_rerank_macro = rerank_records_internal[0]
    best_rerank_hard = max(rerank_records_internal, key=lambda item: float(item["val"]["hard"]))

    rerank_leaderboard = []
    for record in rerank_records_internal[: min(20, len(rerank_records_internal))]:
        rerank_leaderboard.append(
            {
                "topk": int(record["topk"]),
                "mode": record["mode"],
                "alpha": float(record["alpha"]),
                "val_macro": float(record["val"]["macro"]),
                "val_hard": float(record["val"]["hard"]),
                "test_macro": float(record["test"]["macro"]),
                "test_hard": float(record["test"]["hard"]),
                "test_easy": float(record["test"]["easy"]),
                "test_medium": float(record["test"]["medium"]),
            }
        )

    semantic_local_records_internal: List[Dict[str, Any]] = []
    semantic_topk = int(best_rerank_macro["topk"])
    semantic_alpha = float(best_rerank_macro["alpha"])
    semantic_base_mode = str(best_rerank_macro["mode"])
    for source_name in semantic_sources:
        if source_name not in prompt_embeddings_per_class:
            raise ValueError(f"Unsupported semantic source: {source_name}")
        for aggregation_mode in semantic_aggregation_modes:
            val_sem_logits = _text_logits_from_prompt_embeddings(
                val_clip["features"],
                prompt_embeddings_per_class[source_name],
                clip_logit_scale,
                aggregation_mode=aggregation_mode,
            )
            test_sem_logits = _text_logits_from_prompt_embeddings(
                test_clip["features"],
                prompt_embeddings_per_class[source_name],
                clip_logit_scale,
                aggregation_mode=aggregation_mode,
            )
            for comparative_mode in comparative_modes:
                if comparative_mode == "none":
                    val_sem_rerank_logits = apply_topk_rerank_fusion(
                        val_base["logits"],
                        val_sem_logits,
                        topk=semantic_topk,
                        alpha=semantic_alpha,
                        mode=semantic_base_mode,
                    )
                    test_sem_rerank_logits = apply_topk_rerank_fusion(
                        test_base["logits"],
                        test_sem_logits,
                        topk=semantic_topk,
                        alpha=semantic_alpha,
                        mode=semantic_base_mode,
                    )
                else:
                    if semantic_base_mode == "add_norm":
                        val_prior_base = normalize_scores_per_sample(val_sem_logits)
                        test_prior_base = normalize_scores_per_sample(test_sem_logits)
                    else:
                        val_prior_base = val_sem_logits
                        test_prior_base = test_sem_logits

                    val_comparative = build_topk_comparative_prior(
                        val_base["logits"],
                        val_prior_base,
                        topk=semantic_topk,
                        mode=comparative_mode,
                    )
                    test_comparative = build_topk_comparative_prior(
                        test_base["logits"],
                        test_prior_base,
                        topk=semantic_topk,
                        mode=comparative_mode,
                    )
                    val_sem_rerank_logits = apply_topk_rerank_fusion(
                        val_base["logits"],
                        val_comparative,
                        topk=semantic_topk,
                        alpha=semantic_alpha,
                        mode="add",
                    )
                    test_sem_rerank_logits = apply_topk_rerank_fusion(
                        test_base["logits"],
                        test_comparative,
                        topk=semantic_topk,
                        alpha=semantic_alpha,
                        mode="add",
                    )

                val_sem_scores = 1.0 / (1.0 + np.exp(-val_sem_rerank_logits))
                test_sem_scores = 1.0 / (1.0 + np.exp(-test_sem_rerank_logits))
                sem_metrics = evaluate_with_validation_threshold(
                    val_sem_scores,
                    val_base["labels"],
                    test_sem_scores,
                    test_base["labels"],
                    use_inference_strategy=args.use_inference_strategy,
                )
                semantic_local_records_internal.append(
                    {
                        "source": source_name,
                        "aggregation_mode": aggregation_mode,
                        "comparative_mode": comparative_mode,
                        "topk": semantic_topk,
                        "alpha": semantic_alpha,
                        "base_mode": semantic_base_mode,
                        "val": sem_metrics["val"],
                        "test": sem_metrics["test"],
                    }
                )

    semantic_local_records_internal.sort(
        key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
        reverse=True,
    )
    best_semantic_local_macro = semantic_local_records_internal[0]
    best_semantic_local_hard = max(
        semantic_local_records_internal, key=lambda item: float(item["val"]["hard"])
    )
    semantic_local_leaderboard = []
    for record in semantic_local_records_internal[: min(30, len(semantic_local_records_internal))]:
        semantic_local_leaderboard.append(
            {
                "source": record["source"],
                "aggregation_mode": record["aggregation_mode"],
                "comparative_mode": record["comparative_mode"],
                "val_macro": float(record["val"]["macro"]),
                "val_hard": float(record["val"]["hard"]),
                "test_macro": float(record["test"]["macro"]),
                "test_hard": float(record["test"]["hard"]),
                "test_easy": float(record["test"]["easy"]),
                "test_medium": float(record["test"]["medium"]),
            }
        )

    semantic_best_by_source: Dict[str, Dict[str, Any]] = {}
    for source_name in semantic_sources:
        source_records = [row for row in semantic_local_records_internal if row["source"] == source_name]
        if source_records:
            semantic_best_by_source[source_name] = max(
                source_records,
                key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
            )

    semantic_best_by_aggregation: Dict[str, Dict[str, Any]] = {}
    for aggregation_mode in semantic_aggregation_modes:
        agg_records = [row for row in semantic_local_records_internal if row["aggregation_mode"] == aggregation_mode]
        if agg_records:
            semantic_best_by_aggregation[aggregation_mode] = max(
                agg_records,
                key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
            )

    semantic_best_by_comparative: Dict[str, Dict[str, Any]] = {}
    for comparative_mode in comparative_modes:
        comp_records = [row for row in semantic_local_records_internal if row["comparative_mode"] == comparative_mode]
        if comp_records:
            semantic_best_by_comparative[comparative_mode] = max(
                comp_records,
                key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
            )

    selective_plain_record = (
        best_rerank_macro if args.selective_base == "best_rerank_macro" else best_rerank_hard
    )
    selective_prior_mode = _resolve_selective_prior_mode(
        str(selective_plain_record["mode"]),
        args.selective_prior_mode,
    )
    val_class_gains = compute_class_gains(
        baseline_metrics["val"]["per_class_f1"],
        selective_plain_record["val"]["per_class_f1"],
    )
    class_gate_specs = _build_class_gate_specs(
        class_gains=val_class_gains,
        gate_modes=class_gate_modes,
        gate_gammas=class_gate_gammas,
    )
    val_uncertainty_specs = _build_uncertainty_gate_specs(
        baseline_logits=val_base["logits"],
        uncertainty_modes=uncertainty_modes,
        delta_list=uncertainty_delta_list,
        tau_list=uncertainty_tau_list,
    )
    test_uncertainty_by_name = {
        spec["name"]: spec["gate"]
        for spec in _build_uncertainty_gate_specs(
            baseline_logits=test_base["logits"],
            uncertainty_modes=uncertainty_modes,
            delta_list=uncertainty_delta_list,
            tau_list=uncertainty_tau_list,
        )
    }

    selective_records_internal: List[Dict[str, Any]] = []

    for class_spec in class_gate_specs:
        val_logits = apply_selective_topk_rerank(
            baseline_logits=val_base["logits"],
            prior_scores=text_logits[rerank_source]["val"],
            topk=int(selective_plain_record["topk"]),
            alpha=float(selective_plain_record["alpha"]),
            prior_mode=selective_prior_mode,
            class_gate=class_spec["gate"],
            uncertainty_gate=None,
            positive_only=False,
        )
        test_logits = apply_selective_topk_rerank(
            baseline_logits=test_base["logits"],
            prior_scores=text_logits[rerank_source]["test"],
            topk=int(selective_plain_record["topk"]),
            alpha=float(selective_plain_record["alpha"]),
            prior_mode=selective_prior_mode,
            class_gate=class_spec["gate"],
            uncertainty_gate=None,
            positive_only=False,
        )
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))
        test_scores_variant = 1.0 / (1.0 + np.exp(-test_logits))
        metrics = evaluate_with_validation_threshold(
            val_scores,
            val_base["labels"],
            test_scores_variant,
            test_base["labels"],
            use_inference_strategy=args.use_inference_strategy,
        )
        selective_records_internal.append(
            {
                "variant": "slr_v1",
                "class_gate_name": class_spec["name"],
                "class_gate_mode": class_spec["mode"],
                "class_gate_gamma": class_spec["gamma"],
                "uncertainty_name": "none",
                "uncertainty_mode": "none",
                "uncertainty_delta": None,
                "uncertainty_tau": None,
                "positive_only": False,
                "val": metrics["val"],
                "test": metrics["test"],
                "test_scores": test_scores_variant,
            }
        )

    for uncertainty_spec in val_uncertainty_specs:
        val_logits = apply_selective_topk_rerank(
            baseline_logits=val_base["logits"],
            prior_scores=text_logits[rerank_source]["val"],
            topk=int(selective_plain_record["topk"]),
            alpha=float(selective_plain_record["alpha"]),
            prior_mode=selective_prior_mode,
            class_gate=None,
            uncertainty_gate=uncertainty_spec["gate"],
            positive_only=False,
        )
        test_logits = apply_selective_topk_rerank(
            baseline_logits=test_base["logits"],
            prior_scores=text_logits[rerank_source]["test"],
            topk=int(selective_plain_record["topk"]),
            alpha=float(selective_plain_record["alpha"]),
            prior_mode=selective_prior_mode,
            class_gate=None,
            uncertainty_gate=test_uncertainty_by_name[uncertainty_spec["name"]],
            positive_only=False,
        )
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))
        test_scores_variant = 1.0 / (1.0 + np.exp(-test_logits))
        metrics = evaluate_with_validation_threshold(
            val_scores,
            val_base["labels"],
            test_scores_variant,
            test_base["labels"],
            use_inference_strategy=args.use_inference_strategy,
        )
        selective_records_internal.append(
            {
                "variant": "slr_v2",
                "class_gate_name": "none",
                "class_gate_mode": "none",
                "class_gate_gamma": None,
                "uncertainty_name": uncertainty_spec["name"],
                "uncertainty_mode": uncertainty_spec["mode"],
                "uncertainty_delta": uncertainty_spec["delta"],
                "uncertainty_tau": uncertainty_spec["tau"],
                "positive_only": False,
                "val": metrics["val"],
                "test": metrics["test"],
                "test_scores": test_scores_variant,
            }
        )

    for class_spec in class_gate_specs:
        for uncertainty_spec in val_uncertainty_specs:
            for positive_only in positive_only_options:
                variant_name = "slr_v4" if positive_only else "slr_v3"
                val_logits = apply_selective_topk_rerank(
                    baseline_logits=val_base["logits"],
                    prior_scores=text_logits[rerank_source]["val"],
                    topk=int(selective_plain_record["topk"]),
                    alpha=float(selective_plain_record["alpha"]),
                    prior_mode=selective_prior_mode,
                    class_gate=class_spec["gate"],
                    uncertainty_gate=uncertainty_spec["gate"],
                    positive_only=positive_only,
                )
                test_logits = apply_selective_topk_rerank(
                    baseline_logits=test_base["logits"],
                    prior_scores=text_logits[rerank_source]["test"],
                    topk=int(selective_plain_record["topk"]),
                    alpha=float(selective_plain_record["alpha"]),
                    prior_mode=selective_prior_mode,
                    class_gate=class_spec["gate"],
                    uncertainty_gate=test_uncertainty_by_name[uncertainty_spec["name"]],
                    positive_only=positive_only,
                )
                val_scores = 1.0 / (1.0 + np.exp(-val_logits))
                test_scores_variant = 1.0 / (1.0 + np.exp(-test_logits))
                metrics = evaluate_with_validation_threshold(
                    val_scores,
                    val_base["labels"],
                    test_scores_variant,
                    test_base["labels"],
                    use_inference_strategy=args.use_inference_strategy,
                )
                selective_records_internal.append(
                    {
                        "variant": variant_name,
                        "class_gate_name": class_spec["name"],
                        "class_gate_mode": class_spec["mode"],
                        "class_gate_gamma": class_spec["gamma"],
                        "uncertainty_name": uncertainty_spec["name"],
                        "uncertainty_mode": uncertainty_spec["mode"],
                        "uncertainty_delta": uncertainty_spec["delta"],
                        "uncertainty_tau": uncertainty_spec["tau"],
                        "positive_only": bool(positive_only),
                        "val": metrics["val"],
                        "test": metrics["test"],
                        "test_scores": test_scores_variant,
                    }
                )

    selective_records_internal.sort(
        key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
        reverse=True,
    )
    best_selective_macro = selective_records_internal[0]
    best_selective_hard = max(
        selective_records_internal, key=lambda item: float(item["val"]["hard"])
    )
    best_selective_by_variant: Dict[str, Dict[str, Any]] = {}
    for variant_name in ["slr_v1", "slr_v2", "slr_v3", "slr_v4"]:
        variant_records = [row for row in selective_records_internal if row["variant"] == variant_name]
        if variant_records:
            best_selective_by_variant[variant_name] = max(
                variant_records,
                key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
            )

    selective_leaderboard = []
    for record in selective_records_internal[: min(30, len(selective_records_internal))]:
        selective_leaderboard.append(
            {
                "variant": record["variant"],
                "class_gate_name": record["class_gate_name"],
                "uncertainty_name": record["uncertainty_name"],
                "positive_only": bool(record["positive_only"]),
                "val_macro": float(record["val"]["macro"]),
                "val_hard": float(record["val"]["hard"]),
                "test_macro": float(record["test"]["macro"]),
                "test_hard": float(record["test"]["hard"]),
                "test_easy": float(record["test"]["easy"]),
                "test_medium": float(record["test"]["medium"]),
            }
        )

    retrieval_priors = {
        "binary_vote": (train_clip["soft_labels"] > 0.0).astype(np.float32),
        "soft_distribution": np.asarray(train_clip["soft_labels"], dtype=np.float32),
    }

    retrieval_prior_results: Dict[str, Any] = {}
    retrieval_records_internal: List[Dict[str, Any]] = []
    for prior_name in retrieval_prior_names:
        if prior_name not in retrieval_priors:
            raise ValueError(f"Unsupported retrieval prior type: {prior_name}")

        prior_targets = retrieval_priors[prior_name]
        prior_only_rows = []
        for k in retrieval_k_list:
            val_prior = _compute_knn_prior(
                train_clip["features"],
                prior_targets,
                val_clip["features"],
                k=k,
                device=device,
            )
            test_prior = _compute_knn_prior(
                train_clip["features"],
                prior_targets,
                test_clip["features"],
                k=k,
                device=device,
            )
            prior_only_metrics = evaluate_with_validation_threshold(
                val_prior,
                val_clip["labels"],
                test_prior,
                test_clip["labels"],
                use_inference_strategy=args.use_inference_strategy,
            )
            prior_only_rows.append(
                {
                    "prior_name": prior_name,
                    "k": int(k),
                    "val_macro": float(prior_only_metrics["val"]["macro"]),
                    "val_hard": float(prior_only_metrics["val"]["hard"]),
                    "test_macro": float(prior_only_metrics["test"]["macro"]),
                    "test_hard": float(prior_only_metrics["test"]["hard"]),
                }
            )

            prior_logits_val = _safe_logit(val_prior)
            prior_logits_test = _safe_logit(test_prior)
            for beta in retrieval_beta_list:
                fused_val_scores = 1.0 / (
                    1.0 + np.exp(-(val_base["logits"] + float(beta) * prior_logits_val))
                )
                fused_test_scores = 1.0 / (
                    1.0 + np.exp(-(test_base["logits"] + float(beta) * prior_logits_test))
                )
                fused_metrics = evaluate_with_validation_threshold(
                    fused_val_scores,
                    val_base["labels"],
                    fused_test_scores,
                    test_base["labels"],
                    use_inference_strategy=args.use_inference_strategy,
                )
                retrieval_records_internal.append(
                    {
                        "prior_name": prior_name,
                        "k": int(k),
                        "beta": float(beta),
                        "val": fused_metrics["val"],
                        "test": fused_metrics["test"],
                    }
                )

        retrieval_prior_results[prior_name] = {
            "prior_only": prior_only_rows,
        }

    retrieval_records_internal.sort(
        key=lambda item: (float(item["val"]["macro"]), float(item["val"]["hard"])),
        reverse=True,
    )
    best_retrieval_macro = retrieval_records_internal[0]
    best_retrieval_hard = max(
        retrieval_records_internal, key=lambda item: float(item["val"]["hard"])
    )

    retrieval_leaderboard = []
    for record in retrieval_records_internal[: min(20, len(retrieval_records_internal))]:
        retrieval_leaderboard.append(
            {
                "prior_name": record["prior_name"],
                "k": int(record["k"]),
                "beta": float(record["beta"]),
                "val_macro": float(record["val"]["macro"]),
                "val_hard": float(record["val"]["hard"]),
                "test_macro": float(record["test"]["macro"]),
                "test_hard": float(record["test"]["hard"]),
                "test_easy": float(record["test"]["easy"]),
                "test_medium": float(record["test"]["medium"]),
            }
        )

    best_llm_predictions_test = threshold_predictions(
        best_rerank_macro["test_scores"],
        threshold=float(best_rerank_macro["val"]["threshold"]),
        use_inference_strategy=args.use_inference_strategy,
    )
    hard_case_rows = _build_hard_case_rows(
        image_ids=test_base["image_ids"],
        image_id_to_path=test_id_to_path,
        targets=test_base["labels"],
        baseline_scores=test_base["scores"],
        baseline_predictions=baseline_predictions_test,
        variant_scores=best_rerank_macro["test_scores"],
        variant_predictions=best_llm_predictions_test,
        text_only_scores=text_scores[rerank_source]["test"],
        class_names=class_names,
        case_limit=args.hard_case_limit,
        variant_label="llm_rerank",
        text_only_label="llm_text_only",
    )
    best_selective_predictions_test = threshold_predictions(
        best_selective_macro["test_scores"],
        threshold=float(best_selective_macro["val"]["threshold"]),
        use_inference_strategy=args.use_inference_strategy,
    )
    selective_hard_case_rows_raw = _build_hard_case_rows(
        image_ids=test_base["image_ids"],
        image_id_to_path=test_id_to_path,
        targets=test_base["labels"],
        baseline_scores=test_base["scores"],
        baseline_predictions=baseline_predictions_test,
        variant_scores=best_selective_macro["test_scores"],
        variant_predictions=best_selective_predictions_test,
        text_only_scores=text_scores[rerank_source]["test"],
        class_names=class_names,
        case_limit=args.hard_case_limit,
        variant_label="selective_rerank",
        text_only_label="llm_text_only",
    )
    selective_hard_case_rows = {
        "baseline_wrong_selective_right": selective_hard_case_rows_raw["baseline_wrong_llm_right"],
        "baseline_right_selective_wrong": selective_hard_case_rows_raw["baseline_right_llm_wrong"],
    }
    plain_prediction_shift = _prediction_shift_summary(
        targets=test_base["labels"],
        baseline_predictions=baseline_predictions_test,
        variant_predictions=best_llm_predictions_test,
        class_names=class_names,
        top_n=args.confusion_top_n,
    )
    selective_prediction_shift = _prediction_shift_summary(
        targets=test_base["labels"],
        baseline_predictions=baseline_predictions_test,
        variant_predictions=best_selective_predictions_test,
        class_names=class_names,
        top_n=args.confusion_top_n,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "logs" / "analysis" / f"{timestamp}_text_prior_boundary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "rerank_leaderboard.csv", rerank_leaderboard)
    _write_csv(output_dir / "semantic_local_rerank_leaderboard.csv", semantic_local_leaderboard)
    _write_csv(output_dir / "selective_rerank_leaderboard.csv", selective_leaderboard)
    _write_csv(output_dir / "retrieval_leaderboard.csv", retrieval_leaderboard)
    _write_csv(
        output_dir / "hard_cases_baseline_wrong_llm_right.csv",
        [
            {
                "image_id": row["image_id"],
                "sample_f1_delta": row["sample_f1_delta"],
                "recovered_labels": "|".join(row["recovered_labels"]),
                "ground_truth_labels": "|".join(row["ground_truth_labels"]),
                "baseline_pred_labels": "|".join(row["baseline_pred_labels"]),
                "llm_rerank_pred_labels": "|".join(row["llm_rerank_pred_labels"]),
            }
            for row in hard_case_rows["baseline_wrong_llm_right"]
        ],
    )
    _write_csv(
        output_dir / "hard_cases_baseline_right_llm_wrong.csv",
        [
            {
                "image_id": row["image_id"],
                "sample_f1_delta": row["sample_f1_delta"],
                "dropped_true_labels": "|".join(row["dropped_true_labels"]),
                "new_false_positive_labels": "|".join(row["new_false_positive_labels"]),
                "ground_truth_labels": "|".join(row["ground_truth_labels"]),
                "baseline_pred_labels": "|".join(row["baseline_pred_labels"]),
                "llm_rerank_pred_labels": "|".join(row["llm_rerank_pred_labels"]),
            }
            for row in hard_case_rows["baseline_right_llm_wrong"]
        ],
    )
    (output_dir / "hard_cases_baseline_wrong_llm_right.json").write_text(
        json.dumps(_json_ready(hard_case_rows["baseline_wrong_llm_right"]), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "hard_cases_baseline_right_llm_wrong.json").write_text(
        json.dumps(_json_ready(hard_case_rows["baseline_right_llm_wrong"]), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "hard_cases_baseline_wrong_selective_right.csv",
        [
            {
                "image_id": row["image_id"],
                "sample_f1_delta": row["sample_f1_delta"],
                "recovered_labels": "|".join(row["recovered_labels"]),
                "ground_truth_labels": "|".join(row["ground_truth_labels"]),
                "baseline_pred_labels": "|".join(row["baseline_pred_labels"]),
                "selective_rerank_pred_labels": "|".join(row["selective_rerank_pred_labels"]),
            }
            for row in selective_hard_case_rows["baseline_wrong_selective_right"]
        ],
    )
    _write_csv(
        output_dir / "hard_cases_baseline_right_selective_wrong.csv",
        [
            {
                "image_id": row["image_id"],
                "sample_f1_delta": row["sample_f1_delta"],
                "dropped_true_labels": "|".join(row["dropped_true_labels"]),
                "new_false_positive_labels": "|".join(row["new_false_positive_labels"]),
                "ground_truth_labels": "|".join(row["ground_truth_labels"]),
                "baseline_pred_labels": "|".join(row["baseline_pred_labels"]),
                "selective_rerank_pred_labels": "|".join(row["selective_rerank_pred_labels"]),
            }
            for row in selective_hard_case_rows["baseline_right_selective_wrong"]
        ],
    )
    (output_dir / "hard_cases_baseline_wrong_selective_right.json").write_text(
        json.dumps(
            _json_ready(selective_hard_case_rows["baseline_wrong_selective_right"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "hard_cases_baseline_right_selective_wrong.json").write_text(
        json.dumps(
            _json_ready(selective_hard_case_rows["baseline_right_selective_wrong"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = {
        "metadata": {
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "gemini_file": str(gemini_file),
            "device": str(device),
            "clip_logit_scale": clip_logit_scale,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "text_sources": zero_shot_sources,
            "rerank_source": rerank_source,
            "selective_base": args.selective_base,
            "selective_prior_mode": selective_prior_mode,
            "max_samples": args.max_samples,
            "output_dir": str(output_dir),
        },
        "definitions": {
            "zero_shot": "Use frozen CLIP image features against text embeddings from short label names, detailed intent definitions, or LLM abstract+visual descriptions.",
            "rerank": "Apply text-prior logit fusion only on the baseline top-k candidate classes, then threshold with a validation-tuned global threshold.",
            "selective_rerank": "Build class-wise gates from validation per-class gains, uncertainty gates from baseline logits, then apply selective top-k additive rerank with optional positive-only prior clipping.",
            "retrieval_prior": "Use train-set CLIP kNN neighbors to form binary-vote or soft-distribution priors, and add their safe-logit to the baseline logits with a small weight.",
            "hard_case": "Focus on test samples involving hard classes; export cases where LLM rerank recovers baseline misses and cases where it hurts previously correct predictions.",
        },
        "split_sizes": {
            "train": int(train_clip["features"].shape[0]),
            "val": int(val_base["scores"].shape[0]),
            "test": int(test_base["scores"].shape[0]),
        },
        "baseline": {
            "val": _metrics_for_json(baseline_metrics["val"], include_per_class=True),
            "test": _metrics_for_json(baseline_metrics["test"], include_per_class=True),
        },
        "zero_shot": {
            "sources": zero_shot_results,
            "comparisons": zero_shot_comparisons,
        },
        "rerank": {
            "leaderboard_top20": rerank_leaderboard,
            "best_by_val_macro": {
                "config": {
                    "topk": int(best_rerank_macro["topk"]),
                    "mode": best_rerank_macro["mode"],
                    "alpha": float(best_rerank_macro["alpha"]),
                },
                "val": _metrics_for_json(best_rerank_macro["val"], include_per_class=True),
                "test": _metrics_for_json(best_rerank_macro["test"], include_per_class=True),
                "test_gain_over_baseline": {
                    "macro": float(best_rerank_macro["test"]["macro"] - baseline_metrics["test"]["macro"]),
                    "hard": float(best_rerank_macro["test"]["hard"] - baseline_metrics["test"]["hard"]),
                },
                "top_gain_classes": class_gain_rows(
                    baseline_test_per_class,
                    best_rerank_macro["test"]["per_class_f1"],
                    class_names,
                    top_n=10,
                ),
            },
            "best_by_val_hard": {
                "config": {
                    "topk": int(best_rerank_hard["topk"]),
                    "mode": best_rerank_hard["mode"],
                    "alpha": float(best_rerank_hard["alpha"]),
                },
                "val": _metrics_for_json(best_rerank_hard["val"]),
                "test": _metrics_for_json(best_rerank_hard["test"]),
            },
            "prediction_shift_vs_baseline": plain_prediction_shift,
        },
        "semantic_local_rerank": {
            "base_plain_rerank": {
                "config": {
                    "source": rerank_source,
                    "topk": semantic_topk,
                    "base_mode": semantic_base_mode,
                    "alpha": semantic_alpha,
                },
                "val": _metrics_for_json(best_rerank_macro["val"], include_per_class=True),
                "test": _metrics_for_json(best_rerank_macro["test"], include_per_class=True),
            },
            "leaderboard_top30": semantic_local_leaderboard,
            "best_overall_by_val_macro": {
                "config": {
                    "source": best_semantic_local_macro["source"],
                    "aggregation_mode": best_semantic_local_macro["aggregation_mode"],
                    "comparative_mode": best_semantic_local_macro["comparative_mode"],
                    "topk": int(best_semantic_local_macro["topk"]),
                    "alpha": float(best_semantic_local_macro["alpha"]),
                    "base_mode": best_semantic_local_macro["base_mode"],
                },
                "val": _metrics_for_json(best_semantic_local_macro["val"], include_per_class=True),
                "test": _metrics_for_json(best_semantic_local_macro["test"], include_per_class=True),
                "test_gain_over_plain_rerank": {
                    "macro": float(best_semantic_local_macro["test"]["macro"] - best_rerank_macro["test"]["macro"]),
                    "hard": float(best_semantic_local_macro["test"]["hard"] - best_rerank_macro["test"]["hard"]),
                },
            },
            "best_overall_by_val_hard": {
                "config": {
                    "source": best_semantic_local_hard["source"],
                    "aggregation_mode": best_semantic_local_hard["aggregation_mode"],
                    "comparative_mode": best_semantic_local_hard["comparative_mode"],
                    "topk": int(best_semantic_local_hard["topk"]),
                    "alpha": float(best_semantic_local_hard["alpha"]),
                    "base_mode": best_semantic_local_hard["base_mode"],
                },
                "val": _metrics_for_json(best_semantic_local_hard["val"]),
                "test": _metrics_for_json(best_semantic_local_hard["test"]),
                "test_gain_over_plain_rerank": {
                    "macro": float(best_semantic_local_hard["test"]["macro"] - best_rerank_macro["test"]["macro"]),
                    "hard": float(best_semantic_local_hard["test"]["hard"] - best_rerank_macro["test"]["hard"]),
                },
            },
            "best_by_source": {
                source_name: {
                    "config": {
                        "aggregation_mode": record["aggregation_mode"],
                        "comparative_mode": record["comparative_mode"],
                        "topk": int(record["topk"]),
                        "alpha": float(record["alpha"]),
                        "base_mode": record["base_mode"],
                    },
                    "val": _metrics_for_json(record["val"]),
                    "test": _metrics_for_json(record["test"]),
                }
                for source_name, record in semantic_best_by_source.items()
            },
            "best_by_aggregation": {
                agg_mode: {
                    "config": {
                        "source": record["source"],
                        "comparative_mode": record["comparative_mode"],
                        "topk": int(record["topk"]),
                        "alpha": float(record["alpha"]),
                        "base_mode": record["base_mode"],
                    },
                    "val": _metrics_for_json(record["val"]),
                    "test": _metrics_for_json(record["test"]),
                }
                for agg_mode, record in semantic_best_by_aggregation.items()
            },
            "best_by_comparative_mode": {
                comp_mode: {
                    "config": {
                        "source": record["source"],
                        "aggregation_mode": record["aggregation_mode"],
                        "topk": int(record["topk"]),
                        "alpha": float(record["alpha"]),
                        "base_mode": record["base_mode"],
                    },
                    "val": _metrics_for_json(record["val"]),
                    "test": _metrics_for_json(record["test"]),
                }
                for comp_mode, record in semantic_best_by_comparative.items()
            },
        },
        "selective_rerank": {
            "base_plain_rerank": {
                "selection": args.selective_base,
                "config": {
                    "topk": int(selective_plain_record["topk"]),
                    "mode": str(selective_plain_record["mode"]),
                    "alpha": float(selective_plain_record["alpha"]),
                    "prior_mode": selective_prior_mode,
                },
                "val": _metrics_for_json(selective_plain_record["val"], include_per_class=True),
                "test": _metrics_for_json(selective_plain_record["test"], include_per_class=True),
            },
            "class_gain_source": {
                "top_positive_gain_classes": class_gain_rows(
                    baseline_metrics["val"]["per_class_f1"],
                    selective_plain_record["val"]["per_class_f1"],
                    class_names,
                    top_n=10,
                ),
            },
            "leaderboard_top30": selective_leaderboard,
            "best_overall_by_val_macro": {
                "variant": best_selective_macro["variant"],
                "config": {
                    "class_gate_name": best_selective_macro["class_gate_name"],
                    "class_gate_mode": best_selective_macro["class_gate_mode"],
                    "class_gate_gamma": best_selective_macro["class_gate_gamma"],
                    "uncertainty_name": best_selective_macro["uncertainty_name"],
                    "uncertainty_mode": best_selective_macro["uncertainty_mode"],
                    "uncertainty_delta": best_selective_macro["uncertainty_delta"],
                    "uncertainty_tau": best_selective_macro["uncertainty_tau"],
                    "positive_only": bool(best_selective_macro["positive_only"]),
                },
                "val": _metrics_for_json(best_selective_macro["val"], include_per_class=True),
                "test": _metrics_for_json(best_selective_macro["test"], include_per_class=True),
                "test_gain_over_baseline": {
                    "macro": float(best_selective_macro["test"]["macro"] - baseline_metrics["test"]["macro"]),
                    "hard": float(best_selective_macro["test"]["hard"] - baseline_metrics["test"]["hard"]),
                },
                "test_gain_over_plain_rerank": {
                    "macro": float(best_selective_macro["test"]["macro"] - selective_plain_record["test"]["macro"]),
                    "hard": float(best_selective_macro["test"]["hard"] - selective_plain_record["test"]["hard"]),
                },
                "top_gain_classes": class_gain_rows(
                    baseline_test_per_class,
                    best_selective_macro["test"]["per_class_f1"],
                    class_names,
                    top_n=10,
                ),
                "prediction_shift_vs_baseline": selective_prediction_shift,
            },
            "best_overall_by_val_hard": {
                "variant": best_selective_hard["variant"],
                "config": {
                    "class_gate_name": best_selective_hard["class_gate_name"],
                    "class_gate_mode": best_selective_hard["class_gate_mode"],
                    "class_gate_gamma": best_selective_hard["class_gate_gamma"],
                    "uncertainty_name": best_selective_hard["uncertainty_name"],
                    "uncertainty_mode": best_selective_hard["uncertainty_mode"],
                    "uncertainty_delta": best_selective_hard["uncertainty_delta"],
                    "uncertainty_tau": best_selective_hard["uncertainty_tau"],
                    "positive_only": bool(best_selective_hard["positive_only"]),
                },
                "val": _metrics_for_json(best_selective_hard["val"]),
                "test": _metrics_for_json(best_selective_hard["test"]),
            },
            "best_by_variant": {
                variant_name: {
                    "config": {
                        "class_gate_name": record["class_gate_name"],
                        "class_gate_mode": record["class_gate_mode"],
                        "class_gate_gamma": record["class_gate_gamma"],
                        "uncertainty_name": record["uncertainty_name"],
                        "uncertainty_mode": record["uncertainty_mode"],
                        "uncertainty_delta": record["uncertainty_delta"],
                        "uncertainty_tau": record["uncertainty_tau"],
                        "positive_only": bool(record["positive_only"]),
                    },
                    "val": _metrics_for_json(record["val"]),
                    "test": _metrics_for_json(record["test"]),
                }
                for variant_name, record in best_selective_by_variant.items()
            },
            "hard_cases": {
                "baseline_wrong_selective_right_count": len(selective_hard_case_rows["baseline_wrong_selective_right"]),
                "baseline_right_selective_wrong_count": len(selective_hard_case_rows["baseline_right_selective_wrong"]),
                "baseline_wrong_selective_right": selective_hard_case_rows["baseline_wrong_selective_right"],
                "baseline_right_selective_wrong": selective_hard_case_rows["baseline_right_selective_wrong"],
            },
        },
        "retrieval": {
            "prior_only": retrieval_prior_results,
            "leaderboard_top20": retrieval_leaderboard,
            "best_by_val_macro": {
                "config": {
                    "prior_name": best_retrieval_macro["prior_name"],
                    "k": int(best_retrieval_macro["k"]),
                    "beta": float(best_retrieval_macro["beta"]),
                },
                "val": _metrics_for_json(best_retrieval_macro["val"], include_per_class=True),
                "test": _metrics_for_json(best_retrieval_macro["test"], include_per_class=True),
                "test_gain_over_baseline": {
                    "macro": float(
                        best_retrieval_macro["test"]["macro"] - baseline_metrics["test"]["macro"]
                    ),
                    "hard": float(
                        best_retrieval_macro["test"]["hard"] - baseline_metrics["test"]["hard"]
                    ),
                },
            },
            "best_by_val_hard": {
                "config": {
                    "prior_name": best_retrieval_hard["prior_name"],
                    "k": int(best_retrieval_hard["k"]),
                    "beta": float(best_retrieval_hard["beta"]),
                },
                "val": _metrics_for_json(best_retrieval_hard["val"]),
                "test": _metrics_for_json(best_retrieval_hard["test"]),
            },
        },
        "hard_cases": {
            "baseline_wrong_llm_right_count": len(hard_case_rows["baseline_wrong_llm_right"]),
            "baseline_right_llm_wrong_count": len(hard_case_rows["baseline_right_llm_wrong"]),
            "baseline_wrong_llm_right": hard_case_rows["baseline_wrong_llm_right"],
            "baseline_right_llm_wrong": hard_case_rows["baseline_right_llm_wrong"],
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
