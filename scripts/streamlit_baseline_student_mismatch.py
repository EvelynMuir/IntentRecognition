#!/usr/bin/env python3
"""Streamlit app for browsing baseline/full-method disagreement cases."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_privileged_distillation import TeacherMLP, _predict_teacher
from src.models.intentonomy_clip_vit_slot_module import INTENTONOMY_DESCRIPTIONS

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is required for the SADIR full-method browser. "
        "Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc


DEFAULT_RUN_DIR = PROJECT_ROOT / "logs" / "analysis" / "privileged_distillation_full_20260316"
DEFAULT_SADIR_FULL_DIR = PROJECT_ROOT / "logs" / "analysis" / "distillation_slrc_lcs_rebuild_20260327"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_TEXT_DIR = PROJECT_ROOT / "logs" / "analysis" / "vlm_full_20260316"
DEFAULT_IMAGE_DIR = PROJECT_ROOT.parent / "Intentonomy" / "data" / "images" / "low"
DEFAULT_GEMINI_FILE = PROJECT_ROOT.parent / "Intentonomy" / "data" / "intent_description_gemini.json"
DEFAULT_ANNOTATIONS = {
    "train": PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_train2020.json",
    "val": PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_val2020.json",
    "test": PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_test2020.json",
}
DEFAULT_TEXT_NPZ = {
    "train": DEFAULT_TEXT_DIR / "rationale_full_bge_features.npz",
    "val": DEFAULT_TEXT_DIR / "val_rationale_baseline_pred_bge_features.npz",
    "test": DEFAULT_TEXT_DIR / "test_rationale_baseline_pred_bge_features.npz",
}
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


@dataclass
class LoadedArtifacts:
    image_ids: List[str]
    labels: np.ndarray
    soft_labels: np.ndarray
    features: np.ndarray
    class_names: List[str]
    image_paths: Dict[str, str]
    baseline_scores: np.ndarray
    full_scores: np.ndarray
    baseline_thresholds: np.ndarray
    full_thresholds: np.ndarray
    baseline_pred: np.ndarray
    full_pred: np.ndarray
    rationale_texts: List[str]
    prior_scores: np.ndarray | None
    prior_mode: str | None
    full_method_name: str


class LegacyStudentMLP(nn.Module):
    def __init__(self, image_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.net(image_features)


class CurrentStudentMLP(nn.Module):
    def __init__(
        self,
        image_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        feature_proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, feature_proj_dim),
            nn.LayerNorm(feature_proj_dim),
            nn.GELU(),
            nn.Linear(feature_proj_dim, feature_proj_dim),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(image_features)
        return self.classifier(hidden)


class ResidualStudent(nn.Module):
    def __init__(self, image_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features: torch.Tensor, slr_logits: torch.Tensor) -> torch.Tensor:
        return slr_logits + self.net(image_features)


def _load_class_names_and_paths(annotation_path: Path, image_dir: Path) -> tuple[List[str], Dict[str, str]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(
        data["categories"],
        key=lambda item: int(item.get("id", item.get("category_id"))),
    )
    class_names = [str(item["name"]) for item in categories]

    image_paths: Dict[str, str] = {}
    for item in data["images"]:
        filename = str(item["filename"])
        if filename.startswith("low/"):
            filename = filename[4:]
        image_paths[str(item["id"])] = str((image_dir / filename).resolve())
    return class_names, image_paths


def _load_npz(path: Path) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    loaded: Dict[str, Any] = {}
    for key in arr.files:
        value = arr[key]
        loaded[key] = np.asarray(value) if key != "image_ids" else [str(x) for x in value.tolist()]
    return loaded


def _align_bundle_by_image_ids(bundle: Mapping[str, Any], image_ids: Sequence[str]) -> Dict[str, Any]:
    source_ids = [str(item) for item in bundle["image_ids"]]
    id_to_idx = {image_id: idx for idx, image_id in enumerate(source_ids)}
    row_ids = np.asarray([id_to_idx[str(image_id)] for image_id in image_ids], dtype=np.int64)
    aligned: Dict[str, Any] = {}
    for key, value in bundle.items():
        if key == "image_ids":
            aligned[key] = [source_ids[idx] for idx in row_ids.tolist()]
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(source_ids):
            aligned[key] = value[row_ids]
        else:
            aligned[key] = value
    return aligned


def _build_student_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    image_dim: int,
    num_classes: int,
    hidden_dim: int = 768,
    dropout: float = 0.1,
) -> nn.Module:
    if any(key.startswith("encoder.") or key.startswith("classifier.") for key in state_dict):
        model = CurrentStudentMLP(
            image_dim=image_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        model.load_state_dict(state_dict, strict=False)
        return model.eval()
    model = LegacyStudentMLP(
        image_dim=image_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def _build_teacher_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    text_dim: int,
    image_dim: int,
    num_classes: int,
    dropout: float = 0.1,
) -> tuple[TeacherMLP, str]:
    first_weight = state_dict["net.0.weight"]
    hidden_dim = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    if input_dim == int(text_dim):
        input_mode = "text_only"
        model = TeacherMLP(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            input_mode=input_mode,
        )
    elif input_dim == int(text_dim) + int(image_dim):
        input_mode = "image_text"
        model = TeacherMLP(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            input_mode=input_mode,
            image_dim=image_dim,
        )
    else:
        raise ValueError(
            f"Unable to infer teacher input mode from state_dict: input_dim={input_dim}, "
            f"text_dim={text_dim}, image_dim={image_dim}"
        )
    model.load_state_dict(state_dict, strict=False)
    return model.eval(), input_mode


def _build_residual_student_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    image_dim: int,
    num_classes: int,
    dropout: float = 0.1,
) -> ResidualStudent:
    first_weight = state_dict["net.0.weight"]
    hidden_dim = int(first_weight.shape[0])
    model = ResidualStudent(
        image_dim=image_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


@torch.inference_mode()
def _predict_scores(model: nn.Module, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
    logits = _predict_logits(model, features, batch_size=batch_size)
    return 1.0 / (1.0 + np.exp(-logits))


@torch.inference_mode()
def _predict_logits(model: nn.Module, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
    outputs: List[np.ndarray] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for start in range(0, features.shape[0], int(batch_size)):
        batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits = model(batch)
        outputs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


@torch.inference_mode()
def _predict_residual_logits(
    model: ResidualStudent,
    features: np.ndarray,
    slr_logits: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for start in range(0, features.shape[0], int(batch_size)):
        feat_batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        slr_batch = torch.as_tensor(slr_logits[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits = model(feat_batch, slr_batch)
        outputs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _threshold(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (scores > thresholds[None, :]).astype(np.int32)


def _label_names(row: np.ndarray, class_names: Sequence[str]) -> List[str]:
    return [class_names[idx] for idx in np.where(np.asarray(row) > 0.5)[0].tolist()]


def _resolve_text_keys(text_feature_source: str) -> tuple[str, str]:
    normalized = str(text_feature_source).strip().lower()
    if normalized == "full":
        return "features", "texts"
    if normalized == "step1_only":
        return "step1_features", "step1_texts"
    if normalized == "step1_step2":
        return "pos_features", "pos_texts"
    raise ValueError(f"Unsupported text feature source: {text_feature_source}")


def _format_soft_labels(soft_row: np.ndarray, class_names: Sequence[str]) -> str:
    positive_ids = np.where(np.asarray(soft_row) > 0.0)[0].tolist()
    if not positive_ids:
        return "(none)"
    return ", ".join(f"{class_names[idx]}={float(soft_row[idx]):.3f}" for idx in positive_ids)


def _ordered_unique(strings: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in strings:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _build_lcs_text_pools(gemini_file: Path, num_classes: int) -> Dict[str, List[List[str]]]:
    lexical_pools = [[phrase] for phrase in INTENTONOMY_LEXICAL_PHRASES[:num_classes]]
    canonical_pools = [[desc] for desc in INTENTONOMY_DESCRIPTIONS[:num_classes]]
    data = json.loads(gemini_file.read_text(encoding="utf-8"))

    scenario_pools: List[List[str]] = []
    for item in data[:num_classes]:
        scenario_texts = [str(desc.get("Text Query", "")) for desc in item.get("description", [])]
        scenario_pools.append(_ordered_unique(scenario_texts))
    while len(scenario_pools) < num_classes:
        scenario_pools.append([canonical_pools[len(scenario_pools)][0]])

    return {
        "lexical": lexical_pools[:num_classes],
        "canonical": canonical_pools[:num_classes],
        "scenario": scenario_pools[:num_classes],
    }


def _encode_text_pool(
    clip_model: torch.nn.Module,
    texts_per_class: Sequence[Sequence[str]],
    wrap_prompt: bool,
) -> np.ndarray:
    device = next(clip_model.parameters()).device
    embeddings: List[torch.Tensor] = []
    with torch.inference_mode():
        for text_group in texts_per_class:
            prompts = [
                DEFAULT_PROMPT_TEMPLATE.format(text) if wrap_prompt else str(text)
                for text in text_group
            ]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
            mean_feature = text_features.mean(dim=0)
            mean_feature = F.normalize(mean_feature, dim=0)
            embeddings.append(mean_feature.detach().cpu())
    return torch.stack(embeddings, dim=0).numpy()


def _normalize_scores_per_sample(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = np.maximum(scores.std(axis=1, keepdims=True), eps)
    return (scores - mean) / std


def _text_logits_from_features(
    image_features: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    return (
        np.asarray(image_features, dtype=np.float32)
        @ np.asarray(text_embeddings, dtype=np.float32).T
        * float(logit_scale)
    )


def _apply_topk_rerank(
    baseline_logits: np.ndarray,
    prior_logits: np.ndarray,
    topk: int,
    alpha: float,
) -> np.ndarray:
    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    fused_prior = _normalize_scores_per_sample(prior_logits)
    output = baseline_logits.copy()
    topk = max(1, min(int(topk), baseline_logits.shape[1]))
    topk_idx = np.argpartition(-baseline_logits, kth=topk - 1, axis=1)[:, :topk]
    row_idx = np.arange(baseline_logits.shape[0])[:, None]
    output[row_idx, topk_idx] = baseline_logits[row_idx, topk_idx] + float(alpha) * fused_prior[row_idx, topk_idx]
    return output


@st.cache_resource(show_spinner=False)
def load_artifacts(
    split: str,
    run_dir_str: str,
    cache_dir_str: str,
    annotation_path_str: str,
    image_dir_str: str,
    baseline_ckpt_name: str,
    full_ckpt_name: str,
    baseline_summary_key: str,
    full_summary_key: str,
    text_npz_path_str: str,
    text_feature_source: str,
    teacher_ckpt_name: str,
    sadir_full_dir_str: str,
    gemini_file_str: str,
    batch_size: int,
) -> LoadedArtifacts:
    split = str(split)
    run_dir = Path(run_dir_str)
    cache_dir = Path(cache_dir_str)
    annotation_path = Path(annotation_path_str)
    image_dir = Path(image_dir_str)
    text_npz_path = Path(text_npz_path_str)
    sadir_full_dir = Path(sadir_full_dir_str)
    gemini_file = Path(gemini_file_str)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    clip_bundle = _load_npz(cache_dir / f"{split}_clip.npz")
    class_names, image_paths = _load_class_names_and_paths(annotation_path, image_dir)

    features = np.asarray(clip_bundle["features"], dtype=np.float32)
    labels = np.asarray(clip_bundle["labels"], dtype=np.float32)
    soft_labels = np.asarray(clip_bundle.get("soft_labels", clip_bundle["labels"]), dtype=np.float32)
    image_ids = [str(x) for x in clip_bundle["image_ids"]]
    num_classes = int(labels.shape[1])
    image_dim = int(features.shape[1])

    baseline_state = torch.load(run_dir / baseline_ckpt_name, map_location="cpu", weights_only=True)
    full_state = torch.load(run_dir / full_ckpt_name, map_location="cpu", weights_only=True)

    baseline_model = _build_student_from_state_dict(
        baseline_state,
        image_dim=image_dim,
        num_classes=num_classes,
    )
    full_model = _build_student_from_state_dict(
        full_state,
        image_dim=image_dim,
        num_classes=num_classes,
    )

    baseline_scores = _predict_scores(baseline_model, features, batch_size=int(batch_size))
    full_scores = _predict_scores(full_model, features, batch_size=int(batch_size))
    full_method_name = "UTD only"

    baseline_thresholds = np.asarray(
        summary[baseline_summary_key]["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )
    full_thresholds = np.asarray(
        summary[full_summary_key]["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )

    baseline_pred = _threshold(baseline_scores, baseline_thresholds)
    full_pred = _threshold(full_scores, full_thresholds)

    rationale_texts = [""] * len(image_ids)
    prior_scores: np.ndarray | None = None
    prior_mode: str | None = None

    if text_npz_path.exists():
        text_feature_key, text_string_key = _resolve_text_keys(text_feature_source)
        text_bundle = _align_bundle_by_image_ids(_load_npz(text_npz_path), image_ids)
        if text_string_key in text_bundle:
            rationale_texts = [str(item) for item in np.asarray(text_bundle[text_string_key]).tolist()]
        elif "texts" in text_bundle:
            rationale_texts = [str(item) for item in np.asarray(text_bundle["texts"]).tolist()]

        teacher_path = run_dir / teacher_ckpt_name
        if teacher_path.exists() and text_feature_key in text_bundle:
            teacher_state = torch.load(teacher_path, map_location="cpu", weights_only=True)
            text_features = np.asarray(text_bundle[text_feature_key], dtype=np.float32)
            teacher_model, prior_mode = _build_teacher_from_state_dict(
                teacher_state,
                text_dim=int(text_features.shape[1]),
                image_dim=image_dim,
                num_classes=num_classes,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            teacher_model = teacher_model.to(device).eval()
            image_features = features if prior_mode == "image_text" else None
            prior_scores = _predict_teacher(
                teacher_model,
                text_features=text_features,
                image_features=image_features,
                device=device,
                batch_size=int(batch_size),
            )

    if sadir_full_dir.exists() and gemini_file.exists():
        sadir_summary_path = sadir_full_dir / "summary.json"
        if sadir_summary_path.exists():
            sadir_summary = json.loads(sadir_summary_path.read_text(encoding="utf-8"))
            utd_logits = _predict_logits(full_model, features, batch_size=int(batch_size))
            clip_model, _ = clip.load("ViT-L/14", device="cpu")
            clip_model = clip_model.eval()
            text_pools = _build_lcs_text_pools(gemini_file=gemini_file, num_classes=num_classes)
            text_embeddings = {
                "lexical": _encode_text_pool(clip_model, text_pools["lexical"], wrap_prompt=True),
                "canonical": _encode_text_pool(clip_model, text_pools["canonical"], wrap_prompt=True),
                "scenario": _encode_text_pool(clip_model, text_pools["scenario"], wrap_prompt=False),
            }
            logit_scale = float(getattr(clip_model, "logit_scale", torch.tensor(1.0)).exp().item())
            lexical_logits = _text_logits_from_features(features, text_embeddings["lexical"], logit_scale)
            canonical_logits = _text_logits_from_features(features, text_embeddings["canonical"], logit_scale)
            scenario_logits = _text_logits_from_features(features, text_embeddings["scenario"], logit_scale)
            prior_scores = (
                _normalize_scores_per_sample(lexical_logits)
                + _normalize_scores_per_sample(canonical_logits)
                + _normalize_scores_per_sample(scenario_logits)
            ) / 3.0
            prior_mode = "lexical+canonical+scenario"
            slr_logits = _apply_topk_rerank(
                baseline_logits=utd_logits,
                prior_logits=prior_scores,
                topk=10,
                alpha=0.3,
            )
            residual_ckpt = sadir_full_dir / "slr_c_residual_dynamic_kd_best.pt"
            if residual_ckpt.exists():
                residual_state = torch.load(residual_ckpt, map_location="cpu", weights_only=True)
                residual_model = _build_residual_student_from_state_dict(
                    residual_state,
                    image_dim=image_dim,
                    num_classes=num_classes,
                )
                sadir_logits = _predict_residual_logits(
                    residual_model,
                    features=features,
                    slr_logits=slr_logits,
                    batch_size=int(batch_size),
                )
                full_method_name = "SADIR full (checkpoint)"
            else:
                sadir_logits = slr_logits
                full_method_name = "SADIR full (SLR-C reconstructed)"
            full_scores = 1.0 / (1.0 + np.exp(-sadir_logits))
            full_thresholds = np.asarray(
                sadir_summary["slr_c_residual_dynamic_kd"]["bundle"]["classwise"]["val"]["class_thresholds"],
                dtype=np.float32,
            )
            full_pred = _threshold(full_scores, full_thresholds)

    return LoadedArtifacts(
        image_ids=image_ids,
        labels=labels,
        soft_labels=soft_labels,
        features=features,
        class_names=class_names,
        image_paths=image_paths,
        baseline_scores=baseline_scores,
        full_scores=full_scores,
        baseline_thresholds=baseline_thresholds,
        full_thresholds=full_thresholds,
        baseline_pred=baseline_pred,
        full_pred=full_pred,
        rationale_texts=rationale_texts,
        prior_scores=prior_scores,
        prior_mode=prior_mode,
        full_method_name=full_method_name,
    )


def main() -> None:
    st.set_page_config(page_title="Baseline vs Full Method Browser", layout="wide")
    st.title("Baseline vs Full Method Browser")

    with st.sidebar:
        split = st.selectbox("Split", ["train", "val", "test"], index=0)
        run_dir = st.text_input("Run Dir", str(DEFAULT_RUN_DIR))
        sadir_full_dir = st.text_input("SADIR Full Summary Dir", str(DEFAULT_SADIR_FULL_DIR))
        cache_dir = st.text_input("Cache Dir", str(DEFAULT_CACHE_DIR))
        annotation_path = st.text_input("Annotation", str(DEFAULT_ANNOTATIONS[split]))
        image_dir = st.text_input("Image Dir", str(DEFAULT_IMAGE_DIR))
        gemini_file = st.text_input("Gemini Scenario File", str(DEFAULT_GEMINI_FILE))
        baseline_ckpt_name = st.text_input("Baseline Checkpoint", "baseline_best.pt")
        full_ckpt_name = st.text_input("Full Method Checkpoint", "dynamic_gated_kd_best.pt")
        baseline_summary_key = st.text_input("Baseline Summary Key", "baseline")
        full_summary_key = st.text_input("Full Summary Key", "dynamic_gated_kd")
        teacher_ckpt_name = st.text_input("Prior Teacher Checkpoint", "teacher_best.pt")
        text_npz_path = st.text_input("Rationale Text NPZ", str(DEFAULT_TEXT_NPZ[split]))
        text_feature_source = st.selectbox("Text Source", ["full", "step1_only", "step1_step2"], index=0)
        batch_size = st.number_input("Inference Batch Size", min_value=64, max_value=2048, value=512, step=64)
        view_mode = st.selectbox(
            "View Mode",
            [
                "both_correct",
                "baseline_wrong_full_correct",
                "baseline_correct_full_wrong",
                "both_wrong",
            ],
            format_func=lambda token: {
                "both_correct": "都对",
                "baseline_wrong_full_correct": "Baseline 错，Full Method 对",
                "baseline_correct_full_wrong": "Baseline 对，Full Method 错",
                "both_wrong": "都错",
            }[token],
        )

    artifacts = load_artifacts(
        split=split,
        run_dir_str=run_dir,
        cache_dir_str=cache_dir,
        annotation_path_str=annotation_path,
        image_dir_str=image_dir,
        baseline_ckpt_name=baseline_ckpt_name,
        full_ckpt_name=full_ckpt_name,
        baseline_summary_key=baseline_summary_key,
        full_summary_key=full_summary_key,
        text_npz_path_str=text_npz_path,
        text_feature_source=text_feature_source,
        teacher_ckpt_name=teacher_ckpt_name,
        sadir_full_dir_str=sadir_full_dir,
        gemini_file_str=gemini_file,
        batch_size=int(batch_size),
    )

    gt = artifacts.labels.astype(np.int32)
    baseline_correct = np.all(artifacts.baseline_pred == gt, axis=1)
    full_correct = np.all(artifacts.full_pred == gt, axis=1)

    if view_mode == "both_correct":
        candidate_mask = baseline_correct & full_correct
        caption = f"筛选条件：baseline 与 {artifacts.full_method_name} 的预测标签集合都与硬标签完全一致。"
        selector_prefix = "both ok"
        selector_errors = np.zeros(gt.shape[0], dtype=np.int32)
        sort_options = ["baseline_confidence_on_gt", "prior_confidence_on_gt"]
    elif view_mode == "baseline_wrong_full_correct":
        candidate_mask = (~baseline_correct) & full_correct
        caption = f"筛选条件：baseline 预测标签集合与硬标签不一致，{artifacts.full_method_name} 预测标签集合与硬标签完全一致。"
        selector_prefix = "baseline err, full ok"
        selector_errors = np.abs(artifacts.baseline_pred - gt).sum(axis=1)
        sort_options = ["baseline_error_count", "prior_confidence_on_gt", "baseline_confidence_on_gt"]
    elif view_mode == "baseline_correct_full_wrong":
        candidate_mask = baseline_correct & (~full_correct)
        caption = f"筛选条件：baseline 预测标签集合与硬标签完全一致，{artifacts.full_method_name} 预测标签集合与硬标签不一致。"
        selector_prefix = "baseline ok, full err"
        selector_errors = np.abs(artifacts.full_pred - gt).sum(axis=1)
        sort_options = ["full_error_count", "prior_confidence_on_gt", "baseline_confidence_on_gt"]
    else:
        candidate_mask = (~baseline_correct) & (~full_correct)
        caption = f"筛选条件：baseline 与 {artifacts.full_method_name} 的预测标签集合都与硬标签不一致。"
        selector_prefix = "both err"
        selector_errors = (
            np.abs(artifacts.baseline_pred - gt).sum(axis=1)
            + np.abs(artifacts.full_pred - gt).sum(axis=1)
        )
        sort_options = ["combined_error_count", "prior_confidence_on_gt", "baseline_confidence_on_gt"]
    candidate_indices = np.where(candidate_mask)[0].tolist()

    st.metric("Candidate Count", len(candidate_indices))
    st.caption(caption)
    if artifacts.prior_scores is not None:
        st.caption(f"Full method: {artifacts.full_method_name}; prior source: `{artifacts.prior_mode}`.")

    if not candidate_indices:
        st.warning("当前配置下没有找到符合条件的样本。")
        return

    with st.sidebar:
        sort_mode = st.selectbox("Sort By", sort_options)
        show_count = st.slider(
            "Max Examples",
            min_value=10,
            max_value=min(500, len(candidate_indices)),
            value=min(80, len(candidate_indices)),
        )

    def _mean_score(scores: np.ndarray, idx: int) -> float:
        gt_ids = np.where(gt[idx] > 0)[0]
        if len(gt_ids) == 0:
            return 0.0
        return float(scores[idx, gt_ids].mean())

    if sort_mode in {"baseline_error_count", "full_error_count", "combined_error_count"}:
        ordered = [candidate_indices[idx] for idx in np.argsort(-selector_errors[candidate_indices]).tolist()]
    elif sort_mode == "prior_confidence_on_gt" and artifacts.prior_scores is not None:
        ordered = [candidate_indices[idx] for idx in np.argsort(-np.asarray([_mean_score(artifacts.prior_scores, i) for i in candidate_indices])).tolist()]
    else:
        ordered = [candidate_indices[idx] for idx in np.argsort(-np.asarray([_mean_score(artifacts.baseline_scores, i) for i in candidate_indices])).tolist()]

    ordered = ordered[: int(show_count)]
    browse_key = f"browse_pos::{split}::{view_mode}::{sort_mode}::{len(ordered)}"
    if browse_key not in st.session_state:
        st.session_state[browse_key] = 0
    current_pos = int(st.session_state[browse_key])
    current_pos = max(0, min(current_pos, len(ordered) - 1))

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])
    with nav_col1:
        if st.button("上一条", use_container_width=True, disabled=current_pos <= 0):
            current_pos = max(0, current_pos - 1)
            st.session_state[browse_key] = current_pos
            st.rerun()
    with nav_col2:
        if st.button("载入下一条", use_container_width=True, disabled=current_pos >= len(ordered) - 1):
            current_pos = min(len(ordered) - 1, current_pos + 1)
            st.session_state[browse_key] = current_pos
            st.rerun()
    with nav_col3:
        st.caption(
            f"当前样本 {current_pos + 1}/{len(ordered)}: "
            f"{artifacts.image_ids[ordered[current_pos]]} | {selector_prefix}={int(selector_errors[ordered[current_pos]])}"
        )

    selected = ordered[current_pos]

    image_id = artifacts.image_ids[selected]
    image_path = artifacts.image_paths.get(image_id, "")
    gt_names = _label_names(gt[selected], artifacts.class_names)
    baseline_names = _label_names(artifacts.baseline_pred[selected], artifacts.class_names)
    full_names = _label_names(artifacts.full_pred[selected], artifacts.class_names)
    soft_label_text = _format_soft_labels(artifacts.soft_labels[selected], artifacts.class_names)

    missing_by_baseline = [name for name in gt_names if name not in baseline_names]
    extra_by_baseline = [name for name in baseline_names if name not in gt_names]
    missing_by_full = [name for name in gt_names if name not in full_names]
    extra_by_full = [name for name in full_names if name not in gt_names]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Image")
        if image_path and Path(image_path).exists():
            st.image(Image.open(image_path), use_container_width=True)
        else:
            st.warning(f"Image not found: {image_path}")
        st.code(image_path or image_id)

    with col2:
        st.subheader("Labels")
        st.markdown(f"**Hard Labels**: {', '.join(gt_names) if gt_names else '(none)'}")
        st.markdown(f"**Soft Labels**: {soft_label_text}")
        st.markdown(f"**Baseline**: {', '.join(baseline_names) if baseline_names else '(none)'}")
        st.markdown(f"**Baseline Missing**: {', '.join(missing_by_baseline) if missing_by_baseline else '(none)'}")
        st.markdown(f"**Baseline Extra**: {', '.join(extra_by_baseline) if extra_by_baseline else '(none)'}")
        st.markdown(f"**{artifacts.full_method_name}**: {', '.join(full_names) if full_names else '(none)'}")
        st.markdown(f"**Full Missing**: {', '.join(missing_by_full) if missing_by_full else '(none)'}")
        st.markdown(f"**Full Extra**: {', '.join(extra_by_full) if extra_by_full else '(none)'}")

    focus_ids = set(np.where(gt[selected] > 0)[0].tolist())
    focus_ids.update(np.where(artifacts.baseline_pred[selected] > 0)[0].tolist())
    focus_ids.update(np.where(artifacts.full_pred[selected] > 0)[0].tolist())
    focus_ids.update(np.argsort(-artifacts.baseline_scores[selected])[:8].tolist())
    focus_ids.update(np.argsort(-artifacts.full_scores[selected])[:8].tolist())
    if artifacts.prior_scores is not None:
        focus_ids.update(np.argsort(-artifacts.prior_scores[selected])[:8].tolist())
    ordered_focus = sorted(
        list(focus_ids),
        key=lambda class_idx: (
            float(artifacts.soft_labels[selected, class_idx]),
            float(artifacts.full_scores[selected, class_idx]),
            float(artifacts.baseline_scores[selected, class_idx]),
        ),
        reverse=True,
    )

    score_rows = []
    for class_idx in ordered_focus:
        row = {
            "class": artifacts.class_names[class_idx],
            "soft": float(artifacts.soft_labels[selected, class_idx]),
            "hard": int(gt[selected, class_idx]),
            "baseline_score": float(artifacts.baseline_scores[selected, class_idx]),
            "baseline_thr": float(artifacts.baseline_thresholds[class_idx]),
            "baseline_pred": int(artifacts.baseline_pred[selected, class_idx]),
            "full_score": float(artifacts.full_scores[selected, class_idx]),
            "full_thr": float(artifacts.full_thresholds[class_idx]),
            "full_pred": int(artifacts.full_pred[selected, class_idx]),
        }
        if artifacts.prior_scores is not None:
            row["prior_score"] = float(artifacts.prior_scores[selected, class_idx])
        score_rows.append(row)

    st.subheader("Focused Class Table")
    st.dataframe(score_rows, use_container_width=True)

    if artifacts.prior_scores is not None:
        top_prior = np.argsort(-artifacts.prior_scores[selected])[:10].tolist()
        prior_rows = []
        for class_idx in top_prior:
            prior_rows.append(
                {
                    "class": artifacts.class_names[class_idx],
                    "prior_score": float(artifacts.prior_scores[selected, class_idx]),
                    "soft": float(artifacts.soft_labels[selected, class_idx]),
                    "hard": int(gt[selected, class_idx]),
                    "baseline_pred": int(artifacts.baseline_pred[selected, class_idx]),
                    "full_pred": int(artifacts.full_pred[selected, class_idx]),
                }
            )
        st.subheader("Top Prior Scores")
        st.dataframe(prior_rows, use_container_width=True)

    rationale_text = artifacts.rationale_texts[selected].strip()
    st.subheader("Rationale")
    if rationale_text:
        st.text_area("Rationale Text", rationale_text, height=420)
    else:
        st.info("当前没有找到对应的 rationale 文本。")


if __name__ == "__main__":
    main()
