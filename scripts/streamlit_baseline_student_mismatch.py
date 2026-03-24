#!/usr/bin/env python3
"""Streamlit app for browsing baseline/student disagreement cases against GT."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUN_DIR = PROJECT_ROOT / "logs" / "analysis" / "privileged_distillation_text_teacher_seedfix_20260316"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "logs" / "analysis" / "min_agent_evidence_verification_v2_comparative_add_20260312" / "_cache"
)
DEFAULT_TEST_ANNOTATION = (
    PROJECT_ROOT.parent / "Intentonomy" / "data" / "annotation" / "intentonomy_test2020.json"
)
DEFAULT_IMAGE_DIR = PROJECT_ROOT.parent / "Intentonomy" / "data" / "images" / "low"


@dataclass
class LoadedArtifacts:
    image_ids: List[str]
    labels: np.ndarray
    features: np.ndarray
    class_names: List[str]
    image_paths: Dict[str, str]
    baseline_scores: np.ndarray
    student_scores: np.ndarray
    baseline_thresholds: np.ndarray
    student_thresholds: np.ndarray
    baseline_pred: np.ndarray
    student_pred: np.ndarray


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
    return {
        key: np.asarray(arr[key]) if key != "image_ids" else [str(x) for x in arr[key].tolist()]
        for key in arr.files
    }


def _build_model_from_state_dict(
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


@torch.inference_mode()
def _predict_scores(model: nn.Module, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
    outputs: List[np.ndarray] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for start in range(0, features.shape[0], int(batch_size)):
        batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits = model(batch)
        outputs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _threshold(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (scores > thresholds[None, :]).astype(np.int32)


def _label_names(row: np.ndarray, class_names: Sequence[str]) -> List[str]:
    return [class_names[idx] for idx in np.where(np.asarray(row) > 0.5)[0].tolist()]


@st.cache_resource(show_spinner=False)
def load_artifacts(
    run_dir_str: str,
    cache_dir_str: str,
    annotation_path_str: str,
    image_dir_str: str,
    batch_size: int,
) -> LoadedArtifacts:
    run_dir = Path(run_dir_str)
    cache_dir = Path(cache_dir_str)
    annotation_path = Path(annotation_path_str)
    image_dir = Path(image_dir_str)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    test_bundle = _load_npz(cache_dir / "test_clip.npz")
    class_names, image_paths = _load_class_names_and_paths(annotation_path, image_dir)

    features = np.asarray(test_bundle["features"], dtype=np.float32)
    labels = np.asarray(test_bundle["labels"], dtype=np.float32)
    image_ids = [str(x) for x in test_bundle["image_ids"]]
    num_classes = int(labels.shape[1])
    image_dim = int(features.shape[1])

    baseline_state = torch.load(run_dir / "baseline_best.pt", map_location="cpu", weights_only=True)
    student_state = torch.load(run_dir / "dynamic_gated_kd_best.pt", map_location="cpu", weights_only=True)

    baseline_model = _build_model_from_state_dict(
        baseline_state,
        image_dim=image_dim,
        num_classes=num_classes,
    )
    student_model = _build_model_from_state_dict(
        student_state,
        image_dim=image_dim,
        num_classes=num_classes,
    )

    baseline_scores = _predict_scores(baseline_model, features, batch_size=int(batch_size))
    student_scores = _predict_scores(student_model, features, batch_size=int(batch_size))

    baseline_thresholds = np.asarray(
        summary["baseline"]["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )
    student_thresholds = np.asarray(
        summary["dynamic_gated_kd"]["bundle"]["classwise"]["val"]["class_thresholds"],
        dtype=np.float32,
    )

    baseline_pred = _threshold(baseline_scores, baseline_thresholds)
    student_pred = _threshold(student_scores, student_thresholds)

    return LoadedArtifacts(
        image_ids=image_ids,
        labels=labels,
        features=features,
        class_names=class_names,
        image_paths=image_paths,
        baseline_scores=baseline_scores,
        student_scores=student_scores,
        baseline_thresholds=baseline_thresholds,
        student_thresholds=student_thresholds,
        baseline_pred=baseline_pred,
        student_pred=student_pred,
    )


def main() -> None:
    st.set_page_config(page_title="Baseline vs Student Mismatch", layout="wide")
    st.title("Baseline vs Student Disagreement Browser")

    with st.sidebar:
        run_dir = st.text_input("Run Dir", str(DEFAULT_RUN_DIR))
        cache_dir = st.text_input("Cache Dir", str(DEFAULT_CACHE_DIR))
        annotation_path = st.text_input("Test Annotation", str(DEFAULT_TEST_ANNOTATION))
        image_dir = st.text_input("Image Dir", str(DEFAULT_IMAGE_DIR))
        batch_size = st.number_input("Inference Batch Size", min_value=64, max_value=2048, value=512, step=64)
        view_mode = st.selectbox(
            "View Mode",
            [
                "baseline_wrong_student_correct",
                "baseline_correct_student_wrong",
            ],
            format_func=lambda token: {
                "baseline_wrong_student_correct": "Baseline 错，Student 对",
                "baseline_correct_student_wrong": "Baseline 对，Student 错",
            }[token],
        )
        st.caption("默认使用 baseline_best.pt 与 dynamic_gated_kd_best.pt。")

    artifacts = load_artifacts(run_dir, cache_dir, annotation_path, image_dir, int(batch_size))

    gt = artifacts.labels.astype(np.int32)
    baseline_correct = np.all(artifacts.baseline_pred == gt, axis=1)
    student_correct = np.all(artifacts.student_pred == gt, axis=1)
    if view_mode == "baseline_wrong_student_correct":
        candidate_mask = (~baseline_correct) & student_correct
        caption = "筛选条件：baseline 预测标签集合与 GT 不一致，student 预测标签集合与 GT 完全一致。"
        selector_prefix = "baseline err, student ok"
        selector_errors = np.abs(artifacts.baseline_pred - gt).sum(axis=1)
    else:
        candidate_mask = baseline_correct & (~student_correct)
        caption = "筛选条件：baseline 预测标签集合与 GT 完全一致，student 预测标签集合与 GT 不一致。"
        selector_prefix = "baseline ok, student err"
        selector_errors = np.abs(artifacts.student_pred - gt).sum(axis=1)
    candidate_indices = np.where(candidate_mask)[0].tolist()

    st.metric("Candidate Count", len(candidate_indices))
    st.caption(caption)

    if not candidate_indices:
        st.warning("当前配置下没有找到符合条件的样本。")
        return

    with st.sidebar:
        sort_mode = st.selectbox("Sort By", ["student_error_count", "baseline_confidence"])
        show_count = st.slider("Max Examples", min_value=10, max_value=min(200, len(candidate_indices)), value=min(50, len(candidate_indices)))

    if sort_mode == "student_error_count":
        error_counts = np.abs(artifacts.student_pred[candidate_indices] - gt[candidate_indices]).sum(axis=1)
        ordered = [candidate_indices[idx] for idx in np.argsort(-error_counts).tolist()]
    else:
        confidence = []
        for idx in candidate_indices:
            gt_ids = np.where(gt[idx] > 0)[0]
            if len(gt_ids) == 0:
                confidence.append(0.0)
            else:
                confidence.append(float(artifacts.baseline_scores[idx, gt_ids].mean()))
        ordered = [candidate_indices[idx] for idx in np.argsort(-np.asarray(confidence)).tolist()]

    ordered = ordered[: int(show_count)]
    selected = st.selectbox(
        "Select Example",
        ordered,
        format_func=lambda idx: f"{artifacts.image_ids[idx]} | {selector_prefix}={int(selector_errors[idx])}",
    )

    image_id = artifacts.image_ids[selected]
    image_path = artifacts.image_paths.get(image_id, "")
    gt_names = _label_names(gt[selected], artifacts.class_names)
    baseline_names = _label_names(artifacts.baseline_pred[selected], artifacts.class_names)
    student_names = _label_names(artifacts.student_pred[selected], artifacts.class_names)

    missing_by_student = [name for name in gt_names if name not in student_names]
    extra_by_student = [name for name in student_names if name not in gt_names]
    missing_by_baseline = [name for name in gt_names if name not in baseline_names]
    extra_by_baseline = [name for name in baseline_names if name not in gt_names]

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
        st.markdown(f"**GT**: {', '.join(gt_names) if gt_names else '(none)'}")
        st.markdown(f"**Baseline**: {', '.join(baseline_names) if baseline_names else '(none)'}")
        st.markdown(f"**Baseline Missing**: {', '.join(missing_by_baseline) if missing_by_baseline else '(none)'}")
        st.markdown(f"**Baseline Extra**: {', '.join(extra_by_baseline) if extra_by_baseline else '(none)'}")
        st.markdown(f"**Student**: {', '.join(student_names) if student_names else '(none)'}")
        st.markdown(f"**Student Missing**: {', '.join(missing_by_student) if missing_by_student else '(none)'}")
        st.markdown(f"**Student Extra**: {', '.join(extra_by_student) if extra_by_student else '(none)'}")

    gt_ids = np.where(gt[selected] > 0)[0].tolist()
    top_baseline = np.argsort(-artifacts.baseline_scores[selected])[:10].tolist()
    top_student = np.argsort(-artifacts.student_scores[selected])[:10].tolist()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Baseline Scores")
        rows = []
        for class_idx in top_baseline:
            rows.append(
                {
                    "class": artifacts.class_names[class_idx],
                    "score": float(artifacts.baseline_scores[selected, class_idx]),
                    "thr": float(artifacts.baseline_thresholds[class_idx]),
                    "gt": int(gt[selected, class_idx]),
                    "pred": int(artifacts.baseline_pred[selected, class_idx]),
                }
            )
        st.dataframe(rows, use_container_width=True)

    with col4:
        st.subheader("Student Scores")
        rows = []
        for class_idx in top_student:
            rows.append(
                {
                    "class": artifacts.class_names[class_idx],
                    "score": float(artifacts.student_scores[selected, class_idx]),
                    "thr": float(artifacts.student_thresholds[class_idx]),
                    "gt": int(gt[selected, class_idx]),
                    "pred": int(artifacts.student_pred[selected, class_idx]),
                }
            )
        st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
