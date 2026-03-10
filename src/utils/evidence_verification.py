from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is required for evidence_verification utilities. "
        "Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc

from src.models.components.clip_text_anchors import (
    COCO_CLASSES,
    STANFORD40_ACTIONS,
)


EXPERT_NAMES = ("object", "scene", "style", "activity")
DEFAULT_EXPERT_PROMPTS = {
    "object": "a photo of {}",
    "scene": "a photo of {}",
    "style": "a {} photo",
    "activity": "a photo of a person {}",
}

# Flickr Style labels from Karayev et al., "Recognizing Image Style" (2013).
FLICKR_STYLE_20_CLASSES = (
    "Bokeh",
    "Bright",
    "Depth of Field",
    "Detailed",
    "Ethereal",
    "Geometric Composition",
    "HDR",
    "Hazy",
    "Horror",
    "Long Exposure",
    "Macro",
    "Melancholy",
    "Minimal",
    "Noir",
    "Pastel",
    "Romantic",
    "Serene",
    "Sunny",
    "Texture",
    "Vintage",
)

STYLE_HINTS = (
    "vibe",
    "lighting",
    "sunlight",
    "spotlight",
    "glow",
    "glowing",
    "warm",
    "dramatic",
    "vibrant",
    "colorful",
    "color",
    "colour",
    "candid",
    "expressive",
    "elegant",
    "glamorous",
    "bold",
    "refined",
    "cozy",
    "cosy",
    "tranquil",
    "serene",
    "mysterious",
    "modern vibe",
    "modern aesthetic",
    "artistic",
    "aesthetic",
    "joyful",
    "playful",
    "romantic",
    "energetic vibe",
    "powerful vibe",
    "epic vibe",
)

SCENE_HINTS = (
    "background",
    "street",
    "cafe",
    "patio",
    "track",
    "stage",
    "studio",
    "lab",
    "office",
    "park",
    "beach",
    "forest",
    "market",
    "mountain",
    "terrain",
    "reef",
    "underwater",
    "courtyard",
    "gallery",
    "museum",
    "classroom",
    "kitchen",
    "bedroom",
    "living room",
    "restaurant",
    "auditorium",
    "highway",
    "sidewalk",
    "road",
    "trail",
    "garden",
    "temple",
    "church",
    "gym",
    "stadium",
    "court",
    "lake",
    "river",
    "shore",
    "desk",
    "workshop",
    "hall",
    "room",
    "nature",
    "outdoor",
    "indoor",
    "city",
)

ACTIVITY_HINTS = (
    "running",
    "walking",
    "stretching",
    "posing",
    "looking",
    "holding",
    "leaning",
    "swimming",
    "crouching",
    "inspecting",
    "painting",
    "creating",
    "speaking",
    "talking",
    "gesturing",
    "celebrating",
    "training",
    "exercising",
    "studying",
    "teaching",
    "working",
    "smiling",
    "laughing",
    "hugging",
    "kissing",
    "jumping",
    "dancing",
    "playing",
    "praying",
    "meditating",
    "writing",
    "reading",
    "shopping",
    "browsing",
    "traveling",
    "travelling",
    "commuting",
    "packing",
    "driving",
    "cooking",
    "eating",
    "drinking",
    "grooming",
    "observing",
    "guiding",
    "repairing",
    "building",
    "performing",
    "pointing",
    "moving",
    "seated",
    "sitting",
    "listening",
    "making eye contact",
)

OBJECT_OVERRIDE_HINTS = (
    "gown",
    "tuxedo",
    "jewelry",
    "makeup",
    "hair",
    "wear",
    "attire",
    "streetwear",
    "denim",
    "uniform",
    "suit",
    "blazer",
    "ribbon",
    "trophy",
    "brush",
    "brushes",
    "canvas",
    "instrument",
    "megaphone",
    "microphone",
    "camera",
    "notebook",
    "backpack",
    "backpacks",
    "boots",
    "tripod",
    "monitor",
    "headset",
    "goggles",
    "coat",
    "jersey",
    "chessboard",
    "pieces",
    "table",
    "clock",
    "chair",
    "chairs",
    "smartphone",
    "whiteboard",
    "circuit",
    "robotic arm",
    "signs",
)

_TOKEN_PATTERN = re.compile(r"[a-z]+")


def _ordered_unique(strings: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in strings:
        text = normalize_phrase(item)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def normalize_phrase(text: str) -> str:
    phrase = str(text).strip().lower()
    phrase = phrase.replace("\n", " ").replace("\t", " ")
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = phrase.strip(" ,.;:-")
    return phrase


def split_visual_elements(raw_text: str) -> List[str]:
    text = str(raw_text or "")
    if not text.strip():
        return []
    parts = [normalize_phrase(chunk) for chunk in text.split(",")]
    return [part for part in parts if part]


def _phrase_tokens(phrase: str) -> List[str]:
    return _TOKEN_PATTERN.findall(normalize_phrase(phrase))


def classify_evidence_type(phrase: str) -> str:
    text = normalize_phrase(phrase)
    if not text:
        return "object"

    object_hits = sum(1 for hint in OBJECT_OVERRIDE_HINTS if hint in text)
    style_hits = sum(1 for hint in STYLE_HINTS if hint in text)
    scene_hits = sum(1 for hint in SCENE_HINTS if hint in text)
    activity_hits = sum(1 for hint in ACTIVITY_HINTS if hint in text)

    tokens = _phrase_tokens(text)
    if any(token.endswith("ing") for token in tokens):
        activity_hits += 1

    if style_hits > 0:
        return "style"
    if object_hits > max(scene_hits, activity_hits):
        return "object"
    if activity_hits > scene_hits and activity_hits > 0:
        return "activity"
    if scene_hits > 0:
        return "scene"
    if activity_hits > 0:
        return "activity"
    return "object"


def resolve_default_intent2concepts_file(
    data_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path | None:
    candidates: List[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root) / "intent2concepts.json")
    if project_root is not None:
        candidates.append(Path(project_root).parent / "Intentonomy" / "data" / "intent2concepts.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_intent_concepts_entries(intent2concepts_file: str | Path) -> List[Dict[str, Any]]:
    path = Path(intent2concepts_file)
    raw = path.read_text(encoding="utf-8")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
        for pos, ch in enumerate(raw):
            if ch != "[":
                continue
            try:
                candidate = json.loads(raw[pos:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, list):
                data = candidate
                break
        if data is None:
            raise

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")

    normalized_entries: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip()
        concepts = item.get("concepts", {})
        if not intent or not isinstance(concepts, dict):
            continue
        clean_concepts: Dict[str, float] = {}
        for key, value in concepts.items():
            phrase = normalize_phrase(key)
            if not phrase:
                continue
            try:
                clean_concepts[phrase] = float(value)
            except (TypeError, ValueError):
                continue
        if clean_concepts:
            normalized_entries.append({"intent": intent, "concepts": clean_concepts})
    return normalized_entries


def _merge_template_item(
    container: Dict[str, Dict[str, Any]],
    phrase: str,
    weight: float,
    source: str,
) -> None:
    normalized = normalize_phrase(phrase)
    if not normalized:
        return
    existing = container.get(normalized)
    if existing is None or float(weight) > float(existing["weight"]):
        container[normalized] = {
            "phrase": normalized,
            "weight": float(weight),
            "source": str(source),
        }


def build_intent_evidence_templates(
    class_names: Sequence[str],
    gemini_file: str | Path,
    intent2concepts_file: str | Path | None = None,
    max_items_per_expert: int = 8,
    visual_element_weight: float = 1.0,
    concept_weight_scale: float = 1.0,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    gemini_path = Path(gemini_file)
    gemini_data = json.loads(gemini_path.read_text(encoding="utf-8"))
    if not isinstance(gemini_data, list):
        raise ValueError(f"Expected a list in {gemini_path}, got {type(gemini_data).__name__}")

    concept_entries: List[Dict[str, Any]] = []
    if intent2concepts_file is not None and Path(intent2concepts_file).exists():
        concept_entries = load_intent_concepts_entries(intent2concepts_file)

    templates: List[Dict[str, List[Dict[str, Any]]]] = []
    for class_idx, _class_name in enumerate(class_names):
        raw_items = {expert: {} for expert in EXPERT_NAMES}

        if class_idx < len(gemini_data):
            descriptions = gemini_data[class_idx].get("description", [])
            if isinstance(descriptions, list):
                for desc in descriptions:
                    if not isinstance(desc, dict):
                        continue
                    for phrase in split_visual_elements(desc.get("Visual Elements", "")):
                        expert = classify_evidence_type(phrase)
                        _merge_template_item(
                            raw_items[expert],
                            phrase=phrase,
                            weight=float(visual_element_weight),
                            source="visual_elements",
                        )

        if class_idx < len(concept_entries):
            for phrase, value in concept_entries[class_idx].get("concepts", {}).items():
                expert = classify_evidence_type(phrase)
                _merge_template_item(
                    raw_items[expert],
                    phrase=phrase,
                    weight=float(value) * float(concept_weight_scale),
                    source="intent2concepts",
                )

        class_template: Dict[str, List[Dict[str, Any]]] = {}
        for expert in EXPERT_NAMES:
            items = sorted(
                raw_items[expert].values(),
                key=lambda item: (-float(item["weight"]), str(item["phrase"])),
            )
            class_template[expert] = items[: max(0, int(max_items_per_expert))]
        templates.append(class_template)

    return templates


def build_expert_phrase_banks(
    templates: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]],
    extra_phrases: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, List[str]]:
    banks: Dict[str, List[str]] = {expert: [] for expert in EXPERT_NAMES}

    for expert in EXPERT_NAMES:
        phrases: List[str] = []
        for template in templates:
            for item in template.get(expert, []):
                phrase = normalize_phrase(item.get("phrase", ""))
                if phrase:
                    phrases.append(phrase)
        if extra_phrases is not None:
            phrases.extend(extra_phrases.get(expert, []))
        banks[expert] = _ordered_unique(phrases)

    return banks


def resolve_default_places365_file() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[3] / "Intentonomy" / "data" / "place365.txt",
        Path("/home/evelynmuir/lambda/projects/IntentRecognition/Intentonomy/data/place365.txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_places365_benchmark_classes(file_path: str | Path | None = None) -> List[str]:
    path = Path(file_path) if file_path is not None else resolve_default_places365_file()
    if path is None or not path.exists():
        raise FileNotFoundError("Could not resolve Places365 label file.")

    classes: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw_label = line.split()[0].strip("/")
        path_parts = [part for part in raw_label.split("/") if part]
        if path_parts and len(path_parts[0]) == 1:
            path_parts = path_parts[1:]
        readable = " ".join("_".join(path_parts).split("_")).strip()
        if readable:
            classes.append(readable)
    return _ordered_unique(classes)


def build_benchmark_expert_phrase_banks(
    include_activity: bool = False,
    places365_file: str | Path | None = None,
) -> Dict[str, List[str]]:
    banks: Dict[str, List[str]] = {
        "object": _ordered_unique(COCO_CLASSES),
        "scene": load_places365_benchmark_classes(places365_file),
        "style": _ordered_unique(FLICKR_STYLE_20_CLASSES),
        "activity": _ordered_unique(STANFORD40_ACTIONS) if include_activity else [],
    }
    return banks


def build_generic_expert_phrase_banks(include_activity: bool = False) -> Dict[str, List[str]]:
    return build_benchmark_expert_phrase_banks(include_activity=include_activity)


def encode_phrase_bank(
    clip_model: torch.nn.Module,
    phrases: Sequence[str],
    prompt_template: str,
    batch_size: int = 64,
) -> np.ndarray:
    if not phrases:
        hidden_dim = int(getattr(clip_model.visual, "output_dim", 0) or 768)
        return np.zeros((0, hidden_dim), dtype=np.float32)

    device = next(clip_model.parameters()).device
    clip_model.eval()
    chunks: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(phrases), max(1, int(batch_size))):
            batch = phrases[start : start + max(1, int(batch_size))]
            prompts = [prompt_template.format(phrase) for phrase in batch]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            features = clip_model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1)
            chunks.append(features.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def encode_expert_phrase_banks(
    clip_model: torch.nn.Module,
    phrase_banks: Mapping[str, Sequence[str]],
    prompt_templates: Mapping[str, str] | None = None,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    templates = dict(DEFAULT_EXPERT_PROMPTS)
    if prompt_templates is not None:
        templates.update({str(key): str(value) for key, value in prompt_templates.items()})

    return {
        expert: encode_phrase_bank(
            clip_model=clip_model,
            phrases=phrase_banks.get(expert, []),
            prompt_template=templates[expert],
            batch_size=batch_size,
        )
        for expert in EXPERT_NAMES
    }


def encode_template_phrase_sets(
    clip_model: torch.nn.Module,
    templates: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]],
    prompt_templates: Mapping[str, str] | None = None,
    batch_size: int = 64,
) -> Dict[str, List[np.ndarray]]:
    prompt_map = dict(DEFAULT_EXPERT_PROMPTS)
    if prompt_templates is not None:
        prompt_map.update({str(key): str(value) for key, value in prompt_templates.items()})

    output: Dict[str, List[np.ndarray]] = {expert: [] for expert in EXPERT_NAMES}
    for expert in EXPERT_NAMES:
        for template in templates:
            phrases = [
                normalize_phrase(item.get("phrase", ""))
                for item in template.get(expert, [])
                if normalize_phrase(item.get("phrase", ""))
            ]
            output[expert].append(
                encode_phrase_bank(
                    clip_model=clip_model,
                    phrases=phrases,
                    prompt_template=prompt_map[expert],
                    batch_size=batch_size,
                )
            )
    return output


def compute_expert_phrase_scores(
    image_features: np.ndarray,
    phrase_bank_embeddings: Mapping[str, np.ndarray],
    logit_scale: float,
) -> Dict[str, np.ndarray]:
    image_features = np.asarray(image_features, dtype=np.float32)
    scores: Dict[str, np.ndarray] = {}
    for expert, embeddings in phrase_bank_embeddings.items():
        bank = np.asarray(embeddings, dtype=np.float32)
        if bank.size == 0:
            scores[expert] = np.zeros((image_features.shape[0], 0), dtype=np.float32)
            continue
        scores[expert] = (image_features @ bank.T) * float(logit_scale)
    return scores


def build_indexed_templates(
    templates: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]],
    phrase_banks: Mapping[str, Sequence[str]],
) -> List[Dict[str, Dict[str, Any]]]:
    bank_indices = {
        expert: {normalize_phrase(phrase): idx for idx, phrase in enumerate(phrase_banks.get(expert, []))}
        for expert in EXPERT_NAMES
    }

    indexed_templates: List[Dict[str, Dict[str, Any]]] = []
    for template in templates:
        class_template: Dict[str, Dict[str, Any]] = {}
        for expert in EXPERT_NAMES:
            indices: List[int] = []
            weights: List[float] = []
            phrases: List[str] = []
            sources: List[str] = []
            for item in template.get(expert, []):
                phrase = normalize_phrase(item.get("phrase", ""))
                idx = bank_indices[expert].get(phrase)
                if idx is None:
                    continue
                indices.append(int(idx))
                weights.append(float(item.get("weight", 1.0)))
                phrases.append(phrase)
                sources.append(str(item.get("source", "")))

            weights_arr = np.asarray(weights, dtype=np.float32)
            if weights_arr.size > 0:
                weight_sum = float(weights_arr.sum())
                if weight_sum <= 0.0:
                    weights_arr = np.full_like(weights_arr, 1.0 / float(len(weights_arr)))
                else:
                    weights_arr = weights_arr / weight_sum

            class_template[expert] = {
                "indices": np.asarray(indices, dtype=np.int64),
                "weights": weights_arr,
                "phrases": phrases,
                "sources": sources,
            }
        indexed_templates.append(class_template)
    return indexed_templates


def compute_expert_match_scores(
    expert_phrase_scores: Mapping[str, np.ndarray],
    indexed_templates: Sequence[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, np.ndarray]:
    if not indexed_templates:
        return {expert: np.zeros((0, 0), dtype=np.float32) for expert in EXPERT_NAMES}

    sample_count = next(iter(expert_phrase_scores.values())).shape[0]
    class_count = len(indexed_templates)
    match_scores: Dict[str, np.ndarray] = {}

    for expert in EXPERT_NAMES:
        bank_scores = np.asarray(expert_phrase_scores.get(expert), dtype=np.float32)
        class_matrix = np.zeros((sample_count, class_count), dtype=np.float32)
        for class_idx, class_template in enumerate(indexed_templates):
            item = class_template.get(expert, {})
            indices = np.asarray(item.get("indices", []), dtype=np.int64)
            weights = np.asarray(item.get("weights", []), dtype=np.float32)
            if indices.size == 0:
                continue
            selected = bank_scores[:, indices]
            if weights.size == 0:
                class_matrix[:, class_idx] = selected.mean(axis=1)
            else:
                class_matrix[:, class_idx] = selected @ weights
        match_scores[expert] = class_matrix

    return match_scores


def _aggregate_phrase_matrix(
    phrase_scores: np.ndarray,
    mode: str = "average",
) -> np.ndarray:
    scores = np.asarray(phrase_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D phrase score matrix, got shape {scores.shape}")
    if scores.shape[1] == 0:
        return np.zeros(scores.shape[0], dtype=np.float32)
    if mode == "average":
        return scores.mean(axis=1)
    if mode == "max":
        return scores.max(axis=1)
    if mode == "top2_avg":
        top_n = min(2, scores.shape[1])
        part = np.partition(scores, kth=scores.shape[1] - top_n, axis=1)
        return part[:, -top_n:].mean(axis=1)
    if mode == "logsumexp":
        row_max = scores.max(axis=1, keepdims=True)
        stabilized = scores - row_max
        return (
            np.log(np.exp(stabilized).sum(axis=1))
            + row_max.squeeze(axis=1)
            - np.log(float(scores.shape[1]))
        )
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def compute_expert_match_scores_with_similarity(
    expert_phrase_scores: Mapping[str, np.ndarray],
    phrase_bank_embeddings: Mapping[str, np.ndarray],
    template_phrase_embeddings: Mapping[str, Sequence[np.ndarray]],
    aggregation_mode: str = "average",
    positive_only: bool = True,
) -> Dict[str, np.ndarray]:
    if not template_phrase_embeddings:
        return {expert: np.zeros((0, 0), dtype=np.float32) for expert in EXPERT_NAMES}

    sample_count = next(iter(expert_phrase_scores.values())).shape[0]
    class_count = len(next(iter(template_phrase_embeddings.values())))
    match_scores: Dict[str, np.ndarray] = {}

    for expert in EXPERT_NAMES:
        bank_scores = np.asarray(expert_phrase_scores.get(expert), dtype=np.float32)
        bank_embeddings = np.asarray(phrase_bank_embeddings.get(expert), dtype=np.float32)
        class_matrix = np.zeros((sample_count, class_count), dtype=np.float32)
        if bank_scores.size == 0 or bank_embeddings.size == 0:
            match_scores[expert] = class_matrix
            continue

        support_scores = np.maximum(bank_scores, 0.0) if positive_only else bank_scores
        for class_idx, template_embeddings in enumerate(template_phrase_embeddings.get(expert, [])):
            template_embeddings = np.asarray(template_embeddings, dtype=np.float32)
            if template_embeddings.size == 0:
                continue
            similarity = template_embeddings @ bank_embeddings.T
            similarity = np.clip(similarity, 0.0, 1.0)
            phrase_level_support = (
                support_scores[:, None, :] * similarity[None, :, :]
            ).max(axis=2)
            class_matrix[:, class_idx] = _aggregate_phrase_matrix(
                phrase_level_support,
                mode=aggregation_mode,
            )
        match_scores[expert] = class_matrix

    return match_scores


def aggregate_verification_scores(
    expert_match_scores: Mapping[str, np.ndarray],
    indexed_templates: Sequence[Mapping[str, Mapping[str, Any]]],
    selected_experts: Sequence[str] | None = None,
    expert_weights: Mapping[str, float] | None = None,
    weight_mode: str = "template_aware",
) -> np.ndarray:
    if not indexed_templates:
        return np.zeros((0, 0), dtype=np.float32)

    selected = list(selected_experts) if selected_experts is not None else list(EXPERT_NAMES)
    selected = [expert for expert in selected if expert in EXPERT_NAMES]
    if not selected:
        raise ValueError("selected_experts must include at least one valid expert.")

    sample_count = next(iter(expert_match_scores.values())).shape[0]
    class_count = len(indexed_templates)
    output = np.zeros((sample_count, class_count), dtype=np.float32)

    base_weights = {
        expert: float(expert_weights.get(expert, 1.0) if expert_weights is not None else 1.0)
        for expert in selected
    }

    for class_idx, class_template in enumerate(indexed_templates):
        active_experts: List[str] = []
        active_weights: List[float] = []
        for expert in selected:
            has_template = len(class_template.get(expert, {}).get("indices", [])) > 0
            if weight_mode == "template_aware":
                if not has_template:
                    continue
                active_experts.append(expert)
                active_weights.append(base_weights[expert])
            elif weight_mode == "equal":
                active_experts.append(expert)
                active_weights.append(1.0)
            elif weight_mode == "fixed":
                active_experts.append(expert)
                active_weights.append(base_weights[expert])
            else:
                raise ValueError(f"Unsupported weight_mode: {weight_mode}")

        if not active_experts:
            continue

        weight_arr = np.asarray(active_weights, dtype=np.float32)
        weight_sum = float(weight_arr.sum())
        if weight_sum <= 0.0:
            weight_arr = np.full_like(weight_arr, 1.0 / float(len(weight_arr)))
        else:
            weight_arr = weight_arr / weight_sum

        class_score = np.zeros(sample_count, dtype=np.float32)
        for weight, expert in zip(weight_arr.tolist(), active_experts, strict=True):
            class_score += float(weight) * np.asarray(expert_match_scores[expert][:, class_idx], dtype=np.float32)
        output[:, class_idx] = class_score

    return output


def build_classwise_routing_matrix(
    expert_val_per_class_f1: Mapping[str, np.ndarray],
    slr_val_per_class_f1: np.ndarray,
    selected_experts: Sequence[str] | None = None,
    mode: str = "top1_positive",
    gamma: float = 8.0,
    gain_floor: float = 0.0,
    top_n: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    experts = list(selected_experts) if selected_experts is not None else list(EXPERT_NAMES)
    experts = [expert for expert in experts if expert in EXPERT_NAMES]
    if not experts:
        raise ValueError("selected_experts must include at least one valid expert.")

    slr_scores = np.asarray(slr_val_per_class_f1, dtype=np.float32)
    num_classes = slr_scores.shape[0]
    routing = np.zeros((num_classes, len(experts)), dtype=np.float32)
    rows: list[dict[str, Any]] = []

    expert_scores = {
        expert: np.asarray(expert_val_per_class_f1[expert], dtype=np.float32)
        for expert in experts
    }

    for expert, values in expert_scores.items():
        if values.shape != slr_scores.shape:
            raise ValueError(
                f"expert_val_per_class_f1[{expert}] shape {values.shape} != slr shape {slr_scores.shape}"
            )

    gain_matrix = np.stack([expert_scores[expert] - slr_scores for expert in experts], axis=1)

    for class_idx in range(num_classes):
        gains = gain_matrix[class_idx]
        weights = np.zeros(len(experts), dtype=np.float32)

        if mode == "top1_always":
            weights[int(np.argmax(gains))] = 1.0
        elif mode == "top1_positive":
            best_idx = int(np.argmax(gains))
            if float(gains[best_idx]) > float(gain_floor):
                weights[best_idx] = 1.0
        elif mode == "top2_soft":
            positive_ids = np.where(gains > float(gain_floor))[0]
            if positive_ids.size > 0:
                keep = positive_ids[np.argsort(-gains[positive_ids])[: max(1, int(top_n))]]
                scaled = float(gamma) * gains[keep]
                scaled = scaled - float(np.max(scaled))
                soft = np.exp(scaled)
                soft = soft / np.maximum(float(soft.sum()), 1e-8)
                weights[keep] = soft.astype(np.float32)
        elif mode == "soft_all":
            positive_ids = np.where(gains > float(gain_floor))[0]
            if positive_ids.size > 0:
                scaled = float(gamma) * gains[positive_ids]
                scaled = scaled - float(np.max(scaled))
                soft = np.exp(scaled)
                soft = soft / np.maximum(float(soft.sum()), 1e-8)
                weights[positive_ids] = soft.astype(np.float32)
        else:
            raise ValueError(f"Unsupported routing mode: {mode}")

        routing[class_idx] = weights
        active = [
            {
                "expert": experts[idx],
                "weight": float(weights[idx]),
                "gain_vs_slr": float(gains[idx]),
                "val_f1": float(expert_scores[experts[idx]][class_idx]),
            }
            for idx in np.where(weights > 0.0)[0].tolist()
        ]
        rows.append(
            {
                "class_id": int(class_idx),
                "mode": mode,
                "selected_experts": active,
                "slr_val_f1": float(slr_scores[class_idx]),
            }
        )

    return routing, rows


def aggregate_verification_scores_with_routing(
    expert_match_scores: Mapping[str, np.ndarray],
    indexed_templates: Sequence[Mapping[str, Mapping[str, Any]]],
    routing_weights: np.ndarray,
    selected_experts: Sequence[str] | None = None,
    renormalize_active: bool = True,
) -> np.ndarray:
    if not indexed_templates:
        return np.zeros((0, 0), dtype=np.float32)

    experts = list(selected_experts) if selected_experts is not None else list(EXPERT_NAMES)
    experts = [expert for expert in experts if expert in EXPERT_NAMES]
    if not experts:
        raise ValueError("selected_experts must include at least one valid expert.")

    first_scores = np.asarray(expert_match_scores[experts[0]], dtype=np.float32)
    num_samples, num_classes = first_scores.shape
    weights = np.asarray(routing_weights, dtype=np.float32)
    if weights.shape != (num_classes, len(experts)):
        raise ValueError(
            f"routing_weights shape {weights.shape} != expected {(num_classes, len(experts))}"
        )

    output = np.zeros((num_samples, num_classes), dtype=np.float32)
    for class_idx, class_template in enumerate(indexed_templates):
        active_ids: list[int] = []
        active_weights: list[float] = []
        for expert_idx, expert in enumerate(experts):
            has_template = len(class_template.get(expert, {}).get("indices", [])) > 0
            routed_weight = float(weights[class_idx, expert_idx])
            if not has_template or routed_weight <= 0.0:
                continue
            active_ids.append(expert_idx)
            active_weights.append(routed_weight)

        if not active_ids:
            continue

        active_weight_arr = np.asarray(active_weights, dtype=np.float32)
        if renormalize_active:
            denom = float(active_weight_arr.sum())
            if denom > 0.0:
                active_weight_arr = active_weight_arr / denom

        class_score = np.zeros(num_samples, dtype=np.float32)
        for local_idx, expert_idx in enumerate(active_ids):
            expert = experts[expert_idx]
            class_score += (
                float(active_weight_arr[local_idx])
                * np.asarray(expert_match_scores[expert][:, class_idx], dtype=np.float32)
            )
        output[:, class_idx] = class_score

    return output


def build_template_statistics(
    templates: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    per_expert_counts = {expert: [] for expert in EXPERT_NAMES}
    for template in templates:
        for expert in EXPERT_NAMES:
            per_expert_counts[expert].append(len(template.get(expert, [])))

    return {
        "num_classes": len(templates),
        "avg_items_per_expert": {
            expert: float(np.mean(counts)) if counts else 0.0
            for expert, counts in per_expert_counts.items()
        },
        "non_empty_classes": {
            expert: int(sum(1 for count in counts if count > 0))
            for expert, counts in per_expert_counts.items()
        },
        "max_items_per_expert": {
            expert: int(max(counts)) if counts else 0
            for expert, counts in per_expert_counts.items()
        },
    }


def build_expert_bank_statistics(
    phrase_banks: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    return {
        expert: {
            "size": int(len(phrases)),
            "sample_phrases": [str(phrase) for phrase in list(phrases)[:10]],
        }
        for expert, phrases in phrase_banks.items()
    }


def top_evidence_rows(
    expert_phrase_scores: Mapping[str, np.ndarray],
    phrase_banks: Mapping[str, Sequence[str]],
    image_ids: Sequence[str],
    top_m: int = 5,
) -> List[Dict[str, Any]]:
    if not image_ids:
        return []

    rows: List[Dict[str, Any]] = []
    sample_count = min(len(image_ids), next(iter(expert_phrase_scores.values())).shape[0])
    for sample_idx in range(sample_count):
        record: Dict[str, Any] = {
            "image_id": str(image_ids[sample_idx]),
        }
        for expert in EXPERT_NAMES:
            scores = np.asarray(expert_phrase_scores.get(expert), dtype=np.float32)
            phrases = list(phrase_banks.get(expert, []))
            if scores.shape[1] == 0 or not phrases:
                record[f"{expert}_top"] = []
                continue
            top_k = min(max(1, int(top_m)), scores.shape[1])
            idx = np.argpartition(-scores[sample_idx], kth=top_k - 1)[:top_k]
            order = idx[np.argsort(-scores[sample_idx, idx])]
            record[f"{expert}_top"] = [
                {
                    "phrase": phrases[int(bank_idx)],
                    "score": float(scores[sample_idx, int(bank_idx)]),
                }
                for bank_idx in order.tolist()
            ]
        rows.append(record)
    return rows
