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


def _resolve_selected_experts(selected_experts: Sequence[str] | None = None) -> List[str]:
    experts = list(selected_experts) if selected_experts is not None else list(EXPERT_NAMES)
    experts = [expert for expert in experts if expert in EXPERT_NAMES]
    if not experts:
        raise ValueError("selected_experts must include at least one valid expert.")
    return experts


def _build_global_profile_mask(
    relation_matrices: Mapping[str, np.ndarray],
    selected_experts: Sequence[str] | None = None,
    top_n: int | None = None,
    by_abs: bool = False,
) -> Dict[str, np.ndarray]:
    experts = _resolve_selected_experts(selected_experts)
    masks = {
        expert: np.ones_like(np.asarray(relation_matrices[expert], dtype=np.float32), dtype=np.float32)
        for expert in experts
    }
    if top_n is None:
        return masks

    keep_n = int(top_n)
    if keep_n <= 0:
        return {
            expert: np.zeros_like(np.asarray(relation_matrices[expert], dtype=np.float32), dtype=np.float32)
            for expert in experts
        }

    class_count = next(iter(masks.values())).shape[0]
    total_elements = int(sum(mask.shape[1] for mask in masks.values()))
    if keep_n >= total_elements:
        return masks

    for expert in experts:
        masks[expert].fill(0.0)

    offsets: Dict[str, tuple[int, int]] = {}
    start = 0
    for expert in experts:
        width = int(masks[expert].shape[1])
        offsets[expert] = (start, start + width)
        start += width

    for class_idx in range(class_count):
        ranking_chunks: List[np.ndarray] = []
        raw_chunks: List[np.ndarray] = []
        for expert in experts:
            values = np.asarray(relation_matrices[expert][class_idx], dtype=np.float32)
            raw_chunks.append(values)
            ranking_chunks.append(np.abs(values) if by_abs else values)

        ranking_values = np.concatenate(ranking_chunks, axis=0)
        raw_values = np.concatenate(raw_chunks, axis=0)
        if by_abs:
            candidate_ids = np.where(ranking_values > 0.0)[0]
            candidate_scores = ranking_values[candidate_ids]
        else:
            candidate_ids = np.where(raw_values > 0.0)[0]
            candidate_scores = raw_values[candidate_ids]

        if candidate_ids.size == 0:
            continue

        if candidate_ids.size <= keep_n:
            chosen_ids = candidate_ids
        else:
            order = np.argsort(-candidate_scores)[:keep_n]
            chosen_ids = candidate_ids[order]

        for expert in experts:
            left, right = offsets[expert]
            local_ids = chosen_ids[(chosen_ids >= left) & (chosen_ids < right)] - left
            if local_ids.size == 0:
                continue
            masks[expert][class_idx, local_ids] = 1.0

    return masks


def _topk_row_sparsify(matrix: np.ndarray, top_k: int | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if top_k is None:
        return values.copy()
    if values.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {values.shape}")
    if values.shape[1] == 0:
        return values.copy()

    keep_k = max(0, min(int(top_k), values.shape[1]))
    output = np.zeros_like(values, dtype=np.float32)
    if keep_k == 0:
        return output
    if keep_k >= values.shape[1]:
        return values.copy()

    row_idx = np.arange(values.shape[0])[:, None]
    top_idx = np.argpartition(-values, kth=keep_k - 1, axis=1)[:, :keep_k]
    output[row_idx, top_idx] = values[row_idx, top_idx]
    return output


def _build_hard_negative_ids_from_positive_means(
    positive_mean: Mapping[str, np.ndarray],
    selected_experts: Sequence[str] | None = None,
    top_n: int = 3,
) -> List[List[int]]:
    experts = _resolve_selected_experts(selected_experts)
    if top_n <= 0:
        return [[] for _ in range(next(iter(positive_mean.values())).shape[0])]

    stacked = np.concatenate(
        [np.asarray(positive_mean[expert], dtype=np.float32) for expert in experts],
        axis=1,
    )
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    normalized = stacked / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    keep_n = min(int(top_n), max(similarity.shape[1] - 1, 0))
    output: List[List[int]] = []
    for class_idx in range(similarity.shape[0]):
        if keep_n <= 0:
            output.append([])
            continue
        order = np.argsort(-similarity[class_idx])[:keep_n]
        output.append([int(idx) for idx in order.tolist()])
    return output


def learn_data_driven_relations(
    expert_phrase_scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    selected_experts: Sequence[str] | None = None,
    relation_mode: str = "pos_neg_diff",
    profile_topn: int | None = 10,
    hard_negative_topn: int = 3,
    hard_negative_ids: Sequence[Sequence[int]] | None = None,
    positive_only_scores: bool = True,
) -> Dict[str, Any]:
    experts = _resolve_selected_experts(selected_experts)
    label_matrix = (np.asarray(labels, dtype=np.float32) > 0.0).astype(np.float32)
    if label_matrix.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {label_matrix.shape}")

    class_count = label_matrix.shape[1]
    pos_counts = np.maximum(label_matrix.sum(axis=0, keepdims=True).T, 1.0)
    neg_indicator = 1.0 - label_matrix
    neg_counts = np.maximum(neg_indicator.sum(axis=0, keepdims=True).T, 1.0)

    positive_mean: Dict[str, np.ndarray] = {}
    negative_mean: Dict[str, np.ndarray] = {}
    for expert in experts:
        scores = np.asarray(expert_phrase_scores[expert], dtype=np.float32)
        if scores.ndim != 2:
            raise ValueError(f"Expected 2D expert scores for {expert}, got shape {scores.shape}")
        if scores.shape[0] != label_matrix.shape[0]:
            raise ValueError(
                f"expert_phrase_scores[{expert}] rows {scores.shape[0]} != labels rows {label_matrix.shape[0]}"
            )
        working_scores = np.maximum(scores, 0.0) if positive_only_scores else scores
        positive_mean[expert] = (label_matrix.T @ working_scores) / pos_counts
        negative_mean[expert] = (neg_indicator.T @ working_scores) / neg_counts

    if hard_negative_ids is None:
        hard_negative_rows = _build_hard_negative_ids_from_positive_means(
            positive_mean,
            selected_experts=experts,
            top_n=int(hard_negative_topn),
        )
    else:
        hard_negative_rows = [
            [int(idx) for idx in row if 0 <= int(idx) < class_count]
            for row in hard_negative_ids
        ]
        if len(hard_negative_rows) != class_count:
            raise ValueError(
                f"hard_negative_ids length {len(hard_negative_rows)} != num classes {class_count}"
            )

    hard_negative_mean: Dict[str, np.ndarray] = {}
    for expert in experts:
        rows: List[np.ndarray] = []
        expert_positive = np.asarray(positive_mean[expert], dtype=np.float32)
        for class_idx, negatives in enumerate(hard_negative_rows):
            valid_negatives = [neg for neg in negatives if neg != class_idx]
            if not valid_negatives:
                rows.append(np.zeros(expert_positive.shape[1], dtype=np.float32))
                continue
            rows.append(expert_positive[valid_negatives].mean(axis=0).astype(np.float32))
        hard_negative_mean[expert] = np.stack(rows, axis=0)

    support: Dict[str, np.ndarray] = {}
    contradiction: Dict[str, np.ndarray] = {}
    for expert in experts:
        mu_pos = np.asarray(positive_mean[expert], dtype=np.float32)
        mu_neg = np.asarray(negative_mean[expert], dtype=np.float32)
        mu_hard = np.asarray(hard_negative_mean[expert], dtype=np.float32)

        if relation_mode == "positive_mean":
            support[expert] = mu_pos.copy()
            contradiction[expert] = np.zeros_like(mu_pos, dtype=np.float32)
        elif relation_mode == "pos_neg_diff":
            diff = mu_pos - mu_neg
            support[expert] = diff
            contradiction[expert] = np.zeros_like(diff, dtype=np.float32)
        elif relation_mode == "hard_negative_diff":
            diff = mu_pos - mu_hard
            support[expert] = diff
            contradiction[expert] = np.zeros_like(diff, dtype=np.float32)
        elif relation_mode == "support_only":
            diff = mu_pos - mu_neg
            support[expert] = np.maximum(diff, 0.0)
            contradiction[expert] = np.zeros_like(diff, dtype=np.float32)
        elif relation_mode == "support_contradiction":
            diff = mu_pos - mu_neg
            support[expert] = np.maximum(diff, 0.0)
            contradiction[expert] = np.maximum(-diff, 0.0)
        else:
            raise ValueError(f"Unsupported relation_mode: {relation_mode}")

    if profile_topn is not None:
        support_masks = _build_global_profile_mask(
            support,
            selected_experts=experts,
            top_n=int(profile_topn),
            by_abs=relation_mode in {"pos_neg_diff", "hard_negative_diff"},
        )
        for expert in experts:
            support[expert] = support[expert] * support_masks[expert]

        if any(np.any(np.asarray(contradiction[expert], dtype=np.float32) > 0.0) for expert in experts):
            contradiction_masks = _build_global_profile_mask(
                contradiction,
                selected_experts=experts,
                top_n=int(profile_topn),
                by_abs=False,
            )
            for expert in experts:
                contradiction[expert] = contradiction[expert] * contradiction_masks[expert]

    return {
        "selected_experts": experts,
        "relation_mode": relation_mode,
        "profile_topn": None if profile_topn is None else int(profile_topn),
        "hard_negative_topn": int(hard_negative_topn),
        "hard_negative_ids": hard_negative_rows,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "hard_negative_mean": hard_negative_mean,
        "support": support,
        "contradiction": contradiction,
    }


def compute_data_driven_verification_scores(
    expert_phrase_scores: Mapping[str, np.ndarray],
    relation_bundle: Mapping[str, Any],
    selected_experts: Sequence[str] | None = None,
    activation_topm: int | None = 5,
    contradiction_lambda: float = 1.0,
    activation_positive_only: bool = True,
) -> np.ndarray:
    experts = _resolve_selected_experts(
        selected_experts if selected_experts is not None else relation_bundle.get("selected_experts")
    )
    support = relation_bundle.get("support", {})
    contradiction = relation_bundle.get("contradiction", {})

    first_expert = experts[0]
    first_relation = np.asarray(support[first_expert], dtype=np.float32)
    class_count = first_relation.shape[0]
    sample_count = np.asarray(expert_phrase_scores[first_expert], dtype=np.float32).shape[0]
    output = np.zeros((sample_count, class_count), dtype=np.float32)

    for expert in experts:
        sample_scores = np.asarray(expert_phrase_scores[expert], dtype=np.float32)
        active_scores = np.maximum(sample_scores, 0.0) if activation_positive_only else sample_scores
        active_scores = _topk_row_sparsify(active_scores, top_k=activation_topm)

        support_matrix = np.asarray(support[expert], dtype=np.float32)
        output += active_scores @ support_matrix.T

        contradiction_matrix = np.asarray(
            contradiction.get(expert, np.zeros_like(support_matrix, dtype=np.float32)),
            dtype=np.float32,
        )
        if contradiction_matrix.size > 0 and np.any(contradiction_matrix > 0.0):
            output -= float(contradiction_lambda) * (active_scores @ contradiction_matrix.T)

    return output.astype(np.float32)


def build_confusion_neighborhoods(
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    topk: int = 10,
    top_n: int = 3,
    use_rank_weight: bool = True,
) -> List[List[int]]:
    scores = np.asarray(candidate_logits, dtype=np.float32)
    targets = (np.asarray(labels, dtype=np.float32) > 0.0).astype(np.int32)
    if scores.shape != targets.shape:
        raise ValueError(f"candidate_logits shape {scores.shape} != labels shape {targets.shape}")

    num_samples, num_classes = scores.shape
    topk = max(1, min(int(topk), num_classes))
    top_n = max(0, min(int(top_n), max(num_classes - 1, 0)))

    topk_idx = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
    weighted_counts = np.zeros((num_classes, num_classes), dtype=np.float32)

    for sample_idx in range(num_samples):
        ordered = topk_idx[sample_idx][np.argsort(-scores[sample_idx, topk_idx[sample_idx]])]
        positive_ids = np.where(targets[sample_idx] > 0)[0].tolist()
        if not positive_ids:
            continue
        negative_candidates = [int(idx) for idx in ordered.tolist() if targets[sample_idx, int(idx)] == 0]
        if not negative_candidates:
            continue
        for pos_class in positive_ids:
            for rank, neg_class in enumerate(negative_candidates):
                weight = 1.0 / float(rank + 1) if use_rank_weight else 1.0
                weighted_counts[int(pos_class), int(neg_class)] += float(weight)

    neighborhoods: List[List[int]] = []
    for class_idx in range(num_classes):
        row = weighted_counts[class_idx].copy()
        row[class_idx] = -np.inf
        valid_ids = np.where(np.isfinite(row) & (row > 0.0))[0]
        if valid_ids.size == 0 or top_n == 0:
            neighborhoods.append([])
            continue
        order = valid_ids[np.argsort(-row[valid_ids])[:top_n]]
        neighborhoods.append([int(idx) for idx in order.tolist()])
    return neighborhoods


def build_pairwise_relation_profiles(
    relation_bundle: Mapping[str, Any],
    selected_experts: Sequence[str] | None = None,
    pair_profile_topn: int | None = 10,
    contradiction_lambda: float = 1.0,
) -> Dict[str, Any]:
    experts = _resolve_selected_experts(
        selected_experts if selected_experts is not None else relation_bundle.get("selected_experts")
    )
    support = relation_bundle.get("support", {})
    contradiction = relation_bundle.get("contradiction", {})

    pair_profiles: Dict[str, Any] = {
        "selected_experts": experts,
        "pair_profile_topn": None if pair_profile_topn is None else int(pair_profile_topn),
        "contradiction_lambda": float(contradiction_lambda),
        "profiles": {},
    }

    for expert in experts:
        support_matrix = np.asarray(support[expert], dtype=np.float32)
        contradiction_matrix = np.asarray(
            contradiction.get(
                expert,
                np.zeros_like(support_matrix, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        relation_matrix = support_matrix - float(contradiction_lambda) * contradiction_matrix
        class_count = relation_matrix.shape[0]
        expert_profiles = {
            "indices": [[np.zeros(0, dtype=np.int64) for _ in range(class_count)] for _ in range(class_count)],
            "weights": [[np.zeros(0, dtype=np.float32) for _ in range(class_count)] for _ in range(class_count)],
            "relation_matrix": relation_matrix,
        }
        for class_i in range(class_count):
            for class_j in range(class_count):
                if class_i == class_j:
                    continue
                diff = np.asarray(relation_matrix[class_i] - relation_matrix[class_j], dtype=np.float32)
                positive_ids = np.where(diff > 0.0)[0]
                if positive_ids.size == 0:
                    continue
                if pair_profile_topn is not None and positive_ids.size > int(pair_profile_topn):
                    order = np.argsort(-diff[positive_ids])[: int(pair_profile_topn)]
                    positive_ids = positive_ids[order]
                expert_profiles["indices"][class_i][class_j] = positive_ids.astype(np.int64)
                expert_profiles["weights"][class_i][class_j] = diff[positive_ids].astype(np.float32)
        pair_profiles["profiles"][expert] = expert_profiles

    return pair_profiles


def compute_pairwise_comparative_scores(
    expert_phrase_scores: Mapping[str, np.ndarray],
    pairwise_profiles: Mapping[str, Any],
    candidate_logits: np.ndarray,
    selected_experts: Sequence[str] | None = None,
    candidate_topk: int = 10,
    activation_topm: int | None = 5,
    activation_positive_only: bool = True,
    aggregate_mode: str = "mean",
) -> np.ndarray:
    experts = _resolve_selected_experts(
        selected_experts if selected_experts is not None else pairwise_profiles.get("selected_experts")
    )
    base_logits = np.asarray(candidate_logits, dtype=np.float32)
    num_samples, num_classes = base_logits.shape
    candidate_topk = max(1, min(int(candidate_topk), num_classes))

    active_scores: Dict[str, np.ndarray] = {}
    for expert in experts:
        scores = np.asarray(expert_phrase_scores[expert], dtype=np.float32)
        if scores.shape[0] != num_samples:
            raise ValueError(
                f"expert_phrase_scores[{expert}] rows {scores.shape[0]} != candidate rows {num_samples}"
            )
        working = np.maximum(scores, 0.0) if activation_positive_only else scores
        active_scores[expert] = _topk_row_sparsify(working, top_k=activation_topm)

    output = np.zeros((num_samples, num_classes), dtype=np.float32)
    topk_idx = np.argpartition(-base_logits, kth=candidate_topk - 1, axis=1)[:, :candidate_topk]

    for sample_idx in range(num_samples):
        ordered = topk_idx[sample_idx][np.argsort(-base_logits[sample_idx, topk_idx[sample_idx]])]
        candidate_scores = np.zeros(ordered.shape[0], dtype=np.float32)

        for left_idx in range(len(ordered)):
            class_i = int(ordered[left_idx])
            for right_idx in range(left_idx + 1, len(ordered)):
                class_j = int(ordered[right_idx])
                margin = 0.0
                for expert in experts:
                    expert_profile = pairwise_profiles["profiles"][expert]
                    active = active_scores[expert][sample_idx]

                    idx_i = expert_profile["indices"][class_i][class_j]
                    weight_i = expert_profile["weights"][class_i][class_j]
                    score_i = float(active[idx_i] @ weight_i) if idx_i.size > 0 else 0.0

                    idx_j = expert_profile["indices"][class_j][class_i]
                    weight_j = expert_profile["weights"][class_j][class_i]
                    score_j = float(active[idx_j] @ weight_j) if idx_j.size > 0 else 0.0

                    margin += score_i - score_j

                candidate_scores[left_idx] += margin
                candidate_scores[right_idx] -= margin

        if aggregate_mode == "mean" and len(ordered) > 1:
            candidate_scores = candidate_scores / float(len(ordered) - 1)
        elif aggregate_mode != "sum":
            raise ValueError(f"Unsupported aggregate_mode: {aggregate_mode}")

        output[sample_idx, ordered] = candidate_scores.astype(np.float32)

    return output


def build_margin_aware_gate(
    candidate_logits: np.ndarray,
    mode: str = "exp",
    gamma: float = 1.0,
    tau: float = 0.5,
) -> np.ndarray:
    scores = np.asarray(candidate_logits, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"Expected [num_samples, num_classes>=2], got shape {scores.shape}")

    top2_idx = np.argpartition(-scores, kth=1, axis=1)[:, :2]
    top2_scores = np.take_along_axis(scores, top2_idx, axis=1)
    top2_scores.sort(axis=1)
    margin = top2_scores[:, 1] - top2_scores[:, 0]
    margin = np.maximum(margin, 0.0)

    if mode == "none":
        gate = np.ones(scores.shape[0], dtype=np.float32)
    elif mode == "exp":
        gate = np.exp(-float(gamma) * margin).astype(np.float32)
    elif mode == "binary":
        gate = (margin < float(tau)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported gate mode: {mode}")
    return gate.astype(np.float32)


def summarize_data_driven_profiles(
    relation_bundle: Mapping[str, Any],
    phrase_banks: Mapping[str, Sequence[str]],
    class_names: Sequence[str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    experts = _resolve_selected_experts(relation_bundle.get("selected_experts"))
    support = relation_bundle.get("support", {})
    contradiction = relation_bundle.get("contradiction", {})

    rows: List[Dict[str, Any]] = []
    for class_idx, class_name in enumerate(class_names):
        record: Dict[str, Any] = {
            "class_id": int(class_idx),
            "class_name": str(class_name),
            "support": {},
            "contradiction": {},
        }
        for expert in experts:
            phrases = list(phrase_banks.get(expert, []))
            support_values = np.asarray(support[expert][class_idx], dtype=np.float32)
            positive_ids = np.where(support_values > 0.0)[0]
            if positive_ids.size > 0:
                order = positive_ids[np.argsort(-support_values[positive_ids])[: max(1, int(top_n))]]
                record["support"][expert] = [
                    {
                        "phrase": phrases[int(idx)],
                        "weight": float(support_values[int(idx)]),
                    }
                    for idx in order.tolist()
                ]
            else:
                record["support"][expert] = []

            contradiction_matrix = np.asarray(
                contradiction.get(
                    expert,
                    np.zeros_like(np.asarray(support[expert], dtype=np.float32), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            contradiction_values = np.asarray(contradiction_matrix[class_idx], dtype=np.float32)
            contradiction_ids = np.where(contradiction_values > 0.0)[0]
            if contradiction_ids.size > 0:
                order = contradiction_ids[
                    np.argsort(-contradiction_values[contradiction_ids])[: max(1, int(top_n))]
                ]
                record["contradiction"][expert] = [
                    {
                        "phrase": phrases[int(idx)],
                        "weight": float(contradiction_values[int(idx)]),
                    }
                    for idx in order.tolist()
                ]
            else:
                record["contradiction"][expert] = []
        rows.append(record)
    return rows


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
