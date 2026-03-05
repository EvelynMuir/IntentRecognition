"""CLIP ViT with Intent Concept Reasoning Network (ICRN) head for Intentonomy."""

from __future__ import annotations

import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is not installed. Please install it with: "
        "pip install git+https://github.com/openai/CLIP.git"
    ) from exc

from src.models.intentonomy_clip_vit_slot_module import (
    INTENTONOMY_DESCRIPTIONS,
    _compute_intent_embeddings_from_gemini,
)


DEFAULT_CONCEPTS = [
    "hugging",
    "holding hands",
    "handshake",
    "high five",
    "group gathering",
    "conversation",
    "eye contact",
    "smiling",
    "laughing",
    "crying",
    "comforting",
    "supporting",
    "arguing",
    "celebrating",
    "teaching",
    "learning",
    "performing",
    "presenting",
    "collaborating",
    "helping another person",
    "greeting",
    "waving",
    "apologizing",
    "thanking",
    "encouraging",
    "cheering",
    "playing together",
    "dancing",
    "singing",
    "sports activity",
    "running",
    "walking",
    "sitting together",
    "working at desk",
    "reading",
    "writing",
    "using computer",
    "using phone",
    "taking photo",
    "eating",
    "drinking",
    "cooking",
    "shopping",
    "traveling",
    "driving",
    "cycling",
    "outdoor scene",
    "indoor scene",
    "home environment",
    "office environment",
    "classroom environment",
    "nature background",
    "city background",
    "family interaction",
    "friendship",
    "romantic interaction",
    "caregiving",
    "leadership",
    "teamwork",
    "competition",
    "achievement",
    "creativity",
    "curiosity",
    "exploration",
    "planning",
    "organizing",
    "problem solving",
    "focus",
    "relaxation",
    "playfulness",
    "confidence",
    "joy",
    "sadness",
    "surprise",
    "anger",
    "fear",
    "calmness",
    "gratitude",
    "affection",
    "physical contact",
    "social support",
    "emotional support",
    "shared activity",
    "group attention",
    "public setting",
    "private setting",
    "formal setting",
    "informal setting",
    "event participation",
    "ceremony",
    "celebration event",
    "training session",
    "meeting",
    "presentation stage",
    "artistic activity",
    "volunteering",
    "community service",
    "help seeking",
    "mutual assistance",
]


class ClipVisionTransformerICRN(nn.Module):
    """ICRN model with frozen CLIP visual encoder and concept reasoning head."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
        clip_model_name: str = "ViT-L/14",
        concept_list: Optional[List[str]] = None,
        intent_concepts_gemini_file: Optional[str] = None,
        concept_prompt_template: str = "A photo of {}.",
        use_cls_patch_concat: bool = True,
        l2_normalize_visual: bool = True,
        graph_use_relu: bool = False,
        graph_temperature: float = 1.0,
        init_with_llm_prior: bool = False,
        intent_gemini_file: Optional[str] = None,
        intent_descriptions: Optional[List[str]] = None,
        intent_prompt_template: str = "A photo that expresses the intent of {}.",
    ) -> None:
        super().__init__()

        if not pretrained:
            raise ValueError("CLIP models must be pretrained. Set pretrained=True.")

        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.backbone = self.clip_model.visual
        self.clip_model_name = clip_model_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cls_patch_concat = use_cls_patch_concat
        self.l2_normalize_visual = l2_normalize_visual
        self.graph_use_relu = graph_use_relu

        projected_dim = self._get_projected_dim()
        self.visual_dim = projected_dim * (2 if self.use_cls_patch_concat else 1)

        concept_entries = None
        if intent_concepts_gemini_file is not None:
            concept_entries = self._load_intent_concepts_entries(intent_concepts_gemini_file)

        if concept_entries is not None and len(concept_entries) > 0:
            concepts = self._build_concept_list_from_entries(concept_entries)
        else:
            concepts = concept_list if concept_list else DEFAULT_CONCEPTS
        self.concept_list = [str(c) for c in concepts]
        self.num_concepts = len(self.concept_list)

        concept_embeddings = self._build_concept_embeddings(concept_prompt_template)
        self.register_buffer("concept_embeddings", concept_embeddings, persistent=True)

        concept_graph = self._build_concept_graph(temperature=graph_temperature)
        self.register_buffer("concept_graph", concept_graph, persistent=True)

        self.intent_bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))
        self.intent_composition = nn.Parameter(
            torch.empty(self.num_concepts, num_classes, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.intent_composition)

        self.register_buffer("intent_concept_prior_scores", None, persistent=True)
        if concept_entries is not None and len(concept_entries) > 0:
            prior_scores = self._build_prior_from_intent_concepts(
                concept_entries=concept_entries,
                concept_list=self.concept_list,
                num_classes=self.num_classes,
            )
            self.intent_concept_prior_scores = prior_scores

        self.register_buffer("prior_target", None, persistent=True)
        if init_with_llm_prior:
            prior_target = None
            if self.intent_concept_prior_scores is not None:
                prior_target = self.intent_concept_prior_scores.t().contiguous()
            elif intent_gemini_file is not None:
                prior_target = self._build_prior_target(
                    intent_gemini_file=intent_gemini_file,
                    intent_descriptions=intent_descriptions,
                    intent_prompt_template=intent_prompt_template,
                )

            if prior_target is not None:
                self.prior_target = prior_target
                with torch.no_grad():
                    self.intent_composition.copy_(prior_target)

    def _load_intent_concepts_entries(self, gemini_file: str) -> List[dict]:
        with open(gemini_file, "r", encoding="utf-8") as f:
            raw = f.read()

        # Prefer strict JSON parsing; fall back to recovery for partially concatenated files.
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
            raise ValueError(f"Expected a list in {gemini_file}, got {type(data).__name__}")

        normalized_entries = []
        for item in data:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", ""))
            concepts = item.get("concepts", {})
            if not isinstance(concepts, dict):
                continue
            clean_concepts = {}
            for key, value in concepts.items():
                try:
                    clean_concepts[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if intent and clean_concepts:
                normalized_entries.append({"intent": intent, "concepts": clean_concepts})
        return normalized_entries

    def _build_concept_list_from_entries(self, concept_entries: List[dict]) -> List[str]:
        concept_list = []
        seen = set()
        for item in concept_entries:
            concepts = item.get("concepts", {})
            for concept in concepts.keys():
                concept_name = str(concept)
                if concept_name in seen:
                    continue
                seen.add(concept_name)
                concept_list.append(concept_name)
        return concept_list if concept_list else DEFAULT_CONCEPTS

    def _build_prior_from_intent_concepts(
        self,
        concept_entries: List[dict],
        concept_list: List[str],
        num_classes: int,
    ) -> torch.Tensor:
        concept_to_idx = {name: idx for idx, name in enumerate(concept_list)}
        scores = torch.zeros(num_classes, len(concept_list), dtype=torch.float32)

        for intent_idx, item in enumerate(concept_entries[:num_classes]):
            concepts = item.get("concepts", {})
            for concept_name, value in concepts.items():
                idx = concept_to_idx.get(str(concept_name))
                if idx is None:
                    continue
                scores[intent_idx, idx] = float(value)
        return scores

    def _get_projected_dim(self) -> int:
        if hasattr(self.backbone, "proj") and self.backbone.proj is not None:
            return int(self.backbone.proj.shape[1])
        if hasattr(self.backbone, "output_dim"):
            return int(self.backbone.output_dim)
        if "ViT-L/14" in self.clip_model_name:
            return 768
        return 512

    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokenized = clip.tokenize(prompts)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized).float()
        return F.normalize(text_features, dim=-1)

    def _build_concept_embeddings(self, prompt_template: str) -> torch.Tensor:
        prompts = [prompt_template.format(concept) for concept in self.concept_list]
        text_features = self._encode_text(prompts)

        text_dim = int(text_features.shape[-1])
        if text_dim == self.visual_dim:
            return text_features

        # Keep this projection fixed to preserve frozen-text prior semantics.
        proj = torch.zeros(text_dim, self.visual_dim, dtype=text_features.dtype)
        shared_dim = min(text_dim, self.visual_dim)
        proj[:shared_dim, :shared_dim] = torch.eye(shared_dim, dtype=text_features.dtype)
        projected = text_features @ proj
        return F.normalize(projected, dim=-1)

    def _build_concept_graph(self, temperature: float = 1.0) -> torch.Tensor:
        sim = torch.matmul(self.concept_embeddings, self.concept_embeddings.t())
        scale = max(float(temperature), 1e-6)
        return F.softmax(sim / scale, dim=-1)

    def _resolve_intent_descriptions(
        self,
        intent_descriptions: Optional[List[str]],
    ) -> List[str]:
        if intent_descriptions and len(intent_descriptions) > 0:
            descriptions = [str(x) for x in intent_descriptions]
        elif self.num_classes == len(INTENTONOMY_DESCRIPTIONS):
            descriptions = INTENTONOMY_DESCRIPTIONS.copy()
        else:
            descriptions = [f"class_{i}" for i in range(self.num_classes)]

        if len(descriptions) < self.num_classes:
            descriptions.extend([f"class_{i}" for i in range(len(descriptions), self.num_classes)])
        return descriptions[: self.num_classes]

    def _build_prior_target(
        self,
        intent_gemini_file: str,
        intent_descriptions: Optional[List[str]],
        intent_prompt_template: str,
    ) -> torch.Tensor:
        with torch.no_grad():
            try:
                intent_embeddings = _compute_intent_embeddings_from_gemini(
                    clip_model=self.clip_model,
                    gemini_file=intent_gemini_file,
                    num_classes=self.num_classes,
                ).float()
            except Exception:
                descriptions = self._resolve_intent_descriptions(intent_descriptions)
                prompts = [intent_prompt_template.format(desc) for desc in descriptions]
                intent_embeddings = self._encode_text(prompts)

            text_dim = int(intent_embeddings.shape[-1])
            if text_dim != self.visual_dim:
                proj = torch.zeros(text_dim, self.visual_dim, dtype=intent_embeddings.dtype)
                shared_dim = min(text_dim, self.visual_dim)
                proj[:shared_dim, :shared_dim] = torch.eye(shared_dim, dtype=intent_embeddings.dtype)
                intent_embeddings = intent_embeddings @ proj
                intent_embeddings = F.normalize(intent_embeddings, dim=-1)

            sim = torch.matmul(intent_embeddings, self.concept_embeddings.t())
            return sim.t().contiguous()

    def _extract_visual_feature(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone

        x = backbone.conv1(x)
        bsz, ch, h, w = x.shape
        x = x.reshape(bsz, ch, h * w).permute(0, 2, 1)

        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + backbone.positional_embedding.unsqueeze(0)
        x = backbone.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = backbone.transformer(x)
        x = x.permute(1, 0, 2)

        if hasattr(backbone, "ln_post"):
            x = backbone.ln_post(x)

        cls_feat = x[:, 0:1, :]
        patch_feat = x[:, 1:, :]

        if hasattr(backbone, "proj") and backbone.proj is not None:
            cls_feat = cls_feat @ backbone.proj
            patch_feat = patch_feat @ backbone.proj

        cls_feat = cls_feat.squeeze(1)
        patch_mean = patch_feat.mean(dim=1)

        if self.use_cls_patch_concat:
            visual_feature = torch.cat([cls_feat, patch_mean], dim=-1)
        else:
            visual_feature = patch_mean

        if self.l2_normalize_visual:
            visual_feature = F.normalize(visual_feature, dim=-1)

        return visual_feature

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        visual_feature = self._extract_visual_feature(x)

        concept_logits = torch.matmul(visual_feature, self.concept_embeddings.t())
        concept_activation = torch.sigmoid(concept_logits)

        refined = torch.matmul(concept_activation, self.concept_graph.t())
        if self.graph_use_relu:
            refined = F.relu(refined)

        logits = torch.matmul(refined, self.intent_composition) + self.intent_bias

        if not return_aux:
            return logits

        aux = {
            "concept_activation": concept_activation,
            "refined_concepts": refined,
            "prior_target": self.prior_target,
        }
        return logits, aux
