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
        concepts_file: Optional[str] = None,
        intent2concepts_file: Optional[str] = None,
        concept_prompt_template: str = "A photo of {}.",
        use_cls_patch_concat: bool = True,
        l2_normalize_visual: bool = True,
        use_visual_adapter: bool = True,
        adapter_hidden_ratio: float = 0.25,
        adapter_dropout: float = 0.1,
        renormalize_after_adapter: bool = True,
        concept_temperature: float = 0.2,
        graph_use_relu: bool = False,
        graph_temperature: float = 1.0,
        graph_topk: int = 16,
        graph_residual_alpha: float = 0.5,
        base_dropout: float = 0.1,
        base_hidden_dim: Optional[int] = None,
        base_layer_idx: Optional[int] = None,
        alpha_init: float = 0.0,
        cls_mean_patch_ckpt_path: Optional[str] = None,
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
        self.use_visual_adapter = use_visual_adapter
        self.renormalize_after_adapter = renormalize_after_adapter
        self.concept_temperature = max(float(concept_temperature), 1e-6)
        self.graph_use_relu = graph_use_relu
        self.graph_residual_alpha = float(min(max(graph_residual_alpha, 0.0), 1.0))
        self.base_layer_idx = int(base_layer_idx) if base_layer_idx is not None else None

        projected_dim = self._get_projected_dim()
        self.visual_dim = projected_dim * (2 if self.use_cls_patch_concat else 1)
        self.backbone_hidden_dim = int(getattr(self.backbone, "width", self.backbone.conv1.out_channels))
        self.base_feature_dim = self.backbone_hidden_dim * 2
        self.base_hidden_dim = int(base_hidden_dim) if base_hidden_dim is not None else projected_dim

        if not (hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "resblocks")):
            raise ValueError("Could not find transformer.resblocks in CLIP visual backbone.")
        self.base_num_layers = len(self.backbone.transformer.resblocks)
        if self.base_layer_idx is None:
            self.base_layer_idx = self.base_num_layers
        if self.base_layer_idx < 1 or self.base_layer_idx > self.base_num_layers:
            raise ValueError(
                f"base_layer_idx must be in range [1, {self.base_num_layers}], got {self.base_layer_idx}"
            )

        if self.use_visual_adapter:
            adapter_hidden_dim = max(int(self.visual_dim * float(adapter_hidden_ratio)), 64)
            self.visual_adapter = nn.Sequential(
                nn.Linear(self.visual_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(adapter_dropout)),
                nn.Linear(adapter_hidden_dim, self.visual_dim),
            )
            # Start close to identity: v' = v + Adapter(v), with Adapter initially near 0.
            nn.init.zeros_(self.visual_adapter[-1].weight)
            nn.init.zeros_(self.visual_adapter[-1].bias)
        else:
            self.visual_adapter = None

        # Base branch (cls_mean_patch style): z_base = MLP([CLS; mean_patch])
        self.base_head = nn.Sequential(
            nn.Linear(self.base_feature_dim, self.base_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(base_dropout)),
            nn.Linear(self.base_hidden_dim, num_classes),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        if cls_mean_patch_ckpt_path:
            self._load_base_head_from_cls_mean_patch_ckpt(cls_mean_patch_ckpt_path)

        concept_entries = None
        if intent2concepts_file is not None:
            concept_entries = self._load_intent_concepts_entries(intent2concepts_file)

        if concepts_file is not None:
            concepts = self._load_concept_list(concepts_file)
        elif concept_entries is not None and len(concept_entries) > 0:
            concepts = self._build_concept_list_from_entries(concept_entries)
        else:
            concepts = concept_list if concept_list else DEFAULT_CONCEPTS
        self.concept_list = [str(c) for c in concepts]
        self.num_concepts = len(self.concept_list)

        concept_embeddings = self._build_concept_embeddings(concept_prompt_template)
        self.register_buffer("concept_embeddings", concept_embeddings, persistent=True)

        concept_graph = self._build_concept_graph(
            temperature=graph_temperature,
            topk=graph_topk,
        )
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

    def _load_base_head_from_cls_mean_patch_ckpt(self, ckpt_path: str) -> None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[ICRN] Failed to load cls_mean_patch checkpoint {ckpt_path}: {exc}")
            return

        state_dict = ckpt.get("state_dict", ckpt)
        if not isinstance(state_dict, dict):
            print(f"[ICRN] Invalid checkpoint format in {ckpt_path}, skip base head init.")
            return

        base_state = self.base_head.state_dict()
        loaded = 0

        candidate_prefixes = ("net.heads.", "net._orig_mod.heads.")
        for key, tensor in base_state.items():
            matched_tensor = None
            for prefix in candidate_prefixes:
                candidate_key = prefix + key
                if candidate_key in state_dict:
                    matched_tensor = state_dict[candidate_key]
                    break
            if matched_tensor is None:
                continue
            if matched_tensor.shape != tensor.shape:
                continue
            base_state[key] = matched_tensor.detach().to(dtype=tensor.dtype)
            loaded += 1

        if loaded > 0:
            self.base_head.load_state_dict(base_state)
            print(f"[ICRN] Loaded {loaded} base_head tensors from {ckpt_path}")
        else:
            print(f"[ICRN] No compatible base_head tensors found in {ckpt_path}")

    def _load_concept_list(self, concepts_file: str) -> List[str]:
        with open(concepts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {concepts_file}, got {type(data).__name__}")
        concept_list = [str(x).strip() for x in data if str(x).strip()]
        if not concept_list:
            raise ValueError(f"No valid concepts found in {concepts_file}")
        return concept_list

    def _load_intent_concepts_entries(self, intent2concepts_file: str) -> List[dict]:
        with open(intent2concepts_file, "r", encoding="utf-8") as f:
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
            raise ValueError(
                f"Expected a list in {intent2concepts_file}, got {type(data).__name__}"
            )

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

    def _build_concept_graph(self, temperature: float = 1.0, topk: int = 0) -> torch.Tensor:
        sim = torch.matmul(self.concept_embeddings, self.concept_embeddings.t())
        scale = max(float(temperature), 1e-6)
        num_concepts = sim.shape[-1]
        k = int(topk)
        if k <= 0 or k >= num_concepts:
            return F.softmax(sim / scale, dim=-1)

        topk_vals, topk_idx = torch.topk(sim, k=k, dim=-1)
        sparse_logits = torch.full_like(sim, -1e4)
        sparse_logits.scatter_(1, topk_idx, topk_vals)
        return F.softmax(sparse_logits / scale, dim=-1)

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

    def _extract_base_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [CLS; mean_patch] features using cls_mean_patch-compatible flow."""
        backbone = self.backbone

        x = backbone.conv1(x)
        bsz, ch, h, w = x.shape
        x = x.reshape(bsz, ch, h * w).permute(0, 2, 1)

        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + backbone.positional_embedding.unsqueeze(0)
        x = backbone.ln_pre(x)

        x = x.permute(1, 0, 2)
        for i in range(self.base_layer_idx):
            x = backbone.transformer.resblocks[i](x)
        x = x.permute(1, 0, 2)

        cls_feat = x[:, 0, :]
        patch_mean = x[:, 1:, :].mean(dim=1)
        return torch.cat([cls_feat, patch_mean], dim=-1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        base_feature = self._extract_base_feature(x)
        z_base = self.base_head(base_feature)

        visual_feature = self._extract_visual_feature(x)

        concept_feature = visual_feature
        if self.visual_adapter is not None:
            concept_feature = concept_feature + self.visual_adapter(concept_feature)
            if self.renormalize_after_adapter:
                concept_feature = F.normalize(concept_feature, dim=-1)

        concept_logits = torch.matmul(concept_feature, self.concept_embeddings.t())
        concept_activation = torch.sigmoid(concept_logits)
        scaled_scores = concept_logits / self.concept_temperature
        graph_message = torch.matmul(scaled_scores, self.concept_graph.t())
        refined = (
            self.graph_residual_alpha * scaled_scores
            + (1.0 - self.graph_residual_alpha) * graph_message
        )
        if self.graph_use_relu:
            refined = F.relu(refined)

        z_concept = torch.matmul(refined, self.intent_composition) + self.intent_bias
        logits = z_base + self.alpha * z_concept

        if not return_aux:
            return logits

        aux = {
            "z_base": z_base,
            "z_concept": z_concept,
            "alpha": self.alpha,
            "concept_logits": concept_logits,
            "concept_activation": concept_activation,
            "scaled_scores": scaled_scores,
            "graph_message": graph_message,
            "refined_concepts": refined,
            "prior_target": self.prior_target,
        }
        return logits, aux
