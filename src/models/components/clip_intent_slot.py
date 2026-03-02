from __future__ import annotations

import json
import math
import os
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

# Default Intentonomy intent descriptions
INTENTONOMY_DESCRIPTIONS = [
    "Being good looking, attractive.",
    "Beat people in a competition.",
    "To communicate or express myself.",
    "Being creative (e.g., artistically, scientifically, intellectually). Being unique or different.",
    "Exploration - Being curious and adventurous. Having an exciting, stimulating life.",
    "Having an easy and comfortable life.",
    "Enjoying life.",
    "Appreciating fine design (man-made wonders like architectures).",
    "Appreciating fine design (artwork).",
    "Appreciating other cultures.",
    "Being a good parent (teaching, transmitting values). Being emotionally close to my children.",
    "Being happy and content. Feeling satisfied with one's life. Feeling good about myself.",
    "Being ambitious, hard-working.",
    "Achieving harmony and oneness (with self and the universe).",
    "Being physically active, fit, healthy, e.g. maintaining a healthy weight, eating nutritious foods. To be physically able to do my daily/routine activities. Having athletic ability.",
    "Being in love.",
    "Being in love with animal.",
    "Inspiring others, Influencing, persuading others.",
    "To keep things manageable. To make plans.",
    "Experiencing natural beauty.",
    "Being really passionate about something.",
    "Being playful, carefree, lighthearted.",
    "Sharing my feelings with others.",
    "Being part of a social group. Having people to do things with. Having close friends, others to rely on. Making friends, drawing others near.",
    "Being successful in my occupation. Having a good job.",
    "Teaching others.",
    "Keeping things in order (my desk, office, house, etc.).",
    "Having work I really like."
]


def _load_intent_descriptions_from_annotation(annotation_file: Optional[str]) -> List[str]:
    if annotation_file is None or not os.path.exists(annotation_file):
        return []

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    if not categories:
        return []

    def _cat_key(cat: dict) -> int:
        if "id" in cat:
            return int(cat["id"])
        if "category_id" in cat:
            return int(cat["category_id"])
        return 0

    ordered = sorted(categories, key=_cat_key)
    return [str(cat.get("name", f"class_{idx}")) for idx, cat in enumerate(ordered)]


def _load_text_queries_from_gemini(gemini_file: str) -> List[List[str]]:
    """Load text queries from gemini JSON file.
    
    Args:
        gemini_file: Path to intent_description_gemini.json
        
    Returns:
        List of lists, where each inner list contains all Text Query strings for one intent
    """
    if not os.path.exists(gemini_file):
        raise FileNotFoundError(f"Gemini description file not found: {gemini_file}")
    
    with open(gemini_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_text_queries = []
    
    for intent_item in data:
        intent_queries = []
        descriptions = intent_item.get("description", [])
        
        for desc in descriptions:
            text_query = desc.get("Text Query", "")
            if text_query:
                intent_queries.append(text_query)
        
        all_text_queries.append(intent_queries)
    
    return all_text_queries


def _compute_intent_embeddings_from_gemini(
    clip_model: nn.Module,
    gemini_file: str,
    num_classes: int,
) -> torch.Tensor:
    """Compute intent embeddings by averaging text query embeddings.
    
    Args:
        clip_model: CLIP model for text encoding
        gemini_file: Path to intent_description_gemini.json
        num_classes: Number of intent classes
        
    Returns:
        Tensor of shape (num_classes, embedding_dim) with averaged embeddings
    """
    text_queries_per_intent = _load_text_queries_from_gemini(gemini_file)
    
    embeddings_list = []
    
    for intent_idx, queries in enumerate(text_queries_per_intent[:num_classes]):
        if not queries:
            # If no queries found, use a default query
            queries = [f"class_{intent_idx}"]
        
        tokenized = clip.tokenize(queries)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized).float()
        
        # Compute mean embedding across all text queries for this intent
        mean_embedding = text_features.mean(dim=0)
        mean_embedding = F.normalize(mean_embedding, dim=-1)
        embeddings_list.append(mean_embedding)
    
    return torch.stack(embeddings_list, dim=0)


def build_intent_descriptions(
    num_classes: int,
    intent_descriptions: Optional[List[str]] = None,
    annotation_file: Optional[str] = None,
    gemini_file: Optional[str] = None,
    mode: str = "detailed",
) -> List[str]:
    """Build intent descriptions based on mode.
    
    Args:
        num_classes: Number of classes
        intent_descriptions: Explicit list of descriptions (overrides mode)
        annotation_file: Path to annotation file for 'short' mode
        gemini_file: Path to gemini JSON file for 'llm' mode
        mode: Description mode - 'generic', 'short', 'detailed', or 'llm'
            - 'generic': Uses class_0, class_1, etc.
            - 'short': Uses names from annotation file (Attractive, BeatCompete, etc.)
            - 'detailed': Uses detailed descriptions (Being good looking, attractive., etc.)
            - 'llm': Placeholder descriptions (actual embeddings computed separately)
    """
    if intent_descriptions is not None and len(intent_descriptions) > 0:
        descriptions = [str(x) for x in intent_descriptions]
    elif mode == "generic":
        # Use generic class_0, class_1, etc.
        descriptions = [f"class_{i}" for i in range(num_classes)]
    elif mode == "short":
        # Use short names from annotation file
        descriptions = _load_intent_descriptions_from_annotation(annotation_file)
        if not descriptions:
            descriptions = [f"class_{i}" for i in range(num_classes)]
    elif mode == "detailed":
        # Use detailed Intentonomy descriptions
        if num_classes == len(INTENTONOMY_DESCRIPTIONS):
            descriptions = INTENTONOMY_DESCRIPTIONS.copy()
        else:
            descriptions = []
    elif mode == "llm":
        # Use placeholder descriptions for llm mode
        # Actual embeddings will be computed from gemini.json
        descriptions = [f"intent_{i}" for i in range(num_classes)]
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'generic', 'short', 'detailed', or 'llm'.")

    if len(descriptions) < num_classes:
        descriptions = descriptions + [f"class_{i}" for i in range(len(descriptions), num_classes)]
    return descriptions[:num_classes]


class CLIPBackbone(nn.Module):
    def __init__(
        self,
        clip_visual: nn.Module,
        selected_layers: List[int],
        aligned_dim: int,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = clip_visual
        self.selected_layers = selected_layers
        self.use_cls_token = use_cls_token

        if not hasattr(self.backbone, "transformer") or not hasattr(
            self.backbone.transformer, "resblocks"
        ):
            raise ValueError("Current CLIP visual backbone does not expose transformer.resblocks.")

        self.num_blocks = len(self.backbone.transformer.resblocks)
        if any(layer < 1 or layer > self.num_blocks for layer in selected_layers):
            raise ValueError(
                f"selected_layers={selected_layers} out of range [1, {self.num_blocks}]."
            )

        width = int(self.backbone.conv1.out_channels)
        self.aligners = nn.ModuleList([nn.Linear(width, aligned_dim) for _ in selected_layers])

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone.conv1(images)
        bsz, ch, h, w = x.shape
        x = x.reshape(bsz, ch, h * w).permute(0, 2, 1)

        cls_token = self.backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.positional_embedding.unsqueeze(0)
        x = self.backbone.ln_pre(x)

        x = x.permute(1, 0, 2)
        selected = {}
        for idx, block in enumerate(self.backbone.transformer.resblocks, start=1):
            x = block(x)
            if idx in self.selected_layers:
                tokens = x.permute(1, 0, 2)
                selected[idx] = tokens if self.use_cls_token else tokens[:, 1:, :]

        outputs = []
        for i, layer_idx in enumerate(self.selected_layers):
            outputs.append(self.aligners[i](selected[layer_idx]))
        return outputs


class TokenAggregator(nn.Module):
    def __init__(self, dim: int, num_layers: int, use_layernorm: bool = True) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def forward(self, tokens_per_layer: List[torch.Tensor]) -> torch.Tensor:
        normalized = [self.norms[i](tok) for i, tok in enumerate(tokens_per_layer)]
        return torch.cat(normalized, dim=1)


class IntentQueryBank(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        intent_descriptions: List[str],
        trainable_queries: bool = True,
        prompt_template: str = "A photo that expresses the intent of {}.",
        pre_computed_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if pre_computed_embeddings is not None:
            # Use pre-computed embeddings (e.g., from llm mode)
            text_features = pre_computed_embeddings
        else:
            # Compute embeddings from descriptions
            prompts = [prompt_template.format(desc) for desc in intent_descriptions]
            tokenized = clip.tokenize(prompts)
            with torch.no_grad():
                text_features = clip_model.encode_text(tokenized).float()
                text_features = F.normalize(text_features, dim=-1)

        if trainable_queries:
            self.intent_queries = nn.Parameter(text_features.clone())
        else:
            self.register_buffer("intent_queries", text_features)

    def get_queries(self) -> torch.Tensor:
        return self.intent_queries


class IntentConditionedSlotAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_slots: int = 4,
        iters: int = 3,
        use_intent_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.use_intent_conditioning = use_intent_conditioning

        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.slot_sigma = nn.Parameter(torch.ones(1, num_slots, dim) * 0.1)

        self.to_q_slot = nn.Linear(dim, dim)
        self.to_q_intent = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, tokens: torch.Tensor, intent_query: torch.Tensor) -> torch.Tensor:
        bsz = tokens.size(0)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        slots = self.slot_mu + torch.randn(
            bsz, self.num_slots, self.dim, device=tokens.device, dtype=tokens.dtype
        ) * self.slot_sigma

        scale = 1.0 / math.sqrt(self.dim)
        q_intent = self.to_q_intent(intent_query).unsqueeze(1)

        for _ in range(self.iters):
            q_slot = self.to_q_slot(slots)
            if self.use_intent_conditioning:
                q_slot = q_slot + q_intent  # intent modulates slots

            logits = torch.matmul(q_slot, k.transpose(-1, -2)) * scale

            attn = F.softmax(logits, dim=-1)
            updates = torch.matmul(attn, v)
            slots = self.norm_slots(slots + updates)
            slots = slots + self.ffn(self.norm_mlp(slots))

        return slots


class IntentClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        num_intents: int,
        num_slots: int = 4,
        slot_iters: int = 3,
        use_intent_conditioning: bool = True,
        use_learnable_temperature: bool = False,
        init_temperature: float = 1.0,
        use_cls_fusion: bool = False,
        init_alpha: float = 1.0,
        use_proto_classifier: bool = False,
        proto_momentum: float = 0.99,
        use_decoupled_classifier: bool = False,
    ) -> None:
        super().__init__()
        self.num_intents = num_intents
        self.hidden_dim = dim
        self.use_cls_fusion = use_cls_fusion
        self.use_proto_classifier = use_proto_classifier
        self.use_decoupled_classifier = use_decoupled_classifier
        self.slot_attention = IntentConditionedSlotAttention(
            dim=dim,
            num_slots=num_slots,
            iters=slot_iters,
            use_intent_conditioning=use_intent_conditioning,
        )
        if use_cls_fusion:
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(init_alpha)))
        if use_learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(float(init_temperature)))

        # ===== PROTOTYPE-BASED CLASSIFIER =====
        if use_proto_classifier:
            self.register_buffer(
                "intent_proto",
                torch.zeros(self.num_intents, self.hidden_dim)
            )
            self.register_buffer("proto_momentum", torch.tensor(float(proto_momentum)))
            self.register_buffer("proto_initialized", torch.tensor(False, dtype=torch.bool))
        else:
            self.register_buffer(
                "intent_proto",
                torch.zeros(1)  # dummy buffer, not used
            )
            self.register_buffer("proto_momentum", torch.tensor(0.0))
            self.register_buffer("proto_initialized", torch.tensor(False, dtype=torch.bool))

        # ===== DECOUPLED CLASSIFIER MODULES (optional) =====
        if use_decoupled_classifier:
            # (Fix 1) classifier space projection
            self.intent_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

            # (Fix 2) slot importance scorer
            self.slot_score = nn.Linear(self.hidden_dim, 1)

            # (Fix 3) classifier heads after concat fusion
            fusion_dim = self.hidden_dim * 2 if self.use_cls_fusion else self.hidden_dim

            # Only create intent_classifiers if not using proto_classifier
            if not use_proto_classifier:
                self.intent_classifiers = nn.ModuleList([
                    nn.Linear(fusion_dim, 1, bias=False)
                    for _ in range(self.num_intents)
                ])
            else:
                self.intent_classifiers = None

    def init_prototypes(self, intent_queries: torch.Tensor) -> None:
        """Initialize prototypes from intent queries.
        
        Args:
            intent_queries: [num_intents, D]
        """
        if not self.use_proto_classifier:
            return
        
        with torch.no_grad():
            proto = F.normalize(intent_queries, dim=-1)
            self.intent_proto.copy_(proto)
            self.proto_initialized.fill_(True)

    def forward(
        self,
        tokens: torch.Tensor,
        cls_embed: torch.Tensor | None,
        intent_queries: torch.Tensor,
        return_slots: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: [B, N, D]
        cls_embed: [B, D]
        intent_queries: [num_intents, D] (LLM initialized)
        """

        bsz = tokens.shape[0]
        scores = []
        slots_all = []

        # Auto-initialize prototypes on first forward pass during training
        if self.use_proto_classifier and self.training and not self.proto_initialized:
            self.init_prototypes(intent_queries)

        if self.use_cls_fusion:
            if cls_embed is None:
                raise ValueError("cls_embed required when use_cls_fusion=True")
            cls_n = F.normalize(cls_embed, dim=-1)

        # ============================================================
        # LOOP OVER INTENTS
        # ============================================================

        if self.use_decoupled_classifier:
            # ========================================================
            # NEW DECOUPLED CLASSIFIER METHOD
            # ========================================================
            for idx in range(self.num_intents):

                # --------------------------------------------------------
                # Fix 1: separate extractor query & classifier query
                # --------------------------------------------------------
                q_extract = intent_queries[idx]              # LLM prior
                q_extract = q_extract.unsqueeze(0).expand(bsz, -1)

                q_class = self.intent_proj(q_extract)        # learnable classifier space
                qn = F.normalize(q_class, dim=-1)

                # --------------------------------------------------------
                # SLOT ATTENTION (unchanged logic)
                # --------------------------------------------------------
                slots = self.slot_attention(tokens, q_extract)
                # slots: [B, K, D]

                sn = F.normalize(slots, dim=-1)

                # --------------------------------------------------------
                # Fix 2: slot weighting independent of query
                # --------------------------------------------------------
                attn_logits = self.slot_score(slots).squeeze(-1)  # [B, K]
                attn = F.softmax(attn_logits, dim=-1)

                slot_summary = torch.sum(
                    attn.unsqueeze(-1) * slots,
                    dim=1
                )  # [B, D]

                slot_summary = F.normalize(slot_summary, dim=-1)

                # --------------------------------------------------------
                # Fix 3: CLS fusion via CONCAT (not addition)
                # --------------------------------------------------------
                if self.use_cls_fusion:
                    fused = torch.cat([cls_n, slot_summary], dim=-1)
                else:
                    fused = slot_summary

                # --------------------------------------------------------
                # classification head (decoupled from query)
                # --------------------------------------------------------
                if self.use_proto_classifier:
                    # Prototype-based classification with momentum update
                    with torch.no_grad():
                        batch_proto = slot_summary.mean(dim=0)  # [D]
                        if self.training:
                            self.intent_proto[idx] = (
                                self.proto_momentum * self.intent_proto[idx]
                                + (1 - self.proto_momentum) * batch_proto
                            )
                    
                    proto = F.normalize(self.intent_proto[idx], dim=-1)
                    score = (slot_summary * proto).sum(dim=-1)
                else:
                    # Learnable classifier head
                    score = self.intent_classifiers[idx](fused).squeeze(-1)

                scores.append(score)

                if return_slots:
                    slots_all.append(slots)
        else:
            # ========================================================
            # ORIGINAL METHOD (backward compatibility)
            # ========================================================
            for idx in range(self.num_intents):
                q = intent_queries[idx].unsqueeze(0).expand(bsz, -1)
                qn = F.normalize(q, dim=-1)

                slots = self.slot_attention(tokens, q)
                sn = F.normalize(slots, dim=-1)
                
                if self.use_cls_fusion:
                    attn_logits = (sn * qn.unsqueeze(1)).sum(dim=-1)
                    attn = F.softmax(attn_logits, dim=-1)
                    slot_summary = (attn.unsqueeze(-1) * slots).sum(dim=1)

                    fused = cls_n + self.alpha * F.normalize(slot_summary, dim=-1)
                    score = (fused * qn).sum(dim=-1)
                else:
                    score = (sn * qn.unsqueeze(1)).sum(dim=-1).sum(dim=-1)
                
                scores.append(score)
                if return_slots:
                    slots_all.append(slots)

        # ============================================================
        logits = torch.stack(scores, dim=1)
        logits = logits / self.temperature.clamp_min(1e-4)

        if return_slots:
            return logits, torch.stack(slots_all, dim=1)

        return logits, None


class CLIPIntentSlotModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 28,
        clip_model_name: str = "ViT-B/32",
        selected_layers: Optional[List[int]] = None,
        use_cls_token: bool = False,
        aggregator_layernorm: bool = True,
        num_slots: int = 4,
        slot_iters: int = 3,
        use_intent_conditioning: bool = True,
        intent_descriptions: Optional[List[str]] = None,
        intent_annotation_file: Optional[str] = None,
        intent_description_mode: str = "detailed",
        intent_gemini_file: Optional[str] = None,
        trainable_intent_queries: bool = True,
        freeze_backbone: bool = True,
        use_learnable_temperature: bool = False,
        init_temperature: float = 1.0,
        use_cls_fusion: bool = False,
        init_alpha: float = 1.0,
        use_proto_classifier: bool = False,
        proto_momentum: float = 0.99,
        use_decoupled_classifier: bool = False,
    ) -> None:
        super().__init__()
        if selected_layers is None:
            selected_layers = [6, 9, 12]

        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.backbone = self.clip_model.visual
        if not freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True

        with torch.no_grad():
            text_dim = self.clip_model.encode_text(clip.tokenize(["dummy"])).shape[-1]

        descriptions = build_intent_descriptions(
            num_classes=num_classes,
            intent_descriptions=intent_descriptions,
            annotation_file=intent_annotation_file,
            gemini_file=intent_gemini_file,
            mode=intent_description_mode,
        )

        # Compute pre-computed embeddings for llm mode
        pre_computed_embeddings = None
        if intent_description_mode == "llm" and intent_gemini_file is not None:
            pre_computed_embeddings = _compute_intent_embeddings_from_gemini(
                clip_model=self.clip_model,
                gemini_file=intent_gemini_file,
                num_classes=num_classes,
            )

        self.clip_backbone = CLIPBackbone(
            clip_visual=self.backbone,
            selected_layers=selected_layers,
            aligned_dim=text_dim,
            use_cls_token=use_cls_token,
        )
        self.aggregator = TokenAggregator(
            dim=text_dim,
            num_layers=len(selected_layers),
            use_layernorm=aggregator_layernorm,
        )
        self.intent_bank = IntentQueryBank(
            clip_model=self.clip_model,
            intent_descriptions=descriptions,
            trainable_queries=trainable_intent_queries,
            pre_computed_embeddings=pre_computed_embeddings,
        )
        self.intent_classifier = IntentClassifier(
            dim=text_dim,
            num_intents=num_classes,
            num_slots=num_slots,
            slot_iters=slot_iters,
            use_intent_conditioning=use_intent_conditioning,
            use_learnable_temperature=use_learnable_temperature,
            init_temperature=init_temperature,
            use_cls_fusion=use_cls_fusion,
            init_alpha=init_alpha,
            use_proto_classifier=use_proto_classifier,
            proto_momentum=proto_momentum,
            use_decoupled_classifier=use_decoupled_classifier,
        )

    def forward(
        self, images: torch.Tensor, return_slots: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens_per_layer = self.clip_backbone(images)
        tokens = self.aggregator(tokens_per_layer)
        if self.intent_classifier.use_cls_fusion:
            if self.clip_backbone.use_cls_token:
                cls_embed = tokens_per_layer[-1][:, 0, :]
            else:
                cls_embed = tokens.mean(dim=1)
        else:
            cls_embed = None
        intent_queries = self.intent_bank.get_queries()
        return self.intent_classifier(
            tokens=tokens,
            cls_embed=cls_embed,
            intent_queries=intent_queries,
            return_slots=return_slots,
        )

