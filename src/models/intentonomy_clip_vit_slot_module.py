from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

try:
    import clip
except ImportError as exc:
    raise ImportError(
        "CLIP package is not installed. Please install it with: "
        "pip install git+https://github.com/openai/CLIP.git"
    ) from exc

from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.metrics import eval_test_set_both_strategies, eval_validation_set

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


class SlotAttention(nn.Module):
    """Pure Slot Attention without intent conditioning."""
    
    def __init__(self, dim: int, num_slots: int = 4, iters: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters

        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.slot_sigma = nn.Parameter(torch.ones(1, num_slots, dim) * 0.1)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, D]"""
        bsz = tokens.size(0)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        slots = self.slot_mu + torch.randn(
            bsz, self.num_slots, self.dim, device=tokens.device, dtype=tokens.dtype
        ) * self.slot_sigma

        scale = 1.0 / math.sqrt(self.dim)

        for _ in range(self.iters):
            q = self.to_q(slots)
            logits = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = F.softmax(logits, dim=-1)
            updates = torch.matmul(attn, v)
            slots = self.norm_slots(slots + updates)
            slots = slots + self.ffn(self.norm_mlp(slots))

        return slots


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


class PatchSlotClassifier(nn.Module):
    """
    Minimal Slot Experiment
    patch tokens -> slot attention -> flatten -> concat CLS -> MLP
    """

    def __init__(self, dim: int, num_intents: int, num_slots: int = 4, slot_iters: int = 3) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_intents = num_intents

        # 纯 slot attention（无 conditioning）
        self.slot_attention = SlotAttention(
            dim=dim,
            num_slots=num_slots,
            iters=slot_iters,
        )

        mlp_input_dim = (1 + num_slots) * dim
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_input_dim // 2, num_intents),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cls_embed: torch.Tensor,
        return_slots: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, N, D] patch tokens
            cls_embed: [B, D] cls embedding
            return_slots: whether to return slots for visualization

        Returns:
            logits: [B, num_intents]
            slots: [B, num_slots, D] if return_slots=True, else None
        """
        bsz = tokens.shape[0]

        # normalization（保持和 baseline 一致）
        tokens = F.normalize(tokens, dim=-1)
        cls_embed = F.normalize(cls_embed, dim=-1)

        slots = self.slot_attention(tokens)  # [B, K, D]

        slots_flat = slots.reshape(bsz, -1)

        fused = torch.cat([cls_embed, slots_flat], dim=-1)

        logits = self.mlp_head(fused)

        if return_slots:
            return logits, slots

        return logits, None


class RandomSlotClassifier(nn.Module):
    """
    Random Slot Baseline
    tokens -> random K groups -> group mean as fake slots -> flatten -> concat CLS -> MLP
    """

    def __init__(self, dim: int, num_intents: int, num_slots: int = 4) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_intents = num_intents

        mlp_input_dim = (1 + num_slots) * dim
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_input_dim // 2, num_intents),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cls_embed: torch.Tensor,
        token_perm: Optional[torch.Tensor] = None,
        return_slots: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, N, D]
            cls_embed: [B, D]
            return_slots: whether to return fake slots
        """
        bsz, n_tokens, dim = tokens.shape
        if self.num_slots > n_tokens:
            raise ValueError(
                f"num_slots ({self.num_slots}) cannot exceed number of tokens ({n_tokens})."
            )

        tokens = F.normalize(tokens, dim=-1)
        cls_embed = F.normalize(cls_embed, dim=-1)

        # Use dataset-provided fixed perm if available; otherwise random perm every forward.
        if token_perm is not None:
            if token_perm.dim() != 2:
                raise ValueError(
                    f"token_perm must be 2D [B, N], got shape {tuple(token_perm.shape)}."
                )
            if token_perm.shape[0] != bsz or token_perm.shape[1] != n_tokens:
                raise ValueError(
                    f"token_perm shape {tuple(token_perm.shape)} does not match tokens "
                    f"shape [B, N]=[{bsz}, {n_tokens}]."
                )
            perm = token_perm.to(device=tokens.device, dtype=torch.long)
        else:
            perm = torch.argsort(torch.rand(bsz, n_tokens, device=tokens.device), dim=1)

        tokens_shuffled = tokens.gather(1, perm.unsqueeze(-1).expand(-1, -1, dim))
        chunks = tokens_shuffled.chunk(self.num_slots, dim=1)
        fake_slots = torch.stack([c.mean(dim=1) for c in chunks], dim=1)

        fused = torch.cat([cls_embed, fake_slots.reshape(bsz, -1)], dim=-1)
        logits = self.mlp_head(fused)

        if return_slots:
            return logits, fake_slots
        return logits, None


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
        use_late_fusion: bool = False,
        use_single_layer_patch_slot: bool = False,
        use_random_slots: bool = False,
    ) -> None:
        super().__init__()
        self.num_intents = num_intents
        self.hidden_dim = dim
        self.use_cls_fusion = use_cls_fusion
        self.use_proto_classifier = use_proto_classifier
        self.use_decoupled_classifier = use_decoupled_classifier
        self.use_late_fusion = use_late_fusion
        self.use_single_layer_patch_slot = use_single_layer_patch_slot
        self.use_random_slots = use_random_slots

        if self.use_single_layer_patch_slot and self.use_random_slots:
            raise ValueError(
                "use_single_layer_patch_slot and use_random_slots cannot both be True."
            )

        # 如果使用新的patch-slot分类器，创建它
        if use_single_layer_patch_slot:
            self.patch_slot_classifier = PatchSlotClassifier(
                dim=dim,
                num_intents=num_intents,
                num_slots=num_slots,
                slot_iters=slot_iters,
            )
            self.random_slot_classifier = None
        elif use_random_slots:
            self.patch_slot_classifier = None
            self.random_slot_classifier = RandomSlotClassifier(
                dim=dim,
                num_intents=num_intents,
                num_slots=num_slots,
            )
        else:
            self.patch_slot_classifier = None
            self.random_slot_classifier = None
            self.slot_attention = IntentConditionedSlotAttention(
                dim=dim,
                num_slots=num_slots,
                iters=slot_iters,
                use_intent_conditioning=use_intent_conditioning,
            )
        # Alpha is initialized in the residual fusion section above for late fusion
        # For backward compatibility with fusion modes not using late fusion
        if not (use_cls_fusion and use_late_fusion):
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
        
        # ===== RESIDUAL LOGITS FUSION MODULES (optional) =====
        # When using cls_fusion with late fusion (residual logits fusion)
        if use_cls_fusion and use_late_fusion:
            # Separate heads for cls and slot logits
            self.cls_head = nn.Linear(self.hidden_dim, self.num_intents, bias=False)
            self.slot_head = nn.Linear(self.hidden_dim, self.num_intents, bias=False)
            # Learnable weight for residual fusion: logits = logits_cls + alpha * logits_slot
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.cls_head = None
            self.slot_head = None

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
        token_perm: Optional[torch.Tensor] = None,
        return_slots: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: [B, N, D]
        cls_embed: [B, D]
        intent_queries: [num_intents, D] (LLM initialized)
        """

        # 路由到新的patch-slot分类器
        if self.use_single_layer_patch_slot:
            if cls_embed is None:
                raise ValueError("cls_embed required when use_single_layer_patch_slot=True")
            return self.patch_slot_classifier(
                tokens=tokens,
                cls_embed=cls_embed,
                return_slots=return_slots,
            )
        if self.use_random_slots:
            if cls_embed is None:
                raise ValueError("cls_embed required when use_random_slots=True")
            return self.random_slot_classifier(
                tokens=tokens,
                cls_embed=cls_embed,
                token_perm=token_perm,
                return_slots=return_slots,
            )

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
            
            # Check if using late fusion (residual logits fusion)
            use_residual_logits_fusion = (self.use_cls_fusion and 
                                         self.use_late_fusion and
                                         hasattr(self, 'cls_head') and 
                                         self.cls_head is not None)
            
            if use_residual_logits_fusion:
                # ========================================================
                # RESIDUAL LOGITS FUSION: logits = logits_cls + alpha * logits_slot
                # ========================================================
                # Generate logits for all intents at once
                logits_cls_all = self.cls_head(cls_n)  # [B, num_intents]
                
                for idx in range(self.num_intents):
                    q = intent_queries[idx].unsqueeze(0).expand(bsz, -1)
                    
                    slots = self.slot_attention(tokens, q)
                    
                    # Compute slot summary via attention
                    sn = F.normalize(slots, dim=-1)
                    qn = F.normalize(q, dim=-1)
                    attn_logits = (sn * qn.unsqueeze(1)).sum(dim=-1)
                    attn = F.softmax(attn_logits, dim=-1)
                    slot_summary = (attn.unsqueeze(-1) * slots).sum(dim=1)  # [B, D]
                    
                    # Generate slot logits for this intent
                    logits_slot = self.slot_head(F.normalize(slot_summary, dim=-1))  # [B, num_intents]
                    
                    # Residual fusion: combine cls and slot logits
                    # logits = logits_cls + alpha * logits_slot
                    score = logits_cls_all[:, idx] + self.alpha * logits_slot[:, idx]
                    scores.append(score)
                    
                    if return_slots:
                        slots_all.append(slots)
            else:
                # Original fusion method
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
        use_late_fusion: bool = False,
        use_single_layer_patch_slot: bool = False,
        use_random_slots: bool = False,
    ) -> None:
        super().__init__()
        if selected_layers is None:
            selected_layers = [6, 9, 12]
        if use_cls_token:
            warnings.warn(
                "`use_cls_token=True` is ignored in CLIPIntentSlotModel. "
                "Slot methods in this module now always use patch tokens only.",
                UserWarning,
                stacklevel=2,
            )

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
            use_cls_token=False,
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
            use_late_fusion=use_late_fusion,
            use_single_layer_patch_slot=use_single_layer_patch_slot,
            use_random_slots=use_random_slots,
        )

    def forward(
        self,
        images: torch.Tensor,
        token_perm: Optional[torch.Tensor] = None,
        return_slots: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens_per_layer = self.clip_backbone(images)
        tokens = self.aggregator(tokens_per_layer)
        need_cls_embed = (
            self.intent_classifier.use_cls_fusion
            or self.intent_classifier.use_single_layer_patch_slot
            or self.intent_classifier.use_random_slots
        )
        if need_cls_embed:
            cls_embed = tokens.mean(dim=1)
        else:
            cls_embed = None
        intent_queries = self.intent_bank.get_queries()
        return self.intent_classifier(
            tokens=tokens,
            cls_embed=cls_embed,
            intent_queries=intent_queries,
            token_perm=token_perm,
            return_slots=return_slots,
        )


def slot_orthogonality_loss(slots: torch.Tensor) -> torch.Tensor:
    """Compute ||S S^T - I||_F for normalized slots."""
    # slots: [B, I, K, D] or [B, K, D]
    if slots.dim() == 3:
        slots = slots.unsqueeze(1)
    elif slots.dim() != 4:
        raise ValueError(f"Expected slots to be 3D or 4D, got shape {tuple(slots.shape)}")

    slots = F.normalize(slots, dim=-1)
    gram = torch.matmul(slots, slots.transpose(-1, -2))
    k = slots.size(-2)
    eye = torch.eye(k, device=slots.device, dtype=slots.dtype).view(1, 1, k, k)
    diff = gram - eye
    return torch.sqrt(torch.clamp((diff * diff).sum(dim=(-1, -2)), min=1e-8)).mean()


class IntentonomyClipViTSlotModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        num_classes: int = 28,
        compile: bool = False,
        criterion: Optional[torch.nn.Module] = None,
        use_slot_orthogonality: bool = True,
        slot_orthogonality_weight: float = 0.1,
        use_cls_fusion: bool = False,
        init_alpha: float = 1.0,
        use_proto_classifier: bool = False,
        proto_momentum: float = 0.99,
        use_decoupled_classifier: bool = False,
        use_decoupled_cls_fusion: Optional[bool] = None,
        use_late_fusion: bool = False,
        use_single_layer_patch_slot: bool = False,
        use_random_slots: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net
        self.num_classes = num_classes
        self.compile = compile
        self.use_slot_orthogonality = use_slot_orthogonality
        self.slot_orthogonality_weight = slot_orthogonality_weight
        self.use_cls_fusion = use_cls_fusion
        self.init_alpha = init_alpha
        self.use_proto_classifier = use_proto_classifier
        self.proto_momentum = proto_momentum
        self.use_single_layer_patch_slot = use_single_layer_patch_slot
        self.use_random_slots = use_random_slots
        if use_decoupled_cls_fusion is not None:
            warnings.warn(
                "`use_decoupled_cls_fusion` is deprecated and will be removed in a future version. "
                "Use `use_decoupled_classifier` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            effective_use_decoupled_classifier = use_decoupled_cls_fusion
        else:
            effective_use_decoupled_classifier = use_decoupled_classifier

        self.use_decoupled_classifier = effective_use_decoupled_classifier
        self.use_decoupled_cls_fusion = effective_use_decoupled_classifier
        self.use_late_fusion = use_late_fusion
    
        if use_cls_fusion and use_late_fusion:
            with torch.no_grad():
                classifier = self.net.intent_classifier
                # Only initialize alpha if using late fusion
                if hasattr(classifier, 'alpha') and isinstance(classifier.alpha, torch.nn.Parameter):
                    classifier.alpha.copy_(
                        torch.tensor(float(init_alpha), device=classifier.alpha.device)
                    )

        self.criterion = criterion if criterion is not None else AsymmetricLossOptimized()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_slot_orth_loss = MeanMetric()
        self.val_slot_orth_loss = MeanMetric()
        self.test_slot_orth_loss = MeanMetric()

        self.val_f1_mean_best = MaxMetric()
        self.val_f1_macro_best = MaxMetric()
        self.val_preds_list = []
        self.val_targets_list = []
        self.test_preds_list = []
        self.test_targets_list = []

    def forward(
        self,
        x: torch.Tensor,
        token_perm: Optional[torch.Tensor] = None,
        return_slots: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.net(x, token_perm=token_perm, return_slots=return_slots)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_f1_mean_best.reset()
        self.val_f1_macro_best.reset()
        self.val_preds_list.clear()
        self.val_targets_list.clear()
        
        # Initialize prototypes from intent queries for proto-based classifier
        if self.use_proto_classifier:
            intent_queries = self.net.intent_bank.get_queries()
            self.net.intent_classifier.init_prototypes(intent_queries)

    def model_step(
        self, batch: Dict[str, torch.Tensor], return_slots: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["image"]
        y = batch["labels"]
        token_perm = batch.get("token_perm")

        logits, slots = self.forward(x, token_perm=token_perm, return_slots=return_slots)
        cls_loss = self.criterion(logits, y)

        if self.use_slot_orthogonality and slots is not None:
            orth_loss = slot_orthogonality_loss(slots)
        else:
            orth_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        loss = cls_loss + self.slot_orthogonality_weight * orth_loss
        preds = torch.sigmoid(logits)
        return loss, orth_loss, preds, y

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, orth_loss, _, _ = self.model_step(batch, return_slots=True)
        self.train_loss(loss)
        self.train_slot_orth_loss(orth_loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/slot_orth_loss",
            self.train_slot_orth_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, orth_loss, preds, targets = self.model_step(batch, return_slots=True)
        self.val_loss(loss)
        self.val_slot_orth_loss(orth_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/slot_orth_loss",
            self.val_slot_orth_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if len(self.val_preds_list) == 0:
            return
        val_preds_all = torch.cat(self.val_preds_list, dim=0).numpy()
        val_targets_all = torch.cat(self.val_targets_list, dim=0).numpy()
        f1_dict = eval_validation_set(val_preds_all, val_targets_all)
        val_f1_mean = (f1_dict["val_micro"] + f1_dict["val_macro"] + f1_dict["val_samples"]) / 3.0

        self.val_f1_mean_best(val_f1_mean)
        self.val_f1_macro_best(f1_dict["val_macro"])
        self.log("val/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
        self.log("val/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
        self.log("val/f1_samples", f1_dict["val_samples"], sync_dist=True)
        self.log("val/f1_mean", val_f1_mean, sync_dist=True, prog_bar=True)
        self.log("val/f1_mean_best", self.val_f1_mean_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
        self.log("val/threshold", f1_dict["threshold"], sync_dist=True)
        self.log("val/easy", f1_dict["val_easy"], sync_dist=True)
        self.log("val/medium", f1_dict["val_medium"], sync_dist=True)
        self.log("val/hard", f1_dict["val_hard"], sync_dist=True)

        self.val_preds_list.clear()
        self.val_targets_list.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, orth_loss, preds, targets = self.model_step(batch, return_slots=True)
        self.test_loss(loss)
        self.test_slot_orth_loss(orth_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/slot_orth_loss",
            self.test_slot_orth_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if len(self.test_preds_list) == 0:
            return
        test_preds_all = torch.cat(self.test_preds_list, dim=0).numpy()
        test_targets_all = torch.cat(self.test_targets_list, dim=0).numpy()
        dual_f1_dict = eval_test_set_both_strategies(test_preds_all, test_targets_all)
        for strategy_name, metrics in dual_f1_dict.items():
            self.log(f"test/{strategy_name}/f1_micro", metrics["val_micro"], sync_dist=True, prog_bar=True)
            self.log(f"test/{strategy_name}/f1_macro", metrics["val_macro"], sync_dist=True, prog_bar=True)
            self.log(f"test/{strategy_name}/f1_samples", metrics["val_samples"], sync_dist=True)
            self.log(f"test/{strategy_name}/f1_mean", (metrics["val_micro"] + metrics["val_macro"] + metrics["val_samples"]) / 3.0, sync_dist=True)
            self.log(f"test/{strategy_name}/mAP", metrics["val_mAP"], sync_dist=True, prog_bar=True)
            self.log(f"test/{strategy_name}/threshold", metrics["threshold"], sync_dist=True)
            self.log(f"test/{strategy_name}/easy", metrics["val_easy"], sync_dist=True)
            self.log(f"test/{strategy_name}/medium", metrics["val_medium"], sync_dist=True)
            self.log(f"test/{strategy_name}/hard", metrics["val_hard"], sync_dist=True)

        # Backward-compatible aliases (legacy behavior = no inference strategy)
        f1_dict = dual_f1_dict["no_inference_strategy"]
        self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
        self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
        self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
        self.log("test/f1_mean", (f1_dict["val_micro"] + f1_dict["val_macro"] + f1_dict["val_samples"]) / 3.0, sync_dist=True)
        self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
        self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
        self.log("test/easy", f1_dict["val_easy"], sync_dist=True)
        self.log("test/medium", f1_dict["val_medium"], sync_dist=True)
        self.log("test/hard", f1_dict["val_hard"], sync_dist=True)

        self.test_preds_list.clear()
        self.test_targets_list.clear()

    def setup(self, stage: str) -> None:
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        params_to_optimize = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=params_to_optimize)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1_mean",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
