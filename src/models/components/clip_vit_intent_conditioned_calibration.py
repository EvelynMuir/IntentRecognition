"""CLIP ViT with Intent-Conditioned Feature Calibration for Intentonomy."""

from __future__ import annotations

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


class ClipVisionTransformerIntentConditionedCalibration(nn.Module):
    """CLIP ViT architecture based on Intent-Conditioned Feature Calibration."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
        clip_model_name: str = "ViT-L/14",
        layer_idx: int = 24,
        intent_description_mode: str = "llm",
        intent_gemini_file: Optional[str] = None,
        intent_descriptions: Optional[List[str]] = None,
        prompt_template: str = "A photo that expresses the intent of {}.",
        lambda_init: float = 0.1,
        use_vector_lambda: bool = False,
        mlp_hidden_dim: Optional[int] = None,
        use_logit_correlation: bool = False,
    ) -> None:
        super().__init__()

        if not pretrained:
            raise ValueError("CLIP models must be pretrained. Set pretrained=True.")

        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        self.clip_model.eval()

        # Strictly freeze both visual and text encoders.
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.backbone = self.clip_model.visual
        self.clip_model_name = clip_model_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.layer_idx = layer_idx
        self.use_vector_lambda = use_vector_lambda
        self.use_logit_correlation = use_logit_correlation

        if not (hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "resblocks")):
            raise ValueError("Could not find transformer.resblocks in CLIP backbone")

        self.num_layers = len(self.backbone.transformer.resblocks)
        if layer_idx < 1 or layer_idx > self.num_layers:
            raise ValueError(f"layer_idx must be in range [1, {self.num_layers}], got {layer_idx}")

        self.hidden_dim = self._get_visual_hidden_dim()
        with torch.no_grad():
            self.text_dim = int(self.clip_model.encode_text(clip.tokenize(["dummy"])).shape[-1])
        self.text_to_hidden_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self._init_text_to_hidden_proj()

        initial_queries = self._initialize_intent_queries(
            num_classes=num_classes,
            mode=intent_description_mode,
            gemini_file=intent_gemini_file,
            intent_descriptions=intent_descriptions,
            prompt_template=prompt_template,
        )

        # Learnable intent query matrix in text space.
        self.intent_queries = nn.Parameter(initial_queries.clone(), requires_grad=True)

        # Learnable residual calibration scalar or per-intent vector.
        if self.use_vector_lambda:
            self.lambda_val = nn.Parameter(
                torch.full((self.num_classes,), float(lambda_init), dtype=torch.float32)
            )
        else:
            self.lambda_val = nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.float32))
        # Temperature for cosine-attention logits (CLIP-style init).
        self.attn_logit_scale = nn.Parameter(torch.tensor(2.6592, dtype=torch.float32))

        if mlp_hidden_dim is None:
            mlp_hidden_dim = self.hidden_dim

        # Shared lightweight 2-layer MLP for per-intent logits.
        self.heads = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, 1),
        )
        self.logit_correlation = nn.Linear(self.num_classes, self.num_classes) if self.use_logit_correlation else None
        if self.logit_correlation is not None:
            self._init_logit_correlation_layer()

    def _get_visual_hidden_dim(self) -> int:
        if "ViT-B/32" in self.clip_model_name:
            return 512
        if "ViT-B/16" in self.clip_model_name:
            return 768
        if "ViT-L/14" in self.clip_model_name:
            return 1024
        if hasattr(self.backbone, "width"):
            return int(self.backbone.width)
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "width"):
            return int(self.backbone.transformer.width)
        return 512

    def _build_fallback_descriptions(self, num_classes: int) -> List[str]:
        if num_classes == len(INTENTONOMY_DESCRIPTIONS):
            return INTENTONOMY_DESCRIPTIONS.copy()
        return [f"class_{i}" for i in range(num_classes)]

    def _init_text_to_hidden_proj(self) -> None:
        """Initialize projection as identity-preserving map (learnable)."""
        with torch.no_grad():
            self.text_to_hidden_proj.weight.zero_()
            self.text_to_hidden_proj.bias.zero_()
            shared_dim = min(self.text_dim, self.hidden_dim)
            self.text_to_hidden_proj.weight[:shared_dim, :shared_dim] = torch.eye(shared_dim)

    def _init_logit_correlation_layer(self) -> None:
        """Initialize logits correlation layer near identity."""
        if self.logit_correlation is None:
            return
        with torch.no_grad():
            self.logit_correlation.weight.copy_(torch.eye(self.num_classes))
            self.logit_correlation.bias.zero_()

    def _initialize_intent_queries(
        self,
        num_classes: int,
        mode: str,
        gemini_file: Optional[str],
        intent_descriptions: Optional[List[str]],
        prompt_template: str,
    ) -> torch.Tensor:
        with torch.no_grad():
            if mode == "llm" and gemini_file is not None:
                text_features = _compute_intent_embeddings_from_gemini(
                    clip_model=self.clip_model,
                    gemini_file=gemini_file,
                    num_classes=num_classes,
                )
            else:
                descriptions = (
                    [str(x) for x in intent_descriptions]
                    if intent_descriptions
                    else self._build_fallback_descriptions(num_classes)
                )
                prompts = [prompt_template.format(desc) for desc in descriptions[:num_classes]]
                tokenized = clip.tokenize(prompts)
                text_features = self.clip_model.encode_text(tokenized).float()

            return F.normalize(text_features, dim=-1)

    def _extract_cls_and_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS and patch tokens from the specified transformer layer."""
        backbone = self.backbone

        x = backbone.conv1(x)
        bsz, ch, h, w = x.shape
        x = x.reshape(bsz, ch, h * w).permute(0, 2, 1)

        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + backbone.positional_embedding.unsqueeze(0)
        x = backbone.ln_pre(x)

        x = x.permute(1, 0, 2)
        for i in range(self.layer_idx):
            x = backbone.transformer.resblocks[i](x)
        x = x.permute(1, 0, 2)

        cls_token_features = x[:, 0, :]
        patch_tokens = x[:, 1:, :]

        return cls_token_features, patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with intent-conditioned attention and residual calibration."""
        cls_token, patch_tokens = self._extract_cls_and_patches(x)

        # Global mean feature f_g: [B, D]
        f_g = patch_tokens.mean(dim=1)

        # Project text-space queries to visual hidden dim.
        projected_queries = self.text_to_hidden_proj(self.intent_queries)

        # Normalize patch tokens and intent queries before attention.
        q_norm = F.normalize(projected_queries, dim=-1)
        p_norm = F.normalize(patch_tokens, dim=-1)

        # Cosine-attention logits with learnable temperature.
        logit_scale = self.attn_logit_scale.exp().clamp(max=100.0)
        attn_scores = torch.einsum("id,bnd->bin", q_norm, p_norm) * logit_scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        f_attn = torch.einsum("bin,bnd->bid", attn_weights, patch_tokens)

        # Residual calibration: f_calibrated = f_g + lambda * (f_attn - f_g)
        f_g_expanded = f_g.unsqueeze(1)
        if self.use_vector_lambda:
            lambda_term = self.lambda_val.view(1, self.num_classes, 1)
        else:
            lambda_term = self.lambda_val
        f_calibrated = f_g_expanded + lambda_term * (f_attn - f_g_expanded)

        # Fuse expanded CLS token with calibrated intent features.
        cls_expanded = cls_token.unsqueeze(1).expand(-1, self.num_classes, -1)
        fused = torch.cat([cls_expanded, f_calibrated], dim=-1)

        logits = self.heads(fused).squeeze(-1)
        if self.logit_correlation is not None:
            logits = self.logit_correlation(logits)
        return logits
