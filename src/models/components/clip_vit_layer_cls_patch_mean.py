"""CLIP Vision Transformer using CLS token + patch mean pooling from a specific layer."""

from typing import Optional

import torch
import torch.nn as nn

try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP package is not installed. Please install it with: "
        "pip install git+https://github.com/openai/CLIP.git"
    )


class ClipVisionTransformerLayerClsPatchMean(nn.Module):
    """CLIP Vision Transformer using concatenation of CLS token and mean-pooled patch tokens from a specific layer."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
        clip_model_name: str = "ViT-B/32",
        layer_idx: int = 24,
    ) -> None:
        """Initialize CLIP Vision Transformer with CLS + patch mean token extraction.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained CLIP weights.
        :param image_size: Input image size.
        :param clip_model_name: CLIP model name. Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", etc.
        :param layer_idx: Index of transformer layer to extract tokens from (default: 24 for ViT-L/14).
        """
        super().__init__()

        if pretrained:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
            self.clip_model.eval()
        else:
            raise ValueError("CLIP models must be pretrained. Set pretrained=True.")

        self.backbone = self.clip_model.visual
        self.clip_model_name = clip_model_name
        self.image_size = image_size
        self.layer_idx = layer_idx

        if not (hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "resblocks")):
            raise ValueError("Could not find transformer.resblocks in CLIP backbone")

        self.num_layers = len(self.backbone.transformer.resblocks)
        if layer_idx < 1 or layer_idx > self.num_layers:
            raise ValueError(f"layer_idx must be in range [1, {self.num_layers}], got {layer_idx}")

        # Rely on the loaded CLIP backbone instead of model-name heuristics. In
        # OpenAI CLIP, ViT-B/32 projects to 512-dim output but its transformer
        # hidden width is 768, so hard-coding from model name is error-prone.
        if hasattr(self.backbone, "width"):
            self.hidden_dim = self.backbone.width
        elif hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "width"):
            self.hidden_dim = self.backbone.transformer.width
        else:
            if "ViT-L/14" in clip_model_name:
                self.hidden_dim = 1024
            elif "ViT-B/16" in clip_model_name or "ViT-B/32" in clip_model_name:
                self.hidden_dim = 768
            else:
                self.hidden_dim = getattr(self.backbone, "output_dim", 512)

        if hasattr(self.backbone, "proj") and self.backbone.proj is not None:
            self.output_dim = self.backbone.proj.shape[1]
        elif hasattr(self.backbone, "output_dim"):
            self.output_dim = self.backbone.output_dim
        else:
            if "ViT-L/14" in clip_model_name:
                self.output_dim = 768
            else:
                self.output_dim = 512

        # Concatenated feature dimension: hidden_dim (cls) + hidden_dim (mean patch) = 2 * hidden_dim
        self.concat_dim = self.hidden_dim * 2

        # MLP classifier head
        # Input dimension is concatenated feature dimension
        self.heads = nn.Sequential(
            nn.Linear(self.concat_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, num_classes),
        )

    def _extract_cls_and_patch_mean_from_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract CLS token and mean-pooled patch tokens from a specific transformer layer.

        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :param layer_idx: Index of transformer layer to extract from.
        :return: Concatenated features of shape (batch_size, 2 * hidden_dim).
        """
        backbone = self.backbone

        # Patch embedding (conv1)
        x = backbone.conv1(x)
        bsz, ch, h, w = x.shape
        x = x.reshape(bsz, ch, h * w).permute(0, 2, 1)

        # Add CLS token
        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embedding
        x = x + backbone.positional_embedding.unsqueeze(0)
        x = backbone.ln_pre(x)

        # Forward through transformer blocks up to layer_idx
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        for i in range(layer_idx):
            x = backbone.transformer.resblocks[i](x)

        x = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        # Extract CLS token (first token)
        cls_token_features = x[:, 0, :]  # [batch_size, hidden_dim]

        # Extract patch tokens and compute mean
        patch_tokens = x[:, 1:, :]  # [batch_size, num_patches, hidden_dim]
        patch_mean_features = patch_tokens.mean(dim=1)  # [batch_size, hidden_dim]

        # Concatenate CLS token and mean patch token
        concat_features = torch.cat([cls_token_features, patch_mean_features], dim=1)  # [batch_size, 2*hidden_dim]
        return concat_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Output logits of shape (batch_size, num_classes).
        """
        concat_features = self._extract_cls_and_patch_mean_from_layer(x, self.layer_idx)
        logits = self.heads(concat_features)
        return logits
