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


class ClipVisionTransformerLayerPatchMean(nn.Module):
    """CLIP Vision Transformer using patch-token mean pooling from a specific layer."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
        clip_model_name: str = "ViT-B/32",
        layer_idx: int = 24,
    ) -> None:
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

        if "ViT-B/32" in clip_model_name:
            self.hidden_dim = 512
        elif "ViT-B/16" in clip_model_name:
            self.hidden_dim = 768
        elif "ViT-L/14" in clip_model_name:
            self.hidden_dim = 1024
        else:
            if hasattr(self.backbone, "width"):
                self.hidden_dim = self.backbone.width
            elif hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "width"):
                self.hidden_dim = self.backbone.transformer.width
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

        self.heads = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, num_classes),
        )

    def _extract_patch_mean_from_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        backbone = self.backbone

        x = backbone.conv1(x)
        bsz, ch, h, w = x.shape
        x = x.reshape(bsz, ch, h * w).permute(0, 2, 1)

        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + backbone.positional_embedding.unsqueeze(0)
        x = backbone.ln_pre(x)

        x = x.permute(1, 0, 2)
        for i in range(layer_idx):
            x = backbone.transformer.resblocks[i](x)

        x = x.permute(1, 0, 2)
        patch_tokens = x[:, 1:, :]
        patch_mean = patch_tokens.mean(dim=1)
        return patch_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_mean_features = self._extract_patch_mean_from_layer(x, self.layer_idx)
        logits = self.heads(patch_mean_features)
        return logits
