from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class VisionTransformer(nn.Module):
    """Vision Transformer model for multi-label classification."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        """Initialize Vision Transformer.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained weights.
        :param image_size: Input image size.
        """
        super().__init__()

        # Load pretrained ViT-B/16
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)

        # Replace the classifier head for multi-label classification
        hidden_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Output logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)

