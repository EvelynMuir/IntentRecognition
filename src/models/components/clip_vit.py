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


class ClipVisionTransformer(nn.Module):
    """CLIP Vision Transformer model for multi-label classification.
    
    This class wraps CLIP's visual encoder (ViT) as a backbone.
    """

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
        clip_model_name: str = "ViT-B/32",
    ) -> None:
        """Initialize CLIP Vision Transformer.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained CLIP weights.
        :param image_size: Input image size.
        :param clip_model_name: CLIP model name. Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", etc.
        """
        super().__init__()

        # Load pretrained CLIP model
        if pretrained:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
            self.clip_model.eval()
        else:
            raise ValueError("CLIP models must be pretrained. Set pretrained=True.")

        # Extract the visual encoder (ViT backbone)
        self.backbone = self.clip_model.visual
        
        # Store model configuration
        self.clip_model_name = clip_model_name
        self.image_size = image_size
        
        # Get hidden dimension from CLIP model
        # ViT-B/32: 512, ViT-B/16: 768, ViT-L/14: 1024
        if "ViT-B/32" in clip_model_name:
            self.hidden_dim = 512
        elif "ViT-B/16" in clip_model_name:
            self.hidden_dim = 768
        elif "ViT-L/14" in clip_model_name:
            self.hidden_dim = 1024
        else:
            # Try to infer from model structure
            # CLIP's visual encoder has a width attribute
            if hasattr(self.backbone, 'width'):
                self.hidden_dim = self.backbone.width
            else:
                # Fallback: try to get from transformer
                if hasattr(self.backbone, 'transformer'):
                    if hasattr(self.backbone.transformer, 'width'):
                        self.hidden_dim = self.backbone.transformer.width
                    else:
                        # Last resort: use the output dimension
                        self.hidden_dim = self.backbone.output_dim if hasattr(self.backbone, 'output_dim') else 512
                else:
                    self.hidden_dim = 512

        # Replace the classifier head for multi-label classification
        # Note: CLIP's visual encoder doesn't have a classifier head by default
        # We'll add one similar to torchvision's ViT
        self.heads = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Output logits of shape (batch_size, num_classes).
        """
        # CLIP's visual encoder forward pass
        # The visual encoder returns features of shape [batch_size, hidden_dim]
        x = self.backbone(x)
        # Apply classifier head
        x = self.heads(x)
        
        return x

