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


class ClipVisionTransformerLayerCls(nn.Module):
    """CLIP Vision Transformer model for multi-label classification using CLS token from a specific layer.
    
    This class wraps CLIP's visual encoder (ViT) as a backbone and extracts CLS token
    from a specified transformer layer (0-23) for classification.
    """

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        image_size: int = 224,
        clip_model_name: str = "ViT-B/32",
        layer_idx: int = 0,
    ) -> None:
        """Initialize CLIP Vision Transformer with layer-specific CLS token extraction.

        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained CLIP weights.
        :param image_size: Input image size.
        :param clip_model_name: CLIP model name. Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", etc.
        :param layer_idx: Index of transformer layer to extract CLS token from (0-23).
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
        self.layer_idx = layer_idx
        
        # Validate layer_idx
        if hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'resblocks'):
            num_layers = len(self.backbone.transformer.resblocks)
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"layer_idx must be in range [0, {num_layers-1}], got {layer_idx}")
        else:
            raise ValueError("Could not find transformer.resblocks in CLIP backbone")
        
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
            if hasattr(self.backbone, 'width'):
                self.hidden_dim = self.backbone.width
            else:
                if hasattr(self.backbone, 'transformer'):
                    if hasattr(self.backbone.transformer, 'width'):
                        self.hidden_dim = self.backbone.transformer.width
                    else:
                        self.hidden_dim = self.backbone.output_dim if hasattr(self.backbone, 'output_dim') else 512
                else:
                    self.hidden_dim = 512

        # Get output dimension after projection (if exists)
        # This is the dimension we'll use for the classifier
        if hasattr(self.backbone, 'proj') and self.backbone.proj is not None:
            # proj shape: [width, output_dim]
            self.output_dim = self.backbone.proj.shape[1]
        elif hasattr(self.backbone, 'output_dim'):
            self.output_dim = self.backbone.output_dim
        else:
            # Fallback: infer from model name
            if "ViT-L/14" in clip_model_name:
                self.output_dim = 768
            elif "ViT-B/16" in clip_model_name:
                self.output_dim = 512
            elif "ViT-B/32" in clip_model_name:
                self.output_dim = 512
            else:
                self.output_dim = 512

        # MLP classifier head
        # Input dimension is output_dim (after projection if exists, otherwise hidden_dim)
        self.heads = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, num_classes),
        )

    def _extract_cls_token_from_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract CLS token from a specific transformer layer.
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :param layer_idx: Index of transformer layer to extract from (0-23).
        :return: CLS token features of shape (batch_size, output_dim).
        """
        backbone = self.backbone
        
        # CLIP's ViT structure: conv1 -> reshape -> add CLS token -> add positional embedding -> ln_pre -> transformer -> ln_post -> proj
        # 1. Patch embedding (conv1)
        x = backbone.conv1(x)  # [B, hidden_dim, H', W']
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N_patches, hidden_dim]
        
        # 2. Add CLS token
        batch_size = x.shape[0]
        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N_patches, hidden_dim]
        
        # 3. Add positional embedding
        x = x + backbone.positional_embedding.unsqueeze(0)  # [B, 1+N_patches, hidden_dim]
        
        # 4. Pre-layer norm
        x = backbone.ln_pre(x)  # [B, 1+N_patches, hidden_dim]
        
        # 5. Through transformer up to specified layer
        # CLIP's transformer expects input as [N, B, hidden_dim] format (seq_len, batch, hidden_dim)
        x = x.permute(1, 0, 2)  # [1+N_patches, B, hidden_dim]
        
        # Forward through transformer blocks up to layer_idx
        for i in range(layer_idx + 1):
            x = backbone.transformer.resblocks[i](x)  # [1+N_patches, B, hidden_dim]
        
        # 6. Extract CLS token (first token)
        x = x.permute(1, 0, 2)  # [B, 1+N_patches, hidden_dim]
        cls_token_features = x[:, 0, :]  # [B, hidden_dim]
        
        # # 7. Apply post-layer norm if exists (but only if we're at the last layer)
        # # For intermediate layers, we don't apply ln_post
        # is_last_layer = layer_idx == len(backbone.transformer.resblocks) - 1
        # if is_last_layer:
        #     if hasattr(backbone, 'ln_post'):
        #         cls_token_features = backbone.ln_post(cls_token_features.unsqueeze(1)).squeeze(1)  # [B, hidden_dim]
            
        #     # 8. Apply projection if exists
        #     # For the last layer, we apply projection to ensure dimension consistency
        #     # The projection layer maps hidden_dim to output_dim, which is needed for the classifier head
        #     if hasattr(backbone, 'proj') and backbone.proj is not None:
        #         # proj shape: [width, output_dim]
        #         cls_token_features = cls_token_features @ backbone.proj  # [B, output_dim]
        
        return cls_token_features  # [B, output_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Output logits of shape (batch_size, num_classes).
        """
        # Extract CLS token from specified layer
        cls_token_features = self._extract_cls_token_from_layer(x, self.layer_idx)  # [B, output_dim]
        
        # Apply classifier head
        logits = self.heads(cls_token_features)  # [B, num_classes]
        
        return logits

