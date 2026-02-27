"""Attention map extraction utilities for CLIP ViT models."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionExtractor:
    """Extract attention maps from CLIP ViT transformer layers.
    
    This class uses forward hooks to capture attention weights from
    the self-attention layers in CLIP's Vision Transformer.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize the attention extractor.
        
        :param model: The CLIP ViT model (IntentonomyClipViTModule).
        """
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        
        # Get the CLIP ViT backbone
        self.backbone = model.net.backbone
        
        # Register hooks for all transformer layers
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks on all transformer layers."""
        # CLIP's transformer structure: backbone.transformer.resblocks[i].attn
        if not hasattr(self.backbone, 'transformer'):
            raise ValueError("Model does not have transformer attribute")
        
        transformer = self.backbone.transformer
        if not hasattr(transformer, 'resblocks'):
            raise ValueError("Transformer does not have resblocks attribute")
        
        # Register hook for each transformer block's attention layer
        for i, resblock in enumerate(transformer.resblocks):
            if hasattr(resblock, 'attn'):
                attn_module = resblock.attn
                
                # Wrap the forward method to capture attention weights
                self._wrap_attention_forward(attn_module, i)
    
    def _wrap_attention_forward(self, attn_module: nn.Module, layer_idx: int) -> None:
        """Wrap the attention module's forward to capture attention weights.
        
        :param attn_module: The attention module.
        :param layer_idx: Index of the layer.
        """
        original_forward = attn_module.forward
        
        def forward_with_attention(*args, **kwargs):
            """Forward pass that captures attention weights."""
            # Get input tensor
            x = args[0] if args else kwargs.get('x')
            
            # CLIP's attention module structure
            # CLIP transformer uses [seq_len, batch, embed_dim] format
            # We need to manually compute attention weights
            try:
                # Determine input format
                original_format = None
                if x.dim() == 3:
                    dim0, dim1, embed_dim = x.shape
                    # CLIP uses [seq_len, batch, embed_dim] where seq_len > batch typically
                    if dim0 >= dim1:  # Usually seq_len >= batch_size
                        seq_len, batch_size, embed_dim = x.shape
                        original_format = 'seq_batch'
                        # Convert to [batch, seq_len, embed_dim] for easier processing
                        x_batch = x.transpose(0, 1)  # [batch, seq_len, embed_dim]
                    else:
                        batch_size, seq_len, embed_dim = x.shape
                        original_format = 'batch_seq'
                        x_batch = x
                elif x.dim() == 2:
                    seq_len, embed_dim = x.shape
                    batch_size = 1
                    original_format = 'seq_only'
                    x_batch = x.unsqueeze(1).transpose(0, 1)  # [1, seq_len, embed_dim]
                else:
                    raise ValueError(f"Unexpected input dimension: {x.dim()}")
                
                # Get Q, K, V projections
                # CLIP attention modules use separate q_proj, k_proj, v_proj
                if hasattr(attn_module, 'q_proj'):
                    # Custom CLIP attention structure (separate projections)
                    # Projections work on [seq_len, batch, embed_dim] format
                    if original_format == 'seq_batch':
                        q = attn_module.q_proj(x)  # [seq_len, batch, embed_dim]
                        k = attn_module.k_proj(x)
                        v = attn_module.v_proj(x)
                        # Convert to [batch, seq_len, embed_dim]
                        q = q.transpose(0, 1)  # [batch, seq_len, embed_dim]
                        k = k.transpose(0, 1)
                        v = v.transpose(0, 1)
                    elif original_format == 'batch_seq':
                        q = attn_module.q_proj(x.transpose(0, 1)).transpose(0, 1)
                        k = attn_module.k_proj(x.transpose(0, 1)).transpose(0, 1)
                        v = attn_module.v_proj(x.transpose(0, 1)).transpose(0, 1)
                    else:  # seq_only
                        q = attn_module.q_proj(x).unsqueeze(1).transpose(0, 1)
                        k = attn_module.k_proj(x).unsqueeze(1).transpose(0, 1)
                        v = attn_module.v_proj(x).unsqueeze(1).transpose(0, 1)
                elif hasattr(attn_module, 'in_proj_weight'):
                    # Standard MultiheadAttention structure
                    qkv = F.linear(x_batch, attn_module.in_proj_weight, attn_module.in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)
                else:
                    # Fallback: try to call original forward and see if it returns attention
                    result = original_forward(*args, **kwargs)
                    if isinstance(result, tuple) and len(result) == 2:
                        output, attn_weights = result
                        self.attention_maps[layer_idx] = attn_weights.detach().cpu()
                    return result
                
                # Get number of heads and head dimension
                if hasattr(attn_module, 'num_heads'):
                    num_heads = attn_module.num_heads
                else:
                    # Try to infer from weight shapes
                    if hasattr(attn_module, 'q_proj'):
                        embed_dim = attn_module.q_proj.weight.shape[0]
                        num_heads = getattr(attn_module, 'num_heads', embed_dim // 64)
                    elif hasattr(attn_module, 'in_proj_weight'):
                        embed_dim = attn_module.in_proj_weight.shape[0] // 3
                        num_heads = getattr(attn_module, 'num_heads', embed_dim // 64)
                    else:
                        num_heads = 8  # Fallback
                
                head_dim = embed_dim // num_heads
                
                # Now q, k, v are in [batch, seq_len, embed_dim] format
                # Reshape Q, K, V for multi-head attention
                q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, L, D]
                k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, L, D]
                v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, L, D]
                
                # Compute attention scores
                scale = (head_dim ** -0.5)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Store attention weights
                self.attention_maps[layer_idx] = attn_weights.detach().cpu()
                
                # Apply attention to values
                attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
                
                # Apply output projection if exists
                if hasattr(attn_module, 'c_proj'):
                    # CLIP uses c_proj, which expects [seq_len, batch, embed_dim]
                    attn_output_seq = attn_output.transpose(0, 1)  # [seq_len, batch, embed_dim]
                    attn_output_seq = attn_module.c_proj(attn_output_seq)
                    attn_output = attn_output_seq.transpose(0, 1)  # Back to [batch, seq_len, embed_dim]
                elif hasattr(attn_module, 'out_proj'):
                    attn_output = attn_module.out_proj(attn_output)
                
                # Convert back to original format
                if original_format == 'seq_batch':
                    attn_output = attn_output.transpose(0, 1)  # [seq_len, batch, embed_dim]
                elif original_format == 'seq_only':
                    attn_output = attn_output.squeeze(0).transpose(0, 1)  # [seq_len, embed_dim]
                # If batch_seq, keep as is
                
                return attn_output
            except Exception as e:
                # If anything fails, fall back to original forward
                import warnings
                warnings.warn(f"Failed to extract attention for layer {layer_idx}: {e}. Using original forward.")
                import traceback
                warnings.warn(traceback.format_exc())
                return original_forward(*args, **kwargs)
        
        # Replace the forward method
        attn_module.forward = forward_with_attention
        self.hooks.append((attn_module, original_forward))
    
    def extract_attention_maps(
        self, 
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """Extract attention maps for given input.
        
        :param x: Input tensor of shape (batch_size, 3, H, W).
        :param layer_indices: List of layer indices to extract. If None, extract all layers.
        :return: Dictionary mapping layer index to attention weights.
        """
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Set model to eval mode
        self.model.eval()
        
        # Forward pass (this will trigger wrapped forward methods)
        with torch.no_grad():
            _ = self.model(x)
        
        # Filter by layer_indices if specified
        if layer_indices is not None:
            return {idx: self.attention_maps[idx] for idx in layer_indices if idx in self.attention_maps}
        
        return self.attention_maps.copy()
    
    def get_cls_attention_map(
        self,
        attention_weights: torch.Tensor,
        image_size: int = 224,
        patch_size: int = 32
    ) -> np.ndarray:
        """Extract CLS token attention map from attention weights.
        
        :param attention_weights: Attention weights of shape [batch, num_heads, seq_len, seq_len].
        :param image_size: Input image size (default 224).
        :param patch_size: Patch size (default 32 for ViT-B/32).
        :return: Attention map of shape [H, W] where H*W = num_patches.
        """
        # Validate input shape
        if attention_weights.dim() != 4:
            raise ValueError(f"Expected 4D attention weights, got shape: {attention_weights.shape}")
        
        batch_size, num_heads, seq_len, seq_len2 = attention_weights.shape
        
        if seq_len != seq_len2:
            raise ValueError(f"Attention weights must be square, got shape: {attention_weights.shape}")
        
        if seq_len < 2:
            raise ValueError(f"Sequence length must be at least 2 (CLS + patches), got: {seq_len}")
        
        # Extract CLS token attention (first token attends to all tokens)
        # attention_weights[:, :, 0, :] is CLS token attending to all tokens
        # We want CLS token attending to patch tokens (excluding CLS token itself)
        cls_attention = attention_weights[:, :, 0, 1:]  # [batch, num_heads, num_patches]
        
        # Average across attention heads
        cls_attention = cls_attention.mean(dim=1)  # [batch, num_patches]
        
        # Take first sample in batch
        if isinstance(cls_attention, torch.Tensor):
            cls_attention = cls_attention[0].cpu().numpy()  # [num_patches]
        else:
            cls_attention = cls_attention[0]  # [num_patches]
        
        # Validate that we have patches
        num_patches = cls_attention.size
        if num_patches == 0:
            raise ValueError("No patches found in attention weights")
        
        # Reshape to spatial dimensions
        num_patches_per_side = int(np.sqrt(num_patches))
        
        if num_patches_per_side * num_patches_per_side != num_patches:
            # If not a perfect square, try to infer from image size and patch size
            num_patches_per_side = image_size // patch_size
            if num_patches_per_side * num_patches_per_side != num_patches:
                # Still doesn't match, try to find closest valid dimensions
                # For ViT-B/32: 224/32 = 7, so 7x7 = 49 patches
                # For ViT-B/16: 224/16 = 14, so 14x14 = 196 patches
                # Try common configurations
                for test_size in [7, 14, 16]:
                    if test_size * test_size == num_patches:
                        num_patches_per_side = test_size
                        break
                else:
                    # Last resort: use floor of sqrt
                    num_patches_per_side = int(np.floor(np.sqrt(num_patches)))
                    # Pad or truncate if needed
                    expected_patches = num_patches_per_side * num_patches_per_side
                    if expected_patches < num_patches:
                        cls_attention = cls_attention[:expected_patches]
                    elif expected_patches > num_patches:
                        # Pad with zeros
                        padding = np.zeros(expected_patches - num_patches)
                        cls_attention = np.concatenate([cls_attention, padding])
        
        attention_map = cls_attention.reshape(num_patches_per_side, num_patches_per_side)
        
        return attention_map
    
    def cleanup(self) -> None:
        """Remove all registered hooks and restore original forward methods."""
        # Restore original forward methods
        for attn_module, original_forward in self.hooks:
            attn_module.forward = original_forward
        
        self.hooks.clear()
        self.attention_maps.clear()


def extract_attention_from_clip_vit(
    model: nn.Module,
    x: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
    image_size: int = 224,
    patch_size: int = 32
) -> Dict[int, np.ndarray]:
    """Convenience function to extract attention maps from CLIP ViT.
    
    :param model: The CLIP ViT model (IntentonomyClipViTModule).
    :param x: Input tensor of shape (batch_size, 3, H, W).
    :param layer_indices: List of layer indices to extract. If None, extract all layers.
    :param image_size: Input image size (default 224).
    :param patch_size: Patch size (default 32 for ViT-B/32).
    :return: Dictionary mapping layer index to attention map [H, W].
    """
    extractor = AttentionExtractor(model)
    try:
        # Extract raw attention weights
        attention_weights_dict = extractor.extract_attention_maps(x, layer_indices)
        
        # Convert to attention maps
        attention_maps = {}
        for layer_idx, attn_weights in attention_weights_dict.items():
            attention_map = extractor.get_cls_attention_map(
                attn_weights, image_size, patch_size
            )
            attention_maps[layer_idx] = attention_map
        
        return attention_maps
    finally:
        extractor.cleanup()
