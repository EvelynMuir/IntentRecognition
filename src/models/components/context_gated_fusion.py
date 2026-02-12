import torch
import torch.nn as nn

class ContextGatedFusion(nn.Module):
    """
    Context-aware Gated Fusion for multi-stream models.
    Takes 5 factor features and predicts weights for each to perform a weighted sum.
    """
    def __init__(self, 
                 num_factors=5, 
                 input_dim=768,  # Matches vit_projected_dim
                 hidden_dim=1024):
        super().__init__()
        
        self.num_factors = num_factors
        self.input_dim = input_dim
        
        # 1. Total dimension after flattening factors
        self.flat_dim = num_factors * input_dim
        
        # 2. Gating Network: Predicts weights for each factor
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_factors), # Output weights for 5 factors
            nn.Sigmoid() # Range [0, 1]
        )

    def forward(self, quantized_factors):
        """
        Args:
            quantized_factors: [B, 5, 768] - Pooled features from 5 VQ factors
        Returns:
            weighted_sum: [B, 768] - Fused global features
            gates: [B, 5] - Importance weights for each factor
        """
        B = quantized_factors.shape[0]
        
        # Step A: Flatten for gating network
        # [B, 5, 768] -> [B, 5 * 768]
        flat_features = quantized_factors.view(B, -1)
        
        # Step B: Calculate Gates (Importance Weights)
        # [B, flat_dim] -> [B, 5]
        gates = self.gate_mlp(flat_features)
        
        # Step C: Apply Gates (Reweighting and Weighted Sum)
        # quantized_factors: [B, 5, 768]
        # gates.unsqueeze(-1): [B, 5, 1]
        weighted_factors = quantized_factors * gates.unsqueeze(-1)
        
        # Weighted sum across factors (dim 1)
        # [B, 5, 768] -> [B, 768]
        fused_features = weighted_factors.sum(dim=1)
        
        return fused_features, gates
