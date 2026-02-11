"""Vector Quantizer for codebook-based models."""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        anchor_embeddings: Optional[torch.Tensor] = None,
        freeze_codebook: bool = False,
        use_cosine_similarity: bool = False,
    ):
        """
        Initialize the Vector Quantizer.
        
        Args:
            num_embeddings: Number of embeddings in the codebook.
            embedding_dim: Dimension of the embeddings.
            commitment_cost: Weight for the commitment loss.
            anchor_embeddings: Optional tensor to initialize the codebook.
            freeze_codebook: Whether to freeze the codebook (no gradient updates).
            use_cosine_similarity: Whether to use cosine similarity (instead of L2 distance).
        """
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._use_cosine_similarity = use_cosine_similarity

        if anchor_embeddings is not None:
            num_anchors, dim = anchor_embeddings.shape
            if dim != embedding_dim:
                raise ValueError(
                    f"Anchor embedding dimension {dim} does not match "
                    f"specified embedding_dim {embedding_dim}"
                )
            if num_anchors != num_embeddings:
                 # If anchors provided but size mismatch, we might want to warn or error.
                 # For now, let's assume if anchors are provided, they dictate the codebook size 
                 # OR the user ensures they match.
                 # If implied that num_embeddings should be equal to anchor_embeddings, override it?
                 # But standard practice is usually they match.
                 # Let's trust the user or raise error if critical.
                 # In SemanticSpatialVQ, num_codes comes from anchors.
                 if num_anchors != num_embeddings:
                     print(f"Warning: Num anchors {num_anchors} != num_embeddings {num_embeddings}. Using num_embeddings.")
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        
        if anchor_embeddings is not None:
            # Initialize with anchors
            # If anchors > num_embeddings, we truncate? Or if <, we cycle?
            # Assuming exact match for now based on typical usage.
            with torch.no_grad():
                if anchor_embeddings.shape[0] == self._num_embeddings:
                    self._embedding.weight.data.copy_(anchor_embeddings)
                else:
                     # Handle mismatch if necessary, or just copy valid part
                    min_rows = min(anchor_embeddings.shape[0], self._num_embeddings)
                    self._embedding.weight.data[:min_rows].copy_(anchor_embeddings[:min_rows])
            
            # Save initial anchors for consistency loss if needed
            self.register_buffer('initial_anchors', anchor_embeddings.clone())
        else:
            self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
            self.register_buffer('initial_anchors', self._embedding.weight.data.clone())

        if freeze_codebook:
            self._embedding.weight.requires_grad = False
            
    def forward(self, inputs: torch.Tensor, return_indices: bool = False):
        input_shape = inputs.shape
        # inputs: [B, ..., D]
        
        flat_input = inputs.contiguous().view(-1, self._embedding_dim)

        if self._use_cosine_similarity:
            # Cosine similarity logic (from SemanticSpatialVQ)
            # Normalize inputs and codebook
            flat_input_norm = F.normalize(flat_input, p=2, dim=1)
            codebook_norm = F.normalize(self._embedding.weight, p=2, dim=1)
            
            # Distance = -similarity (we want to minimize distance -> maximize similarity)
            distances = -torch.matmul(flat_input_norm, codebook_norm.t())
        else:
            # Euclidean distance logic
            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(self._embedding.weight ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [N, 1]

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device).type_as(inputs)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Return format matching original: loss, quantized, perplexity, encodings, encoding_indices
        # Note: SemanticSpatialVQ returned (quantized, vq_loss, perplexity, [indices])
        # Original VectorQuantizer returned (loss, quantized, perplexity, encodings, encoding_indices)
        # We stick to original VectorQuantizer signature to minimize breaking changes in existing code,
        # but the unified module will need to handle it.
        
        if return_indices:
             return loss, quantized, perplexity, encodings, encoding_indices
        
        return loss, quantized, perplexity, encodings, encoding_indices

    def get_encoding(self, code):
        batch = code.shape[0]
        # Assuming code is indices [B, ...]? 
        # The original code handled shape [B, prompt_num] -> view(-1,1)
        # We try to keep compatibility
        
        if code.dim() > 1:
            flat_code = code.view(-1, 1)
        else:
            flat_code = code.unsqueeze(1)
            
        encodings = torch.zeros(flat_code.shape[0], self._num_embeddings, device=flat_code.device).type_as(self._embedding.weight)
        encodings.scatter_(1, flat_code, 1)

        quantized = torch.matmul(encodings, self._embedding.weight)
        
        if code.dim() > 1:
            quantized = quantized.view(batch, code.shape[1], -1)
        else:
            quantized = quantized.view(batch, -1)
            
        return quantized

    def get_semantic_consistency_loss(self) -> torch.Tensor:
        """
        Calculate semantic consistency loss (prevent codebook drift).
        """
        if not self._embedding.weight.requires_grad:
            return torch.tensor(0.0, device=self._embedding.weight.device)
        
        # Calculate MSE between current codebook and initial anchors
        # Using registered buffer 'initial_anchors'
        loss = F.mse_loss(self._embedding.weight, self.initial_anchors)
        return loss
    
    def get_code_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get code indices for inputs.
        """
        _, _, _, _, encoding_indices = self.forward(inputs)
        # encoding_indices is [N, 1] flattened
        # reshape back to [B, ...]
        B = inputs.shape[0]
        # Assuming inputs is [B, N_patches, D]
        if inputs.dim() == 3:
            N_patches = inputs.shape[1]
            return encoding_indices.view(B, N_patches)
        return encoding_indices.view(B, -1)


