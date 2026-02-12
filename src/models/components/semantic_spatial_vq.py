"""
Semantic Spatial Vector Quantizer

对空间特征图进行语义量化，使用预定义的语义锚点初始化codebook。
保留空间维度，对每个patch位置独立进行量化。
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSpatialVQ(nn.Module):
    """
    语义空间向量量化器
    
    使用语义锚点初始化codebook，对空间特征图的每个位置独立进行量化。
    使用Cosine相似度计算距离（更适合CLIP特征空间）。
    """
    
    def __init__(
        self,
        anchor_embeddings: torch.Tensor,
        embedding_dim: int = 1024,
        commitment_cost: float = 0.25,
        freeze_codebook: bool = True,
    ):
        """
        初始化SemanticSpatialVQ
        
        Args:
            anchor_embeddings: 语义锚点embeddings [num_codes, embedding_dim]
            embedding_dim: embedding维度（通常与CLIP hidden dim相同，如1024）
            commitment_cost: VQ commitment loss权重
            freeze_codebook: 是否冻结codebook（默认True）
        """
        super().__init__()
        
        num_codes, dim = anchor_embeddings.shape
        
        if dim != embedding_dim:
            raise ValueError(
                f"Anchor embedding dimension {dim} does not match "
                f"specified embedding_dim {embedding_dim}"
            )
        
        self.embedding_dim = embedding_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        
        # 使用语义锚点初始化codebook
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.copy_(anchor_embeddings)
        
        # 保存初始锚点（用于语义一致性loss）
        self.register_buffer('initial_anchors', anchor_embeddings.clone())
        
        # 是否冻结codebook
        if freeze_codebook:
            self.embedding.weight.requires_grad = False
    
    def forward(
        self, 
        inputs: torch.Tensor,
        return_indices: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: 输入特征图 [B, N_patches, embedding_dim]
            return_indices: 是否返回编码索引
        
        Returns:
            - vq_loss: VQ损失
            - quantized: 量化后的特征 [B, N_patches, embedding_dim]
            - perplexity: 困惑度（用于监控）
            - encoding_indices: 编码索引 [B, N_patches] (如果return_indices=True)
        """
        input_shape = inputs.shape
        B, N_patches, D = input_shape
        
        # 展平为 [B*N_patches, D]
        flat_input = inputs.contiguous().view(-1, D)
        
        # 归一化输入和codebook（用于Cosine相似度）
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        codebook_norm = F.normalize(self.embedding.weight, p=2, dim=1)
        
        # 计算Cosine相似度（负距离，因为我们要找最相似的）
        # Cosine相似度范围 [-1, 1]，越大越相似
        # 我们使用负的相似度作为距离，所以argmin找到最相似的
        distances = -torch.matmul(flat_input_norm, codebook_norm.t())  # [B*N_patches, num_codes]
        
        # 获取编码索引
        encoding_indices = torch.argmin(distances, dim=1)  # [B*N_patches]
        
        # 创建one-hot编码
        encodings = torch.zeros(
            encoding_indices.shape[0], 
            self.num_codes, 
            device=inputs.device
        ).type_as(inputs)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # 量化：使用codebook中的embeddings
        quantized = torch.matmul(encodings, self.embedding.weight)  # [B*N_patches, D]
        quantized = quantized.view(B, N_patches, D)  # [B, N_patches, D]
        
        # VQ Loss
        # e_latent_loss: 鼓励输入接近量化结果
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss: 鼓励量化结果接近输入（Straight-Through Estimator）
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator: 前向使用量化值，反向传播梯度到输入
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算困惑度（perplexity）
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 重塑编码索引
        encoding_indices = encoding_indices.view(B, N_patches)  # [B, N_patches]
        
        if return_indices:
            return quantized, vq_loss, perplexity, encoding_indices
        else:
            return quantized, vq_loss, perplexity
    
    def get_semantic_consistency_loss(self) -> torch.Tensor:
        """
        计算语义一致性loss（防止codebook漂移）
        
        Returns:
            语义一致性loss（如果codebook被冻结，返回0）
        """
        if not self.embedding.weight.requires_grad:
            # Codebook被冻结，不需要一致性loss
            return torch.tensor(0.0, device=self.embedding.weight.device)
        
        # 计算当前codebook与初始锚点的L2距离
        loss = F.mse_loss(self.embedding.weight, self.initial_anchors)
        return loss
    
    def get_code_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        获取编码索引（用于可视化）
        
        Args:
            inputs: 输入特征图 [B, N_patches, embedding_dim]
        
        Returns:
            编码索引 [B, N_patches]
        """
        _, _, _, indices = self.forward(inputs, return_indices=True)
        return indices

