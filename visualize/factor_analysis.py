"""Factor analysis utilities for codebook-based models."""
from typing import List

import torch


def get_quantized_features(model, images: torch.Tensor) -> torch.Tensor:
    """获取量化后的语义块特征。
    
    Args:
        model: IntentonomyClipViTCodebookModule 模型实例
        images: 图像张量，形状为 [B, 3, H, W]
    
    Returns:
        z_quantized: 量化后的语义块，形状为 [B, F, D]，其中 F 是 factor 数量，D 是每个块的维度
    """
    with torch.no_grad():
        _, _, _, z_quantized = model.forward(images, return_vq_info=True)
    return z_quantized


def factor_drop_test(model, z_q_split: torch.Tensor, intents: List[str] = None) -> torch.Tensor:
    """测试每个 factor 被 drop 后对各个 intent 预测的影响。
    
    Args:
        model: 模型实例，需要有 classifier 属性
        z_q_split: 量化后的语义块，形状为 [B, F, D]
        intents: 意图类别名称列表（可选，用于调试）
    
    Returns:
        drops: 形状为 [F, num_intents] 的张量，表示每个 factor 被 drop 后对每个 intent 的影响
    """
    B, F, D = z_q_split.shape
    device = z_q_split.device
    
    # 获取基础预测（所有 factor 都保留）
    z_flat_base = z_q_split.view(B, -1)  # [B, F*D]
    base_pred = model.classifier(z_flat_base).sigmoid()  # [B, num_intents]
    
    drops = []
    
    # 对每个 factor 进行 drop 测试
    for f in range(F):
        # 创建 drop 后的特征（将第 f 个 factor 置零）
        z_drop = z_q_split.clone()
        z_drop[:, f] = 0  # 将第 f 个 factor 置零
        
        # 获取 drop 后的预测
        z_flat_drop = z_drop.view(B, -1)  # [B, F*D]
        pred = model.classifier(z_flat_drop).sigmoid()  # [B, num_intents]
        
        # 计算预测变化（绝对值平均）
        drop = (base_pred - pred).abs().mean(dim=0)  # [num_intents]
        drops.append(drop)
    
    # 堆叠所有 factor 的结果
    return torch.stack(drops)  # [F, num_intents]

