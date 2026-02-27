from typing import Any, Dict, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.metrics import eval_test_set_both_strategies, eval_validation_set
from src.utils.ema import ModelEma


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphReasoningLayer(nn.Module):
    """
    单层图卷积推理：让选中的 Patch 之间交换信息
    Formula: H' = ReLU(Norm(A * H * W)) + H
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj):
        """
        x: [B, K, D] - 节点特征
        adj: [B, K, K] - 邻接矩阵
        """
        # 1. Message Passing (聚合邻居信息)
        # [B, K, K] @ [B, K, D] -> [B, K, D]
        out = torch.bmm(adj, x)
        
        # 2. Node Update (节点内部变换)
        out = self.linear(out)
        out = self.norm(out)
        out = self.act(out)
        
        # 3. Residual Connection (残差连接，防止梯度消失)
        return x + self.dropout(out)


class LabelGCN(nn.Module):
    """
    标签图卷积网络：使用预计算的邻接矩阵增强标签嵌入之间的交互
    Formula: H' = ReLU(A * H * W) + H (带残差连接)
    """
    def __init__(self, dim, adj_matrix):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        # 将邻接矩阵注册为 buffer (不可学习，但随模型保存)
        # adj_matrix 应该是 [Num_Classes, Num_Classes]，行归一化过
        self.register_buffer('adj', adj_matrix)

    def forward(self, x):
        """
        x: [Num_Classes, Dim] - 标签嵌入特征
        返回: [Num_Classes, Dim] - 交互后的标签特征
        """
        # 公式: H' = ReLU(A * H * W) + H
        x_proj = self.fc(x)       # [C, D]
        x_adj = torch.mm(self.adj, x_proj)  # [C, C] * [C, D] -> [C, D]
        return x + self.act(x_adj)  # 残差连接，防止把 Embedding 搞乱


class LabelGuidedVerifier(nn.Module):
    def __init__(self, vis_dim, num_classes, num_heads=4):
        super().__init__()
        # 标准的 Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=num_heads, batch_first=True)
        
        # 【核心创新】为每个类别定义一个可学习的温度参数
        # 初始化为 0 (即 exp(0) = 1.0，标准 Attention)
        self.log_temperature = nn.Parameter(torch.zeros(num_classes)) 
        nn.init.constant_(self.log_temperature, -1.0)
        
    def forward(self, label_queries, visual_evidence):
        """
        label_queries: [B, Num_Classes, D] - 意图标签 Embedding
        visual_evidence: [B, K, D] - 经过 GCN 的视觉线索
        """
        # 1. 计算温度 tau
        # 使用 exp 保证温度恒为正数，且数值稳定
        # shape: [Num_Classes] -> [1, Num_Classes, 1] 以便广播
        # 限制范围！防止除以 0 或过大
        tau = torch.clamp(torch.exp(self.log_temperature), min=0.01, max=5.0)
        tau = tau.view(1, -1, 1) # [1, Num_Classes, 1]
        
        # 2. 调节 Query 的"锐度" (Temperature Scaling)
        # 如果 tau < 1: Query 变大 -> Softmax 变尖 -> 关注局部 (Focal)
        # 如果 tau > 1: Query 变小 -> Softmax 变平 -> 关注全局 (Global)
        scaled_queries = label_queries / tau
        
        # 3. 传入标准 Attention
        # 这里的 scaled_queries 已经携带了类别特有的"聚焦偏好"
        attn_out, attn_weights = self.attn(query=scaled_queries, 
                                           key=visual_evidence, 
                                           value=visual_evidence)
        
        return attn_out, attn_weights, tau


class DiscriminativeClueMiner(nn.Module):
    def __init__(self, 
                 vis_dim=768,       # ViT 输出维度 (Base: 768, Large: 1024)
                 text_dim=1536,     # LLM 标签 Embedding 维度 (e.g., BERT/GPT)
                 num_classes=28,    # 意图类别数
                 k_patches=16,      # 显式保留的关键 Patch 数量
                 gcn_depth=2,       # GCN 推理层数
                 use_adaptive_temperature=False,  # 是否使用自适应温度缩放
                 use_global_feature=False,  # 是否使用全局特征拼接
                 use_learnable_fusion=False,  # 是否使用可学习的融合权重
                 use_threshold_loss=False,  # 是否使用阈值loss（决定是否创建阈值头）
                 use_image_level_temperature=False,  # 是否使用 Image-Level 温度头
                 use_cosine_similarity_temperature=False,  # 是否使用基于余弦相似度的温度计算
                 use_max_pooling_temperature=False,  # 是否使用 Max Pooling 辅助温度计算（双通道一致性）
                 use_consistency_bias=False,  # 是否使用 consistency bias（默认关闭）
                 adj_matrix=None):  # 标签共现概率邻接矩阵 [Num_Classes, Num_Classes]
        super().__init__()
        
        self.vis_dim = vis_dim
        self.k = k_patches
        self.use_adaptive_temperature = use_adaptive_temperature
        self.use_global_feature = use_global_feature
        self.use_learnable_fusion = use_learnable_fusion
        self.use_threshold_loss = use_threshold_loss
        self.use_image_level_temperature = use_image_level_temperature
        self.use_cosine_similarity_temperature = use_cosine_similarity_temperature
        self.use_max_pooling_temperature = use_max_pooling_temperature
        self.use_consistency_bias = use_consistency_bias
        self.num_classes = num_classes  # 修复：保存 num_classes 供 forward 使用
        
        # -----------------------------------------------------------
        # 1. Clue Proposal Network (Patch Scoring)
        # -----------------------------------------------------------
        # 给每个 Patch 打分，用于筛选 Top-K
        self.scorer = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4),
            nn.ReLU(),
            nn.Linear(vis_dim // 4, 1),
            nn.Sigmoid() 
        )

        # -----------------------------------------------------------
        # 2. Dynamic Structure Learning (GCN)
        # -----------------------------------------------------------
        # 用于在选中的 K 个 Patch 之间建立推理关系
        self.gcn_layers = nn.ModuleList([
            GraphReasoningLayer(vis_dim) for _ in range(gcn_depth)
        ])
        
        # -----------------------------------------------------------
        # 3. Label-Guided Alignment (The "Verifier")
        # -----------------------------------------------------------
        # 将文本维度的 Label 映射到视觉维度
        self.label_proj = nn.Linear(text_dim, vis_dim)
        
        # 【新增】Label GCN 模块：让标签之间交互
        if adj_matrix is not None:
            self.label_gcn = LabelGCN(vis_dim, adj_matrix)
        else:
            self.label_gcn = nn.Identity()
        
        # 根据 use_adaptive_temperature 选择使用哪种 Verifier
        if use_adaptive_temperature:
            # 使用带自适应温度缩放的 LabelGuidedVerifier
            self.verifier = LabelGuidedVerifier(vis_dim, num_classes, num_heads=4)
        else:
            # 使用标准的 MultiheadAttention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=vis_dim, 
                num_heads=8, 
                batch_first=True,
                dropout=0.1
            )

        if use_global_feature:
            self.global_head = nn.Linear(vis_dim, num_classes)
        else:
            self.global_head = None

        self.ln_local = nn.LayerNorm(vis_dim)
        self.local_head = nn.Linear(vis_dim, 1)
        
        # 【必杀技】零初始化 Local Head
        # 这确保了在 Epoch 0，DCM 分支的输出全为 0，不干扰 Global 分支
        nn.init.constant_(self.local_head.weight, 0)
        nn.init.constant_(self.local_head.bias, 0)
        
        # 阈值头：用于预测每个样本的阈值（仅在 use_threshold_loss=True 时创建）
        # 输入维度是 vis_dim (ResNet global feature dim, e.g., 2048 or 1024)
        if use_threshold_loss:
            self.thresh_head = nn.Sequential(
                nn.Linear(vis_dim, vis_dim // 4),
                nn.ReLU(),
                nn.Linear(vis_dim // 4, 1),
                nn.Sigmoid()  # 必须有 Sigmoid，确保输出在 0-1 之间
            )
        else:
            self.thresh_head = None
        
        # 最后的 Logit 缩放因子 (可选，有助于收敛)
        self.scale = nn.Parameter(torch.ones([]) * (vis_dim ** -0.5))
        
        # 可学习的全局特征融合权重（仅在 use_learnable_fusion=True 时创建）
        if use_learnable_fusion:
            self.global_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.global_scale = None
        
        # 【核心新增】Image-Level 温度头
        # 极其简单：Global Feature -> Scalar
        # 用于根据全局特征自适应计算温度，对 logits 进行缩放
        if use_image_level_temperature:
            self.temp_head = nn.Sequential(
                nn.Linear(vis_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid() 
            )
        else:
            self.temp_head = None
        
        # 【新增】基于 Global-Local Cosine Similarity 的温度计算
        # 使用可学习的 logit_scale 参数来动态决定温度
        if use_cosine_similarity_temperature:
            # 初始化为 ln(100) ≈ 4.6052，这在对比学习中是一个经典的初始值
            # 这是一个可学习参数，它决定了模型对"一致性"有多敏感
            self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        else:
            self.logit_scale = None
        
        # 【新增】Dropout 层：在训练时给 DCM 加一点 Dropout
        # self.clue_dropout = nn.Dropout(0.1)
        
        # 【新增】Consistency Bias：用于调整 consistency 值的可学习参数
        if use_consistency_bias:
            self.consistency_bias = nn.Parameter(torch.tensor(-0.5))
        else:
            self.consistency_bias = None

    def forward(self, x, label_embeddings):
        """
        Args:
            x: [B, N_patches, D] - ViT Backbone 输出 (无 CLS token)
            label_embeddings: [Num_Classes, Text_D] - 冻结的 LLM 特征
        Returns:
            logits: [B, Num_Classes]
            aux_dict: 用于计算 Loss 和可视化的信息
        """
        B, N, D = x.shape

        # 计算全局特征（如果使用 global_feature、threshold_loss、image_level_temperature 或 cosine_similarity_temperature，需要计算）
        if self.use_global_feature or self.use_threshold_loss or self.use_image_level_temperature or self.use_cosine_similarity_temperature:
            global_feat = x.mean(dim=1) # [B, D]
        else:
            global_feat = None
        
        if self.use_global_feature:
            logits_global = self.global_head(global_feat) # [B, num_classes]
        else:
            logits_global = torch.zeros(B, self.num_classes, device=x.device)
        
        # 计算阈值预测（仅在 use_threshold_loss=True 时）
        if self.use_threshold_loss and self.thresh_head is not None:
            pred_thresh = self.thresh_head(global_feat)  # [B, 1]
        else:
            pred_thresh = None
        
        # =======================================================
        # Step 1: Discriminative Clue Mining (去噪与筛选)
        # =======================================================
        
        # 1.1 计算分数
        scores = self.scorer(x)  # [B, N, 1]
        
        # 1.2 Top-K 筛选
        # topk_indices: [B, K, 1]
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=1)
        
        # 1.3 提取特征
        # 扩展 indices 以适配 gather: [B, K, D]
        gather_indices = topk_indices.expand(-1, -1, D)
        selected_features = torch.gather(x, 1, gather_indices) # [B, K, D]
        
        # *关键*: 将分数乘回特征。
        # 作用 A: 让梯度流回 scorer，使模型学会给重要 Patch 打高分。
        # 作用 B: 作为一个 Soft Gate，进一步抑制不确定的特征。
        selected_features = selected_features * topk_scores
        
        # =======================================================
        # Step 2: Dynamic Graph Construction (动态构图)
        # =======================================================
        
        # 计算 Patch 间的语义相似度作为边权重
        # normalize 确保余弦相似度计算
        feats_norm = F.normalize(selected_features, p=2, dim=-1)
        # [B, K, D] @ [B, D, K] -> [B, K, K]
        adj_matrix = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
        
        # 归一化邻接矩阵 (Row-wise Softmax)，作为 GCN 的权重
        # 乘以 10 作为 temperature，让关注点更集中
        adj_matrix = F.softmax(adj_matrix * 10, dim=-1)
        
        # =======================================================
        # Step 3: Compositional Reasoning (GCN 推理)
        # =======================================================
        
        # "Reasoned Features": 此时的特征不再是孤立的 Patch，而是包含了上下文关系
        gcn_feat = selected_features
        for layer in self.gcn_layers:
            gcn_feat = layer(gcn_feat, adj_matrix)
            
        # =======================================================
        # Step 4: Label-Guided Verification (意图验证)
        # =======================================================
        
        # 4.1 准备 Queries
        # label_embeddings: [Num_Classes, Text_D]
        label_feats = self.label_proj(label_embeddings) # [Num_Classes, Vis_D]
        
        # 4.1.1 【关键】让 Label 之间交互
        # 交互后的 label_feats 包含了共现信息
        # 比如 "Party" 的向量里会混入一点 "Friends" 的信息
        label_feats = self.label_gcn(label_feats)  # [Num_Classes, Vis_D]
        
        # 扩展到 Batch: [B, Num_Classes, Vis_D]
        label_queries = label_feats.unsqueeze(0).expand(B, -1, -1)
        
        # 4.2 根据是否使用自适应温度选择不同的 Verifier
        if self.use_adaptive_temperature:
            # 使用带温度缩放的 LabelGuidedVerifier
            # "我看这 K 个线索(Key/Value)中，有没有符合 Label(Query) 描述的组合？"
            # attn_out: [B, Num_Classes, Vis_D]
            attn_out, attn_weights, learned_tau = self.verifier(label_queries, gcn_feat)
        else:
            # 使用标准的 Cross Attention
            # attn_out: [B, Num_Classes, Vis_D]
            attn_out, attn_weights = self.cross_attn(
                query=label_queries, 
                key=gcn_feat, 
                value=gcn_feat
            )
            learned_tau = None
        
        # 在训练时给 DCM 加一点 Dropout
        # attn_out = self.clue_dropout(attn_out)
        
        # 4.3 预测 Logits (Visual-Semantic Similarity)
        # 计算对齐后的视觉特征与标签语义的点积
        # [B, NC, D] * [B, NC, D] -> sum -> [B, NC]
        # logits = (attn_out * label_queries).sum(dim=-1) * self.scale
        attn_out = self.ln_local(attn_out)
        if self.use_learnable_fusion:
            logits = self.local_head(attn_out).squeeze(-1) + self.global_scale * logits_global
        else:
            logits = self.local_head(attn_out).squeeze(-1) + torch.tanh(logits_global) * 2.0
        
        # 【关键一步】计算全局自适应温度
        # 0.1 (锐化，用于弱样本) ~ 1.0 (保持，用于强样本)
        temperature = None
        
        # 优先使用基于余弦相似度的温度计算
        if self.use_cosine_similarity_temperature and self.logit_scale is not None:
            # 确保 global_feat 已计算
            if global_feat is None:
                global_feat = x.mean(dim=1)  # [B, D]
            
            # === 核心修改：双通道一致性 (Dual-Stream Consistency) ===
            
            # A. 准备特征
            global_norm = F.normalize(global_feat, dim=-1)  # [B, D]
            
            if self.use_max_pooling_temperature:
                # 通道 1: 平均线索 (代表整体氛围)
                feat_mean_norm = F.normalize(attn_out.mean(dim=1), dim=-1)  # [B, Num_Classes, Vis_D] -> [B, Vis_D]
                consistency_mean = (global_norm * feat_mean_norm).sum(dim=-1, keepdim=True)  # [B, 1]
                
                # 通道 2: 峰值线索 (代表最强证据)
                # 只要有一个 Patch 和全局特征对上了，就是强信号！
                # attn_out: [B, Num_Classes, Vis_D] -> max -> [B, Vis_D]
                feat_max, _ = attn_out.max(dim=1) 
                feat_max_norm = F.normalize(feat_max, dim=-1)
                consistency_max = (global_norm * feat_max_norm).sum(dim=-1, keepdim=True)  # [B, 1]
                
                # B. 融合策略：取两者的最大值 (Winner Takes All)
                # 逻辑：只要"整体像" 或者 "局部有个地方特别像"，我就认为一致性高
                # 这能极大挽救那些背景杂乱的弱样本！
                consistency = torch.max(consistency_mean, consistency_max)
            else:
                # 原有逻辑：仅使用平均线索
                # 1. DCM 挖掘线索 - 从 attn_out 聚合得到局部特征
                local_feat_agg = attn_out.mean(dim=1)  # [B, Num_Classes, Vis_D] -> [B, Vis_D]
                
                # 2. 归一化并计算余弦相似度
                local_norm = F.normalize(local_feat_agg, dim=-1)  # [B, Vis_D]
                
                # consistency: [-1, 1]
                consistency = (global_norm * local_norm).sum(dim=-1, keepdim=True)  # [B, 1]
            
            # C. 计算温度 (保持之前的逻辑)
            # .exp() 保证系数恒为正
            # 这个系数通常会学到很大 (比如 10~50)，用来放大微小的相似度差异
            scale = self.logit_scale.exp()
            
            # 动态温度系数 alpha: 0(难/不一致) ~ 1(易/一致)
            # 如果启用 consistency_bias，则在计算前加上 bias
            if self.use_consistency_bias and self.consistency_bias is not None:
                alpha = torch.sigmoid((consistency + self.consistency_bias) * scale)  # [B, 1]
            else:
                alpha = torch.sigmoid(consistency * scale)  # [B, 1]
            
            # 最终温度：[0.1, 1.0]
            # alpha 越小 -> T 越接近 0.1 (锐化，帮助弱样本)
            # alpha 越大 -> T 越接近 1.0 (保持，防止过拟合)
            temperature = alpha * 0.9 + 0.1  # 范围 [0.1, 1.0]

            # DEBUG: print temperature
            # print(f"Temp Mean: {temperature.mean().item():.3f} | Min: {temperature.min().item():.3f} | Max: {temperature.max().item():.3f}")
            
            # 应用基于共识的温度
            logits = logits / temperature
        # 向后兼容：使用原有的 Image-Level 温度头
        elif self.use_image_level_temperature and self.temp_head is not None:
            # 确保 global_feat 已计算
            if global_feat is None:
                global_feat = x.mean(dim=1)  # [B, D]
            
            # 计算温度：alpha 在 [0, 1]，temperature 在 [0.1, 1.0]
            alpha = self.temp_head(global_feat)  # [B, 1]
            temperature = alpha * 0.9 + 0.1  # 范围 [0.1, 1.0]
            
            # 放大 Logits：这就是 Hard Label 能 Work 的核心魔法
            logits = logits / temperature
        
        # 构建返回字典
        aux_dict = {
            "patch_scores": scores,        # 用于 Sparsity Loss
            "topk_indices": topk_indices,  # 用于可视化 (Fig 5)
            "adj_matrix": adj_matrix,      # 用于可视化图结构
            "attn_weights": attn_weights   # Attention 权重
        }
        
        # 只有在使用阈值loss时才添加 pred_thresh
        if pred_thresh is not None:
            aux_dict["pred_thresh"] = pred_thresh  # 阈值预测 [B, 1]
        
        # 只有在使用自适应温度时才添加 learned_tau
        if self.use_adaptive_temperature:
            aux_dict["learned_tau"] = learned_tau  # 学习到的温度参数，用于做分析图表！
        
        # 只有在使用 Image-Level 温度时才添加 temperature
        if temperature is not None:
            aux_dict["image_level_temperature"] = temperature  # Image-Level 温度，用于分析和可视化
        
        return logits, aux_dict


def sparsity_loss(patch_scores, target_sparsity=0.2):
    """
    强迫 Scorer 网络只激活少量的 Patch，防止所有 Patch 分数都很高。
    这能增强 'DCM' 的去噪能力。
    """
    # L1 约束：希望平均分接近 target_sparsity
    mean_score = torch.mean(patch_scores)
    loss = torch.abs(mean_score - target_sparsity)
    return loss


def sparsity_loss_smart(patch_scores, k=3):
    """
    智能稀疏：保护分数最高的 K 个 Patch 不受惩罚，只惩罚剩下的背景。
    
    Args:
        patch_scores: [B, N, 1] 或 [B, N] - Patch 分数
        k: int - 保护前 K 个最高分数的 Patch
    
    Returns:
        loss: 背景 Patch 分数的平均绝对值
    """
    # 处理形状：如果是 [B, N, 1]，先 squeeze 为 [B, N]
    if patch_scores.dim() == 3:
        patch_scores = patch_scores.squeeze(-1)  # [B, N, 1] -> [B, N]
    
    # 1. 对 Patch 分数排序
    sorted_scores, _ = torch.sort(patch_scores, dim=1, descending=True)
    
    # 2. 只惩罚排在 K 名之后的 Patch (认为是背景)
    # 这样如果有 10 个有效线索，模型可以大胆激活前 10 个，只要把第 11 个以后的背景压住就行
    background_scores = sorted_scores[:, k:] 
    
    loss = torch.mean(torch.abs(background_scores))
    return loss


def correlation_loss(logits, targets):
    """
    计算预测的相关性与真实标签的相关性之间的差异。
    通过比较预测概率的相关性矩阵和真实标签的相关性矩阵来约束模型学习标签之间的相关性。
    
    Args:
        logits: [B, Num_Classes] - 模型输出的 logits
        targets: [B, Num_Classes] - 真实标签（0/1）
    
    Returns:
        loss: 相关性损失值
    """
    # 预测的相关性
    probs = torch.sigmoid(logits)
    probs_norm = F.normalize(probs, dim=0)  # 跨 Batch 归一化
    pred_corr = torch.mm(probs_norm.t(), probs_norm)
    
    # 真实的相关性
    targets_norm = F.normalize(targets.float(), dim=0)
    gt_corr = torch.mm(targets_norm.t(), targets_norm)
    
    return F.mse_loss(pred_corr, gt_corr)


def threshold_consistency_loss(scaled_logits, targets, class_thresholds, margin=0.1):
    """
    阈值一致性损失：让学习到的阈值能够把正负样本分开。
    正样本应该 > threshold + margin
    负样本应该 < threshold - margin
    
    Args:
        scaled_logits: [B, Num_Classes] - 经过 Temperature 缩放后的 Logits
        targets: [B, Num_Classes] - 真实标签（0/1）
        class_thresholds: [Num_Classes] - 类别级可学习阈值参数（未经过 Sigmoid）
        margin: float - 边界值，默认 0.1
    
    Returns:
        loss: 阈值一致性损失值
    """
    # 计算阈值和概率
    thresholds = torch.sigmoid(class_thresholds).unsqueeze(0)  # [1, Num_Classes]
    probs = torch.sigmoid(scaled_logits)  # [B, Num_Classes]
    
    # 正样本和负样本的掩码
    pos_mask = (targets == 1)
    neg_mask = (targets == 0)
    
    # 惩罚那些低于阈值的正样本
    loss_thresh_pos = 0.0
    if pos_mask.sum() > 0:
        loss_thresh_pos = F.relu(thresholds - probs + margin)[pos_mask].mean()
    
    # 惩罚那些高于阈值的负样本
    loss_thresh_neg = 0.0
    if neg_mask.sum() > 0:
        loss_thresh_neg = F.relu(probs - thresholds + margin)[neg_mask].mean()
    
    return loss_thresh_pos + loss_thresh_neg


class IntentonomyViTModule(LightningModule):
    """`LightningModule` for Intentonomy multi-label classification using Vision Transformer with DiscriminativeClueMiner.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        # DiscriminativeClueMiner 参数
        use_mcc: bool = True,
        k_patches: int = 16,
        gcn_layers: int = 2,
        input_dim: int = 768,
        use_adaptive_temperature: bool = False,  # 是否使用自适应温度缩放
        use_global_feature: bool = False,  # 是否使用全局特征拼接
        use_learnable_fusion: bool = False,  # 是否使用可学习的融合权重
        use_image_level_temperature: bool = False,  # 是否使用 Image-Level 温度头
        use_cosine_similarity_temperature: bool = False,  # 是否使用基于余弦相似度的温度计算
        use_max_pooling_temperature: bool = False,  # 是否使用 Max Pooling 辅助温度计算（双通道一致性）
        use_consistency_bias: bool = False,  # 是否使用 consistency bias（默认关闭）
        # Label embedding 参数
        label_embedding_path: str = "Intentonomy/data/label_embedding_300_28",
        label_embedding_type: str = "default",  # "default" 或 "clip"，用于自动选择embedding路径
        # Label adjacency matrix 参数
        label_adjacency_matrix_path: str = None,  # 标签共现概率邻接矩阵路径
        # Sparsity loss 参数
        use_sparsity_loss: bool = True,
        sparsity_loss_weight: float = 0.1,
        target_sparsity: float = 0.2,
        sparsity_loss_type: str = "default",  # "default" 或 "smart"
        sparsity_loss_k: int = 3,  # 仅在 sparsity_loss_type="smart" 时使用
        # Correlation loss 参数
        use_correlation_loss: bool = True,
        correlation_loss_weight: float = 0.1,
        # Class thresholds 参数
        use_class_thresholds: bool = False,  # 是否使用类别级可学习阈值（默认关闭）
        # EMA 参数
        use_ema: bool = True,
        ema_decay: float = 0.9997,
    ) -> None:
        """Initialize a `IntentonomyViTModule` with DiscriminativeClueMiner.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized with default params.
        :param use_mcc: Whether to use DiscriminativeClueMiner. If False, use standard forward.
        :param k_patches: Number of top-k patches to select in DiscriminativeClueMiner.
        :param gcn_layers: Number of GCN layers in DiscriminativeClueMiner.
        :param input_dim: Input dimension for DiscriminativeClueMiner (should match ViT hidden dim).
        :param use_adaptive_temperature: Whether to use adaptive temperature scaling in LabelGuidedVerifier (default False).
        :param use_global_feature: Whether to concatenate global average pooled feature with top-K features (default False).
        :param use_learnable_fusion: Whether to use learnable fusion weight for combining local and global features (default False).
        :param use_image_level_temperature: Whether to use Image-Level temperature head for adaptive logits scaling (default False).
        :param use_cosine_similarity_temperature: Whether to use Global-Local Cosine Similarity for dynamic temperature calculation (default False).
        :param use_consistency_bias: Whether to use consistency bias for adjusting consistency value before alpha calculation (default False).
        :param label_embedding_path: Path to label embedding file. If None and use_mcc=True, will raise error.
        :param label_embedding_type: Type of label embedding to use. Options: "default" (use label_embedding_path), 
                                     "clip" (use CLIP embedding from label_embedding_clip). Default: "default".
        :param use_sparsity_loss: Whether to use sparsity loss (default True).
        :param sparsity_loss_weight: Weight for sparsity loss (default 0.1).
        :param target_sparsity: Target sparsity ratio for patch scores (default 0.2, i.e., 20%).
        :param sparsity_loss_type: Type of sparsity loss to use. Options: "default" or "smart" (default "default").
        :param sparsity_loss_k: Number of top patches to protect from penalty when using "smart" loss (default 3).
        :param use_correlation_loss: Whether to use correlation loss (default True).
        :param correlation_loss_weight: Weight for correlation loss (default 0.1).
        :param use_class_thresholds: Whether to use class-level learnable thresholds (default False).
        :param use_ema: Whether to use Exponential Moving Average for model parameters (default True).
        :param ema_decay: Decay factor for EMA (default 0.9997).
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.num_classes = num_classes
        self.use_mcc = use_mcc
        self.use_sparsity_loss = use_sparsity_loss
        self.sparsity_loss_weight = sparsity_loss_weight
        self.target_sparsity = target_sparsity
        self.sparsity_loss_type = sparsity_loss_type
        self.sparsity_loss_k = sparsity_loss_k
        self.use_correlation_loss = use_correlation_loss
        self.correlation_loss_weight = correlation_loss_weight
        self.use_class_thresholds = use_class_thresholds

        # 加载 label_embedding
        if self.use_mcc:
            import os
            # 根据 label_embedding_type 自动选择路径
            if label_embedding_type == "clip":
                # 使用CLIP embedding
                base_dir = "/share/lmcp/tangyin/projects/IntentRecognition/Intentonomy/data"
                clip_embedding_path = os.path.join(base_dir, "label_embedding_clip")
                if not os.path.exists(clip_embedding_path):
                    raise FileNotFoundError(
                        f"CLIP embedding file not found at {clip_embedding_path}. "
                        f"Please run generate_clip_embeddings.py first to generate CLIP embeddings."
                    )
                actual_embedding_path = clip_embedding_path
            else:
                # 使用 label_embedding_path（如果为None，使用默认路径）
                if label_embedding_path is None:
                    # 尝试多个可能的路径
                    possible_paths = [
                        os.path.join("/share/lmcp/tangyin/projects/IntentRecognition", "Intentonomy/data/label_embedding_300_28"),
                        os.path.join("/home/evelynmuir/lambda/projects/IntentRecognition", "Intentonomy/data/label_embedding_300_28"),
                        "Intentonomy/data/label_embedding_300_28"
                    ]
                    actual_embedding_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            actual_embedding_path = path
                            break
                    if actual_embedding_path is None:
                        raise FileNotFoundError(
                            f"Label embedding file not found. Tried paths: {possible_paths}. "
                            f"Please specify label_embedding_path in config."
                        )
                else:
                    actual_embedding_path = label_embedding_path
            
            # 使用 torch.load(weights_only=False) 加载 label_embedding
            if not os.path.exists(actual_embedding_path):
                raise FileNotFoundError(
                    f"Label embedding file not found at {actual_embedding_path}. "
                    f"Please check the path in config."
                )
            label_embedding_data = torch.load(actual_embedding_path, weights_only=False)
            # 确保是 tensor 格式，并限制到 num_classes
            if isinstance(label_embedding_data, torch.Tensor):
                label_embedding_data = label_embedding_data[:num_classes].float()  # 强制转换为 float32
            else:
                # 处理 list 或 numpy array 格式
                import numpy as np
                if isinstance(label_embedding_data, list):
                    # 检查是否是字符串列表（错误的数据格式）
                    if len(label_embedding_data) > 0 and isinstance(label_embedding_data[0], str):
                        raise ValueError(
                            f"label_embedding_path points to a file containing strings, not embeddings. "
                            f"Expected numerical embeddings but got string list. "
                            f"Please check the file path. First element: {label_embedding_data[0][:50]}..."
                        )
                    # 转换为 numpy array 再转为 tensor
                    label_embedding_data = np.array(label_embedding_data[:num_classes])
                elif isinstance(label_embedding_data, np.ndarray):
                    label_embedding_data = label_embedding_data[:num_classes]
                label_embedding_data = torch.tensor(label_embedding_data, dtype=torch.float32)
            
            # 获取 embedding 维度
            embedding_dim = label_embedding_data.shape[-1]
            
            # 创建 nn.Embedding 层
            self.label_embedding = nn.Embedding(num_classes, embedding_dim)
            # 使用加载的数据初始化 embedding 权重
            with torch.no_grad():
                self.label_embedding.weight.data = label_embedding_data
            
            # 加载标签邻接矩阵（如果提供）
            adj_matrix = None
            if label_adjacency_matrix_path is not None:
                import os
                if os.path.exists(label_adjacency_matrix_path):
                    adj_matrix = torch.load(label_adjacency_matrix_path, weights_only=False)
                    if isinstance(adj_matrix, torch.Tensor):
                        # 确保形状正确
                        if adj_matrix.shape != (num_classes, num_classes):
                            print(f"警告: 邻接矩阵形状 {adj_matrix.shape} 与 num_classes {num_classes} 不匹配，将截断或填充")
                            if adj_matrix.shape[0] >= num_classes and adj_matrix.shape[1] >= num_classes:
                                adj_matrix = adj_matrix[:num_classes, :num_classes]
                            else:
                                print(f"错误: 邻接矩阵太小，无法截断")
                                adj_matrix = None
                    else:
                        print(f"警告: 邻接矩阵不是 torch.Tensor 格式，跳过")
                        adj_matrix = None
                    if adj_matrix is not None:
                        print(f"成功加载标签邻接矩阵: {label_adjacency_matrix_path}, 形状: {adj_matrix.shape}")
                else:
                    print(f"警告: 邻接矩阵文件不存在: {label_adjacency_matrix_path}，将不使用 LabelGCN")
            
            # 初始化 DiscriminativeClueMiner
            self.mcc_miner = DiscriminativeClueMiner(
                vis_dim=input_dim,
                text_dim=embedding_dim,  # 使用 embedding 的维度
                num_classes=num_classes,
                k_patches=k_patches,
                gcn_depth=gcn_layers,
                use_adaptive_temperature=use_adaptive_temperature,
                use_global_feature=use_global_feature,
                use_learnable_fusion=use_learnable_fusion,
                use_image_level_temperature=use_image_level_temperature,
                use_cosine_similarity_temperature=use_cosine_similarity_temperature,  # 新增
                use_max_pooling_temperature=use_max_pooling_temperature,  # 新增：Max Pooling 辅助温度计算
                use_consistency_bias=use_consistency_bias,  # 新增：Consistency Bias
                adj_matrix=adj_matrix
            )
        else:
            self.mcc_miner = None
            self.label_embedding = None

        # loss function for multi-label classification
        if criterion is None:
            self.criterion = AsymmetricLossOptimized()
        else:
            self.criterion = criterion

        # 【新增】类别级可学习阈值（可选）
        # 初始化为 0，经过 Sigmoid 后就是 0.5 (标准阈值)
        # 这是一个 Parameter，会随着训练自动调整
        if self.use_class_thresholds:
            self.class_thresholds = nn.Parameter(torch.zeros(num_classes))
        else:
            self.class_thresholds = None

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking sparsity loss
        self.train_sparsity_loss = MeanMetric()
        self.val_sparsity_loss = MeanMetric()
        
        # for tracking correlation loss
        self.train_correlation_loss = MeanMetric()
        self.val_correlation_loss = MeanMetric()
        
        # for tracking threshold consistency loss
        self.train_threshold_consistency_loss = MeanMetric()
        self.val_threshold_consistency_loss = MeanMetric()

        # for tracking best so far validation metrics (使用 HLEG 计算的 macro F1)
        self.val_f1_macro_best = MaxMetric()
        
        # 用于收集验证和测试的预测和标签，以便使用 HLEG 的计算方式
        self.val_preds_list = []
        self.val_targets_list = []
        self.test_preds_list = []
        self.test_targets_list = []
        
        # EMA 相关
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_model = None  # 将在 on_train_start 中初始化
        # EMA模型的预测列表
        self.val_ema_preds_list = []
        self.val_ema_targets_list = []
        self.test_ema_preds_list = []
        self.test_ema_targets_list = []

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """从 ViT backbone 中提取 patch features.
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Patch features of shape (batch_size, num_patches, hidden_dim).
        """
        # 访问 ViT backbone
        backbone = self.net.backbone
        
        # torchvision 的 ViT 结构: conv_proj -> encoder -> heads
        # 将图像转换为 patches
        x = backbone.conv_proj(x)  # [B, hidden_dim, H', W']
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N_patches, hidden_dim]
        
        # 添加 CLS token
        # 注意：encoder.forward() 内部会自动添加位置编码，所以这里不需要手动添加
        batch_size = x.shape[0]
        cls_token = backbone.class_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N_patches, hidden_dim]
        
        # 通过 encoder
        # torchvision ViT encoder 的 forward 方法期望输入为 [B, N, hidden_dim] 格式
        # encoder.forward() 内部会：
        # 1. 添加位置编码：input = input + self.pos_embedding
        # 2. 通过 transformer layers
        # 3. 应用最后的 layer norm：return self.ln(self.layers(self.dropout(input)))
        x = backbone.encoder(x)  # [B, 1+N_patches, hidden_dim]
        
        # 返回 patch tokens (去掉 CLS token，即 x[:, 1:, :])
        # DiscriminativeClueMiner 需要的是 patch features，不包括 CLS token
        # 根据 DiscriminativeClueMiner 的注释，应该传入 x[:, 1:, :]
        return x[:, 1:, :]  # [B, N_patches, hidden_dim]

    def forward(self, x: torch.Tensor, return_selection_info: bool = False) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: A tensor of images.
        :param return_selection_info: Whether to return selection info (patch_scores, etc.) for loss computation.
        :return: A tensor of logits, or (logits, selection_info) if return_selection_info=True.
        """
        if self.use_mcc and self.mcc_miner is not None:
            # 使用 DiscriminativeClueMiner
            patch_features = self._extract_patch_features(x)
            # 传入 label_embedding 的权重
            logits, selection_info = self.mcc_miner(patch_features, self.label_embedding.weight)
            # print(selection_info)
            if return_selection_info:
                return logits, selection_info
            return logits
        else:
            # 标准 forward
            return self.net(x)
    
    def _forward_with_ema(self, x: torch.Tensor, return_selection_info: bool = False) -> torch.Tensor:
        """Perform a forward pass through the EMA model.

        :param x: A tensor of images.
        :param return_selection_info: Whether to return selection info (patch_scores, etc.) for loss computation.
        :return: A tensor of logits, or (logits, selection_info) if return_selection_info=True.
        """
        if self.ema_model is None:
            raise RuntimeError("EMA model is not initialized. Call on_train_start first.")
        
        ema_module = self.ema_model.module
        if ema_module.use_mcc and ema_module.mcc_miner is not None:
            # 使用EMA模型的DiscriminativeClueMiner
            # 从EMA模型的backbone提取特征
            backbone = ema_module.net.backbone
            x_patches = backbone.conv_proj(x)
            B, C, H, W = x_patches.shape
            x_patches = x_patches.reshape(B, C, H * W).permute(0, 2, 1)
            batch_size = x_patches.shape[0]
            cls_token = backbone.class_token.expand(batch_size, -1, -1)
            x_patches = torch.cat([cls_token, x_patches], dim=1)
            x_patches = backbone.encoder(x_patches)
            patch_features = x_patches[:, 1:, :]
            
            # 使用EMA模型的mcc_miner和label_embedding
            logits, selection_info = ema_module.mcc_miner(patch_features, ema_module.label_embedding.weight)
            if return_selection_info:
                return logits, selection_info
            return logits
        else:
            # 标准 forward
            return ema_module.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_f1_macro_best.reset()
        self.val_preds_list.clear()
        self.val_targets_list.clear()
        if self.use_mcc and self.use_sparsity_loss:
            self.train_sparsity_loss.reset()
            self.val_sparsity_loss.reset()
        if self.use_correlation_loss:
            self.train_correlation_loss.reset()
            self.val_correlation_loss.reset()
        self.train_threshold_consistency_loss.reset()
        self.val_threshold_consistency_loss.reset()
        
        # 初始化EMA模型
        if self.use_ema and self.ema_model is None:
            self.ema_model = ModelEma(self, decay=self.ema_decay)
            # 初始化EMA模型参数为当前模型参数
            self.ema_model.set(self)

    def model_step(
        self, batch: Dict[str, torch.Tensor], use_ema_model: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param use_ema_model: Whether to use EMA model for forward pass (default False).

        :return: A tuple containing (in order):
            - A tensor of total losses (classification + sparsity + correlation + threshold_consistency).
            - A tensor of predictions (probabilities after sigmoid).
            - A tensor of target labels.
            - A tensor of sparsity loss values.
            - A tensor of correlation loss values.
            - A tensor of threshold consistency loss values.
        """
        x = batch["image"]
        y = batch["labels"]
        
        # Forward pass
        if use_ema_model:
            forward_fn = self._forward_with_ema
        else:
            forward_fn = self.forward
        
        if self.use_mcc and self.mcc_miner is not None:
            logits, selection_info = forward_fn(x, return_selection_info=True)
            # 计算分类 loss
            classification_loss = self.criterion(logits, y)
            
            # 计算 sparsity loss（如果启用）
            if self.use_sparsity_loss:
                patch_scores = selection_info["patch_scores"]  # [B, N, 1]
                if self.sparsity_loss_type == "smart":
                    sparsity_loss_val = sparsity_loss_smart(patch_scores, k=self.sparsity_loss_k)
                else:
                    sparsity_loss_val = sparsity_loss(patch_scores, target_sparsity=self.target_sparsity)
            else:
                sparsity_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算 correlation loss（如果启用）
            if self.use_correlation_loss:
                correlation_loss_val = correlation_loss(logits, y)
            else:
                correlation_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算阈值一致性损失（如果启用）
            if self.use_class_thresholds and self.class_thresholds is not None:
                threshold_consistency_loss_val = threshold_consistency_loss(
                    logits, y, self.class_thresholds, margin=0.1
                )
            else:
                threshold_consistency_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 总 loss: Loss_Total = Loss_ASL + weight_sparsity * Loss_Sparsity + weight_corr * Loss_Corr + 0.1 * Loss_Threshold
            loss = classification_loss + self.sparsity_loss_weight * sparsity_loss_val + self.correlation_loss_weight * correlation_loss_val + 0.1 * threshold_consistency_loss_val
        else:
            logits = forward_fn(x)
            classification_loss = self.criterion(logits, y)
            sparsity_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算 correlation loss（如果启用）
            if self.use_correlation_loss:
                correlation_loss_val = correlation_loss(logits, y)
            else:
                correlation_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算阈值一致性损失（如果启用）
            if self.use_class_thresholds and self.class_thresholds is not None:
                threshold_consistency_loss_val = threshold_consistency_loss(
                    logits, y, self.class_thresholds, margin=0.1
                )
            else:
                threshold_consistency_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 总 loss
            loss = classification_loss + self.correlation_loss_weight * correlation_loss_val + 0.1 * threshold_consistency_loss_val
        
        preds = torch.sigmoid(logits)  # Convert logits to probabilities
        return loss, preds, y, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # update and log sparsity loss
        if self.use_mcc and self.use_sparsity_loss:
            self.train_sparsity_loss(sparsity_loss_val)
            self.log("train/sparsity_loss", self.train_sparsity_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log correlation loss
        if self.use_correlation_loss:
            self.train_correlation_loss(correlation_loss_val)
            self.log("train/correlation_loss", self.train_correlation_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log threshold consistency loss
        self.train_threshold_consistency_loss(threshold_consistency_loss_val)
        self.log("train/threshold_consistency_loss", self.train_threshold_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Lightning hook that is called after a training batch ends."""
        # 更新EMA模型
        if self.use_ema and self.ema_model is not None:
            self.ema_model.update(self)
    
    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # 使用原始模型
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val = self.model_step(batch, use_ema_model=False)

        # update and log loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # update and log sparsity loss
        if self.use_mcc and self.use_sparsity_loss:
            self.val_sparsity_loss(sparsity_loss_val)
            self.log("val/sparsity_loss", self.val_sparsity_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log correlation loss
        if self.use_correlation_loss:
            self.val_correlation_loss(correlation_loss_val)
            self.log("val/correlation_loss", self.val_correlation_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log threshold consistency loss
        self.val_threshold_consistency_loss(threshold_consistency_loss_val)
        self.log("val/threshold_consistency_loss", self.val_threshold_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        # 收集预测和标签用于 HLEG 计算方式（原始模型）
        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())
        
        # 如果使用EMA，也使用EMA模型进行推理
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.val_ema_preds_list.append(ema_preds.detach().cpu())
            self.val_ema_targets_list.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # 获取类别级阈值（如果启用）
        thresholds = None
        if self.use_class_thresholds and self.class_thresholds is not None:
            thresholds = torch.sigmoid(self.class_thresholds).detach().cpu().numpy()
        
        # 使用 HLEG 的计算方式计算 metrics（原始模型）
        if len(self.val_preds_list) > 0:
            # 合并所有批次的预测和标签
            val_preds_all = torch.cat(self.val_preds_list, dim=0).numpy()
            val_targets_all = torch.cat(self.val_targets_list, dim=0).numpy()
            
            # 使用 HLEG 的计算方式，传递类别级阈值（如果启用）
            f1_dict = eval_validation_set(val_preds_all, val_targets_all, class_thresholds=thresholds)
            
            # 更新最佳 macro F1
            self.val_f1_macro_best(f1_dict["val_macro"])
            
            # 记录 metrics
            self.log("val/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val/threshold", f1_dict["threshold"], sync_dist=True)
            self.log("val/easy", f1_dict["val_easy"], sync_dist=True)
            self.log("val/medium", f1_dict["val_medium"], sync_dist=True)
            self.log("val/hard", f1_dict["val_hard"], sync_dist=True)
            
            # 清空列表以便下次验证
            self.val_preds_list.clear()
            self.val_targets_list.clear()
        
        # 如果使用EMA，也计算EMA模型的metrics
        if self.use_ema and self.ema_model is not None and len(self.val_ema_preds_list) > 0:
            val_ema_preds_all = torch.cat(self.val_ema_preds_list, dim=0).numpy()
            val_ema_targets_all = torch.cat(self.val_ema_targets_list, dim=0).numpy()
            
            # 使用EMA模型的阈值（如果启用）
            ema_thresholds = None
            if self.use_class_thresholds:
                ema_module = self.ema_model.module
                if ema_module.class_thresholds is not None:
                    ema_thresholds = torch.sigmoid(ema_module.class_thresholds).detach().cpu().numpy()
            f1_dict_ema = eval_validation_set(val_ema_preds_all, val_ema_targets_all, class_thresholds=ema_thresholds)
            
            # 记录EMA模型的metrics
            self.log("val_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("val_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            self.log("val_ema/easy", f1_dict_ema["val_easy"], sync_dist=True)
            self.log("val_ema/medium", f1_dict_ema["val_medium"], sync_dist=True)
            self.log("val_ema/hard", f1_dict_ema["val_hard"], sync_dist=True)
            
            # 清空EMA列表
            self.val_ema_preds_list.clear()
            self.val_ema_targets_list.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # 使用原始模型
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val = self.model_step(batch, use_ema_model=False)

        # update and log loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # log sparsity loss (test阶段通常不需要累积)
        if self.use_mcc and self.use_sparsity_loss:
            self.log("test/sparsity_loss", sparsity_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        # log correlation loss (test阶段通常不需要累积)
        if self.use_correlation_loss:
            self.log("test/correlation_loss", correlation_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        # log threshold consistency loss (test阶段通常不需要累积)
        self.log("test/threshold_consistency_loss", threshold_consistency_loss_val, on_step=False, on_epoch=True, prog_bar=False)

        # 收集预测和标签用于 HLEG 计算方式（原始模型）
        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())
        
        # 如果使用EMA，也使用EMA模型进行推理
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.test_ema_preds_list.append(ema_preds.detach().cpu())
            self.test_ema_targets_list.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # 获取类别级阈值（如果启用）
        thresholds = None
        if self.use_class_thresholds and self.class_thresholds is not None:
            thresholds = torch.sigmoid(self.class_thresholds).detach().cpu().numpy()
        
        # 使用 HLEG 的计算方式计算 metrics（原始模型）
        if len(self.test_preds_list) > 0:
            # 合并所有批次的预测和标签
            test_preds_all = torch.cat(self.test_preds_list, dim=0).numpy()
            test_targets_all = torch.cat(self.test_targets_list, dim=0).numpy()
            
            dual_f1_dict = eval_test_set_both_strategies(
                test_preds_all, test_targets_all, class_thresholds=thresholds
            )
            for strategy_name, metrics in dual_f1_dict.items():
                self.log(f"test/{strategy_name}/f1_micro", metrics["val_micro"], sync_dist=True, prog_bar=True)
                self.log(f"test/{strategy_name}/f1_macro", metrics["val_macro"], sync_dist=True, prog_bar=True)
                self.log(f"test/{strategy_name}/f1_samples", metrics["val_samples"], sync_dist=True)
                self.log(f"test/{strategy_name}/f1_mean", (metrics["val_micro"] + metrics["val_macro"] + metrics["val_samples"]) / 3.0, sync_dist=True)
                self.log(f"test/{strategy_name}/mAP", metrics["val_mAP"], sync_dist=True, prog_bar=True)
                self.log(f"test/{strategy_name}/threshold", metrics["threshold"], sync_dist=True)
                self.log(f"test/{strategy_name}/easy", metrics["val_easy"], sync_dist=True)
                self.log(f"test/{strategy_name}/medium", metrics["val_medium"], sync_dist=True)
                self.log(f"test/{strategy_name}/hard", metrics["val_hard"], sync_dist=True)

            # Backward-compatible aliases (legacy behavior = no inference strategy)
            f1_dict = dual_f1_dict["no_inference_strategy"]
            self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("test/f1_mean", (f1_dict["val_micro"] + f1_dict["val_macro"] + f1_dict["val_samples"]) / 3.0, sync_dist=True)
            self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
            self.log("test/easy", f1_dict["val_easy"], sync_dist=True)
            self.log("test/medium", f1_dict["val_medium"], sync_dist=True)
            self.log("test/hard", f1_dict["val_hard"], sync_dist=True)
            
            # 清空列表
            self.test_preds_list.clear()
            self.test_targets_list.clear()
        
        # 如果使用EMA，也计算EMA模型的metrics
        if self.use_ema and self.ema_model is not None and len(self.test_ema_preds_list) > 0:
            test_ema_preds_all = torch.cat(self.test_ema_preds_list, dim=0).numpy()
            test_ema_targets_all = torch.cat(self.test_ema_targets_list, dim=0).numpy()
            
            # 使用EMA模型的阈值（如果启用）
            ema_thresholds = None
            if self.use_class_thresholds:
                ema_module = self.ema_model.module
                if ema_module.class_thresholds is not None:
                    ema_thresholds = torch.sigmoid(ema_module.class_thresholds).detach().cpu().numpy()
            dual_f1_dict_ema = eval_test_set_both_strategies(
                test_ema_preds_all, test_ema_targets_all, class_thresholds=ema_thresholds
            )
            for strategy_name, metrics in dual_f1_dict_ema.items():
                self.log(f"test_ema/{strategy_name}/f1_micro", metrics["val_micro"], sync_dist=True, prog_bar=True)
                self.log(f"test_ema/{strategy_name}/f1_macro", metrics["val_macro"], sync_dist=True, prog_bar=True)
                self.log(f"test_ema/{strategy_name}/f1_samples", metrics["val_samples"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/f1_mean", (metrics["val_micro"] + metrics["val_macro"] + metrics["val_samples"]) / 3.0, sync_dist=True)
                self.log(f"test_ema/{strategy_name}/mAP", metrics["val_mAP"], sync_dist=True, prog_bar=True)
                self.log(f"test_ema/{strategy_name}/threshold", metrics["threshold"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/easy", metrics["val_easy"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/medium", metrics["val_medium"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/hard", metrics["val_hard"], sync_dist=True)

            # Backward-compatible aliases (legacy behavior = no inference strategy)
            f1_dict_ema = dual_f1_dict_ema["no_inference_strategy"]
            self.log("test_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("test_ema/f1_mean", (f1_dict_ema["val_micro"] + f1_dict_ema["val_macro"] + f1_dict_ema["val_samples"]) / 3.0, sync_dist=True)
            self.log("test_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            self.log("test_ema/easy", f1_dict_ema["val_easy"], sync_dist=True)
            self.log("test_ema/medium", f1_dict_ema["val_medium"], sync_dist=True)
            self.log("test_ema/hard", f1_dict_ema["val_hard"], sync_dist=True)
            
            # 清空EMA列表
            self.test_ema_preds_list.clear()
            self.test_ema_targets_list.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # 获取基础学习率
        # 从 optimizer 的 keywords 中获取 lr（因为使用了 _partial_）
        if hasattr(self.hparams.optimizer, 'keywords'):
            base_lr = self.hparams.optimizer.keywords.get('lr', 0.00001)
        else:
            # 如果没有 keywords 属性，使用默认值
            base_lr = 0.00001
        
        # 分离 ViT、DCM 和 log_temperature 的参数
        vit_params = []
        dcm_params = []
        log_temperature_params = []
        
        # ViT backbone 的参数
        if self.net is not None:
            vit_params.extend(self.net.parameters())
        
        # DCM 的参数（如果使用）
        if self.use_mcc and self.mcc_miner is not None:
            # 检查是否有 verifier 和 log_temperature 参数
            if hasattr(self.mcc_miner, 'verifier') and hasattr(self.mcc_miner.verifier, 'log_temperature'):
                # 将 log_temperature 单独分离出来
                log_temperature_params.append(self.mcc_miner.verifier.log_temperature)
                # 获取 DCM 的其他参数（排除 log_temperature）
                for name, param in self.mcc_miner.named_parameters():
                    if 'log_temperature' not in name:
                        dcm_params.append(param)
            else:
                # 如果没有 log_temperature，使用所有 DCM 参数
                dcm_params.extend(self.mcc_miner.parameters())
        
        # 创建参数组：
        # - ViT 使用基础学习率
        # - DCM 使用 10 倍学习率
        # - log_temperature 使用 base_lr / 10
        param_groups = []
        if vit_params:
            param_groups.append({
                'params': vit_params,
                'lr': base_lr
            })
        if dcm_params:
            param_groups.append({
                'params': dcm_params,
                'lr': base_lr * 10.0
            })
        if log_temperature_params:
            param_groups.append({
                'params': log_temperature_params,
                'lr': base_lr / 10.0
            })
        
        # 如果没有分离参数，使用所有参数（向后兼容）
        if not param_groups:
            param_groups = [{'params': self.parameters()}]
        
        # 创建优化器，使用参数组
        optimizer = self.hparams.optimizer(params=param_groups)
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
