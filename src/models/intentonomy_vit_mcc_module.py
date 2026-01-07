from typing import Any, Dict, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.metrics import eval_validation_set


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


class LabelGuidedVerifier(nn.Module):
    def __init__(self, vis_dim, num_classes, num_heads=4):
        super().__init__()
        # 标准的 Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dim=vis_dim, num_heads=num_heads, batch_first=True)
        
        # 【核心创新】为每个类别定义一个可学习的温度参数
        # 初始化为 0 (即 exp(0) = 1.0，标准 Attention)
        self.log_temperature = nn.Parameter(torch.zeros(num_classes)) 
        
    def forward(self, label_queries, visual_evidence):
        """
        label_queries: [B, Num_Classes, D] - 意图标签 Embedding
        visual_evidence: [B, K, D] - 经过 GCN 的视觉线索
        """
        # 1. 计算温度 tau
        # 使用 exp 保证温度恒为正数，且数值稳定
        # shape: [Num_Classes] -> [1, Num_Classes, 1] 以便广播
        tau = torch.exp(self.log_temperature).view(1, -1, 1)
        
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
                 use_adaptive_temperature=False):  # 是否使用自适应温度缩放
        super().__init__()
        
        self.vis_dim = vis_dim
        self.k = k_patches
        self.use_adaptive_temperature = use_adaptive_temperature
        
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
        
        # 最后的 Logit 缩放因子 (可选，有助于收敛)
        self.scale = nn.Parameter(torch.ones([]) * (vis_dim ** -0.5))

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
        label_queries = self.label_proj(label_embeddings) # [Num_Classes, Vis_D]
        # 扩展到 Batch: [B, Num_Classes, Vis_D]
        label_queries = label_queries.unsqueeze(0).expand(B, -1, -1)
        
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
        
        # 4.3 预测 Logits (Visual-Semantic Similarity)
        # 计算对齐后的视觉特征与标签语义的点积
        # [B, NC, D] * [B, NC, D] -> sum -> [B, NC]
        logits = (attn_out * label_queries).sum(dim=-1) * self.scale
        
        # 构建返回字典
        aux_dict = {
            "patch_scores": scores,        # 用于 Sparsity Loss
            "topk_indices": topk_indices,  # 用于可视化 (Fig 5)
            "adj_matrix": adj_matrix,      # 用于可视化图结构
            "attn_weights": attn_weights   # Attention 权重
        }
        
        # 只有在使用自适应温度时才添加 learned_tau
        if self.use_adaptive_temperature:
            aux_dict["learned_tau"] = learned_tau  # 学习到的温度参数，用于做分析图表！
        
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
        # Label embedding 参数
        label_embedding_path: str = None,
        # Sparsity loss 参数
        use_sparsity_loss: bool = True,
        sparsity_loss_weight: float = 0.1,
        target_sparsity: float = 0.2,
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
        :param label_embedding_path: Path to label embedding file. If None and use_mcc=True, will raise error.
        :param use_sparsity_loss: Whether to use sparsity loss (default True).
        :param sparsity_loss_weight: Weight for sparsity loss (default 0.1).
        :param target_sparsity: Target sparsity ratio for patch scores (default 0.2, i.e., 20%).
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

        # 加载 label_embedding
        if self.use_mcc:
            if label_embedding_path is None:
                raise ValueError("label_embedding_path must be provided when use_mcc=True")
            # 使用 torch.load(weights_only=False) 加载 label_embedding
            label_embedding = torch.load(label_embedding_path, weights_only=False)
            # 确保是 tensor 格式，并限制到 num_classes
            if isinstance(label_embedding, torch.Tensor):
                label_embedding = label_embedding[:num_classes]
            else:
                # 处理 list 或 numpy array 格式
                import numpy as np
                if isinstance(label_embedding, list):
                    # 检查是否是字符串列表（错误的数据格式）
                    if len(label_embedding) > 0 and isinstance(label_embedding[0], str):
                        raise ValueError(
                            f"label_embedding_path points to a file containing strings, not embeddings. "
                            f"Expected numerical embeddings but got string list. "
                            f"Please check the file path. First element: {label_embedding[0][:50]}..."
                        )
                    # 转换为 numpy array 再转为 tensor
                    label_embedding = np.array(label_embedding[:num_classes])
                elif isinstance(label_embedding, np.ndarray):
                    label_embedding = label_embedding[:num_classes]
                label_embedding = torch.tensor(label_embedding, dtype=torch.float32)
            # 注册为 buffer，这样会被包含在 state_dict 中，但不会作为可训练参数
            # 同时会自动处理设备移动
            self.register_buffer('label_embedding', label_embedding)
            
            # 初始化 DiscriminativeClueMiner
            self.mcc_miner = DiscriminativeClueMiner(
                vis_dim=input_dim,
                text_dim=label_embedding.shape[-1],  # 使用 label_embedding 的维度
                num_classes=num_classes,
                k_patches=k_patches,
                gcn_depth=gcn_layers,
                use_adaptive_temperature=use_adaptive_temperature
            )
        else:
            self.mcc_miner = None
            self.label_embedding = None

        # loss function for multi-label classification
        if criterion is None:
            self.criterion = AsymmetricLossOptimized()
        else:
            self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking sparsity loss
        self.train_sparsity_loss = MeanMetric()
        self.val_sparsity_loss = MeanMetric()

        # for tracking best so far validation metrics (使用 HLEG 计算的 macro F1)
        self.val_f1_macro_best = MaxMetric()
        
        # 用于收集验证和测试的预测和标签，以便使用 HLEG 的计算方式
        self.val_preds_list = []
        self.val_targets_list = []
        self.test_preds_list = []
        self.test_targets_list = []

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
            # 传入 label_embedding
            logits, selection_info = self.mcc_miner(patch_features, self.label_embedding)
            # print(selection_info)
            if return_selection_info:
                return logits, selection_info
            return logits
        else:
            # 标准 forward
            return self.net(x)

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

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of total losses (classification + sparsity).
            - A tensor of predictions (probabilities after sigmoid).
            - A tensor of target labels.
            - A tensor of sparsity loss values.
        """
        x = batch["image"]
        y = batch["labels"]
        
        # Forward pass
        if self.use_mcc and self.mcc_miner is not None:
            logits, selection_info = self.forward(x, return_selection_info=True)
            # 计算分类 loss
            classification_loss = self.criterion(logits, y)
            # 计算 sparsity loss（如果启用）
            if self.use_sparsity_loss:
                patch_scores = selection_info["patch_scores"]  # [B, N, 1]
                sparsity_loss_val = sparsity_loss(patch_scores, target_sparsity=self.target_sparsity)
                # 总 loss
                loss = classification_loss + self.sparsity_loss_weight * sparsity_loss_val
            else:
                loss = classification_loss
                sparsity_loss_val = torch.tensor(0.0, device=loss.device)
        else:
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            sparsity_loss_val = torch.tensor(0.0, device=loss.device)
        
        preds = torch.sigmoid(logits)  # Convert logits to probabilities
        return loss, preds, y, sparsity_loss_val

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, sparsity_loss_val = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # update and log sparsity loss
        if self.use_mcc and self.use_sparsity_loss:
            self.train_sparsity_loss(sparsity_loss_val)
            self.log("train/sparsity_loss", self.train_sparsity_loss, on_step=False, on_epoch=True, prog_bar=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, sparsity_loss_val = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # update and log sparsity loss
        if self.use_mcc and self.use_sparsity_loss:
            self.val_sparsity_loss(sparsity_loss_val)
            self.log("val/sparsity_loss", self.val_sparsity_loss, on_step=False, on_epoch=True, prog_bar=False)

        # 收集预测和标签用于 HLEG 计算方式
        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # 使用 HLEG 的计算方式计算 metrics
        if len(self.val_preds_list) > 0:
            # 合并所有批次的预测和标签
            val_preds_all = torch.cat(self.val_preds_list, dim=0).numpy()
            val_targets_all = torch.cat(self.val_targets_list, dim=0).numpy()
            
            # 使用 HLEG 的计算方式
            f1_dict = eval_validation_set(val_preds_all, val_targets_all)
            
            # 更新最佳 macro F1
            self.val_f1_macro_best(f1_dict["val_macro"])
            
            # 记录 metrics
            self.log("val/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val/threshold", f1_dict["threshold"], sync_dist=True)
            
            # 清空列表以便下次验证
            self.val_preds_list.clear()
            self.val_targets_list.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, sparsity_loss_val = self.model_step(batch)

        # update and log loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # log sparsity loss (test阶段通常不需要累积)
        if self.use_mcc and self.use_sparsity_loss:
            self.log("test/sparsity_loss", sparsity_loss_val, on_step=False, on_epoch=True, prog_bar=False)

        # 收集预测和标签用于 HLEG 计算方式
        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # 使用 HLEG 的计算方式计算 metrics
        if len(self.test_preds_list) > 0:
            # 合并所有批次的预测和标签
            test_preds_all = torch.cat(self.test_preds_list, dim=0).numpy()
            test_targets_all = torch.cat(self.test_targets_list, dim=0).numpy()
            
            # 使用 HLEG 的计算方式
            f1_dict = eval_validation_set(test_preds_all, test_targets_all)
            
            # 记录 metrics
            self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
            
            # 清空列表
            self.test_preds_list.clear()
            self.test_targets_list.clear()

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
        
        # 分离 ViT 和 DCM 的参数
        vit_params = []
        dcm_params = []
        
        # ViT backbone 的参数
        if self.net is not None:
            vit_params.extend(self.net.parameters())
        
        # DCM 的参数（如果使用）
        if self.use_mcc and self.mcc_miner is not None:
            dcm_params.extend(self.mcc_miner.parameters())
        
        # 创建参数组：ViT 使用基础学习率，DCM 使用 10 倍学习率
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

