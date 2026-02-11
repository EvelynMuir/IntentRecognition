"""
CLIP-ViT version of Intentonomy module.
This module uses CLIP's Vision Transformer as backbone and CLIP label embeddings.
"""
from typing import Any, Dict, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

# Import all the shared components from the ViT module
from src.models.intentonomy_vit_mcc_module import (
    GraphReasoningLayer,
    LabelGCN,
    LabelGuidedVerifier,
    DiscriminativeClueMiner,
    sparsity_loss,
    sparsity_loss_smart,
    correlation_loss,
    threshold_consistency_loss,
)
from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.metrics import eval_validation_set
from src.utils.ema import ModelEma


class IntentonomyClipViTModule(LightningModule):
    """`LightningModule` for Intentonomy multi-label classification using CLIP Vision Transformer with DiscriminativeClueMiner.

    This module is similar to IntentonomyViTModule but uses CLIP's ViT backbone and CLIP label embeddings.
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
        input_dim: int = 512,  # CLIP ViT-B/32: 512, ViT-B/16: 768
        use_adaptive_temperature: bool = False,
        use_global_feature: bool = False,
        use_learnable_fusion: bool = False,
        use_image_level_temperature: bool = False,
        use_cosine_similarity_temperature: bool = False,
        use_max_pooling_temperature: bool = False,
        use_consistency_bias: bool = False,
        # Label embedding 参数 - 默认使用CLIP embedding
        label_embedding_path: str = None,
        label_embedding_type: str = "clip",  # 默认使用CLIP embedding
        # Label adjacency matrix 参数
        label_adjacency_matrix_path: str = None,
        # Sparsity loss 参数
        use_sparsity_loss: bool = True,
        sparsity_loss_weight: float = 0.1,
        target_sparsity: float = 0.2,
        sparsity_loss_type: str = "default",
        sparsity_loss_k: int = 3,
        # Correlation loss 参数
        use_correlation_loss: bool = True,
        correlation_loss_weight: float = 0.1,
        # Class thresholds 参数
        use_class_thresholds: bool = False,
        # EMA 参数
        use_ema: bool = True,
        ema_decay: float = 0.9997,
    ) -> None:
        """Initialize a `IntentonomyClipViTModule` with DiscriminativeClueMiner.

        :param net: The model to train (should be ClipVisionTransformer).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized with default params.
        :param use_mcc: Whether to use DiscriminativeClueMiner. If False, use standard forward.
        :param k_patches: Number of top-k patches to select in DiscriminativeClueMiner.
        :param gcn_layers: Number of GCN layers in DiscriminativeClueMiner.
        :param input_dim: Input dimension for DiscriminativeClueMiner (CLIP ViT-B/32: 512, ViT-B/16: 768).
        :param label_embedding_type: Type of label embedding to use. Default: "clip".
        :param use_sparsity_loss: Whether to use sparsity loss (default True).
        :param sparsity_loss_weight: Weight for sparsity loss (default 0.1).
        :param target_sparsity: Target sparsity ratio for patch scores (default 0.2).
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

        # 加载 label_embedding - 默认使用CLIP embedding
        if self.use_mcc:
            import os
            # 根据 label_embedding_type 自动选择路径
            if label_embedding_type == "clip":
                # 使用CLIP embedding - 尝试多个可能的路径
                possible_paths = [
                    os.path.join("/share/lmcp/tangyin/projects/IntentRecognition", "Intentonomy/data/label_embedding_clip"),
                    os.path.join("/home/evelynmuir/lambda/projects/IntentRecognition", "Intentonomy/data/label_embedding_clip"),
                    "Intentonomy/data/label_embedding_clip"
                ]
                actual_embedding_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        actual_embedding_path = path
                        break
                if actual_embedding_path is None:
                    raise FileNotFoundError(
                        f"CLIP embedding file not found. Tried paths: {possible_paths}. "
                        f"Please run generate_clip_embeddings.py first to generate CLIP embeddings."
                    )
            else:
                # 使用 label_embedding_path（如果为None，使用默认路径）
                if label_embedding_path is None:
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
                label_embedding_data = label_embedding_data[:num_classes].float()
            else:
                import numpy as np
                if isinstance(label_embedding_data, list):
                    if len(label_embedding_data) > 0 and isinstance(label_embedding_data[0], str):
                        raise ValueError(
                            f"label_embedding_path points to a file containing strings, not embeddings. "
                            f"Expected numerical embeddings but got string list. "
                            f"Please check the file path. First element: {label_embedding_data[0][:50]}..."
                        )
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
                text_dim=embedding_dim,
                num_classes=num_classes,
                k_patches=k_patches,
                gcn_depth=gcn_layers,
                use_adaptive_temperature=use_adaptive_temperature,
                use_global_feature=use_global_feature,
                use_learnable_fusion=use_learnable_fusion,
                use_image_level_temperature=use_image_level_temperature,
                use_cosine_similarity_temperature=use_cosine_similarity_temperature,
                use_max_pooling_temperature=use_max_pooling_temperature,
                use_consistency_bias=use_consistency_bias,
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

        # 类别级可学习阈值（可选）
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

        # for tracking best so far validation metrics
        self.val_f1_macro_best = MaxMetric()
        
        # 用于收集验证和测试的预测和标签
        self.val_preds_list = []
        self.val_targets_list = []
        self.test_preds_list = []
        self.test_targets_list = []
        
        # EMA 相关
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_model = None
        self.val_ema_preds_list = []
        self.val_ema_targets_list = []
        self.test_ema_preds_list = []
        self.test_ema_targets_list = []

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """从 CLIP ViT backbone 中提取 patch features.
        
        CLIP的ViT结构与torchvision不同，需要适配提取逻辑。
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Patch features of shape (batch_size, num_patches, hidden_dim).
        """
        # 访问 CLIP ViT backbone
        backbone = self.net.backbone
        
        # CLIP的ViT结构: conv1 -> reshape -> add CLS token -> add positional embedding -> ln_pre -> transformer -> ln_post
        # 1. Patch embedding (conv1)
        x = backbone.conv1(x)  # [B, hidden_dim, H', W']
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N_patches, hidden_dim]
        
        # 2. 添加 CLS token
        # CLIP的class_embedding是一个可学习的Parameter，形状为 [hidden_dim]
        batch_size = x.shape[0]
        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N_patches, hidden_dim]
        
        # 3. 添加位置编码
        # CLIP的positional_embedding是固定的，形状为 [1+N_patches, hidden_dim]
        x = x + backbone.positional_embedding.unsqueeze(0)  # [B, 1+N_patches, hidden_dim]
        
        # 4. Pre-layer norm
        x = backbone.ln_pre(x)  # [B, 1+N_patches, hidden_dim]
        
        # 5. 通过 transformer
        # CLIP的transformer期望输入为 [N, B, hidden_dim] 格式（seq_len, batch, hidden_dim）
        x = x.permute(1, 0, 2)  # [1+N_patches, B, hidden_dim]
        x = backbone.transformer(x)  # [1+N_patches, B, hidden_dim]
        x = x.permute(1, 0, 2)  # [B, 1+N_patches, hidden_dim]
        
        # 6. Post-layer norm (如果存在)
        if hasattr(backbone, 'ln_post'):
            x = backbone.ln_post(x)  # [B, 1+N_patches, hidden_dim]
        
        # 返回 patch tokens (去掉 CLS token，即 x[:, 1:, :])
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
            # 从EMA模型的backbone提取特征（使用CLIP ViT的提取方式）
            backbone = ema_module.net.backbone
            x_patches = backbone.conv1(x)
            B, C, H, W = x_patches.shape
            x_patches = x_patches.reshape(B, C, H * W).permute(0, 2, 1)
            batch_size = x_patches.shape[0]
            cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            x_patches = torch.cat([cls_token, x_patches], dim=1)
            x_patches = x_patches + backbone.positional_embedding.unsqueeze(0)
            x_patches = backbone.ln_pre(x_patches)
            x_patches = x_patches.permute(1, 0, 2)
            x_patches = backbone.transformer(x_patches)
            x_patches = x_patches.permute(1, 0, 2)
            if hasattr(backbone, 'ln_post'):
                x_patches = backbone.ln_post(x_patches)
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
                patch_scores = selection_info["patch_scores"]
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
            
            # 总 loss
            loss = classification_loss + self.sparsity_loss_weight * sparsity_loss_val + self.correlation_loss_weight * correlation_loss_val + 0.1 * threshold_consistency_loss_val
        else:
            logits = forward_fn(x)
            classification_loss = self.criterion(logits, y)
            sparsity_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            if self.use_correlation_loss:
                correlation_loss_val = correlation_loss(logits, y)
            else:
                correlation_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            if self.use_class_thresholds and self.class_thresholds is not None:
                threshold_consistency_loss_val = threshold_consistency_loss(
                    logits, y, self.class_thresholds, margin=0.1
                )
            else:
                threshold_consistency_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            loss = classification_loss + self.correlation_loss_weight * correlation_loss_val + 0.1 * threshold_consistency_loss_val
        
        preds = torch.sigmoid(logits)
        return loss, preds, y, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.use_mcc and self.use_sparsity_loss:
            self.train_sparsity_loss(sparsity_loss_val)
            self.log("train/sparsity_loss", self.train_sparsity_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        if self.use_correlation_loss:
            self.train_correlation_loss(correlation_loss_val)
            self.log("train/correlation_loss", self.train_correlation_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        self.train_threshold_consistency_loss(threshold_consistency_loss_val)
        self.log("train/threshold_consistency_loss", self.train_threshold_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Lightning hook that is called after a training batch ends."""
        if self.use_ema and self.ema_model is not None:
            self.ema_model.update(self)
    
    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val = self.model_step(batch, use_ema_model=False)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.use_mcc and self.use_sparsity_loss:
            self.val_sparsity_loss(sparsity_loss_val)
            self.log("val/sparsity_loss", self.val_sparsity_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        if self.use_correlation_loss:
            self.val_correlation_loss(correlation_loss_val)
            self.log("val/correlation_loss", self.val_correlation_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        self.val_threshold_consistency_loss(threshold_consistency_loss_val)
        self.log("val/threshold_consistency_loss", self.val_threshold_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())
        
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.val_ema_preds_list.append(ema_preds.detach().cpu())
            self.val_ema_targets_list.append(targets.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        thresholds = None
        if self.use_class_thresholds and self.class_thresholds is not None:
            thresholds = torch.sigmoid(self.class_thresholds).detach().cpu().numpy()
        
        if len(self.val_preds_list) > 0:
            val_preds_all = torch.cat(self.val_preds_list, dim=0).numpy()
            val_targets_all = torch.cat(self.val_targets_list, dim=0).numpy()
            
            f1_dict = eval_validation_set(val_preds_all, val_targets_all, class_thresholds=thresholds)
            
            self.val_f1_macro_best(f1_dict["val_macro"])
            
            self.log("val/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val/threshold", f1_dict["threshold"], sync_dist=True)
            
            self.val_preds_list.clear()
            self.val_targets_list.clear()
        
        if self.use_ema and self.ema_model is not None and len(self.val_ema_preds_list) > 0:
            val_ema_preds_all = torch.cat(self.val_ema_preds_list, dim=0).numpy()
            val_ema_targets_all = torch.cat(self.val_ema_targets_list, dim=0).numpy()
            
            ema_thresholds = None
            if self.use_class_thresholds:
                ema_module = self.ema_model.module
                if ema_module.class_thresholds is not None:
                    ema_thresholds = torch.sigmoid(ema_module.class_thresholds).detach().cpu().numpy()
            f1_dict_ema = eval_validation_set(val_ema_preds_all, val_ema_targets_all, class_thresholds=ema_thresholds)
            
            self.log("val_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("val_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            
            self.val_ema_preds_list.clear()
            self.val_ema_targets_list.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_consistency_loss_val = self.model_step(batch, use_ema_model=False)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.use_mcc and self.use_sparsity_loss:
            self.log("test/sparsity_loss", sparsity_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        if self.use_correlation_loss:
            self.log("test/correlation_loss", correlation_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        self.log("test/threshold_consistency_loss", threshold_consistency_loss_val, on_step=False, on_epoch=True, prog_bar=False)

        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())
        
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.test_ema_preds_list.append(ema_preds.detach().cpu())
            self.test_ema_targets_list.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        thresholds = None
        if self.use_class_thresholds and self.class_thresholds is not None:
            thresholds = torch.sigmoid(self.class_thresholds).detach().cpu().numpy()
        
        if len(self.test_preds_list) > 0:
            test_preds_all = torch.cat(self.test_preds_list, dim=0).numpy()
            test_targets_all = torch.cat(self.test_targets_list, dim=0).numpy()
            
            f1_dict = eval_validation_set(test_preds_all, test_targets_all, class_thresholds=thresholds)
            
            self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
            
            self.test_preds_list.clear()
            self.test_targets_list.clear()
        
        if self.use_ema and self.ema_model is not None and len(self.test_ema_preds_list) > 0:
            test_ema_preds_all = torch.cat(self.test_ema_preds_list, dim=0).numpy()
            test_ema_targets_all = torch.cat(self.test_ema_targets_list, dim=0).numpy()
            
            ema_thresholds = None
            if self.use_class_thresholds:
                ema_module = self.ema_model.module
                if ema_module.class_thresholds is not None:
                    ema_thresholds = torch.sigmoid(ema_module.class_thresholds).detach().cpu().numpy()
            f1_dict_ema = eval_validation_set(test_ema_preds_all, test_ema_targets_all, class_thresholds=ema_thresholds)
            
            self.log("test_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("test_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            
            self.test_ema_preds_list.clear()
            self.test_ema_targets_list.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit, validate, test, or predict."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        if hasattr(self.hparams.optimizer, 'keywords'):
            base_lr = self.hparams.optimizer.keywords.get('lr', 0.00001)
        else:
            base_lr = 0.00001
        
        vit_params = []
        dcm_params = []
        log_temperature_params = []
        
        if self.net is not None:
            vit_params.extend(self.net.parameters())
        
        if self.use_mcc and self.mcc_miner is not None:
            if hasattr(self.mcc_miner, 'verifier') and hasattr(self.mcc_miner.verifier, 'log_temperature'):
                log_temperature_params.append(self.mcc_miner.verifier.log_temperature)
                for name, param in self.mcc_miner.named_parameters():
                    if 'log_temperature' not in name:
                        dcm_params.append(param)
            else:
                dcm_params.extend(self.mcc_miner.parameters())
        
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
        
        if not param_groups:
            param_groups = [{'params': self.parameters()}]
        
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

