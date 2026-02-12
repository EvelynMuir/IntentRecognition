from typing import Any, Dict, Tuple
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.metrics import eval_validation_set
from src.utils.ema import ModelEma
from src.models.intentonomy_vit_mcc_module import (
    DiscriminativeClueMiner,
    sparsity_loss,
    sparsity_loss_smart,
    correlation_loss,
    threshold_consistency_loss
)


# def aggressive_gap_threshold_loss(pred_thresh, logits, soft_targets):
    # """
    # 让阈值学会钻缝隙：必须低于 Weak Positive (0.33)，但高于 Negative (0.1)。
    # """
    # probs = torch.sigmoid(logits).detach()  # 梯度只传给 threshold head
    # loss_batch = 0.0
    # valid_cnt = 0
    
    # for i in range(logits.shape[0]):
    #     p = probs[i]
    #     t = soft_targets[i]
        
    #     # 定义正负样本 (激进版：0.33 也是正)
    #     neg_scores = p[t < 0.1]
    #     pos_scores = p[t > 0.1]  # 包含 0.33, 0.66, 1.0
        
    #     if len(pos_scores) > 0:
    #         min_pos = pos_scores.min()
    #         if len(neg_scores) > 0:
    #             max_neg = neg_scores.max()
    #             # 理想情况：阈值在 max_neg 和 min_pos 中间
    #             # 如果混叠了 (max_neg > min_pos)，则强制阈值去逼近 min_pos - 0.05
    #             target = (max_neg + min_pos) / 2.0 if max_neg < min_pos else (min_pos - 0.05)
    #         else:
    #             target = min_pos - 0.05
    #     else:
    #         # 全负样本：阈值设高点，或者比最大负样本高
    #         target = neg_scores.max() + 0.1 if len(neg_scores) > 0 else torch.tensor(0.5).to(p.device)
            
    #     # 限制 target 范围，防止 tensor 越界
    #     target = target.clamp(0.01, 0.99)
    #     loss_batch += F.mse_loss(pred_thresh[i], target.unsqueeze(0))
    #     valid_cnt += 1
        
    # return loss_batch / (valid_cnt + 1e-6)

def aggressive_gap_threshold_loss(pred_thresh, logits, soft_targets):
    """
    【最终推荐版】SOTA 杀手：Aggressive Gap-Aware Threshold Loss
    针对 Micro F1 和 Samples F1 进行了特化优化。
    """
    # 梯度截断：只训练阈值头，不干扰 Backbone 的特征学习
    probs = torch.sigmoid(logits).detach()
    
    loss_batch = 0.0
    valid_cnt = 0
    device = pred_thresh.device
    
    for i in range(logits.shape[0]):
        p = probs[i]
        t = soft_targets[i]
        
        # === 1. 定义 SOTA 视角下的正负 ===
        # 激进定义：只要 soft_target > 0.1 (即包含 0.33) 全都算正样本
        pos_scores = p[t > 0.1] 
        # 只有 soft_target 接近 0 才是负样本
        neg_scores = p[t < 0.1]
        
        # 默认初始化
        target_t = torch.tensor(0.5).to(device)
        
        # === 2. 寻找缝隙 (Gap Finding) ===
        if len(pos_scores) > 0:
            # 找到最弱的正样本 (也就是我们需要捞回来的 0.33)
            min_pos = pos_scores.min()
            
            if len(neg_scores) > 0:
                max_neg = neg_scores.max()
                
                if max_neg < min_pos:
                    # 情况 A：分得开 -> 阈值设在中间 (最稳健)
                    target_t = (max_neg + min_pos) / 2.0
                else:
                    # 情况 B：分不开 (混叠) -> 优先保 Recall！
                    # 区别点：只退后 0.02，贴着正样本下沿切
                    target_t = min_pos - 0.02
            else:
                # 情况 C：只有正样本 -> 阈值设在最小正样本下面
                target_t = min_pos - 0.02
        else:
            # 情况 D：全负样本 -> 阈值设在最大负样本上面
            if len(neg_scores) > 0:
                target_t = neg_scores.max() + 0.1
            else:
                target_t = torch.tensor(0.5).to(device)
            
        # === 3. 激进截断 (The Clamp Trick) ===
        # 区别点：强制上限 0.6。
        # 即使全是负样本，也不让阈值飙太高，防止下一张图有弱信号时反应不过来。
        # 同时也防止阈值太低变成 0.0
        target_t = target_t.clamp(0.01, 0.6)
        
        loss_batch += F.mse_loss(pred_thresh[i], target_t.unsqueeze(0))
        valid_cnt += 1
        
    return loss_batch / (valid_cnt + 1e-6)


def clean_state_dict_for_loading(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理state_dict，移除torch.compile产生的_orig_mod前缀和EMA相关前缀。
    
    :param state_dict: 原始state_dict
    :return: 清理后的state_dict
    """
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        new_key = k
        
        # 移除 ema_model.module. 前缀（如果存在）
        if new_key.startswith("ema_model.module."):
            # new_key = new_key[len("ema_model.module."):]
            continue
        
        # 移除 net._orig_mod. 前缀（torch.compile产生）
        if new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod."):]
        
        new_state_dict[new_key] = v
    
    return new_state_dict


def soft_margin_boosting_loss(logits, soft_targets):
    """
    强迫 0.33 的预测值 > 0.5，强迫 0.1 的预测值 < 0.05
    """
    probs = torch.sigmoid(logits)
    
    # 选中所有的 Weak Positives (0.33)
    weak_pos_mask = (soft_targets > 0.3) & (soft_targets < 0.4)
    # 选中所有的 Negatives
    neg_mask = soft_targets < 0.1
    
    loss = 0.0
    
    # Boosting: 让 0.33 的预测值至少达到 0.5 (One-sided Loss)
    if weak_pos_mask.sum() > 0:
        preds = probs[weak_pos_mask]
        # 只惩罚小于 0.5 的部分
        loss += F.mse_loss(preds, torch.max(preds, torch.tensor(0.5).to(preds.device)))

    # Suppressing: 让 0.1 的预测值压到 0.05 以下
    if neg_mask.sum() > 0:
        preds = probs[neg_mask]
        # 只惩罚大于 0.05 的部分
        loss += F.mse_loss(preds, torch.min(preds, torch.tensor(0.05).to(preds.device)))
        
    return loss


class IntentonomyResNet101Module(LightningModule):
    """`LightningModule` for Intentonomy multi-label classification using ResNet101 with DiscriminativeClueMiner.

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
        input_dim: int = 2048,  # ResNet101 的输出通道数
        use_adaptive_temperature: bool = False,  # 是否使用自适应温度缩放
        use_global_feature: bool = False,  # 是否使用全局特征拼接
        use_learnable_fusion: bool = False,  # 是否使用可学习的融合权重
        use_image_level_temperature: bool = False,  # 是否使用 Image-Level 温度头
        use_cosine_similarity_temperature: bool = False,  # 是否使用基于余弦相似度的温度计算
        use_max_pooling_temperature: bool = False,  # 是否使用 Max Pooling 辅助温度计算（双通道一致性）
        use_consistency_bias: bool = False,  # 是否使用 consistency bias（默认关闭）
        # Label embedding 参数
        label_embedding_path: str = None,
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
        # Threshold loss 参数
        use_threshold_loss: bool = False,  # 默认关闭
        threshold_loss_weight: float = 1.0,
        # Margin boosting loss 参数
        use_margin_boosting_loss: bool = False,  # 默认关闭
        margin_boosting_loss_weight: float = 0.5,
        # Inference strategy 参数
        use_inference_strategy: bool = False,  # 是否使用 inference_strategy（默认关闭）
        # Class thresholds 参数
        use_class_thresholds: bool = False,  # 是否使用类别级可学习阈值（默认关闭）
        # EMA 参数
        use_ema: bool = True,
        ema_decay: float = 0.9997,
    ) -> None:
        """Initialize a `IntentonomyResNet101Module` with DiscriminativeClueMiner.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized with default params.
        :param use_mcc: Whether to use DiscriminativeClueMiner. If False, use standard forward.
        :param k_patches: Number of top-k patches to select in DiscriminativeClueMiner.
        :param gcn_layers: Number of GCN layers in DiscriminativeClueMiner.
        :param input_dim: Input dimension for DiscriminativeClueMiner (should match ResNet101 output channels, 2048).
        :param use_adaptive_temperature: Whether to use adaptive temperature scaling in LabelGuidedVerifier (default False).
        :param use_global_feature: Whether to concatenate global average pooled feature with top-K features (default False).
        :param use_learnable_fusion: Whether to use learnable fusion weight for combining local and global features (default False).
        :param use_image_level_temperature: Whether to use Image-Level temperature head for adaptive logits scaling (default False).
        :param use_cosine_similarity_temperature: Whether to use Global-Local Cosine Similarity for dynamic temperature calculation (default False).
        :param use_max_pooling_temperature: Whether to use Max Pooling to assist temperature calculation (dual-stream consistency, default False).
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
        :param use_threshold_loss: Whether to use threshold loss (default False).
        :param threshold_loss_weight: Weight for threshold loss (default 1.0).
        :param use_margin_boosting_loss: Whether to use margin boosting loss (default False).
        :param margin_boosting_loss_weight: Weight for margin boosting loss (default 0.5).
        :param use_inference_strategy: Whether to use inference_strategy in validation and test steps (default False).
                                      When enabled, uses adaptive threshold and fallback mechanism for predictions.
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
        self.use_threshold_loss = use_threshold_loss
        self.threshold_loss_weight = threshold_loss_weight
        self.use_margin_boosting_loss = use_margin_boosting_loss
        self.margin_boosting_loss_weight = margin_boosting_loss_weight
        self.use_inference_strategy = use_inference_strategy
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
                    # 使用默认路径
                    import os
                    base_dir = "/share/lmcp/tangyin/projects/IntentRecognition"
                    default_path = os.path.join(base_dir, "Intentonomy/data/label_embedding_300_28")
                    if not os.path.exists(default_path):
                        # 尝试另一个可能的路径
                        base_dir = "/home/evelynmuir/lambda/projects/IntentRecognition"
                        default_path = os.path.join(base_dir, "Intentonomy/data/label_embedding_300_28")
                    actual_embedding_path = default_path
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
                use_threshold_loss=use_threshold_loss,  # 传递阈值loss参数
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
        
        # for tracking threshold loss
        self.train_threshold_loss = MeanMetric()
        self.val_threshold_loss = MeanMetric()
        
        # for tracking margin boosting loss
        self.train_margin_boosting_loss = MeanMetric()
        self.val_margin_boosting_loss = MeanMetric()
        
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
        """从 ResNet101 backbone 中提取特征并转换为序列格式.
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Patch features of shape (batch_size, num_patches, feature_dim).
                 对于 224x224 输入，输出为 (batch_size, 49, 2048) (7x7=49).
        """
        # 访问 ResNet101 backbone
        backbone = self.net.backbone
        
        # ResNet101 特征提取
        # Input: [B, 3, 224, 224]
        feat_map = backbone(x) 
        # Output: [B, 2048, 7, 7] (假设输入 224x224，经过5次下采样，32倍)
        
        # 【关键步骤】格式转换 (Flatten)
        # 目标：变成 [B, N, D] 以适配 DCM
        B, C, H, W = feat_map.shape
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        feat_seq = feat_map.view(B, C, -1).permute(0, 2, 1) 
        # 现在 feat_seq 是 [B, 49, 2048] (对于 224x224 输入)
        
        return feat_seq

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
            # 从EMA模型的backbone提取特征
            backbone = ema_module.net.backbone
            feat_map = backbone(x)
            B, C, H, W = feat_map.shape
            feat_seq = feat_map.view(B, C, -1).permute(0, 2, 1)
            
            # 使用EMA模型的mcc_miner和label_embedding
            logits, selection_info = ema_module.mcc_miner(feat_seq, ema_module.label_embedding.weight)
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
        if self.use_threshold_loss:
            self.train_threshold_loss.reset()
            self.val_threshold_loss.reset()
        if self.use_margin_boosting_loss:
            self.train_margin_boosting_loss.reset()
            self.val_margin_boosting_loss.reset()
        self.train_threshold_consistency_loss.reset()
        self.val_threshold_consistency_loss.reset()
        
        # 初始化EMA模型
        if self.use_ema and self.ema_model is None:
            self.ema_model = ModelEma(self, decay=self.ema_decay)
            # 初始化EMA模型参数为当前模型参数
            self.ema_model.set(self)

    def model_step(
        self, batch: Dict[str, torch.Tensor], use_ema_model: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param use_ema_model: Whether to use EMA model for forward pass (default False).

        :return: A tuple containing (in order):
            - A tensor of total losses (classification + sparsity + correlation + threshold + margin_boosting + threshold_consistency).
            - A tensor of predictions (probabilities after sigmoid).
            - A tensor of target labels.
            - A tensor of sparsity loss values.
            - A tensor of correlation loss values.
            - A tensor of threshold loss values.
            - A tensor of margin boosting loss values.
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
            
            # 计算 threshold loss（如果启用）
            if self.use_threshold_loss:
                pred_thresh = selection_info.get("pred_thresh")  # [B, 1]
                if pred_thresh is not None:
                    threshold_loss_val = aggressive_gap_threshold_loss(pred_thresh, logits, y)
                else:
                    threshold_loss_val = torch.tensor(0.0, device=classification_loss.device)
            else:
                threshold_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算 margin boosting loss（如果启用）
            if self.use_margin_boosting_loss:
                margin_boosting_loss_val = soft_margin_boosting_loss(logits, y)
            else:
                margin_boosting_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算阈值一致性损失（如果启用）
            if self.use_class_thresholds and self.class_thresholds is not None:
                threshold_consistency_loss_val = threshold_consistency_loss(
                    logits, y, self.class_thresholds, margin=0.1
                )
            else:
                threshold_consistency_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 总 loss: Loss_Total = Loss_ASL + weight_sparsity * Loss_Sparsity + weight_corr * Loss_Corr + weight_thresh * Loss_Thresh + weight_margin * Loss_Margin + 0.1 * Loss_Threshold_Consistency
            loss = classification_loss + self.sparsity_loss_weight * sparsity_loss_val + self.correlation_loss_weight * correlation_loss_val + self.threshold_loss_weight * threshold_loss_val + self.margin_boosting_loss_weight * margin_boosting_loss_val + 0.1 * threshold_consistency_loss_val
        else:
            logits = forward_fn(x)
            classification_loss = self.criterion(logits, y)
            sparsity_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算 correlation loss（如果启用）
            if self.use_correlation_loss:
                correlation_loss_val = correlation_loss(logits, y)
            else:
                correlation_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算 threshold loss（如果启用，但在非MCC模式下无法获取pred_thresh，设为0）
            threshold_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算 margin boosting loss（如果启用）
            if self.use_margin_boosting_loss:
                margin_boosting_loss_val = soft_margin_boosting_loss(logits, y)
            else:
                margin_boosting_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 计算阈值一致性损失（如果启用）
            if self.use_class_thresholds and self.class_thresholds is not None:
                threshold_consistency_loss_val = threshold_consistency_loss(
                    logits, y, self.class_thresholds, margin=0.1
                )
            else:
                threshold_consistency_loss_val = torch.tensor(0.0, device=classification_loss.device)
            
            # 总 loss
            loss = classification_loss + self.correlation_loss_weight * correlation_loss_val + self.margin_boosting_loss_weight * margin_boosting_loss_val + 0.1 * threshold_consistency_loss_val
        
        preds = torch.sigmoid(logits)  # Convert logits to probabilities
        return loss, preds, y, sparsity_loss_val, correlation_loss_val, threshold_loss_val, margin_boosting_loss_val, threshold_consistency_loss_val

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_loss_val, margin_boosting_loss_val, threshold_consistency_loss_val = self.model_step(batch)

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
        
        # update and log threshold loss
        if self.use_threshold_loss:
            self.train_threshold_loss(threshold_loss_val)
            self.log("train/threshold_loss", self.train_threshold_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log margin boosting loss
        if self.use_margin_boosting_loss:
            self.train_margin_boosting_loss(margin_boosting_loss_val)
            self.log("train/margin_boosting_loss", self.train_margin_boosting_loss, on_step=False, on_epoch=True, prog_bar=False)

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
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_loss_val, margin_boosting_loss_val, threshold_consistency_loss_val = self.model_step(batch, use_ema_model=False)

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
        
        # update and log threshold loss
        if self.use_threshold_loss:
            self.val_threshold_loss(threshold_loss_val)
            self.log("val/threshold_loss", self.val_threshold_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log margin boosting loss
        if self.use_margin_boosting_loss:
            self.val_margin_boosting_loss(margin_boosting_loss_val)
            self.log("val/margin_boosting_loss", self.val_margin_boosting_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        # update and log threshold consistency loss
        self.val_threshold_consistency_loss(threshold_consistency_loss_val)
        self.log("val/threshold_consistency_loss", self.val_threshold_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)

        # 收集预测和标签用于 HLEG 计算方式（原始模型）
        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())
        
        # 如果使用EMA，也使用EMA模型进行推理
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _, _, _, _ = self.model_step(batch, use_ema_model=True)
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
            f1_dict = eval_validation_set(val_preds_all, val_targets_all, use_inference_strategy=self.use_inference_strategy, class_thresholds=thresholds)
            
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
            f1_dict_ema = eval_validation_set(val_ema_preds_all, val_ema_targets_all, use_inference_strategy=self.use_inference_strategy, class_thresholds=ema_thresholds)
            
            # 记录EMA模型的metrics
            self.log("val_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("val_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            
            # 清空EMA列表
            self.val_ema_preds_list.clear()
            self.val_ema_targets_list.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # 使用原始模型
        loss, preds, targets, sparsity_loss_val, correlation_loss_val, threshold_loss_val, margin_boosting_loss_val, threshold_consistency_loss_val = self.model_step(batch, use_ema_model=False)

        # update and log loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # log sparsity loss (test阶段通常不需要累积)
        if self.use_mcc and self.use_sparsity_loss:
            self.log("test/sparsity_loss", sparsity_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        # log correlation loss (test阶段通常不需要累积)
        if self.use_correlation_loss:
            self.log("test/correlation_loss", correlation_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        # log threshold loss (test阶段通常不需要累积)
        if self.use_threshold_loss:
            self.log("test/threshold_loss", threshold_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        # log margin boosting loss (test阶段通常不需要累积)
        if self.use_margin_boosting_loss:
            self.log("test/margin_boosting_loss", margin_boosting_loss_val, on_step=False, on_epoch=True, prog_bar=False)
        
        # log threshold consistency loss (test阶段通常不需要累积)
        self.log("test/threshold_consistency_loss", threshold_consistency_loss_val, on_step=False, on_epoch=True, prog_bar=False)

        # 收集预测和标签用于 HLEG 计算方式（原始模型）
        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())
        
        # 如果使用EMA，也使用EMA模型进行推理
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _, _, _, _ = self.model_step(batch, use_ema_model=True)
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
            
            # 使用 HLEG 的计算方式，传递类别级阈值（如果启用）
            f1_dict = eval_validation_set(test_preds_all, test_targets_all, use_inference_strategy=self.use_inference_strategy, class_thresholds=thresholds)
            
            # 记录 metrics
            self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
            
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
            f1_dict_ema = eval_validation_set(test_ema_preds_all, test_ema_targets_all, use_inference_strategy=self.use_inference_strategy, class_thresholds=ema_thresholds)
            
            # 记录EMA模型的metrics
            self.log("test_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("test_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            
            # 清空EMA列表
            self.test_ema_preds_list.clear()
            self.test_ema_targets_list.clear()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Lightning hook called when loading a checkpoint.
        
        在加载checkpoint之前清理state_dict，移除torch.compile产生的_orig_mod前缀。
        
        :param checkpoint: The checkpoint dictionary.
        """
        if "state_dict" in checkpoint:
            original_state_dict = checkpoint["state_dict"]
            # print(original_state_dict.keys())
            cleaned_state_dict = clean_state_dict_for_loading(original_state_dict)
            # print(cleaned_state_dict.keys())
            checkpoint["state_dict"] = cleaned_state_dict

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
            base_lr = self.hparams.optimizer.keywords.get('lr', 0.0001)
        else:
            # 如果没有 keywords 属性，使用默认值
            base_lr = 0.0001
        
        # 分离 ResNet101 和 DCM 的参数
        backbone_params = []
        dcm_params = []
        
        # ResNet101 backbone 的参数
        if self.net is not None:
            backbone_params.extend(self.net.parameters())
        
        # DCM 的参数（如果使用）
        if self.use_mcc and self.mcc_miner is not None:
            dcm_params.extend(self.mcc_miner.parameters())
        
        # 创建参数组：ResNet101 使用基础学习率，DCM 使用 10 倍学习率
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
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

