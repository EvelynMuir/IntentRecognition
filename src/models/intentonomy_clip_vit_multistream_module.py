"""
CLIP-ViT Multi-Stream Semantic Spatial VQ Module

使用5个语义维度的空间向量量化器（COCO, Places365, Emotion, AVA, Stanford 40 Actions）
对CLIP ViT的patch tokens进行语义量化，然后融合进行分类。

更新：整合codebook模块的新功能（factor attention、anchor loss等），
同时保留multistream中用anchor初始化VQ的核心功能。
"""
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.intentonomy_clip_vit_base_module import IntentonomyClipViTBaseModule
from src.models.components.aslloss import AsymmetricLossOptimized
from src.models.components.semantic_spatial_vq import SemanticSpatialVQ
from src.models.components.clip_text_anchors import generate_text_anchors
from src.models.components.context_gated_fusion import ContextGatedFusion
from src.utils.metrics import eval_validation_set
from src.utils.ema import ModelEma
import clip


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
            continue
        
        # 移除 net._orig_mod. 前缀（torch.compile产生）
        if new_key.startswith("net._orig_mod."):
            new_key = "net." + new_key[len("net._orig_mod."):]
        
        new_state_dict[new_key] = v
    
    return new_state_dict


class IntentonomyClipViTMultiStreamModule(IntentonomyClipViTBaseModule):
    """`LightningModule` for Intentonomy multi-label classification using CLIP Vision Transformer with Multi-Stream Semantic Spatial VQ.
    
    Model architecture:
    1. CLIP ViT encoder extracts all patch tokens (no global pooling) [B, N_patches, vit_projected_dim]
    2. Factor attention机制：降维bottleneck + factor attention权重
    3. 5个SemanticSpatialVQ并行处理：COCO, Places, Emotion, AVA, Actions（使用anchor初始化codebook）
    4. Sum融合5个VQ输出（保留空间对应关系）
    5. Global pooling (mean) 得到全局特征
    6. Classification head输出28类logits
    
    更新：整合codebook模块的新功能：
    - Factor attention机制
    - Factor separation loss
    - 文本anchor生成（可选，保留从文件加载anchor的方式）
    - Checkpoint和mapping加载功能
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        # Semantic anchors
        semantic_anchors_path: str = "semantic_anchors.pth",
        use_text_anchors: bool = False,  # 是否使用文本anchor（如果True，忽略semantic_anchors_path）
        # VQ parameters
        vq_commitment_cost: float = 0.25,
        freeze_codebook: bool = True,  # 是否冻结codebook（默认True）
        # Semantic consistency loss
        use_semantic_consistency_loss: bool = False,
        semantic_consistency_weight: float = 0.1,
        # Factor separation loss
        use_factor_separation_loss: bool = False,
        factor_separation_loss_weight: float = 0.1,
        # EMA parameters
        use_ema: bool = True,
        ema_decay: float = 0.9997,
        # Backbone freezing
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,  # Number of last transformer blocks to unfreeze
        lr_vit: float = 1e-5,  # Learning rate for unfrozen ViT blocks
        # Optimizer learning rates (可分别为5个VQ设置)
        lr_pre_proj: float = 1e-3,  # Learning rate for pre_proj
        lr_factor_attention: float = 1e-3,  # Learning rate for factor_attention
        lr_vq_coco: float = 1e-3,
        lr_vq_places: float = 1e-3,
        lr_vq_emotion: float = 1e-3,
        lr_vq_ava: float = 1e-3,
        lr_vq_actions: float = 1e-3,
        lr_classifier: float = 3e-4,
        # Weight decay
        wd_pre_proj: float = 1e-6,
        wd_factor_attention: float = 1e-6,
        wd_vq: float = 0.0,
        wd_head: float = 1e-4,
        # Intent loss weight scheduling
        intent_loss_weight_warmup_epochs: int = 2,
        intent_loss_weight_warmup: float = 0.2,
        intent_loss_weight_normal: float = 1.0,
        # Pretrained checkpoint
        pretrained_ckpt_path: str = None,
        mapping_path: str = None,
        # Gated Fusion
        use_gated_fusion: bool = False,
        gated_fusion_hidden_dim: int = 1024,
        lr_gated_fusion: float = 1e-3,
        wd_gated_fusion: float = 1e-6,
        # Feature extraction mode
        token_mode: str = "patch",  # "cls", "patch", or "mixed". 
                                    # "cls": all factors use CLS token
                                    # "patch": all factors use patch tokens
                                    # "mixed": places and ava use CLS token, others use patch tokens
    ) -> None:
        """Initialize a `IntentonomyClipViTMultiStreamModule`.
        
        :param net: The model to train (should be ClipVisionTransformer).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized with default params.
        :param semantic_anchors_path: Path to semantic anchors file (used if use_text_anchors=False).
        :param use_text_anchors: Whether to use text anchors generated from CLIP (default False).
        :param vq_commitment_cost: Weight for VQ commitment loss.
        :param freeze_codebook: Whether to freeze codebook (default True).
        :param use_semantic_consistency_loss: Whether to use semantic consistency loss.
        :param semantic_consistency_weight: Weight for semantic consistency loss.
        :param use_factor_separation_loss: Whether to use factor separation loss.
        :param factor_separation_loss_weight: Weight for factor separation loss.
        :param use_ema: Whether to use Exponential Moving Average for model parameters.
        :param ema_decay: Decay factor for EMA.
        :param freeze_backbone: Whether to freeze CLIP ViT backbone parameters.
        :param unfreeze_last_n_blocks: Number of last transformer blocks to unfreeze.
        :param lr_vit: Learning rate for unfrozen ViT blocks.
        :param lr_pre_proj: Learning rate for pre_proj.
        :param lr_factor_attention: Learning rate for factor_attention.
        :param lr_vq_*: Learning rates for each VQ module.
        :param lr_classifier: Learning rate for classifier (head).
        :param wd_pre_proj: Weight decay for pre_proj.
        :param wd_factor_attention: Weight decay for factor_attention.
        :param wd_vq: Weight decay for vector quantizers.
        :param wd_head: Weight decay for classifier (head).
        :param intent_loss_weight_warmup_epochs: Number of epochs with reduced intent loss weight.
        :param intent_loss_weight_warmup: Intent loss weight during warmup epochs.
        :param intent_loss_weight_normal: Intent loss weight after warmup.
        :param pretrained_ckpt_path: Path to pretrained checkpoint to load.
        :param mapping_path: Path to mapping file (from compute_mapping.py) to load.
        :param token_mode: Token extraction mode. "cls" for all CLS tokens, "patch" for all patch tokens, 
                           "mixed" for places/ava using CLS token and others using patch tokens.
        """
        # Validate token_mode
        if token_mode not in ["cls", "patch", "mixed"]:
            raise ValueError(f"token_mode must be one of ['cls', 'patch', 'mixed'], got {token_mode}")
        
        # Initialize base class
        # For base class, use_cls_token=True only for "cls" mode, False otherwise
        # (base class doesn't support mixed mode, so we handle it in this class)
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=num_classes,
            compile=compile,
            criterion=criterion,
            use_ema=use_ema,
            ema_decay=ema_decay,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            lr_vit=lr_vit,
            intent_loss_weight_warmup_epochs=intent_loss_weight_warmup_epochs,
            intent_loss_weight_warmup=intent_loss_weight_warmup,
            intent_loss_weight_normal=intent_loss_weight_normal,
            use_cls_token=(token_mode == "cls"),  # Only True for "cls" mode
        )
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.semantic_anchors_path = semantic_anchors_path
        self.use_text_anchors = use_text_anchors
        self.vq_commitment_cost = vq_commitment_cost
        self.freeze_codebook = freeze_codebook
        self.use_semantic_consistency_loss = use_semantic_consistency_loss
        self.semantic_consistency_weight = semantic_consistency_weight
        self.use_factor_separation_loss = use_factor_separation_loss
        self.factor_separation_loss_weight = factor_separation_loss_weight
        self.lr_pre_proj = lr_pre_proj
        self.lr_factor_attention = lr_factor_attention
        self.lr_vq_coco = lr_vq_coco
        self.lr_vq_places = lr_vq_places
        self.lr_vq_emotion = lr_vq_emotion
        self.lr_vq_ava = lr_vq_ava
        self.lr_vq_actions = lr_vq_actions
        self.lr_classifier = lr_classifier
        self.wd_pre_proj = wd_pre_proj
        self.wd_factor_attention = wd_factor_attention
        self.wd_vq = wd_vq
        self.wd_head = wd_head
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.mapping_path = mapping_path
        self.use_gated_fusion = use_gated_fusion
        self.lr_gated_fusion = lr_gated_fusion
        self.wd_gated_fusion = wd_gated_fusion
        self.token_mode = token_mode
        
        # Get CLIP ViT dimensions
        vit_hidden_dim = self._get_vit_hidden_dim()
        vit_projected_dim = self._get_vit_projected_dim()  # Get projected dimension (768)
        
        # Factor attention机制：降维bottleneck + factor attention权重
        self.pre_proj = nn.Sequential(
            nn.Linear(vit_projected_dim, 256),
            nn.GELU()
        )
        
        # Factor attention权重: [5, 256]
        self.factor_attention = nn.Parameter(torch.randn(5, 256))
        nn.init.xavier_uniform_(self.factor_attention)
        
        # 温度参数
        self.tau = 0.1
        
        # Factor heads: 5个独立的MLP用于拆解factor特征
        self.factor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vit_projected_dim, vit_projected_dim),  # 768 -> 768
                nn.LayerNorm(vit_projected_dim),
                nn.GELU(),
                nn.Linear(vit_projected_dim, vit_projected_dim)  # 768 -> 768，输出给VQ
            ) for _ in range(5)  # 5个factors
        ])
        
        # Adapters: 5个独立的适配器，用于将patch特征映射到Codebook空间
        # 每个适配器负责把视觉特征映射到对应的Codebook空间
        feature_dim = vit_projected_dim  # 768
        vq_dim = vit_projected_dim  # 768
        num_factors = 5
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),  # 先做一次混合
                nn.LayerNorm(feature_dim),            # 归一化很重要
                nn.GELU(),
                nn.Linear(feature_dim, vq_dim)        # 映射到 VQ 维度
            ) for _ in range(num_factors)
        ])
        
        # Load semantic anchors (从文件或生成文本anchors)
        if use_text_anchors:
            anchors = self._generate_text_anchors()
        else:
            anchors = self._load_semantic_anchors()
        
        # 创建5个SemanticSpatialVQ (使用投影后的维度768)
        # 核心：使用anchor初始化VQ的codebook（保留multistream的核心功能）
        self.vq_coco = SemanticSpatialVQ(
            anchor_embeddings=anchors["coco"],
            embedding_dim=vit_projected_dim,  # 768 after projection
            commitment_cost=vq_commitment_cost,
            freeze_codebook=freeze_codebook
        )
        self.vq_places = SemanticSpatialVQ(
            anchor_embeddings=anchors["places"],
            embedding_dim=vit_projected_dim,  # 768 after projection
            commitment_cost=vq_commitment_cost,
            freeze_codebook=freeze_codebook
        )
        self.vq_emotion = SemanticSpatialVQ(
            anchor_embeddings=anchors["emotion"],
            embedding_dim=vit_projected_dim,  # 768 after projection
            commitment_cost=vq_commitment_cost,
            freeze_codebook=freeze_codebook
        )
        self.vq_ava = SemanticSpatialVQ(
            anchor_embeddings=anchors["ava"],
            embedding_dim=vit_projected_dim,  # 768 after projection
            commitment_cost=vq_commitment_cost,
            freeze_codebook=freeze_codebook
        )
        self.vq_actions = SemanticSpatialVQ(
            anchor_embeddings=anchors["actions"],
            embedding_dim=vit_projected_dim,  # 768 after projection
            commitment_cost=vq_commitment_cost,
            freeze_codebook=freeze_codebook
        )
        
        # VQ列表，方便循环调用
        self.vqs = [self.vq_coco, self.vq_places, self.vq_emotion, self.vq_ava, self.vq_actions]
        
        self.classifier = nn.Sequential(
            nn.Linear(vit_projected_dim, 512),  # 768 -> 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
        
        # Gated Fusion
        if use_gated_fusion:
            self.gated_fusion = ContextGatedFusion(
                num_factors=5,
                input_dim=vit_projected_dim,
                hidden_dim=gated_fusion_hidden_dim
            )
        else:
            self.register_module("gated_fusion", None)
        
        # Loss function and metrics are initialized in base class
    
    # _get_vit_projected_dim is inherited from base class
    
    
    def _load_semantic_anchors(self) -> Dict[str, torch.Tensor]:
        """Load semantic anchors from file."""
        anchors = torch.load(self.semantic_anchors_path, map_location="cpu")
        print(f"Loaded semantic anchors from {self.semantic_anchors_path}")
        for factor, emb in anchors.items():
            print(f"  {factor}: shape {emb.shape}")
        return anchors
    
    def _generate_text_anchors(self) -> Dict[str, torch.Tensor]:
        """生成5个数据集的CLIP文本anchor并返回字典格式（兼容_load_semantic_anchors的返回格式）。
        
        返回的anchors格式：每个factor对应一个tensor，shape为[num_codes, embedding_dim]或[embedding_dim]
        """
        try:
            # 访问CLIP模型
            clip_model = self.net.clip_model
            clip_tokenize = clip.tokenize
            
            # 使用CPU设备（初始化时模型可能在CPU上）
            device = "cpu"
            if hasattr(clip_model, 'device'):
                device = clip_model.device
            elif next(clip_model.parameters()).is_cuda:
                device = "cuda"
            
            # 生成anchors
            anchors, anchor_names = generate_text_anchors(
                clip_model=clip_model,
                clip_tokenize=clip_tokenize,
                device=device
            )
            
            # 存储原始文本anchors为buffer
            self.register_buffer('text_anchors', anchors)
            print(f"Generated {len(anchor_names)} text anchors: {anchor_names}")
            print(f"Text anchors shape: {anchors.shape}")
            
            # 将anchors转换为字典格式（兼容从文件加载的格式）
            # 每个anchor是一个[embedding_dim]的tensor，需要扩展为[num_codes, embedding_dim]
            # 为了兼容SemanticSpatialVQ，我们使用单个anchor作为codebook的初始化
            # 但SemanticSpatialVQ期望[num_codes, embedding_dim]，所以我们创建一个包含单个code的codebook
            vit_projected_dim = self._get_vit_projected_dim()
            clip_embedding_dim = anchors.shape[1]
            
            def pad_to_vit_dim(x, target_dim):
                """Pad or truncate x to target_dim"""
                if x.shape[0] >= target_dim:
                    return x[:target_dim]
                pad = torch.zeros(target_dim - x.shape[0], device=x.device)
                return torch.cat([x, pad], dim=0)
            
            # 为每个factor创建anchor（单个code，shape为[1, vit_projected_dim]）
            anchors_dict = {}
            factor_names = ["coco", "places", "emotion", "ava", "actions"]
            for i, factor_name in enumerate(factor_names):
                anchor = pad_to_vit_dim(anchors[i], vit_projected_dim)
                # SemanticSpatialVQ期望[num_codes, embedding_dim]，所以unsqueeze
                anchors_dict[factor_name] = anchor.unsqueeze(0)  # [1, vit_projected_dim]
            
            return anchors_dict
        except Exception as e:
            print(f"Warning: Failed to generate text anchors: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，回退到从文件加载
            print(f"Falling back to loading anchors from file: {self.semantic_anchors_path}")
            return self._load_semantic_anchors()
    
    # _extract_patch_tokens is inherited from base class
    
    def forward(self, x: torch.Tensor, return_vq_info: bool = False) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :param return_vq_info: Whether to return VQ loss, perplexity, and quantized features.
        :return: A tensor of logits of shape (batch_size, num_classes), or 
                 (logits, vq_loss, perplexity, quantized_dict, gates) if return_vq_info=True.
        
        Token mode options:
        - "cls": All factors use CLS token
        - "patch": All factors use patch tokens
        - "mixed": places (index 1) and ava (index 3) use CLS token, 
                   coco (index 0), emotion (index 2), actions (index 4) use patch tokens
        """
        # Factor indices: 0=coco, 1=places, 2=emotion, 3=ava, 4=actions
        
        if self.token_mode == "cls":
            # Mode 1: All factors use CLS token
            features = self._extract_patch_tokens(x)  # [B, 1, vit_projected_dim=768]
            B, _, vit_projected_dim = features.shape
            global_semantic = features.squeeze(1)  # [B, 1, D] -> [B, D]
            factor_features = []
            for i in range(5):
                factor_feat = self.factor_heads[i](global_semantic)  # [B, D]
                factor_features.append(factor_feat)
            z_f = torch.stack(factor_features, dim=1)  # [B, 5, D]
            
        elif self.token_mode == "patch":
            # Mode 2: All factors use patch tokens with factor attention
            patch_tokens = self._extract_patch_tokens(x)  # [B, N_patches, vit_projected_dim=768]
            B, N_patches, vit_projected_dim = patch_tokens.shape
            
            # Factor attention机制
            h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
            
            # Factor attention (温度缩放sigmoid)
            A = torch.sigmoid(h @ self.factor_attention.T / self.tau)  # [B, N_patches, 5]
            
            # Normalize attention weights (dim=1: 跨patch求和为1)
            weights = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 5]
            
            # Aggregate (使用einsum)
            # patch_tokens: [B, N_patches, D], weights: [B, N_patches, 5]
            # z_f: [B, 5, D]
            z_f = torch.einsum('bpd,bpf->bfd', patch_tokens, weights)
            
        elif self.token_mode == "mixed":
            # Mode 3: Mixed mode - places and ava use CLS token, others use patch tokens
            # Extract both CLS token and patch tokens from CLIP ViT
            features_dict = self._extract_all_features(x)
            cls_token = features_dict["cls_token"]  # [B, 1, vit_projected_dim=768]
            patch_tokens = features_dict["patch_tokens"]  # [B, N_patches, vit_projected_dim=768]
            B, N_patches, vit_projected_dim = patch_tokens.shape
            
            z_f = torch.zeros(B, 5, vit_projected_dim, device=x.device, dtype=patch_tokens.dtype)
            
            # Process CLS token factors (places=1, ava=3)
            cls_semantic = cls_token.squeeze(1)  # [B, 1, D] -> [B, D]
            z_f[:, 1, :] = self.factor_heads[1](cls_semantic)  # places
            z_f[:, 3, :] = self.factor_heads[3](cls_semantic)  # ava
            
            # Process patch token factors (coco=0, emotion=2, actions=4) with factor attention
            h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
            
            # Factor attention for patch token factors: coco (0), emotion (2), actions (4)
            patch_factor_indices = [0, 2, 4]  # coco, emotion, actions
            patch_factor_attention = self.factor_attention[patch_factor_indices, :]  # [3, 256]
            
            A_patch = torch.sigmoid(h @ patch_factor_attention.T / self.tau)  # [B, N_patches, 3]
            weights_patch = A_patch / (A_patch.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 3]
            
            # Aggregate for each patch token factor
            z_patch_factors = torch.einsum('bpd,bpf->bfd', patch_tokens, weights_patch)  # [B, 3, D]
            
            # Assign to z_f: coco=0, emotion=2, actions=4
            z_f[:, 0, :] = z_patch_factors[:, 0, :]  # coco
            z_f[:, 2, :] = z_patch_factors[:, 1, :]  # emotion
            z_f[:, 4, :] = z_patch_factors[:, 2, :]  # actions
        else:
            raise ValueError(f"Unknown token_mode: {self.token_mode}")
        
        # 保存原始特征
        z_f_raw = z_f
        
        # 6. Parallel VQ: 5个VQ分别处理对应的factor
        # 根据token_mode决定是否使用适配器：
        # - cls模式: 所有因子都不使用适配器
        # - patch模式: 所有因子都使用适配器
        # - mixed模式: 因子0, 2, 4 (coco, emotion, actions) 使用适配器，因子1, 3 (places, ava) 不使用适配器
        quantized_list = []
        vq_losses = []
        perplexities = []
        
        for i in range(5):
            # 判断是否需要使用适配器
            use_adapter = False
            if self.token_mode == "patch":
                # patch模式：所有因子都使用适配器
                use_adapter = True
            elif self.token_mode == "mixed":
                # mixed模式：仅patch因子使用适配器 (0, 2, 4)
                if i in [0, 2, 4]:  # coco, emotion, actions
                    use_adapter = True
            # cls模式：所有因子都不使用适配器，use_adapter保持False
            
            # 取出第 i 个因子的原始特征
            raw_feat = z_f_raw[:, i, :]  # [B, 768]
            
            if use_adapter:
                # 通过适配器：变成"潜在的语义向量"
                adapted_feat = self.adapters[i](raw_feat)  # [B, vq_dim=768]
                # 输入冻结的 VQ，此时 adapted_feat 会努力通过训练去靠近 Codebook
                vq_input = adapted_feat.unsqueeze(1)  # [B, 1, 768]
            else:
                # 直接使用原始特征，不经过适配器
                vq_input = raw_feat.unsqueeze(1)  # [B, 1, 768]
            
            # 输入VQ
            z_q, vq_loss, perplexity = self.vqs[i](vq_input)
            quantized_list.append(z_q)
            vq_losses.append(vq_loss)
            perplexities.append(perplexity)
        
        # 解包结果
        z_coco, z_places, z_emotion, z_ava, z_actions = quantized_list
        vq_loss_coco, vq_loss_places, vq_loss_emotion, vq_loss_ava, vq_loss_actions = vq_losses
        perplexity_coco, perplexity_places, perplexity_emotion, perplexity_ava, perplexity_actions = perplexities
        
        # 5. Fusion: 融合5个VQ输出
        gates = None
        if self.use_gated_fusion:
            # Step A: Pool factors individually
            # z_*: [B, 1, 768] (CLS mode) or [B, N_patches, 768] (patch mode)
            z_factors = [
                z_coco.mean(dim=1),  # [B, 768]
                z_places.mean(dim=1),
                z_emotion.mean(dim=1),
                z_ava.mean(dim=1),
                z_actions.mean(dim=1)
            ]
            # Stack factors: [B, 5, 768]
            z_stacked = torch.stack(z_factors, dim=1)
            # Gated Fusion: [B, 5, 768] -> [B, 768]
            z_global, gates = self.gated_fusion(z_stacked)
        else:
            # Default Sum Fusion: 保留空间对应关系再池化
            # z_*: [B, 1, 768] (CLS mode) or [B, N_patches, 768] (patch mode)
            z_fused = z_coco + z_places + z_emotion + z_ava + z_actions  # [B, 1, 768] or [B, N_patches, 768]
            # Global Pooling (mean)
            z_global = z_fused.mean(dim=1)  # [B, 768]
        
        # 7. Classification
        logits = self.classifier(z_global)  # [B, num_classes]
        
        if return_vq_info:
            # Sum all VQ losses
            vq_loss = vq_loss_coco + vq_loss_places + vq_loss_emotion + vq_loss_ava + vq_loss_actions
            # Average perplexity
            perplexity = (perplexity_coco + perplexity_places + perplexity_emotion + 
                         perplexity_ava + perplexity_actions) / 5.0
            # Quantized features dict
            quantized_dict = {
                "coco": z_coco,
                "places": z_places,
                "emotion": z_emotion,
                "ava": z_ava,
                "actions": z_actions
            }
            return logits, vq_loss, perplexity, quantized_dict, gates
        
        return logits
    
    def get_code_indices(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get code indices for each factor for each image.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :return: A dictionary of code indices for each factor, each of shape [B, 1].
        
        Token mode options (same as forward method):
        - "cls": All factors use CLS token
        - "patch": All factors use patch tokens
        - "mixed": places (index 1) and ava (index 3) use CLS token, 
                   coco (index 0), emotion (index 2), actions (index 4) use patch tokens
        """
        # Factor indices: 0=coco, 1=places, 2=emotion, 3=ava, 4=actions
        
        if self.token_mode == "cls":
            # Mode 1: All factors use CLS token
            features = self._extract_patch_tokens(x)  # [B, 1, vit_projected_dim=768]
            B, _, vit_projected_dim = features.shape
            global_semantic = features.squeeze(1)  # [B, 1, D] -> [B, D]
            factor_features = []
            for i in range(5):
                factor_feat = self.factor_heads[i](global_semantic)  # [B, D]
                factor_features.append(factor_feat)
            z_f = torch.stack(factor_features, dim=1)  # [B, 5, D]
            
        elif self.token_mode == "patch":
            # Mode 2: All factors use patch tokens with factor attention
            patch_tokens = self._extract_patch_tokens(x)  # [B, N_patches, vit_projected_dim=768]
            B, N_patches, vit_projected_dim = patch_tokens.shape
            
            # Factor attention机制
            h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
            
            # Factor attention (温度缩放sigmoid)
            A = torch.sigmoid(h @ self.factor_attention.T / self.tau)  # [B, N_patches, 5]
            weights = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 5]
            
            # Aggregate (使用einsum)
            z_f = torch.einsum('bpd,bpf->bfd', patch_tokens, weights)  # [B, 5, D]
            
        elif self.token_mode == "mixed":
            # Mode 3: Mixed mode - places and ava use CLS token, others use patch tokens
            # Extract both CLS token and patch tokens from CLIP ViT
            features_dict = self._extract_all_features(x)
            cls_token = features_dict["cls_token"]  # [B, 1, vit_projected_dim=768]
            patch_tokens = features_dict["patch_tokens"]  # [B, N_patches, vit_projected_dim=768]
            B, N_patches, vit_projected_dim = patch_tokens.shape
            
            z_f = torch.zeros(B, 5, vit_projected_dim, device=x.device, dtype=patch_tokens.dtype)
            
            # Process CLS token factors (places=1, ava=3)
            cls_semantic = cls_token.squeeze(1)  # [B, 1, D] -> [B, D]
            z_f[:, 1, :] = self.factor_heads[1](cls_semantic)  # places
            z_f[:, 3, :] = self.factor_heads[3](cls_semantic)  # ava
            
            # Process patch token factors (coco=0, emotion=2, actions=4) with factor attention
            h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
            
            # Factor attention for patch token factors: coco (0), emotion (2), actions (4)
            patch_factor_indices = [0, 2, 4]  # coco, emotion, actions
            patch_factor_attention = self.factor_attention[patch_factor_indices, :]  # [3, 256]
            
            A_patch = torch.sigmoid(h @ patch_factor_attention.T / self.tau)  # [B, N_patches, 3]
            weights_patch = A_patch / (A_patch.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 3]
            
            # Aggregate for each patch token factor
            z_patch_factors = torch.einsum('bpd,bpf->bfd', patch_tokens, weights_patch)  # [B, 3, D]
            
            # Assign to z_f: coco=0, emotion=2, actions=4
            z_f[:, 0, :] = z_patch_factors[:, 0, :]  # coco
            z_f[:, 2, :] = z_patch_factors[:, 1, :]  # emotion
            z_f[:, 4, :] = z_patch_factors[:, 2, :]  # actions
        else:
            raise ValueError(f"Unknown token_mode: {self.token_mode}")
        
        # 保存原始特征
        z_f_raw = z_f
        
        # 根据token_mode决定是否使用适配器（与forward方法保持一致）
        indices = {}
        factor_names = ["coco", "places", "emotion", "ava", "actions"]
        
        for i, factor_name in enumerate(factor_names):
            # 判断是否需要使用适配器
            use_adapter = False
            if self.token_mode == "patch":
                # patch模式：所有因子都使用适配器
                use_adapter = True
            elif self.token_mode == "mixed":
                # mixed模式：仅patch因子使用适配器 (0, 2, 4)
                if i in [0, 2, 4]:  # coco, emotion, actions
                    use_adapter = True
            # cls模式：所有因子都不使用适配器，use_adapter保持False
            
            # 取出第 i 个因子的原始特征
            raw_feat = z_f_raw[:, i, :]  # [B, 768]
            
            if use_adapter:
                # 通过适配器：变成"潜在的语义向量"
                adapted_feat = self.adapters[i](raw_feat)  # [B, vq_dim=768]
                # 输入VQ
                vq_input = adapted_feat.unsqueeze(1)  # [B, 1, 768]
            else:
                # 直接使用原始特征，不经过适配器
                vq_input = raw_feat.unsqueeze(1)  # [B, 1, 768]
            
            # 获取code indices
            indices[factor_name] = self.vqs[i].get_code_indices(vq_input)
        
        return indices
    
    def factor_separation_loss(self, quantized_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute factor separation loss to encourage diversity among factors.
        
        :param quantized_dict: Dictionary of quantized factor vectors, each of shape [B, N_patches, D].
        :return: Factor separation loss scalar.
        """
        # Extract quantized features for each factor
        z_list = [
            quantized_dict["coco"].mean(dim=1),  # [B, D] - global average pooling
            quantized_dict["places"].mean(dim=1),
            quantized_dict["emotion"].mean(dim=1),
            quantized_dict["ava"].mean(dim=1),
            quantized_dict["actions"].mean(dim=1),
        ]
        z = torch.stack(z_list, dim=1)  # [B, 5, D]
        
        # Normalize
        z = F.normalize(z, dim=-1)  # [B, 5, D]
        K = z.shape[1]  # Number of factors (5)
        eye = torch.eye(K, device=z.device).unsqueeze(0)  # [1, 5, 5]
        sim = z @ z.transpose(-2, -1)  # [B, 5, 5]
        sim = sim * (1 - eye)  # Mask out diagonal elements
        
        loss = (sim ** 2).mean()
        return loss
    
    def model_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        use_ema_model: bool = False, 
        intent_loss_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        
        :param batch: A batch of data containing the input tensor of images and target labels.
        :param use_ema_model: Whether to use EMA model for forward pass.
        :param intent_loss_weight: Weight for intent (classification) loss.
        
        :return: A tuple containing (loss, preds, targets, vq_loss, perplexity).
        """
        x = batch["image"]
        y = batch["labels"]
        
        # Forward pass
        if use_ema_model:
            if self.ema_model is None:
                raise RuntimeError("EMA model is not initialized. Call on_train_start first.")
            ema_module = self.ema_model.module
            logits, vq_loss, perplexity, quantized_dict, gates = \
                ema_module.forward(x, return_vq_info=True)
            # logits = ema_module(x)
            # vq_loss = torch.tensor(0.0, device=logits.device)
            # perplexity = torch.tensor(0.0, device=logits.device)
            # quantized_dict = None
            # gates = None
        else:
            logits, vq_loss, perplexity, quantized_dict, gates = self.forward(x, return_vq_info=True)
        
        # Classification loss
        classification_loss = self.criterion(logits, y)
        
        # Semantic consistency loss (if enabled and codebook is learnable)
        semantic_consistency_loss = torch.tensor(0.0, device=logits.device)
        if self.use_semantic_consistency_loss and not self.freeze_codebook:
            semantic_consistency_loss = (
                self.vq_coco.get_semantic_consistency_loss() +
                self.vq_places.get_semantic_consistency_loss() +
                self.vq_emotion.get_semantic_consistency_loss() +
                self.vq_ava.get_semantic_consistency_loss() +
                self.vq_actions.get_semantic_consistency_loss()
            ) / 5.0
        
        if self.use_factor_separation_loss and quantized_dict is not None:
            factor_sep_loss = self.factor_separation_loss(quantized_dict)
        
        # Total loss
        loss = (vq_loss + 
                intent_loss_weight * classification_loss + 
                self.semantic_consistency_weight * semantic_consistency_loss +
                self.factor_separation_loss_weight * factor_sep_loss)
        
        preds = torch.sigmoid(logits)
        return loss, preds, y, vq_loss, perplexity
    
    # training_step, validation_step, test_step, on_train_batch_end, on_train_epoch_end,
    # on_validation_epoch_end, on_test_epoch_end, on_train_start are inherited from base class
    
    def setup(self, stage: str) -> None:
        """Setup model for training/validation/testing."""
        # Call base class setup first
        super().setup(stage)
        
        # Load pretrained checkpoint if provided
        if self.pretrained_ckpt_path is not None and stage == "fit":
            self._load_pretrained_checkpoint(self.pretrained_ckpt_path)
    
    def _load_pretrained_checkpoint(self, ckpt_path: str) -> None:
        """Load pretrained checkpoint weights.
        
        :param ckpt_path: Path to the pretrained checkpoint file.
        """
        print(f"Loading pretrained checkpoint from: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            # Extract state_dict
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Clean state_dict (remove torch.compile and EMA prefixes)
            cleaned_state_dict = clean_state_dict_for_loading(state_dict)
            
            # Load state_dict with strict=False to allow partial loading
            missing_keys, unexpected_keys = self.load_state_dict(cleaned_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys when loading pretrained checkpoint: {missing_keys[:10]}...")
                if len(missing_keys) > 10:
                    print(f"  (and {len(missing_keys) - 10} more)")
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in pretrained checkpoint: {unexpected_keys[:10]}...")
                if len(unexpected_keys) > 10:
                    print(f"  (and {len(unexpected_keys) - 10} more)")
            
            # Print epoch info if available
            if "epoch" in checkpoint:
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            
            print("Successfully loaded pretrained checkpoint!")
            
        except Exception as e:
            print(f"Error loading pretrained checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_mapping(self, mapping_path: str) -> None:
        """Load factor2anchor mapping from file (saved by compute_mapping.py).
        
        :param mapping_path: Path to mapping_result.pth file.
        """
        print(f"Loading mapping from: {mapping_path}")
        try:
            mapping_result = torch.load(mapping_path, map_location="cpu", weights_only=False)
            
            if "mapping" not in mapping_result:
                raise ValueError(f"mapping_result.pth must contain 'mapping' key, got keys: {list(mapping_result.keys())}")
            
            mapping = mapping_result["mapping"]
            if not isinstance(mapping, torch.Tensor):
                mapping = torch.tensor(mapping, dtype=torch.long)
            
            self.set_factor2anchor(mapping)
            print(f"Successfully loaded mapping: {mapping.tolist()}")
            
            # Print similarity matrix if available
            if "similarity_matrix" in mapping_result:
                sim_matrix = mapping_result["similarity_matrix"]
                print(f"Similarity matrix:\n{sim_matrix}")
            
        except Exception as e:
            print(f"Error loading mapping: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Separate parameters: pre_proj, factor_attention, VQs, classifier
        pre_proj_params = list(self.pre_proj.parameters())
        factor_attention_params = [self.factor_attention]
        vq_coco_params = list(self.vq_coco.parameters())
        vq_places_params = list(self.vq_places.parameters())
        vq_emotion_params = list(self.vq_emotion.parameters())
        vq_ava_params = list(self.vq_ava.parameters())
        vq_actions_params = list(self.vq_actions.parameters())
        classifier_params = list(self.classifier.parameters())
        
        gated_fusion_params = []
        if self.use_gated_fusion:
            gated_fusion_params = list(self.gated_fusion.parameters())
        
        # Unfrozen ViT transformer blocks parameters
        vit_blocks_params = []
        if self.unfreeze_last_n_blocks > 0:
            backbone = self.net.backbone
            if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'resblocks'):
                resblocks = backbone.transformer.resblocks
                total_blocks = len(resblocks)
                start_idx = total_blocks - self.unfreeze_last_n_blocks
                for i in range(start_idx, total_blocks):
                    vit_blocks_params.extend(resblocks[i].parameters())
        
        # Create parameter groups
        param_groups = []
        
        if pre_proj_params:
            param_groups.append({
                'params': pre_proj_params,
                'lr': self.lr_pre_proj,
                'weight_decay': self.wd_pre_proj
            })
        
        if factor_attention_params:
            param_groups.append({
                'params': factor_attention_params,
                'lr': self.lr_factor_attention,
                'weight_decay': self.wd_factor_attention
            })
        
        if vq_coco_params and any(p.requires_grad for p in vq_coco_params):
            param_groups.append({
                'params': [p for p in vq_coco_params if p.requires_grad],
                'lr': self.lr_vq_coco,
                'weight_decay': self.wd_vq
            })
        
        if vq_places_params and any(p.requires_grad for p in vq_places_params):
            param_groups.append({
                'params': [p for p in vq_places_params if p.requires_grad],
                'lr': self.lr_vq_places,
                'weight_decay': self.wd_vq
            })
        
        if vq_emotion_params and any(p.requires_grad for p in vq_emotion_params):
            param_groups.append({
                'params': [p for p in vq_emotion_params if p.requires_grad],
                'lr': self.lr_vq_emotion,
                'weight_decay': self.wd_vq
            })
        
        if vq_ava_params and any(p.requires_grad for p in vq_ava_params):
            param_groups.append({
                'params': [p for p in vq_ava_params if p.requires_grad],
                'lr': self.lr_vq_ava,
                'weight_decay': self.wd_vq
            })
        
        if vq_actions_params and any(p.requires_grad for p in vq_actions_params):
            param_groups.append({
                'params': [p for p in vq_actions_params if p.requires_grad],
                'lr': self.lr_vq_actions,
                'weight_decay': self.wd_vq
            })
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': self.lr_classifier,
                'weight_decay': self.wd_head
            })
        
        if gated_fusion_params:
            param_groups.append({
                'params': gated_fusion_params,
                'lr': self.lr_gated_fusion,
                'weight_decay': self.wd_gated_fusion
            })
        
        if vit_blocks_params:
            param_groups.append({
                'params': vit_blocks_params,
                'lr': self.lr_vit,
                'weight_decay': self.wd_pre_proj  # Use same weight decay as pre_proj
            })
        
        if not param_groups:
            param_groups = [{'params': self.parameters(), 'weight_decay': 0.0}]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        
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

