"""
CLIP-ViT Codebook version of Intentonomy module.
This module uses CLIP's Vision Transformer as backbone with codebook-based quantization.
"""
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric

from src.models.intentonomy_clip_vit_base_module import IntentonomyClipViTBaseModule
from src.models.components.aslloss import AsymmetricLossOptimized
from src.models.components.vector_quantizer import VectorQuantizer
from src.models.components.clip_text_anchors import generate_text_anchors
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


class IntentonomyClipViTCodebookModule(IntentonomyClipViTBaseModule):
    """`LightningModule` for Intentonomy multi-label classification using CLIP Vision Transformer with Codebook.
    
    Model architecture:
    1. CLIP ViT encoder extracts all patch tokens (no global pooling)
    2. MLP Projector projects each patch to K*D dimensions and reshapes to K semantic blocks
    3. Each patch is independently quantized: each semantic block is quantized using nearest neighbor lookup in codebook
    4. All patches' quantized codes are fused (mean pooling) before classification
    5. Fused features are used for classification
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        # Codebook parameters
        k_semantic_blocks: int = 5,  # K: number of semantic blocks
        block_dim: int = 768,  # D: dimension of each semantic block (changed to 768)
        codebook_size: int = 128,  # N: number of codebook vectors
        vq_commitment_cost: float = 0.07,  # beta in VQ-VAE paper
        # EMA parameters for model weights
        use_ema: bool = True,
        ema_decay: float = 0.9997,
        # Backbone freezing
        freeze_backbone: bool = True,  # Whether to freeze CLIP ViT backbone (default True)
        unfreeze_last_n_blocks: int = 0,  # Number of last transformer blocks to unfreeze (0 = disabled, 2 = unfreeze last 2 blocks)
        # Optimizer learning rates
        lr_projector: float = 1e-3,  # Learning rate for projector
        lr_vq: float = 1e-3,  # Learning rate for vector quantizers
        lr_classifier: float = 3e-4,  # Learning rate for classifier (head)
        lr_vit: float = 1e-5,  # Learning rate for unfrozen ViT blocks
        weight_decay: float = 1e-4,  # Weight decay for optimizer (deprecated, use wd_projector/wd_vq/wd_head instead)
        # Weight decay for different parameter groups
        wd_projector: float = 1e-6,  # Weight decay for projector
        wd_vq: float = 0.0,  # Weight decay for vector quantizers
        wd_head: float = 1e-4,  # Weight decay for classifier (head)
        # Intent loss weight scheduling
        intent_loss_weight_warmup_epochs: int = 2,  # Number of epochs with reduced intent loss weight
        intent_loss_weight_warmup: float = 0.2,  # Intent loss weight during warmup epochs
        intent_loss_weight_normal: float = 1.0,  # Intent loss weight after warmup
        # Factor separation loss
        use_factor_separation_loss: bool = False,  # Whether to use factor separation loss (default False)
        factor_separation_loss_weight: float = 0.1,  # Weight for factor separation loss (default 0.1)
        # Anchor loss
        use_anchor_loss: bool = False,  # Whether to use anchor loss (default False)
        anchor_loss_weight: float = 0.05,  # Weight for anchor loss (default 0.05)
        # Pretrained checkpoint
        pretrained_ckpt_path: str = None,  # Path to pretrained checkpoint to load (default None)
        mapping_path: str = None,  # Path to mapping file (from compute_mapping.py) to load (default None)
        # Feature extraction mode
        use_cls_token: bool = False,  # If True, use CLS token as feature; If False, use patch tokens with learnable backbone.proj
    ) -> None:
        """Initialize a `IntentonomyClipViTCodebookModule`.
        
        :param net: The model to train (should be ClipVisionTransformer).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized with default params.
        :param k_semantic_blocks: Number of semantic blocks (K), default 5.
        :param block_dim: Dimension of each semantic block (D), default 768.
        :param codebook_size: Number of codebook vectors (N), default 128.
        :param vq_commitment_cost: Weight for VQ commitment loss, default 0.25.
        :param use_ema: Whether to use Exponential Moving Average for model parameters (default True).
        :param ema_decay: Decay factor for EMA (default 0.9997).
        :param freeze_backbone: Whether to freeze CLIP ViT backbone parameters (default True).
        :param lr_projector: Learning rate for projector, default 1e-3.
        :param lr_vq: Learning rate for vector quantizers, default 1e-3.
        :param lr_classifier: Learning rate for classifier (head), default 3e-4.
        :param weight_decay: Weight decay for optimizer, default 1e-4 (deprecated, use wd_projector/wd_vq/wd_head instead).
        :param wd_projector: Weight decay for projector, default 1e-6.
        :param wd_vq: Weight decay for vector quantizers, default 0.0.
        :param wd_head: Weight decay for classifier (head), default 1e-4.
        :param intent_loss_weight_warmup_epochs: Number of epochs with reduced intent loss weight, default 2.
        :param intent_loss_weight_warmup: Intent loss weight during warmup epochs, default 0.2.
        :param intent_loss_weight_normal: Intent loss weight after warmup, default 1.0.
        :param use_factor_separation_loss: Whether to use factor separation loss, default False.
        :param factor_separation_loss_weight: Weight for factor separation loss, default 0.1.
        :param unfreeze_last_n_blocks: Number of last transformer blocks to unfreeze, default 0.
        :param lr_vit: Learning rate for unfrozen ViT blocks, default 1e-5.
        :param use_anchor_loss: Whether to use anchor loss, default False.
        :param anchor_loss_weight: Weight for anchor loss, default 0.05.
        :param pretrained_ckpt_path: Path to pretrained checkpoint to load, default None.
        :param mapping_path: Path to mapping file (saved by compute_mapping.py) to load, default None.
        :param use_cls_token: If True, use CLS token as feature; If False, use patch tokens with learnable backbone.proj.
        """
        # Initialize base class
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
            use_cls_token=use_cls_token,
        )
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.k_semantic_blocks = k_semantic_blocks
        self.block_dim = block_dim
        self.codebook_size = codebook_size
        self.vq_commitment_cost = vq_commitment_cost
        self.lr_projector = lr_projector
        self.lr_vq = lr_vq
        self.lr_classifier = lr_classifier
        self.weight_decay = weight_decay
        self.wd_projector = wd_projector
        self.wd_vq = wd_vq
        self.wd_head = wd_head
        self.use_factor_separation_loss = use_factor_separation_loss
        self.factor_separation_loss_weight = factor_separation_loss_weight
        self.use_anchor_loss = use_anchor_loss
        self.anchor_loss_weight = anchor_loss_weight
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.mapping_path = mapping_path
        
        # Get CLIP ViT output dimension automatically
        vit_hidden_dim = self._get_vit_hidden_dim()
        
        # Get CLIP visual projector dimension (projected dimension, typically 768)
        vit_projected_dim = self._get_vit_projected_dim()
        
        # 降维bottleneck (共享): 在factor attention之前降维
        self.pre_proj = nn.Sequential(
            nn.Linear(vit_hidden_dim, 256),
            nn.GELU()
        )
        
        # Factor attention权重: [5, 256] (注意：维度是256，不是vit_hidden_dim)
        self.factor_attention = nn.Parameter(torch.randn(5, 256))
        # 初始化factor attention权重
        nn.init.xavier_uniform_(self.factor_attention)
        
        # 温度参数
        self.tau = 0.1
        
        # 5个独立的投影层，每个factor一个（输入是vit_projected_dim，输出是block_dim=768）
        # 使用CLIP的visual projector初始化
        self.projectors = nn.ModuleList([
            nn.Linear(vit_projected_dim, block_dim) for _ in range(5)
        ])
        
        # 使用CLIP的visual projector初始化projector
        self._init_projectors_with_clip_proj()
        
        # 5个独立的Vector Quantizers（每个factor一个）
        self.vector_quantizers = nn.ModuleList([
            VectorQuantizer(
                num_embeddings=codebook_size,
                embedding_dim=block_dim,
                commitment_cost=vq_commitment_cost,
            )
            for _ in range(5)
        ])
        
        # 两层MLP融合层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(5 * block_dim, 1024),  # 5*768=3840 -> 1024
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),  # 1024 -> 1024
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 分类头（独立的MLP）
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),  # 512 -> 28
        )
        
        if self.use_anchor_loss:
            # 生成CLIP文本anchors
            self._generate_text_anchors()
            
            # 注册 factor2anchor mapping（默认 identity mapping）
            self.register_buffer('factor2anchor', torch.arange(5, dtype=torch.long))
        
        # Metrics are initialized in base class
    
    # _get_vit_hidden_dim and _get_vit_projected_dim are inherited from base class
    
    def _init_projectors_with_clip_proj(self) -> None:
        """Initialize projectors using CLIP's visual projector weights.
        
        CLIP's visual projector (backbone.proj) is a linear layer that projects
        visual features to the same dimension as text features.
        We use this to initialize our factor projectors.
        """
        backbone = self.net.backbone
        
        # Get CLIP's visual projector
        if hasattr(backbone, 'proj') and backbone.proj is not None:
            clip_proj = backbone.proj  # [width, output_dim]
            clip_proj_weight = clip_proj.T  # [output_dim, width] for Linear layer format
            
            # Initialize each projector with CLIP's visual projector weights
            for projector in self.projectors:
                # projector is nn.Linear(vit_projected_dim, block_dim)
                # We want to initialize it with CLIP's proj weights
                if projector.weight.shape == clip_proj_weight.shape:
                    # If dimensions match exactly, copy weights
                    projector.weight.data.copy_(clip_proj_weight)
                    print(f"Initialized projector with CLIP visual projector weights (shape: {clip_proj_weight.shape})")
                elif projector.weight.shape[0] == clip_proj_weight.shape[0]:
                    # If output dim matches but input dim doesn't, use output dim part
                    projector.weight.data.copy_(clip_proj_weight[:, :projector.weight.shape[1]])
                    print(f"Initialized projector with CLIP visual projector weights (partial, shape: {projector.weight.shape})")
                else:
                    # If dimensions don't match, use Xavier initialization (already done by default)
                    print(f"Warning: Projector shape {projector.weight.shape} doesn't match CLIP proj shape {clip_proj_weight.shape}, using default initialization")
        else:
            print("Warning: CLIP visual projector not found, using default initialization for projectors")
    
    def _generate_text_anchors(self) -> None:
        """生成5个数据集的CLIP文本anchor并存储为buffer"""
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
            
            # 存储为buffer（不参与梯度更新）
            self.register_buffer('text_anchors', anchors)
            print(f"Generated {len(anchor_names)} text anchors: {anchor_names}")
            print(f"Text anchors shape: {anchors.shape}")
            
            # Process text anchors: pad to vit_projected_dim, then project to block_dim
            text_anchors_clip = anchors  # [5, clip_embedding_dim]
            clip_embedding_dim = text_anchors_clip.shape[1]
            vit_projected_dim = self._get_vit_projected_dim()
            
            def pad_to_vit_dim(x, target_dim):
                """Pad or truncate x to target_dim"""
                if x.shape[0] >= target_dim:
                    return x[:target_dim]
                pad = torch.zeros(target_dim - x.shape[0], device=x.device)
                return torch.cat([x, pad], dim=0)
            
            # Project anchors using corresponding projectors
            proj_anchors = []
            for f in range(5):
                t = pad_to_vit_dim(text_anchors_clip[f], vit_projected_dim)  # [vit_projected_dim]
                t_proj = self.projectors[f](t.unsqueeze(0)).squeeze(0)  # [block_dim=768]
                t_proj = F.normalize(t_proj, dim=0)
                proj_anchors.append(t_proj)
            
            proj_anchors = torch.stack(proj_anchors)  # [5, block_dim]
            self.register_buffer('proj_text_anchors', proj_anchors)
            print(f"Projected text anchors shape: {proj_anchors.shape}")
        except Exception as e:
            print(f"Warning: Failed to generate text anchors: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，创建一个dummy anchor
            vit_projected_dim = self._get_vit_projected_dim()
            dummy_anchors = torch.zeros(5, vit_projected_dim)
            self.register_buffer('text_anchors', dummy_anchors)
            # Also create dummy proj_text_anchors
            dummy_proj_anchors = torch.zeros(5, self.block_dim)
            self.register_buffer('proj_text_anchors', dummy_proj_anchors)
    
    # _extract_patch_tokens is inherited from base class
    
    def forward(self, x: torch.Tensor, return_vq_info: bool = False) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :param return_vq_info: Whether to return VQ loss, perplexity, and z_quantized (for training).
        :return: A tensor of logits of shape (batch_size, num_classes), or 
                 (logits, vq_loss, perplexity, z_quantized) if return_vq_info=True.
        
        Note: If you need both CLS token and patch tokens simultaneously, you can use:
              features_dict = self._extract_all_features(x)
              cls_token = features_dict["cls_token"]  # [B, 1, output_dim]
              patch_tokens = features_dict["patch_tokens"]  # [B, N_patches, output_dim]
        """
        # 1. Extract features from CLIP ViT (patch tokens or CLS token)
        # To use both features simultaneously, use: self._extract_all_features(x)
        features = self._extract_patch_tokens(x)  # [B, N, vit_projected_dim] where N=1 (CLS) or N_patches
        B, N, vit_projected_dim = features.shape
        
        # 2. Handle two modes: CLS token vs patch tokens
        if self.use_cls_token:
            # Mode 1: Use CLS token as feature
            # CLS token: [B, 1, D] -> expand to [B, 5, D] for 5 factors (each factor uses the same CLS token)
            z_f = features.expand(B, 5, vit_projected_dim)  # [B, 5, D]
        else:
            # Mode 2: Use patch tokens with factor attention
            patch_tokens = features  # [B, N_patches, D]
            
            # 降维bottleneck
            h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
            
            # Factor Attention (温度缩放sigmoid)
            A = torch.sigmoid(h @ self.factor_attention.T / self.tau)  # [B, N_patches, 5]
            
            # Normalize attention weights
            weights = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 5]
            
            # Aggregate (使用einsum)
            # patch_tokens: [B, N_patches, vit_projected_dim], weights: [B, N_patches, 5]
            # z_f: [B, 5, vit_projected_dim]
            z_f = torch.einsum('bpd,bpf->bfd', patch_tokens, weights)
        
        # 6. Project (5个独立投影层，输入是vit_projected_dim，输出是block_dim=768)
        z_f_proj = []
        for f in range(5):
            z_f_proj.append(self.projectors[f](z_f[:, f, :]))  # [B, block_dim=768]
        z_f_proj = torch.stack(z_f_proj, dim=1)  # [B, 5, block_dim=768]
        
        # 7. VQ (5个独立，使用stop-grad)
        z_quantized_list = []
        vq_losses = []
        perplexities = []
        for f in range(5):
            z_f = z_f_proj[:, f, :]  # [B, block_dim]
            vq_loss_f, z_q_f, perplexity_f, _, _ = self.vector_quantizers[f](z_f)
            # Stop-grad技巧
            z_q_f = z_q_f.detach() + (z_f - z_f.detach())
            z_quantized_list.append(z_q_f)
            vq_losses.append(vq_loss_f)
            perplexities.append(perplexity_f)
        
        # Sum VQ losses from all factors
        vq_loss = sum(vq_losses)
        
        # Average perplexity across all factors
        perplexity = torch.stack(perplexities).mean()
        
        # 8. Extract and concatenate 5 factor vectors
        z1, z2, z3, z4, z5 = [z_quantized_list[i] for i in range(5)]
        fusion = torch.cat([z1, z2, z3, z4, z5], dim=-1)  # [B, 5*768=3840]
        
        # 9. Fusion MLP (两层)
        fusion_mlp_out = self.fusion_mlp(fusion)  # [B, 1024]
        
        # 10. Classification
        logits = self.classifier(fusion_mlp_out)  # [B, num_classes]
        
        if return_vq_info:
            # Stack quantized factors for return
            z_quantized = torch.stack(z_quantized_list, dim=1)  # [B, 5, block_dim]
            return logits, vq_loss, perplexity, z_quantized
        return logits
    
    def get_code_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Get code indices for each factor in the given images.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :return: A tensor of code indices of shape (batch_size, 5).
        """
        # 1. Extract features from CLIP ViT (patch tokens or CLS token)
        features = self._extract_patch_tokens(x)  # [B, N, vit_projected_dim] where N=1 (CLS) or N_patches
        B, N, vit_projected_dim = features.shape
        
        # 2. Handle two modes: CLS token vs patch tokens (与forward保持一致)
        if self.use_cls_token:
            # Mode 1: Use CLS token as feature
            z_f = features.expand(B, 5, vit_projected_dim)  # [B, 5, D]
        else:
            # Mode 2: Use patch tokens with factor attention
            patch_tokens = features  # [B, N_patches, D]
            
            # 降维bottleneck
            h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
            
            # Factor Attention (温度缩放sigmoid)
            A = torch.sigmoid(h @ self.factor_attention.T / self.tau)  # [B, N_patches, 5]
            
            # Normalize attention weights
            weights = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 5]
            
            # Aggregate (使用einsum)
            z_f = torch.einsum('bpd,bpf->bfd', patch_tokens, weights)  # [B, 5, vit_projected_dim]
        
        # 6. Project (5个独立投影层，输入是vit_projected_dim，输出是block_dim=768)
        z_f_proj = []
        for f in range(5):
            z_f_proj.append(self.projectors[f](z_f[:, f, :]))  # [B, block_dim=768]
        z_f_proj = torch.stack(z_f_proj, dim=1)  # [B, 5, block_dim=768]
        
        # 7. Quantize and collect encoding_indices
        code_indices_list = []
        for f in range(5):
            z_f = z_f_proj[:, f, :]  # [B, block_dim]
            # Quantize using f-th VQ
            _, _, _, _, encoding_indices = self.vector_quantizers[f](z_f)
            # Squeeze to [B] and collect
            code_indices_list.append(encoding_indices.squeeze(-1))
        
        # Stack to [B, 5]
        code_indices = torch.stack(code_indices_list, dim=1)  # [B, 5]
        
        return code_indices
    
    def factor_separation_loss(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """Compute factor separation loss to encourage diversity among factors.
        
        :param z_quantized: Quantized factor vectors of shape [B, 5, block_dim].
        :return: Factor separation loss scalar.
        """
        # z_quantized shape: [B, 5, block_dim]
        z = F.normalize(z_quantized, dim=-1)  # [B, 5, block_dim]
        K = z.shape[1]  # Number of factors (5)
        eye = torch.eye(K, device=z.device).unsqueeze(0)  # [1, 5, 5]
        sim = z @ z.transpose(-2, -1)  # [B, 5, 5]
        sim = sim * (1 - eye)  # Mask out diagonal elements
        
        loss = (sim ** 2).mean()
        return loss
    
    @staticmethod
    def compute_mapping(z_all: torch.Tensor, proj_text_anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mapping from factors to text anchors.
        
        :param z_all: Quantized factor vectors of shape [N, 5, D].
        :param proj_text_anchors: Projected text anchors of shape [5, D].
        :return: Tuple of (mapping, similarity_matrix) where mapping is [5] and similarity_matrix is [5, 5].
        """
        z = F.normalize(z_all, dim=-1)             # [N, 5, D]
        t = F.normalize(proj_text_anchors, dim=-1) # [5, D]
        
        sim = torch.einsum('nfd,ad->nfa', z, t)    # [N, 5, 5]
        sim = sim.mean(0)                          # [5, 5]
        
        mapping = sim.argmax(dim=1)                # [5]
        return mapping, sim
    
    def anchor_loss(self, z_q_split: torch.Tensor) -> torch.Tensor:
        """Compute anchor loss between quantized factors and text anchors.
        
        :param z_q_split: Quantized factor vectors of shape [B, F, D] where F=5, D=block_dim.
        :return: Anchor loss scalar.
        """
        B, F, D = z_q_split.shape
        
        # 根据 mapping 重排 anchors
        anchors = self.proj_text_anchors[self.factor2anchor]  # [5, D]
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)  # [B, 5, D]
        
        # Cosine similarity
        z_norm = F.normalize(z_q_split, dim=-1)
        t_norm = F.normalize(anchors, dim=-1)
        
        sim = (z_norm * t_norm).sum(-1)  # [B, 5]
        return (1 - sim).mean()
    
    def set_factor2anchor(self, mapping: torch.Tensor) -> None:
        """Set factor2anchor mapping from external source.
        
        :param mapping: Tensor of shape [5] with anchor indices for each factor.
        """
        if mapping.shape != (5,):
            raise ValueError(f"mapping must have shape [5], got {mapping.shape}")
        self.register_buffer('factor2anchor', mapping.to(self.device))
    
    def model_step(
        self, batch: Dict[str, torch.Tensor], use_ema_model: bool = False, intent_loss_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        
        :param batch: A batch of data containing the input tensor of images and target labels.
        :param use_ema_model: Whether to use EMA model for forward pass (default False).
        :param intent_loss_weight: Weight for intent (classification) loss (default 1.0).
        
        :return: A tuple containing (in order):
            - A tensor of total losses (VQ loss + weighted classification loss).
            - A tensor of predictions (probabilities after sigmoid).
            - A tensor of target labels.
            - A tensor of VQ loss values.
            - A tensor of perplexity values.
        """
        x = batch["image"]
        y = batch["labels"]
        
        # Forward pass
        if use_ema_model:
            if self.ema_model is None:
                raise RuntimeError("EMA model is not initialized. Call on_train_start first.")
            ema_module = self.ema_model.module
            logits = ema_module(x)
            # For EMA model, we don't compute VQ loss (it's only for training)
            vq_loss = torch.tensor(0.0, device=logits.device)
            perplexity = torch.tensor(0.0, device=logits.device)
            z_quantized = None
        else:
            # Forward pass with VQ info
            logits, vq_loss, perplexity, z_quantized = self.forward(x, return_vq_info=True)
        
        # Classification loss (intent loss)
        classification_loss = self.criterion(logits, y)
        
        # Factor separation loss (optional)
        factor_sep_loss = torch.tensor(0.0, device=logits.device)
        if self.use_factor_separation_loss and z_quantized is not None:
            factor_sep_loss = self.factor_separation_loss(z_quantized)
        
        # Anchor loss (optional)
        anchor_loss_val = torch.tensor(0.0, device=logits.device)
        if self.use_anchor_loss and z_quantized is not None:
            anchor_loss_val = self.anchor_loss(z_quantized)
        
        # Total loss: vq_loss + intent_loss_weight * classification_loss + factor_separation_loss_weight * factor_separation_loss + anchor_loss_weight * anchor_loss
        loss = vq_loss + intent_loss_weight * classification_loss + \
               self.factor_separation_loss_weight * factor_sep_loss + \
               self.anchor_loss_weight * anchor_loss_val
        
        preds = torch.sigmoid(logits)  # Convert logits to probabilities
        return loss, preds, y, vq_loss, perplexity
    
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        
        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Get current epoch
        current_epoch = self.current_epoch
        
        # Set intent loss weight based on warmup configuration
        if current_epoch < self.intent_loss_weight_warmup_epochs:
            intent_loss_weight = self.intent_loss_weight_warmup
        else:
            intent_loss_weight = self.intent_loss_weight_normal
        
        loss, preds, targets, vq_loss, perplexity = self.model_step(batch, intent_loss_weight=intent_loss_weight)
        
        # Update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.train_vq_loss(vq_loss)
        self.log("train/vq_loss", self.train_vq_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        self.train_perplexity(perplexity)
        self.log("train/perplexity", self.train_perplexity, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log intent loss weight for monitoring
        self.log("train/intent_loss_weight", intent_loss_weight, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Lightning hook that is called after a training batch ends."""
        # Update EMA model
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
        # Use original model
        loss, preds, targets, vq_loss, perplexity = self.model_step(batch, use_ema_model=False)
        
        # Update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_vq_loss(vq_loss)
        self.log("val/vq_loss", self.val_vq_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        self.val_perplexity(perplexity)
        self.log("val/perplexity", self.val_perplexity, on_step=False, on_epoch=True, prog_bar=False)
        
        # Collect predictions and targets for HLEG computation
        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())
        
        # If using EMA, also use EMA model for inference
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.val_ema_preds_list.append(ema_preds.detach().cpu())
            self.val_ema_targets_list.append(targets.detach().cpu())
    
    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Compute metrics using HLEG computation method (original model)
        if len(self.val_preds_list) > 0:
            val_preds_all = torch.cat(self.val_preds_list, dim=0).numpy()
            val_targets_all = torch.cat(self.val_targets_list, dim=0).numpy()
            
            f1_dict = eval_validation_set(val_preds_all, val_targets_all)
            
            # Update best macro F1
            self.val_f1_macro_best(f1_dict["val_macro"])
            
            # Log metrics
            self.log("val/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val/threshold", f1_dict["threshold"], sync_dist=True)
            
            # Clear lists
            self.val_preds_list.clear()
            self.val_targets_list.clear()
        
        # If using EMA, also compute EMA model metrics
        if self.use_ema and self.ema_model is not None and len(self.val_ema_preds_list) > 0:
            val_ema_preds_all = torch.cat(self.val_ema_preds_list, dim=0).numpy()
            val_ema_targets_all = torch.cat(self.val_ema_targets_list, dim=0).numpy()
            
            f1_dict_ema = eval_validation_set(val_ema_preds_all, val_ema_targets_all)
            
            # Log EMA model metrics
            self.log("val_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("val_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            
            # Clear EMA lists
            self.val_ema_preds_list.clear()
            self.val_ema_targets_list.clear()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.
        
        :param batch: A batch of data containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        # Use original model
        loss, preds, targets, vq_loss, perplexity = self.model_step(batch, use_ema_model=False)
        
        # Update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("test/vq_loss", vq_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=False)
        
        # Collect predictions and targets for HLEG computation
        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())
        
        # If using EMA, also use EMA model for inference
        if self.use_ema and self.ema_model is not None:
            _, ema_preds, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.test_ema_preds_list.append(ema_preds.detach().cpu())
            self.test_ema_targets_list.append(targets.detach().cpu())
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Compute metrics using HLEG computation method (original model)
        if len(self.test_preds_list) > 0:
            test_preds_all = torch.cat(self.test_preds_list, dim=0).numpy()
            test_targets_all = torch.cat(self.test_targets_list, dim=0).numpy()
            
            f1_dict = eval_validation_set(test_preds_all, test_targets_all)
            
            # Log metrics
            self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
            
            # Clear lists
            self.test_preds_list.clear()
            self.test_targets_list.clear()
        
        # If using EMA, also compute EMA model metrics
        if self.use_ema and self.ema_model is not None and len(self.test_ema_preds_list) > 0:
            test_ema_preds_all = torch.cat(self.test_ema_preds_list, dim=0).numpy()
            test_ema_targets_all = torch.cat(self.test_ema_targets_list, dim=0).numpy()
            
            f1_dict_ema = eval_validation_set(test_ema_preds_all, test_ema_targets_all)
            
            # Log EMA model metrics
            self.log("test_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("test_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            
            # Clear EMA lists
            self.test_ema_preds_list.clear()
            self.test_ema_targets_list.clear()
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_f1_macro_best.reset()
        self.val_preds_list.clear()
        self.val_targets_list.clear()
        
        # Initialize EMA model
        if self.use_ema and self.ema_model is None:
            self.ema_model = ModelEma(self, decay=self.ema_decay)
            self.ema_model.set(self)
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit, validate, test, or predict."""
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
            
            # Load mapping from separate file if provided
            if self.mapping_path is not None:
                self._load_mapping(self.mapping_path)
            
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
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        # Separate parameters: pre_proj, factor_attention, projectors, vq, fusion_mlp, classifier
        pre_proj_params = []
        factor_attention_params = []
        projector_params = []
        vq_params = []
        fusion_mlp_params = []
        classifier_params = []
        
        # Pre-projection parameters
        pre_proj_params.extend(self.pre_proj.parameters())
        
        # Factor attention parameters
        factor_attention_params.append(self.factor_attention)
        
        # Projector parameters (5个独立投影层)
        for projector in self.projectors:
            projector_params.extend(projector.parameters())
        
        # Vector Quantizer parameters
        for vq in self.vector_quantizers:
            vq_params.extend(vq.parameters())
        
        # Fusion MLP parameters
        fusion_mlp_params.extend(self.fusion_mlp.parameters())
        
        # Classifier (head) parameters
        classifier_params.extend(self.classifier.parameters())
        
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
        
        # Create parameter groups with configurable learning rates and weight decay
        param_groups = []
        if pre_proj_params:
            param_groups.append({
                'params': pre_proj_params,
                'lr': self.lr_projector,
                'weight_decay': self.wd_projector
            })
        if factor_attention_params:
            param_groups.append({
                'params': factor_attention_params,
                'lr': self.lr_projector,
                'weight_decay': self.wd_projector
            })
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': self.lr_projector,
                'weight_decay': self.wd_projector
            })
        if vq_params:
            param_groups.append({
                'params': vq_params,
                'lr': self.lr_vq,
                'weight_decay': self.wd_vq
            })
        if fusion_mlp_params:
            param_groups.append({
                'params': fusion_mlp_params,
                'lr': self.lr_classifier,
                'weight_decay': self.wd_head
            })
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': self.lr_classifier,
                'weight_decay': self.wd_head
            })
        if vit_blocks_params:
            param_groups.append({
                'params': vit_blocks_params,
                'lr': self.lr_vit,
                'weight_decay': self.wd_projector  # Use same weight decay as projector
            })
        
        # Fallback if no groups
        if not param_groups:
            param_groups = [{'params': self.parameters(), 'weight_decay': self.weight_decay}]
        
        # Create optimizer (weight_decay is set per parameter group)
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=0.0  # Set to 0 since we set weight_decay per group
        )
        
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

