"""
CLIP-ViT Codebook version of Intentonomy module.
This module uses CLIP's Vision Transformer as backbone with codebook-based quantization.
"""
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.aslloss import AsymmetricLossOptimized
from src.models.components.vector_quantizer import VectorQuantizer
from src.models.components.clip_text_anchors import generate_text_anchors
from src.utils.metrics import eval_validation_set
from src.utils.ema import ModelEma
import clip


class IntentonomyClipViTCodebookModule(LightningModule):
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
        block_dim: int = 256,  # D: dimension of each semantic block
        codebook_size: int = 128,  # N: number of codebook vectors
        vq_commitment_cost: float = 0.07,  # beta in VQ-VAE paper
        # EMA parameters for model weights
        use_ema: bool = True,
        ema_decay: float = 0.9997,
        # Backbone freezing
        freeze_backbone: bool = True,  # Whether to freeze CLIP ViT backbone (default True)
        # Optimizer learning rates
        lr_projector: float = 1e-3,  # Learning rate for projector
        lr_vq: float = 1e-3,  # Learning rate for vector quantizers
        lr_classifier: float = 3e-4,  # Learning rate for classifier (head)
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
    ) -> None:
        """Initialize a `IntentonomyClipViTCodebookModule`.
        
        :param net: The model to train (should be ClipVisionTransformer).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized with default params.
        :param k_semantic_blocks: Number of semantic blocks (K), default 5.
        :param block_dim: Dimension of each semantic block (D), default 256.
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
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.net = net
        self.num_classes = num_classes
        self.k_semantic_blocks = k_semantic_blocks
        self.block_dim = block_dim
        self.codebook_size = codebook_size
        self.vq_commitment_cost = vq_commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.freeze_backbone = freeze_backbone
        self.lr_projector = lr_projector
        self.lr_vq = lr_vq
        self.lr_classifier = lr_classifier
        self.weight_decay = weight_decay
        self.wd_projector = wd_projector
        self.wd_vq = wd_vq
        self.wd_head = wd_head
        self.intent_loss_weight_warmup_epochs = intent_loss_weight_warmup_epochs
        self.intent_loss_weight_warmup = intent_loss_weight_warmup
        self.intent_loss_weight_normal = intent_loss_weight_normal
        self.use_factor_separation_loss = use_factor_separation_loss
        self.factor_separation_loss_weight = factor_separation_loss_weight
        
        # Freeze CLIP ViT backbone if requested
        if freeze_backbone:
            for param in self.net.backbone.parameters():
                param.requires_grad = False
            print(f"CLIP ViT backbone frozen (requires_grad=False for all parameters)")
        
        # Get CLIP ViT output dimension automatically
        vit_hidden_dim = self._get_vit_hidden_dim()
        
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
        
        # 5个独立的投影层，每个factor一个（输入是vit_hidden_dim，输出是block_dim）
        self.projectors = nn.ModuleList([
            nn.Linear(vit_hidden_dim, block_dim) for _ in range(5)
        ])
        
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
            nn.Linear(5 * block_dim, 1024),  # 1280 -> 1024
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
        
        # 生成CLIP文本anchors
        self._generate_text_anchors()
        
        # Loss function for multi-label classification
        if criterion is None:
            self.criterion = AsymmetricLossOptimized()
        else:
            self.criterion = criterion
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_vq_loss = MeanMetric()
        self.val_vq_loss = MeanMetric()
        self.test_vq_loss = MeanMetric()
        self.train_perplexity = MeanMetric()
        self.val_perplexity = MeanMetric()
        self.test_perplexity = MeanMetric()
        
        # For tracking best validation metrics
        self.val_f1_macro_best = MaxMetric()
        
        # For collecting predictions and targets
        self.val_preds_list = []
        self.val_targets_list = []
        self.test_preds_list = []
        self.test_targets_list = []
        
        # EMA model
        self.ema_model = None
        self.val_ema_preds_list = []
        self.val_ema_targets_list = []
        self.test_ema_preds_list = []
        self.test_ema_targets_list = []
    
    def _get_vit_hidden_dim(self) -> int:
        """Get CLIP ViT output dimension automatically.
        
        :return: Hidden dimension of CLIP ViT backbone.
        """
        backbone = self.net.backbone
        
        # Try to get from backbone.width
        if hasattr(backbone, 'width'):
            return backbone.width
        
        # Try to get from transformer.width
        if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'width'):
            return backbone.transformer.width
        
        # Try to get from net.hidden_dim (if ClipVisionTransformer stores it)
        if hasattr(self.net, 'hidden_dim'):
            return self.net.hidden_dim
        
        # Infer from clip_model_name
        if hasattr(self.net, 'clip_model_name'):
            clip_model_name = self.net.clip_model_name
            if "ViT-B/32" in clip_model_name:
                return 512
            elif "ViT-B/16" in clip_model_name:
                return 768
            elif "ViT-L/14" in clip_model_name:
                return 1024
        
        # Default fallback
        return 512
    
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
        except Exception as e:
            print(f"Warning: Failed to generate text anchors: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，创建一个dummy anchor
            vit_hidden_dim = self._get_vit_hidden_dim()
            dummy_anchors = torch.zeros(5, vit_hidden_dim)
            self.register_buffer('text_anchors', dummy_anchors)
    
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract all patch tokens from CLIP ViT backbone (excluding CLS token).
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Patch tokens of shape (batch_size, N_patches, hidden_dim).
        """
        backbone = self.net.backbone
        
        # CLIP's ViT structure: conv1 -> reshape -> add CLS token -> add positional embedding -> ln_pre -> transformer -> ln_post
        # 1. Patch embedding (conv1)
        x = backbone.conv1(x)  # [B, hidden_dim, H', W']
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N_patches, hidden_dim]
        
        # 2. Add CLS token
        batch_size = x.shape[0]
        cls_token = backbone.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N_patches, hidden_dim]
        
        # 3. Add positional embedding
        x = x + backbone.positional_embedding.unsqueeze(0)  # [B, 1+N_patches, hidden_dim]
        
        # 4. Pre-layer norm
        x = backbone.ln_pre(x)  # [B, 1+N_patches, hidden_dim]
        
        # 5. Through transformer
        x = x.permute(1, 0, 2)  # [1+N_patches, B, hidden_dim]
        x = backbone.transformer(x)  # [1+N_patches, B, hidden_dim]
        x = x.permute(1, 0, 2)  # [B, 1+N_patches, hidden_dim]
        
        # 6. Post-layer norm (if exists)
        if hasattr(backbone, 'ln_post'):
            x = backbone.ln_post(x)  # [B, 1+N_patches, hidden_dim]
        
        # Return all patch tokens (excluding CLS token, which is at index 0)
        return x[:, 1:, :]  # [B, N_patches, hidden_dim]
    
    def forward(self, x: torch.Tensor, return_vq_info: bool = False) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :param return_vq_info: Whether to return VQ loss, perplexity, and z_quantized (for training).
        :return: A tensor of logits of shape (batch_size, num_classes), or 
                 (logits, vq_loss, perplexity, z_quantized) if return_vq_info=True.
        """
        # 1. Extract all patch tokens from CLIP ViT (no global pooling)
        patch_tokens = self._extract_patch_tokens(x)  # [B, N_patches, vit_hidden_dim]
        B, N_patches, vit_hidden_dim = patch_tokens.shape
        
        # 2. 降维bottleneck
        h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
        
        # 3. Factor Attention (温度缩放sigmoid)
        A = torch.sigmoid(h @ self.factor_attention.T / self.tau)  # [B, N_patches, 5]
        
        # 4. Normalize attention weights
        weights = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 5]
        
        # 5. Aggregate (使用einsum)
        # patch_tokens: [B, N_patches, vit_hidden_dim], weights: [B, N_patches, 5]
        # z_f: [B, 5, vit_hidden_dim]
        z_f = torch.einsum('bpd,bpf->bfd', patch_tokens, weights)
        
        # 6. Project (5个独立投影层)
        z_f_proj = []
        for f in range(5):
            z_f_proj.append(self.projectors[f](z_f[:, f, :]))  # [B, block_dim]
        z_f_proj = torch.stack(z_f_proj, dim=1)  # [B, 5, block_dim]
        
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
        fusion = torch.cat([z1, z2, z3, z4, z5], dim=-1)  # [B, 1280]
        
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
        # 1. Extract all patch tokens from CLIP ViT (no global pooling)
        patch_tokens = self._extract_patch_tokens(x)  # [B, N_patches, vit_hidden_dim]
        B, N_patches, vit_hidden_dim = patch_tokens.shape
        
        # 2. 降维bottleneck
        h = self.pre_proj(patch_tokens)  # [B, N_patches, 256]
        
        # 3. Factor Attention (温度缩放sigmoid)
        A = torch.sigmoid(h @ self.factor_attention.T / self.tau)  # [B, N_patches, 5]
        
        # 4. Normalize attention weights
        weights = A / (A.sum(dim=1, keepdim=True) + 1e-6)  # [B, N_patches, 5]
        
        # 5. Aggregate (使用einsum)
        z_f = torch.einsum('bpd,bpf->bfd', patch_tokens, weights)  # [B, 5, vit_hidden_dim]
        
        # 6. Project (5个独立投影层)
        z_f_proj = []
        for f in range(5):
            z_f_proj.append(self.projectors[f](z_f[:, f, :]))  # [B, block_dim]
        z_f_proj = torch.stack(z_f_proj, dim=1)  # [B, 5, block_dim]
        
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
        
        # Total loss: vq_loss + intent_loss_weight * classification_loss + factor_separation_loss_weight * factor_separation_loss
        loss = vq_loss + intent_loss_weight * classification_loss + self.factor_separation_loss_weight * factor_sep_loss
        
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
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    
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

