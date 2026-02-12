"""
Base module for CLIP-ViT Intentonomy modules.
Contains common functionality shared between codebook and multistream versions.
"""
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.aslloss import AsymmetricLossOptimized
from src.utils.metrics import eval_validation_set
from src.utils.ema import ModelEma


class IntentonomyClipViTBaseModule(LightningModule):
    """Base class for CLIP-ViT Intentonomy modules.
    
    This class contains common functionality shared between codebook and multistream versions:
    - Common metrics and EMA setup
    - Common training/validation/test steps
    - Common patch token extraction with CLIP projection
    - Common dimension utilities
    """
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        # EMA parameters
        use_ema: bool = True,
        ema_decay: float = 0.9997,
        # Backbone freezing
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        lr_vit: float = 1e-5,
        # Intent loss weight scheduling
        intent_loss_weight_warmup_epochs: int = 2,
        intent_loss_weight_warmup: float = 0.2,
        intent_loss_weight_normal: float = 1.0,
        # Feature extraction mode
        use_cls_token: bool = False,  # If True, use CLS token as feature; If False, use patch tokens with learnable backbone.proj
    ) -> None:
        """Initialize base module.
        
        :param net: The model to train (should be ClipVisionTransformer).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized.
        :param use_ema: Whether to use Exponential Moving Average.
        :param ema_decay: Decay factor for EMA.
        :param freeze_backbone: Whether to freeze CLIP ViT backbone.
        :param unfreeze_last_n_blocks: Number of last transformer blocks to unfreeze.
        :param lr_vit: Learning rate for unfrozen ViT blocks.
        :param intent_loss_weight_warmup_epochs: Number of epochs with reduced intent loss weight.
        :param intent_loss_weight_warmup: Intent loss weight during warmup epochs.
        :param intent_loss_weight_normal: Intent loss weight after warmup.
        :param use_cls_token: If True, use CLS token as feature; If False, use patch tokens with learnable backbone.proj.
        """
        super().__init__()
        
        self.net = net
        self.num_classes = num_classes
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.lr_vit = lr_vit
        self.intent_loss_weight_warmup_epochs = intent_loss_weight_warmup_epochs
        self.intent_loss_weight_warmup = intent_loss_weight_warmup
        self.intent_loss_weight_normal = intent_loss_weight_normal
        self.use_cls_token = use_cls_token
        
        # Freeze CLIP ViT backbone if requested
        if freeze_backbone:
            for param in self.net.backbone.parameters():
                param.requires_grad = False
            print(f"CLIP ViT backbone frozen (requires_grad=False for all parameters)")
            
            # Unfreeze last N transformer blocks if requested
            if unfreeze_last_n_blocks > 0:
                self._unfreeze_last_n_blocks(unfreeze_last_n_blocks)
        
        # Loss function
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
    
    def _unfreeze_last_n_blocks(self, n_blocks: int) -> None:
        """Unfreeze last N transformer blocks.
        
        :param n_blocks: Number of blocks to unfreeze.
        """
        backbone = self.net.backbone
        if hasattr(backbone, 'transformer'):
            transformer = backbone.transformer
            if hasattr(transformer, 'resblocks'):
                resblocks = transformer.resblocks
                total_blocks = len(resblocks)
                if n_blocks > total_blocks:
                    print(f"Warning: unfreeze_last_n_blocks ({n_blocks}) > total blocks ({total_blocks}), unfreezing all blocks")
                    n_blocks = total_blocks
                
                start_idx = total_blocks - n_blocks
                for i in range(start_idx, total_blocks):
                    for param in resblocks[i].parameters():
                        param.requires_grad = True
                print(f"Unfroze last {n_blocks} transformer blocks (indices {start_idx} to {total_blocks-1})")
            else:
                print(f"Warning: Could not find 'resblocks' in transformer. Available attributes: {dir(transformer)}")
        else:
            print(f"Warning: Could not find 'transformer' in backbone. Available attributes: {dir(backbone)}")
    
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
    
    def _get_vit_projected_dim(self) -> int:
        """Get CLIP ViT output dimension after Image Projection Layer.
        
        Returns the projected dimension (output_dim), which is typically 768 for ViT-L/14.
        """
        backbone = self.net.backbone
        
        # Check if proj layer exists and get its output dimension
        if hasattr(backbone, 'proj') and backbone.proj is not None:
            # proj shape: [width, output_dim]
            return backbone.proj.shape[1]  # output_dim
        
        # Fallback: check output_dim attribute
        if hasattr(backbone, 'output_dim'):
            return backbone.output_dim
        
        # Fallback: infer from model name
        if hasattr(self.net, 'clip_model_name'):
            clip_model_name = self.net.clip_model_name
            if "ViT-L/14" in clip_model_name:
                return 768  # ViT-L/14 projects to 768
            elif "ViT-B/16" in clip_model_name:
                return 512  # ViT-B/16 projects to 512
            elif "ViT-B/32" in clip_model_name:
                return 512  # ViT-B/32 projects to 512
        
        return 768  # Default: assume 768-dim projection
    
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from CLIP ViT backbone.
        
        Two modes:
        1. use_cls_token=True: Extract CLS token and project using backbone.proj -> [B, 1, output_dim]
        2. use_cls_token=False: Extract patch tokens and project using backbone.proj -> [B, N_patches, output_dim]
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: Features of shape (batch_size, 1, output_dim) if use_cls_token=True,
                 or (batch_size, N_patches, output_dim) if use_cls_token=False.
        """
        backbone = self.net.backbone
        
        # CLIP's ViT structure: conv1 -> reshape -> add CLS token -> add positional embedding -> ln_pre -> transformer -> ln_post -> proj
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
        
        # 7. Extract features based on mode
        if self.use_cls_token:
            # Extract CLS token (first token)
            features = x[:, 0:1, :]  # [B, 1, hidden_dim]
        else:
            # Extract patch tokens (excluding CLS token)
            features = x[:, 1:, :]  # [B, N_patches, hidden_dim]
        
        # 8. Project using CLIP's Image Projection Layer (learnable backbone.proj)
        if hasattr(backbone, 'proj') and backbone.proj is not None:
            # proj shape: [width, output_dim]
            # features: [B, N, width] where N=1 (CLS) or N_patches
            # Project: [B, N, width] @ [width, output_dim] -> [B, N, output_dim]
            features = features @ backbone.proj  # [B, N, output_dim]
        
        return features  # [B, 1, output_dim] if use_cls_token=True, or [B, N_patches, output_dim] if False
    
    def _extract_all_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract both CLS token and patch tokens from CLIP ViT backbone.
        
        This method always extracts both features regardless of use_cls_token setting,
        allowing forward methods to use both features simultaneously.
        
        :param x: Input tensor of shape (batch_size, 3, image_size, image_size).
        :return: A dictionary containing:
                 - "cls_token": CLS token of shape (batch_size, 1, output_dim) after projection
                 - "patch_tokens": Patch tokens of shape (batch_size, N_patches, output_dim) after projection
        """
        backbone = self.net.backbone
        
        # CLIP's ViT structure: conv1 -> reshape -> add CLS token -> add positional embedding -> ln_pre -> transformer -> ln_post -> proj
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
        
        # 7. Extract both CLS token and patch tokens
        cls_token_features = x[:, 0:1, :]  # [B, 1, hidden_dim]
        patch_tokens_features = x[:, 1:, :]  # [B, N_patches, hidden_dim]
        
        # 8. Project using CLIP's Image Projection Layer (learnable backbone.proj)
        if hasattr(backbone, 'proj') and backbone.proj is not None:
            # proj shape: [width, output_dim]
            # Project CLS token: [B, 1, width] @ [width, output_dim] -> [B, 1, output_dim]
            cls_token_features = cls_token_features @ backbone.proj
            # Project patch tokens: [B, N_patches, width] @ [width, output_dim] -> [B, N_patches, output_dim]
            patch_tokens_features = patch_tokens_features @ backbone.proj
        
        return {
            "cls_token": cls_token_features,  # [B, 1, output_dim]
            "patch_tokens": patch_tokens_features  # [B, N_patches, output_dim]
        }
    
    def forward(self, x: torch.Tensor, return_vq_info: bool = False) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        This method should be overridden by subclasses.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :param return_vq_info: Whether to return VQ loss, perplexity, and quantized features.
        :return: A tensor of logits of shape (batch_size, num_classes), or 
                 (logits, vq_loss, perplexity, ...) if return_vq_info=True.
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def model_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        use_ema_model: bool = False, 
        intent_loss_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        
        This method should be overridden by subclasses to implement their specific loss computation.
        
        :param batch: A batch of data containing the input tensor of images and target labels.
        :param use_ema_model: Whether to use EMA model for forward pass.
        :param intent_loss_weight: Weight for intent (classification) loss.
        
        :return: A tuple containing (loss, preds, targets, vq_loss, perplexity).
        """
        raise NotImplementedError("Subclasses must implement model_step method")
    
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

