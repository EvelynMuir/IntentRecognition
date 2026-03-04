"""
Lightning module for CLIP ViT layer-specific CLS token classification.
Extracts CLS token from a specified transformer layer (1-based) and trains an MLP classifier.
"""
from typing import Dict, Tuple

import torch
from lightning import LightningModule

from src.models.intentonomy_clip_vit_base_module import IntentonomyClipViTBaseModule


class IntentonomyClipViTLayerClsModule(IntentonomyClipViTBaseModule):
    """Lightning module for CLIP ViT layer-specific CLS token classification.
    
    This module extracts CLS token from a specified transformer layer (1-based)
    and trains an MLP classifier for multi-label classification.
    """
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        # Layer-specific parameters
        layer_idx: int = 1,
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
    ) -> None:
        """Initialize the module.
        
        :param net: The model to train (should be ClipVisionTransformerLayerCls).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized.
        :param layer_idx: 1-based transformer layer index to extract CLS token from.
        :param use_ema: Whether to use Exponential Moving Average.
        :param ema_decay: Decay factor for EMA.
        :param freeze_backbone: Whether to freeze CLIP ViT backbone.
        :param unfreeze_last_n_blocks: Number of last transformer blocks to unfreeze.
        :param lr_vit: Learning rate for unfrozen ViT blocks.
        :param intent_loss_weight_warmup_epochs: Number of epochs with reduced intent loss weight.
        :param intent_loss_weight_warmup: Intent loss weight during warmup epochs.
        :param intent_loss_weight_normal: Intent loss weight after warmup.
        """
        # Initialize base module (use_cls_token is not relevant here, but we set it to True for consistency)
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
            use_cls_token=True,  # We're using CLS token, so set to True
        )
        
        # Save hyperparameters to make them accessible via self.hparams
        # This is needed for the setup method in base module to access self.hparams.compile
        # Note: optimizer and scheduler are NOT ignored so they can be accessed via self.hparams
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        
        self.layer_idx = layer_idx
        
        # Store layer_idx in net if it has the attribute
        if hasattr(self.net, 'layer_idx'):
            self.net.layer_idx = layer_idx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :return: A tensor of logits of shape (batch_size, num_classes).
        """
        return self.net(x)
    
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
                 For this module, vq_loss and perplexity are always 0.
        """
        images = batch["image"]
        targets = batch["labels"]
        
        # Use EMA model if requested
        if use_ema_model and self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self
        
        # Forward pass
        logits = model(images)  # [B, num_classes]
        
        # Compute loss
        # AsymmetricLossOptimized already returns the mean loss (scalar)
        loss = self.criterion(logits, targets)
        loss = loss * intent_loss_weight
        
        # Get predictions (probabilities)
        preds = torch.sigmoid(logits)  # [B, num_classes]
        
        # No VQ loss or perplexity for this module
        vq_loss = torch.tensor(0.0, device=loss.device)
        perplexity = torch.tensor(0.0, device=loss.device)
        
        return loss, preds, targets, vq_loss, perplexity
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get base learning rate from optimizer config
        if hasattr(self.hparams.optimizer, 'keywords'):
            base_lr = self.hparams.optimizer.keywords.get('lr', 0.005)
        else:
            base_lr = 0.005
        
        # Collect parameters to optimize
        param_groups = []
        
        # Classifier head parameters (always trainable)
        classifier_params = [p for p in self.net.heads.parameters() if p.requires_grad]
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': base_lr
            })
        
        # Unfrozen ViT transformer blocks parameters (if any)
        if self.unfreeze_last_n_blocks > 0:
            vit_blocks_params = []
            backbone = self.net.backbone
            if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'resblocks'):
                resblocks = backbone.transformer.resblocks
                total_blocks = len(resblocks)
                start_idx = total_blocks - self.unfreeze_last_n_blocks
                for i in range(start_idx, total_blocks):
                    vit_blocks_params.extend([p for p in resblocks[i].parameters() if p.requires_grad])
            
            if vit_blocks_params:
                param_groups.append({
                    'params': vit_blocks_params,
                    'lr': self.lr_vit
                })
        
        # If no param groups, fall back to all trainable parameters
        if not param_groups:
            param_groups = [{'params': [p for p in self.parameters() if p.requires_grad]}]
        
        optimizer = self.hparams.optimizer(params=param_groups)
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1_macro",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
