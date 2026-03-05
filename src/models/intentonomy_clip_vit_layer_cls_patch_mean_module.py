"""Lightning module for CLIP ViT layer-specific CLS token + patch mean classification.

This module concatenates the CLS token and mean-pooled patch tokens from a specified layer,
then classifies with MLP.
"""

from typing import Dict, Optional, Tuple

import torch

from src.models.components.aslloss import AsymmetricLossOptimized
from src.models.intentonomy_clip_vit_base_module import IntentonomyClipViTBaseModule


class IntentonomyClipViTLayerClsPatchMeanModule(IntentonomyClipViTBaseModule):
    """Extract CLS token and patch tokens from a specified layer, concatenate, then classify with MLP.
    
    This module extracts both the CLS token and mean-pooled patch tokens from a specified 
    transformer layer (default: layer 24), concatenates them, and trains an MLP classifier 
    for multi-label classification.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        layer_idx: int = 24,
        use_ema: bool = True,
        ema_decay: float = 0.9997,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        lr_vit: float = 1e-5,
        intent_loss_weight_warmup_epochs: int = 2,
        intent_loss_weight_warmup: float = 0.2,
        intent_loss_weight_normal: float = 1.0,
        use_semantic_weighted_loss: bool = False,
        alpha: float = 0.2,
        intent_gemini_file: Optional[str] = None,
        semantic_sim_clamp_min: Optional[float] = 0.0,
    ) -> None:
        """Initialize the module.
        
        :param net: The model to train (should be ClipVisionTransformerLayerClsPatchMean).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param num_classes: Number of classes.
        :param compile: Whether to compile the model.
        :param criterion: Loss function. If None, will use AsymmetricLossOptimized.
        :param layer_idx: Index of transformer layer to extract tokens from (default: 24).
        :param use_ema: Whether to use Exponential Moving Average.
        :param ema_decay: Decay factor for EMA.
        :param freeze_backbone: Whether to freeze CLIP ViT backbone.
        :param unfreeze_last_n_blocks: Number of last transformer blocks to unfreeze.
        :param lr_vit: Learning rate for unfrozen ViT blocks.
        :param intent_loss_weight_warmup_epochs: Number of epochs with reduced intent loss weight.
        :param intent_loss_weight_warmup: Intent loss weight during warmup epochs.
        :param intent_loss_weight_normal: Intent loss weight after warmup.
        :param use_semantic_weighted_loss: Whether to apply semantic weighted ASL in model_step.
        :param alpha: Scaling factor for semantic class weights.
        :param intent_gemini_file: Path to LLM intent description JSON for text embedding averages.
        :param semantic_sim_clamp_min: Optional lower bound for semantic cosine similarity.
        """
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
            use_cls_token=False,
        )

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        self.layer_idx = layer_idx
        self.use_semantic_weighted_loss = use_semantic_weighted_loss
        self.alpha = alpha
        self.intent_gemini_file = intent_gemini_file
        self.semantic_sim_clamp_min = semantic_sim_clamp_min

        self.register_buffer(
            "semantic_weight",
            torch.eye(num_classes, dtype=torch.float32),
            persistent=True,
        )
        self.semantic_weight_ready = False

        if self.use_semantic_weighted_loss:
            self.semantic_weight = self._build_semantic_weight_matrix(
                num_classes=num_classes,
                intent_gemini_file=intent_gemini_file,
                semantic_sim_clamp_min=semantic_sim_clamp_min,
            )
            self.semantic_weight_ready = True

        if hasattr(self.net, "layer_idx"):
            self.net.layer_idx = layer_idx

    def _build_semantic_weight_matrix(
        self,
        num_classes: int,
        intent_gemini_file: Optional[str],
        semantic_sim_clamp_min: Optional[float],
    ) -> torch.Tensor:
        """Build semantic class weight matrix from CLIP text embeddings averaged over LLM descriptions."""
        if intent_gemini_file is None:
            raise ValueError(
                "intent_gemini_file must be provided when use_semantic_weighted_loss=True."
            )
        if not hasattr(self.net, "clip_model"):
            raise ValueError(
                "Current net does not expose clip_model required for text encoding."
            )

        from src.models.intentonomy_clip_vit_slot_module import _compute_intent_embeddings_from_gemini

        with torch.no_grad():
            text_embeddings = _compute_intent_embeddings_from_gemini(
                clip_model=self.net.clip_model,
                gemini_file=intent_gemini_file,
                num_classes=num_classes,
            ).float()

            if text_embeddings.shape[0] != num_classes:
                raise ValueError(
                    f"Expected {num_classes} class text embeddings, got {text_embeddings.shape[0]}."
                )

            semantic_sim = torch.matmul(text_embeddings, text_embeddings.t())
            if semantic_sim_clamp_min is not None:
                semantic_sim = semantic_sim.clamp(min=semantic_sim_clamp_min)
            semantic_weight = 1.0 - semantic_sim

        return semantic_weight.to(dtype=torch.float32)

    def _compute_semantic_weighted_asl_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute semantic-weighted ASL using per-class losses."""
        if not isinstance(self.criterion, AsymmetricLossOptimized):
            raise TypeError(
                "Semantic weighted loss requires AsymmetricLossOptimized criterion."
            )

        loss_per_class = self.criterion(logits, targets, reduction="none")
        semantic_weight = self.semantic_weight.to(device=targets.device, dtype=targets.dtype)
        weight = torch.matmul(targets, semantic_weight)
        weight = 1.0 + self.alpha * weight
        weighted_loss = loss_per_class * weight
        return weighted_loss.mean()

    def forward(
        self, x: torch.Tensor, return_slots: bool | None = None
    ) -> torch.Tensor | Tuple[torch.Tensor, None]:
        """Perform a forward pass through the model.
        
        :param x: A tensor of images of shape (batch_size, 3, image_size, image_size).
        :param return_slots: When set, return a (logits, slots) tuple for API parity.
        :return: Logits tensor, or (logits, None) when return_slots is set.
        """
        logits = self.net(x)
        if return_slots is None:
            return logits
        return logits, None

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_ema_model: bool = False,
        intent_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        
        :param batch: A batch of data containing the input tensor of images and target labels.
        :param use_ema_model: Whether to use EMA model for forward pass.
        :param intent_loss_weight: Weight for intent (classification) loss.
        
        :return: A tuple containing (loss, preds, targets, vq_loss, perplexity).
                 For this module, vq_loss and perplexity are always 0.
        """
        images = batch["image"]
        targets = batch["labels"].float()

        # Use EMA model if requested
        if use_ema_model and self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self

        # Forward pass
        logits = model(images)
        
        # Compute loss
        if self.use_semantic_weighted_loss:
            if not self.semantic_weight_ready:
                raise RuntimeError("semantic_weight is not initialized.")
            loss = self._compute_semantic_weighted_asl_loss(logits, targets)
        else:
            loss = self.criterion(logits, targets)
        loss = loss * intent_loss_weight

        # Get predictions (probabilities)
        preds = torch.sigmoid(logits)
        
        # No VQ loss or perplexity for this module
        vq_loss = torch.tensor(0.0, device=loss.device)
        perplexity = torch.tensor(0.0, device=loss.device)

        return loss, preds, targets, vq_loss, perplexity

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get base learning rate from optimizer config
        if hasattr(self.hparams.optimizer, "keywords"):
            base_lr = self.hparams.optimizer.keywords.get("lr", 0.005)
        else:
            base_lr = 0.005

        param_groups = []

        # Classifier head parameters (always trainable)
        classifier_params = [p for p in self.net.heads.parameters() if p.requires_grad]
        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": base_lr})

        # Unfrozen ViT transformer blocks parameters (if any)
        if self.unfreeze_last_n_blocks > 0:
            vit_blocks_params = []
            backbone = self.net.backbone
            if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "resblocks"):
                resblocks = backbone.transformer.resblocks
                total_blocks = len(resblocks)
                start_idx = total_blocks - self.unfreeze_last_n_blocks
                for i in range(start_idx, total_blocks):
                    vit_blocks_params.extend([p for p in resblocks[i].parameters() if p.requires_grad])

            if vit_blocks_params:
                param_groups.append({"params": vit_blocks_params, "lr": self.lr_vit})

        # If no param groups, fall back to all trainable parameters
        if not param_groups:
            param_groups = [{"params": [p for p in self.parameters() if p.requires_grad]}]

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
