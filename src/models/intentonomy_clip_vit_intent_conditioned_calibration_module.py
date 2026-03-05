"""Lightning module for CLIP ViT Intent-Conditioned Feature Calibration."""

from typing import Dict, Tuple

import torch

from src.models.intentonomy_clip_vit_base_module import IntentonomyClipViTBaseModule


class IntentonomyClipViTIntentConditionedCalibrationModule(IntentonomyClipViTBaseModule):
    """Intentonomy Lightning module with intent-conditioned feature calibration."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 28,
        compile: bool = False,
        criterion: torch.nn.Module = None,
        use_ema: bool = True,
        ema_decay: float = 0.9997,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        lr_vit: float = 1e-5,
        intent_loss_weight_warmup_epochs: int = 2,
        intent_loss_weight_warmup: float = 0.2,
        intent_loss_weight_normal: float = 1.0,
    ) -> None:
        if not freeze_backbone:
            raise ValueError(
                "This method requires frozen vision/text encoders. Set freeze_backbone=true."
            )
        if unfreeze_last_n_blocks > 0:
            raise ValueError(
                "This method enforces fully frozen CLIP encoder. Set unfreeze_last_n_blocks=0."
            )

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
            unfreeze_last_n_blocks=0,
            lr_vit=lr_vit,
            intent_loss_weight_warmup_epochs=intent_loss_weight_warmup_epochs,
            intent_loss_weight_warmup=intent_loss_weight_warmup,
            intent_loss_weight_normal=intent_loss_weight_normal,
            use_cls_token=False,
        )

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_ema_model: bool = False,
        intent_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]
        targets = batch["labels"]

        if use_ema_model and self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self

        logits = model(images)
        loss = self.criterion(logits, targets)
        loss = loss * intent_loss_weight

        preds = torch.sigmoid(logits)

        vq_loss = torch.tensor(0.0, device=loss.device)
        perplexity = torch.tensor(0.0, device=loss.device)

        return loss, preds, targets, vq_loss, perplexity

    def configure_optimizers(self):
        if hasattr(self.hparams.optimizer, "keywords"):
            base_lr = self.hparams.optimizer.keywords.get("lr", 0.005)
        else:
            base_lr = 0.005

        trainable_params = [
            p for n, p in self.net.named_parameters() if p.requires_grad and not n.startswith("backbone.")
        ]

        param_groups = []
        if trainable_params:
            param_groups.append({"params": trainable_params, "lr": base_lr})

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
