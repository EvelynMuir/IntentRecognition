"""Lightning module for CLIP ViT Intent Concept Reasoning Network (ICRN)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from src.models.intentonomy_clip_vit_base_module import IntentonomyClipViTBaseModule


class IntentonomyClipViTICRNModule(IntentonomyClipViTBaseModule):
    """Intentonomy module using concept grounding and graph reasoning."""

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
        use_prior_regularization: bool = True,
        lambda_prior: float = 0.1,
        use_sparse_regularization: bool = False,
        lambda_sparse: float = 0.01,
    ) -> None:
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
        self.use_prior_regularization = use_prior_regularization
        self.lambda_prior = lambda_prior
        self.use_sparse_regularization = use_sparse_regularization
        self.lambda_sparse = lambda_sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_ema_model: bool = False,
        intent_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]
        targets = batch["labels"].float()

        if use_ema_model and self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self

        if use_ema_model:
            logits = model(images)
            aux = None
        else:
            logits, aux = model.net(images, return_aux=True)

        cls_loss = self.criterion(logits, targets)

        prior_loss = torch.tensor(0.0, device=logits.device)
        sparse_loss = torch.tensor(0.0, device=logits.device)

        if (
            aux is not None
            and self.use_prior_regularization
            and aux.get("prior_target") is not None
            and hasattr(self.net, "intent_composition")
        ):
            prior_target = aux["prior_target"].to(
                device=self.net.intent_composition.device,
                dtype=self.net.intent_composition.dtype,
            )
            prior_loss = torch.mean((self.net.intent_composition - prior_target) ** 2)

        if aux is not None and self.use_sparse_regularization and "concept_activation" in aux:
            sparse_loss = torch.mean(torch.abs(aux["concept_activation"]))

        loss = cls_loss + self.lambda_prior * prior_loss + self.lambda_sparse * sparse_loss
        loss = loss * intent_loss_weight

        if self.training and not use_ema_model:
            self.log("train/loss_cls", cls_loss.detach(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/loss_prior", prior_loss.detach(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/loss_sparse", sparse_loss.detach(), on_step=False, on_epoch=True, prog_bar=False)

        preds = torch.sigmoid(logits)
        vq_loss = torch.tensor(0.0, device=loss.device)
        perplexity = torch.tensor(0.0, device=loss.device)

        return loss, preds, targets, vq_loss, perplexity

    def configure_optimizers(self):
        if hasattr(self.hparams.optimizer, "keywords"):
            base_lr = self.hparams.optimizer.keywords.get("lr", 0.005)
        else:
            base_lr = 0.005

        param_groups = []

        trainable_non_backbone = [
            p for n, p in self.net.named_parameters() if p.requires_grad and not n.startswith("backbone.")
        ]
        if trainable_non_backbone:
            param_groups.append({"params": trainable_non_backbone, "lr": base_lr})

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
