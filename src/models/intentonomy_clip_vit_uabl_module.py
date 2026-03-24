"""UABL module built on top of the CLIP ViT CLS+patch-mean baseline."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric

from src.models.intentonomy_clip_vit_layer_cls_patch_mean_module import (
    IntentonomyClipViTLayerClsPatchMeanModule,
)


class IntentonomyClipViTUABLModule(IntentonomyClipViTLayerClsPatchMeanModule):
    """Uncertainty-aware boundary learning on top of a strong CLIP baseline."""

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
        uncertainty_feature_dim: Optional[int] = None,
        uncertainty_hidden_dim: int = 512,
        uncertainty_dropout: float = 0.1,
        uncertainty_loss_weight: float = 0.1,
        uncertainty_target_mode: str = "positive_inverse",
        detach_logits_for_uncertainty: bool = True,
        adaptation_hidden_dim: int = 16,
        adaptation_dropout: float = 0.0,
        adaptation_scale_limit: float = 1.0,
        adaptation_bias_limit: float = 1.0,
        identity_regularization_weight: float = 1e-3,
        uncertainty_lr: Optional[float] = None,
        adaptation_lr: Optional[float] = None,
    ) -> None:
        if use_semantic_weighted_loss:
            raise ValueError("UABL does not support semantic weighted loss.")

        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=num_classes,
            compile=compile,
            criterion=criterion,
            layer_idx=layer_idx,
            use_ema=use_ema,
            ema_decay=ema_decay,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            lr_vit=lr_vit,
            intent_loss_weight_warmup_epochs=intent_loss_weight_warmup_epochs,
            intent_loss_weight_warmup=intent_loss_weight_warmup,
            intent_loss_weight_normal=intent_loss_weight_normal,
            use_semantic_weighted_loss=False,
            alpha=alpha,
            intent_gemini_file=intent_gemini_file,
            semantic_sim_clamp_min=semantic_sim_clamp_min,
        )

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        feature_dim = uncertainty_feature_dim
        if feature_dim is None:
            feature_dim = int(getattr(net, "concat_dim", 0))
        self.uncertainty_feature_dim = int(feature_dim)
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.uncertainty_target_mode = uncertainty_target_mode
        self.detach_logits_for_uncertainty = detach_logits_for_uncertainty
        self.identity_regularization_weight = identity_regularization_weight
        self.uncertainty_lr = uncertainty_lr
        self.adaptation_lr = adaptation_lr
        self.adaptation_scale_limit = float(adaptation_scale_limit)
        self.adaptation_bias_limit = float(adaptation_bias_limit)

        uncertainty_input_dim = self.uncertainty_feature_dim + num_classes
        self.uncertainty_head = nn.Sequential(
            nn.Linear(uncertainty_input_dim, uncertainty_hidden_dim),
            nn.ReLU(),
            nn.Dropout(uncertainty_dropout),
            nn.Linear(uncertainty_hidden_dim, num_classes),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(1, adaptation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adaptation_dropout),
            nn.Linear(adaptation_hidden_dim, 1),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(1, adaptation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(adaptation_dropout),
            nn.Linear(adaptation_hidden_dim, 1),
        )
        self._init_adaptation_heads()

        self.train_classification_loss = MeanMetric()
        self.val_classification_loss = MeanMetric()
        self.test_classification_loss = MeanMetric()
        self.train_uncertainty_loss = MeanMetric()
        self.val_uncertainty_loss = MeanMetric()
        self.test_uncertainty_loss = MeanMetric()
        self.train_identity_reg = MeanMetric()
        self.val_identity_reg = MeanMetric()
        self.test_identity_reg = MeanMetric()

    def _init_adaptation_heads(self) -> None:
        for head in (self.scale_head, self.bias_head):
            final_layer = head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)

    def _get_operable_net(self, module: Optional["IntentonomyClipViTUABLModule"] = None) -> torch.nn.Module:
        current_module = module if module is not None else self
        net = current_module.net
        return net._orig_mod if hasattr(net, "_orig_mod") else net

    def _prepare_targets_and_soft_labels(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = batch["labels"].float()
        targets = (labels > 0).float()
        soft_labels = batch.get("soft_labels")
        if soft_labels is None:
            soft_labels = labels
        else:
            soft_labels = soft_labels.float()
        return targets, soft_labels

    def _extract_features_and_logits(
        self,
        images: torch.Tensor,
        module: Optional["IntentonomyClipViTUABLModule"] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        net = self._get_operable_net(module=module)
        if hasattr(net, "_extract_cls_and_patch_mean_from_layer") and hasattr(net, "heads"):
            layer_idx = int(getattr(net, "layer_idx", self.layer_idx))
            features = net._extract_cls_and_patch_mean_from_layer(images, layer_idx)
            logits = net.heads(features)
            return features, logits

        logits = net(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        return None, logits

    def _build_uncertainty_inputs(
        self,
        features: Optional[torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        logits_for_head = logits.detach() if self.detach_logits_for_uncertainty else logits
        if self.uncertainty_feature_dim <= 0:
            return logits_for_head

        if features is None:
            feature_tensor = logits_for_head.new_zeros(logits.shape[0], self.uncertainty_feature_dim)
        else:
            feature_tensor = features
            if feature_tensor.shape[1] != self.uncertainty_feature_dim:
                raise ValueError(
                    "Feature dimension mismatch for uncertainty head: "
                    f"expected {self.uncertainty_feature_dim}, got {feature_tensor.shape[1]}"
                )
        return torch.cat([feature_tensor, logits_for_head], dim=1)

    def _predict_uncertainty(
        self,
        features: Optional[torch.Tensor],
        logits: torch.Tensor,
        module: Optional["IntentonomyClipViTUABLModule"] = None,
    ) -> torch.Tensor:
        current_module = module if module is not None else self
        uncertainty_inputs = self._build_uncertainty_inputs(features, logits)
        return torch.sigmoid(current_module.uncertainty_head(uncertainty_inputs))

    def _compute_uncertainty_targets(
        self,
        targets: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        soft_labels = soft_labels.clamp(min=0.0, max=1.0)
        if self.uncertainty_target_mode == "positive_inverse":
            return torch.where(targets > 0, 1.0 - soft_labels, torch.zeros_like(soft_labels))
        if self.uncertainty_target_mode == "inverse":
            return 1.0 - soft_labels
        raise ValueError(f"Unsupported uncertainty target mode: {self.uncertainty_target_mode}")

    def _apply_boundary_adaptation(
        self,
        logits: torch.Tensor,
        uncertainty: torch.Tensor,
        module: Optional["IntentonomyClipViTUABLModule"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_module = module if module is not None else self
        uncertainty_input = uncertainty.unsqueeze(-1)
        raw_scale = current_module.scale_head(uncertainty_input).squeeze(-1)
        raw_bias = current_module.bias_head(uncertainty_input).squeeze(-1)

        scale = 1.0 + current_module.adaptation_scale_limit * torch.tanh(raw_scale)
        bias = current_module.adaptation_bias_limit * torch.tanh(raw_bias)
        adapted_logits = scale * logits + bias
        return adapted_logits, scale, bias

    def _compute_identity_regularization(
        self,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return (scale - 1.0).abs().mean() + bias.abs().mean()

    def _compute_uncertainty_supervision_loss(
        self,
        uncertainty: torch.Tensor,
        targets: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        uncertainty_targets = self._compute_uncertainty_targets(targets, soft_labels)
        return F.mse_loss(uncertainty, uncertainty_targets)

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features, raw_logits = self._extract_features_and_logits(x)
        uncertainty = self._predict_uncertainty(features, raw_logits)
        adapted_logits, _, _ = self._apply_boundary_adaptation(raw_logits, uncertainty)
        if return_uncertainty:
            return adapted_logits, uncertainty
        return adapted_logits

    def model_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_ema_model: bool = False,
        intent_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]
        targets, soft_labels = self._prepare_targets_and_soft_labels(batch)

        if use_ema_model and self.ema_model is not None:
            model = self.ema_model.module
        else:
            model = self

        features, raw_logits = model._extract_features_and_logits(images, module=model)
        uncertainty = model._predict_uncertainty(features, raw_logits, module=model)
        adapted_logits, scale, bias = model._apply_boundary_adaptation(
            raw_logits, uncertainty, module=model
        )

        classification_loss = model.criterion(adapted_logits, targets)
        uncertainty_loss = model._compute_uncertainty_supervision_loss(
            uncertainty=uncertainty,
            targets=targets,
            soft_labels=soft_labels,
        )
        identity_reg = model._compute_identity_regularization(scale=scale, bias=bias)
        loss = (
            classification_loss * intent_loss_weight
            + model.uncertainty_loss_weight * uncertainty_loss
            + model.identity_regularization_weight * identity_reg
        )

        preds = torch.sigmoid(adapted_logits)
        return loss, preds, targets, classification_loss, uncertainty_loss, identity_reg

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        current_epoch = self.current_epoch
        if current_epoch < self.intent_loss_weight_warmup_epochs:
            intent_loss_weight = self.intent_loss_weight_warmup
        else:
            intent_loss_weight = self.intent_loss_weight_normal

        loss, _, _, classification_loss, uncertainty_loss, identity_reg = self.model_step(
            batch,
            intent_loss_weight=intent_loss_weight,
        )

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_classification_loss(classification_loss)
        self.log(
            "train/classification_loss",
            self.train_classification_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.train_uncertainty_loss(uncertainty_loss)
        self.log(
            "train/uncertainty_loss",
            self.train_uncertainty_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.train_identity_reg(identity_reg)
        self.log(
            "train/identity_reg",
            self.train_identity_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/intent_loss_weight",
            intent_loss_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, classification_loss, uncertainty_loss, identity_reg = self.model_step(
            batch,
            use_ema_model=False,
        )

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_classification_loss(classification_loss)
        self.log(
            "val/classification_loss",
            self.val_classification_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.val_uncertainty_loss(uncertainty_loss)
        self.log(
            "val/uncertainty_loss",
            self.val_uncertainty_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.val_identity_reg(identity_reg)
        self.log(
            "val/identity_reg",
            self.val_identity_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.val_preds_list.append(preds.detach().cpu())
        self.val_targets_list.append(targets.detach().cpu())

        if self.use_ema and self.ema_model is not None:
            _, ema_preds, ema_targets, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.val_ema_preds_list.append(ema_preds.detach().cpu())
            self.val_ema_targets_list.append(ema_targets.detach().cpu())

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, classification_loss, uncertainty_loss, identity_reg = self.model_step(
            batch,
            use_ema_model=False,
        )

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_classification_loss(classification_loss)
        self.log(
            "test/classification_loss",
            self.test_classification_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.test_uncertainty_loss(uncertainty_loss)
        self.log(
            "test/uncertainty_loss",
            self.test_uncertainty_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.test_identity_reg(identity_reg)
        self.log(
            "test/identity_reg",
            self.test_identity_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.test_preds_list.append(preds.detach().cpu())
        self.test_targets_list.append(targets.detach().cpu())

        if self.use_ema and self.ema_model is not None:
            _, ema_preds, ema_targets, _, _, _ = self.model_step(batch, use_ema_model=True)
            self.test_ema_preds_list.append(ema_preds.detach().cpu())
            self.test_ema_targets_list.append(ema_targets.detach().cpu())

    def configure_optimizers(self):
        if hasattr(self.hparams.optimizer, "keywords"):
            base_lr = self.hparams.optimizer.keywords.get("lr", 0.005)
        else:
            base_lr = 0.005

        net = self._get_operable_net()
        param_groups = []
        seen_params = set()

        def _append_group(params, lr):
            unique_params = []
            for param in params:
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in seen_params:
                    continue
                seen_params.add(param_id)
                unique_params.append(param)
            if unique_params:
                param_groups.append({"params": unique_params, "lr": lr})

        if hasattr(net, "heads"):
            _append_group(net.heads.parameters(), base_lr)

        _append_group(
            self.uncertainty_head.parameters(),
            self.uncertainty_lr if self.uncertainty_lr is not None else base_lr,
        )
        _append_group(
            list(self.scale_head.parameters()) + list(self.bias_head.parameters()),
            self.adaptation_lr if self.adaptation_lr is not None else base_lr,
        )

        if self.unfreeze_last_n_blocks > 0 and hasattr(net, "backbone"):
            vit_blocks_params = []
            backbone = net.backbone
            if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "resblocks"):
                resblocks = backbone.transformer.resblocks
                total_blocks = len(resblocks)
                start_idx = total_blocks - self.unfreeze_last_n_blocks
                for i in range(start_idx, total_blocks):
                    vit_blocks_params.extend(resblocks[i].parameters())
            _append_group(vit_blocks_params, self.lr_vit)

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
