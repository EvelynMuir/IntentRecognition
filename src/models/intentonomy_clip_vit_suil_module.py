"""SUIL module built on top of the CLIP ViT CLS+patch-mean baseline."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from src.models.components.aslloss import AsymmetricLossOptimized
from src.models.components.intentonomy_hierarchy import (
    build_hierarchy_probabilities,
    build_hierarchy_targets,
    coarse_supervision_loss,
)
from src.models.intentonomy_clip_vit_layer_cls_patch_mean_module import (
    IntentonomyClipViTLayerClsPatchMeanModule,
)
from src.utils.metrics import eval_test_set_both_strategies, eval_validation_set


def _inverse_softplus(value: float) -> float:
    """Return a numerically stable inverse softplus for positive scalars."""
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(value_tensor)).item()


class IntentonomyClipViTSUILModule(IntentonomyClipViTLayerClsPatchMeanModule):
    """Structure- and uncertainty-aware intent learning on top of a strong CLIP baseline."""

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
        use_confidence_aware_supervision: bool = True,
        force_binarize_targets: bool = True,
        confidence_mapping: str = "discrete",
        confidence_lambda: float = 0.4,
        confidence_score_low: float = 1.0 / 3.0,
        confidence_score_mid: float = 2.0 / 3.0,
        confidence_score_high: float = 1.0,
        confidence_weight_low: float = 1.0,
        confidence_weight_mid: float = 1.15,
        confidence_weight_high: float = 1.3,
        use_hierarchy_regularization: bool = True,
        hierarchy_aggregation: str = "noisy_or",
        hierarchy_margin: float = 0.0,
        hierarchy_loss_weight: float = 0.1,
        use_coarse_auxiliary_loss: bool = False,
        hierarchy_coarse_loss_weight: float = 0.0,
        use_classwise_calibration: bool = True,
        calibration_mode: str = "bias",
        calibration_lr: Optional[float] = None,
        calibration_regularization_weight: float = 1e-4,
        initial_calibration_bias: float = 0.0,
        initial_calibration_scale: float = 1.0,
        use_learned_thresholds_for_eval: bool = True,
    ) -> None:
        if use_semantic_weighted_loss:
            raise ValueError("SUIL does not support semantic weighted loss.")

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

        self.use_confidence_aware_supervision = use_confidence_aware_supervision
        self.force_binarize_targets = force_binarize_targets
        self.confidence_mapping = confidence_mapping
        self.confidence_lambda = confidence_lambda
        self.use_hierarchy_regularization = use_hierarchy_regularization
        self.hierarchy_aggregation = hierarchy_aggregation
        self.hierarchy_margin = hierarchy_margin
        self.hierarchy_loss_weight = hierarchy_loss_weight
        self.use_coarse_auxiliary_loss = use_coarse_auxiliary_loss
        self.hierarchy_coarse_loss_weight = hierarchy_coarse_loss_weight
        self.use_classwise_calibration = use_classwise_calibration
        self.calibration_mode = calibration_mode
        self.calibration_lr = calibration_lr
        self.calibration_regularization_weight = calibration_regularization_weight
        self.use_learned_thresholds_for_eval = use_learned_thresholds_for_eval

        self.register_buffer(
            "confidence_scores",
            torch.tensor(
                [confidence_score_low, confidence_score_mid, confidence_score_high],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "confidence_weights",
            torch.tensor(
                [confidence_weight_low, confidence_weight_mid, confidence_weight_high],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.class_bias: Optional[nn.Parameter]
        self.class_scale_rho: Optional[nn.Parameter]
        if self.use_classwise_calibration:
            self.class_bias = nn.Parameter(
                torch.full((num_classes,), float(initial_calibration_bias), dtype=torch.float32)
            )
            if self.calibration_mode == "affine":
                init_rho = _inverse_softplus(initial_calibration_scale)
                self.class_scale_rho = nn.Parameter(
                    torch.full((num_classes,), init_rho, dtype=torch.float32)
                )
            elif self.calibration_mode == "bias":
                self.class_scale_rho = None
            else:
                raise ValueError(f"Unsupported calibration mode: {self.calibration_mode}")
        else:
            self.class_bias = None
            self.class_scale_rho = None

        self.train_classification_loss = MeanMetric()
        self.val_classification_loss = MeanMetric()
        self.test_classification_loss = MeanMetric()
        self.train_hierarchy_loss = MeanMetric()
        self.val_hierarchy_loss = MeanMetric()
        self.test_hierarchy_loss = MeanMetric()
        self.train_calibration_reg = MeanMetric()
        self.val_calibration_reg = MeanMetric()
        self.test_calibration_reg = MeanMetric()

    def _prepare_targets_and_soft_labels(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = batch["labels"].float()
        targets = (labels > 0).float() if self.force_binarize_targets else labels
        soft_labels = batch.get("soft_labels")
        if soft_labels is None:
            soft_labels = labels
        else:
            soft_labels = soft_labels.float()
        return targets, soft_labels

    def _compute_confidence_weights(
        self,
        targets: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.ones_like(targets)
        if not self.use_confidence_aware_supervision:
            return weights

        positive_mask = targets > 0
        soft_labels = soft_labels.clamp(min=0.0, max=1.0)

        if self.confidence_mapping == "linear":
            positive_weights = 1.0 + self.confidence_lambda * soft_labels
        elif self.confidence_mapping == "discrete":
            positive_weights = torch.ones_like(soft_labels)
            matched = torch.zeros_like(positive_mask, dtype=torch.bool)
            for score, weight in zip(self.confidence_scores, self.confidence_weights):
                score_mask = torch.isclose(
                    soft_labels,
                    score.to(device=soft_labels.device, dtype=soft_labels.dtype),
                    atol=1e-4,
                    rtol=0.0,
                )
                positive_weights = torch.where(
                    score_mask,
                    weight.to(device=soft_labels.device, dtype=soft_labels.dtype),
                    positive_weights,
                )
                matched = matched | score_mask
            fallback_mask = positive_mask & (~matched)
            fallback_weights = 1.0 + self.confidence_lambda * soft_labels
            positive_weights = torch.where(fallback_mask, fallback_weights, positive_weights)
        else:
            raise ValueError(f"Unsupported confidence mapping: {self.confidence_mapping}")

        return torch.where(positive_mask, positive_weights, weights)

    def _get_classwise_scale(self, module: Optional["IntentonomyClipViTSUILModule"] = None) -> torch.Tensor:
        current_module = module if module is not None else self
        if current_module.class_scale_rho is None:
            return torch.ones(
                current_module.num_classes,
                device=current_module.class_bias.device if current_module.class_bias is not None else None,
            )
        return torch.nn.functional.softplus(current_module.class_scale_rho)

    def _apply_classwise_calibration(
        self,
        logits: torch.Tensor,
        module: Optional["IntentonomyClipViTSUILModule"] = None,
    ) -> torch.Tensor:
        current_module = module if module is not None else self
        if not current_module.use_classwise_calibration or current_module.class_bias is None:
            return logits

        calibrated = logits + current_module.class_bias.unsqueeze(0)
        if current_module.calibration_mode == "affine":
            scales = self._get_classwise_scale(module=current_module).unsqueeze(0)
            calibrated = logits * scales + current_module.class_bias.unsqueeze(0)
        return calibrated

    def _compute_calibration_regularization(
        self,
        module: Optional["IntentonomyClipViTSUILModule"] = None,
    ) -> torch.Tensor:
        current_module = module if module is not None else self
        if not current_module.use_classwise_calibration or current_module.class_bias is None:
            return next(self.parameters()).new_tensor(0.0)

        bias_reg = current_module.class_bias.pow(2).mean()
        if current_module.calibration_mode == "affine":
            scale_reg = (self._get_classwise_scale(module=current_module) - 1.0).pow(2).mean()
            return bias_reg + scale_reg
        return bias_reg

    def _compute_classification_loss(
        self,
        calibrated_logits: torch.Tensor,
        targets: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_confidence_aware_supervision:
            if not isinstance(self.criterion, AsymmetricLossOptimized):
                raise TypeError("Confidence-aware supervision requires AsymmetricLossOptimized.")
            loss_per_class = self.criterion(calibrated_logits, targets, reduction="none")
            confidence_weights = self._compute_confidence_weights(targets, soft_labels)
            return (loss_per_class * confidence_weights).mean()
        return self.criterion(calibrated_logits, targets)

    def _compute_hierarchy_loss(
        self,
        calibrated_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_hierarchy_regularization:
            return calibrated_probs.new_tensor(0.0)

        level_probs = build_hierarchy_probabilities(
            calibrated_probs, mode=self.hierarchy_aggregation
        )
        level_targets = build_hierarchy_targets(targets)

        # Parent probabilities are deterministically aggregated from child probabilities, so
        # a parent-child consistency penalty would collapse to ~0. Coarse-level BCE gives the
        # hierarchy branch a real training signal on the induced parent predictions.
        hierarchy_loss = coarse_supervision_loss(level_probs, level_targets)

        if self.use_coarse_auxiliary_loss and self.hierarchy_coarse_loss_weight > 0.0:
            coarse_loss = coarse_supervision_loss(level_probs, level_targets, start_level=2)
            hierarchy_loss = hierarchy_loss + self.hierarchy_coarse_loss_weight * coarse_loss

        return hierarchy_loss

    def _compute_eval_class_thresholds(
        self,
        module: Optional["IntentonomyClipViTSUILModule"] = None,
    ):
        current_module = module if module is not None else self
        if (
            not current_module.use_classwise_calibration
            or current_module.class_bias is None
            or not current_module.use_learned_thresholds_for_eval
        ):
            return None

        # Validation/test now consume calibrated probabilities directly, so the learned
        # class-wise bias/scale is already baked into the scores. The equivalent decision
        # boundary is therefore the standard 0.5 threshold for every class.
        return current_module.class_bias.detach().new_full(
            (current_module.num_classes,), 0.5
        ).cpu().numpy()

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

        raw_logits = model(images)
        calibrated_logits = model._apply_classwise_calibration(raw_logits, module=model)
        calibrated_probs = torch.sigmoid(calibrated_logits)

        classification_loss = model._compute_classification_loss(
            calibrated_logits=calibrated_logits,
            targets=targets,
            soft_labels=soft_labels,
        )
        hierarchy_loss = model._compute_hierarchy_loss(
            calibrated_probs=calibrated_probs,
            targets=targets,
        )
        calibration_reg = model._compute_calibration_regularization(module=model)

        loss = (
            classification_loss * intent_loss_weight
            + model.hierarchy_loss_weight * hierarchy_loss
            + model.calibration_regularization_weight * calibration_reg
        )
        return loss, calibrated_probs, targets, classification_loss, hierarchy_loss, calibration_reg

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        current_epoch = self.current_epoch
        if current_epoch < self.intent_loss_weight_warmup_epochs:
            intent_loss_weight = self.intent_loss_weight_warmup
        else:
            intent_loss_weight = self.intent_loss_weight_normal

        loss, _, _, classification_loss, hierarchy_loss, calibration_reg = self.model_step(
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
        self.train_hierarchy_loss(hierarchy_loss)
        self.log(
            "train/hierarchy_loss",
            self.train_hierarchy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.train_calibration_reg(calibration_reg)
        self.log(
            "train/calibration_reg",
            self.train_calibration_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train/intent_loss_weight", intent_loss_weight, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, classification_loss, hierarchy_loss, calibration_reg = self.model_step(
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
        self.val_hierarchy_loss(hierarchy_loss)
        self.log(
            "val/hierarchy_loss",
            self.val_hierarchy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.val_calibration_reg(calibration_reg)
        self.log(
            "val/calibration_reg",
            self.val_calibration_reg,
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

    def on_validation_epoch_end(self) -> None:
        thresholds = self._compute_eval_class_thresholds()
        if len(self.val_preds_list) > 0:
            val_preds_all = torch.cat(self.val_preds_list, dim=0).numpy()
            val_targets_all = torch.cat(self.val_targets_list, dim=0).numpy()

            f1_dict = eval_validation_set(
                val_preds_all,
                val_targets_all,
                class_thresholds=thresholds,
            )

            self.val_f1_macro_best(f1_dict["val_macro"])
            self.log("val/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log("val/f1_macro_best", self.val_f1_macro_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val/threshold", f1_dict["threshold"], sync_dist=True)
            self.log("val/easy", f1_dict["val_easy"], sync_dist=True)
            self.log("val/medium", f1_dict["val_medium"], sync_dist=True)
            self.log("val/hard", f1_dict["val_hard"], sync_dist=True)

            self.val_preds_list.clear()
            self.val_targets_list.clear()

        if self.use_ema and self.ema_model is not None and len(self.val_ema_preds_list) > 0:
            ema_thresholds = self._compute_eval_class_thresholds(module=self.ema_model.module)
            val_ema_preds_all = torch.cat(self.val_ema_preds_list, dim=0).numpy()
            val_ema_targets_all = torch.cat(self.val_ema_targets_list, dim=0).numpy()

            f1_dict_ema = eval_validation_set(
                val_ema_preds_all,
                val_ema_targets_all,
                class_thresholds=ema_thresholds,
            )

            self.log("val_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("val_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log("val_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("val_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            self.log("val_ema/easy", f1_dict_ema["val_easy"], sync_dist=True)
            self.log("val_ema/medium", f1_dict_ema["val_medium"], sync_dist=True)
            self.log("val_ema/hard", f1_dict_ema["val_hard"], sync_dist=True)

            self.val_ema_preds_list.clear()
            self.val_ema_targets_list.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, classification_loss, hierarchy_loss, calibration_reg = self.model_step(
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
        self.test_hierarchy_loss(hierarchy_loss)
        self.log(
            "test/hierarchy_loss",
            self.test_hierarchy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.test_calibration_reg(calibration_reg)
        self.log(
            "test/calibration_reg",
            self.test_calibration_reg,
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

    def on_test_epoch_end(self) -> None:
        thresholds = self._compute_eval_class_thresholds()
        if len(self.test_preds_list) > 0:
            test_preds_all = torch.cat(self.test_preds_list, dim=0).numpy()
            test_targets_all = torch.cat(self.test_targets_list, dim=0).numpy()
            self.test_preds_all = test_preds_all
            self.test_targets_all = test_targets_all

            dual_f1_dict = eval_test_set_both_strategies(
                test_preds_all,
                test_targets_all,
                class_thresholds=thresholds,
            )
            for strategy_name, metrics in dual_f1_dict.items():
                self.log(f"test/{strategy_name}/f1_micro", metrics["val_micro"], sync_dist=True, prog_bar=True)
                self.log(f"test/{strategy_name}/f1_macro", metrics["val_macro"], sync_dist=True, prog_bar=True)
                self.log(f"test/{strategy_name}/f1_samples", metrics["val_samples"], sync_dist=True)
                self.log(
                    f"test/{strategy_name}/f1_mean",
                    (metrics["val_micro"] + metrics["val_macro"] + metrics["val_samples"]) / 3.0,
                    sync_dist=True,
                )
                self.log(f"test/{strategy_name}/mAP", metrics["val_mAP"], sync_dist=True, prog_bar=True)
                self.log(f"test/{strategy_name}/threshold", metrics["threshold"], sync_dist=True)
                self.log(f"test/{strategy_name}/easy", metrics["val_easy"], sync_dist=True)
                self.log(f"test/{strategy_name}/medium", metrics["val_medium"], sync_dist=True)
                self.log(f"test/{strategy_name}/hard", metrics["val_hard"], sync_dist=True)

            f1_dict = dual_f1_dict["no_inference_strategy"]
            self.log("test/f1_micro", f1_dict["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_macro", f1_dict["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test/f1_samples", f1_dict["val_samples"], sync_dist=True)
            self.log(
                "test/f1_mean",
                (f1_dict["val_micro"] + f1_dict["val_macro"] + f1_dict["val_samples"]) / 3.0,
                sync_dist=True,
            )
            self.log("test/mAP", f1_dict["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test/threshold", f1_dict["threshold"], sync_dist=True)
            self.log("test/easy", f1_dict["val_easy"], sync_dist=True)
            self.log("test/medium", f1_dict["val_medium"], sync_dist=True)
            self.log("test/hard", f1_dict["val_hard"], sync_dist=True)

            self.test_preds_list.clear()
            self.test_targets_list.clear()

        if self.use_ema and self.ema_model is not None and len(self.test_ema_preds_list) > 0:
            ema_thresholds = self._compute_eval_class_thresholds(module=self.ema_model.module)
            test_ema_preds_all = torch.cat(self.test_ema_preds_list, dim=0).numpy()
            test_ema_targets_all = torch.cat(self.test_ema_targets_list, dim=0).numpy()

            dual_f1_dict_ema = eval_test_set_both_strategies(
                test_ema_preds_all,
                test_ema_targets_all,
                class_thresholds=ema_thresholds,
            )
            for strategy_name, metrics in dual_f1_dict_ema.items():
                self.log(f"test_ema/{strategy_name}/f1_micro", metrics["val_micro"], sync_dist=True, prog_bar=True)
                self.log(f"test_ema/{strategy_name}/f1_macro", metrics["val_macro"], sync_dist=True, prog_bar=True)
                self.log(f"test_ema/{strategy_name}/f1_samples", metrics["val_samples"], sync_dist=True)
                self.log(
                    f"test_ema/{strategy_name}/f1_mean",
                    (metrics["val_micro"] + metrics["val_macro"] + metrics["val_samples"]) / 3.0,
                    sync_dist=True,
                )
                self.log(f"test_ema/{strategy_name}/mAP", metrics["val_mAP"], sync_dist=True, prog_bar=True)
                self.log(f"test_ema/{strategy_name}/threshold", metrics["threshold"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/easy", metrics["val_easy"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/medium", metrics["val_medium"], sync_dist=True)
                self.log(f"test_ema/{strategy_name}/hard", metrics["val_hard"], sync_dist=True)

            f1_dict_ema = dual_f1_dict_ema["no_inference_strategy"]
            self.log("test_ema/f1_micro", f1_dict_ema["val_micro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_macro", f1_dict_ema["val_macro"], sync_dist=True, prog_bar=True)
            self.log("test_ema/f1_samples", f1_dict_ema["val_samples"], sync_dist=True)
            self.log(
                "test_ema/f1_mean",
                (f1_dict_ema["val_micro"] + f1_dict_ema["val_macro"] + f1_dict_ema["val_samples"]) / 3.0,
                sync_dist=True,
            )
            self.log("test_ema/mAP", f1_dict_ema["val_mAP"], sync_dist=True, prog_bar=True)
            self.log("test_ema/threshold", f1_dict_ema["threshold"], sync_dist=True)
            self.log("test_ema/easy", f1_dict_ema["val_easy"], sync_dist=True)
            self.log("test_ema/medium", f1_dict_ema["val_medium"], sync_dist=True)
            self.log("test_ema/hard", f1_dict_ema["val_hard"], sync_dist=True)

            self.test_ema_preds_list.clear()
            self.test_ema_targets_list.clear()

    def configure_optimizers(self):
        if hasattr(self.hparams.optimizer, "keywords"):
            base_lr = self.hparams.optimizer.keywords.get("lr", 0.005)
        else:
            base_lr = 0.005

        param_groups = []

        classifier_params = [p for p in self.net.heads.parameters() if p.requires_grad]
        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": base_lr})

        calibration_params = []
        if self.use_classwise_calibration and self.class_bias is not None and self.class_bias.requires_grad:
            calibration_params.append(self.class_bias)
        if self.use_classwise_calibration and self.class_scale_rho is not None and self.class_scale_rho.requires_grad:
            calibration_params.append(self.class_scale_rho)
        if calibration_params:
            param_groups.append(
                {
                    "params": calibration_params,
                    "lr": self.calibration_lr if self.calibration_lr is not None else base_lr,
                }
            )

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
