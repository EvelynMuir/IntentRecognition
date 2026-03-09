from functools import partial

import numpy as np
import torch
import torch.nn as nn

from src.models.intentonomy_clip_vit_suil_module import IntentonomyClipViTSUILModule


class _IdentityNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.heads = nn.Identity()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _build_suil_module(**kwargs) -> IntentonomyClipViTSUILModule:
    return IntentonomyClipViTSUILModule(
        net=_IdentityNet(num_classes=28),
        optimizer=partial(torch.optim.SGD, lr=1e-3),
        scheduler=None,
        num_classes=28,
        compile=False,
        criterion=nn.BCEWithLogitsLoss(),
        use_ema=False,
        freeze_backbone=True,
        use_confidence_aware_supervision=False,
        **kwargs,
    )


def test_suil_hierarchy_loss_penalizes_coarse_false_positive() -> None:
    module = _build_suil_module(
        use_hierarchy_regularization=True,
        use_classwise_calibration=False,
    )

    calibrated_probs = torch.zeros(1, 28)
    calibrated_probs[0, 24] = 0.9
    targets = torch.zeros_like(calibrated_probs)

    hierarchy_loss = module._compute_hierarchy_loss(calibrated_probs, targets)

    assert hierarchy_loss.item() > 0.1


def test_suil_model_step_returns_calibrated_probabilities_for_eval() -> None:
    module = _build_suil_module(
        use_classwise_calibration=True,
        calibration_mode="bias",
        use_hierarchy_regularization=False,
    )
    with torch.no_grad():
        module.class_bias.fill_(5.0)

    batch = {
        "image": torch.zeros(2, 28),
        "labels": torch.zeros(2, 28),
        "soft_labels": torch.zeros(2, 28),
    }

    _, preds, _, _, _, _ = module.model_step(batch)
    thresholds = module._compute_eval_class_thresholds()

    expected_prob = torch.sigmoid(torch.tensor(5.0))
    assert torch.allclose(preds, torch.full_like(preds, expected_prob), atol=1e-6)
    assert np.allclose(thresholds, np.full(28, 0.5, dtype=np.float32))
