from functools import partial

import torch
import torch.nn as nn

from src.models.intentonomy_clip_vit_uabl_module import IntentonomyClipViTUABLModule


class _FeatureIdentityNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.heads = nn.Identity()
        self.concat_dim = num_classes
        self.layer_idx = 1

    def _extract_cls_and_patch_mean_from_layer(
        self, x: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        del layer_idx
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(x)


def _build_uabl_module(num_classes: int = 3, **kwargs) -> IntentonomyClipViTUABLModule:
    return IntentonomyClipViTUABLModule(
        net=_FeatureIdentityNet(num_classes=num_classes),
        optimizer=partial(torch.optim.SGD, lr=1e-3),
        scheduler=None,
        num_classes=num_classes,
        compile=False,
        criterion=nn.BCEWithLogitsLoss(),
        use_ema=False,
        freeze_backbone=True,
        uncertainty_hidden_dim=8,
        adaptation_hidden_dim=1,
        adaptation_dropout=0.0,
        **kwargs,
    )


def test_uabl_uncertainty_targets_follow_positive_soft_labels() -> None:
    module = _build_uabl_module()

    targets = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    soft_labels = torch.tensor(
        [[1.0 / 3.0, 1.0, 0.0], [0.0, 2.0 / 3.0, 1.0]],
        dtype=torch.float32,
    )

    uncertainty_targets = module._compute_uncertainty_targets(targets, soft_labels)

    expected = torch.tensor(
        [[2.0 / 3.0, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(uncertainty_targets, expected, atol=1e-6)


def test_uabl_forward_defaults_to_identity_adaptation() -> None:
    module = _build_uabl_module()
    images = torch.tensor([[0.2, -0.4, 1.0], [1.0, 0.0, -0.5]], dtype=torch.float32)

    raw_logits = module.net(images)
    adapted_logits, uncertainty = module(images, return_uncertainty=True)

    assert torch.allclose(adapted_logits, raw_logits, atol=1e-6)
    assert uncertainty.shape == raw_logits.shape


def test_uabl_boundary_adaptation_can_rescale_logits_from_uncertainty() -> None:
    module = _build_uabl_module(adaptation_scale_limit=1.0, adaptation_bias_limit=1.0)
    with torch.no_grad():
        module.scale_head[0].weight.fill_(1.0)
        module.scale_head[0].bias.zero_()
        module.scale_head[3].weight.fill_(1.0)
        module.scale_head[3].bias.zero_()

        module.bias_head[0].weight.fill_(1.0)
        module.bias_head[0].bias.zero_()
        module.bias_head[3].weight.fill_(0.5)
        module.bias_head[3].bias.zero_()

    logits = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    uncertainty = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)

    adapted_logits, scale, bias = module._apply_boundary_adaptation(logits, uncertainty)

    expected_scale = 1.0 + torch.tanh(uncertainty)
    expected_bias = torch.tanh(0.5 * uncertainty)
    expected_logits = expected_scale * logits + expected_bias

    assert torch.allclose(scale, expected_scale, atol=1e-6)
    assert torch.allclose(bias, expected_bias, atol=1e-6)
    assert torch.allclose(adapted_logits, expected_logits, atol=1e-6)


def test_uabl_model_step_supervises_uncertainty_from_soft_labels() -> None:
    module = _build_uabl_module(uncertainty_loss_weight=1.0, identity_regularization_weight=0.0)
    with torch.no_grad():
        for parameter in module.uncertainty_head.parameters():
            parameter.zero_()

    batch = {
        "image": torch.zeros(1, 3),
        "labels": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "soft_labels": torch.tensor([[1.0 / 3.0, 1.0, 0.0]], dtype=torch.float32),
    }

    _, preds, targets, _, uncertainty_loss, identity_reg = module.model_step(batch)

    expected_uncertainty = torch.full((1, 3), 0.5, dtype=torch.float32)
    expected_targets = torch.tensor([[2.0 / 3.0, 0.0, 0.0]], dtype=torch.float32)
    expected_loss = torch.mean((expected_uncertainty - expected_targets) ** 2)

    assert torch.allclose(targets, batch["labels"], atol=1e-6)
    assert torch.allclose(preds, torch.full_like(preds, 0.5), atol=1e-6)
    assert torch.allclose(uncertainty_loss, expected_loss, atol=1e-6)
    assert torch.allclose(identity_reg, torch.tensor(0.0), atol=1e-6)
