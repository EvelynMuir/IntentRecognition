import torch

from src.utils.learnable_local_fusion import (
    LearnableLocalFusionAdapter,
    build_normalized_rank_feature,
    build_topk_mask,
)


def test_build_topk_mask_marks_only_topk() -> None:
    logits = torch.tensor([[0.9, 0.1, 0.4, -0.2]], dtype=torch.float32)
    mask = build_topk_mask(logits, topk=2)

    expected = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    assert torch.equal(mask, expected)


def test_classwise_affine_starts_from_alpha_init() -> None:
    logits = torch.tensor([[0.9, 0.1, 0.4]], dtype=torch.float32)
    prior = torch.tensor([[0.2, -0.3, 1.1]], dtype=torch.float32)
    mask = build_topk_mask(logits, topk=2)

    affine = LearnableLocalFusionAdapter(num_classes=3, mode="classwise_affine", alpha_init=0.3)
    fused_affine, delta_affine = affine(logits, prior, mask)

    expected_delta = torch.tensor([[0.06, -0.09, 0.33]], dtype=torch.float32)
    expected_fused = logits + mask * expected_delta

    assert torch.allclose(delta_affine, expected_delta)
    assert torch.allclose(fused_affine, expected_fused)


def test_shared_mlp_zero_last_layer_starts_as_identity() -> None:
    logits = torch.tensor([[0.9, 0.1, 0.4]], dtype=torch.float32)
    prior = torch.tensor([[0.2, -0.3, 1.1]], dtype=torch.float32)
    mask = build_topk_mask(logits, topk=2)

    mlp = LearnableLocalFusionAdapter(num_classes=3, mode="shared_mlp", hidden_dim=4, feature_set="zs")
    fused_mlp, delta_mlp = mlp(logits, prior, mask)

    assert torch.allclose(delta_mlp, torch.zeros_like(delta_mlp))
    assert torch.allclose(fused_mlp, logits)


def test_build_normalized_rank_feature_descending() -> None:
    logits = torch.tensor([[0.9, 0.1, 0.4]], dtype=torch.float32)
    rank = build_normalized_rank_feature(logits)

    expected = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
    assert torch.allclose(rank, expected)
