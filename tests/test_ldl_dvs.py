import numpy as np
import torch

from scripts.run_ldl_dvs_clip import pairwise_divisiveness_loss, smooth_min


def test_smooth_min_is_symmetric_and_approaches_minimum() -> None:
    left = torch.tensor([0.1, 0.8])
    right = torch.tensor([0.7, 0.2])
    assert torch.allclose(smooth_min(left, right, 10.0), smooth_min(right, left, 10.0))
    assert torch.allclose(smooth_min(left, right, 100.0), torch.minimum(left, right), atol=1e-4)


def test_pairwise_divisiveness_loss_is_minimal_for_identity() -> None:
    targets = torch.tensor([[0.2, 0.1, 0.05, 0.2, 0.1, 0.15, 0.1, 0.1]], dtype=torch.float32)
    identity = pairwise_divisiveness_loss(targets, targets)
    shifted = pairwise_divisiveness_loss(
        targets, torch.tensor([[0.05, 0.25, 0.05, 0.05, 0.2, 0.05, 0.2, 0.15]])
    )
    expected_floor = len(POSITIVE_IDS := (0, 3, 5)) * len(NEGATIVE_IDS := (1, 4, 6, 7))
    expected_floor *= np.finfo(np.float32).eps / (8 * 7)
    assert abs(float(identity) - expected_floor) < 1e-10
    assert shifted > identity
