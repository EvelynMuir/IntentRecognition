import torch

from scripts.run_ldl_alternative_objectives import delta_ldl_loss, dpa_regularizers, lrr_ranking_loss


def test_delta_ldl_prefers_lower_kld() -> None:
    low = delta_ldl_loss(torch.tensor([0.01, 0.02]), delta_0=0.5)
    high = delta_ldl_loss(torch.tensor([0.8, 1.0]), delta_0=0.5)
    assert low < high


def test_lrr_prefers_correct_distribution_order() -> None:
    target = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32)
    correctly_ordered = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32)
    reversed_order = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)
    assert lrr_ranking_loss(target, correctly_ordered) < lrr_ranking_loss(target, reversed_order)


def test_dpa_prefers_correct_rank_and_variance() -> None:
    target = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    ranks = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    correct = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    reversed_prediction = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
    correct_rank, correct_variance = dpa_regularizers(ranks, target, correct)
    reverse_rank, reverse_variance = dpa_regularizers(ranks, target, reversed_prediction)
    assert correct_rank < reverse_rank
    assert torch.allclose(correct_variance, reverse_variance)
