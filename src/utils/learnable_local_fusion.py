from __future__ import annotations

import torch
import torch.nn as nn


def build_topk_mask(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Build a float mask that is 1.0 on the top-k classes of each sample."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {tuple(logits.shape)}")
    num_classes = logits.shape[1]
    topk = max(1, min(int(topk), num_classes))
    topk_idx = torch.topk(logits, k=topk, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(logits, dtype=logits.dtype)
    mask.scatter_(1, topk_idx, 1.0)
    return mask


def build_normalized_rank_feature(logits: torch.Tensor) -> torch.Tensor:
    """Return normalized descending rank in [0, 1] for each class."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {tuple(logits.shape)}")
    order = torch.argsort(logits, dim=1, descending=True)
    rank = torch.empty_like(order, dtype=torch.float32)
    rank.scatter_(1, order, torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32)[None, :])
    denom = max(logits.shape[1] - 1, 1)
    return rank / float(denom)


class LearnableLocalFusionAdapter(nn.Module):
    """Tiny local fusion adapter: class-wise affine or shared tiny MLP."""

    def __init__(
        self,
        num_classes: int,
        mode: str = "classwise_affine",
        hidden_dim: int = 8,
        alpha_init: float = 0.3,
        feature_set: str = "zs",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.mode = mode
        self.feature_set = feature_set

        if mode == "classwise_affine":
            self.a = nn.Parameter(torch.full((self.num_classes,), float(alpha_init), dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(self.num_classes, dtype=torch.float32))
            self.alpha_init = float(alpha_init)
            self.net = None
        elif mode == "shared_mlp":
            input_dim = 2 if feature_set == "zs" else 4
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.alpha_init = float(alpha_init)
            self._init_zero_last_layer()
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def _init_zero_last_layer(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        baseline_logits: torch.Tensor,
        text_prior: torch.Tensor,
        topk_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if baseline_logits.shape != text_prior.shape or baseline_logits.shape != topk_mask.shape:
            raise ValueError("baseline_logits, text_prior, and topk_mask must share the same shape.")

        if self.mode == "classwise_affine":
            delta = self.a[None, :] * text_prior + self.b[None, :]
        else:
            if self.feature_set == "zs":
                x = torch.stack([baseline_logits, text_prior], dim=-1)
            elif self.feature_set == "zspr":
                probs = torch.sigmoid(baseline_logits)
                rank = build_normalized_rank_feature(baseline_logits)
                x = torch.stack([baseline_logits, text_prior, probs, rank], dim=-1)
            else:
                raise ValueError(f"Unsupported feature_set: {self.feature_set}")
            delta = self.net(x).squeeze(-1)
        fused_logits = baseline_logits + topk_mask * delta
        return fused_logits, delta
