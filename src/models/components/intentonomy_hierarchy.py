"""Hierarchy utilities for Intentonomy.

The hierarchy definition is derived from ``Intentonomy/data/label_tree.ipynb`` and
captures the 28 fine classes together with three coarse levels (18, 15, 9).
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


# fine (28) -> hierarchy_1 (18)
HIERARCHY_LEVEL_1: List[List[int]] = [
    [24, 27],
    [5],
    [13],
    [11],
    [3, 7, 8, 9, 19],
    [4],
    [20],
    [6, 21],
    [15, 16],
    [23],
    [2, 22],
    [0],
    [17, 25],
    [10],
    [1],
    [12],
    [18, 26],
    [14],
]

# hierarchy_1 (18) -> hierarchy_2 (15)
HIERARCHY_LEVEL_2: List[List[int]] = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5, 6, 7],
    [8],
    [9, 10],
    [11],
    [12],
    [13],
    [14],
    [15],
    [16],
    [17],
]

# hierarchy_2 (15) -> hierarchy_3 (9)
HIERARCHY_LEVEL_3: List[List[int]] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7, 8],
    [9],
    [10],
    [11],
    [12, 13],
    [14],
]

INTENTONOMY_HIERARCHY: List[List[List[int]]] = [
    HIERARCHY_LEVEL_1,
    HIERARCHY_LEVEL_2,
    HIERARCHY_LEVEL_3,
]


def aggregate_parent_probs(
    child_probs: torch.Tensor,
    groups: Sequence[Sequence[int]],
    mode: str = "noisy_or",
) -> torch.Tensor:
    """Aggregate child probabilities into parent probabilities."""
    parent_probs = []
    for child_indices in groups:
        current = child_probs[:, child_indices]
        if mode == "noisy_or":
            parent = 1.0 - torch.prod(1.0 - current, dim=1)
        elif mode == "max":
            parent = current.max(dim=1).values
        else:
            raise ValueError(f"Unsupported hierarchy aggregation mode: {mode}")
        parent_probs.append(parent)

    return torch.stack(parent_probs, dim=1)


def build_hierarchy_probabilities(
    fine_probs: torch.Tensor,
    mode: str = "noisy_or",
) -> List[torch.Tensor]:
    """Build probabilities for all hierarchy levels from fine probabilities."""
    level_probs = [fine_probs]
    current = fine_probs
    for groups in INTENTONOMY_HIERARCHY:
        current = aggregate_parent_probs(current, groups, mode=mode)
        level_probs.append(current)
    return level_probs


def build_hierarchy_targets(fine_targets: torch.Tensor) -> List[torch.Tensor]:
    """Build binary targets for all hierarchy levels from fine labels."""
    level_targets = [fine_targets]
    current = fine_targets
    for groups in INTENTONOMY_HIERARCHY:
        coarse = []
        for child_indices in groups:
            coarse.append(current[:, child_indices].amax(dim=1))
        current = torch.stack(coarse, dim=1)
        level_targets.append(current)
    return level_targets


def parent_child_consistency_loss(
    level_probs: Sequence[torch.Tensor],
    groups_per_level: Sequence[Sequence[Sequence[int]]] | None = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize children whose probability exceeds their parent's probability."""
    if groups_per_level is None:
        groups_per_level = INTENTONOMY_HIERARCHY

    total_loss = None
    total_edges = 0
    for child_probs, parent_probs, groups in zip(level_probs[:-1], level_probs[1:], groups_per_level):
        for parent_idx, child_indices in enumerate(groups):
            diffs = F.relu(child_probs[:, child_indices] - parent_probs[:, parent_idx].unsqueeze(1) - margin)
            loss = diffs.sum()
            total_loss = loss if total_loss is None else total_loss + loss
            total_edges += diffs.numel()

    if total_loss is None or total_edges == 0:
        return level_probs[0].new_tensor(0.0)

    return total_loss / float(total_edges)


def coarse_supervision_loss(
    level_probs: Sequence[torch.Tensor],
    level_targets: Sequence[torch.Tensor],
    start_level: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply BCE on coarse probabilities built from fine predictions."""
    losses = []
    for probs, targets in zip(level_probs[start_level:], level_targets[start_level:]):
        probs = probs.clamp(min=eps, max=1.0 - eps)
        losses.append(F.binary_cross_entropy(probs, targets.float()))

    if not losses:
        return level_probs[0].new_tensor(0.0)

    return torch.stack(losses).mean()
