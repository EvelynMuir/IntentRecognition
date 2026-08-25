#!/usr/bin/env python3
"""Shared semantic-prior normalisation + SLR-C reconstruction.

`analyze_distillation_slrc._apply_slr` hard-codes a per-sample z-score of the prior
(`zscore` here) followed by an additive top-k update.  The EMOTIC sweep
(`sweep_emotic_slrc.py`) needs to vary that normalisation, so the two pieces live
here and both call sites import them.

`_norm_prior(prior, "zscore") -> _apply(...)` is numerically identical to
`analyze_distillation_slrc._apply_slr(base, prior, topk, alpha)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NORMS = [
    "zscore", "rank", "minmax", "centered_softmax",
    "class_zscore", "class_rank", "class_then_sample_zscore",
]
# normalisations whose statistics run over the sample axis and must therefore be
# fitted on the fold's train rows only
FOLD_FITTED = {"class_zscore", "class_rank", "class_then_sample_zscore"}


def _sample_zscore(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = np.maximum(scores.std(axis=1, keepdims=True), eps)
    return (scores - mean) / std


def norm_prior(prior: np.ndarray, mode: str, fit_idx: np.ndarray | None = None) -> np.ndarray:
    """Normalise the semantic prior.  `fit_idx` restricts class-wise statistics to
    the fold's train rows so nothing is fitted on val/test rows."""
    prior = np.asarray(prior, dtype=np.float32)
    fit = prior if fit_idx is None else prior[np.asarray(fit_idx, dtype=np.int64)]
    if mode == "zscore":
        return _sample_zscore(prior)
    if mode == "minmax":
        lo = prior.min(axis=1, keepdims=True)
        hi = prior.max(axis=1, keepdims=True)
        return (prior - lo) / np.maximum(hi - lo, 1e-6)
    if mode == "rank":
        order = np.argsort(np.argsort(prior, axis=1), axis=1).astype(np.float32)
        return order / max(prior.shape[1] - 1, 1)
    if mode == "centered_softmax":
        z = prior - prior.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
        return (p - p.mean(axis=1, keepdims=True)) * prior.shape[1]
    if mode == "class_zscore":
        mu = fit.mean(axis=0, keepdims=True)
        sd = np.maximum(fit.std(axis=0, keepdims=True), 1e-6)
        return (prior - mu) / sd
    if mode == "class_rank":
        out = np.empty_like(prior)
        for c in range(prior.shape[1]):
            ref = np.sort(fit[:, c])
            out[:, c] = np.searchsorted(ref, prior[:, c], side="left") / max(ref.size, 1)
        return (out - 0.5) * 2.0
    if mode == "class_then_sample_zscore":
        mu = fit.mean(axis=0, keepdims=True)
        sd = np.maximum(fit.std(axis=0, keepdims=True), 1e-6)
        return _sample_zscore((prior - mu) / sd)
    raise ValueError(f"unknown prior normalisation: {mode}")


def apply_slr(base_logits: np.ndarray, prior_normed: np.ndarray, topk: int, alpha: float) -> np.ndarray:
    """Additive SLR-C reconstruction on the top-k baseline classes.
    `prior_normed` must already be normalised by `norm_prior`."""
    base = np.asarray(base_logits, dtype=np.float32)
    prior_normed = np.asarray(prior_normed, dtype=np.float32)
    out = base.copy()
    c = base.shape[1]
    topk = max(1, min(int(topk), c))
    if topk >= c:
        return base + float(alpha) * prior_normed
    idx = np.argpartition(-base, kth=topk - 1, axis=1)[:, :topk]
    rows = np.arange(base.shape[0])[:, None]
    out[rows, idx] = base[rows, idx] + float(alpha) * prior_normed[rows, idx]
    return out
