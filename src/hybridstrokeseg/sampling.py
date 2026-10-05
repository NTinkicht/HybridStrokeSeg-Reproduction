"""Balanced lesion/non-lesion pixel sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalancedSample:
    """A balanced binary training sample."""

    X: np.ndarray
    y: np.ndarray
    positive_count: int
    negative_count: int


def balanced_binary_sample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_per_class: int = 15_000,
    seed: int = 2026,
) -> BalancedSample:
    """Sample equal lesion and non-lesion observations without replacement."""
    X = np.asarray(X)
    y = np.asarray(y).astype(np.uint8, copy=False).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X must be 2-D and aligned with y")
    if target_per_class < 1:
        raise ValueError("target_per_class must be positive")

    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both lesion and non-lesion samples are required")

    count = min(target_per_class, positive.size, negative.size)
    rng = np.random.default_rng(seed)
    indices = np.concatenate(
        [
            rng.choice(positive, size=count, replace=False),
            rng.choice(negative, size=count, replace=False),
        ]
    )
    indices = indices[rng.permutation(indices.size)]
    return BalancedSample(
        X=X[indices],
        y=y[indices],
        positive_count=count,
        negative_count=count,
    )
