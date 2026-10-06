"""Small reusable helpers for deliberately non-clinical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PixelPartition:
    """Per-case random pixel partition used only for leakage diagnostics."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def random_pixel_partition(
    n_pixels: int,
    *,
    train_fraction: float,
    validation_fraction: float = 0.0,
    seed: int,
) -> PixelPartition:
    """Partition pixel indices deterministically without overlap.

    This helper intentionally enables within-patient train/test overlap at the
    *patient* level. It exists only to test the manuscript's contradictory
    random-splitting language and must not be used for clinical evaluation.
    """
    if n_pixels < 1:
        raise ValueError("n_pixels must be at least 1")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_pixels)
    n_train = int(np.floor(train_fraction * n_pixels))
    n_validation = int(np.floor(validation_fraction * n_pixels))
    train = np.sort(indices[:n_train])
    validation = np.sort(indices[n_train : n_train + n_validation])
    test = np.sort(indices[n_train + n_validation :])
    return PixelPartition(train=train, validation=validation, test=test)
