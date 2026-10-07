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


def histogram_match_values(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match one 1-D intensity distribution to another by empirical CDF.

    This is a deterministic clean-room implementation of classical histogram
    specification. It is used only for sensitivity analysis because the
    manuscript does not report the reference image or implementation details.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if src.size == 0 or ref.size == 0:
        raise ValueError("source and reference must be non-empty")
    if not np.isfinite(src).all() or not np.isfinite(ref).all():
        raise ValueError("source and reference must contain only finite values")

    src_values, src_inverse, src_counts = np.unique(
        src, return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(ref, return_counts=True)

    src_quantiles = np.cumsum(src_counts, dtype=np.float64)
    src_quantiles /= src_quantiles[-1]
    ref_quantiles = np.cumsum(ref_counts, dtype=np.float64)
    ref_quantiles /= ref_quantiles[-1]

    mapped_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    return mapped_values[src_inverse].astype(np.float32, copy=False)
