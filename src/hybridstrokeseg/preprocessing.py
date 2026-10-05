"""Explicit preprocessing alternatives for the clean-room reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

PreprocessMode = Literal["paper_minimal", "robust_uint8", "zscore_brain"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for FLAIR preprocessing sensitivity experiments."""

    mode: PreprocessMode = "paper_minimal"
    median_size: int = 3
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0


def brain_mask_from_flair(flair: np.ndarray) -> np.ndarray:
    """Approximate the skull-stripped brain support as finite non-zero FLAIR."""
    image = np.asarray(flair)
    return np.isfinite(image) & (image > 0)


def preprocess_flair(
    flair: np.ndarray,
    config: PreprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return preprocessed FLAIR and a brain mask.

    ``paper_minimal`` keeps the challenge intensity scale and applies only a
    small median filter. Other modes are sensitivity analyses for the paper's
    underspecified histogram/intensity normalization step.
    """
    cfg = config or PreprocessConfig()
    image = np.asarray(flair, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("preprocess_flair expects a 2-D slice")
    if cfg.median_size < 1 or cfg.median_size % 2 == 0:
        raise ValueError("median_size must be a positive odd integer")

    brain = brain_mask_from_flair(image)
    filtered = ndimage.median_filter(image, size=cfg.median_size, mode="nearest")
    filtered = np.where(brain, filtered, 0.0).astype(np.float32, copy=False)
    if not brain.any():
        return filtered, brain

    values = filtered[brain]
    if cfg.mode == "paper_minimal":
        output = filtered
    elif cfg.mode == "robust_uint8":
        low, high = np.percentile(values, [cfg.lower_percentile, cfg.upper_percentile])
        if high <= low:
            output = np.zeros_like(filtered)
        else:
            output = np.clip((filtered - low) / (high - low), 0.0, 1.0) * 255.0
            output[~brain] = 0.0
    elif cfg.mode == "zscore_brain":
        mean = float(values.mean())
        std = float(values.std())
        output = np.zeros_like(filtered)
        if std > 0:
            output[brain] = (values - mean) / std
    else:
        raise ValueError(f"Unknown preprocessing mode: {cfg.mode}")

    return np.asarray(output, dtype=np.float32), brain
