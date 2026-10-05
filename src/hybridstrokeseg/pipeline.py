"""Reusable pieces of the single-slice historical reproduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .features import FeatureConfig, extract_nine_features
from .postprocessing import morphological_close_2d
from .preprocessing import PreprocessConfig, preprocess_flair

SliceStrategy = Literal["max_lesion", "middle_brain"]


@dataclass(frozen=True)
class PreparedSlice:
    """Features, labels and masks for one selected axial slice."""

    X: np.ndarray
    y: np.ndarray | None
    brain_mask: np.ndarray
    image_shape: tuple[int, int]
    preprocessed: np.ndarray


def select_slice_index(
    flair_volume: np.ndarray,
    lesion_volume: np.ndarray | None = None,
    *,
    strategy: SliceStrategy = "middle_brain",
) -> int:
    """Select a single axial slice.

    ``max_lesion`` intentionally uses ground truth and is therefore an oracle,
    label-informed approximation of the paper's unspecified slice selection.
    ``middle_brain`` avoids ground-truth use but may miss a lesion entirely.
    """
    flair = np.asarray(flair_volume)
    if flair.ndim != 3:
        raise ValueError("flair_volume must be 3-D")

    if strategy == "max_lesion":
        if lesion_volume is None:
            raise ValueError("max_lesion strategy requires lesion_volume")
        lesion = np.asarray(lesion_volume)
        if lesion.shape != flair.shape:
            raise ValueError("lesion_volume must match flair_volume")
        return int(np.argmax((lesion > 0).sum(axis=(1, 2))))

    if strategy == "middle_brain":
        areas = (np.isfinite(flair) & (flair > 0)).sum(axis=(1, 2))
        nonempty = np.flatnonzero(areas > 0)
        if nonempty.size == 0:
            return flair.shape[0] // 2
        return int(nonempty[nonempty.size // 2])

    raise ValueError(f"Unknown slice strategy: {strategy}")


def prepare_slice(
    flair_slice: np.ndarray,
    lesion_slice: np.ndarray | None = None,
    *,
    preprocess_config: PreprocessConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> PreparedSlice:
    """Preprocess a slice and extract the nine features at brain pixels."""
    processed, brain = preprocess_flair(flair_slice, preprocess_config)
    features = extract_nine_features(processed, feature_config)
    X = features[brain]
    y = None
    if lesion_slice is not None:
        lesion = np.asarray(lesion_slice)
        if lesion.shape != processed.shape:
            raise ValueError("lesion_slice must match flair_slice")
        y = (lesion[brain] > 0).astype(np.uint8)
    return PreparedSlice(
        X=X,
        y=y,
        brain_mask=brain,
        image_shape=processed.shape,
        preprocessed=processed,
    )


def restore_slice_prediction(
    predicted_brain_pixels: np.ndarray,
    brain_mask: np.ndarray,
    *,
    close_radius: int = 1,
) -> np.ndarray:
    """Restore vector predictions to a 2-D mask and apply morphological closing."""
    brain = np.asarray(brain_mask, dtype=bool)
    values = np.asarray(predicted_brain_pixels, dtype=bool).reshape(-1)
    if values.size != int(brain.sum()):
        raise ValueError("predicted_brain_pixels does not match brain_mask")
    restored = np.zeros(brain.shape, dtype=bool)
    restored[brain] = values
    closed = morphological_close_2d(restored, radius=close_radius)
    return closed & brain
