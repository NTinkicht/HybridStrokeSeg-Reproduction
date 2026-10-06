"""Handcrafted 2-D feature reconstruction from the paper.

The paper reports nine MLP inputs but does not define every mathematical detail.
This module therefore separates paper-stated values from explicit reconstruction
assumptions. All ambiguous choices are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

Direction = tuple[int, int]
RunMode = Literal["threshold_count", "centered_run", "max_run"]

DIRECTIONS: tuple[Direction, ...] = (
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # 45 degrees
    (1, -1),  # 135 degrees
)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the reconstructed nine-feature descriptor.

    Paper-stated values:
      - a 3x3 neighbourhood is used;
      - run-length-like features use four directions;
      - the directional window has length 25;
      - the reported threshold is 20.

    Reconstruction assumptions:
      - "neighbourhood" is represented by one 3x3 box mean so that the total
        dimensionality remains nine;
      - "weighted local mean" uses a normalized 3x3 binomial kernel;
      - the exact directional descriptor is not specified by the paper. Three
        explicit sensitivity variants are available:
          * ``threshold_count``: number of thresholded samples in the window;
          * ``centered_run``: contiguous thresholded run through the center;
          * ``max_run``: longest thresholded run anywhere in the window;
      - spatial coordinates are raw row/column indices by default.
    """

    threshold: float = 20.0
    run_window: int = 25
    run_mode: RunMode = "threshold_count"
    coordinate_mode: Literal["raw", "normalized"] = "raw"
    boundary_mode: Literal["nearest", "reflect", "constant"] = "nearest"

    def __post_init__(self) -> None:
        if self.run_window < 1 or self.run_window % 2 == 0:
            raise ValueError("run_window must be a positive odd integer")
        if self.run_mode not in {"threshold_count", "centered_run", "max_run"}:
            raise ValueError("Unsupported run_mode")
        if self.coordinate_mode not in {"raw", "normalized"}:
            raise ValueError("coordinate_mode must be 'raw' or 'normalized'")


def feature_names() -> tuple[str, ...]:
    """Return feature names in the model input order."""
    return (
        "row",
        "col",
        "intensity",
        "neighborhood_mean_3x3",
        "weighted_local_mean_3x3",
        "run_horizontal",
        "run_vertical",
        "run_45deg",
        "run_135deg",
    )


def _coordinates(shape: tuple[int, int], mode: str) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=np.float32)
    if mode == "normalized":
        if shape[0] > 1:
            rows /= float(shape[0] - 1)
        if shape[1] > 1:
            cols /= float(shape[1] - 1)
    return rows, cols


def _box_mean(image: np.ndarray, boundary_mode: str) -> np.ndarray:
    return ndimage.uniform_filter(image, size=3, mode=boundary_mode)


def _weighted_local_mean(image: np.ndarray, boundary_mode: str) -> np.ndarray:
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    return ndimage.convolve(image, kernel, mode=boundary_mode)


def _shift_with_constant(
    mask: np.ndarray,
    dr: int,
    dc: int,
    fill: bool = False,
) -> np.ndarray:
    """Return mask shifted so output[r,c] samples input[r+dr,c+dc]."""
    out = np.full(mask.shape, fill, dtype=mask.dtype)

    r_src_start = max(0, dr)
    r_src_end = mask.shape[0] + min(0, dr)
    c_src_start = max(0, dc)
    c_src_end = mask.shape[1] + min(0, dc)

    r_dst_start = max(0, -dr)
    r_dst_end = mask.shape[0] - max(0, dr)
    c_dst_start = max(0, -dc)
    c_dst_end = mask.shape[1] - max(0, dc)

    if r_src_start < r_src_end and c_src_start < c_src_end:
        out[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = mask[
            r_src_start:r_src_end,
            c_src_start:c_src_end,
        ]
    return out


def _validate_directional_inputs(
    image: np.ndarray,
    direction: Direction,
    window: int,
) -> tuple[np.ndarray, int, int, int]:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("directional descriptor expects a 2-D image")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    dr, dc = direction
    if (dr, dc) == (0, 0):
        raise ValueError("direction cannot be (0, 0)")
    return image, dr, dc, window // 2


def directional_threshold_count(
    image: np.ndarray,
    direction: Direction,
    *,
    threshold: float = 20.0,
    window: int = 25,
) -> np.ndarray:
    """Count thresholded samples in a centered directional window.

    This is the first clean-room interpretation implemented for the paper's
    underspecified "run-length" feature. Values outside the image are treated
    as below threshold.
    """
    image, dr, dc, radius = _validate_directional_inputs(image, direction, window)
    high = image >= threshold
    counts = np.zeros(image.shape, dtype=np.float32)
    for offset in range(-radius, radius + 1):
        counts += _shift_with_constant(high, offset * dr, offset * dc).astype(np.float32)
    return counts


def directional_centered_run(
    image: np.ndarray,
    direction: Direction,
    *,
    threshold: float = 20.0,
    window: int = 25,
) -> np.ndarray:
    """Return the contiguous thresholded run length passing through each pixel.

    A pixel below threshold receives zero. For a pixel above threshold, the run
    expands in both directions until the first below-threshold sample or the
    edge of the centered window. This is a plausible but unverified reading of
    the manuscript's use of the term "run-length".
    """
    image, dr, dc, radius = _validate_directional_inputs(image, direction, window)
    high = image >= threshold
    counts = high.astype(np.float32)
    active_forward = high.copy()
    active_backward = high.copy()

    for step in range(1, radius + 1):
        active_forward &= _shift_with_constant(high, step * dr, step * dc)
        active_backward &= _shift_with_constant(high, -step * dr, -step * dc)
        counts += active_forward.astype(np.float32)
        counts += active_backward.astype(np.float32)
    return counts


def directional_max_run(
    image: np.ndarray,
    direction: Direction,
    *,
    threshold: float = 20.0,
    window: int = 25,
) -> np.ndarray:
    """Return the longest thresholded run anywhere in the directional window.

    This variant tests another plausible reading of the manuscript phrase that
    a maximum value is assigned within the 1x25 directional support. It is an
    explicit sensitivity assumption, not a claim about the unpublished code.
    """
    image, dr, dc, radius = _validate_directional_inputs(image, direction, window)
    high = image >= threshold
    current = np.zeros(image.shape, dtype=np.float32)
    maximum = np.zeros(image.shape, dtype=np.float32)

    for offset in range(-radius, radius + 1):
        sample = _shift_with_constant(high, offset * dr, offset * dc)
        current = np.where(sample, current + 1.0, 0.0).astype(np.float32, copy=False)
        maximum = np.maximum(maximum, current)
    return maximum


def directional_run_descriptor(
    image: np.ndarray,
    direction: Direction,
    *,
    threshold: float,
    window: int,
    mode: RunMode,
) -> np.ndarray:
    """Dispatch one documented directional reconstruction variant."""
    if mode == "threshold_count":
        return directional_threshold_count(
            image,
            direction,
            threshold=threshold,
            window=window,
        )
    if mode == "centered_run":
        return directional_centered_run(
            image,
            direction,
            threshold=threshold,
            window=window,
        )
    if mode == "max_run":
        return directional_max_run(
            image,
            direction,
            threshold=threshold,
            window=window,
        )
    raise ValueError(f"Unknown run mode: {mode}")


def extract_nine_features(
    flair_slice: np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Extract the reconstructed 9-D feature vector for every pixel.

    Parameters
    ----------
    flair_slice:
        A preprocessed two-dimensional FLAIR image. Preprocessing is kept
        separate because the paper does not specify enough detail to reproduce
        filtering, histogram specification, and brain extraction exactly.
    config:
        Feature reconstruction settings.

    Returns
    -------
    np.ndarray
        Array with shape ``(height, width, 9)``.
    """
    cfg = config or FeatureConfig()
    image = np.asarray(flair_slice, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("extract_nine_features expects a 2-D FLAIR slice")
    if not np.isfinite(image).all():
        raise ValueError("flair_slice contains NaN or infinite values")

    row, col = _coordinates(image.shape, cfg.coordinate_mode)
    neighborhood = _box_mean(image, cfg.boundary_mode)
    weighted = _weighted_local_mean(image, cfg.boundary_mode)
    run_features = [
        directional_run_descriptor(
            image,
            direction,
            threshold=cfg.threshold,
            window=cfg.run_window,
            mode=cfg.run_mode,
        )
        for direction in DIRECTIONS
    ]

    stacked = np.stack(
        [row, col, image, neighborhood, weighted, *run_features],
        axis=-1,
    ).astype(np.float32, copy=False)

    if stacked.shape[-1] != 9:
        raise AssertionError(f"Expected 9 features, got {stacked.shape[-1]}")
    return stacked
