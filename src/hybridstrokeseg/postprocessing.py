"""Morphological post-processing for the historical reconstruction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def disk(radius: int) -> np.ndarray:
    """Return a 2-D disk structuring element."""
    if radius < 0:
        raise ValueError("radius cannot be negative")
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def morphological_close_2d(mask: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Apply dilation followed by erosion, as described by the manuscript."""
    binary = np.asarray(mask, dtype=bool)
    structure = disk(radius)
    dilated = ndimage.binary_dilation(binary, structure=structure)
    return ndimage.binary_erosion(dilated, structure=structure)
