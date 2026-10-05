"""ISLES 2022 challenge-style segmentation metrics.

These implementations mirror the definitions published by the official ISLES'22
repository while using NumPy/SciPy APIs that remain current in modern Python.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ISLES22Metrics:
    dice: float
    absolute_volume_difference_ml: float
    absolute_lesion_count_difference: int
    lesion_f1: float
    gt_voxels: int
    pred_voxels: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_bool_pair(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(
            f"ground_truth and prediction must have identical shapes, got {gt.shape} and {pred.shape}"
        )
    if gt.ndim != 3:
        raise ValueError(f"ISLES 2022 challenge metrics expect 3-D masks, got {gt.ndim}-D")
    return gt, pred


def dice_score(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    empty_value: float = 1.0,
) -> float:
    gt, pred = _as_bool_pair(ground_truth, prediction)
    denominator = int(gt.sum()) + int(pred.sum())
    if denominator == 0:
        return float(empty_value)
    intersection = int(np.logical_and(gt, pred).sum())
    return float((2.0 * intersection) / denominator)


def absolute_volume_difference_ml(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    spacing_xyz_mm: Sequence[float],
) -> float:
    gt, pred = _as_bool_pair(ground_truth, prediction)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz_mm must contain three positive values")
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    return float(abs(int(gt.sum()) - int(pred.sum())) * voxel_volume_ml)


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("connectivity must be one of 6, 18, or 26")


def _label(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    labeled, count = ndimage.label(mask, structure=_structure(connectivity))
    return labeled, int(count)


def absolute_lesion_count_difference(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    connectivity: int = 26,
) -> int:
    gt, pred = _as_bool_pair(ground_truth, prediction)
    _, gt_count = _label(gt, connectivity)
    _, pred_count = _label(pred, connectivity)
    return abs(gt_count - pred_count)


def lesion_f1_score(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    empty_value: float = 1.0,
    connectivity: int = 26,
) -> float:
    """Compute the official overlap-based lesion-wise F1 definition.

    A ground-truth connected component is a TP when any of its voxels overlaps
    the prediction; otherwise it is an FN. A predicted connected component with
    no overlap with the ground truth is an FP.
    """
    gt, pred = _as_bool_pair(ground_truth, prediction)
    gt_labels, gt_count = _label(gt, connectivity)
    pred_labels, pred_count = _label(pred, connectivity)

    tp = 0
    fn = 0
    for label_id in range(1, gt_count + 1):
        component = gt_labels == label_id
        if np.logical_and(component, pred).any():
            tp += 1
        else:
            fn += 1

    fp = 0
    for label_id in range(1, pred_count + 1):
        component = pred_labels == label_id
        if not np.logical_and(component, gt).any():
            fp += 1

    if tp + fp + fn == 0:
        return float(empty_value)
    return float(tp / (tp + (fp + fn) / 2.0))


def evaluate_isles2022(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing_xyz_mm: Sequence[float],
    connectivity: int = 26,
) -> ISLES22Metrics:
    gt, pred = _as_bool_pair(ground_truth, prediction)
    return ISLES22Metrics(
        dice=dice_score(gt, pred),
        absolute_volume_difference_ml=absolute_volume_difference_ml(
            gt,
            pred,
            spacing_xyz_mm,
        ),
        absolute_lesion_count_difference=absolute_lesion_count_difference(
            gt,
            pred,
            connectivity=connectivity,
        ),
        lesion_f1=lesion_f1_score(gt, pred, connectivity=connectivity),
        gt_voxels=int(gt.sum()),
        pred_voxels=int(pred.sum()),
    )
