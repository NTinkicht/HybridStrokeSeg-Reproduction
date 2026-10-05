"""Segmentation metrics used by the historical reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class SegmentationMetrics:
    """Per-case binary segmentation metrics."""

    dice: float
    precision: float
    recall: float
    auc: float
    gt_voxels: int
    pred_voxels: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_binary_segmentation(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    scores: np.ndarray | None = None,
) -> SegmentationMetrics:
    """Evaluate binary predictions with explicit empty-mask behavior."""
    gt = np.asarray(ground_truth, dtype=bool).reshape(-1)
    pred = np.asarray(prediction, dtype=bool).reshape(-1)
    if gt.size != pred.size:
        raise ValueError("ground_truth and prediction must have the same size")

    tp = int(np.count_nonzero(gt & pred))
    gt_count = int(np.count_nonzero(gt))
    pred_count = int(np.count_nonzero(pred))
    dice_denominator = gt_count + pred_count
    dice = 1.0 if dice_denominator == 0 else (2.0 * tp) / dice_denominator
    precision = 1.0 if pred_count == 0 and gt_count == 0 else tp / max(pred_count, 1)
    recall = 1.0 if gt_count == 0 and pred_count == 0 else tp / max(gt_count, 1)

    auc = float("nan")
    if scores is not None and np.unique(gt).size == 2:
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_array.size != gt.size:
            raise ValueError("scores must align with ground_truth")
        auc = float(roc_auc_score(gt.astype(np.uint8), score_array))

    return SegmentationMetrics(
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        auc=auc,
        gt_voxels=gt_count,
        pred_voxels=pred_count,
    )
