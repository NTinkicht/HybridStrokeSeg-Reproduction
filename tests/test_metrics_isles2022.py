import numpy as np

from hybridstrokeseg.metrics_isles2022 import (
    absolute_lesion_count_difference,
    absolute_volume_difference_ml,
    dice_score,
    evaluate_isles2022,
    lesion_f1_score,
)


def test_empty_masks_follow_challenge_convention() -> None:
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    pred = np.zeros_like(gt)
    assert dice_score(gt, pred) == 1.0
    assert lesion_f1_score(gt, pred) == 1.0
    assert absolute_lesion_count_difference(gt, pred) == 0
    assert absolute_volume_difference_ml(gt, pred, (1.0, 1.0, 1.0)) == 0.0


def test_lesion_f1_matches_overlap_definition() -> None:
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[1, 1, 1] = 1
    gt[6, 6, 6] = 1

    pred[1, 1, 1] = 1  # TP lesion
    pred[1, 1, 2] = 1  # same predicted component as TP
    pred[4, 4, 4] = 1  # FP lesion

    # tp=1, fn=1, fp=1 -> 1 / (1 + (1+1)/2) = 0.5
    assert lesion_f1_score(gt, pred) == 0.5
    assert absolute_lesion_count_difference(gt, pred) == 0


def test_volume_difference_uses_mm3_to_ml_conversion() -> None:
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[1, 1, 1] = 1
    spacing_xyz_mm = (2.0, 2.0, 5.0)
    assert absolute_volume_difference_ml(gt, pred, spacing_xyz_mm) == 0.02


def test_evaluate_isles2022_returns_complete_record() -> None:
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[1, 1, 1] = 1
    pred[1, 1, 1] = 1

    metrics = evaluate_isles2022(gt, pred, spacing_xyz_mm=(1.0, 1.0, 1.0))
    assert metrics.dice == 1.0
    assert metrics.absolute_volume_difference_ml == 0.0
    assert metrics.absolute_lesion_count_difference == 0
    assert metrics.lesion_f1 == 1.0
    assert metrics.gt_voxels == 1
    assert metrics.pred_voxels == 1
