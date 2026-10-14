from types import SimpleNamespace

import numpy as np
import pytest

from hybridstrokeseg.isles2024_eval import (
    absolute_volume_difference_ml,
    bootstrap_mean_ci,
    load_binary_mask,
    validate_oof_splits,
)


def test_validate_oof_splits_requires_unique_complete_coverage():
    splits = [
        {"train": ["b", "c"], "val": ["a"]},
        {"train": ["a", "c"], "val": ["b"]},
        {"train": ["a", "b"], "val": ["c"]},
    ]
    assert validate_oof_splits(splits, {"a", "b", "c"}) == ["a", "b", "c"]


def test_validate_oof_splits_rejects_duplicate_validation_case():
    splits = [
        {"train": ["b"], "val": ["a"]},
        {"train": ["b"], "val": ["a"]},
    ]
    with pytest.raises(ValueError, match="duplicated"):
        validate_oof_splits(splits, {"a", "b"})


def test_absolute_volume_difference_is_reported_in_ml():
    reference = np.zeros((2, 2, 2), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference.flat[:3] = 1
    prediction.flat[:1] = 1

    # Two-voxel difference * 0.125 ml/voxel = 0.25 ml.
    assert absolute_volume_difference_ml(reference, prediction, 0.125) == pytest.approx(0.25)


def test_bootstrap_mean_ci_is_deterministic():
    values = np.array([0.0, 0.25, 0.5, 1.0])
    first = bootstrap_mean_ci(values, repetitions=500, seed=2026)
    second = bootstrap_mean_ci(values, repetitions=500, seed=2026)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_validate_oof_splits_rejects_patient_train_validation_leakage():
    splits = [
        {"train": ["a", "b"], "val": ["a"]},
        {"train": ["a"], "val": ["b"]},
    ]
    with pytest.raises(ValueError, match="train/validation patient leakage"):
        validate_oof_splits(splits, {"a", "b"})


def test_validate_oof_splits_rejects_incomplete_training_partition():
    splits = [
        {"train": ["b"], "val": ["a"]},
        {"train": ["a", "c"], "val": ["b"]},
        {"train": ["a", "b"], "val": ["c"]},
    ]
    with pytest.raises(ValueError, match="training partition is incomplete"):
        validate_oof_splits(splits, {"a", "b", "c"})


def test_validate_oof_splits_rejects_duplicate_training_subject():
    splits = [
        {"train": ["b", "b"], "val": ["a"]},
        {"train": ["a"], "val": ["b"]},
    ]
    with pytest.raises(ValueError, match="duplicate training case IDs"):
        validate_oof_splits(splits, {"a", "b"})


def test_load_binary_mask_decodes_nonzero_voxels_without_float64_cache():
    image = SimpleNamespace(dataobj=np.array([[[0, 1], [-2, 0]]], dtype=np.int16))
    mask = load_binary_mask(image)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [[[False, True], [False, False]]]
