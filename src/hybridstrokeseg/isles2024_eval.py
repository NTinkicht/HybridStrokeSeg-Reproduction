"""ISLES'24 out-of-fold evaluation helpers.

The challenge's public evaluator computes four case-level metrics:
Dice, absolute volume difference, lesion-wise F1, and absolute lesion-count
difference. Lesion matching uses connected components and Panoptica with a
0.2 matching threshold.

Reference implementation:
https://github.com/ezequieldlrosa/isles24/blob/main/utils/eval_utils.py

This module keeps the challenge semantics while adding explicit geometry,
cohort, and reproducibility checks for our 149-case training-set OOF study.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OFFICIAL_EVAL_SOURCE = (
    "https://github.com/ezequieldlrosa/isles24/blob/"
    "6b4dea6f07bf1717cca638ea1fe9345a47c3cd41/utils/eval_utils.py"
)
PANOPTICA_MATCHING_THRESHOLD = 0.2


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    fold: int
    dice: float
    absolute_volume_difference_ml: float
    lesion_f1: float
    absolute_lesion_count_difference: int
    prediction_volume_ml: float
    reference_volume_ml: float
    prediction_lesion_count: int
    reference_lesion_count: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_splits(path: Path) -> list[dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty split list in {path}")
    normalized: list[dict[str, list[str]]] = []
    for index, split in enumerate(payload):
        if not isinstance(split, dict):
            raise TypeError(f"Split {index} is not an object")
        train = [str(case) for case in split.get("train", [])]
        val = [str(case) for case in split.get("val", [])]
        if not val:
            raise ValueError(f"Split {index} has no validation cases")
        normalized.append({"train": train, "val": val})
    return normalized


def validate_oof_splits(
    splits: list[dict[str, list[str]]],
    expected_cases: Iterable[str] | None = None,
) -> list[str]:
    validation_cases = [case for split in splits for case in split["val"]]
    unique = set(validation_cases)
    if len(validation_cases) != len(unique):
        duplicates = sorted(
            case for case in unique if validation_cases.count(case) > 1
        )
        raise ValueError(f"Validation cases are duplicated across folds: {duplicates}")

    train_cases = {case for split in splits for case in split["train"]}
    if expected_cases is None:
        expected = train_cases | unique
    else:
        expected = {str(case) for case in expected_cases}

    missing = sorted(expected - unique)
    extra = sorted(unique - expected)
    if missing or extra:
        raise ValueError(
            f"OOF validation coverage mismatch. Missing={missing}, extra={extra}"
        )
    # Patient-level validation must also ensure each fold is a genuine
    # partition: every held-out subject is absent from the training list,
    # and every other expected subject is present exactly once.
    for fold_index, split in enumerate(splits):
        train_ids = split["train"]
        val_ids = split["val"]
        train_set = set(train_ids)
        val_set = set(val_ids)
        if len(train_ids) != len(train_set):
            raise ValueError(f"Fold {fold_index} has duplicate training case IDs")
        overlap = sorted(train_set & val_set)
        if overlap:
            raise ValueError(
                f"Fold {fold_index} has train/validation patient leakage: {overlap}"
            )
        expected_train = expected - val_set
        if train_set != expected_train:
            raise ValueError(
                f"Fold {fold_index} training partition is incomplete or unexpected: "
                f"missing={sorted(expected_train - train_set)}, "
                f"extra={sorted(train_set - expected_train)}"
            )

    return sorted(unique)


def load_binary_mask(image) -> np.ndarray:
    """Decode a NIfTI proxy directly to bool without a float64 get_fdata cache.

    nibabel proxies apply any on-disk scaling during np.asanyarray, so the
    foreground definition remains exactly the original strictly-positive threshold.
    """
    return np.asanyarray(image.dataobj) > 0


def absolute_volume_difference_ml(
    reference: np.ndarray,
    prediction: np.ndarray,
    voxel_volume_ml: float,
) -> float:
    reference_bool = np.asarray(reference).astype(bool)
    prediction_bool = np.asarray(prediction).astype(bool)
    if reference_bool.shape != prediction_bool.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference_bool.shape}, "
            f"prediction={prediction_bool.shape}"
        )
    return float(
        abs(
            reference_bool.sum(dtype=np.int64) * float(voxel_volume_ml)
            - prediction_bool.sum(dtype=np.int64) * float(voxel_volume_ml)
        )
    )


def official_dice_f1_and_lesion_difference(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    empty_value: float = 1.0,
) -> tuple[float, int, float, int, int]:
    """Match the public ISLES'24 Panoptica evaluator.

    Returns lesion_f1, absolute_lesion_count_difference, dice,
    reference_lesion_count, prediction_lesion_count.
    """
    try:
        from panoptica import (
            ConnectedComponentsInstanceApproximator,
            InputType,
            NaiveThresholdMatching,
            Panoptica_Evaluator,
        )
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Panoptica is required for ISLES'24 lesion-wise evaluation. "
            "Install the project with: pip install -e '.[evaluation]'"
        ) from exc

    reference_int = np.asarray(reference).astype(int)
    prediction_int = np.asarray(prediction).astype(int)
    if reference_int.shape != prediction_int.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference_int.shape}, "
            f"prediction={prediction_int.shape}"
        )

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(
            matching_threshold=PANOPTICA_MATCHING_THRESHOLD
        ),
    )
    result_payload = evaluator.evaluate(
        prediction_int,
        reference_int,
        verbose=False,
    )["ungrouped"]
    # The organizer's Panoptica API returned (result, diagnostics); the
    # compatible 1.x release returns the result object directly. Neither
    # variant changes the metrics or matching threshold.
    result = result_payload[0] if isinstance(result_payload, tuple) else result_payload

    reference_count = int(result.num_ref_instances)
    prediction_count = int(result.num_pred_instances)
    count_difference = abs(reference_count - prediction_count)

    if reference_count == 0 and prediction_count == 0:
        lesion_f1 = float(empty_value)
        dice = float(empty_value)
    else:
        lesion_f1 = float(result.rq)
        dice = float(result.global_bin_dsc)

    return lesion_f1, count_difference, dice, reference_count, prediction_count


def apply_brain_mask(
    reference: np.ndarray,
    prediction: np.ndarray,
    brain_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    reference_bool = np.asarray(reference).astype(bool)
    prediction_bool = np.asarray(prediction).astype(bool)
    if reference_bool.shape != prediction_bool.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference_bool.shape}, "
            f"prediction={prediction_bool.shape}"
        )
    if brain_mask is None:
        return reference_bool, prediction_bool

    brain_bool = np.asarray(brain_mask).astype(bool)
    if brain_bool.shape != reference_bool.shape:
        raise ValueError(
            f"Brain-mask shape mismatch: mask={brain_bool.shape}, "
            f"reference={reference_bool.shape}"
        )
    return reference_bool & brain_bool, prediction_bool & brain_bool


def evaluate_case(
    *,
    case_id: str,
    fold: int,
    reference: np.ndarray,
    prediction: np.ndarray,
    voxel_volume_ml: float,
    brain_mask: np.ndarray | None = None,
) -> CaseMetrics:
    reference_bool, prediction_bool = apply_brain_mask(
        reference,
        prediction,
        brain_mask,
    )
    lesion_f1, lesion_diff, dice, ref_count, pred_count = (
        official_dice_f1_and_lesion_difference(
            reference_bool,
            prediction_bool,
        )
    )
    avd = absolute_volume_difference_ml(
        reference_bool,
        prediction_bool,
        voxel_volume_ml,
    )
    reference_volume = float(reference_bool.sum(dtype=np.int64) * voxel_volume_ml)
    prediction_volume = float(prediction_bool.sum(dtype=np.int64) * voxel_volume_ml)

    return CaseMetrics(
        case_id=case_id,
        fold=fold,
        dice=dice,
        absolute_volume_difference_ml=avd,
        lesion_f1=lesion_f1,
        absolute_lesion_count_difference=lesion_diff,
        prediction_volume_ml=prediction_volume,
        reference_volume_ml=reference_volume,
        prediction_lesion_count=pred_count,
        reference_lesion_count=ref_count,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty 1D array")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[samples].mean(axis=1)
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)
