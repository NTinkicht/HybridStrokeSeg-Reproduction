"""Deterministic patient-level splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatientSplit:
    """A leakage-safe patient-level train/test split."""

    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class PatientFold:
    """One deterministic patient-level cross-validation fold."""

    fold: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    seed: int


def _validate_ids(case_ids: list[str] | tuple[str, ...]) -> list[str]:
    ids = sorted(str(case_id) for case_id in case_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("case_ids must be unique")
    if len(ids) < 2:
        raise ValueError("At least two cases are required")
    return ids


def make_patient_split(
    case_ids: list[str] | tuple[str, ...],
    *,
    train_size: int | None = None,
    test_fraction: float = 0.30,
    seed: int = 2026,
) -> PatientSplit:
    """Create a deterministic patient-level split.

    For the 28-case historical SISS cohort, callers can request ``train_size=19``
    to mirror the manuscript's stated 19/9 partition. No voxel from a test patient
    can enter the training set.
    """
    ids = _validate_ids(case_ids)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie between 0 and 1")

    if train_size is None:
        test_size = max(1, round(len(ids) * test_fraction))
        train_size = len(ids) - test_size
    if not 1 <= train_size < len(ids):
        raise ValueError("train_size must leave at least one train and one test case")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ids, dtype=object)[rng.permutation(len(ids))]
    train = tuple(str(value) for value in shuffled[:train_size])
    test = tuple(str(value) for value in shuffled[train_size:])
    return PatientSplit(train_ids=train, test_ids=test, seed=seed)


def make_patient_kfolds(
    case_ids: list[str] | tuple[str, ...],
    *,
    n_splits: int = 5,
    seed: int = 2026,
) -> tuple[PatientFold, ...]:
    """Create deterministic, exhaustive patient-level cross-validation folds.

    Each patient appears in exactly one validation fold and never appears in the
    corresponding training partition. Fold sizes differ by at most one patient.
    """
    ids = _validate_ids(case_ids)
    if not 2 <= n_splits <= len(ids):
        raise ValueError("n_splits must be between 2 and the number of cases")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ids, dtype=object)[rng.permutation(len(ids))]
    validation_parts = np.array_split(shuffled, n_splits)
    folds: list[PatientFold] = []

    for fold_index, validation_part in enumerate(validation_parts):
        validation = tuple(str(value) for value in validation_part)
        validation_set = set(validation)
        train = tuple(str(value) for value in shuffled if str(value) not in validation_set)
        folds.append(
            PatientFold(
                fold=fold_index,
                train_ids=train,
                validation_ids=validation,
                seed=seed,
            )
        )

    return tuple(folds)
