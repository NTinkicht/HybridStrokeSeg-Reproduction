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
    ids = sorted(str(case_id) for case_id in case_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("case_ids must be unique")
    if len(ids) < 2:
        raise ValueError("At least two cases are required")
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
