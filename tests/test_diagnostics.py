import numpy as np
import pytest

from hybridstrokeseg.diagnostics import random_pixel_partition


def test_random_pixel_partition_is_deterministic_disjoint_and_exhaustive():
    first = random_pixel_partition(
        100,
        train_fraction=0.70,
        validation_fraction=0.15,
        seed=2026,
    )
    second = random_pixel_partition(
        100,
        train_fraction=0.70,
        validation_fraction=0.15,
        seed=2026,
    )

    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    assert len(first.train) == 70
    assert len(first.validation) == 15
    assert len(first.test) == 15

    combined = np.concatenate([first.train, first.validation, first.test])
    assert np.array_equal(np.sort(combined), np.arange(100))
    assert len(np.unique(combined)) == 100


def test_random_pixel_partition_supports_70_30():
    partition = random_pixel_partition(
        101,
        train_fraction=0.70,
        validation_fraction=0.0,
        seed=7,
    )
    assert len(partition.train) == 70
    assert len(partition.validation) == 0
    assert len(partition.test) == 31


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.7, -0.1),
        (0.7, 0.3),
    ],
)
def test_random_pixel_partition_rejects_invalid_fractions(train_fraction, validation_fraction):
    with pytest.raises(ValueError):
        random_pixel_partition(
            10,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=1,
        )
