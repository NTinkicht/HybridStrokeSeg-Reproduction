import numpy as np
import pytest

from hybridstrokeseg.diagnostics import histogram_match_values, random_pixel_partition


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


def test_histogram_match_values_maps_empirical_quantiles():
    source = np.array([0, 0, 1, 1], dtype=float)
    reference = np.array([10, 10, 20, 20], dtype=float)
    matched = histogram_match_values(source, reference)
    assert matched.dtype == np.float32
    assert np.array_equal(matched, np.array([10, 10, 20, 20], dtype=np.float32))


def test_histogram_match_values_rejects_empty_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        histogram_match_values(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError):
        histogram_match_values(np.array([1.0]), np.array([np.nan]))
