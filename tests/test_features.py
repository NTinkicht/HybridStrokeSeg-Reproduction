import numpy as np
import pytest

from hybridstrokeseg.features import (
    FeatureConfig,
    directional_threshold_count,
    extract_nine_features,
    feature_names,
)


def test_feature_names_and_shape():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    features = extract_nine_features(image, FeatureConfig(run_window=3))

    assert features.shape == (5, 5, 9)
    assert len(feature_names()) == 9
    np.testing.assert_array_equal(features[..., 2], image)


def test_raw_coordinates_are_row_and_column_indices():
    image = np.zeros((3, 4), dtype=np.float32)
    features = extract_nine_features(image, FeatureConfig(run_window=3))

    assert features[2, 1, 0] == 2
    assert features[2, 1, 1] == 1


def test_normalized_coordinates():
    image = np.zeros((3, 5), dtype=np.float32)
    features = extract_nine_features(
        image,
        FeatureConfig(run_window=3, coordinate_mode="normalized"),
    )

    assert features[2, 4, 0] == pytest.approx(1.0)
    assert features[2, 4, 1] == pytest.approx(1.0)


def test_directional_threshold_count_handles_edges_without_wraparound():
    image = np.zeros((3, 5), dtype=np.float32)
    image[1, 0:3] = 20.0

    counts = directional_threshold_count(
        image,
        (0, 1),
        threshold=20.0,
        window=3,
    )

    assert counts[1, 0] == 2
    assert counts[1, 1] == 3
    assert counts[1, 2] == 2
    assert counts[1, 4] == 0


def test_even_run_window_is_rejected():
    with pytest.raises(ValueError):
        FeatureConfig(run_window=24)
