import numpy as np

from hybridstrokeseg.postprocessing import disk, morphological_close_2d


def test_radius_zero_is_noop() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[3, 3] = False

    result = morphological_close_2d(mask, radius=0)

    assert np.array_equal(result, mask)


def test_closing_fills_single_pixel_hole() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[3, 3] = False

    result = morphological_close_2d(mask, radius=1)

    assert result[3, 3]


def test_disk_radius_two_has_expected_extent() -> None:
    structure = disk(2)

    assert structure.shape == (5, 5)
    assert structure[2, 2]
    assert structure[0, 2]
    assert structure[2, 0]
    assert not structure[0, 0]
