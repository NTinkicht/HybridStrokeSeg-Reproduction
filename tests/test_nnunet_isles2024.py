from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from hybridstrokeseg.data.isles2022 import geometry_signature
from hybridstrokeseg.nnunet_isles2024 import (
    _materialize_label_on_reference,
    build_isles2024_dataset_json,
)


def test_isles2024_dataset_json_accepts_only_acute_channels():
    metadata = build_isles2024_dataset_json(
        channels=("ncct", "tmax", "cbf"),
        num_training=149,
    )
    assert metadata["numTraining"] == 149
    assert metadata["channel_names"] == {
        "0": "NCCT",
        "1": "TMAX",
        "2": "CBF",
    }


@pytest.mark.parametrize(
    "channel",
    ("dwi", "adc", "lesion_mask", "outcome"),
)
def test_isles2024_dataset_json_rejects_followup_or_label_channels(channel):
    with pytest.raises(ValueError):
        build_isles2024_dataset_json(
            channels=("ncct", channel),
            num_training=149,
        )


def test_materialize_label_on_reference_copies_header_without_resampling(tmp_path: Path):
    reference_path = tmp_path / "ncct.nii.gz"
    source_path = tmp_path / "mask.nii.gz"
    destination_path = tmp_path / "staged_mask.nii.gz"

    reference_array = np.zeros((3, 4, 5), dtype=np.int16)
    mask_array = np.zeros((3, 4, 5), dtype=np.uint8)
    mask_array[1, 2, 3] = 1

    reference = sitk.GetImageFromArray(reference_array)
    reference.SetSpacing((0.34, 0.34, 4.0))
    reference.SetOrigin((97.0, 335.0, -704.0))
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(reference, str(reference_path))

    mask = sitk.GetImageFromArray(mask_array)
    mask.SetSpacing((0.34, 0.34, 4.0))
    mask.SetOrigin((97.0, 335.0, -704.0))
    mask.SetDirection((1.0, 0.0, 1.5e-5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(mask, str(source_path))

    _materialize_label_on_reference(
        source_path,
        reference_path,
        destination_path,
    )

    staged = sitk.GetArrayFromImage(sitk.ReadImage(str(destination_path)))
    assert np.array_equal(staged, mask_array)
    assert geometry_signature(destination_path) == geometry_signature(reference_path)
