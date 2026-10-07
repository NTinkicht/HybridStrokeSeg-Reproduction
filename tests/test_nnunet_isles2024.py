import pytest

from hybridstrokeseg.nnunet_isles2024 import build_isles2024_dataset_json


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
