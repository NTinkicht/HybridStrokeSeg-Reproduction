from hybridstrokeseg.nnunet import build_dataset_json, build_splits_final


def test_build_dataset_json_matches_nnunet_v2_contract():
    payload = build_dataset_json(channels=("dwi", "adc", "flair"), num_training=250)
    assert payload == {
        "channel_names": {"0": "DWI", "1": "ADC", "2": "FLAIR"},
        "labels": {"background": 0, "stroke": 1},
        "numTraining": 250,
        "file_ending": ".nii.gz",
    }


def test_build_splits_final_is_exhaustive_and_deterministic():
    ids = [f"case_{index:03d}" for index in range(250)]
    splits_a = build_splits_final(ids, n_splits=5, seed=19)
    splits_b = build_splits_final(ids, n_splits=5, seed=19)
    assert splits_a == splits_b
    assert len(splits_a) == 5

    all_validation: list[str] = []
    for split in splits_a:
        assert set(split) == {"train", "val"}
        assert len(split["train"]) == 200
        assert len(split["val"]) == 50
        assert set(split["train"]).isdisjoint(split["val"])
        all_validation.extend(split["val"])

    assert sorted(all_validation) == sorted(ids)
