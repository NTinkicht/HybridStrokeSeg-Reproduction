from pathlib import Path

from hybridstrokeseg.data.isles2024 import discover_isles2024_cases


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_discover_isles2024_official_structure_and_anti_leakage(tmp_path: Path) -> None:
    subject = "sub-stroke0001"
    acute_raw = tmp_path / "rawdata" / subject / "ses-01"
    followup_raw = tmp_path / "rawdata" / subject / "ses-02"
    acute_derivative = tmp_path / "derivatives" / subject / "ses-01"
    followup_derivative = tmp_path / "derivatives" / subject / "ses-02"

    _touch(acute_raw / f"{subject}_ses-01_ncct.nii.gz")
    _touch(acute_raw / f"{subject}_ses-01_ctp.nii.gz")
    _touch(followup_raw / f"{subject}_ses-02_dwi.nii.gz")
    _touch(followup_raw / f"{subject}_ses-02_adc.nii.gz")

    _touch(acute_derivative / f"{subject}_ses-01_space-ncct_cta.nii.gz")
    perfusion = acute_derivative / "perfusion-maps"
    for suffix in ("tmax", "cbf", "cbv", "mtt"):
        _touch(perfusion / f"{subject}_ses-01_space-ncct_{suffix}.nii.gz")
    _touch(followup_derivative / f"{subject}_ses-02_lesion-msk.nii.gz")

    cases = discover_isles2024_cases(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == subject

    model_inputs = case.acute_model_inputs()
    assert set(model_inputs) == {"ncct", "cta", "tmax", "cbf", "cbv", "mtt"}
    assert case.dwi_followup not in model_inputs.values()
    assert case.adc_followup not in model_inputs.values()
    assert case.lesion_mask_ncct not in model_inputs.values()


def test_discover_ncct_only_selective_extraction(tmp_path: Path) -> None:
    subject = "sub-stroke0002"
    raw = tmp_path / "rawdata" / subject / "ses-01"
    followup = tmp_path / "derivatives" / subject / "ses-02"

    _touch(raw / f"{subject}_ses-01_ncct.nii.gz")
    _touch(followup / f"{subject}_ses-02_lesion-msk.nii.gz")

    cases = discover_isles2024_cases(tmp_path, required_channels=("ncct",))
    assert [case.case_id for case in cases] == [subject]
    assert set(cases[0].acute_model_inputs()) == {"ncct"}


def test_discover_isles2024_resolves_one_archive_wrapper_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "train"
    subject = "sub-stroke0003"
    raw = dataset / "raw_data" / subject / "ses-01"
    followup = dataset / "derivatives" / subject / "ses-02"

    _touch(raw / f"{subject}_ses-01_ncct.nii.gz")
    _touch(followup / f"{subject}_ses-02_lesion-msk.nii.gz")

    cases = discover_isles2024_cases(tmp_path, required_channels=("ncct",))
    assert [case.case_id for case in cases] == [subject]


def test_discover_isles2024_accepts_legacy_long_session_labels(tmp_path: Path) -> None:
    subject = "sub-stroke0099"
    raw = tmp_path / "rawdata" / subject / "ses-0001"
    followup = tmp_path / "derivatives" / subject / "ses-0002"

    _touch(raw / f"{subject}_ses-0001_ncct.nii.gz")
    _touch(followup / f"{subject}_ses-0002_space-ncct_lesion-msk.nii.gz")

    cases = discover_isles2024_cases(tmp_path, required_channels=("ncct",))
    assert [case.case_id for case in cases] == [subject]
