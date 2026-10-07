from pathlib import Path

from hybridstrokeseg.data.isles2024 import discover_isles2024_cases


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_discover_isles2024_case_and_separates_followup_from_inputs(tmp_path: Path) -> None:
    subject = "sub-strokecase0001"
    raw = tmp_path / "raw_data" / subject / "ses-0001"
    acute = tmp_path / "derivatives" / subject / "ses-0001"
    followup = tmp_path / "derivatives" / subject / "ses-0002"

    _touch(raw / f"{subject}_ses-0001_ncct.nii.gz")
    _touch(raw / f"{subject}_ses-0001_ctp.nii.gz")

    _touch(acute / f"{subject}_ses-0001_space-ncct_cta.nii.gz")
    _touch(acute / "perfusion-maps" / f"{subject}_ses-0001_space-ncct_tmax.nii.gz")
    _touch(acute / "perfusion-maps" / f"{subject}_ses-0001_space-ncct_cbf.nii.gz")
    _touch(acute / "perfusion-maps" / f"{subject}_ses-0001_space-ncct_cbv.nii.gz")
    _touch(acute / "perfusion-maps" / f"{subject}_ses-0001_space-ncct_mtt.nii.gz")
    _touch(acute / f"{subject}_ses-0001_space-ncct_lvo-msk.nii.gz")
    _touch(acute / f"{subject}_ses-0001_space-ncct_cow-msk.nii.gz")

    _touch(followup / f"{subject}_ses-0002_space-ncct_dwi.nii.gz")
    _touch(followup / f"{subject}_ses-0002_space-ncct_adc.nii.gz")
    _touch(followup / f"{subject}_ses-0002_space-ncct_lesion-msk.nii.gz")

    cases = discover_isles2024_cases(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == subject

    model_inputs = case.acute_model_inputs()
    assert set(model_inputs) == {"ncct", "cta", "tmax", "cbf", "cbv", "mtt"}
    assert case.dwi_followup_ncct not in model_inputs.values()
    assert case.adc_followup_ncct not in model_inputs.values()
    assert case.lesion_mask_ncct not in model_inputs.values()


def test_discover_isles2024_accepts_rawdata_spelling(tmp_path: Path) -> None:
    subject = "sub-strokecase0002"
    raw = tmp_path / "rawdata" / subject / "ses-0001"
    acute = tmp_path / "derivatives" / subject / "ses-0001"
    followup = tmp_path / "derivatives" / subject / "ses-0002"

    _touch(raw / f"{subject}_ses-0001_ncct.nii.gz")
    for suffix in ("cta", "tmax", "cbf", "cbv", "mtt"):
        _touch(acute / f"{subject}_ses-0001_space-ncct_{suffix}.nii.gz")
    for suffix in ("dwi", "adc", "lesion-msk"):
        _touch(followup / f"{subject}_ses-0002_space-ncct_{suffix}.nii.gz")

    cases = discover_isles2024_cases(tmp_path)
    assert [case.case_id for case in cases] == [subject]
