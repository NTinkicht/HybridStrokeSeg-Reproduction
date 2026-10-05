from pathlib import Path

from hybridstrokeseg.data.isles2022 import discover_isles2022_cases, geometry_equal


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_discover_isles2022_case(tmp_path: Path) -> None:
    subject = "sub-strokecase0009"
    session = "ses-0001"
    raw = tmp_path / "rawdata" / subject / session
    derivatives = tmp_path / "derivatives" / subject / session

    _touch(raw / f"{subject}_{session}_adc.nii.gz")
    _touch(raw / f"{subject}_{session}_dwi.nii.gz")
    _touch(raw / f"{subject}_{session}_flair.nii.gz")
    _touch(derivatives / f"{subject}_{session}_msk.nii.gz")

    cases = discover_isles2022_cases(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == f"{subject}_{session}"
    assert case.adc.name.endswith("_adc.nii.gz")
    assert case.dwi.name.endswith("_dwi.nii.gz")
    assert case.flair.name.endswith("_flair.nii.gz")
    assert case.mask.name.endswith("_msk.nii.gz")


def test_geometry_equal_requires_same_grid() -> None:
    base = {
        "size_xyz": (128, 128, 32),
        "spacing_xyz": (1.0, 1.0, 4.0),
        "origin_xyz": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }
    same = dict(base)
    shifted = dict(base)
    shifted["origin_xyz"] = (0.0, 0.0, 1.0)
    resized = dict(base)
    resized["size_xyz"] = (128, 128, 31)

    assert geometry_equal(base, same)
    assert not geometry_equal(base, shifted)
    assert not geometry_equal(base, resized)
