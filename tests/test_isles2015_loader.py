from pathlib import Path

from hybridstrokeseg.data.isles2015 import discover_siss_cases, infer_modality


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_complete_case(case: Path, *, apple_double: bool = False) -> None:
    prefix = "._" if apple_double else ""
    _touch(case / "MR_Flair" / f"{prefix}flair.mha")
    _touch(case / "MR_T1" / f"{prefix}t1.mha")
    _touch(case / "MR_T2" / f"{prefix}t2.mha")
    _touch(case / "MR_DWI" / f"{prefix}dwi.mha")
    _touch(case / "OT" / f"{prefix}ot.mha")


def test_infer_modality_from_challenge_style_names():
    assert infer_modality(Path("VSD.Brain.XX.O.MR_Flair.123/VSD.Brain.XX.O.MR_Flair.123.mha")) == "flair"
    assert infer_modality(Path("VSD.Brain.XX.O.MR_T1.123/a.mha")) == "t1"
    assert infer_modality(Path("VSD.Brain.XX.O.MR_T2.123/a.mha")) == "t2"
    assert infer_modality(Path("VSD.Brain.XX.O.MR_DWI.123/a.mha")) == "dwi"
    assert infer_modality(Path("VSD.Brain_3more.XX.OT.123/a.mha")) == "ot"


def test_discover_complete_case_with_modality_subfolders(tmp_path):
    case = tmp_path / "SISS2015_Training" / "case_01"
    _make_complete_case(case)

    found = discover_siss_cases(tmp_path)

    assert len(found) == 1
    assert found[0].flair.name == "flair.mha"
    assert found[0].ot.name == "ot.mha"
    assert "case_01" in found[0].case_id


def test_incomplete_case_is_ignored(tmp_path):
    case = tmp_path / "case_02"
    _touch(case / "MR_Flair" / "flair.mha")
    _touch(case / "OT" / "ot.mha")

    assert discover_siss_cases(tmp_path) == []


def test_macos_resource_fork_tree_is_ignored(tmp_path):
    real_case = tmp_path / "ISLES2015" / "ISLES2015_SISS" / "training_1"
    fake_case = tmp_path / "__MACOSX" / "ISLES2015" / "ISLES2015_SISS" / "training_1"
    _make_complete_case(real_case)
    _make_complete_case(fake_case, apple_double=True)

    found = discover_siss_cases(tmp_path)

    assert len(found) == 1
    assert "__MACOSX" not in found[0].case_id
    assert all("__MACOSX" not in str(path) for path in found[0].as_dict().values())
