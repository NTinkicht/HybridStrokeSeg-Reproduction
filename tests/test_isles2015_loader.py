from pathlib import Path

from hybridstrokeseg.data.isles2015 import discover_siss_cases, infer_modality


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_infer_modality_from_challenge_style_names():
    assert infer_modality(Path("VSD.Brain.XX.O.MR_Flair.123/VSD.Brain.XX.O.MR_Flair.123.mha")) == "flair"
    assert infer_modality(Path("VSD.Brain.XX.O.MR_T1.123/a.mha")) == "t1"
    assert infer_modality(Path("VSD.Brain.XX.O.MR_T2.123/a.mha")) == "t2"
    assert infer_modality(Path("VSD.Brain.XX.O.MR_DWI.123/a.mha")) == "dwi"
    assert infer_modality(Path("VSD.Brain_3more.XX.OT.123/a.mha")) == "ot"


def test_discover_complete_case_with_modality_subfolders(tmp_path):
    case = tmp_path / "SISS2015_Training" / "case_01"

    _touch(case / "MR_Flair" / "flair.mha")
    _touch(case / "MR_T1" / "t1.mha")
    _touch(case / "MR_T2" / "t2.mha")
    _touch(case / "MR_DWI" / "dwi.mha")
    _touch(case / "OT" / "ot.mha")

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
