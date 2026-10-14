"""Regression tests for crash-safe OOF progress and cache invalidation."""

from pathlib import Path

import pytest

from hybridstrokeseg.isles2024_oof_progress import (
    input_signature,
    load_case,
    record_progress,
    save_case,
)


def test_case_cache_roundtrip_and_input_change_invalidates(tmp_path: Path):
    prediction = tmp_path / "prediction.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    prediction.write_bytes(b"pred")
    reference.write_bytes(b"ref")
    signature = input_signature("case-a", 0, reference, prediction)
    row = {"case_id": "case-a", "fold": 0, "dice": 0.25}
    cache = tmp_path / "case.json"
    save_case(cache, signature, row)
    assert load_case(cache, signature) == row
    assert not cache.with_name(cache.name + ".tmp").exists()

    prediction.write_bytes(b"changed")
    new_signature = input_signature("case-a", 0, reference, prediction)
    assert load_case(cache, new_signature) is None


def test_case_cache_fail_closed_on_wrong_identity(tmp_path: Path):
    reference = tmp_path / "ref.nii.gz"
    prediction = tmp_path / "pred.nii.gz"
    reference.write_bytes(b"ref")
    prediction.write_bytes(b"pred")
    signature = input_signature("case-b", 2, reference, prediction)
    with pytest.raises(ValueError, match="mismatched"):
        save_case(tmp_path / "bad.json", signature, {"case_id": "case-b", "fold": 3})


def test_progress_tracks_percent_status_and_failure(tmp_path: Path):
    output = tmp_path / "status.json"
    state = record_progress(
        output, completed=3, total=149, fold=0, case_id="case-a", status="running"
    )
    assert state["percent"] == 2.01
    assert state["completed_cases"] == 3
    assert state["status"] == "running"
    assert output.exists()
    failed = record_progress(
        output, completed=3, total=149, fold=0, case_id="case-b",
        status="failed", detail="example exception",
    )
    assert failed["status"] == "failed"
    assert failed["detail"] == "example exception"


def test_progress_rejects_impossible_counts(tmp_path: Path):
    with pytest.raises(ValueError, match="between 0 and total"):
        record_progress(
            tmp_path / "bad.json", completed=150, total=149,
            fold=None, case_id=None, status="running",
        )
