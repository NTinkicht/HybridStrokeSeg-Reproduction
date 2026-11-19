"""Read-only preprocessed Dataset505 tensor audit regression tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

AUDIT = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "audit_isles2024_preprocessed_tensors.py")
)


def test_blockwise_full_scan_counts_extrema_and_nonfinite():
    volume = np.zeros((5, 5, 3, 4), dtype=np.float32)
    volume[4] = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
    volume[4, 0, 0, 0] = np.nan
    volume[4, 1, 1, 1] = -np.inf
    volume[4, 2, 2, 3] = np.inf
    volume[4, 3, 0, 2] = -1_000.5
    volume[4, 4, 2, 0] = 1_001.2
    unchanged = volume.copy()
    full = AUDIT["summarize_channel"](volume, 4, slices_per_block=2, sample_max=1000)
    assert full["n_voxels"] == 60
    assert full["n_nonfinite"] == 3
    assert full["n_nan"] == 1
    assert full["n_posinf"] == 1
    assert full["n_neginf"] == 1
    assert full["n_abs_gt_1000"] == 2
    assert full["n_abs_gt_100"] == 2
    assert full["min"] == -1000.5
    assert full["max"] == pytest.approx(1001.2, abs=1e-3)
    assert full["near_constant"] is False
    assert full["mean"] == pytest.approx(float(np.nanmean(volume[4][np.isfinite(volume[4])])), rel=1e-6)
    np.testing.assert_array_equal(unchanged, volume)


def test_unchanged_channel_flagged_near_constant():
    tensor = np.ones((5, 3, 2, 2), dtype=np.float32)
    result = AUDIT["summarize_channel"](tensor, 2, slices_per_block=1)
    assert result["near_constant"] is True
    assert result["n_nonfinite"] == 0
    assert result["mean"] == 1
    assert result["std"] == 0
    assert result["n_abs_gt_6"] == 0


def test_wrong_channel_count_fails_closed():
    with pytest.raises(ValueError, match="Expected"):
        AUDIT["summarize_channel"](np.zeros((4, 2, 2, 2), dtype=np.float32), 2)


def test_149_case_resumable_output_and_raw_comparison(tmp_path, monkeypatch):
    # Execute the real driver against synthetic minimal inputs, without reading
    # real clinical data or requiring Blosc2 on CI runners.
    dataset = tmp_path / "Dataset505_ISLES2024_NCCT_PERFUSION"
    config = dataset / "nnUNetPlans_3d_fullres"
    config.mkdir(parents=True)
    (dataset / "dataset.json").write_text(
        json.dumps({"channel_names": {str(i): n for i, n in enumerate(AUDIT["CHANNELS"])}})
    )
    (dataset / "nnUNetPlans.json").write_text(
        json.dumps({"configurations": {"3d_fullres": {
            "normalization_schemes": ["ZScoreNormalization"] * 5
        }}})
    )
    cases = [f"sub_stroke{i:04d}" for i in range(1, 149)] + ["sub_stroke0152"]
    folds = []
    offsets = [0, 30, 60, 90, 120, 149]
    for i in range(5):
        val = cases[offsets[i]:offsets[i + 1]]
        folds.append({"train": [p for p in cases if p not in val], "val": val})
    split_file = tmp_path / "splits_final.json"
    split_file.write_text(json.dumps(folds))
    AUDIT["frozen_cases"].__globals__["FROZEN_SPLIT_SHA"] = hashlib.sha256(split_file.read_bytes()).hexdigest()
    for case in cases:
        for ending in (".b2nd", "_seg.b2nd", ".pkl"):
            (config / f"{case}{ending}").write_bytes(b"synthetic placeholder")
    raw = tmp_path / "raw_audit.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "case_id", "channel", "min", "max", "n_nonfinite", "n_voxels",
            "count_below_minus_10000"
        ])
        writer.writeheader()
        for case in cases:
            for name in AUDIT["CHANNELS"]:
                writer.writerow({"case_id": case, "channel": name, "min": -100,
                                 "max": 100, "n_nonfinite": 1 if name == "MTT" else 0,
                                 "n_voxels": 60, "count_below_minus_10000": 0})
    fake_stat = AUDIT["summarize_channel"](np.zeros((5, 2, 2, 2), dtype=np.float32), 0)
    calls = []

    def fake_audit(case, data, label, *, block, retries):
        calls.append(case)
        return {"shape": (5, 2, 2, 2), "dtype": "float32", "seg_shape": (1, 2, 2, 2),
                "channels": {ch: dict(fake_stat) for ch in AUDIT["CHANNELS"]}}

    monkeypatch.setitem(AUDIT["run"].__globals__, "audit_case", fake_audit)
    out = tmp_path / "output"
    AUDIT["run"](dataset, split_file, raw, out, block=2)
    assert len(calls) == 149
    result = json.loads((out / "isles2024_preprocessed_tensor_audit_aggregate.json").read_text())
    assert result["n_patients"] == 149
    assert result["n_channel_volumes"] == 745
    assert result["channels"]["MTT"]["raw_cases_with_nonfinite"] == 149
    assert result["channels"]["MTT"]["pre_cases_with_nonfinite"] == 0
    assert len(list((out / "patient_checkpoints").glob("*.json"))) == 149
    AUDIT["run"](dataset, split_file, raw, out, block=2)
    assert len(calls) == 149, "A resumed run must not rescan unchanged cases"
    # Changing source metadata invalidates the corresponding cached case.
    (config / f"{cases[0]}.b2nd").write_bytes(b"synthetic new input")
    AUDIT["run"](dataset, split_file, raw, out, block=2)
    assert calls[-1] == cases[0] and len(calls) == 150
