"""Patient-free fail-close regression checks for WU20D full native-volume audit."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE = run_path(str(ROOT / "scripts" / "qualify_isles2024_full_volume.py"))


def test_nonfinite_and_extreme_partition_counts_are_exact():
    a = np.array([-50000., 20000., np.inf, np.nan, 4., -10001., -2., 40000.],
                 dtype=np.float32).reshape(2, 2, 2)
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask.flat[[0, 2, 3, 6]] = True
    before = a.copy()
    r = MODULE["raw_channel_diagnostics"](a, mask)
    assert r["n_voxels"] == 8
    assert r["n_nonfinite_full"] == 2
    assert r["n_nonfinite_within_support"] == 2
    assert r["n_nonfinite_outside_support"] == 0
    assert r["n_below_neg10000_full"] == 2
    assert r["n_below_neg10000_within_support"] == 1
    assert r["n_above_pos10000_full"] == 2
    assert r["n_above_pos10000_within_support"] == 0
    np.testing.assert_array_equal(a, before, strict=True)


def test_reject_invalid_anatomical_support():
    with pytest.raises(ValueError, match="shape mismatch"):
        MODULE["raw_channel_diagnostics"](np.zeros((2, 2, 2)), np.ones((2, 2), bool))


def test_pausing_or_stale_checkpoint_fails_closed(tmp_path):
    p = tmp_path / "test.json"
    assert MODULE["existing_completed"](p, "signature", "sub_stroke0016", 2) is None
    data = {"schema": MODULE["SCHEMA"], "case": "sub_stroke0016", "fold": 2,
            "input_sha256": "signature", "status": "FULL_VOLUME_TECHNICAL_PASS"}
    MODULE["atomic_json"](p, data)
    assert MODULE["existing_completed"](p, "signature", "sub_stroke0016", 2) == data
    with pytest.raises(ValueError, match="stale"):
        MODULE["existing_completed"](p, "changed", "sub_stroke0016", 2)
    data["status"] = "PARTIAL"
    MODULE["atomic_json"](p, data)
    with pytest.raises(ValueError, match="Unfinished"):
        MODULE["existing_completed"](p, "signature", "sub_stroke0016", 2)


def test_output_isolated_from_frozen_sources(tmp_path):
    raw = tmp_path / "nnUNet_raw" / "Dataset505_ISLES2024_NCCT_PERFUSION"
    plans = tmp_path / "nnUNet_preprocessed" / "Dataset505_ISLES2024_NCCT_PERFUSION" / "nnUNetPlans.json"
    splits = plans.parent / "splits_final.json"
    for path in (raw, plans.parent):
        path.mkdir(parents=True)
    for unsafe in (raw, raw / "results", raw.parent / "reports", plans.parent / "results"):
        with pytest.raises(ValueError, match="frozen"):
            MODULE["ensure_output_isolated"](unsafe, raw, plans, splits)
    MODULE["ensure_output_isolated"](tmp_path / "results" / "wu20d", raw, plans, splits)


def test_postresample_pair_equivalence_and_overshoot_is_detected():
    a = np.zeros((1, 4, 5, 6), dtype=np.float32)
    b = np.zeros((5, 4, 5, 6), dtype=np.float32)
    props = {"bbox_used_for_cropping": [[0, 4], [0, 5], [0, 6]]}
    got = MODULE["verify_pair_after_preprocessing"](a, b, props, props)
    assert got["postresample_ncct_max_abs_difference"] == 0
    assert got["postresample_perfusion_voxels_abs_gt_8"] == 0
    b[4, 0, 0, 0] = 8.5
    got = MODULE["verify_pair_after_preprocessing"](a, b, props, props)
    assert got["postresample_excess_requires_review"] is True
    assert got["postresample_perfusion_voxels_abs_gt_8"] == 1
    b[0, 0, 0, 0] = 0.01
    with pytest.raises(ValueError, match="NCCT mismatch"):
        MODULE["verify_pair_after_preprocessing"](a, b, props, props)
    b[0, 0, 0, 0] = 0
    b[1, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        MODULE["verify_pair_after_preprocessing"](a, b, props, props)


def test_case_fold_pins_and_no_label_or_training_commands():
    assert MODULE["CASE_FOLDS"] == {
        "sub_stroke0016": 2, "sub_stroke0113": 1,
        "sub_stroke0137": 4, "sub_stroke0152": 3,
    }
    script = (ROOT / "scripts" / "qualify_isles2024_full_volume.py").read_text()
    assert "nnUNetv2_train" not in script
    assert "labelsTr" not in script
    ast.parse(script)


def test_notebook_23_is_cpu_only_and_parses():
    path = ROOT / "notebooks" / "23_ISLES2024_WU20D_FULL_NATIVE_VOLUME.ipynb"
    notebook = json.loads(path.read_text(encoding="utf8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    joined = "\n".join("".join(x["source"]) for x in notebook["cells"])
    assert "nnunetv2==2.8.1" in joined
    assert "--case" in joined
    assert "nnUNetv2_train" not in joined
    assert "--fold-manifest-dir" in joined
