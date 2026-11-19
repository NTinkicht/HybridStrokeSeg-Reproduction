"""WU20B fail-closed, training-only fold calibration regression tests."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_isles2024_all_folds.py"
NOTEBOOK = ROOT / "notebooks" / "21_ISLES2024_WU20B_FULL_FOLD_CALIBRATION.ipynb"
if not SCRIPT.exists():
    SCRIPT = ROOT / "wu20b" / "calibrate_isles2024_all_folds.py"
if not NOTEBOOK.exists():
    NOTEBOOK = ROOT / "wu20b" / "21_ISLES2024_WU20B_FULL_FOLD_CALIBRATION.ipynb"
AUDIT = run_path(str(SCRIPT))


def test_atomic_checkpoint_detects_changed_source_and_corrupted_samples(tmp_path, monkeypatch):
    source = tmp_path / "Dataset505_ISLES2024_NCCT_PERFUSION"
    img = source / "imagesTr"
    img.mkdir(parents=True)
    case = "sub_stroke0001"
    for i in range(5):
        (img / f"{case}_{i:04d}.nii.gz").write_bytes(bytes([65 + i]) * 40)
    calls = []

    def fake_case(*_args):
        calls.append(1)
        return ({ch: np.linspace(0, 1, 4096, dtype=np.float32)
                 for ch in AUDIT["CHANNELS"]},
                {"support": {"coverage_fraction": 0.25},
                 "nonfinite_in_original_channels": {"MTT": 0}})

    monkeypatch.setitem(AUDIT["create_checkpoint"].__globals__, "calculate_case", fake_case)
    output = tmp_path / "results"
    entry = AUDIT["create_checkpoint"](output, source, case)
    meta_path, npz_path = AUDIT["case_paths"](output, case)
    assert len(calls) == 1
    assert AUDIT["verify_checkpoint"](entry, entry["inputs"], npz_path, case)
    assert json.loads(meta_path.read_text())["sample_sha256"] == AUDIT["file_sha256"](npz_path)
    # Tampering with raw input invalidates patient-level resume.
    with (img / f"{case}_0004.nii.gz").open("ab") as f:
        f.write(b"tamper")
    current = AUDIT["source_signatures"](source, case)
    assert not AUDIT["verify_checkpoint"](entry, current, npz_path, case)
    with npz_path.open("ab") as f:
        f.write(b"tamper")
    assert not AUDIT["verify_checkpoint"](entry, entry["inputs"], npz_path, case)


def make_fold_cases():
    ids = [f"sub_stroke{i:04d}" for i in range(1, 150)]
    folds = []
    cursor = 0
    for count in (30, 30, 30, 30, 29):
        val = ids[cursor:cursor + count]
        folds.append(([c for c in ids if c not in val], val))
        cursor += count
    return ids, folds


def patch_fake_inputs(monkeypatch, tmp_path, *, fail_one=False):
    ids, fold_cases = make_fold_cases()
    gl = AUDIT["full_calibration"].__globals__
    monkeypatch.setitem(gl, "get_split_fold_cases", lambda *_: (ids, fold_cases))
    monkeypatch.setitem(gl, "validate_source", lambda *_: {"plans_sha256": "plan"})
    monkeypatch.setitem(gl, "source_signatures", lambda raw, case: [{"name": case}])
    monkeypatch.setitem(gl, "verify_checkpoint", lambda *_: False)

    def create(out, _raw, case):
        if fail_one and case == ids[-1]:
            raise ValueError("bad source case")
        return {"sample_sha256": hashlib.sha256(case.encode()).hexdigest(),
                "qc": {"support": {"coverage_fraction": 0.20},
                       "nonfinite_in_original_channels": {"MTT": 0}}}

    monkeypatch.setitem(gl, "create_checkpoint", create)
    # Synthetic non-constant, channel-specific deterministic per-patient samples.
    all_samples = {
        name: {
            case: np.linspace(1, 80, 4096, dtype=np.float32) + idx * 0.02 + j * 0.1
            for idx, case in enumerate(ids)
        }
        for j, name in enumerate(AUDIT["CHANNELS"])
    }
    monkeypatch.setitem(gl, "load_samples", lambda *_: all_samples)
    raw = tmp_path / "nnUNet_raw" / "Dataset505_ISLES2024_NCCT_PERFUSION"
    raw.mkdir(parents=True)
    paths = (raw, tmp_path / "splits.json", tmp_path / "nnUNetPlans.json",
             tmp_path / "results")
    paths[1].write_text("frozen")
    paths[2].write_text("plans")
    return paths, ids, fold_cases


def test_all_five_folds_use_exact_119_or_120_train_cases(monkeypatch, tmp_path):
    paths, ids, folds = patch_fake_inputs(monkeypatch, tmp_path)
    result = AUDIT["full_calibration"](*paths, render_overlays=False)
    assert result["status"] == "FULL_FOLD_PROVISIONAL_FIT_COMPLETE"
    assert result["training_ready"] is False
    assert result["n_patients"] == 149
    assert len(result["folds"]) == 5
    assert (paths[3] / "WU20B_progress.json").is_file()
    for fold, (train, val) in enumerate(folds):
        output = json.loads((paths[3] / "fold_private" /
                             f"WU20B_fold{fold}_parameters_PRIVATE.json").read_text())
        assert len(train) in (119, 120)
        assert len(val) in (29, 30)
        assert output["training_ready"] is False
        assert set(output["source_sample_sha256"]) == set(train)
        assert set(output["source_sample_sha256"]).isdisjoint(val)
        for ch, params in output["provisional_fit"].items():
            assert set(params["training_ids"]) == set(train)
            assert set(params["validation_ids_excluded"]) == set(val)
            assert params["n_training_cases"] == len(train)
            assert params["n_samples"] == len(train) * 4096
            assert params["status"] == "candidate_not_authorized_for_training"
        assert output["provisional_fit"]["NCCT"]["scheme"] == "CT_style_fold_only"


def test_bad_source_blocks_all_fold_manifests(monkeypatch, tmp_path):
    paths, ids, _ = patch_fake_inputs(monkeypatch, tmp_path, fail_one=True)
    result = AUDIT["full_calibration"](*paths, render_overlays=False)
    assert result["status"] == "PARTIAL_OR_FAILED"
    assert result["completed_verified"] == 148
    assert result["n_errors"] == 1
    assert result["training_ready"] is False
    assert not (paths[3] / "fold_private").exists()
    assert json.loads((paths[3] / "WU20B_aggregate.json").read_text())["training_ready"] is False


def test_mask_qc_priorities_and_outliers_are_always_selected():
    ids, _ = make_fold_cases()
    entries = {case: {"qc": {"support": {"coverage_fraction": 0.18 + i * 0.0001}}}
               for i, case in enumerate(ids)}
    entries["sub_stroke0016"]["qc"]["support"]["coverage_fraction"] = 0.0738
    overlays = AUDIT["select_overlays"](entries)
    assert set(AUDIT["PRIORITY"]).intersection(entries).issubset(overlays)
    assert len(overlays) <= 16
    assert "sub_stroke0016" in overlays
    with pytest.raises(ValueError):
        AUDIT["select_overlays"](entries, limit=3)


def test_safety_gate_does_not_read_labels_or_prediction_arrays():
    code = SCRIPT.read_text(encoding="utf8")
    assert '"labelsTr"' not in code
    assert "nnUNetv2_train" not in code
    assert "prediction" not in code.split("def main()")[1]
    assert "training_ready\": False" in code
    assert "candidate_not_authorized_for_training" in code or "fit_fold_training_only" in code


def test_notebook_is_cpu_only_and_parses():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    assert "--raw505" in code
    assert "--splits" in code
    assert "--plans505" in code
    assert "--output-dir" in code
    assert "nnUNetv2_train" not in code
    assert "nnUNetv2_plan_and_preprocess" not in code
    assert "training_ready" in code
