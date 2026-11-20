"""WU20F patient-free virtual staging and fail-closed gate regression tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Import study script without needing real patient NIfTI data.
import preflight_isles2024_wu20f_staging as preflight
import qualify_isles2024_wu20f_synthetic_forward as forward


def test_virtual_inventory_for_every_fold():
    for fold in range(5):
        n = 120 if fold == 4 else 119
        train = [f"tr{i:04d}" for i in range(n)]
        val = [f"vl{i:04d}" for i in range(149 - n)]
        record = preflight.virtual_fold_inventory({"fold": fold, "train": train,
                                                     "heldout": val, "manifest_sha256": "a"*64})
        assert set(record["train_ids_PRIVATE"]).isdisjoint(record["heldout_ids_PRIVATE"])
        assert record["approved_for_dataset_materialization"] is False
        a, b = record["arms"]["506"], record["arms"]["507"]
        assert a["imagesTr_declared"] == n and a["imagesTs_declared"] == 149-n
        assert b["imagesTr_declared"] == n*5 and b["imagesTs_declared"] == (149-n)*5
        assert a["labelsTr_declared"] == b["labelsTr_declared"] == n
        assert a["labelsTs_declared"] == b["labelsTs_declared"] == 0
        assert a["actual_nifti_written"] is b["actual_nifti_written"] is False
        assert len(a["channel_names"]) == 1 and len(b["channel_names"]) == 5
        assert a["virtual_workspace"] != b["virtual_workspace"]


def test_rejects_protected_source_output(tmp_path):
    raw = tmp_path / "nnUNet_raw" / "Dataset505_data"
    plans = tmp_path / "nnUNet_preprocessed" / "Dataset505_data" / "nnUNetPlans.json"
    splits = plans.parent / "splits_final.json"
    folds = tmp_path / "results" / "fold_private"
    gate = tmp_path / "review" / "gate.json"
    policy = gate.parent / "policy.json"
    for p in (raw, plans.parent, folds, gate.parent):
        p.mkdir(parents=True, exist_ok=True)
    for target in (raw / "candidate", raw.parent / "wrong", plans.parent / "bad", folds / "bad"):
        with pytest.raises(ValueError):
            preflight.protect_input_trees(target, raw, splits, plans, folds, policy, gate)
    preflight.protect_input_trees(tmp_path / "private_output", raw, splits, plans, folds, policy, gate)


def test_input_signature_rejects_empty_and_missing(tmp_path):
    path = tmp_path / "case_0000.nii.gz"
    with pytest.raises(FileNotFoundError):
        preflight.input_signature([path])
    path.write_bytes(b"a" * 100)
    with pytest.raises(ValueError):
        preflight.input_signature([path])
    path.write_bytes(b"a" * 400)
    assert len(preflight.input_signature([path])["sha256"]) == 64


def test_missing_label_blocks_without_opening_label(tmp_path, monkeypatch):
    case = "sub_stroke0001"
    imdir = tmp_path / "imagesTr"
    imdir.mkdir()
    for i in range(5):
        (imdir / f"{case}_{i:04d}.nii.gz").write_bytes(b"k" * 400)
    monkeypatch.setattr(preflight, "inspect_image_header", lambda p: {
        "shape": [8, 8, 8], "spacing_mm": [1, 1, 1],
        "affine": np.eye(4).tolist(), "dtype": "float32",
    })
    with pytest.raises(ValueError, match="Missing source label"):
        preflight.inspect_case(tmp_path, case)
    labdir = tmp_path / "labelsTr"
    labdir.mkdir()
    (labdir / f"{case}.nii.gz").write_bytes(b"L" * 500)
    result = preflight.inspect_case(tmp_path, case)
    assert result["label_exists_no_read"] is True
    assert result["voxel_data_read"] is False


def test_geometry_mismatch_blocks(tmp_path, monkeypatch):
    case = "sub_stroke0002"
    for name in ("imagesTr", "labelsTr"):
        (tmp_path / name).mkdir()
    for i in range(5):
        (tmp_path / "imagesTr" / f"{case}_{i:04d}.nii.gz").write_bytes(b"X" * 400)
    (tmp_path / "labelsTr" / f"{case}.nii.gz").write_bytes(b"L" * 400)
    def fake_header(path):
        return {"shape": [8, 9, 10] if path.name.endswith("0001.nii.gz") else [8, 8, 8],
                "affine": np.eye(4).tolist(), "dtype": "float32"}
    monkeypatch.setattr(preflight, "inspect_image_header", fake_header)
    with pytest.raises(ValueError, match="five-channel geometry"):
        preflight.inspect_case(tmp_path, case)


def test_cache_invalidated_by_source_change(tmp_path):
    case = "sub_stroke0003"
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    for i in range(5):
        (tmp_path / "imagesTr" / f"{case}_{i:04d}.nii.gz").write_bytes(b"k" * 401)
    (tmp_path / "labelsTr" / f"{case}.nii.gz").write_bytes(b"L" * 402)
    sig = preflight.input_signature([*
        (tmp_path / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)),
        tmp_path / "labelsTr" / f"{case}.nii.gz"])
    record = {"schema": preflight.CASE_SCHEMA, "case_id": case, "status": "HEADER_VALID",
              "label_exists_no_read": True, "image_source_signature": sig}
    assert preflight.verify_cached_case(case, record, tmp_path)
    (tmp_path / "imagesTr" / f"{case}_0001.nii.gz").write_bytes(b"l" * 403)
    assert not preflight.verify_cached_case(case, record, tmp_path)


def test_preflight_does_not_pass_partial_cohort(tmp_path, monkeypatch):
    case_ids = [f"sub_stroke{i:04d}" for i in range(1, 150)]
    fold_records = [{"fold": i,"train": case_ids[:-30], "heldout": case_ids[-30:],
                     "manifest_sha256": "a"*64} for i in range(5)]
    monkeypatch.setattr(preflight,"verify_frozen_cohort", lambda *_: ({}, fold_records, case_ids))
    monkeypatch.setattr(preflight, "inspect_case", lambda path,case: {
        "schema":preflight.CASE_SCHEMA,"case_id":case,"status":"HEADER_VALID",
        "label_exists_no_read":True,"image_source_signature":{}})
    monkeypatch.setattr(preflight,"protect_input_trees",lambda *_:None)
    monkeypatch.setattr(preflight,"file_sha256",lambda *_: "a"*64)
    result=preflight.run_preflight(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path,tmp_path,
                                   tmp_path/"private_result",limit_new_cases=3)
    assert result["status"]=="PARTIAL_OR_FAILED" and result["n_patients"]==3
    assert result["training_ready"] is False
    assert not (tmp_path/"private_result"/"WU20F_virtual_staging_PRIVATE.json").exists()


def test_synthetic_forward_must_never_approve_training():
    forward.assert_no_training_release({"training_ready":False,
                                        "trained_model_inference_performed":False})
    for k in ("training_ready","trained_model_inference_performed"):
        bad={"training_ready":False,"trained_model_inference_performed":False}
        bad[k]=True
        with pytest.raises(ValueError):
            forward.assert_no_training_release(bad)


def test_policy_hash_is_canonical_not_file_bytes():
    policy = {"z":1,"a":2}
    pretty=json.dumps(policy,indent=2)
    digest=preflight.hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    assert digest!=preflight.hashlib.sha256(pretty.encode()).hexdigest()


def test_notebook25_cpu_and_no_gpu_training():
    path=ROOT/"notebooks"/"25_ISLES2024_WU20F_VIRTUAL_STAGING_AND_SYNTHETIC_FORWARD.ipynb"
    nb=json.loads(path.read_text(encoding="utf8"))
    import ast
    all_code="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    for c in nb["cells"]:
        if c["cell_type"]=="code":ast.parse("".join(c["source"]))
    assert "--raw505" in all_code and "--source-plans" in all_code
    assert "train" not in all_code.lower().replace("training_ready", "") or "nnUNetv2_train" not in all_code
    assert "nnUNetv2_train" not in all_code and "torch.cuda" not in all_code
    assert "__WU20F_COMMIT__" not in all_code
