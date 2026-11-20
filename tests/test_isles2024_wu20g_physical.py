"""WU20G staging safety and storage tests; all synthetic, no clinical data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import stage_isles2024_wu20g_physical as stage


def cases(shape=(32, 32, 16)):
    return [{"schema": "WU20F-image-header-v1", "case_id": f"case{i:03d}",
             "reference_header": {"shape": list(shape)}} for i in range(149)]


def test_uncompressed_storage_uses_exact_fold_and_channel_counts():
    items = cases()
    out = stage.estimate_storage(items, 149 * 800)
    expected = 149 * 32 * 32 * 16 * 5 * 6 * 4
    assert out["uncompressed_images_bytes"] == expected
    assert out["reference_label_copy_bytes"] == 149 * 800 * 8
    assert out["recommended_free_bytes_at_least"] == 2 * out["combined_uncompressed_bytes"]
    assert out["clinical_output_staged"] is False
    assert out["gzip_size_predictable"] is False


@pytest.mark.parametrize("replace", [[], [32, 0, 16], [16, 16], [12, 10, 7]])
def test_storage_fails_with_incomplete_inventory_or_invalid_shape(replace):
    x = cases()
    if replace:
        x[0]["reference_header"]["shape"] = replace
    else:
        x.pop()
    with pytest.raises(ValueError):
        stage.estimate_storage(x, 10)


def test_source_workspace_never_overlaps_or_resolves_to_source(tmp_path):
    raw = tmp_path / "input" / "raw505"
    raw.mkdir(parents=True)
    with pytest.raises(ValueError, match="overlaps"):
        stage.reject_unsafe_workspace(raw / "candidate", [raw])
    with pytest.raises(ValueError, match="overlaps"):
        stage.reject_unsafe_workspace(tmp_path / "input", [raw])
    stage.reject_unsafe_workspace(tmp_path / "isolated", [raw])
    link = tmp_path / "link"
    link.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        stage.reject_unsafe_workspace(link / "child", [raw])


def test_no_training_command_or_checkpoint_selection_leakage():
    x = stage.make_training_hold_plan()
    assert x["training_ready"] is False
    assert x["default_nnunetv2_train_allowed"] is False
    assert x["heldout_labels_available_to_trainer"] is False
    assert x["training_commands_generated"] is False
    assert x["target_epochs_per_arm_fold"] == 1000


def test_preflight_requires_both_frozen_metadata_and_failed_release():
    ok = {"schema": "WU20F-virtual-staging-v1",
          "status": "VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE",
          "n_errors": 0, "n_patients": 149, "n_source_image_headers_checked": 745,
          "n_label_file_presence_checks": 149, "training_ready": False,
          "staging_materialized": False, "image_voxels_read": False,
          "label_voxels_read": False}
    stage.check_preflight(ok)
    for field, wrong in (("n_patients", 148), ("training_ready", True),
                         ("staging_materialized", True), ("label_voxels_read", True),
                         ("status", "PARTIAL"), ("n_errors", 1)):
        changed = {**ok, field: wrong}
        with pytest.raises(ValueError):
            stage.check_preflight(changed)


def test_stage_mode_unconditionally_denied(tmp_path, monkeypatch):
    argv = ["stage.py", "--mode", "stage", "--plans", str(tmp_path / "plans.json"),
            "--output", str(tmp_path / "unapproved")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(PermissionError, match="DISABLED"):
        stage.main()
    assert not (tmp_path / "unapproved").exists()


def test_invalid_real_case_cannot_be_staged_even_via_helper(tmp_path):
    with pytest.raises(PermissionError, match="synthetic"):
        stage.stage_synthetic_case(tmp_path, tmp_path / "a", tmp_path / "b",
                                   "sub_stroke0016", "imagesTs", {})


def test_synthetic_heldout_never_accepts_label_parameter(tmp_path):
    params = stage.synthetic_parameters()
    with pytest.raises(ValueError, match="held-out label"):
        stage.stage_synthetic_case(tmp_path, tmp_path / "a", tmp_path / "b",
                                   "sub_stroke9001", "imagesTs", params,
                                   label_file=tmp_path / "forbidden.nii.gz")
    with pytest.raises(ValueError, match="requires label"):
        stage.stage_synthetic_case(tmp_path, tmp_path / "a", tmp_path / "b",
                                   "sub_stroke9001", "imagesTr", params)


def test_synthetic_parameters_not_falsely_approved():
    p = stage.synthetic_parameters()
    assert set(p) == set(stage.CHANNELS)
    assert p["NCCT"]["candidate_post_z_cap"] is None
    for v in p.values():
        assert v["status"] == "candidate_not_authorized_for_training"
        assert np.isfinite(v["center"])


def test_plan_checks_149_cases_without_touching_voxels(tmp_path, monkeypatch):
    src = tmp_path / "raw" / "Dataset505"
    files = tmp_path / "results" / "folds"
    source_splits = tmp_path / "pre" / "splits_final.json"
    source_plans = source_splits.parent / "nnUNetPlans.json"
    gate = tmp_path / "results" / "review" / "gate.json"
    policy = gate.parent / "policy.json"
    other = tmp_path / "results" / "wu20f" / "aggregate.json"
    checkpoints = other.parent / "patient_header_checkpoints"
    for d in (src, files, source_plans.parent, gate.parent, checkpoints):
        d.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps({"schema":"WU20F-virtual-staging-v1",
        "status":"VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE","n_errors":0,
        "n_patients":149,"n_source_image_headers_checked":745,
        "n_label_file_presence_checks":149,"training_ready":False,
        "staging_materialized":False,"image_voxels_read":False,"label_voxels_read":False}))
    for p in (source_splits, source_plans, gate, policy):
        p.write_text("{}")
    ids = [f"sub_stroke{i:04d}" for i in range(1,150)]
    for i in ids:
        target = src / "labelsTr" / f"{i}.nii.gz"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"abc" * 200)
        (checkpoints / f"{i}_header_PRIVATE.json").write_text(json.dumps({
            "schema": stage.CASE_SCHEMA,"case_id":i,
            "reference_header":{"shape":[32,32,16]}}))
    mock_folds = [{"fold":i,"manifest_sha256":"a"*64} for i in range(5)]
    monkeypatch.setattr(stage,"verify_frozen_cohort", lambda *_: ({"identity":"locked"},mock_folds,ids))
    monkeypatch.setattr(stage,"verify_cached_case",lambda *_: True)
    report = stage.plan_staging(src,source_splits,source_plans,files,gate,policy,
                                other,checkpoints,tmp_path / "reports",tmp_path / "scratch")
    assert report["n_patients"] == 149
    assert report["n_image_headers_verified"] == 745
    assert report["training_ready"] is False
    assert report["staging_materialized"] is False
    assert len(stage.load_object(tmp_path/"reports"/"WU20G_physical_staging_plan_PRIVATE.json")
               ["candidate_workspaces"]) == 10
    assert not (tmp_path / "scratch").exists()
    # an invalid case invalidates the entire plan
    (checkpoints / f"{ids[0]}_header_PRIVATE.json").unlink()
    with pytest.raises(FileNotFoundError):
        stage.plan_staging(src,source_splits,source_plans,files,gate,policy,
                           other,checkpoints,tmp_path / "reports",tmp_path / "scratch")


def test_notebook26_no_training_commands_and_source_pin():
    notebook = ROOT / "notebooks" / "26_ISLES2024_WU20G_PHYSICAL_STAGING_QUALIFICATION.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf8"))
    import ast
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "--mode" in code and "'plan'" in code and "'synthetic'" in code
    assert "nnUNetv2_train" not in code and "torch.cuda" not in code
    assert "__WU20G_SOURCE_COMMIT__" not in code
    assert "training_ready" in code
