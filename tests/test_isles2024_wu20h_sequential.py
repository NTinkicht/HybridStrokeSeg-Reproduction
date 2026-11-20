"""WU20H patient-free regression for sequential capacity and trainer access contract."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import audit_isles2024_wu20h_training_access as trace
import plan_isles2024_wu20h_sequential as h


@pytest.fixture
def fold_fixture():
    tr = {f"tr_{i:03d}" for i in range(119)}
    val = {f"val_{i:03d}" for i in range(30)}
    def arm(ident):
        n = 1 if ident == 506 else 5
        return {
            "arm": ident, "numTraining": 119, "numHeldout": 30,
            "actual_nifti_written": False, "labels_read": False,
            "labelsTs_declared": 0, "imagesTr_declared": 119*n,
            "imagesTs_declared": 30*n, "labelsTr_declared": 119,
            "ncct_copy_policy": "future_507_0000_shall_copy_exact_506_0000_bytes",
            "channel_names": {str(i): "noNorm" for i in range(n)},
            "expected_files_PRIVATE": {
                "imagesTr": sorted(f"{c}_{i:04d}.nii.gz" for c in tr for i in range(n)),
                "imagesTs": sorted(f"{c}_{i:04d}.nii.gz" for c in val for i in range(n)),
                "labelsTr": sorted(f"{c}.nii.gz" for c in tr),
                "labelsTs": [],
            },
        }
    return {"fold": 0, "train_ids_PRIVATE": sorted(tr), "heldout_ids_PRIVATE": sorted(val),
            "approved_for_dataset_materialization": False,
            "manifest_sha256": "a"*64, "arms": {"506": arm(506), "507": arm(507)}}, tr, val


def test_virtual_fold_exact_inventory_pass(fold_fixture):
    row, tr, val = fold_fixture
    actual = h.validate_virtual_record(row, 0, tr, val)
    assert actual["train_labels_each_arm"] == 119
    assert actual["heldout_labels_for_trainer"] == 0


@pytest.mark.parametrize("mutation", ["leak", "drop", "duplicate", "enable", "renorm", "bad_ncct"])
def test_virtual_fold_rejects_leak_and_tamper(fold_fixture, mutation):
    row, tr, val = fold_fixture
    a = row["arms"]["507"]
    if mutation == "leak":
        a["expected_files_PRIVATE"]["labelsTr"][0] = f"{next(iter(val))}.nii.gz"
    elif mutation == "drop":
        a["expected_files_PRIVATE"]["imagesTs"].pop()
    elif mutation == "duplicate":
        a["expected_files_PRIVATE"]["imagesTr"][0] = a["expected_files_PRIVATE"]["imagesTr"][1]
    elif mutation == "enable":
        row["approved_for_dataset_materialization"] = True
    elif mutation == "renorm":
        a["channel_names"]["0"] = "ZScoreNormalization"
    else:
        a["ncct_copy_policy"] = "recompute_independently"
    with pytest.raises(ValueError):
        h.validate_virtual_record(row, 0, tr, val)


def test_sequential_storage_uses_conservative_label_upper_bound():
    record = {"uncompressed_images_bytes": 339171655680,
              "reference_label_copy_bytes": 189978112,
              "combined_uncompressed_bytes": 339361633792}
    s = h.sequential_storage(record, 241870721024)
    assert s["one_fold_combined_upper_bound_bytes"] == 67881825664
    assert s["conservative_one_fold_2x_reservation_bytes"] == 135763651328
    assert s["five_fold_all_at_once_2x_reservation_bytes"] == 678723267584
    assert s["provisional_staging_space_meets_2x"] is True
    assert s["full_training_capacity_qualified"] is False
    assert h.sequential_storage(record, 120_000_000_000)["provisional_staging_space_meets_2x"] is False


def test_storage_rejects_fabricated_negative_or_inconsistent():
    good = {"uncompressed_images_bytes": 50_000, "reference_label_copy_bytes": 800,
            "combined_uncompressed_bytes": 50_800}
    for bad in ({**good, "combined_uncompressed_bytes": 50_801},
                {**good, "reference_label_copy_bytes": -1},
                {**good, "uncompressed_images_bytes": 50_001}):
        with pytest.raises(ValueError):
            h.sequential_storage(bad, 100000)


def test_schedule_never_permits_staging_training_or_deleting():
    records = [{"fold": i, "train_ids_PRIVATE": [f"train_{i}"],
                "heldout_ids_PRIVATE": [f"val_{i}"]} for i in range(5)]
    schedule = h.make_fold_schedule(records)
    assert len(schedule) == 5
    assert all(x["may_stage"] is x["may_train"] is x["may_delete"] is False for x in schedule)
    assert all("nnUNetv2_train" not in json.dumps(x) for x in schedule)
    with pytest.raises(ValueError):
        h.make_fold_schedule(records[:-1])


def test_protect_output_rejects_overwrite_and_symlink(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError):
        h.protect_outputs(src / "report", scratch, [src])
    with pytest.raises(ValueError):
        h.protect_outputs(scratch / "report", scratch, [src])
    link = tmp_path / "link"
    link.symlink_to(src, target_is_directory=True)
    with pytest.raises(ValueError):
        h.protect_outputs(link / "report", scratch, [src])
    h.protect_outputs(tmp_path / "report", scratch, [src])


def test_review_status_does_not_authenticate_approvals(tmp_path):
    csv_path = tmp_path / "review.csv"
    with csv_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "decision", "reviewer", "review_date"])
        writer.writeheader()
        for i in range(15):
            writer.writerow({"case_id": f"case{i:02d}", "decision": "APPROVED",
                             "reviewer": "fake", "review_date": "2026-11-20"})
    q = h.review_status(csv_path)
    assert q["approved_unverified"] == 15
    assert q["authentic_clinical_signature_verified"] is False



def test_invalid_new_review_or_policy_cannot_preserve_prior_success(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    h.atomic_json(output / "WU20H_sequential_aggregate.json", {
        "schema": h.SCHEMA, "status": "SEQUENTIAL_CAPACITY_AND_TRAINER_HOLD_PLAN_COMPLETE",
        "training_ready": False,
    })
    h.atomic_json(output / "WU20H_sequential_PRIVATE.json", {"prior_complete": True})
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}")
    monkeypatch.setattr(h, "protect_outputs", lambda *_: None)
    monkeypatch.setattr(h, "load_json", lambda *_: {})
    monkeypatch.setattr(h, "sha256", lambda *_: "f"*64)
    with pytest.raises(ValueError, match="policy/evidence"):
        h.create_plan(dummy, dummy, dummy, dummy, dummy, dummy,
                      dummy, dummy, tmp_path, tmp_path, output)
    result = json.loads((output / "WU20H_sequential_aggregate.json").read_text())
    assert result["status"] == "INCOMPLETE" and result["training_ready"] is False
    assert not (output / "WU20H_sequential_PRIVATE.json").exists()
    assert len(list((output / "superseded_NOT_CURRENT").glob("*_OLD_PRIVATE.json"))) == 1


def fixture_partition():
    tr = [f"fixture_train_{i:03d}" for i in range(119)]
    val = [f"fixture_heldout_{i:03d}" for i in range(30)]
    return tr, val


def test_complete_synthetic_training_trace_passes_but_cannot_authorize():
    train, heldout = fixture_partition()
    events = trace.make_synthetic_trace(train, heldout)
    result = trace.inspect_trace(events, set(train), set(heldout))
    assert result["synthetic_training_steps"] == 1000
    assert result["synthetic_heldout_images"] == 30
    assert result["actual_nnunet_trainer_hook_verified"] is False
    assert result["training_ready"] is False


@pytest.mark.parametrize("tamper", ["heldout_label_before_archive", "heldout_train_label",
                                 "early_checkpoint", "best_checkpoint", "skip_epoch",
                                 "outer_label_before_prediction_seal", "skip_prediction", "unknown_event"])
def test_training_trace_adversarial_fail_closed(tamper):
    train, heldout = fixture_partition()
    events = trace.make_synthetic_trace(train, heldout)
    if tamper == "heldout_label_before_archive":
        events.insert(3, {"type": "outer_evaluation_label_open", "case_id": heldout[0]})
    elif tamper == "heldout_train_label":
        events[0]["case_id"] = heldout[0]
    elif tamper == "early_checkpoint":
        events[1119]["epoch"] = 999
    elif tamper == "best_checkpoint":
        events.insert(100, {"type": "checkpoint_best_written", "epoch": 30})
    elif tamper == "skip_epoch":
        events = [e for e in events if not (e["type"] == "train_step" and e["epoch"] == 500)]
    elif tamper == "outer_label_before_prediction_seal":
        events.insert(-1, {"type": "outer_evaluation_label_open", "case_id": heldout[0]})
    elif tamper == "skip_prediction":
        events = [e for e in events if not (e.get("type") == "outer_inference_image_open" and e.get("case_id") == heldout[0])]
    else:
        events.append({"type": "erase_unarchived_training_data"})
    with pytest.raises(ValueError):
        trace.inspect_trace(events, set(train), set(heldout))


def test_notebook27_pin_and_no_live_action():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "27_ISLES2024_WU20H_SEQUENTIAL_RELEASE_HOLD.ipynb"
    if not path.exists():
        pytest.skip("Notebook not yet generated in local development")
    import ast
    nb = json.loads(path.read_text(encoding="utf8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "--mode', 'plan'" in code or "--mode','plan'" in code
    for item in ("__WU20H_COMMIT__", "nnUNetv2_train", "torch.cuda", "--mode','stage'"):
        assert item not in code
