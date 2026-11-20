"""Patient-free WU20E tests; all public fixtures use synthetic case identifiers."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "scripts" / "audit_isles2024_wu20e_release_gate.py"
SPEC = importlib.util.spec_from_file_location("wu20e_gate", SRC)
WU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WU)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "input"
    roots = {p: root / p for p in ("fold_private", "overlays")}
    for p in roots.values():
        p.mkdir(parents=True)
    cases = [f"case{i:03d}" for i in range(149)]
    valgroups = [cases[i * 30: (i + 1) * 30] for i in range(4)] + [cases[120:]]
    splits = [{"train": [c for c in cases if c not in val], "val": val}
              for val in valgroups]
    split_path = root / "splits_final.json"
    split_path.write_text(json.dumps(splits), encoding="utf8")
    monkeypatch.setattr(WU, "SPLIT_SHA", WU.sha256(split_path))
    for i, orig in enumerate(splits):
        train, val = orig["train"], orig["val"]
        params = {}
        for channel in WU.CHANNELS:
            params[channel] = {
                "channel": channel, "n_training_cases": len(train),
                "training_ids": train, "validation_ids_excluded": val,
                "n_samples": len(train) * 4096,
                "status": "candidate_not_authorized_for_training",
                "clip_low": -11., "clip_high": 20., "center": 2., "scale": 3.,
            }
        obj = {"fold": i, "frozen_split_sha256": WU.SPLIT_SHA,
               "training_ready": False, "status": "TRAIN_ONLY_FIT_COMPLETE_QC_PENDING",
               "train_ids": train, "heldout_ids": val, "provisional_fit": params}
        (roots["fold_private"] / f"WU20B_fold{i}_parameters_PRIVATE.json").write_text(
            json.dumps(obj), encoding="utf8")
    overlays = roots["overlays"]
    review = root / "review.csv"
    names = [f"sub_stroke{i:04d}" for i in range(1, 16)]
    with review.open("w", newline="", encoding="utf8") as out:
        fields = sorted(WU.REVIEW_COLUMNS)
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for case in names:
            (overlays / f"{case}_support_QC.png").write_bytes(b"synthetic-bytes-not-a-patient-image")
            writer.writerow({"case_id": case,
                             "overlay_filename": f"{case}_support_QC.png",
                             "decision": "PENDING"})
    b = {"status": "FULL_FOLD_PROVISIONAL_FIT_COMPLETE", "n_patients": 149,
         "n_folds": 5, "n_channel_case_samples": 745, "n_review_overlays": 15,
         "training_ready": False, "frozen_split_sha256": WU.SPLIT_SHA,
         "source_plans_sha256": WU.PLANS_SHA,
         "qc_aggregate": {"n_cases": 149, "n_support_below_0_10": 1,
                          "n_support_above_0_40": 0}}
    c = {"status": "CPU_PREPROCESSING_PROBE_COMPLETE_NOT_TRAINING_READY",
         "source_roi_only_not_full_volume": True, "real_model_inference_performed": False,
         "training_ready": False, "frozen_split_sha256": WU.SPLIT_SHA,
         "source_plans_sha256": WU.PLANS_SHA, "locked_architecture_spacing": True,
         "postresample_ncct_max_abs_difference": 0,
         "postresample_perfusion_voxels_abs_gt_8": 0,
         "input_noNorm_checks": {i: {"actual_normalizer_no_change": True,
                                    "nnunetv2_version": "2.8.1"}
                                 for i in ("506", "507")}}
    d = {"case_status": dict.fromkeys(WU.EXPECTED_CASES, "FULL_VOLUME_TECHNICAL_PASS"),
         "patients_qualified": 4, "dataset504_505_unchanged": True,
         "real_model_inference_performed": False, "training_ready": False,
         "per_case_postresample_overshoot_voxels": dict.fromkeys(WU.EXPECTED_CASES, 0),
         "per_case_raw_mtt_nonfinite": {"sub_stroke0016": 9,
                                         "sub_stroke0113": 225045,
                                         "sub_stroke0137": 0,
                                         "sub_stroke0152": 0}}
    for name, obj in (("b.json", b), ("c.json", c), ("d.json", d)):
        (root / name).write_text(json.dumps(obj), encoding="utf8")
    return root, split_path, review, overlays


def test_five_folds_exactly_match_frozen_splits(tmp_path, monkeypatch):
    root, splits, _, _ = fixture(tmp_path, monkeypatch)
    result = WU.verify_folds(root / "fold_private", splits)
    assert [x["n_heldout"] for x in result["folds"]] == [30, 30, 30, 30, 29]
    assert result["fold_inventory_verified"] is True


def test_frozen_split_hash_change_blocks(tmp_path, monkeypatch):
    root, splits, _, _ = fixture(tmp_path, monkeypatch)
    splits.write_bytes(splits.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        WU.verify_folds(root / "fold_private", splits)


def test_leaking_validation_patient_blocks(tmp_path, monkeypatch):
    root, splits, _, _ = fixture(tmp_path, monkeypatch)
    path = root / "fold_private" / "WU20B_fold2_parameters_PRIVATE.json"
    o = json.loads(path.read_text())
    o["provisional_fit"]["MTT"]["training_ids"][0] = o["heldout_ids"][0]
    path.write_text(json.dumps(o), encoding="utf8")
    with pytest.raises(ValueError, match="leaky fit"):
        WU.verify_folds(root / "fold_private", splits)


def test_fold_with_missing_patient_blocks(tmp_path, monkeypatch):
    root, splits, _, _ = fixture(tmp_path, monkeypatch)
    path = root / "fold_private" / "WU20B_fold0_parameters_PRIVATE.json"
    o = json.loads(path.read_text())
    o["train_ids"] = o["train_ids"][:-1]
    path.write_text(json.dumps(o), encoding="utf8")
    with pytest.raises(ValueError, match="does not match"):
        WU.verify_folds(root / "fold_private", splits)


def test_review_all_pending_is_not_approved(tmp_path, monkeypatch):
    _, _, review, overlays = fixture(tmp_path, monkeypatch)
    result = WU.inspect_review(review, overlays)
    assert result["review_rows"] == 15
    assert result["review_decisions"] == {"PENDING": 15}
    assert result["review_complete"] is False


def test_review_duplicate_case_blocks(tmp_path, monkeypatch):
    _, _, review, overlays = fixture(tmp_path, monkeypatch)
    data = review.read_text()
    rows = data.splitlines()
    review.write_text("\n".join([rows[0], rows[1], rows[1], *rows[3:]]) + "\n")
    with pytest.raises(ValueError, match="case inventory"):
        WU.inspect_review(review, overlays)


def test_approved_row_without_review_credentials_is_not_approved(tmp_path, monkeypatch):
    _, _, review, overlays = fixture(tmp_path, monkeypatch)
    with review.open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    for row in rows:
        row.update({"decision": "APPROVED", "brain_tissue_preserved": "YES",
                    "correction_required": "NO", "extracranial_tissue_included": "NO"})
    with review.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = WU.inspect_review(review, overlays)
    assert result["review_complete"] is False
    assert any("identity/date" in t for t in result["review_blockers"])


def test_signoff_fields_cannot_authorize_training(tmp_path, monkeypatch):
    root, splits, review, overlays = fixture(tmp_path, monkeypatch)
    with review.open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    for row in rows:
        row.update({"decision": "APPROVED", "brain_tissue_preserved": "YES",
                    "correction_required": "NO", "extracranial_tissue_included": "NO",
                    "reviewer": "Synthetic Reviewer", "review_date": "2026-11-19"})
    with review.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "results"
    aggregate = WU.evaluate(root / "b.json", root / "c.json", root / "d.json",
                            root / "fold_private", splits, review, overlays, out)
    assert aggregate["n_review_decisions_pending"] == 0
    assert aggregate["training_ready"] is False
    assert json.loads((out / "WU20E_release_gate_PRIVATE.json").read_text())["training_ready"] is False


def test_missing_full_volume_overshoot_checks_fail_closed(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch)
    b, c, d = [WU.read_json(root / name) for name in ("b.json", "c.json", "d.json")]
    d["per_case_postresample_overshoot_voxels"] = {}
    with pytest.raises(ValueError, match="full native-volume"):
        WU.verify_numeric_audits(b, c, d)


def test_output_cannot_mutate_review_inputs(tmp_path, monkeypatch):
    root, splits, review, overlays = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="overwrite"):
        WU.evaluate(root / "b.json", root / "c.json", root / "d.json",
                    root / "fold_private", splits, review, overlays, overlays / "new")


def test_proposed_policy_is_immutable_candidate_without_approval():
    policy = WU.proposed_policy()
    assert policy["training_authorized"] is False
    assert "UNAPPROVED" in policy["status"]
    assert policy["perfusion_post_z_cap"] == 8.0
    assert len(hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()) == 64
