#!/usr/bin/env python3
"""WU20H: read-only, fail-closed sequential one-fold-at-a-time research preflight.

Never opens image/label voxels, stages NIfTI, deletes data, trains, or issues approval.
Patient IDs and per-fold identities stay in PRIVATE results only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SPLIT_SHA = "664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027"
SOURCE_PLANS_SHA = "6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca"
SCHEMA = "WU20H-sequential-release-hold-v1"
FOLD_SIZES = [(119, 30)] * 4 + [(120, 29)]
ARMS = (506, 507)
MAX_SAFE_INT = 2**63 - 1
PROHIBITED_MODES = ("stage", "train", "cleanup", "release", "approve")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected object in {path.name}")
    return obj


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, sort_keys=True, indent=2, allow_nan=False)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temp, path)


def protect_outputs(output: Path, scratch: Path, evidence: list[Path]) -> None:
    """The reports must never overwrite immutable evidence or scratch data."""
    dest = output.absolute()
    if any(p.is_symlink() for p in (dest, *dest.parents)):
        raise ValueError("Output contains a symlink component")
    resolved = dest.resolve()
    for source in [scratch, *evidence]:
        path = source.resolve()
        if resolved == path or resolved in path.parents or path in resolved.parents:
            raise ValueError("Output overlaps scratch or protected evidence")


def frozen_memberships(path: Path) -> list[tuple[set[str], set[str]]]:
    if sha256(path) != SOURCE_SPLIT_SHA:
        raise ValueError("Frozen split checksum changed")
    all_folds = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(all_folds, list) or len(all_folds) != 5:
        raise ValueError("Wrong number of frozen splits")
    result = []
    for i, item in enumerate(all_folds):
        tr, val = item["train"], item["val"]
        if len(tr) != FOLD_SIZES[i][0] or len(val) != FOLD_SIZES[i][1]:
            raise ValueError("Frozen fold patient count changed")
        if len(set(tr)) != len(tr) or len(set(val)) != len(val) or set(tr) & set(val):
            raise ValueError("Duplicate/leaking case within fold")
        result.append((set(tr), set(val)))
    universe = result[0][0] | result[0][1]
    if len(universe) != 149 or any(tr | val != universe for tr, val in result):
        raise ValueError("Inconsistent 149-case frozen universe")
    if len(set().union(*(v for _, v in result))) != 149 or sum(map(lambda x: len(x[1]), result)) != 149:
        raise ValueError("Outer held-out patient appears twice")
    return result


def validate_virtual_record(item: dict, fold: int, tr: set[str], val: set[str]) -> dict:
    """Check both arms' EXACT image and label path inventories, not just counts."""
    if (item.get("fold") != fold or set(item.get("train_ids_PRIVATE", [])) != tr
            or set(item.get("heldout_ids_PRIVATE", [])) != val
            or len(item.get("train_ids_PRIVATE", [])) != len(tr)
            or len(item.get("heldout_ids_PRIVATE", [])) != len(val)
            or item.get("approved_for_dataset_materialization") is not False):
        raise ValueError(f"Fold {fold}: memberships/status incorrect")
    arms = item.get("arms", {})
    if set(arms) != {"506", "507"}:
        raise ValueError("Expected two virtual arms")
    for arm in ARMS:
        a = arms[str(arm)]
        n = 1 if arm == 506 else 5
        if (a.get("arm") != arm or a.get("numTraining") != len(tr)
                or a.get("numHeldout") != len(val)
                or a.get("actual_nifti_written") is not False
                or a.get("labels_read") is not False
                or a.get("labelsTs_declared") != 0
                or a.get("imagesTr_declared") != len(tr)*n
                or a.get("imagesTs_declared") != len(val)*n
                or a.get("labelsTr_declared") != len(tr)
                or a.get("ncct_copy_policy") != "future_507_0000_shall_copy_exact_506_0000_bytes"):
            raise ValueError(f"Fold {fold} arm {arm}: status/count/NCCT contract changed")
        filenames = a.get("expected_files_PRIVATE", {})
        expected = {
            "imagesTr": {f"{c}_{i:04d}.nii.gz" for c in tr for i in range(n)},
            "imagesTs": {f"{c}_{i:04d}.nii.gz" for c in val for i in range(n)},
            "labelsTr": {f"{c}.nii.gz" for c in tr},
            "labelsTs": set(),
        }
        if set(filenames) != set(expected):
            raise ValueError("Missing expected image/label partitions")
        for group, correct in expected.items():
            actual = filenames[group]
            if len(actual) != len(correct) or set(actual) != correct:
                raise ValueError(f"Fold {fold}, arm {arm}: {group} inventory mismatch")
        if a.get("channel_names") != {str(i): "noNorm" for i in range(n)}:
            raise ValueError("Second nnU-Net normalization not permitted")
    return {"fold": fold, "train_ids_PRIVATE": sorted(tr), "heldout_ids_PRIVATE": sorted(val),
            "input_manifest_sha256": item.get("manifest_sha256"),
            "train_images_506": len(tr), "train_images_507": 5*len(tr),
            "heldout_images_506": len(val), "heldout_images_507": 5*len(val),
            "train_labels_each_arm": len(tr), "heldout_labels_for_trainer": 0}


def verify_inputs(plan: dict, virtual: dict, preflight: dict, gate: dict,
                  splits: Path, plans: Path, folds: Path) -> list[dict]:
    if sha256(plans) != SOURCE_PLANS_SHA:
        raise ValueError("Frozen source plan changed")
    members = frozen_memberships(splits)
    if (plan.get("schema") != "WU20G-physical-staging-plan-v1"
            or plan.get("status") != "ENGINEERING_PHYSICAL_STAGING_PLAN_COMPLETE_NOT_AUTHORIZED"
            or plan.get("n_patients") != 149 or plan.get("n_image_headers_verified") != 745
            or plan.get("n_label_stat_checks") != 149 or plan.get("virtual_fold_count") != 5
            or plan.get("training_ready") is not False
            or plan.get("staging_materialized") is not False
            or plan.get("real_patient_voxels_read") is not False
            or plan.get("label_voxels_read") is not False):
        raise ValueError("WU20G physical plan identity/governance mismatch")
    hold = plan.get("training_hold_plan", {})
    if (hold.get("training_ready") is not False or hold.get("training_commands_generated") is not False
            or hold.get("default_nnunetv2_train_allowed") is not False
            or hold.get("target_epochs_per_arm_fold") != 1000
            or hold.get("checkpoint_policy") != "fixed-final-epoch-only; requires trainer audit"
            or hold.get("nnunet_version") != "2.8.1"):
        raise ValueError("Unapproved trainer or epoch rule changed")
    if (virtual.get("schema") != "WU20F-virtual-staging-v1"
            or virtual.get("status") != "VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE"
            or virtual.get("staging_materialized") is not False
            or virtual.get("training_ready") is not False
            or virtual.get("n_patients") != 149
            or virtual.get("image_voxels_read") is not False
            or virtual.get("label_voxels_read") is not False
            or virtual.get("frozen_split_sha256") != SOURCE_SPLIT_SHA
            or virtual.get("original_plans_sha256") != SOURCE_PLANS_SHA):
        raise ValueError("WU20F private virtual inventory invalid")
    if (preflight.get("status") != "VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE"
            or preflight.get("training_ready") is not False
            or preflight.get("n_source_image_headers_checked") != 745
            or preflight.get("n_patients") != 149):
        raise ValueError("WU20F public preflight incomplete")
    if (gate.get("schema") != "WU20E-review-gate-v1"
            or gate.get("training_ready") is not False
            or gate.get("status") != "BLOCKED_PENDING_INDEPENDENT_REVIEW_AND_STAGING"
            or gate.get("n_patients") != 149 or gate.get("n_review_overlays") != 15):
        raise ValueError("Prospective approval gate is not unchanged/unapproved")
    records = virtual.get("fold_inventories", [])
    if len(records) != 5 or {x.get("fold") for x in records} != set(range(5)):
        raise ValueError("Five fold virtual inventory missing or repeated")
    if plan.get("source_folds_sha256", {}).keys() != {str(i) for i in range(5)}:
        raise ValueError("All five fold-source hashes are required")
    result = []
    for i in range(5):
        row = next(x for x in records if x["fold"] == i)
        manifest = folds / f"WU20B_fold{i}_parameters_PRIVATE.json"
        if (not manifest.is_file() or sha256(manifest) != plan["source_folds_sha256"][str(i)]
                or row.get("manifest_sha256") != sha256(manifest)):
            raise ValueError(f"Fold {i}: calibration manifest changed")
        result.append(validate_virtual_record(row, i, *members[i]))
    return result


def review_status(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not {"case_id", "decision", "reviewer", "review_date"}.issubset(reader.fieldnames or []):
            raise ValueError("Clinical review columns missing")
        rows = list(reader)
    if len(rows) != 15 or len({r["case_id"] for r in rows}) != 15:
        raise ValueError("The review roster must contain 15 distinct overlays")
    pending = sum((r.get("decision") or "").strip().upper() == "PENDING" for r in rows)
    approved = sum((r.get("decision") or "").strip().upper() == "APPROVED" for r in rows)
    rejected = len(rows) - pending - approved
    return {"total": 15, "pending": pending, "approved_unverified": approved,
            "other_or_rejected": rejected, "authentic_clinical_signature_verified": False}


def sequential_storage(wu20g_estimate: dict, available_bytes: int) -> dict:
    images = wu20g_estimate.get("uncompressed_images_bytes")
    labels = wu20g_estimate.get("reference_label_copy_bytes")
    total = wu20g_estimate.get("combined_uncompressed_bytes")
    if (any(type(x) is not int or x <= 0 or x >= MAX_SAFE_INT for x in (images, labels, total))
            or images + labels != total or images % 5):
        raise ValueError("Untrusted five-fold byte budget")
    # Each fold contains every image once in 506 and 507: all image budgets equal.
    # Each patient is a training label in FOUR of the five folds, for TWO arms.
    # A single fold has at most two complete source-label inventories; divide
    # the eight-copy five-fold label total by four for a conservative upper bound.
    if labels % 4:
        raise ValueError("Original label-copy budget not divisible by four")
    images_per_fold = images // 5
    labels_one_fold_upper = labels // 4
    bound = images_per_fold + labels_one_fold_upper
    reservation = 2 * bound
    if type(available_bytes) is not int or available_bytes < 0:
        raise ValueError("Invalid scratch free byte count")
    return {
        "image_bytes_each_fold_uncompressed": images_per_fold,
        "label_bytes_each_fold_upper_bound": labels_one_fold_upper,
        "one_fold_combined_upper_bound_bytes": bound,
        "conservative_one_fold_2x_reservation_bytes": reservation,
        "five_fold_all_at_once_2x_reservation_bytes": 2 * total,
        "current_scratch_free_bytes": available_bytes,
        "provisional_staging_space_meets_2x": available_bytes >= reservation,
        "full_training_capacity_qualified": False,
        "additional_preprocessing_checkpoint_and_archive_space": "UNKNOWN_NOT_ESTIMATED",
        "gzip_prediction_valid": False,
    }


def make_fold_schedule(records: list[dict]) -> list[dict]:
    if len(records) != 5 or [x["fold"] for x in records] != list(range(5)):
        raise ValueError("Five immutable folds required, in order")
    schedule = []
    for row in records:
        i = row["fold"]
        schedule.append({
            "fold": i,
            "phase": "HOLD_PENDING_INDEPENDENT_APPROVAL",
            "isolation": f"scratch/outer_fold_{i}/candidate_arms_506_507",
            "plan_only": True,
            "source_train_ids_PRIVATE": row["train_ids_PRIVATE"],
            "source_outer_heldout_ids_PRIVATE": row["heldout_ids_PRIVATE"],
            "allowed_train_labels_PRIVATE": [f"{c}.nii.gz" for c in row["train_ids_PRIVATE"]],
            "outer_heldout_label_access_allowed": False,
            "checkpoint_selection": "fixed_epoch_1000_only",
            "proposed_steps_NOT_EXECUTABLE": [
                "independent_signed_protocol_review",
                "authorize_physical_staging_in_separate_PR",
                "materialize_exact_one_fold_only_after_release",
                "audit_label_geometry_and_NCCT_hashes",
                "audit_actual_preprocessed_dataset_and_trainer_access",
                "train_1000_epochs_after_explicit_release_only",
                "archive_model_and_inference_receipts_before_disposal",
                "manual_cleanup_only_after_provenance_preserved",
            ],
            "may_stage": False,
            "may_train": False,
            "may_delete": False,
        })
    return schedule


def create_plan(wu20g_plan: Path, wu20f_virtual: Path, wu20f_preflight: Path,
                wu20e_gate: Path, policy_file: Path, review_csv: Path,
                splits: Path, plans: Path, folds: Path, scratch: Path, output: Path) -> dict:
    evidence = [wu20g_plan, wu20f_virtual, wu20f_preflight, wu20e_gate, policy_file,
                review_csv, splits, plans, folds]
    protect_outputs(output, scratch, evidence)
    # BEFORE verifying inputs, invalidate any earlier completed report. If
    # authentication fails, the prior PRIVATE plan is archived as not current.
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "WU20H_sequential_aggregate.json", {
        "schema": SCHEMA, "status": "INCOMPLETE", "training_ready": False,
    })
    prior = output / "WU20H_sequential_PRIVATE.json"
    if prior.exists():
        archive = output / "superseded_NOT_CURRENT"
        archive.mkdir(exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        os.replace(prior, archive / f"WU20H_sequential_{now}_OLD_PRIVATE.json")
    plan = load_json(wu20g_plan)
    virtual = load_json(wu20f_virtual)
    preflight = load_json(wu20f_preflight)
    gate = load_json(wu20e_gate)
    policy = load_json(policy_file)
    canonical_digest = hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if (plan.get("source_preflight_sha256") != sha256(wu20f_preflight)
            or plan.get("source_approval_evidence_sha256") != sha256(wu20e_gate)
            or plan.get("source_policy_sha256") != sha256(policy_file)
            or policy.get("schema") != "WU20E-proposed-policy-v1"
            or policy.get("training_authorized") is not False
            or policy.get("status") != "UNAPPROVED_REQUIRES_INDEPENDENT_REVIEW"
            or policy.get("source_split_sha256") != SOURCE_SPLIT_SHA
            or policy.get("original_plans_sha256") != SOURCE_PLANS_SHA
            or canonical_digest != gate.get("candidate_policy_sha256")):
        raise ValueError("WU20E/WU20G policy/evidence hashes or release status changed")
    records = verify_inputs(plan, virtual, preflight, gate, splits, plans, folds)
    reviews = review_status(review_csv)
    if not scratch.is_dir() or scratch.is_symlink():
        raise ValueError("Scratch must be a pre-existing real directory")
    free = shutil.disk_usage(scratch).free
    storage = sequential_storage(plan["storage_estimate"], free)
    schedule = make_fold_schedule(records)
    result = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SEQUENTIAL_CAPACITY_AND_TRAINER_HOLD_PLAN_COMPLETE",
        "n_patients": 149, "n_folds": 5, "n_arms": 2,
        "source_split_sha256": SOURCE_SPLIT_SHA,
        "source_plans_sha256": SOURCE_PLANS_SHA,
        "evidence_SHA256_PRIVATE": {p.name: sha256(p) for p in evidence if p.is_file()},
        "fold_schedule_PRIVATE": schedule,
        "storage": storage,
        "anatomical_review": reviews,
        "candidate_method_signed": False,
        "outer_heldout_label_access": False,
        "physical_staging_authorized": False,
        "training_commands_emitted": False,
        "gpu_training_started": False,
        "output_cleanup_allowed": False,
        "training_ready": False,
        "next_required_gates": [
            "Independent qualified anatomical and 134-other-case review strategy",
            "Genuine prospective signed clipping/imputation/segmentation protocol",
            "Dedicated physical staging release with provenance and full label QC",
            "Actual trainer IO instrumentation on exact nnU-Net 2.8.1 and heldout denial",
            "Real preprocessing/workspace/model checkpoint disk budget verified",
            "Human research release decision; no automatic conversion of JSON to approval",
        ],
    }
    atomic_json(output / "WU20H_sequential_PRIVATE.json", result)
    public = {key: value for key, value in result.items() if key not in (
        "evidence_SHA256_PRIVATE", "fold_schedule_PRIVATE",
    )}
    public["folds"] = [{"fold": row["fold"], "n_train": len(row["train_ids_PRIVATE"]),
                        "n_heldout": len(row["heldout_ids_PRIVATE"]),
                        "stage_enabled": False, "train_enabled": False}
                       for row in records]
    atomic_json(output / "WU20H_sequential_aggregate.json", public)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", *PROHIBITED_MODES), default="plan")
    parser.add_argument("--wu20g-plan", type=Path)
    parser.add_argument("--wu20f-virtual", type=Path)
    parser.add_argument("--wu20f-preflight", type=Path)
    parser.add_argument("--wu20e-gate", type=Path)
    parser.add_argument("--wu20e-policy", type=Path)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode != "plan":
        raise PermissionError("WU20H cannot stage, train, delete, approve or release; human gates unresolved")
    fields = (args.wu20g_plan, args.wu20f_virtual, args.wu20f_preflight,
              args.wu20e_gate, args.wu20e_policy, args.review_csv, args.splits, args.plans,
              args.folds, args.scratch, args.output)
    if any(f is None for f in fields):
        parser.error("plan requires all private evidence and output paths")
    result = create_plan(*fields)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
