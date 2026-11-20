#!/usr/bin/env python3
"""WU-20E: private read-only evidence/reviewer gate; NEVER grants training approval.

Consumes frozen audit summaries and five fold-specific patient calibration files.
Creates private machine-readable QC evidence. Does not access lesion labels,
trained models, DWI/ADC or outcome metrics. Does not process NIfTI volumes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

SPLIT_SHA = "664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027"
PLANS_SHA = "6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca"
CHANNELS = {"NCCT", "TMAX", "CBF", "CBV", "MTT"}
REVIEW_COLUMNS = {
    "case_id", "overlay_filename", "decision", "brain_tissue_preserved",
    "extracranial_tissue_included", "correction_required", "reviewer", "review_date", "notes",
}
EXPECTED_CASES = {"sub_stroke0016", "sub_stroke0113", "sub_stroke0137", "sub_stroke0152"}
SCHEMA = "WU20E-review-gate-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object: {path.name}")
    return obj


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def verify_folds(fold_dir: Path, frozen_splits: Path) -> dict:
    """Verify entire five-fold train-only calibration and exact 149-case partition."""
    if sha256(frozen_splits) != SPLIT_SHA:
        raise ValueError("Frozen five-fold split checksum changed")
    with frozen_splits.open(encoding="utf-8") as stream:
        original_splits = json.load(stream)
    if not isinstance(original_splits, list) or len(original_splits) != 5:
        raise ValueError("Expected exact five frozen folds")
    heldout_all: list[str] = []
    inputs: dict[str, str] = {}
    n_cases = []
    bands: dict[str, list[float]] = {"NCCT_center": [], "NCCT_scale": [], "MTT_lower_clip": []}
    for fold in range(5):
        path = fold_dir / f"WU20B_fold{fold}_parameters_PRIVATE.json"
        o = read_json(path)
        if (o.get("fold") != fold or o.get("frozen_split_sha256") != SPLIT_SHA
                or o.get("training_ready") is not False
                or o.get("status") != "TRAIN_ONLY_FIT_COMPLETE_QC_PENDING"):
            raise ValueError(f"Fold {fold} identity, split or provisional status mismatch")
        train, heldout = o.get("train_ids"), o.get("heldout_ids")
        original = original_splits[fold]
        if (set(train or []) != set(original.get("train", []))
                or set(heldout or []) != set(original.get("val", []))):
            raise ValueError(f"Fold {fold} does not match immutable five-fold split")
        if not isinstance(train, list) or not isinstance(heldout, list):
            raise ValueError(f"Fold {fold}: missing case inventory")
        if (len(train) != (120 if fold == 4 else 119)
                or len(heldout) != (29 if fold == 4 else 30)
                or len(train) != len(set(train)) or len(heldout) != len(set(heldout))
                or set(train) & set(heldout) or len(set(train) | set(heldout)) != 149):
            raise ValueError(f"Fold {fold}: incomplete or leaking case membership")
        p = o.get("provisional_fit", {})
        if set(p) != CHANNELS:
            raise ValueError(f"Fold {fold}: inconsistent fitted modalities")
        for channel, param in p.items():
            if (param.get("channel") != channel
                    or param.get("n_training_cases") != len(train)
                    or len(param.get("training_ids", [])) != len(train)
                    or set(param.get("training_ids", [])) != set(train)
                    or len(param.get("validation_ids_excluded", [])) != len(heldout)
                    or set(param.get("validation_ids_excluded", [])) != set(heldout)
                    or param.get("n_samples") != len(train) * 4096
                    or param.get("status") != "candidate_not_authorized_for_training"):
                raise ValueError(f"Fold {fold}, {channel}: wrong or leaky fit")
            if not all(isinstance(param.get(k), (float, int)) for k in
                       ("clip_low", "clip_high", "center", "scale")):
                raise ValueError(f"Fold {fold}, {channel}: invalid calibration parameters")
            if param["clip_low"] >= param["clip_high"] or param["scale"] <= 0:
                raise ValueError(f"Fold {fold}, {channel}: degenerate calibrated range")
        bands["NCCT_center"].append(p["NCCT"]["center"])
        bands["NCCT_scale"].append(p["NCCT"]["scale"])
        bands["MTT_lower_clip"].append(p["MTT"]["clip_low"])
        heldout_all.extend(heldout)
        inputs[f"fold{fold}"] = sha256(path)
        n_cases.append({"fold": fold, "n_train": len(train), "n_heldout": len(heldout)})
    if len(heldout_all) != 149 or len(set(heldout_all)) != 149:
        raise ValueError("Five-fold validation partition is incomplete or duplicated")
    return {"folds": n_cases, "fold_file_sha256_PRIVATE": inputs,
            "fold_parameter_ranges": {k: [min(v), max(v)] for k, v in bands.items()},
            "fold_inventory_verified": True}


def verify_numeric_audits(wu20b: dict, wu20c: dict, wu20d: dict) -> dict:
    if not (wu20b.get("status") == "FULL_FOLD_PROVISIONAL_FIT_COMPLETE"
            and wu20b.get("n_patients") == 149
            and wu20b.get("n_folds") == 5
            and wu20b.get("n_channel_case_samples") == 745
            and wu20b.get("n_review_overlays") == 15
            and wu20b.get("training_ready") is False
            and wu20b.get("frozen_split_sha256") == SPLIT_SHA
            and wu20b.get("source_plans_sha256") == PLANS_SHA):
        raise ValueError("WU20B cohort calibration is incomplete or changed")
    q = wu20b.get("qc_aggregate", {})
    if (q.get("n_cases") != 149 or q.get("n_support_below_0_10") != 1
            or q.get("n_support_above_0_40") != 0):
        raise ValueError("WU20B source QC counts changed; independent review required")
    norm = wu20c.get("input_noNorm_checks", {})
    if not (wu20c.get("status") == "CPU_PREPROCESSING_PROBE_COMPLETE_NOT_TRAINING_READY"
            and wu20c.get("source_roi_only_not_full_volume") is True
            and wu20c.get("real_model_inference_performed") is False
            and wu20c.get("training_ready") is False
            and wu20c.get("frozen_split_sha256") == SPLIT_SHA
            and wu20c.get("source_plans_sha256") == PLANS_SHA
            and wu20c.get("locked_architecture_spacing") is True
            and wu20c.get("postresample_ncct_max_abs_difference") == 0
            and wu20c.get("postresample_perfusion_voxels_abs_gt_8") == 0
            and all(norm.get(k, {}).get("actual_normalizer_no_change") is True
                    and norm[k].get("nnunetv2_version") == "2.8.1" for k in ("506", "507"))):
        raise ValueError("WU20C technical probe identity or results fail")
    statuses = wu20d.get("case_status", {})
    perf_overshoot = wu20d.get("per_case_postresample_overshoot_voxels", {})
    mtt_input_nonfinite = wu20d.get("per_case_raw_mtt_nonfinite", {})
    if not (set(statuses) == EXPECTED_CASES
            and all(v == "FULL_VOLUME_TECHNICAL_PASS" for v in statuses.values())
            and wu20d.get("patients_qualified") == 4
            and wu20d.get("dataset504_505_unchanged") is True
            and wu20d.get("real_model_inference_performed") is False
            and wu20d.get("training_ready") is False
            and set(perf_overshoot) == EXPECTED_CASES
            and all(v == 0 for v in perf_overshoot.values())
            and set(mtt_input_nonfinite) == EXPECTED_CASES
            and mtt_input_nonfinite.get("sub_stroke0016") == 9
            and mtt_input_nonfinite.get("sub_stroke0113") == 225045):
        raise ValueError("WU20D full native-volume evidence incomplete or changed")
    return {"149_cases_fitted": True, "four_full_native_cases_passed": True,
            "nnunet_2_8_1_roi_parity": True, "full_cohort_native_preprocessing": False,
            "full_model_forward_inference_tested": False}


def inspect_review(review_csv: Path, overlay_dir: Path) -> dict:
    with review_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not REVIEW_COLUMNS.issubset(set(reader.fieldnames or [])):
            raise ValueError("Review worksheet missing required columns")
        rows = list(reader)
    actual = sorted(overlay_dir.glob("*_support_QC.png"))
    if len(actual) != 15 or len(rows) != 15:
        raise ValueError("Review must contain all 15 source overlays and 15 worksheet rows")
    overlay_by_case = {p.name.removesuffix("_support_QC.png"): p for p in actual}
    csv_ids = [r.get("case_id", "") for r in rows]
    if set(csv_ids) != set(overlay_by_case) or len(set(csv_ids)) != 15:
        raise ValueError("Review worksheet doesn't match original overlay case inventory")
    decisions = Counter()
    review_failures = []
    sha = {}
    for row in rows:
        case = row["case_id"]
        image = overlay_by_case[case]
        if row.get("overlay_filename") != image.name:
            raise ValueError(f"Mismatched overlay filename: {case}")
        if not re.fullmatch(r"sub_stroke\d{4}", case):
            raise ValueError("Unrecognized pseudonymous source case")
        sha[case] = sha256(image)
        decision = (row.get("decision") or "").strip().upper()
        decisions[decision] += 1
        if decision != "APPROVED":
            review_failures.append(f"{case}: review decision {decision or '(blank)'}")
            continue
        if (row.get("brain_tissue_preserved", "").strip().upper() != "YES"
                or row.get("correction_required", "").strip().upper() != "NO"
                or row.get("extracranial_tissue_included", "").strip().upper() not in {"NO", "ACCEPTABLE"}):
            review_failures.append(f"{case}: approval fields incomplete or require correction")
        reviewer = (row.get("reviewer") or "").strip()
        when = (row.get("review_date") or "").strip()
        if not reviewer or not when:
            review_failures.append(f"{case}: reviewer identity/date missing")
        else:
            try:
                review_date = date.fromisoformat(when)
                if review_date > datetime.now(timezone.utc).date():
                    review_failures.append(f"{case}: review date is in future")
            except ValueError:
                review_failures.append(f"{case}: invalid ISO-8601 review date")
    return {"review_rows": len(rows), "overlay_sha256_PRIVATE": sha,
            "review_decisions": dict(decisions), "review_complete": not review_failures,
            "review_blockers": review_failures}


def proposed_policy() -> dict:
    """Pre-declared CANDIDATE; this function cannot authorize clinical suitability."""
    return {
        "schema": "WU20E-proposed-policy-v1",
        "status": "UNAPPROVED_REQUIRES_INDEPENDENT_REVIEW",
        "data": "ISLES2024; frozen 149 cases, five held-out folds",
        "source_split_sha256": SPLIT_SHA,
        "original_plans_sha256": PLANS_SHA,
        "ncct": "train-fold support samples; q0.005-q0.995 clipping; training mean/std",
        "perfusion": "train-fold support samples; q0.005-q0.995 clipping; median/IQR divided by 1.349",
        "missing_perfusion": "replace non-finite with fold-fitted median before clipping; outside support set 0",
        "finite_ncct_nonfinite": "fail-closed; do not impute",
        "support": "provisional_head_tissue_v1 (not a clinically validated brain mask)",
        "perfusion_post_z_cap": 8.0,
        "nnunet_version": "2.8.1",
        "nnunet_channel_normalization": "NoNormalization; input noNorm",
        "model_selection": "no use of original OOF Dice, held-out labels, or outcome metrics for preprocessing decisions",
        "required_reviews": [
            "independent anatomical/clinical review, including 15 prioritized overlays and cohort QC",
            "independent methodological review of clipping, median replacement, the ±8 cap and scan coverage",
            "independent protocol signature with image/policy digests; professional credentials authenticated outside automation",
            "full-cohort staging/inference-shape and data-leakage checks before training",
        ],
        "training_authorized": False,
    }


def evaluate(wu20b: Path, wu20c: Path, wu20d: Path, folds: Path,
             splits: Path, review: Path, overlays: Path, output: Path) -> dict:
    protected = [folds.resolve(), overlays.resolve()]
    target = output.resolve()
    if any(target == source or source in target.parents for source in protected):
        raise ValueError("Release output would overwrite frozen or private input evidence")
    numeric = verify_numeric_audits(read_json(wu20b), read_json(wu20c), read_json(wu20d))
    fold_info = verify_folds(folds, splits)
    review_info = inspect_review(review, overlays)
    policy = proposed_policy()
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    blockers = list(review_info["review_blockers"])
    blockers.extend([
        "Independent policy review/signature absent; reviewer credential and attachment authenticity require manual checks",
        "Priority 15 overlays are not a clinical validation of all 149 masks",
        "No full-cohort Dataset506/507 staging dry-run or operational training release signoff",
        "No trained-model inference possible yet; independent synthetic forward/inference-path smoke remains pending",
    ])
    result = {
        "schema": SCHEMA,
        "status": "BLOCKED_PENDING_INDEPENDENT_REVIEW_AND_STAGING",
        "training_ready": False,
        "review_ready": True,
        "technical_evidence_checked": numeric,
        "fold_integrity": fold_info,
        "review": review_info,
        "source_evidence_sha256_PRIVATE": {
            "wu20b": sha256(wu20b), "wu20c": sha256(wu20c), "wu20d": sha256(wu20d),
            "review_csv": sha256(review), "frozen_split": sha256(splits),
        },
        "policy_sha256": digest,
        "blocked_release_gates": blockers,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "WU20E_policy_PROPOSED_NOT_APPROVED_PRIVATE.json", policy)
    atomic_json(output / "WU20E_release_gate_PRIVATE.json", result)
    # This report is private even though numeric results are aggregate,
    # because it embeds patient-level review inventory and digest metadata.
    aggregate = {
        "schema": SCHEMA,
        "status": result["status"],
        "training_ready": False,
        "review_ready": True,
        "n_patients": 149,
        "n_folds": 5,
        "n_full_native_volume_cases_technically_passed": 4,
        "n_review_overlays": 15,
        "n_review_decisions_pending": len(review_info["review_blockers"]),
        "review_decisions": review_info["review_decisions"],
        "candidate_policy_sha256": digest,
        "remaining_independent_approvals": 2,
        "full_cohort_staged": False,
    }
    atomic_json(output / "WU20E_release_gate_aggregate.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("wu20b", "wu20c", "wu20d", "folds", "splits", "review", "overlays", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(evaluate(a.wu20b, a.wu20c, a.wu20d, a.folds,
                              a.splits, a.review, a.overlays, a.output), indent=2))


if __name__ == "__main__":
    main()
