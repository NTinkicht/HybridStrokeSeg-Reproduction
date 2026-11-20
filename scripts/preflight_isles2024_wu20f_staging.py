#!/usr/bin/env python3
"""WU20F: fail-closed image-header-only, five-fold *virtual* staging preflight.

Reads no image voxels, segmentation voxels, patient outcomes, or model weights.
Never creates trainable nnU-Net datasets, transforms, or approves training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qualify_isles2024_nnunet_runtime import (
    SOURCE_PLANS_SHA,
    assert_locked_plans,
    build_locked_plans,
    validate_locked_fold_manifest,
)

from hybridstrokeseg.wu20_fold_preprocessing import (
    CHANNELS,
    FROZEN_SPLIT_SHA,
    file_sha256,
    validated_split,
)

SCHEMA = "WU20F-virtual-staging-v1"
CASE_SCHEMA = "WU20F-image-header-v1"
CHANNEL_INDICES = {ch: i for i, ch in enumerate(CHANNELS)}
EXPECTED_LABELS = {"background": 0, "final_infarct": 1}
CASE_COUNTS = (119, 119, 119, 119, 120)


def load_object(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(result, dict):
        raise TypeError(f"Expected JSON object: {path.name}")
    return result


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf8") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def protect_input_trees(output: Path, raw: Path, splits: Path, plans: Path,
                        folds: Path, policy: Path, release_gate: Path) -> None:
    dest = output.resolve()
    protected = (raw.resolve(), raw.parent.resolve(), splits.parent.resolve(),
                 plans.parent.resolve(), folds.resolve(), policy.resolve(),
                 release_gate.resolve())
    if any(dest == p or p in dest.parents for p in protected):
        raise ValueError("Output cannot modify protected source/evidence trees")
    if output.is_symlink():
        raise ValueError("Symlink report output is prohibited")


def verify_frozen_cohort(raw: Path, splits: Path, plans: Path,
                         folds: Path, gate: Path, policy: Path) -> tuple[dict, list, list]:
    if file_sha256(splits) != FROZEN_SPLIT_SHA or file_sha256(plans) != SOURCE_PLANS_SHA:
        raise ValueError("Frozen split or Dataset505 plan checksum mismatch")
    dataset = load_object(raw / "dataset.json")
    if (tuple(dataset.get("channel_names", {}).get(str(i), "").upper() for i in range(5)) != CHANNELS
            or dataset.get("numTraining") != 149 or dataset.get("file_ending") != ".nii.gz"
            or dataset.get("labels") != EXPECTED_LABELS):
        raise ValueError("Dataset505 source modality/label metadata differs")
    original = load_object(plans)
    release = load_object(gate)
    method = load_object(policy)
    if (release.get("schema") != "WU20E-review-gate-v1"
            or release.get("training_ready") is not False
            or release.get("full_cohort_staged") is not False
            or method.get("status") != "UNAPPROVED_REQUIRES_INDEPENDENT_REVIEW"
            or method.get("training_authorized") is not False
            or method.get("source_split_sha256") != FROZEN_SPLIT_SHA
            or method.get("original_plans_sha256") != SOURCE_PLANS_SHA
            or hashlib.sha256(json.dumps(method, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            != release.get("candidate_policy_sha256")):
        raise ValueError("Unapproved WU20E policy/evidence missing or altered")
    summaries = []
    for fold in range(5):
        train, heldout = validated_split(splits, fold)
        path = folds / f"WU20B_fold{fold}_parameters_PRIVATE.json"
        manifest = load_object(path)
        validate_locked_fold_manifest(manifest, train, heldout, fold)
        for channel, params in manifest["provisional_fit"].items():
            if (len(params["training_ids"]) != len(train) or
                    len(params["validation_ids_excluded"]) != len(heldout) or
                    params["n_samples"] != len(train) * 4096):
                raise ValueError(f"Fold {fold}/{channel}: repeated or incomplete fit samples")
        if len(train) != CASE_COUNTS[fold] or len(heldout) != 149 - CASE_COUNTS[fold]:
            raise ValueError("Frozen fold cardinality mismatch")
        summaries.append({"fold": fold, "train": train, "heldout": heldout,
                          "manifest_sha256": file_sha256(path)})
    expected = set(summaries[0]["train"]) | set(summaries[0]["heldout"])
    if len(expected) != 149 or any(set(x["train"]) | set(x["heldout"]) != expected for x in summaries):
        raise ValueError("Inconsistent 149-case universe")
    if len(set().union(*(set(x["heldout"]) for x in summaries))) != 149:
        raise ValueError("Held-out patient is missing or duplicated")
    if sum(len(x["heldout"]) for x in summaries) != 149:
        raise ValueError("Holdout assignments duplicate cases")
    for dataset_id in (506, 507):
        candidate = build_locked_plans(original, dataset_id=dataset_id)
        assert_locked_plans(original, candidate, dataset_id=dataset_id)
    return original, summaries, sorted(expected)


def input_signature(paths: list[Path]) -> dict:
    """Metadata-level identity; not a content hash of multi-GB source images."""
    parts = []
    for path in paths:
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 256:
            raise ValueError(f"Missing/empty NIfTI source: {path.name}")
        parts.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"files": parts, "sha256": hashlib.sha256(
        json.dumps(parts, sort_keys=True).encode("utf8")
    ).hexdigest()}


def inspect_image_header(path: Path) -> dict:
    """Read compressed NIfTI header, never load or inspect image voxel arrays."""
    import nibabel as nib

    image = nib.load(str(path))
    shape = tuple(int(v) for v in image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    if len(shape) != 3 or any(n < 8 for n in shape):
        raise ValueError(f"Invalid 3-D source shape: {path.name}")
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.isfinite(spacing).all() or min(spacing) <= 0
            or not np.isfinite(np.linalg.det(affine[:3, :3]))
            or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-10):
        raise ValueError(f"Invalid source geometry: {path.name}")
    return {"shape": list(shape), "spacing_mm": list(spacing),
            "affine": affine.round(7).tolist(), "dtype": str(image.get_data_dtype())}


def inspect_case(raw: Path, case: str) -> dict:
    paths = [raw / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)]
    label = raw / "labelsTr" / f"{case}.nii.gz"
    if not label.is_file() or label.stat().st_size <= 256:
        raise ValueError(f"Missing source label (not read): {case}")
    signature = input_signature([*paths, label])
    headers = [inspect_image_header(p) for p in paths]
    if any(h["shape"] != headers[0]["shape"] or not np.allclose(
        h["affine"], headers[0]["affine"], atol=1e-4, rtol=0
    ) for h in headers[1:]):
        raise ValueError(f"Mismatch in native five-channel geometry: {case}")
    # File *presence only*: never read lesion label bytes or headers, even on held-out cases.
    return {"schema": CASE_SCHEMA, "case_id": case, "status": "HEADER_VALID",
            "image_source_signature": signature, "reference_header": headers[0],
            "input_channels": list(CHANNELS), "label_exists_no_read": True,
            "voxel_data_read": False, "training_ready": False}


def verify_cached_case(case: str, cached: dict, raw: Path) -> bool:
    if cached.get("schema") != CASE_SCHEMA or cached.get("case_id") != case:
        return False
    if cached.get("status") != "HEADER_VALID" or cached.get("label_exists_no_read") is not True:
        return False
    paths = [raw / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)]
    label = raw / "labelsTr" / f"{case}.nii.gz"
    if not label.is_file() or label.stat().st_size <= 256:
        return False
    return cached.get("image_source_signature") == input_signature([*paths, label])


def virtual_fold_inventory(fold_record: dict) -> dict:
    """Expected paths are declarations, not actual generated NIfTI files."""
    fold = fold_record["fold"]
    tr, val = fold_record["train"], fold_record["heldout"]
    def describe(arm: int) -> dict:
        n = 1 if arm == 506 else 5
        return {"arm": arm, "virtual_workspace": f"isolated_fold_{fold}/Dataset{arm}",
                "numTraining": len(tr), "numHeldout": len(val),
                "imagesTr_declared": len(tr) * n, "imagesTs_declared": len(val) * n,
                "labelsTr_declared": len(tr), "labelsTs_declared": 0,
                "expected_files_PRIVATE": {
                    "imagesTr": [f"{case}_{i:04d}.nii.gz" for case in tr for i in range(n)],
                    "imagesTs": [f"{case}_{i:04d}.nii.gz" for case in val for i in range(n)],
                    "labelsTr": [f"{case}.nii.gz" for case in tr],
                    "labelsTs": [],
                },
                "channel_names": {str(i): "noNorm" for i in range(n)},
                "ncct_copy_policy": "future_507_0000_shall_copy_exact_506_0000_bytes",
                "actual_nifti_written": False, "labels_read": False}
    return {"fold": fold, "train_ids_PRIVATE": tr, "heldout_ids_PRIVATE": val,
            "manifest_sha256": fold_record["manifest_sha256"],
            "arms": {"506": describe(506), "507": describe(507)},
            "approved_for_dataset_materialization": False}


def run_preflight(raw: Path, splits: Path, plans: Path, folds: Path, gate: Path,
                  policy: Path, output: Path, *, limit_new_cases: int = 0) -> dict:
    if limit_new_cases < 0:
        raise ValueError("Negative scan limit")
    protect_input_trees(output, raw, splits, plans, folds, policy, gate)
    _plans, folds_verified, cases = verify_frozen_cohort(raw, splits, plans, folds, gate, policy)
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous complete aggregate before continuing work.
    atomic_json(output / "WU20F_preflight_aggregate.json", {
        "schema": SCHEMA, "status": "INCOMPLETE", "training_ready": False,
        "staging_materialized": False,
    })
    # Preserve prior result but prevent a completed old inventory being mistaken
    # for the current run when a patient later fails or a Colab run disconnects.
    former = output / "WU20F_virtual_staging_PRIVATE.json"
    if former.exists():
        archive = output / "superseded_not_current"
        archive.mkdir(exist_ok=True)
        date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        os.replace(former, archive / f"WU20F_virtual_staging_{date}_OLD_PRIVATE.json")
    checkpoints = output / "patient_header_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    completed = []
    errors = []
    created = 0
    for i, case in enumerate(cases, 1):
        target = checkpoints / f"{case}_header_PRIVATE.json"
        try:
            if target.exists():
                cached = load_object(target)
                if not verify_cached_case(case, cached, raw):
                    raise ValueError("Existing header checkpoint/source signature changed")
                result = cached
                status = "CACHED"
            else:
                if limit_new_cases and created >= limit_new_cases:
                    continue
                result = inspect_case(raw, case)
                atomic_json(target, result)
                created += 1
                status = "CHECKED"
            completed.append(case)
            print(f"[{i:03d}/149] {case}: {status}", flush=True)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            # Do not silently exclude a patient or proceed with a partial cohort.
            errors.append({"case_id_PRIVATE": case, "error_type": type(exc).__name__,
                           "message": str(exc)[:220]})
            print(f"[{i:03d}/149] {case}: BLOCKED {type(exc).__name__}", flush=True)
    if len(completed) == 149 and not errors:
        private = {"schema": SCHEMA, "status": "VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE",
                   "training_ready": False, "staging_materialized": False,
                   "image_voxels_read": False, "label_voxels_read": False,
                   "n_patients": len(completed), "n_source_image_headers_checked": 745,
                   "n_label_file_presence_checks": 149,
                   "frozen_split_sha256": FROZEN_SPLIT_SHA,
                   "original_plans_sha256": SOURCE_PLANS_SHA,
                   "policy_file_sha256": file_sha256(policy),
                   "wu20e_gate_sha256": file_sha256(gate),
                   "original_504_505_modified": False,
                   "fold_inventories": [virtual_fold_inventory(f) for f in folds_verified],
                   "blocked_release_gates": [
                       "15 prioritized anatomical overlays require independent review",
                       "Provisional clipping and imputation policy not signed/approved",
                       "Virtual metadata inventory is NOT transformed nnU-Net data staging",
                       "Reference label headers/contents not inspected at all",
                       "Actual nnU-Net full-cohort preprocessing and model training not performed",
                   ]}
        atomic_json(output / "WU20F_virtual_staging_PRIVATE.json", private)
    else:
        private = {"schema": SCHEMA, "status": "PARTIAL_OR_FAILED",
                   "training_ready": False, "staging_materialized": False,
                   "n_patients": len(completed), "errors_PRIVATE": errors,
                   "blocked_release_gates": ["All 149 headers must pass without alteration"]}
        atomic_json(output / "WU20F_virtual_staging_PARTIAL_PRIVATE.json", private)
    redacted = {k: v for k, v in private.items() if k not in (
        "fold_inventories", "errors_PRIVATE", "policy_file_sha256",
        "wu20e_gate_sha256",
    )}
    if "fold_inventories" in private:
        redacted["folds"] = [{"fold": r["fold"],
                               "n_train": len(r["train_ids_PRIVATE"]),
                               "n_heldout": len(r["heldout_ids_PRIVATE"]),
                               "images506": r["arms"]["506"]["imagesTr_declared"] + r["arms"]["506"]["imagesTs_declared"],
                               "images507": r["arms"]["507"]["imagesTr_declared"] + r["arms"]["507"]["imagesTs_declared"]}
                              for r in private["fold_inventories"]]
    redacted["n_errors"] = len(errors)
    redacted["generated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "WU20F_preflight_aggregate.json", redacted)
    return redacted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw505", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--wu20e-gate", type=Path, required=True)
    p.add_argument("--wu20e-policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit-new-cases", type=int, default=0)
    args = p.parse_args()
    result = run_preflight(args.raw505, args.splits, args.plans, args.folds,
                           args.wu20e_gate, args.wu20e_policy, args.output,
                           limit_new_cases=args.limit_new_cases)
    print(json.dumps(result, indent=2))
    if result["status"] != "VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
