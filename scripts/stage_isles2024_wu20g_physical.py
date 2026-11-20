#!/usr/bin/env python3
"""WU20G: guarded physical staging implementation and private storage plan.

This module implements staging for synthetic qualification. Production CLI deliberately
has NO physical-staging entrypoint until authenticated independent approvals and a
separate reviewed release PR exist. Never trains, reads heldout labels or edits 504/505.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from preflight_isles2024_wu20f_staging import (
    CASE_SCHEMA,
    load_object,
    protect_input_trees,
    verify_cached_case,
    verify_frozen_cohort,
)
from qualify_isles2024_nnunet_runtime import (
    PINNED_NNUNET,
    SOURCE_PLANS_SHA,
    assert_locked_plans,
    build_locked_plans,
)
from stage_isles2024_fold_pilot import _save_preprocessed

from hybridstrokeseg.wu20_fold_preprocessing import (
    CHANNELS,
    candidate_arms,
    file_sha256,
    ncct_head_tissue_support,
    no_norm_dataset_json,
)

SCHEMA = "WU20G-physical-staging-plan-v1"
RECEIPT_SCHEMA = "WU20G-synthetic-case-receipt-v1"
ALLOWED_MODE = ("plan", "synthetic", "stage")
CASE_PATTERN = re.compile(r"^sub_stroke\d{4}$")
TRAINING_BLOCKERS = (
    "Independent qualified review of all 15 priority overlays and remaining cohort strategy",
    "Prospective clinical/scientific approval of masking and clipping/imputation",
    "Authenticated frozen preprocessing protocol and separate research-release PR",
    "Full-cohort physical staging and actual label-geometry qualification not executed",
    "Audited no-heldout-label training trainer and OOF evaluation protocol pending",
)


def sha_text(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def reject_unsafe_workspace(workspace: Path, protected: list[Path]) -> None:
    """Guard symlink ancestors and any input directory (including the Drive mount)."""
    root = workspace.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Symlink component in staging workspace")
    resolved = root.resolve()
    if "drive" in (p.lower() for p in resolved.parts):
        raise ValueError("Do not stage NIfTI files on mounted Drive")
    for source in protected:
        p = source.resolve()
        if resolved == p or p in resolved.parents or resolved in p.parents:
            raise ValueError("Staging workspace overlaps a frozen source or evidence root")


def estimate_storage(header_cases: list[dict], label_bytes: int) -> dict:
    """Estimate *uncompressed* float32 output; never guess gzip compression."""
    if len(header_cases) != 149 or label_bytes <= 0:
        raise ValueError("Require all 149 verified cases and positive label bytes")
    volume_counts = []
    for case in header_cases:
        header = case.get("reference_header", {})
        shape = header.get("shape", [])
        if len(shape) != 3 or any(not isinstance(v, int) or v < 8 for v in shape):
            raise ValueError("Unverified source geometry in storage planning")
        volume_counts.append(int(np.prod(shape)))
    # Every patient is in exactly five folds, one and five channels per fold.
    images_uncompressed = sum(volume_counts) * 5 * 6 * 4
    # Each patient is TRAIN in 4 folds, in both arms (8 copies total).
    label_copies = label_bytes * 8
    combined = images_uncompressed + label_copies
    return {
        "uncompressed_images_bytes": images_uncompressed,
        "reference_label_copy_bytes": label_copies,
        "combined_uncompressed_bytes": combined,
        "recommended_free_bytes_at_least": combined * 2,
        "estimate_basis": "149 cases * 5 folds * (1+5) float32 channels; labels 4 folds * 2 arms",
        "gzip_size_predictable": False,
        "clinical_output_staged": False,
    }


def make_training_hold_plan() -> dict:
    """No executable GPU command: nnU-Net's default validation consumes labels."""
    return {
        "nnunet_version": PINNED_NNUNET,
        "target_epochs_per_arm_fold": 1000,
        "configuration": "3d_fullres",
        "num_outer_folds": 5,
        "arms": [506, 507],
        "checkpoint_policy": "fixed-final-epoch-only; requires trainer audit",
        "heldout_labels_available_to_trainer": False,
        "default_nnunetv2_train_allowed": False,
        "training_commands_generated": False,
        "reason": "A default nnU-Net validation split can consume held-out labels; custom trainer and fold handling not yet audited",
        "training_ready": False,
    }


def check_preflight(preflight: dict) -> None:
    if (preflight.get("schema") != "WU20F-virtual-staging-v1"
            or preflight.get("status") != "VIRTUAL_FULL_COHORT_PREFLIGHT_COMPLETE"
            or preflight.get("n_errors") != 0 or preflight.get("n_patients") != 149
            or preflight.get("n_source_image_headers_checked") != 745
            or preflight.get("n_label_file_presence_checks") != 149
            or preflight.get("training_ready") is not False
            or preflight.get("staging_materialized") is not False
            or preflight.get("image_voxels_read") is not False
            or preflight.get("label_voxels_read") is not False):
        raise ValueError("WU20F verified preflight unavailable/inconsistent")


def plan_staging(raw: Path, splits: Path, plans: Path, folds: Path, wu20e_gate: Path,
                 policy: Path, wu20f_preflight: Path, checkpoints: Path,
                 output: Path, scratch: Path) -> dict:
    protect_input_trees(output, raw, splits, plans, folds, policy, wu20e_gate)
    reject_unsafe_workspace(scratch, [raw, raw.parent, plans.parent, folds, output, policy.parent])
    output.mkdir(parents=True, exist_ok=True)
    # A new invocation cannot inherit an earlier successful status if any source fails.
    atomic_json(output / "WU20G_physical_staging_plan_aggregate.json", {
        "schema": SCHEMA, "status": "INCOMPLETE", "training_ready": False,
        "staging_materialized": False,
    })
    preflight = load_object(wu20f_preflight)
    check_preflight(preflight)
    original, fold_sets, cases = verify_frozen_cohort(raw, splits, plans, folds, wu20e_gate, policy)
    if not original or len(fold_sets) != 5 or len(cases) != 149:
        raise ValueError("Frozen cohort inventory incomplete")
    checked = []
    label_bytes = 0
    for case in cases:
        record_path = checkpoints / f"{case}_header_PRIVATE.json"
        header = load_object(record_path)
        if header.get("schema") != CASE_SCHEMA or not verify_cached_case(case, header, raw):
            raise ValueError(f"Stale or inconsistent patient header checkpoint {case}")
        label_path = raw / "labelsTr" / f"{case}.nii.gz"
        # File stat only, not label header or voxel content.
        label_bytes += label_path.stat().st_size
        checked.append(header)
    estimate = estimate_storage(checked, label_bytes)
    free = shutil.disk_usage(scratch if scratch.exists() else scratch.parent).free
    reserve_ok = free >= estimate["recommended_free_bytes_at_least"]
    candidate_files = [f"fold_{f['fold']}/{name}" for f in fold_sets for name in (
        "Dataset506_ISLES2024_NCCT_CTCTRL",
        "Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST",
    )]
    result = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ENGINEERING_PHYSICAL_STAGING_PLAN_COMPLETE_NOT_AUTHORIZED",
        "n_patients": len(checked),
        "n_image_headers_verified": 745,
        "n_label_stat_checks": 149,
        "virtual_fold_count": len(fold_sets),
        "candidate_workspaces": candidate_files,
        "source_preflight_sha256": file_sha256(wu20f_preflight),
        "source_policy_sha256": file_sha256(policy),
        "source_approval_evidence_sha256": file_sha256(wu20e_gate),
        "source_folds_sha256": {str(f['fold']): f['manifest_sha256'] for f in fold_sets},
        "storage_estimate": estimate,
        "scratch_free_bytes_at_check": free,
        "scratch_meets_conservative_reservation": reserve_ok,
        "staging_materialized": False,
        "real_patient_voxels_read": False,
        "label_voxels_read": False,
        "training_ready": False,
        "blocked_release_gates": list(TRAINING_BLOCKERS),
        "training_hold_plan": make_training_hold_plan(),
    }
    atomic_json(output / "WU20G_physical_staging_plan_PRIVATE.json", result)
    public = {k: v for k, v in result.items() if k not in (
        "source_folds_sha256", "candidate_workspaces", "source_preflight_sha256",
        "source_policy_sha256", "source_approval_evidence_sha256",
    )}
    atomic_json(output / "WU20G_physical_staging_plan_aggregate.json", public)
    return public


def validate_training_label(label_file: Path, reference) -> dict:
    """Read label voxels only when the caller has explicitly selected TRAIN group."""
    import nibabel as nib

    label = nib.load(str(label_file))
    if label.shape != reference.shape or not np.allclose(label.affine, reference.affine, atol=1e-4):
        raise ValueError("Training label geometry mismatch")
    data = label.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.all(np.isin(np.unique(data), [0.0, 1.0])):
        raise ValueError("Training label is not a finite binary lesion mask")
    return {"source_label_sha256_PRIVATE": file_sha256(label_file),
            "source_label_classes_PRIVATE": [int(x) for x in np.unique(data)]}


def verify_written_case(root506: Path, root507: Path, case: str,
                        group: str, *, has_label: bool) -> dict:
    if group not in ("imagesTr", "imagesTs") or not CASE_PATTERN.fullmatch(case):
        raise ValueError("Invalid case/group")
    h506 = root506 / group / f"{case}_0000.nii.gz"
    h507 = root507 / group / f"{case}_0000.nii.gz"
    sha506 = file_sha256(h506)
    if file_sha256(h507) != sha506:
        raise ValueError("NCCT bytes differ between 506 and 507")
    hashes = {"506_NCCT": sha506, "507_NCCT": sha506}
    import nibabel as nib

    template = nib.load(str(h506))
    if not np.isfinite(template.get_fdata(dtype=np.float32)).all():
        raise ValueError("Non-finite staged NCCT")
    for i, ch in enumerate(CHANNELS[1:], start=1):
        p = root507 / group / f"{case}_{i:04d}.nii.gz"
        im = nib.load(str(p))
        if im.shape != template.shape or not np.allclose(im.affine, template.affine, atol=1e-4):
            raise ValueError(f"Staged perfusion geometry mismatch: {ch}")
        values = im.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 8.0001:
            raise ValueError(f"Invalid transformed perfusion values: {ch}")
        hashes[f"507_{ch}"] = file_sha256(p)
    if has_label:
        for d in (root506, root507):
            lab = d / "labelsTr" / f"{case}.nii.gz"
            label = nib.load(str(lab))
            if label.shape != template.shape or not np.allclose(label.affine, template.affine, atol=1e-4):
                raise ValueError("Staged training label geometry mismatch")
        a = file_sha256(root506 / "labelsTr" / f"{case}.nii.gz")
        b = file_sha256(root507 / "labelsTr" / f"{case}.nii.gz")
        if a != b:
            raise ValueError("Training label bytes differ between arms")
        hashes["paired_labelsTr"] = a
    else:
        if any((d / "labelsTr" / f"{case}.nii.gz").exists() for d in (root506, root507)):
            raise ValueError("Held-out label contamination")
    return hashes


def stage_synthetic_case(source: Path, root506: Path, root507: Path, case: str,
                         group: str, params: dict, *, label_file: Path | None = None) -> dict:
    """Exercise the real transform/writer on SYNTHETIC images, never call for real cohort.

    The production entrypoint cannot reach this function without a separate
    future audited release. Synthetic fixtures must use reserved case IDs > 9000.
    """
    if not CASE_PATTERN.fullmatch(case) or int(case[-4:]) < 9000:
        raise PermissionError("Only reserved synthetic fixture cases may be written in WU20G")
    if group not in ("imagesTr", "imagesTs"):
        raise ValueError("Invalid staging group")
    if (group == "imagesTr") != (label_file is not None):
        raise ValueError("Training case requires label; held-out label access prohibited")
    if set(params) != set(CHANNELS):
        raise ValueError("Incorrect calibration modalities")
    import nibabel as nib

    originals = [source / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)]
    input_hashes = [file_sha256(p) for p in originals]
    imgs = [nib.load(str(p)) for p in originals]
    if len({im.shape for im in imgs}) != 1 or any(
        not np.allclose(im.affine, imgs[0].affine, atol=1e-4) for im in imgs[1:]
    ):
        raise ValueError("Source image geometry mismatch")
    if label_file is not None:
        label_audit = validate_training_label(label_file, imgs[0])
    else:
        label_audit = None
    arrays = {ch: im.get_fdata(dtype=np.float32) for ch, im in zip(CHANNELS, imgs)}
    support, mask_qc = ncct_head_tissue_support(arrays["NCCT"])
    a506, a507, diagnostics = candidate_arms(arrays, params, support=support)
    del arrays, support
    gc.collect()
    if input_hashes != [file_sha256(p) for p in originals]:
        raise ValueError("Source NIfTI content changed during staging")
    # Reject old output, even if a previous interrupted process left one file.
    names = [(root506 / group / f"{case}_0000.nii.gz"), *(
        root507 / group / f"{case}_{i:04d}.nii.gz" for i in range(5))]
    if label_file is not None:
        names.extend([d / "labelsTr" / f"{case}.nii.gz" for d in (root506, root507)])
    if any(p.exists() for p in names):
        raise FileExistsError("Existing staged case requires explicit reconciliation, not overwrite")
    with tempfile.TemporaryDirectory(prefix="wu20g_synthetic_commit_") as temporary:
        tmp = Path(temporary)
        x506, x507 = tmp / "506", tmp / "507"
        for i, channel in enumerate(CHANNELS):
            if i == 0:
                _save_preprocessed(x506 / group / f"{case}_0000.nii.gz", a506["NCCT"], imgs[0])
                dest = x507 / group / f"{case}_0000.nii.gz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(x506 / group / f"{case}_0000.nii.gz", dest)
            else:
                _save_preprocessed(x507 / group / f"{case}_{i:04d}.nii.gz", a507[channel], imgs[i])
        if label_file is not None:
            for d in (x506, x507):
                (d / "labelsTr").mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_file, d / "labelsTr" / f"{case}.nii.gz")
        # Validate staging in scratch BEFORE commit to permanent synthetic dirs.
        evidence = verify_written_case(x506, x507, case, group, has_label=label_file is not None)
        for root, staging in ((root506, x506), (root507, x507)):
            for folder in ("imagesTr", "imagesTs", "labelsTr"):
                for src in (staging / folder).glob("*.nii.gz"):
                    dest = root / folder / src.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        raise FileExistsError("Output appeared during staging")
                    os.replace(src, dest)
    committed = verify_written_case(root506, root507, case, group, has_label=label_file is not None)
    if evidence != committed:
        raise ValueError("Staged output integrity changed after commit")
    return {"schema": RECEIPT_SCHEMA, "case": case, "group": group,
            "source_input_sha256_PRIVATE": input_hashes,
            "source_label_validation_PRIVATE": label_audit,
            "output_sha256_PRIVATE": committed,
            "head_support_is_clinically_approved": False,
            "head_support_coverage": mask_qc["coverage_fraction"],
            "candidate_transform_diagnostics": diagnostics,
            "training_ready": False,
            "synthetic_only": True}


def make_synthetic_inputs(folder: Path, case: str, label: bool) -> Path | None:
    import nibabel as nib

    x, y, _ = np.indices((32, 32, 16))
    head = (x-16)**2 + (y-16)**2 < 110
    ncct = np.where(head, 38.0 + 2.0 * np.sin(x + y), -1000.0).astype(np.float32)
    raw = [ncct]
    for i in range(1, 5):
        arr = np.where(head, 10.0 + 3.0*np.cos(i + x / 4.0), 0.0).astype(np.float32)
        if i == 4:
            arr[13, 13, 6] = np.nan
            arr[14, 13, 6] = -100001.0
        raw.append(arr)
    affine = np.diag([0.6, 0.6, 2.0, 1.0])
    affine[:3, 3] = [-70.0, 20.0, 5.0]
    images = folder / "imagesTr"
    images.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(raw):
        nifti = nib.Nifti1Image(a, affine)
        nifti.set_qform(affine, code=1)
        nifti.set_sform(affine, code=1)
        nib.save(nifti, str(images / f"{case}_{i:04d}.nii.gz"))
    if not label:
        return None
    target = folder / "labelsTr" / f"{case}.nii.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    lab = np.zeros((32, 32, 16), dtype=np.uint8)
    lab[14:18, 14:18, 7:9] = 1
    obj = nib.Nifti1Image(lab, affine)
    obj.set_qform(affine, code=1)
    obj.set_sform(affine, code=1)
    nib.save(obj, str(target))
    return target


def synthetic_parameters() -> dict:
    return {ch: {
        "channel": ch,
        "scheme": "CT_style_fold_only" if ch == "NCCT" else "perfusion_fold_robust",
        "center": 30.0 if ch == "NCCT" else 10.0,
        "scale": 8.0 if ch == "NCCT" else 3.0,
        "clip_low": -50.0 if ch == "NCCT" else 0.0,
        "clip_high": 140.0 if ch == "NCCT" else 30.0,
        "candidate_post_z_cap": None if ch == "NCCT" else 8.0,
        "status": "candidate_not_authorized_for_training",
    } for ch in CHANNELS}


def run_synthetic(output: Path, source_plans: Path) -> dict:
    from qualify_isles2024_nnunet_runtime import check_actual_nnunet_normalization

    if file_sha256(source_plans) != SOURCE_PLANS_SHA:
        raise ValueError("Frozen original nnU-Net plans SHA mismatch")
    if output.resolve() == source_plans.parent.resolve() or source_plans.parent.resolve() in output.resolve().parents:
        raise ValueError("Synthetic report must not overwrite source plans")
    import importlib.metadata
    if importlib.metadata.version("nnunetv2") != PINNED_NNUNET:
        raise RuntimeError("Pinned nnU-Net 2.8.1 is required for official synthetic preprocessing")
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    original = load_object(source_plans)
    with tempfile.TemporaryDirectory(prefix="wu20g_offline_synthetic_") as t:
        root = Path(t)
        source = root / "source"
        d506, d507 = root / "staged" / "506", root / "staged" / "507"
        params = synthetic_parameters()
        a = "sub_stroke9001"
        b = "sub_stroke9002"
        lab = make_synthetic_inputs(source, a, label=True)
        receipt_tr = stage_synthetic_case(source, d506, d507, a, "imagesTr", params, label_file=lab)
        make_synthetic_inputs(source, b, label=False)
        receipt_ts = stage_synthetic_case(source, d506, d507, b, "imagesTs", params)
        official = {}
        for case, group in ((a, "imagesTr"), (b, "imagesTs")):
            arrays = {}
            props = {}
            for ds, root_ds in ((506, d506), (507, d507)):
                locked = build_locked_plans(original, dataset_id=ds)
                assert_locked_plans(original, locked, dataset_id=ds)
                meta = no_norm_dataset_json(n_train=1, ncct_only=ds == 506)
                check_actual_nnunet_normalization(locked, meta, 1 if ds == 506 else 5)
                pm = PlansManager(locked)
                cfg = pm.get_configuration("3d_fullres")
                files = [str(root_ds / group / f"{case}_{i:04d}.nii.gz") for i in range(1 if ds==506 else 5)]
                data, _seg, properties = DefaultPreprocessor(verbose=False).run_case(
                    files, seg_file=None, plans_manager=pm,
                    configuration_manager=cfg, dataset_json=meta,
                )
                if not np.isfinite(data).all():
                    raise ValueError("Official synthetic preprocessing emitted nonfinite values")
                arrays[ds], props[ds] = data, properties
            if arrays[506].shape[1:] != arrays[507].shape[1:]:
                raise ValueError("Synthetic official preprocessing changed spatial shape")
            diff = float(np.max(np.abs(arrays[506][0] - arrays[507][0])))
            if diff > 1e-5 or props[506].get("bbox_used_for_cropping") != props[507].get("bbox_used_for_cropping"):
                raise ValueError("Synthetic official NCCT/geometry mismatch")
            perf_overshoot = int(np.count_nonzero(np.abs(arrays[507][1:]) > 8.0001))
            if perf_overshoot:
                raise ValueError("Synthetic perfusion interpolation overshoot needs review")
            official[group] = {"post_resample_ncct_max_abs_difference": diff,
                               "post_resample_perfusion_overshoot_voxels": perf_overshoot,
                               "shapes": {str(ds): list(arrays[ds].shape) for ds in (506, 507)},
                               "nnunet_noNorm_verified": True}
        result = {"schema": SCHEMA, "status": "SYNTHETIC_PHYSICAL_STAGING_QUALIFIED",
                  "case_count": 2, "training_label_case_count": 1,
                  "heldout_label_opened": False,
                  "synthetic_receipts": [receipt_tr["case"], receipt_ts["case"]],
                  "official_nnunet_preprocessing": official,
                  "nnunet_version": PINNED_NNUNET,
                  "staging_materialized_for_real_patients": False,
                  "training_ready": False}
    atomic_json(output / "WU20G_synthetic_staging_aggregate.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=ALLOWED_MODE, default="plan")
    parser.add_argument("--raw505", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--wu20e-gate", type=Path)
    parser.add_argument("--wu20e-policy", type=Path)
    parser.add_argument("--wu20f", type=Path)
    parser.add_argument("--header-checkpoints", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/content/wu20g_physical_staging_DISABLED"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "stage":
        # No JSON approval flag, environment override or undocumented CLI switch.
        raise PermissionError("REAL PHYSICAL STAGING DISABLED: WU20H independent signed release PR required")
    if args.mode == "synthetic":
        result = run_synthetic(args.output, args.plans)
    else:
        required = (args.raw505, args.splits, args.folds, args.wu20e_gate,
                    args.wu20e_policy, args.wu20f, args.header_checkpoints)
        if any(x is None for x in required):
            parser.error("Plan mode requires frozen source, fold and WU20E/WU20F evidence paths")
        result = plan_staging(args.raw505, args.splits, args.plans, args.folds,
                              args.wu20e_gate, args.wu20e_policy, args.wu20f,
                              args.header_checkpoints, args.output, args.scratch)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
