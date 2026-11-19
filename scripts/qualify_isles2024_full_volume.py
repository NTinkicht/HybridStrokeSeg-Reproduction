#!/usr/bin/env python3
"""WU20D: read-only full native-volume nnU-Net CPU preprocessing audit.

Not inference, not clinical approval, not permission to train. All NIfTI
staging is temporary; only private, pseudonymized QC JSON is persisted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qualify_isles2024_nnunet_runtime import (
    PINNED_NNUNET,
    SOURCE_PLANS_SHA,
    build_locked_plans,
    check_actual_nnunet_normalization,
    validate_locked_fold_manifest,
)
from stage_isles2024_fold_pilot import _save_preprocessed

from hybridstrokeseg.wu20_fold_preprocessing import (
    CHANNELS,
    candidate_arms,
    file_sha256,
    ncct_head_tissue_support,
    no_norm_dataset_json,
    validated_split,
)

SCHEMA = "WU20D-full-native-volume-1.0"
CASE_FOLDS = {
    "sub_stroke0016": 2,
    "sub_stroke0113": 1,
    "sub_stroke0137": 4,
    "sub_stroke0152": 3,
}
CASES = tuple(CASE_FOLDS)


def ensure_output_isolated(output: Path, raw: Path, plans: Path, splits: Path) -> None:
    """Refuse direct writes to original raw/preprocessed trees."""
    target = output.resolve()
    protected = [raw.resolve(), raw.parent.resolve(), plans.parent.resolve(), splits.parent.resolve()]
    if any(target == p or p in target.parents for p in protected):
        raise ValueError("Output may not modify a frozen Dataset505 directory")


def raw_channel_diagnostics(data: np.ndarray, support: np.ndarray) -> dict:
    """Counts full native-volume anomalies inside/outside image-only support."""
    a = np.asarray(data)
    m = np.asarray(support)
    if a.ndim != 3 or m.shape != a.shape or m.dtype != np.bool_:
        raise ValueError("Raw channel/support 3D shape mismatch")
    finite = np.isfinite(a)
    bad = ~finite
    low = finite & (a < -10000)
    high = finite & (a > 10000)
    return {
        "n_voxels": int(a.size),
        "n_nonfinite_full": int(np.count_nonzero(bad)),
        "n_nonfinite_within_support": int(np.count_nonzero(bad & m)),
        "n_nonfinite_outside_support": int(np.count_nonzero(bad & ~m)),
        "n_below_neg10000_full": int(np.count_nonzero(low)),
        "n_below_neg10000_within_support": int(np.count_nonzero(low & m)),
        "n_above_pos10000_full": int(np.count_nonzero(high)),
        "n_above_pos10000_within_support": int(np.count_nonzero(high & m)),
    }


def verify_pair_after_preprocessing(ncct: np.ndarray, perfusion: np.ndarray,
                                    props506: dict, props507: dict,
                                    *, threshold: float = 8.0001) -> dict:
    """Prevent silent NCCT discrepancy; report perfusion interpolation overshoot."""
    a, b = np.asarray(ncct), np.asarray(perfusion)
    if a.ndim != 4 or b.ndim != 4 or a.shape[0] != 1 or b.shape[0] != 5:
        raise ValueError("Unexpected channel count in preprocessed tensors")
    if a.shape[1:] != b.shape[1:] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Spatial mismatch or nonfinite after official nnU-Net preprocessing")
    if props506.get("bbox_used_for_cropping") != props507.get("bbox_used_for_cropping"):
        raise ValueError("Paired arms have different preprocessing crop box")
    diff = float(np.max(np.abs(a[0].astype(np.float64) - b[0].astype(np.float64))))
    if diff > 1e-5:
        raise ValueError(f"NCCT mismatch after nnU-Net preprocessing: {diff}")
    extra = int(np.count_nonzero(np.abs(b[1:]) > threshold))
    return {
        "output_shape_506": [int(v) for v in a.shape],
        "output_shape_507": [int(v) for v in b.shape],
        "postresample_ncct_max_abs_difference": diff,
        "postresample_perfusion_max_abs": float(np.max(np.abs(b[1:]))),
        "postresample_perfusion_voxels_abs_gt_8": extra,
        "postresample_excess_requires_review": bool(extra),
        "preprocess_crop_box_equal": True,
    }


def checksum_inputs(files: list[Path], manifest: Path, plans: Path) -> str:
    d = hashlib.sha256()
    for path in [*files, manifest, plans]:
        if not path.is_file():
            raise FileNotFoundError(path)
        d.update(str(path.name).encode("utf8"))
        d.update(file_sha256(path).encode("ascii"))
    return d.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def existing_completed(path: Path, signature: str, case: str, fold: int) -> dict | None:
    if not path.is_file():
        return None
    old = json.loads(path.read_text(encoding="utf8"))
    if (old.get("schema") != SCHEMA or old.get("case") != case
            or old.get("fold") != fold or old.get("input_sha256") != signature):
        raise ValueError("Existing checkpoint is stale/incompatible; review before explicit removal")
    if old.get("status") not in {"FULL_VOLUME_TECHNICAL_PASS", "FULL_VOLUME_TECHNICAL_REVIEW_REQUIRED"}:
        raise ValueError("Unfinished checkpoint must not count as completed")
    return old


def run_full_case(raw: Path, splits: Path, plans_path: Path, manifest_path: Path,
                  output: Path, *, case: str) -> dict:
    if case not in CASE_FOLDS:
        raise ValueError("Unapproved audit case; expand predefined case roster only by code review")
    fold = CASE_FOLDS[case]
    ensure_output_isolated(output, raw, plans_path, splits)
    if file_sha256(plans_path) != SOURCE_PLANS_SHA:
        raise ValueError("Frozen source nnU-Net plans SHA256 mismatch")
    train, heldout = validated_split(splits, fold)
    if case not in heldout:
        raise ValueError("Audit case is not held out in its assigned fold")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    parameters = validate_locked_fold_manifest(manifest, train, heldout, fold)
    originals = [raw / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)]
    signature = checksum_inputs(originals, manifest_path, plans_path)
    target = output / "private_cases" / f"{case}_full_volume_PRIVATE.json"
    cached = existing_completed(target, signature, case, fold)
    if cached is not None:
        print(f"{case}: verified immutable checkpoint; no repeat preprocessing", flush=True)
        return cached

    if importlib.metadata.version("nnunetv2") != PINNED_NNUNET:
        raise RuntimeError(f"Expected pinned nnunetv2=={PINNED_NNUNET}")
    import nibabel as nib
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    original = json.loads(plans_path.read_text(encoding="utf8"))
    locked = {ds: build_locked_plans(original, dataset_id=ds) for ds in (506, 507)}
    metas = {ds: no_norm_dataset_json(n_train=len(train), ncct_only=ds == 506)
             for ds in (506, 507)}
    checks = {str(ds): check_actual_nnunet_normalization(locked[ds], metas[ds], 1 if ds == 506 else 5)
              for ds in (506, 507)}
    imgs = [nib.load(str(p)) for p in originals]
    if len({img.shape for img in imgs}) != 1 or not all(
        np.allclose(imgs[0].affine, img.affine, atol=1e-4) for img in imgs[1:]
    ):
        raise ValueError("Native source shape/affine mismatch")
    arrays = {ch: img.get_fdata(dtype=np.float32) for ch, img in zip(CHANNELS, imgs)}
    support, support_qc = ncct_head_tissue_support(arrays["NCCT"])
    diagnostics = {ch: raw_channel_diagnostics(arrays[ch], support) for ch in CHANNELS}
    expected_mtt_nonfinite = {"sub_stroke0016": 9, "sub_stroke0113": 225045}
    if case in expected_mtt_nonfinite and diagnostics["MTT"]["n_nonfinite_full"] != expected_mtt_nonfinite[case]:
        raise ValueError("MTT raw nonfinite count differs from validated Notebook17 audit")
    arm506, arm507, transformed = candidate_arms(arrays, parameters, support=support)
    if not all(np.isfinite(v).all() for v in arm507.values()):
        raise ValueError("Nonfinite in candidate transformed volume")
    if transformed["same_ncct_object_in_memory"] is not True:
        raise ValueError("NCCT preprocessing differs between arms")

    # Free native arrays before the official potentially memory-intensive preprocessor.
    del arrays, support
    gc.collect()
    with tempfile.TemporaryDirectory(prefix="wu20d_full_volume_") as scratch:
        home = Path(scratch)
        for dataset, channel_data in ((506, arm506), (507, arm507)):
            d = home / f"Dataset{dataset}"
            (d / "imagesTs").mkdir(parents=True)
            paths = []
            for i, ch in enumerate(channel_data):
                dest = d / "imagesTs" / f"{case}_{i:04d}.nii.gz"
                if dataset == 507 and i == 0:
                    # Identical NCCT file bytes, not merely equal floating point values.
                    shutil.copy2(home / "Dataset506" / "imagesTs" / dest.name, dest)
                else:
                    _save_preprocessed(dest, channel_data[ch], imgs[CHANNELS.index(ch)])
                paths.append(dest)
            if dataset == 506:
                ncct_file_hash = file_sha256(paths[0])
            elif file_sha256(paths[0]) != ncct_file_hash:
                raise ValueError("Dataset506/507 NCCT staging not byte-identical")
            del channel_data
            gc.collect()
            pm = PlansManager(locked[dataset])
            conf = pm.get_configuration("3d_fullres")
            data, _seg, props = DefaultPreprocessor(verbose=False).run_case(
                [str(p) for p in paths], seg_file=None, plans_manager=pm,
                configuration_manager=conf, dataset_json=metas[dataset],
            )
            if dataset == 506:
                ncct_pre = data
                props506 = props
            else:
                comparison = verify_pair_after_preprocessing(ncct_pre, data, props506, props)
                del data, ncct_pre
                gc.collect()
        result = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "case": case, "fold": fold, "input_sha256": signature,
            "status": "FULL_VOLUME_TECHNICAL_REVIEW_REQUIRED"
            if comparison["postresample_excess_requires_review"] else "FULL_VOLUME_TECHNICAL_PASS",
            "training_ready": False, "real_model_inference_performed": False,
            "full_native_volume_tested": True, "heldout_labels_read": False,
            "nnunet_version": PINNED_NNUNET,
            "train_case_count": len(train), "heldout_count": len(heldout),
            "source_image_shape": [int(v) for v in imgs[0].shape],
            "raw_channel_qc": diagnostics,
            "ncct_support_qc": support_qc,
            "candidate_transform_qc": transformed,
            "input_noNorm_checks": checks,
            "official_preprocessing": comparison,
            "blocked_release_gates": [
                "Independent clinical/anatomical review of 15 masks and robust clipping policy",
                "Real model inference not performed: no trained 506/507 checkpoints",
                "Prospective, signed and frozen candidate protocol required",
            ],
        }
    atomic_json(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw505", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--source-plans", required=True, type=Path)
    parser.add_argument("--fold-manifest-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()
    fold = CASE_FOLDS[args.case]
    manifest = args.fold_manifest_dir / f"WU20B_fold{fold}_parameters_PRIVATE.json"
    report = run_full_case(args.raw505, args.splits, args.source_plans, manifest,
                           args.output, case=args.case)
    print(json.dumps({
        "case": args.case,
        "status": report["status"],
        "full_native_volume_tested": report["full_native_volume_tested"],
        "raw_mtt_nonfinite": report["raw_channel_qc"]["MTT"]["n_nonfinite_full"],
        "raw_mtt_nonfinite_within_support": report["raw_channel_qc"]["MTT"]["n_nonfinite_within_support"],
        "postresample": report["official_preprocessing"],
        "training_ready": False,
    }, indent=2))


if __name__ == "__main__":
    main()
