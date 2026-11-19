#!/usr/bin/env python3
"""WU20C: CPU-only nnU-Net 2.8.1 held-out ROI compatibility proof (NOT training approval).

Creates no trained models, changes no frozen data, reads no validation labels.
The ROI is an image-only bounded technical proxy, not full-volume validation.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hybridstrokeseg.wu20_fold_preprocessing import (
    CHANNELS,
    FROZEN_SPLIT_SHA,
    candidate_arms,
    file_sha256,
    ncct_head_tissue_support,
    no_norm_dataset_json,
    validated_split,
)

PINNED_NNUNET = "2.8.1"
SOURCE_PLANS_SHA = "6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca"
DATASET_NAMES = {
    506: "Dataset506_ISLES2024_NCCT_CTCTRL",
    507: "Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST",
}
SCHEMA = "WU20C-runtime-qualification-1.0"
LOCKED_KEYS = (
    "batch_size", "patch_size", "spacing", "architecture", "resampling_fn_data",
    "resampling_fn_data_kwargs", "resampling_fn_seg", "resampling_fn_seg_kwargs",
    "resampling_fn_probabilities", "resampling_fn_probabilities_kwargs",
    "preprocessor_name", "batch_dice", "median_image_size_in_voxels",
)


def build_locked_plans(original: dict, *, dataset_id: int) -> dict:
    """Copy unchanged 505 geometry/architecture, replacing only modality-specific noNorm."""
    if dataset_id not in DATASET_NAMES:
        raise ValueError("Allowed dataset IDs are 506/507")
    if "3d_fullres" not in original.get("configurations", {}):
        raise ValueError("Missing frozen 3d_fullres configuration")
    ref = original["configurations"]["3d_fullres"]
    if len(ref.get("normalization_schemes", [])) != 5:
        raise ValueError("Expected five reference modalities")
    result = copy.deepcopy(original)
    result["dataset_name"] = DATASET_NAMES[dataset_id]
    result["configurations"] = {"3d_fullres": copy.deepcopy(ref)}
    n = 1 if dataset_id == 506 else 5
    cfg = result["configurations"]["3d_fullres"]
    cfg["normalization_schemes"] = ["NoNormalization"] * n
    cfg["use_mask_for_norm"] = [False] * n
    # NoNormalization ignores intensity properties. Do not carry global 505
    # foreground parameters from held-out cases into proposed fold plans.
    result["foreground_intensity_properties_per_channel"] = {str(i): {} for i in range(n)}
    assert_locked_plans(original, result, dataset_id=dataset_id)
    return result


def assert_locked_plans(original: dict, target: dict, *, dataset_id: int) -> None:
    if dataset_id not in DATASET_NAMES or target.get("dataset_name") != DATASET_NAMES[dataset_id]:
        raise ValueError("Wrong dataset identity")
    if list(target.get("configurations", {})) != ["3d_fullres"]:
        raise ValueError("Unexpected/planner-generated configuration")
    a = original["configurations"]["3d_fullres"]
    b = target["configurations"]["3d_fullres"]
    if any(a.get(key) != b.get(key) for key in LOCKED_KEYS):
        raise ValueError("Frozen 3d_fullres architecture/spacing/resampling mismatch")
    n = 1 if dataset_id == 506 else 5
    if b.get("normalization_schemes") != ["NoNormalization"] * n:
        raise ValueError("Preprocessor will normalize intensities again")
    if b.get("use_mask_for_norm") != [False] * n:
        raise ValueError("Unexpected normalization mask")
    if target.get("foreground_intensity_properties_per_channel") != {
        str(i): {} for i in range(n)
    }:
        raise ValueError("Leaky global intensity properties found in folded plans")
    for key in ("transpose_forward", "transpose_backward", "image_reader_writer"):
        if original.get(key) != target.get(key):
            raise ValueError(f"Frozen image orientation or I/O changed: {key}")
    if target.get("image_reader_writer") != "SimpleITKIO":
        raise ValueError("Expected source SimpleITKIO ordering")


def validate_locked_fold_manifest(manifest: dict, train: list[str], heldout: list[str], fold: int) -> dict:
    if manifest.get("frozen_split_sha256") != FROZEN_SPLIT_SHA:
        raise ValueError("Fold manifest split SHA mismatch")
    if manifest.get("fold") != fold:
        raise ValueError("Fold manifest identity mismatch")
    if manifest.get("n_training_patients") != len(train) or manifest.get("n_heldout_patients") != len(heldout):
        raise ValueError("Fold manifest count mismatch")
    if set(manifest.get("heldout_ids", [])) != set(heldout):
        raise ValueError("Held-out patient mismatch")
    params = manifest.get("provisional_fit", {})
    if set(params) != set(CHANNELS) or set(train) & set(heldout):
        raise ValueError("Fold modalities or split overlap")
    for channel, obj in params.items():
        if obj.get("channel") != channel or obj.get("n_training_cases") != len(train):
            raise ValueError(f"Incorrect fit channel/case count: {channel}")
        if set(obj.get("training_ids", [])) != set(train):
            raise ValueError(f"Training calibration leakage/missing cases: {channel}")
        if set(obj.get("validation_ids_excluded", [])) != set(heldout):
            raise ValueError(f"Held-out calibration mismatch: {channel}")
        if obj.get("status") != "candidate_not_authorized_for_training":
            raise ValueError("Unreviewed normalization status changed")
        for k in ("clip_low", "clip_high", "center", "scale"):
            if not np.isfinite(obj[k]):
                raise ValueError(f"Nonfinite parameter: {channel}.{k}")
        if obj["scale"] <= 0 or obj["clip_low"] >= obj["clip_high"]:
            raise ValueError(f"Invalid normalization: {channel}")
    return params


def image_only_roi(support: np.ndarray, max_shape=(96, 96, 24)) -> tuple[slice, slice, slice]:
    if support.ndim != 3 or support.dtype != np.bool_ or not support.any():
        raise ValueError("Invalid image-only support mask")
    where = np.where(support)
    result = []
    for ax, max_dim in enumerate(max_shape):
        if max_dim < 8:
            raise ValueError("ROI dimensions are too small")
        length = min(int(support.shape[ax]), max_dim)
        center = (int(where[ax].min()) + int(where[ax].max())) // 2
        start = max(0, min(center - length // 2, int(support.shape[ax]) - length))
        result.append(slice(start, start + length))
    crop = tuple(result)
    if int(support[crop].sum()) < 128:
        raise ValueError("ROI lacks sufficient image-only support")
    return crop


def shifted_affine(affine: np.ndarray, region: tuple[slice, slice, slice]) -> np.ndarray:
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.all(np.isfinite(a)):
        raise ValueError("Invalid NIfTI affine")
    delta = np.asarray([s.start for s in region], dtype=np.float64)
    shifted = a.copy()
    shifted[:3, 3] = a[:3, 3] + a[:3, :3] @ delta
    return shifted


def safe_write_roi_nifti(path: Path, data: np.ndarray, full_nifti, region) -> None:
    import nibabel as nib

    crop = np.asarray(data, dtype=np.float32)
    if crop.shape != tuple(s.stop - s.start for s in region):
        raise ValueError("Cropped image dimensions do not match ROI")
    if not np.isfinite(crop).all():
        raise ValueError("Refusing nonfinite cropped NIfTI output")
    affine = shifted_affine(full_nifti.affine, region)
    header = full_nifti.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(crop, affine, header=header)
    qcode = int(full_nifti.header["qform_code"])
    scode = int(full_nifti.header["sform_code"])
    if qcode:
        result.set_qform(affine, code=qcode)
    if scode:
        result.set_sform(affine, code=scode)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(path))
    roundtrip = nib.load(str(path))
    if roundtrip.shape != crop.shape or not np.allclose(roundtrip.affine, affine, atol=1e-4):
        raise ValueError("Cropped image affine or geometry changed")
    if not np.isfinite(roundtrip.get_fdata(dtype=np.float32)).all():
        raise ValueError("NIfTI storage introduced nonfinite values")


def check_actual_nnunet_normalization(plans: dict, dataset_json: dict, n: int) -> dict:
    """Require the *installed pinned* nnU-Net class and real _normalize call."""
    version = importlib.metadata.version("nnunetv2")
    if version != PINNED_NNUNET:
        raise RuntimeError(f"nnunetv2 version {version}, expected {PINNED_NNUNET}")
    from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import (
        get_normalization_scheme,
    )
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    scheme = get_normalization_scheme("noNorm")
    if scheme.__name__ != "NoNormalization":
        raise ValueError("noNorm channel mapping mismatch")
    pm = PlansManager(plans)
    cfg = pm.get_configuration("3d_fullres")
    if cfg.normalization_schemes != ["NoNormalization"] * n:
        raise ValueError("Locked plan normalization does not resolve to NoNormalization")
    if list(dataset_json["channel_names"].values()) != ["noNorm"] * n:
        raise ValueError("Dataset channel names inconsistent")
    probe = np.arange(n * 8 * 8 * 8, dtype=np.float32).reshape(n, 8, 8, 8) / 11
    seg = np.zeros((1, 8, 8, 8), dtype=np.int8)
    baseline = probe.copy()
    result = DefaultPreprocessor(verbose=False)._normalize(
        probe, seg, cfg, pm.foreground_intensity_properties_per_channel
    )
    if not np.array_equal(result, baseline):
        raise ValueError("nnU-Net silently applied second normalization")
    return {"nnunetv2_version": version, "mapping": scheme.__name__,
            "actual_normalizer_no_change": True}


def runtime_probe(raw505: Path, splits: Path, source_plans: Path, fold_manifest: Path,
                  output: Path, *, fold: int = 2, case: str = "sub_stroke0016") -> dict:
    import nibabel as nib
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    if file_sha256(source_plans) != SOURCE_PLANS_SHA:
        raise ValueError("Original Dataset505 plan hash mismatch")
    train, val = validated_split(splits, fold)
    if case not in val:
        raise ValueError("Case is not held out in the selected fold")
    if output.resolve() == raw505.resolve() or raw505.resolve() in output.resolve().parents:
        raise ValueError("Cannot write into frozen source data")
    if output.resolve() == source_plans.parent.resolve():
        raise ValueError("Cannot write over original plans")
    manifest = json.loads(fold_manifest.read_text(encoding="utf8"))
    params = validate_locked_fold_manifest(manifest, train, val, fold)
    original = json.loads(source_plans.read_text(encoding="utf8"))
    locked = {ds: build_locked_plans(original, dataset_id=ds) for ds in (506, 507)}
    meta = {ds: no_norm_dataset_json(n_train=len(train), ncct_only=(ds == 506))
            for ds in (506, 507)}
    no_norm_checks = {
        str(ds): check_actual_nnunet_normalization(locked[ds], meta[ds], 1 if ds == 506 else 5)
        for ds in (506, 507)
    }
    src = [raw505 / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)]
    images = [nib.load(str(p)) for p in src]
    if len({im.shape for im in images}) != 1 or not all(
        np.allclose(images[0].affine, im.affine, atol=1e-4) for im in images[1:]
    ):
        raise ValueError("Source NIfTI shape or affine mismatch")
    ncct = images[0].get_fdata(dtype=np.float32)
    mask, mask_qc = ncct_head_tissue_support(ncct)
    region = image_only_roi(mask)
    support_roi = mask[region]
    cropped = {"NCCT": ncct[region]}
    for i, channel in enumerate(CHANNELS[1:], start=1):
        cropped[channel] = np.asarray(images[i].dataobj[region], dtype=np.float32)
    arm506, arm507, audit = candidate_arms(cropped, params, support=support_roi)
    preproc = {}
    with tempfile.TemporaryDirectory(prefix="isles_wu20c_cpu_") as scratch:
        home = Path(scratch)
        for ds, channels in ((506, arm506), (507, arm507)):
            name = DATASET_NAMES[ds]
            d = home / name
            (d / "imagesTs").mkdir(parents=True)
            (d / "dataset.json").write_text(json.dumps(meta[ds]), encoding="utf8")
            (d / "nnUNetPlans.json").write_text(json.dumps(locked[ds]), encoding="utf8")
            for i, (ch, data) in enumerate(channels.items()):
                safe_write_roi_nifti(d / "imagesTs" / f"{case}_{i:04d}.nii.gz", data,
                                     images[CHANNELS.index(ch)], region)
            # A held-out image is never paired with its reference lesion label.
            pm = PlansManager(locked[ds])
            cfg = pm.get_configuration("3d_fullres")
            data, _seg, properties = DefaultPreprocessor(verbose=False).run_case(
                [str(d / "imagesTs" / f"{case}_{i:04d}.nii.gz")
                 for i in range(len(channels))],
                seg_file=None, plans_manager=pm, configuration_manager=cfg, dataset_json=meta[ds],
            )
            if data.shape[0] != len(channels) or not np.isfinite(data).all():
                raise ValueError(f"Dataset{ds}: preprocessing failed finite/channel checks")
            preproc[ds] = {"data": data, "properties": properties}
        a = preproc[506]["data"]
        b = preproc[507]["data"]
        if a.shape[1:] != b.shape[1:]:
            raise ValueError("Paired arm preprocessing changes output voxel geometry")
        ncct_absdiff = float(np.max(np.abs(a[0].astype(np.float64) - b[0].astype(np.float64))))
        if ncct_absdiff > 1e-5:
            raise ValueError(f"Post-resampling NCCT mismatch between 506/507: {ncct_absdiff}")
        props0 = preproc[506]["properties"]
        props1 = preproc[507]["properties"]
        if props0.get("bbox_used_for_cropping") != props1.get("bbox_used_for_cropping"):
            raise ValueError("Different nnU-Net crop boxes in the two arms")
        max_post_perfusion = float(np.max(np.abs(b[1:])))
        # The cap is only applied BEFORE cubic interpolation. Cubic overshoot
        # is measured and flagged rather than silently ignored or re-clipped.
        post_excess = int(np.count_nonzero(np.abs(b[1:]) > 8.0001))
        result = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "CPU_PREPROCESSING_PROBE_COMPLETE_NOT_TRAINING_READY",
            "training_ready": False,
            "real_model_inference_performed": False,
            "frozen_split_sha256": FROZEN_SPLIT_SHA,
            "source_plans_sha256": SOURCE_PLANS_SHA,
            "fold_manifest_sha256": file_sha256(fold_manifest),
            "fold": fold, "heldout_case_PRIVATE": case,
            "heldout_labels_read": False,
            "train_case_count": len(train), "heldout_case_count": len(val),
            "masked_roi_shape": [int(s.stop - s.start) for s in region],
            "mask_coverage_FULL_IMAGE": mask_qc["coverage_fraction"],
            "input_noNorm_checks": no_norm_checks,
            "locked_architecture_spacing": True,
            "output_shapes": {str(ds): list(preproc[ds]["data"].shape) for ds in (506, 507)},
            "postresample_ncct_max_abs_difference": ncct_absdiff,
            "postresample_perfusion_max_abs": max_post_perfusion,
            "postresample_perfusion_voxels_abs_gt_8": post_excess,
            "pretransform_nonfinite_perf_voxels_in_roi": audit["nonfinite_input_voxels"],
            "preprocess_crop_box_equal": True,
            "postresample_excess_requires_review": bool(post_excess),
            "source_roi_only_not_full_volume": True,
            "blocked_release_gates": [
                "Independent clinical review of masks and clipping policy is pending",
                "Full native-volume plan/preprocessing/inference parity not tested (technical ROI only)",
                "No trained 506/507 model checkpoints; no real model inference performed",
                "Signed and locked prospective protocol required before GPU training",
            ],
        }
    # No private NIfTIs leave local scratch. Save only private JSON results.
    output.mkdir(parents=True, exist_ok=True)
    tmp = output / "WU20C_probe_PRIVATE.tmp"
    final = output / "WU20C_probe_PRIVATE.json"
    with tmp.open("w", encoding="utf8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    redacted = {k: v for k, v in result.items() if k != "heldout_case_PRIVATE"}
    (output / "WU20C_probe_aggregate.json").write_text(
        json.dumps(redacted, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    return redacted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw505", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--case", default="sub_stroke0016")
    args = parser.parse_args()
    result = runtime_probe(args.raw505, args.splits, args.source_plans,
                           args.fold_manifest, args.output, fold=args.fold, case=args.case)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
