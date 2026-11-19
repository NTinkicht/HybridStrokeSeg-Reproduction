"""WU-20 fold-isolated, image-only candidate preprocessing (training NOT authorized).

No patient outcomes are used during calibration. Raw NCCT identifies a *head-tissue*
support region, NOT a validated brain mask. All calibration statistics are from
training-fold patients only. The exact same NCCT transform is used for the
506 and 507 arms. This module requires no nnU-Net training installation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from hybridstrokeseg.wu19_normalization_protocol import (
    apply_candidate_transform,
    fit_fold_channel,
)

CHANNELS = ("NCCT", "TMAX", "CBF", "CBV", "MTT")
FROZEN_SPLIT_SHA = "664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027"
SCHEMA = "WU20-pilot-0.1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_split(path: Path, fold: int) -> tuple[list[str], list[str]]:
    if file_sha256(path) != FROZEN_SPLIT_SHA:
        raise ValueError("Frozen 149-patient split hash mismatch")
    all_folds = json.loads(path.read_text(encoding="utf8"))
    if len(all_folds) != 5 or [len(f["val"]) for f in all_folds] != [30, 30, 30, 30, 29]:
        raise ValueError("Unexpected fold sizes")
    if fold not in range(5):
        raise ValueError("Fold must be 0..4")
    full = [case for group in all_folds for case in group["val"]]
    universe = set(full)
    if len(universe) != 149:
        raise ValueError("Duplicate or missing validation patient")
    for group in all_folds:
        if set(group["train"]) != universe - set(group["val"]):
            raise ValueError("Patient leakage or noncomplementary split")
    return sorted(all_folds[fold]["train"]), sorted(all_folds[fold]["val"])


def ncct_head_tissue_support(image: np.ndarray, *, min_coverage: float = 0.015,
                             max_coverage: float = 0.75) -> tuple[np.ndarray, dict]:
    """Make *provisional* NCCT-only tissue support, not a verified brain mask.

    Uses an image-only head envelope above -500 HU (largest 3D component), fills
    holes and includes plausible soft-tissue values [-100, 150]. Never reads a
    lesion mask. This deliberately errs on failing uncertain cropped cases.
    """
    a = np.asarray(image)
    if a.ndim != 3 or not np.issubdtype(a.dtype, np.number):
        raise ValueError("NCCT must be a 3D numeric array")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite NCCT requires manual source QC")
    if min(a.shape) < 8:
        raise ValueError("Insufficient 3D NCCT extent")
    envelope = a > -500.0
    labeled, count = ndi.label(envelope, structure=ndi.generate_binary_structure(3, 1))
    if count < 1:
        raise ValueError("No NCCT head foreground found")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    head = ndi.binary_fill_holes(labeled == int(np.argmax(sizes)))
    support = head & (a >= -100.0) & (a <= 150.0)
    n = int(support.sum())
    fraction = n / a.size
    if not min_coverage <= fraction <= max_coverage or n < 32:
        raise ValueError(f"NCCT support coverage failed QC: {fraction:.4%}")
    bounds = np.where(support)
    bbox = [[int(v.min()), int(v.max())] for v in bounds]
    border = (np.count_nonzero(support[0]) + np.count_nonzero(support[-1])
              + np.count_nonzero(support[:, 0]) + np.count_nonzero(support[:, -1])
              + np.count_nonzero(support[:, :, 0]) + np.count_nonzero(support[:, :, -1]))
    return support, {
        "n_tissue_voxels": n, "coverage_fraction": fraction, "bbox_min_max": bbox,
        "border_support_voxels_with_duplicates": int(border),
        "head_envelope_voxels": int(head.sum()), "algorithm": "provisional_head_tissue_v1",
        "clinical_brain_mask_validated": False,
    }


def deterministic_training_sample(image: np.ndarray, support: np.ndarray, *, case_id: str,
                                  channel: str, max_samples: int = 4096) -> np.ndarray:
    """Draw a reproducible, uniform-without-replacement per-case finite sample."""
    image = np.asarray(image)
    if image.shape != support.shape or not np.issubdtype(image.dtype, np.number):
        raise ValueError("Image/support shape or dtype mismatch")
    if channel not in CHANNELS or max_samples < 16:
        raise ValueError("Unsupported sample channel/count")
    eligible = np.flatnonzero(support.ravel() & np.isfinite(image.ravel()))
    if eligible.size < 16:
        raise ValueError(f"Not enough finite support voxels: {case_id}/{channel}")
    seed = int.from_bytes(hashlib.sha256(f"WU20|{case_id}|{channel}".encode()).digest()[:8], "big")
    random = np.random.default_rng(seed)
    chosen = random.choice(eligible, size=min(max_samples, eligible.size), replace=False)
    return np.asarray(image.ravel()[chosen], dtype=np.float32).copy()


def fit_fold_training_only(samples_by_channel: Mapping[str, Mapping[str, np.ndarray]],
                           *, training_ids: Sequence[str], excluded_ids: Sequence[str]) -> dict:
    """Fit 506/507 *candidate* policy on training patient IDs only.

    The existing WU19 normalizer rejects a validation ID in any input samples.
    """
    if set(samples_by_channel) != set(CHANNELS):
        raise ValueError("Missing/extra modality in fold calibration")
    output = {}
    for channel in CHANNELS:
        output[channel] = fit_fold_channel(
            samples_by_channel[channel], training_ids, excluded_ids, channel=channel
        )
    return output


def candidate_arms(images: Mapping[str, np.ndarray], parameters: dict,
                   *, support: np.ndarray) -> tuple[dict, dict, dict]:
    """Transform both arms once, reusing a *single identical* NCCT array."""
    if set(images) != set(CHANNELS) or set(parameters) != set(CHANNELS):
        raise ValueError("Expected five image channels and parameters")
    for channel in CHANNELS:
        if np.asarray(images[channel]).shape != support.shape:
            raise ValueError(f"Geometry/shape mismatch: {channel}")
    common_ncct = apply_candidate_transform(images["NCCT"], parameters["NCCT"], support=support)
    single = {"NCCT": common_ncct}
    multiple = {"NCCT": common_ncct}
    audit = {"nonfinite_input_voxels": {}, "abs_gt_8_output_voxels": {},
             "same_ncct_object_in_memory": True}
    for channel in CHANNELS[1:]:
        a = np.asarray(images[channel])
        audit["nonfinite_input_voxels"][channel] = int(np.count_nonzero(~np.isfinite(a)))
        transformed = apply_candidate_transform(a, parameters[channel], support=support)
        audit["abs_gt_8_output_voxels"][channel] = int(np.count_nonzero(np.abs(transformed) > 8.000001))
        if not np.isfinite(transformed).all():
            raise ValueError(f"Nonfinite after candidate transform: {channel}")
        multiple[channel] = transformed
    return single, multiple, audit


def no_norm_dataset_json(*, n_train: int, ncct_only: bool) -> dict:
    if n_train < 1:
        raise ValueError("Training patient count must be positive")
    count = 1 if ncct_only else 5
    return {
        "channel_names": {str(i): "noNorm" for i in range(count)},
        "labels": {"background": 0, "final_infarct": 1},
        "numTraining": int(n_train),
        "file_ending": ".nii.gz",
    }


def validate_arm_workspaces(d506: Path, d507: Path, *, train_ids: Sequence[str],
                            val_ids: Sequence[str]) -> dict:
    """Fail if any held-out label is present in imagesTr/labelsTr or NCCT differs."""
    train, val = set(train_ids), set(val_ids)
    if not train or not val or train & val:
        raise ValueError("Invalid train/validation partition")
    output = {}
    for dataset, count, root in ((506, 1, d506), (507, 5, d507)):
        images_tr = {p.name for p in (root / "imagesTr").glob("*.nii.gz")}
        images_ts = {p.name for p in (root / "imagesTs").glob("*.nii.gz")}
        labels = {p.name for p in (root / "labelsTr").glob("*.nii.gz")}
        exp_train = {f"{case}_{i:04d}.nii.gz" for case in train for i in range(count)}
        exp_val = {f"{case}_{i:04d}.nii.gz" for case in val for i in range(count)}
        if images_tr != exp_train or images_ts != exp_val or labels != {f"{c}.nii.gz" for c in train}:
            raise ValueError(f"Dataset{dataset}: staged patient inventory mismatch / possible leakage")
        meta = json.loads((root / "dataset.json").read_text())
        if meta != no_norm_dataset_json(n_train=len(train), ncct_only=(dataset == 506)):
            raise ValueError(f"Dataset{dataset}: second-normalization risk or unexpected metadata")
        output[str(dataset)] = {"n_train_images": len(images_tr), "n_val_imagesTs": len(images_ts),
                                "n_train_labels": len(labels)}
    for case in sorted(train | val):
        group = "imagesTr" if case in train else "imagesTs"
        a = d506 / group / f"{case}_0000.nii.gz"
        b = d507 / group / f"{case}_0000.nii.gz"
        if file_sha256(a) != file_sha256(b):
            raise ValueError(f"NCCT output differs across 506 and 507: {case}")
    return output
