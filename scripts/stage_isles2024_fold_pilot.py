#!/usr/bin/env python3
"""Stage a bounded CPU-only WU20 PILOT: independent train-only fit + paired 506/507.

NOT an approved/released dataset, and NOT a 149-case train-ready pipeline.
All NIfTI data are staged in a local scratch workspace only. Saves private QC
and manifests to a separate report folder; no outcomes/validation labels used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hybridstrokeseg.wu20_fold_preprocessing import (
    CHANNELS,
    SCHEMA,
    candidate_arms,
    deterministic_training_sample,
    file_sha256,
    fit_fold_training_only,
    ncct_head_tissue_support,
    no_norm_dataset_json,
    validate_arm_workspaces,
    validated_split,
)

NAME506 = "Dataset506_ISLES2024_NCCT_CTCTRL"
NAME507 = "Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST"


def _load_nifti(path: Path):
    """Import nibabel lazily so pilot helper tests remain NumPy-only."""
    import nibabel as nib

    if not path.is_file():
        raise FileNotFoundError(path)
    nifti = nib.load(str(path))
    data = nifti.get_fdata(dtype=np.float32)
    if data.ndim != 3 or any(x < 8 for x in data.shape):
        raise ValueError(f"Invalid NIfTI shape: {path}: {data.shape}")
    return nifti, data


def _image_pair(root: Path, case: str, channel: str):
    if channel not in CHANNELS:
        raise ValueError(channel)
    return _load_nifti(root / "imagesTr" / f"{case}_{CHANNELS.index(channel):04d}.nii.gz")


def _save_preprocessed(path: Path, image: np.ndarray, template) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    header = template.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    out = nib.Nifti1Image(np.ascontiguousarray(image, dtype=np.float32),
                          template.affine.copy(), header=header)
    code_q = int(template.header["qform_code"])
    code_s = int(template.header["sform_code"])
    if code_q:
        out.set_qform(template.get_qform(), code=code_q)
    if code_s:
        out.set_sform(template.get_sform(), code=code_s)
    nib.save(out, str(path))
    saved = nib.load(str(path))
    if saved.shape != template.shape or not np.allclose(saved.affine, template.affine, atol=1e-5):
        raise ValueError(f"NIfTI header / geometry changed: {path}")
    if not np.all(np.isfinite(saved.get_fdata(dtype=np.float32))):
        raise ValueError(f"Non-finite generated NIfTI: {path}")


def choose_pilot_ids(training: list[str], validation: list[str], *, n_train: int,
                     validation_case: str | None) -> tuple[list[str], list[str]]:
    if n_train < 3 or n_train > len(training):
        raise ValueError("Pilot needs 3+ and no more than 119/120 training cases")
    # Spread examples deterministically throughout sorted training IDs.
    pick = np.linspace(0, len(training) - 1, n_train, dtype=int)
    training_pilot = [training[int(i)] for i in pick]
    if len(set(training_pilot)) != n_train:
        raise ValueError("Pilot sampling produced duplicate training case")
    if validation_case is not None and validation_case not in validation:
        raise ValueError("Requested pilot validation case is not held-out in this fold")
    chosen_val = [validation_case] if validation_case is not None else [validation[0]]
    if set(chosen_val) & set(training_pilot):
        raise ValueError("Pilot training/validation overlap")
    return training_pilot, chosen_val


def verify_source_geometry(images: dict, *, case: str) -> None:
    template = images["NCCT"][0]
    for channel in CHANNELS:
        nifti, pixels = images[channel]
        if pixels.shape != template.shape or not np.allclose(nifti.affine, template.affine, atol=1e-4):
            raise ValueError(f"{case}: native NIfTI geometry mismatch for {channel}")


def atomic_private_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf8") as fp:
        json.dump(content, fp, sort_keys=True, indent=2, allow_nan=False)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def pilot(raw505: Path, split: Path, *, fold: int, pilot_train_n: int,
          pilot_validation_case: str | None, scratch: Path, private_results: Path) -> dict:
    train, heldout = validated_split(split, fold)
    train_pilot, val_pilot = choose_pilot_ids(
        train, heldout, n_train=pilot_train_n, validation_case=pilot_validation_case
    )
    if scratch.is_symlink() or scratch.resolve() == private_results.resolve():
        raise ValueError("Scratch workspace must be separate from report folder")
    if any(scratch.iterdir()) if scratch.exists() else False:
        raise FileExistsError(f"Refusing nonempty pilot scratch (no overwrite): {scratch}")
    if not (raw505 / "dataset.json").is_file():
        raise FileNotFoundError(raw505 / "dataset.json")
    meta = json.loads((raw505 / "dataset.json").read_text(encoding="utf8"))
    names = tuple(meta["channel_names"][str(i)].upper() for i in range(5))
    if names != CHANNELS or meta["numTraining"] != 149:
        raise ValueError(f"Dataset505 source channel contract differs: {names}")

    print(f"PILOT ONLY: fold={fold}; training sample cases={len(train_pilot)}; heldout cases={val_pilot}", flush=True)
    samples = {channel: {} for channel in CHANNELS}
    support_info = {}
    for i, case in enumerate(train_pilot, 1):
        ncct_nifti, ncct = _image_pair(raw505, case, "NCCT")
        support, info = ncct_head_tissue_support(ncct)
        support_info[case] = info
        for channel in CHANNELS:
            if channel == "NCCT":
                data = ncct
            else:
                nifti, data = _image_pair(raw505, case, channel)
                if data.shape != ncct.shape or not np.allclose(nifti.affine, ncct_nifti.affine, atol=1e-4):
                    raise ValueError(f"{case}: {channel} geometry mismatch before calibration")
            samples[channel][case] = deterministic_training_sample(
                data, support, case_id=case, channel=channel
            )
        print(f"[fit {i}/{len(train_pilot)}] {case}; NCCT support {info['coverage_fraction']:.2%}", flush=True)
    params = fit_fold_training_only(samples, training_ids=train_pilot, excluded_ids=heldout)
    # Record subset limitations: fitting a pilot subset is NOT valid for full-fold training.
    for channel in CHANNELS:
        if set(params[channel]["training_ids"]) != set(train_pilot):
            raise ValueError("Calibrator did not use exactly pilot training cases")
        if set(params[channel]["validation_ids_excluded"]) != set(heldout):
            raise ValueError("Held-out fold exclusions not recorded")
    armroot = scratch / f"fold_{fold}" / "nnUNet_raw"
    d506 = armroot / NAME506
    d507 = armroot / NAME507
    d506.mkdir(parents=True)
    d507.mkdir(parents=True)
    for path, is_ncct_only in ((d506, True), (d507, False)):
        (path / "dataset.json").write_text(
            json.dumps(no_norm_dataset_json(n_train=len(train_pilot), ncct_only=is_ncct_only), indent=2) + "\n",
            encoding="utf8",
        )
        for group in ("imagesTr", "imagesTs", "labelsTr"):
            (path / group).mkdir()
    qc = {}
    for j, case in enumerate(train_pilot + val_pilot, 1):
        images = {channel: _image_pair(raw505, case, channel) for channel in CHANNELS}
        verify_source_geometry(images, case=case)
        support, info = ncct_head_tissue_support(images["NCCT"][1])
        arrays = {k: v[1] for k, v in images.items()}
        arm506, arm507, diagnostics = candidate_arms(arrays, params, support=support)
        group = "imagesTr" if case in train_pilot else "imagesTs"
        dest506 = d506 / group / f"{case}_0000.nii.gz"
        _save_preprocessed(dest506, arm506["NCCT"], images["NCCT"][0])
        shutil.copy2(dest506, d507 / group / dest506.name)
        if not np.array_equal(arm506["NCCT"], arm507["NCCT"]):
            raise ValueError("In-memory NCCT differs across arms")
        for idx, channel in enumerate(CHANNELS[1:], 1):
            _save_preprocessed(d507 / group / f"{case}_{idx:04d}.nii.gz",
                               arm507[channel], images[channel][0])
        if case in train_pilot:
            label_path = raw505 / "labelsTr" / f"{case}.nii.gz"
            label_nifti, _ = _load_nifti(label_path)
            if label_nifti.shape != images["NCCT"][0].shape or not np.allclose(
                label_nifti.affine, images["NCCT"][0].affine, atol=1e-4
            ):
                raise ValueError(f"{case}: label shape/affine mismatch")
            for dataset in (d506, d507):
                shutil.copy2(label_path, dataset / "labelsTr" / label_path.name)
        qc[case] = {
            "source_split": "training" if case in train_pilot else "heldout",
            "head_support": info, "perfusion_sanitization": diagnostics,
            "post_ncct_min": float(arm506["NCCT"].min()),
            "post_ncct_max": float(arm506["NCCT"].max()),
            "post_mtt_min": float(arm507["MTT"].min()),
            "post_mtt_max": float(arm507["MTT"].max()),
        }
        print(f"[stage {j}/{len(train_pilot)+len(val_pilot)}] {case} {group}:",
              f"MTT nonfinite input={diagnostics['nonfinite_input_voxels']['MTT']}", flush=True)

    inventory = validate_arm_workspaces(d506, d507, train_ids=train_pilot, val_ids=val_pilot)
    # No dataset-specific tuning on OOF results. Full fold 119/120 training
    # parameters and clinician-approved supports remain mandatory release gates.
    result = {
        "schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CPU_PILOT_COMPLETE_NOT_TRAINING_APPROVED",
        "training_ready": False, "dataset504_505_unchanged": True,
        "pilot_fold": fold, "full_fold_train_n": len(train), "full_fold_val_n": len(heldout),
        "pilot_train_ids": train_pilot, "pilot_val_ids": val_pilot,
        "frozen_split_sha256": file_sha256(split),
        "normalizer_parameters_PILOT_ONLY": params,
        "pilot_fit_support_QC": support_info,
        "staged_case_QC": qc, "staging_inventory": inventory,
        "noNorm_in_both_arms": True,
        "independent_train_only_calibration": True,
        "same_ncct_bytes_in_both_arms": True,
        "artifacts_local_scratch_only": str(armroot),
        "blocked_release_gates": [
            "Provisional NCCT head-tissue mask is NOT a validated brain mask; needs independent case QC",
            "Normalization pilot fit uses subset of 119/120 training cases; full-fold fits pending",
            "Candidate q0.5-99.5 percentile bounds and |robust z|<=8 require blinded clinical review",
            "No nnU-Net plan/architecture lock or full inference smoke test yet",
            "No full-five-fold Dataset506/507 staging or training authorized",
        ],
    }
    atomic_private_json(private_results / f"WU20_pilot_fold{fold}_PRIVATE.json", result)
    public = {k: v for k, v in result.items() if k not in (
        "pilot_train_ids", "pilot_val_ids", "normalizer_parameters_PILOT_ONLY",
        "pilot_fit_support_QC", "staged_case_QC", "artifacts_local_scratch_only"
    )}
    atomic_private_json(private_results / f"WU20_pilot_fold{fold}_aggregate.json", public)
    return public


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw505", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--pilot-train-count", type=int, default=6)
    p.add_argument("--pilot-validation-case", type=str, default="sub_stroke0016")
    p.add_argument("--scratch", type=Path, default=Path("/content/wu20_fold_pilot"))
    p.add_argument("--private-results", type=Path, required=True)
    args = p.parse_args()
    if "drive" in {part.lower() for part in args.scratch.parts}:
        raise ValueError("Pilot NIfTI scratch must stay outside mounted Drive")
    result = pilot(args.raw505, args.split_file, fold=args.fold,
                   pilot_train_n=args.pilot_train_count,
                   pilot_validation_case=args.pilot_validation_case,
                   scratch=args.scratch, private_results=args.private_results)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
