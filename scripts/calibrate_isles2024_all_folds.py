#!/usr/bin/env python3
"""WU-20B read-only-source, resumable 149-case fold calibration and private mask QC.

Only pre-intervention images are read. No lesion labels, predictions, or outcomes
are read; NO trained images are staged and NO nnU-Net training is authorized.
All patient-derived checkpoints, samples, and PNGs MUST stay in private Drive.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hybridstrokeseg.wu20_fold_preprocessing import (
    CHANNELS,
    deterministic_training_sample,
    file_sha256,
    fit_fold_training_only,
    ncct_head_tissue_support,
    validated_split,
)

SCHEMA = "WU20B-fold-calibration-v1"
SOURCE_NAME = "Dataset505_ISLES2024_NCCT_PERFUSION"
PRIORITY = ("sub_stroke0016", "sub_stroke0113", "sub_stroke0137", "sub_stroke0152")
RAW_PATIENT_CHANNEL_COUNT = 5
SAMPLE_COUNT = 4096


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def file_stat_signature(path: Path) -> dict:
    st = path.stat()
    if st.st_size <= 0:
        raise ValueError(f"Empty input: {path}")
    return {"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def ordered_source_files(raw: Path, case: str) -> list[Path]:
    return [raw / "imagesTr" / f"{case}_{i:04d}.nii.gz" for i in range(5)]


def source_signatures(raw: Path, case: str) -> list[dict]:
    return [file_stat_signature(p) for p in ordered_source_files(raw, case)]


def case_paths(out: Path, case: str) -> tuple[Path, Path]:
    return out / "patient_checkpoints" / f"{case}.json", out / "patient_checkpoints" / f"{case}.npz"


def verify_checkpoint(meta: dict, sigs: list[dict], npz_path: Path, case: str) -> bool:
    if (meta.get("schema") != SCHEMA or meta.get("case_id") != case
            or meta.get("inputs") != sigs or not npz_path.is_file()):
        return False
    if meta.get("sample_sha256") != file_sha256(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            return all(name in d and d[name].shape == (SAMPLE_COUNT,)
                       and np.isfinite(d[name]).all() for name in CHANNELS)
    except (OSError, ValueError, KeyError):
        return False


def load_nifti(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    pixels = img.get_fdata(dtype=np.float32)
    if pixels.ndim != 3 or min(pixels.shape) < 8:
        raise ValueError(f"Unexpected three-dimensional NIfTI shape: {path}")
    return img, pixels


def calculate_case(raw: Path, case: str) -> tuple[dict, dict]:
    """Read one subject's *image-only* five channels; never open a lesion mask."""
    inputs = ordered_source_files(raw, case)
    ncct_img, ncct = load_nifti(inputs[0])
    support, support_qc = ncct_head_tissue_support(ncct)
    modality_samples = {
        "NCCT": deterministic_training_sample(
            ncct, support, case_id=case, channel="NCCT", max_samples=SAMPLE_COUNT
        )
    }
    n_nonfinite = {"NCCT": 0}
    for i, name in enumerate(CHANNELS[1:], 1):
        img, pixels = load_nifti(inputs[i])
        if (img.shape != ncct_img.shape
                or not np.allclose(img.affine, ncct_img.affine, rtol=0, atol=1e-4)):
            raise ValueError(f"Case {case} modality {name}: shape/affine mismatch")
        n_nonfinite[name] = int(np.count_nonzero(~np.isfinite(pixels)))
        modality_samples[name] = deterministic_training_sample(
            pixels, support, case_id=case, channel=name, max_samples=SAMPLE_COUNT
        )
        del pixels, img
    return modality_samples, {
        "support": support_qc,
        "nonfinite_in_original_channels": n_nonfinite,
        "image_shape": list(ncct.shape),
        "n_samples_per_channel": SAMPLE_COUNT,
        "mask_status": "provisional_head_tissue_NOT_clinical_brain_mask",
    }


def create_checkpoint(out: Path, raw: Path, case: str) -> dict:
    """Atomic write for NPZ first, JSON last; reject partial/corrupt resume."""
    sigs = source_signatures(raw, case)
    samples, qc = calculate_case(raw, case)
    json_path, npz_path = case_paths(out, case)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temp = npz_path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **samples)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, npz_path)
    entry = {
        "schema": SCHEMA, "case_id": case, "inputs": sigs,
        "sample_sha256": file_sha256(npz_path), "qc": qc,
        "output_source": "image_only_unreviewed",
    }
    atomic_json(json_path, entry)
    return entry


def load_samples(out: Path, cases: list[str]) -> dict:
    all_samples = {name: {} for name in CHANNELS}
    for case in cases:
        _, data_path = case_paths(out, case)
        with np.load(data_path, allow_pickle=False) as samples:
            for name in CHANNELS:
                v = np.asarray(samples[name], dtype=np.float32)
                if v.shape != (SAMPLE_COUNT,) or not np.isfinite(v).all():
                    raise ValueError(f"Corrupted stored samples {case}/{name}")
                all_samples[name][case] = v.copy()
    return all_samples


def get_split_fold_cases(split: Path) -> tuple[list[str], list[tuple[list[str], list[str]]]]:
    fold_cases = [validated_split(split, fold) for fold in range(5)]
    ids = sorted({case for _, val in fold_cases for case in val})
    if len(ids) != 149 or any(len(t) + len(v) != 149 for t, v in fold_cases):
        raise ValueError("Frozen 149-case split integrity failure")
    return ids, fold_cases


def aggregate_qc(entries: dict[str, dict]) -> dict:
    fractions = [float(x["qc"]["support"]["coverage_fraction"]) for x in entries.values()]
    if len(fractions) != 149:
        raise ValueError("Expected all 149 audited patients before aggregate")
    return {
        "n_cases": len(fractions), "coverage_min": min(fractions),
        "coverage_median": float(np.median(fractions)),
        "coverage_max": max(fractions),
        "n_support_below_0_10": sum(v < 0.10 for v in fractions),
        "n_support_above_0_40": sum(v > 0.40 for v in fractions),
        "n_raw_nonfinite_mtt_cases": sum(
            x["qc"]["nonfinite_in_original_channels"].get("MTT", 0) > 0
            for x in entries.values()
        ),
    }


def select_overlays(entries: dict[str, dict], limit: int = 16) -> list[str]:
    if limit < len(PRIORITY):
        raise ValueError("Overlay limit cannot omit priority cases")
    ranked = sorted(entries, key=lambda c: entries[c]["qc"]["support"]["coverage_fraction"])
    selected = [case for case in PRIORITY if case in entries]
    for case in ranked[:6] + ranked[-6:]:
        if case not in selected and len(selected) < limit:
            selected.append(case)
    return selected


def generate_overlay(raw: Path, case: str, destination: Path) -> None:
    """Private native-grid overlays for *independent review*, no target labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, pixels = load_nifti(ordered_source_files(raw, case)[0])
    mask, _ = ncct_head_tissue_support(pixels)
    where = np.where(mask)
    z0, z1 = int(where[2].min()), int(where[2].max())
    slices = sorted({round(z0 + frac * (z1 - z0)) for frac in (0.2, 0.5, 0.8)})
    fig, axs = plt.subplots(2, len(slices), figsize=(4 * len(slices), 8), squeeze=False)
    for j, k in enumerate(slices):
        original = np.rot90(pixels[:, :, k])
        support = np.rot90(mask[:, :, k])
        axs[0, j].imshow(original, cmap="gray", vmin=-100, vmax=150, interpolation="nearest")
        axs[0, j].set_title(f"Source, native axis-2 index {k}", fontsize=9)
        axs[1, j].imshow(original, cmap="gray", vmin=-100, vmax=150, interpolation="nearest")
        axs[1, j].imshow(np.ma.masked_where(~support, support), cmap="autumn",
                         interpolation="nearest", alpha=0.36, vmin=0, vmax=1)
        axs[1, j].set_title("Provisional tissue support (orange)", fontsize=9)
        for i in range(2):
            axs[i, j].axis("off")
    fig.suptitle(f"{case}: image-derived head-tissue proxy, NOT verified brain mask", fontsize=11)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".png.tmp")
    fig.savefig(tmp, format="png", dpi=95)
    plt.close(fig)
    os.replace(tmp, destination)


def validate_source(raw: Path, plans: Path) -> dict:
    if raw.name != SOURCE_NAME:
        raise ValueError(f"Wrong frozen input dataset: {raw}")
    expected = {str(i): channel for i, channel in enumerate(CHANNELS)}
    ds = json.loads((raw / "dataset.json").read_text(encoding="utf8"))
    if ds.get("channel_names") != expected or ds.get("numTraining") != 149:
        raise ValueError("Dataset505 channel inventory or size has changed")
    source = json.loads(plans.read_text(encoding="utf8"))
    config = source["configurations"]["3d_fullres"]
    if config["normalization_schemes"] != ["ZScoreNormalization"] * 5:
        raise ValueError("Frozen source plans normalization mismatch")
    # These MUST be locked at a later stage in both training arms. No training yet.
    return {
        "plans_sha256": file_sha256(plans),
        "proposed_3d_fullres_lock": {
            "spacing": config["spacing"],
            "patch_size": config["patch_size"],
            "batch_size": config["batch_size"],
            "architecture": config["architecture"],
            "resampling_fn_data": config["resampling_fn_data"],
            "resampling_fn_data_kwargs": config["resampling_fn_data_kwargs"],
            "resampling_fn_seg": config["resampling_fn_seg"],
            "resampling_fn_seg_kwargs": config["resampling_fn_seg_kwargs"],
        },
        "architecture_lock_status": "PROPOSED_NOT_VERIFIED_IN_NEW_NNUNET_PLANS",
    }


def full_calibration(raw: Path, splits: Path, plans: Path, output: Path,
                     *, render_overlays: bool = True, limit_new_cases: int = 0) -> dict:
    """Resume patient image reads, then fit five provisional fold-only parameter sets."""
    if limit_new_cases < 0:
        raise ValueError("Negative case limit")
    source = validate_source(raw, plans)
    cases, folds = get_split_fold_cases(splits)
    protected = [raw.resolve(), raw.parent.resolve()]
    if plans.parent.name.startswith("Dataset505_"):
        protected.append(plans.parent.resolve())
    target = output.resolve()
    if any(p == target or p in target.parents for p in protected):
        raise ValueError("Output must not be inside frozen source folder")
    if output.is_symlink():
        raise ValueError("Refuse symlink output")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "WU20B_progress.json"
    # Fail-closed: any interrupted/restarted audit invalidates previous aggregates.
    atomic_json(output / "WU20B_aggregate.json", {
        "schema": SCHEMA, "status": "IN_PROGRESS_UNVERIFIED", "training_ready": False,
    })
    entries: dict[str, dict] = {}
    errors: dict[str, str] = {}
    newly_processed = 0
    for i, case in enumerate(cases, 1):
        if limit_new_cases and newly_processed >= limit_new_cases:
            break
        try:
            sig = source_signatures(raw, case)
            meta_path, sample_path = case_paths(output, case)
            saved = json.loads(meta_path.read_text(encoding="utf8")) if meta_path.is_file() else {}
            if verify_checkpoint(saved, sig, sample_path, case):
                entry, status = saved, "RESUMED"
            else:
                entry = create_checkpoint(output, raw, case)
                status = "COMPUTED"
                newly_processed += 1
            entries[case] = entry
            print(f"[{i:03d}/149] {case} {status} support={entry['qc']['support']['coverage_fraction']:.2%}", flush=True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            errors[case] = f"{type(exc).__name__}: {exc}"
            print(f"[{i:03d}/149] {case} QC_ERROR {errors[case]}", flush=True)
        atomic_json(progress, {
            "schema": SCHEMA, "status": "RUNNING", "completed_verified": len(entries),
            "errors": errors, "total": len(cases), "current_case": case,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
    if len(entries) != 149 or errors:
        status = {"status": "PARTIAL_OR_FAILED", "completed_verified": len(entries),
                  "total": len(cases), "n_errors": len(errors), "training_ready": False,
                  "run_more": True, "errors": errors}
        atomic_json(progress, {"schema": SCHEMA, **status})
        atomic_json(output / "WU20B_aggregate.json", {"schema": SCHEMA, **status})
        return status
    # Loading checkpoints is cheap and does not touch labels/outcomes.
    samples = load_samples(output, cases)
    fold_reports = []
    for fold, (train, val) in enumerate(folds):
        fitted = fit_fold_training_only(
            {name: {case: samples[name][case] for case in train} for name in CHANNELS},
            training_ids=train, excluded_ids=val,
        )
        if any(len(fitted[name]["training_ids"]) != len(train)
               or set(fitted[name]["validation_ids_excluded"]) != set(val)
               for name in CHANNELS):
            raise ValueError("Fail-closed: fold-only fitting patient audit mismatch")
        provenance = {
            "schema": SCHEMA, "fold": fold,
            "frozen_split_sha256": file_sha256(splits),
            "n_training_patients": len(train), "n_heldout_patients": len(val),
            "train_ids": train, "heldout_ids": val,
            "source_sample_sha256": {case: entries[case]["sample_sha256"] for case in train},
            "provisional_fit": fitted,
            "status": "TRAIN_ONLY_FIT_COMPLETE_QC_PENDING",
            "training_ready": False,
        }
        atomic_json(output / "fold_private" / f"WU20B_fold{fold}_parameters_PRIVATE.json", provenance)
        fold_reports.append({
            "fold": fold, "n_train": len(train), "n_heldout": len(val),
            "n_fit_samples_per_modality": fitted["NCCT"]["n_samples"],
            "shared_ncct_policy_between_arms": True,
        })
    qc_summary = aggregate_qc(entries)
    overlays = select_overlays(entries)
    if render_overlays:
        for case in overlays:
            dest = output / "private_mask_overlays" / f"{case}_support_QC.png"
            if not dest.is_file() or dest.stat().st_size < 100:
                generate_overlay(raw, case, dest)
    summary = {
        "schema": SCHEMA, "status": "FULL_FOLD_PROVISIONAL_FIT_COMPLETE",
        "n_patients": len(cases), "n_channel_case_samples": len(cases) * RAW_PATIENT_CHANNEL_COUNT,
        "n_folds": 5, "folds": fold_reports, "qc_aggregate": qc_summary,
        "n_review_overlays": len(overlays) if render_overlays else 0,
        "source_plans_sha256": source["plans_sha256"],
        "frozen_split_sha256": file_sha256(splits),
        "training_ready": False, "model_training_started": False,
        "provisional_anatomical_mask": True,
        "mask_review_status": "NOT_CLINICALLY_APPROVED",
        "nnunet_plan_inference_parity": "PENDING_REAL_END_TO_END_QUALIFICATION",
        "blocked_release_gates": [
            "Independent clinical/anatomical review of all flagged masks and candidate clipped ranges",
            "Future fold-specific nnU-Net plans/spacing and noNorm mapping verified in pinned version",
            "End-to-end on-disk nnU-Net preprocessing and heldout inference smoke",
            "Prospective signed and frozen QC/normalization policy before training",
        ],
    }
    atomic_json(output / "WU20B_aggregate.json", summary)
    atomic_json(output / "WU20B_proposed_architecture_lock_PRIVATE.json", source)
    atomic_json(progress, {"schema": SCHEMA, "status": "COMPLETE",
                           "completed_verified": 149, "total": 149, "training_ready": False,
                           "updated_utc": datetime.now(timezone.utc).isoformat()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw505", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--plans505", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-new-cases", type=int, default=0,
                        help="Optional number of freshly computed cases per session (0 = unlimited)")
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()
    result = full_calibration(args.raw505, args.splits, args.plans505,
                              args.output_dir, limit_new_cases=args.limit_new_cases,
                              render_overlays=not args.skip_overlays)
    print(json.dumps({"status": result["status"], "training_ready": False,
                      "completed": result.get("n_patients", result.get("completed_verified"))},
                     indent=2))
    if result["status"] == "PARTIAL_OR_FAILED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
