#!/usr/bin/env python3
"""Read-only, resumable patient-level ISLES'24 perfusion intensity audit.

Flags possible extreme perfusion-map values *without assuming that a negative
value is necessarily invalid*. No clipping, changes to models, or overwrites.
Uses whole native volumes (not exactly nnU-Net's cropped ZScore voxels).
Patient-level outputs must stay in private Google Drive.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHANNELS = ("NCCT", "TMAX", "CBF", "CBV", "MTT")
FROZEN_SPLIT_SHA = "664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027"


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def split_cases(split_path: Path) -> list[str]:
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if digest != FROZEN_SPLIT_SHA:
        raise ValueError(f"Frozen fold checksum mismatch: {digest}")
    folds = json.loads(split_path.read_text(encoding="utf8"))
    if len(folds) != 5 or [len(f["val"]) for f in folds] != [30, 30, 30, 30, 29]:
        raise ValueError("Frozen fold count/size mismatch")
    values = [str(case) for fold in folds for case in fold["val"]]
    if len(values) != 149 or len(set(values)) != 149:
        raise ValueError("Expected exactly 149 unique validation cases")
    for i, fold in enumerate(folds):
        if set(fold["train"]) != set(values) - set(fold["val"]):
            raise ValueError(f"Train/validation leakage or incomplete complement: fold {i}")
    return sorted(values)


def summarize_pixels(array: np.ndarray) -> dict:
    """Full-volume extrema and counts; deterministic <=1M-voxel quantiles."""
    arr = np.asarray(array).ravel()
    finite = np.isfinite(arr)
    n = int(arr.size)
    finite_n = int(finite.sum())
    if not finite_n:
        raise ValueError("Volume has no finite voxel intensities")
    good = arr[finite]
    # Full-volume extrema and extreme-value counts (not sample estimates).
    exact_min, exact_max = float(np.min(good)), float(np.max(good))
    below0 = int(np.count_nonzero(good < 0))
    below1k = int(np.count_nonzero(good < -1000))
    below10k = int(np.count_nonzero(good < -10000))
    above10k = int(np.count_nonzero(good > 10000))
    # Quantiles are deterministic subsampled diagnostics for resource control.
    skip = max(1, int(np.ceil(finite_n / 1_000_000)))
    sample = good[::skip]
    q = np.quantile(sample, [0.001, 0.005, 0.5, 0.995, 0.999])
    out = {
        "n_voxels": n,
        "n_finite": finite_n,
        "n_nonfinite": n - finite_n,
        "min": exact_min,
        "max": exact_max,
        "count_negative": below0,
        "percent_negative": round(100.0 * below0 / finite_n, 6),
        "count_below_minus_1000": below1k,
        "count_below_minus_10000": below10k,
        "count_above_plus_10000": above10k,
        "sample_n": int(len(sample)),
        "q001": float(q[0]),
        "q005": float(q[1]),
        "median": float(q[2]),
        "q995": float(q[3]),
        "q999": float(q[4]),
        "mean_full": float(np.mean(good, dtype=np.float64)),
        "std_full": float(np.std(good, dtype=np.float64)),
    }
    del good, sample, finite
    return out


def image_signature(path: Path) -> dict:
    s = path.stat()
    return {"file": str(path), "size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def audit_case(case: str, image_root: Path, *, tries: int) -> dict:
    import nibabel as nib  # Runtime-only; no unnecessary dependency for fast tests.

    signatures = [image_signature(image_root / f"{case}_{idx:04d}.nii.gz") for idx in range(5)]
    channels = {}
    base_shape = None
    for idx, label in enumerate(CHANNELS):
        path = image_root / f"{case}_{idx:04d}.nii.gz"
        last_exc: Exception | None = None
        for retry in range(tries):
            try:
                image = nib.load(str(path))
                if base_shape is None:
                    base_shape = tuple(image.shape)
                elif tuple(image.shape) != base_shape:
                    raise ValueError(f"Shape mismatch: {case}/{label}: {image.shape} != {base_shape}")
                values = np.asanyarray(image.dataobj)
                channels[label] = summarize_pixels(values)
                del values, image
                gc.collect()
                last_exc = None
                break
            except (OSError, EOFError) as exc:
                last_exc = exc
                print(f"Drive read retry {retry+1}/{tries}: {case}/{label}: {exc}", flush=True)
                time.sleep(2)
        if last_exc:
            raise last_exc
    return {"schema": 1, "case_id": case, "channels": channels, "inputs": signatures}


def run(image_root: Path, split_file: Path, out: Path, *, retries: int = 3) -> None:
    if retries < 1:
        raise ValueError("retries must be positive")
    cases = split_cases(split_file)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    out.mkdir(parents=True, exist_ok=True)
    per_case = out / "patient_checkpoints"
    per_case.mkdir(exist_ok=True)
    progress = out / "intensity_audit_progress.json"
    all_rows = []
    for i, case in enumerate(cases):
        checkpoint = per_case / f"{case}.json"
        try:
            signatures = [image_signature(image_root / f"{case}_{idx:04d}.nii.gz") for idx in range(5)]
            saved = json.loads(checkpoint.read_text(encoding="utf8")) if checkpoint.is_file() else None
            if saved and saved.get("schema") == 1 and saved.get("inputs") == signatures:
                entry, state = saved, "RESUMED"
            else:
                entry, state = audit_case(case, image_root, tries=retries), "COMPUTED"
                atomic_json(checkpoint, entry)
            for channel, stats in entry["channels"].items():
                all_rows.append({"case_id": case, "channel": channel, **stats})
            atomic_json(progress, {"status": "running", "completed_cases": i + 1,
                                   "total_cases": len(cases), "last_case": case,
                                   "percent": round(100*(i+1)/len(cases), 2),
                                   "updated_utc": datetime.now(timezone.utc).isoformat()})
            mtt = entry["channels"]["MTT"]
            print(f"[{i+1:03d}/149] {100*(i+1)/149:.1f}% {case} {state}: MTT min={mtt['min']:,.3f} "
                  f"below -10k={mtt['count_below_minus_10000']}", flush=True)
        except Exception as exc:
            atomic_json(progress, {"status": "failed", "completed_cases": i,
                                   "total_cases": len(cases), "last_case": case,
                                   "error": f"{type(exc).__name__}: {exc}"})
            raise
    csv_path = out / "isles2024_perfusion_case_intensity_audit_PRIVATE.csv"
    with csv_path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    aggregate = {"dataset": "Dataset505_ISLES2024_NCCT_PERFUSION",
                 "n_patients": len(cases), "n_channel_volumes": len(all_rows),
                 "split_sha256": FROZEN_SPLIT_SHA, "read_only": True,
                 "raw_data_processed_in_original_units": True,
                 "note": "Full-volume intensities; not nnU-Net cropped training-mask statistics. "
                         "Flags are diagnostic, not automatically validated clinical bounds.",
                 "channels": {}}
    for name in CHANNELS:
        a = [r for r in all_rows if r["channel"] == name]
        aggregate["channels"][name] = {
            "n_patients": len(a),
            "minimum_across_patients": min(x["min"] for x in a),
            "maximum_across_patients": max(x["max"] for x in a),
            "cases_with_nonfinite_voxels": sum(x["n_nonfinite"] > 0 for x in a),
            "cases_with_values_below_minus_10000": sum(x["count_below_minus_10000"] > 0 for x in a),
            "total_voxels_below_minus_10000": sum(x["count_below_minus_10000"] for x in a),
            "cases_with_values_below_minus_1000": sum(x["count_below_minus_1000"] > 0 for x in a),
        }
    atomic_json(out / "isles2024_perfusion_intensity_audit_aggregate.json", aggregate)
    atomic_json(progress, {"status": "complete", "completed_cases": 149,
                           "total_cases": 149, "percent": 100.0,
                           "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Complete: 149/149. Private patient-level CSV: {csv_path}")
    print("Aggregate diagnostic:", out / "isles2024_perfusion_intensity_audit_aggregate.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-tr", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    run(args.images_tr, args.split_file, args.output_dir, retries=args.retries)


if __name__ == "__main__":
    main()