#!/usr/bin/env python3
"""Read-only, resumable ISLES'24 Dataset505 3D full-resolution tensor audit.

Compares *staged raw-volume summaries* with exact nonfinite and extreme-value
counts of persisted preprocessed 5-channel b2nd arrays. This is NOT a
voxel-aligned comparison (nnU-Net crops/resamples/normalizes), and the cached
Drive tensors might differ from tensors reconstructed on local SSD for a
specific training session. Never change any input, model, or raw image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHANNELS = ("NCCT", "TMAX", "CBF", "CBV", "MTT")
FROZEN_SPLIT_SHA = "664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027"
SCHEMA = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def frozen_cases(splits: Path) -> list[str]:
    digest = sha256_file(splits)
    if digest != FROZEN_SPLIT_SHA:
        raise ValueError(f"Frozen split hash mismatch: {digest}")
    folds = json.loads(splits.read_text(encoding="utf-8"))
    if len(folds) != 5 or [len(f["val"]) for f in folds] != [30, 30, 30, 30, 29]:
        raise ValueError("Unexpected fold counts; expected 30/30/30/30/29")
    all_ids = [p for fold in folds for p in fold["val"]]
    if len(all_ids) != 149 or len(set(all_ids)) != 149:
        raise ValueError("Must have 149 unique validation patients")
    expected = set(all_ids)
    for fold in folds:
        if set(fold["train"]) != expected - set(fold["val"]):
            raise ValueError("Patient leakage or incomplete training complement")
    return sorted(expected)


def load_raw_audit(path: Path, cases: list[str]) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "channel", "min", "max", "n_nonfinite", "n_voxels", "count_below_minus_10000"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Raw Notebook 17 audit CSV lacks fields: {required - set(reader.fieldnames or [])}")
        values = list(reader)
    expected_keys = {(case, name) for case in cases for name in CHANNELS}
    pairs = [(x["case_id"], x["channel"]) for x in values]
    if len(pairs) != 745 or set(pairs) != expected_keys or len(set(pairs)) != 745:
        raise ValueError("Raw Notebook 17 audit must contain exactly 745 unique matching patient/channel records")
    return {key: row for key, row in zip(pairs, values, strict=True)}


def input_signature(path: Path) -> dict:
    stat = path.stat()
    if not stat.st_size:
        raise ValueError(f"Empty tensor or metadata: {path}")
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def summarize_channel(array, channel: int, *, slices_per_block: int = 8, sample_max: int = 50000) -> dict:
    """Exact full-volume counts and stable moments, sampled quantiles; bounded RAM.

    ``array`` must provide shape and slice indexing as for blosc2.NDArray.
    Indexing an entire channel at once is intentionally avoided.
    """
    if slices_per_block < 1 or sample_max < 10:
        raise ValueError("Invalid block or sample size")
    shape = tuple(int(n) for n in array.shape)
    if len(shape) != 4 or shape[0] != 5 or any(n < 1 for n in shape):
        raise ValueError(f"Expected (5, z, y, x) normalized tensor; found {shape}")
    if channel not in range(5):
        raise ValueError(f"Invalid channel {channel}")
    slice_size = shape[2] * shape[3]
    spatial_n = shape[1] * slice_size
    step = max(1, (spatial_n + sample_max - 1) // sample_max)
    counts = {"n_voxels": 0, "n_finite": 0, "n_nan": 0, "n_posinf": 0, "n_neginf": 0,
              "n_zero": 0, "n_abs_gt_6": 0, "n_abs_gt_10": 0, "n_abs_gt_30": 0,
              "n_abs_gt_100": 0, "n_abs_gt_1000": 0}
    minimum, maximum, average, sum_of_squares = None, None, 0.0, 0.0
    sample_parts = []
    for z0 in range(0, shape[1], slices_per_block):
        z1 = min(shape[1], z0 + slices_per_block)
        slab = np.asarray(array[channel, z0:z1, :, :])
        if slab.shape != (z1 - z0, shape[2], shape[3]):
            raise ValueError(f"Slice shape mismatch for channel={channel} z={z0}:{z1}")
        flattened = slab.reshape(-1)
        counts["n_voxels"] += flattened.size
        counts["n_nan"] += int(np.count_nonzero(np.isnan(flattened)))
        counts["n_posinf"] += int(np.count_nonzero(np.isposinf(flattened)))
        counts["n_neginf"] += int(np.count_nonzero(np.isneginf(flattened)))
        good = flattened[np.isfinite(flattened)].astype(np.float64)
        n1 = len(good)
        counts["n_zero"] += int(np.count_nonzero(good == 0))
        if n1:
            absvals = np.abs(good)
            for threshold in (6, 10, 30, 100, 1000):
                counts[f"n_abs_gt_{threshold}"] += int(np.count_nonzero(absvals > threshold))
            local_min, local_max = float(good.min()), float(good.max())
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = local_max if maximum is None else max(maximum, local_max)
            local_mean = float(np.mean(good))
            local_m2 = float(np.sum(np.square(good - local_mean), dtype=np.float64))
            n0 = counts["n_finite"]
            n = n0 + n1
            diff = local_mean - average
            average += diff * n1 / n
            sum_of_squares += local_m2 + diff * diff * n0 * n1 / n
            counts["n_finite"] = n
        # Stable global flat offsets, not a biased resampling of image sections.
        first = (-(z0 * slice_size)) % step
        part = flattened[first::step]
        part = part[np.isfinite(part)]
        if part.size:
            sample_parts.append(part.astype(np.float32))
    if counts["n_voxels"] != spatial_n:
        raise ValueError("Preprocessed tensor shape/voxel count mismatch")
    values = np.concatenate(sample_parts) if sample_parts else np.asarray([], dtype=np.float32)
    if values.size:
        q = np.quantile(values.astype(np.float64), [0.001, 0.01, 0.5, 0.99, 0.999])
        quantiles = {key: float(v) for key, v in zip(("q001", "q01", "median", "q99", "q999"), q, strict=True)}
    else:
        quantiles = {key: None for key in ("q001", "q01", "median", "q99", "q999")}
    variance = sum_of_squares / counts["n_finite"] if counts["n_finite"] else None
    std = float(np.sqrt(max(0.0, variance))) if variance is not None else None
    return {
        **counts,
        "min": minimum, "max": maximum,
        "mean": float(average) if counts["n_finite"] else None,
        "std": std,
        "n_nonfinite": counts["n_voxels"] - counts["n_finite"],
        "all_nonfinite": not bool(counts["n_finite"]),
        "near_constant": std is not None and std < 1e-6,
        "sample_n": values.size,
        **quantiles,
    }


def audit_case(case: str, data_path: Path, segmentation_path: Path, *, block: int, retries: int) -> dict:
    import blosc2

    last = None
    for attempt in range(retries):
        try:
            tensor = blosc2.open(str(data_path), mode="r")
            shape = tuple(int(x) for x in tensor.shape)
            if len(shape) != 4 or shape[0] != 5:
                raise ValueError(f"{case}: unexpected tensor shape {shape}")
            label = blosc2.open(str(segmentation_path), mode="r")
            label_shape = tuple(int(x) for x in label.shape)
            if label_shape[-3:] != shape[-3:] or len(label_shape) not in (3, 4):
                raise ValueError(f"{case}: segmentation shape {label_shape} incompatible with {shape}")
            if len(label_shape) == 4 and label_shape[0] != 1:
                raise ValueError(f"{case}: unexpected segmentation channel dimension {label_shape}")
            all_stats = {name: summarize_channel(tensor, i, slices_per_block=block)
                         for i, name in enumerate(CHANNELS)}
            return {"shape": shape, "dtype": str(tensor.dtype), "seg_shape": label_shape,
                    "channels": all_stats}
        except (OSError, EOFError, FileNotFoundError) as exc:
            last = exc
            print(f"Drive I/O retry {attempt + 1}/{retries} {case}: {exc}", flush=True)
            if attempt + 1 < retries:
                time.sleep(3)
    if last is not None:
        raise last
    raise RuntimeError(f"Unreachable state while evaluating {case}")


def run(dataset: Path, splits: Path, raw_csv: Path, out: Path, *, block: int = 8, retries: int = 3) -> None:
    if block < 1 or retries < 1:
        raise ValueError("Block and retries must be positive")
    cases = frozen_cases(splits)
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    plan = dataset / "nnUNetPlans.json"
    dataset_json = dataset / "dataset.json"
    with plan.open(encoding="utf8") as f:
        plans = json.load(f)
    with dataset_json.open(encoding="utf8") as f:
        metadata = json.load(f)
    expected = {str(i): ch for i, ch in enumerate(CHANNELS)}
    if metadata.get("channel_names") != expected:
        raise ValueError("The Dataset505 channel order differs from the frozen NCCT/TMAX/CBF/CBV/MTT protocol")
    normalization = plans["configurations"]["3d_fullres"]["normalization_schemes"]
    if len(normalization) != 5:
        raise ValueError("Unexpected number of normalization schemes")
    config = dataset / "nnUNetPlans_3d_fullres"
    if not config.is_dir():
        raise FileNotFoundError(config)
    raw_sha = sha256_file(raw_csv)
    raw_rows = load_raw_audit(raw_csv, cases)
    plan_sha = sha256_file(plan)
    script_sha = sha256_file(Path(__file__))
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "patient_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    rows = []
    progress = out / "preprocessed_tensor_audit_progress.json"
    for i, case in enumerate(cases):
        tensor_path = config / f"{case}.b2nd"
        segmentation = config / f"{case}_seg.b2nd"
        metadata_path = config / f"{case}.pkl"
        checkpoint = checkpoints / f"{case}.json"
        try:
            signatures = [input_signature(x) for x in (tensor_path, segmentation, metadata_path)]
            signature = {"files": signatures, "raw_audit_sha256": raw_sha, "plans_sha256": plan_sha,
                         "split_sha256": FROZEN_SPLIT_SHA, "script_sha256": script_sha, "block": block}
            prior = json.loads(checkpoint.read_text(encoding="utf8")) if checkpoint.is_file() else None
            valid = prior and prior.get("schema") == SCHEMA and prior.get("signature") == signature
            if valid:
                entry, state = prior, "RESUMED"
            else:
                result = audit_case(case, tensor_path, segmentation, block=block, retries=retries)
                entry = {"schema": SCHEMA, "case_id": case, "signature": signature, **result}
                atomic_json(checkpoint, entry)
                state = "COMPUTED"
            if set(entry["channels"]) != set(CHANNELS):
                raise ValueError(f"Malformed case checkpoint for {case}")
            for ch in CHANNELS:
                raw = raw_rows[(case, ch)]
                z = entry["channels"][ch]
                rows.append({
                    "case_id": case, "channel": ch,
                    "raw_min": float(raw["min"]),
                    "raw_max": float(raw["max"]),
                    "raw_n_nonfinite": int(raw["n_nonfinite"]),
                    "raw_n_below_minus_10000": int(raw["count_below_minus_10000"]),
                    **{f"pre_{key}": value for key, value in z.items()},
                })
            atomic_json(progress, {"status": "running", "completed_cases": i + 1,
                                   "total_cases": 149, "last_case": case,
                                   "updated_utc": datetime.now(timezone.utc).isoformat()})
            mtt = entry["channels"]["MTT"]
            print(f"[{i + 1:03d}/149] {case} {state} "
                  f"MTT pre NaN/Inf={mtt['n_nonfinite']} |z|>100={mtt['n_abs_gt_100']} "
                  f"mean={mtt['mean']}", flush=True)
        except Exception as exc:
            atomic_json(progress, {"status": "failed", "completed_cases": i,
                                   "total_cases": 149, "last_case": case,
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "updated_utc": datetime.now(timezone.utc).isoformat()})
            raise
    if len(rows) != 745 or len({(r['case_id'], r['channel']) for r in rows}) != 745:
        raise ValueError("Full 149x5 tensor scan not complete")
    csv_path = out / "isles2024_preprocessed_tensor_audit_PRIVATE.csv"
    with csv_path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for channel in CHANNELS:
        group = [x for x in rows if x["channel"] == channel]
        grouped[channel] = {
            "n_cases": len(group),
            "raw_cases_with_nonfinite": sum(x["raw_n_nonfinite"] > 0 for x in group),
            "pre_cases_with_nonfinite": sum(x["pre_n_nonfinite"] > 0 for x in group),
            "pre_total_nonfinite": sum(x["pre_n_nonfinite"] for x in group),
            "pre_cases_abs_gt_10": sum(x["pre_n_abs_gt_10"] > 0 for x in group),
            "pre_cases_abs_gt_100": sum(x["pre_n_abs_gt_100"] > 0 for x in group),
            "pre_total_abs_gt_100": sum(x["pre_n_abs_gt_100"] for x in group),
            "pre_cases_near_constant": sum(x["pre_near_constant"] for x in group),
            "pre_cases_all_nonfinite": sum(x["pre_all_nonfinite"] for x in group),
            "raw_cases_below_minus_10000": sum(x["raw_n_below_minus_10000"] > 0 for x in group),
            "pre_minimum": min((x["pre_min"] for x in group if x["pre_min"] is not None), default=None),
            "pre_maximum": max((x["pre_max"] for x in group if x["pre_max"] is not None), default=None),
        }
    summary = {
        "dataset": dataset.name, "configuration": "3d_fullres",
        "n_patients": 149, "n_channel_volumes": 745, "split_sha256": FROZEN_SPLIT_SHA,
        "plans_sha256": plan_sha, "raw_audit_csv_sha256": raw_sha, "script_sha256": script_sha,
        "normalization_schemes": dict(zip(CHANNELS, normalization, strict=True)),
        "read_only_input": True,
        "method": "Exact full-data nonfinite and absolute-z threshold counts; moments aggregated over slabs; percentiles sampled deterministically",
        "limitation": "Read-only persisted Drive tensors, which may differ from training-session local-rebuilt preprocessing; raw and preprocessed volumes are resampled/cropped, not voxel aligned.",
        "channels": grouped,
        "priority_cases": [
            {"case_id": case, "raw_mtt_min": next(r["raw_min"] for r in rows if r["case_id"] == case and r["channel"] == "MTT"),
             "pre_mtt_n_nonfinite": next(r["pre_n_nonfinite"] for r in rows if r["case_id"] == case and r["channel"] == "MTT"),
             "pre_mtt_abs_gt_100": next(r["pre_n_abs_gt_100"] for r in rows if r["case_id"] == case and r["channel"] == "MTT")}
            for case in ("sub_stroke0016", "sub_stroke0113", "sub_stroke0137", "sub_stroke0152")
        ],
    }
    atomic_json(out / "isles2024_preprocessed_tensor_audit_aggregate.json", summary)
    atomic_json(progress, {"status": "complete", "completed_cases": 149, "total_cases": 149,
                           "updated_utc": datetime.now(timezone.utc).isoformat()})
    print("COMPLETE: 149/149 patients x 5 normalized channels")
    print("Aggregate:", out / "isles2024_preprocessed_tensor_audit_aggregate.json")
    print("Patient-level CSV (private):", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dataset", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--raw-audit-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slices-per-block", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    run(args.preprocessed_dataset, args.split_file, args.raw_audit_csv, args.output_dir,
        block=args.slices_per_block, retries=args.retries)


if __name__ == "__main__":
    main()
