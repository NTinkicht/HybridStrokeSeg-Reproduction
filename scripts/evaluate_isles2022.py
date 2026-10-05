#!/usr/bin/env python3
"""Evaluate nnU-Net-style predictions with ISLES 2022 challenge metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hybridstrokeseg.data import (
    discover_isles2022_cases,
    geometry_equal,
    geometry_signature,
    load_volume,
)
from hybridstrokeseg.metrics_isles2022 import evaluate_isles2022
from hybridstrokeseg.nnunet import nnunet_case_id

METRIC_FIELDS = (
    "dice",
    "absolute_volume_difference_ml",
    "absolute_lesion_count_difference",
    "lesion_f1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "predictions",
        type=Path,
        help="Folder containing nnU-Net-style {case_id}.nii.gz predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/isles2022_evaluation"),
    )
    args = parser.parse_args()

    cases = discover_isles2022_cases(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for case in cases:
        prediction_path = args.predictions / f"{nnunet_case_id(case)}.nii.gz"
        if not prediction_path.is_file():
            missing.append(str(prediction_path))
            continue

        if not geometry_equal(geometry_signature(case.mask), geometry_signature(prediction_path)):
            raise ValueError(
                f"Prediction geometry does not match the ground-truth mask for {case.case_id}: "
                f"{prediction_path}"
            )

        ground_truth, metadata = load_volume(case.mask)
        prediction, _ = load_volume(prediction_path)
        metrics = evaluate_isles2022(
            ground_truth > 0,
            prediction > 0,
            spacing_xyz_mm=metadata["spacing_xyz"],
        )
        row: dict[str, object] = {"case_id": case.case_id, "prediction": str(prediction_path)}
        row.update(metrics.as_dict())
        rows.append(row)

    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing predictions for {len(missing)}/{len(cases)} cases:\n{preview}{suffix}"
        )

    per_case_path = args.output_dir / "per_case_metrics.csv"
    with per_case_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "cases": len(rows),
        "connectivity": 26,
        "empty_mask_value_dice": 1.0,
        "empty_mask_value_lesion_f1": 1.0,
        "metrics": {},
    }
    metrics_summary = summary["metrics"]
    assert isinstance(metrics_summary, dict)
    for field in METRIC_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        metrics_summary[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary_path = args.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Evaluated {len(rows)} cases.")
    for field in METRIC_FIELDS:
        stats = metrics_summary[field]
        print(f"{field}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"Per-case CSV: {per_case_path}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
