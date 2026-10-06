#!/usr/bin/env python3
"""Deliberately test voxel-level train/test leakage implied by manuscript wording.

This diagnostic is intentionally *not* a valid clinical protocol. It keeps the
same patients on both sides of the split by randomly partitioning pixels within
each selected slice. Its only purpose is to test whether the manuscript's
contradictory 70/30 and 70/15/15 language could materially inflate performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from hybridstrokeseg.data import discover_siss_cases, load_volume
from hybridstrokeseg.diagnostics import random_pixel_partition
from hybridstrokeseg.features import FeatureConfig
from hybridstrokeseg.metrics import evaluate_binary_segmentation
from hybridstrokeseg.models import SVMConfig, continuous_scores, make_rbf_svm
from hybridstrokeseg.pipeline import prepare_slice, select_slice_index
from hybridstrokeseg.preprocessing import PreprocessConfig
from hybridstrokeseg.sampling import balanced_binary_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/voxel_leakage"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-per-class", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_flair_and_mask(case) -> tuple[np.ndarray, np.ndarray]:
    flair, _ = load_volume(case.flair)
    lesion, _ = load_volume(case.ot)
    if flair.shape != lesion.shape:
        raise ValueError(f"FLAIR/mask shape mismatch for {case.case_id}")
    return np.asarray(flair), np.asarray(lesion)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_case_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in ("dice", "precision", "recall", "auc"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        output[f"{metric}_mean"] = float(finite.mean()) if finite.size else math.nan
        output[f"{metric}_std"] = (
            float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size else math.nan
        )
    return output


def run_condition(
    *,
    cases,
    split_name: str,
    train_fraction: float,
    validation_fraction: float,
    scaler: str,
    seed: int,
    target_per_class: int,
    output_dir: Path,
) -> dict[str, object]:
    summary_path = output_dir / "summary_metrics.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        return {key: float(value) if key not in {"configuration", "split", "scaler"} else value for key, value in row.items()}

    preprocess_config = PreprocessConfig(mode="paper_minimal")
    feature_config = FeatureConfig(
        threshold=20.0,
        run_window=25,
        run_mode="threshold_count",
        coordinate_mode="raw",
    )

    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    heldout: list[tuple[str, np.ndarray, np.ndarray]] = []
    selected_slices: dict[str, int] = {}

    for case_index, case in enumerate(cases):
        flair, lesion = load_flair_and_mask(case)
        slice_index = select_slice_index(flair, lesion, strategy="max_lesion")
        selected_slices[case.case_id] = slice_index
        prepared = prepare_slice(
            flair[slice_index],
            lesion[slice_index],
            preprocess_config=preprocess_config,
            feature_config=feature_config,
        )
        if prepared.y is None:
            raise AssertionError("Labels unexpectedly missing")

        partition = random_pixel_partition(
            prepared.X.shape[0],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=seed + case_index,
        )
        train_X.append(prepared.X[partition.train])
        train_y.append(prepared.y[partition.train])
        heldout.append((case.case_id, prepared.X[partition.test], prepared.y[partition.test]))

    X = np.concatenate(train_X, axis=0)
    y = np.concatenate(train_y, axis=0)
    sample = balanced_binary_sample(
        X,
        y,
        target_per_class=target_per_class,
        seed=seed,
    )
    print(
        f"{split_name}/{scaler}: balanced train sample "
        f"{sample.positive_count} lesion + {sample.negative_count} non-lesion pixels"
    )

    model = make_rbf_svm(SVMConfig(C=1.0, gamma="scale", scaler=scaler))
    model.fit(sample.X, sample.y)

    case_rows: list[dict[str, object]] = []
    for case_id, X_test, y_test in heldout:
        prediction = model.predict(X_test)
        scores = continuous_scores(model, X_test)
        metrics = evaluate_binary_segmentation(y_test, prediction, scores=scores)
        case_rows.append(
            {
                "case_id": case_id,
                "model": "svm",
                **metrics.as_dict(),
            }
        )
        print(
            f"{split_name}/{scaler} {case_id}: Dice={metrics.dice:.3f} "
            f"P={metrics.precision:.3f} R={metrics.recall:.3f} AUC={metrics.auc:.3f}"
        )

    stats = summarize_case_rows(case_rows)
    configuration = f"{split_name}_{scaler}"
    summary = {
        "configuration": configuration,
        "split": split_name,
        "scaler": scaler,
        "cases": len(case_rows),
        **stats,
    }
    write_csv(output_dir / "per_case_metrics.csv", case_rows)
    write_csv(output_dir / "summary_metrics.csv", [summary])
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "purpose": "deliberate voxel-level leakage diagnostic, not a valid evaluation protocol",
                "split": split_name,
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "test_fraction": 1.0 - train_fraction - validation_fraction,
                "same_patients_in_train_and_test": True,
                "oracle_slice_selection": True,
                "selected_slices": selected_slices,
                "target_per_class": sample.positive_count,
                "svm": {"C": 1.0, "gamma": "scale", "scaler": scaler},
                "morphology": "not applicable because evaluation is on randomly held-out pixels",
                "seed": seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = discover_siss_cases(args.data_root)
    if len(cases) != 28:
        raise RuntimeError(f"Expected 28 labeled SISS cases, found {len(cases)}")

    print(
        "LEAKAGE DIAGNOSTIC ONLY: patients intentionally appear in both training and test pixels. "
        "These scores must never be reported as clinically valid segmentation performance."
    )

    conditions = (
        ("pixel_70_30", 0.70, 0.00, "standard"),
        ("pixel_70_30", 0.70, 0.00, "none"),
        ("pixel_70_15_15", 0.70, 0.15, "standard"),
        ("pixel_70_15_15", 0.70, 0.15, "none"),
    )

    rows: list[dict[str, object]] = []
    for index, (split_name, train_fraction, val_fraction, scaler) in enumerate(conditions, start=1):
        output_dir = args.output_root / f"{split_name}_{scaler}"
        print(f"\n[{index}/{len(conditions)}] {split_name} / scaler={scaler}")
        if args.resume and (output_dir / "summary_metrics.csv").exists():
            print("Resuming: existing summary found.")
        rows.append(
            run_condition(
                cases=cases,
                split_name=split_name,
                train_fraction=train_fraction,
                validation_fraction=val_fraction,
                scaler=scaler,
                seed=args.seed,
                target_per_class=args.target_per_class,
                output_dir=output_dir,
            )
        )

    combined_path = args.output_root / "voxel_leakage_summary.csv"
    write_csv(combined_path, rows)
    (args.output_root / "voxel_leakage_metadata.json").write_text(
        json.dumps(
            {
                "purpose": "diagnose whether contradictory manuscript split wording could inflate scores",
                "scientific_status": "invalid for clinical/generalization claims; leakage diagnostic only",
                "conditions": [row["configuration"] for row in rows],
                "target_per_class": args.target_per_class,
                "seed": args.seed,
                "manuscript_svm_dice": 0.56,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\nVoxel-leakage diagnostic summary (NOT clinically valid):")
    for row in sorted(rows, key=lambda item: float(item["dice_mean"]), reverse=True):
        print(
            f"{row['configuration']:>26}: Dice={float(row['dice_mean']):.3f} ± "
            f"{float(row['dice_std']):.3f}; AUC={float(row['auc_mean']):.3f}"
        )
    print(f"\nCombined results written to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
