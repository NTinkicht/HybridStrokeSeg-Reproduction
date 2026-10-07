#!/usr/bin/env python3
"""Diagnose the manuscript's underspecified histogram-specification step.

This is a patient-level 19/9 sensitivity experiment. It keeps the current
paper-like oracle slice selection and compares no histogram specification with
two deterministic classical histogram-matching reference choices. Reference
images are selected from training patients only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from hybridstrokeseg.data import discover_siss_cases, load_volume
from hybridstrokeseg.diagnostics import histogram_match_values
from hybridstrokeseg.features import FeatureConfig, extract_nine_features
from hybridstrokeseg.metrics import evaluate_binary_segmentation
from hybridstrokeseg.models import SVMConfig, continuous_scores, make_rbf_svm
from hybridstrokeseg.pipeline import restore_slice_prediction, select_slice_index
from hybridstrokeseg.preprocessing import PreprocessConfig, preprocess_flair
from hybridstrokeseg.sampling import balanced_binary_sample
from hybridstrokeseg.splits import make_patient_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/histogram_specification")
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-per-class", type=int, default=15_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in ("dice", "precision", "recall", "auc"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        output[f"{metric}_mean"] = float(finite.mean()) if finite.size else math.nan
        output[f"{metric}_std"] = (
            float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size else math.nan
        )
    return output


def load_selected_slices(cases) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    flair_slices: dict[str, np.ndarray] = {}
    lesion_slices: dict[str, np.ndarray] = {}
    selected: dict[str, int] = {}
    for case in cases:
        flair, _ = load_volume(case.flair)
        lesion, _ = load_volume(case.ot)
        flair = np.asarray(flair)
        lesion = np.asarray(lesion)
        index = select_slice_index(flair, lesion, strategy="max_lesion")
        flair_slices[case.case_id] = np.asarray(flair[index], dtype=np.float32)
        lesion_slices[case.case_id] = np.asarray(lesion[index])
        selected[case.case_id] = index
    return flair_slices, lesion_slices, selected


def preprocess_selected_slices(
    flair_slices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    images: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    cfg = PreprocessConfig(mode="paper_minimal")
    for case_id, image in flair_slices.items():
        processed, brain = preprocess_flair(image, cfg)
        images[case_id] = processed
        masks[case_id] = brain
    return images, masks


def choose_reference_ids(
    train_ids: tuple[str, ...],
    processed: dict[str, np.ndarray],
    brains: dict[str, np.ndarray],
) -> dict[str, str]:
    first_id = train_ids[0]
    medians = {
        case_id: float(np.median(processed[case_id][brains[case_id]]))
        for case_id in train_ids
    }
    cohort_median = float(np.median(list(medians.values())))
    median_id = min(
        train_ids,
        key=lambda case_id: (abs(medians[case_id] - cohort_median), case_id),
    )
    return {"first_train_ref": first_id, "median_train_ref": median_id}


def transformed_image(
    case_id: str,
    mode: str,
    *,
    processed: dict[str, np.ndarray],
    brains: dict[str, np.ndarray],
    reference_ids: dict[str, str],
) -> np.ndarray:
    source = processed[case_id]
    if mode == "none":
        return source
    reference_id = reference_ids[mode]
    reference_values = processed[reference_id][brains[reference_id]]
    output = source.copy()
    output[brains[case_id]] = histogram_match_values(
        source[brains[case_id]],
        reference_values,
    )
    return output


def run_condition(
    *,
    condition: str,
    hist_mode: str,
    scaler: str,
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
    processed: dict[str, np.ndarray],
    brains: dict[str, np.ndarray],
    lesions: dict[str, np.ndarray],
    reference_ids: dict[str, str],
    selected_slices: dict[str, int],
    target_per_class: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    summary_path = output_dir / "summary_metrics.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        return {
            key: float(value)
            if key not in {"configuration", "histogram_mode", "scaler", "reference_case"}
            else value
            for key, value in row.items()
        }

    feature_config = FeatureConfig(
        threshold=20.0,
        run_window=25,
        run_mode="threshold_count",
        coordinate_mode="raw",
    )

    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    for case_id in train_ids:
        image = transformed_image(
            case_id,
            hist_mode,
            processed=processed,
            brains=brains,
            reference_ids=reference_ids,
        )
        features = extract_nine_features(image, feature_config)
        brain = brains[case_id]
        train_X.append(features[brain])
        train_y.append((lesions[case_id][brain] > 0).astype(np.uint8))

    sample = balanced_binary_sample(
        np.concatenate(train_X, axis=0),
        np.concatenate(train_y, axis=0),
        target_per_class=target_per_class,
        seed=seed,
    )
    print(
        f"{condition}: balanced train sample {sample.positive_count} lesion + "
        f"{sample.negative_count} non-lesion pixels"
    )

    model = make_rbf_svm(SVMConfig(C=1.0, gamma="scale", scaler=scaler))
    model.fit(sample.X, sample.y)

    case_rows: list[dict[str, object]] = []
    for case_id in test_ids:
        image = transformed_image(
            case_id,
            hist_mode,
            processed=processed,
            brains=brains,
            reference_ids=reference_ids,
        )
        brain = brains[case_id]
        features = extract_nine_features(image, feature_config)
        X_test = features[brain]
        y_test = (lesions[case_id][brain] > 0).astype(np.uint8)

        raw_prediction = model.predict(X_test)
        scores = continuous_scores(model, X_test)
        restored = restore_slice_prediction(raw_prediction, brain, close_radius=1)
        metrics = evaluate_binary_segmentation(
            y_test,
            restored[brain],
            scores=scores,
        )
        case_rows.append(
            {
                "case_id": case_id,
                "slice_index": selected_slices[case_id],
                "model": "svm",
                **metrics.as_dict(),
            }
        )
        print(
            f"{condition} {case_id}: Dice={metrics.dice:.3f} "
            f"P={metrics.precision:.3f} R={metrics.recall:.3f} AUC={metrics.auc:.3f}"
        )

    reference_case = "" if hist_mode == "none" else reference_ids[hist_mode]
    summary = {
        "configuration": condition,
        "histogram_mode": hist_mode,
        "scaler": scaler,
        "reference_case": reference_case,
        "cases": len(case_rows),
        **summarize(case_rows),
    }
    write_csv(output_dir / "per_case_metrics.csv", case_rows)
    write_csv(output_dir / "summary_metrics.csv", [summary])
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "purpose": "histogram-specification reconstruction uncertainty analysis",
                "protocol": "patient-level 19/9 paper-like oracle slice selection",
                "histogram_mode": hist_mode,
                "reference_case": reference_case or None,
                "reference_selected_from_training_only": True,
                "histogram_algorithm": "empirical-CDF intensity matching",
                "pre_histogram_preprocessing": "paper_minimal median filter",
                "target_per_class": sample.positive_count,
                "svm": {"C": 1.0, "gamma": "scale", "scaler": scaler},
                "morphology_close_radius": 1,
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

    split = make_patient_split(
        [case.case_id for case in cases],
        train_size=19,
        test_fraction=0.30,
        seed=args.seed,
    )
    flair_slices, lesions, selected_slices = load_selected_slices(cases)
    processed, brains = preprocess_selected_slices(flair_slices)
    reference_ids = choose_reference_ids(split.train_ids, processed, brains)

    print("Histogram specification references selected from training patients only:")
    print(f"  first_train_ref: {reference_ids['first_train_ref']}")
    print(f"  median_train_ref: {reference_ids['median_train_ref']}")

    conditions = tuple(
        (hist_mode, scaler)
        for hist_mode in ("none", "first_train_ref", "median_train_ref")
        for scaler in ("standard", "none")
    )

    rows: list[dict[str, object]] = []
    for index, (hist_mode, scaler) in enumerate(conditions, start=1):
        condition = f"{hist_mode}_{scaler}"
        output_dir = args.output_root / condition
        print(f"\n[{index}/{len(conditions)}] {condition}")
        if args.resume and (output_dir / "summary_metrics.csv").exists():
            print("Resuming: existing summary found.")
        rows.append(
            run_condition(
                condition=condition,
                hist_mode=hist_mode,
                scaler=scaler,
                train_ids=split.train_ids,
                test_ids=split.test_ids,
                processed=processed,
                brains=brains,
                lesions=lesions,
                reference_ids=reference_ids,
                selected_slices=selected_slices,
                target_per_class=args.target_per_class,
                seed=args.seed,
                output_dir=output_dir,
            )
        )

    combined_path = args.output_root / "histogram_specification_summary.csv"
    write_csv(combined_path, rows)
    (args.output_root / "histogram_specification_metadata.json").write_text(
        json.dumps(
            {
                "purpose": "test plausible classical histogram specification interpretations",
                "scientific_status": "sensitivity analysis; original reference image is unknown",
                "conditions": [row["configuration"] for row in rows],
                "reference_ids": reference_ids,
                "train_ids": list(split.train_ids),
                "test_ids": list(split.test_ids),
                "target_per_class": args.target_per_class,
                "seed": args.seed,
                "manuscript_svm_dice": 0.56,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\nHistogram specification summary (sorted by Dice for inspection only):")
    for row in sorted(rows, key=lambda item: float(item["dice_mean"]), reverse=True):
        print(
            f"{row['configuration']:>30}: Dice={float(row['dice_mean']):.3f} ± "
            f"{float(row['dice_std']):.3f}; AUC={float(row['auc_mean']):.3f}"
        )
    print(f"\nCombined results written to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
