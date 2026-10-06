"""Run the clean-room ISLES 2015 single-slice reproduction experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hybridstrokeseg.data import discover_siss_cases, load_volume
from hybridstrokeseg.features import FeatureConfig
from hybridstrokeseg.metrics import evaluate_binary_segmentation
from hybridstrokeseg.models import (
    MLPConfig,
    SVMConfig,
    continuous_scores,
    make_mlp,
    make_rbf_svm,
)
from hybridstrokeseg.pipeline import prepare_slice, restore_slice_prediction, select_slice_index
from hybridstrokeseg.preprocessing import PreprocessConfig
from hybridstrokeseg.sampling import balanced_binary_sample
from hybridstrokeseg.splits import make_patient_split


def parse_gamma(value: str) -> str | float:
    """Parse an SVC gamma CLI value."""
    if value in {"scale", "auto"}:
        return value
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--protocol",
        choices=("paper-like", "leakage-free-slice"),
        default="paper-like",
        help="paper-like uses the label-informed max-lesion slice; leakage-free uses mid-brain.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("paper_minimal", "robust_uint8", "zscore_brain"),
        default="paper_minimal",
    )
    parser.add_argument("--target-per-class", type=int, default=15_000)
    parser.add_argument("--run-threshold", type=float, default=20.0)
    parser.add_argument("--run-window", type=int, default=25)
    parser.add_argument(
        "--run-mode",
        choices=("threshold_count", "centered_run", "max_run"),
        default="threshold_count",
        help="Explicit reconstruction variant for the manuscript's underspecified run-length feature.",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("raw", "normalized"),
        default="raw",
        help="Sensitivity control for the two spatial-coordinate inputs.",
    )
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=("mlp", "svm"), default=("mlp", "svm"))
    parser.add_argument("--mlp-solver", choices=("lbfgs", "adam", "sgd"), default="lbfgs")
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    return parser.parse_args()


def load_flair_and_mask(case) -> tuple[np.ndarray, np.ndarray]:
    flair, _ = load_volume(case.flair)
    lesion, _ = load_volume(case.ot)
    if flair.shape != lesion.shape:
        raise ValueError(f"FLAIR/mask shape mismatch for {case.case_id}")
    return np.asarray(flair), np.asarray(lesion)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metric_names = ("dice", "precision", "recall", "auc")
    models = sorted({str(row["model"]) for row in rows})
    for model_name in models:
        model_rows = [row for row in rows if row["model"] == model_name]
        record: dict[str, object] = {"model": model_name, "cases": len(model_rows)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(finite.mean()) if finite.size else math.nan
            record[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size else math.nan
            )
        output.append(record)
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_siss_cases(args.data_root)
    if len(cases) < 2:
        raise RuntimeError("At least two complete labeled SISS cases are required")
    if len(cases) != 28:
        print(f"WARNING: expected 28 labeled SISS cases, found {len(cases)}")

    train_size = 19 if len(cases) == 28 else None
    split = make_patient_split(
        [case.case_id for case in cases],
        train_size=train_size,
        test_fraction=0.30,
        seed=args.seed,
    )
    by_id = {case.case_id: case for case in cases}
    slice_strategy = "max_lesion" if args.protocol == "paper-like" else "middle_brain"

    if slice_strategy == "max_lesion":
        print(
            "WARNING: paper-like protocol uses ground-truth masks to choose the max-lesion slice. "
            "This is an ORACLE reconstruction assumption, not a leakage-free clinical protocol."
        )

    preprocess_config = PreprocessConfig(mode=args.preprocessing)
    feature_config = FeatureConfig(
        threshold=args.run_threshold,
        run_window=args.run_window,
        run_mode=args.run_mode,
        coordinate_mode=args.coordinate_mode,
    )

    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    train_slice_indices: dict[str, int] = {}
    for case_id in split.train_ids:
        case = by_id[case_id]
        flair, lesion = load_flair_and_mask(case)
        index = select_slice_index(flair, lesion, strategy=slice_strategy)
        prepared = prepare_slice(
            flair[index],
            lesion[index],
            preprocess_config=preprocess_config,
            feature_config=feature_config,
        )
        if prepared.y is None:
            raise AssertionError("Training labels unexpectedly missing")
        train_X.append(prepared.X)
        train_y.append(prepared.y)
        train_slice_indices[case_id] = index

    X = np.concatenate(train_X, axis=0)
    y = np.concatenate(train_y, axis=0)
    sample = balanced_binary_sample(
        X,
        y,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )
    print(
        f"Balanced training sample: {sample.positive_count} lesion + "
        f"{sample.negative_count} non-lesion pixels"
    )

    models = {}
    if "mlp" in args.models:
        models["mlp"] = make_mlp(
            MLPConfig(
                hidden_layers=(100, 100, 100),
                activation="logistic",
                solver=args.mlp_solver,
                max_iter=args.mlp_max_iter,
                random_state=args.seed,
            )
        )
    if "svm" in args.models:
        models["svm"] = make_rbf_svm(
            SVMConfig(C=args.svm_c, gamma=parse_gamma(args.svm_gamma))
        )

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(sample.X, sample.y)

    rows: list[dict[str, object]] = []
    test_slice_indices: dict[str, int] = {}
    for case_id in split.test_ids:
        case = by_id[case_id]
        flair, lesion = load_flair_and_mask(case)
        index = select_slice_index(flair, lesion, strategy=slice_strategy)
        test_slice_indices[case_id] = index
        prepared = prepare_slice(
            flair[index],
            lesion[index],
            preprocess_config=preprocess_config,
            feature_config=feature_config,
        )
        if prepared.y is None:
            raise AssertionError("Test labels unexpectedly missing")

        for name, model in models.items():
            raw_prediction = model.predict(prepared.X)
            scores = continuous_scores(model, prepared.X)
            mask_2d = restore_slice_prediction(
                raw_prediction,
                prepared.brain_mask,
                close_radius=args.close_radius,
            )
            postprocessed = mask_2d[prepared.brain_mask]
            metrics = evaluate_binary_segmentation(
                prepared.y,
                postprocessed,
                scores=scores,
            )
            row: dict[str, object] = {
                "case_id": case_id,
                "slice_index": index,
                "model": name,
                **metrics.as_dict(),
            }
            rows.append(row)
            print(
                f"{name:>3} {case_id}: Dice={metrics.dice:.3f} "
                f"P={metrics.precision:.3f} R={metrics.recall:.3f} AUC={metrics.auc:.3f}"
            )

    summaries = summarize(rows)
    write_csv(args.output_dir / "per_case_metrics.csv", rows)
    write_csv(args.output_dir / "summary_metrics.csv", summaries)

    split_payload = {
        "seed": split.seed,
        "train_ids": list(split.train_ids),
        "test_ids": list(split.test_ids),
        "train_slice_indices": train_slice_indices,
        "test_slice_indices": test_slice_indices,
    }
    (args.output_dir / "split.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata = {
        "dataset": "ISLES 2015 SISS",
        "protocol": args.protocol,
        "slice_strategy": slice_strategy,
        "oracle_slice_selection": slice_strategy == "max_lesion",
        "preprocessing": asdict(preprocess_config),
        "feature_config": asdict(feature_config),
        "target_per_class_requested": args.target_per_class,
        "target_per_class_used": sample.positive_count,
        "close_radius": args.close_radius,
        "models": list(models),
        "mlp": {
            "reported_optimizer": "scaled conjugate gradient",
            "implemented_optimizer": args.mlp_solver,
            "optimizer_exact_match": False,
            "hidden_layers": [100, 100, 100],
            "activation": "logistic",
            "max_iter": args.mlp_max_iter,
        },
        "svm": {
            "kernel": "rbf",
            "C": args.svm_c,
            "gamma": args.svm_gamma,
            "feature_scaling": "StandardScaler",
        },
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nSummary")
    for record in summaries:
        print(
            f"{record['model']}: Dice {record['dice_mean']:.3f} ± {record['dice_std']:.3f}; "
            f"Precision {record['precision_mean']:.3f}; Recall {record['recall_mean']:.3f}"
        )
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
