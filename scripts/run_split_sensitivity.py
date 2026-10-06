#!/usr/bin/env python3
"""Measure how strongly the unknown 19/9 patient split affects reproduction results.

This script does not search for a split that matches the manuscript. It repeats
the same leakage-safe SVM reconstruction across a predeclared sequence of
patient-level random seeds and summarizes the resulting distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_gamma(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/split_sensitivity"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--target-per-class", type=int, default=2_000)
    parser.add_argument(
        "--preprocessing",
        choices=("paper_minimal", "robust_uint8", "zscore_brain"),
        default="paper_minimal",
    )
    parser.add_argument(
        "--run-mode",
        choices=("threshold_count", "centered_run", "max_run"),
        default="threshold_count",
    )
    parser.add_argument("--run-threshold", type=float, default=20.0)
    parser.add_argument("--run-window", type=int, default=25)
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "q975": float(np.quantile(values, 0.975)),
        "max": float(values.max()),
    }


def main() -> int:
    args = parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")

    runner = Path(__file__).with_name("run_reproduction.py")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    print(
        "Split-sensitivity guardrail: every run remains a patient-level 19/9 holdout. "
        "The purpose is to quantify uncertainty from the manuscript's undisclosed case split, "
        "not to pick the seed that best matches the reported Dice."
    )

    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        output_dir = args.output_root / f"seed_{seed:04d}"
        summary_path = output_dir / "summary_metrics.csv"
        per_case_path = output_dir / "per_case_metrics.csv"
        split_path = output_dir / "split.json"

        print(f"\n[{offset + 1}/{args.num_seeds}] patient split seed {seed}")
        if not (args.resume and summary_path.exists() and per_case_path.exists() and split_path.exists()):
            command = [
                sys.executable,
                str(runner),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--protocol",
                "paper-like",
                "--preprocessing",
                args.preprocessing,
                "--target-per-class",
                str(args.target_per_class),
                "--run-mode",
                args.run_mode,
                "--run-threshold",
                str(args.run_threshold),
                "--run-window",
                str(args.run_window),
                "--close-radius",
                str(args.close_radius),
                "--models",
                "svm",
                "--svm-c",
                str(args.svm_c),
                "--svm-gamma",
                str(parse_gamma(args.svm_gamma)),
            ]
            subprocess.run(command, check=True)
        else:
            print("Resuming: existing outputs found, skipping training.")

        summary_rows = read_csv(summary_path)
        svm_rows = [row for row in summary_rows if row["model"] == "svm"]
        if len(svm_rows) != 1:
            raise RuntimeError(f"Expected exactly one SVM summary row in {summary_path}")
        summary = svm_rows[0]

        case_rows = [row for row in read_csv(per_case_path) if row["model"] == "svm"]
        case_dice = np.asarray([float(row["dice"]) for row in case_rows], dtype=float)
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "seed": seed,
                "dice_mean": float(summary["dice_mean"]),
                "dice_std": float(summary["dice_std"]),
                "precision_mean": float(summary["precision_mean"]),
                "recall_mean": float(summary["recall_mean"]),
                "auc_mean": float(summary["auc_mean"]),
                "case_dice_median": float(np.median(case_dice)),
                "zero_dice_cases": int(np.count_nonzero(case_dice == 0.0)),
                "test_ids": ";".join(str(value) for value in split_payload["test_ids"]),
            }
        )

    combined_path = args.output_root / "split_sensitivity.csv"
    write_csv(combined_path, rows)

    dice_values = np.asarray([float(row["dice_mean"]) for row in rows], dtype=float)
    distribution = summarize_distribution(dice_values)
    metadata = {
        "purpose": "patient-split uncertainty analysis, not score matching",
        "num_seeds": args.num_seeds,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.num_seeds - 1,
        "target_per_class": args.target_per_class,
        "protocol": "paper-like oracle slice selection",
        "preprocessing": args.preprocessing,
        "run_mode": args.run_mode,
        "run_threshold": args.run_threshold,
        "run_window": args.run_window,
        "close_radius": args.close_radius,
        "svm_c": args.svm_c,
        "svm_gamma": args.svm_gamma,
        "dice_distribution": distribution,
    }
    metadata_path = args.output_root / "split_sensitivity_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print("\nPatient-split Dice distribution")
    print(
        f"mean={distribution['mean']:.3f} ± {distribution['std']:.3f}; "
        f"median={distribution['median']:.3f}; "
        f"2.5%-97.5%={distribution['q025']:.3f}-{distribution['q975']:.3f}; "
        f"range={distribution['min']:.3f}-{distribution['max']:.3f}"
    )
    print(f"Detailed results: {combined_path}")
    print(f"Distribution summary: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
