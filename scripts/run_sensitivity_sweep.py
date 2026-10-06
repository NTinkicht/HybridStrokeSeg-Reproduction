#!/usr/bin/env python3
"""Run a targeted sensitivity sweep for underspecified ISLES 2015 reconstruction choices.

This sweep is diagnostic. It is designed to measure how much the unavailable
implementation details affect results, not to tune parameters until the paper's
reported Dice score is matched.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = (
    {
        "name": "minimal_count_t20",
        "preprocessing": "paper_minimal",
        "run_mode": "threshold_count",
        "threshold": 20,
    },
    {
        "name": "minimal_centered_t20",
        "preprocessing": "paper_minimal",
        "run_mode": "centered_run",
        "threshold": 20,
    },
    {
        "name": "minimal_max_t20",
        "preprocessing": "paper_minimal",
        "run_mode": "max_run",
        "threshold": 20,
    },
    {
        "name": "uint8_count_t10",
        "preprocessing": "robust_uint8",
        "run_mode": "threshold_count",
        "threshold": 10,
    },
    {
        "name": "uint8_count_t20",
        "preprocessing": "robust_uint8",
        "run_mode": "threshold_count",
        "threshold": 20,
    },
    {
        "name": "uint8_count_t40",
        "preprocessing": "robust_uint8",
        "run_mode": "threshold_count",
        "threshold": 40,
    },
    {
        "name": "uint8_centered_t10",
        "preprocessing": "robust_uint8",
        "run_mode": "centered_run",
        "threshold": 10,
    },
    {
        "name": "uint8_centered_t20",
        "preprocessing": "robust_uint8",
        "run_mode": "centered_run",
        "threshold": 20,
    },
    {
        "name": "uint8_centered_t40",
        "preprocessing": "robust_uint8",
        "run_mode": "centered_run",
        "threshold": 40,
    },
    {
        "name": "uint8_max_t10",
        "preprocessing": "robust_uint8",
        "run_mode": "max_run",
        "threshold": 10,
    },
    {
        "name": "uint8_max_t20",
        "preprocessing": "robust_uint8",
        "run_mode": "max_run",
        "threshold": 20,
    },
    {
        "name": "uint8_max_t40",
        "preprocessing": "robust_uint8",
        "run_mode": "max_run",
        "threshold": 40,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitivity_quick"))
    parser.add_argument("--target-per-class", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("mlp", "svm"),
        default=("svm",),
        help="SVM-only is the default fast screen; add MLP for a slower diagnostic sweep.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    runner = Path(__file__).with_name("run_reproduction.py")
    combined: list[dict[str, object]] = []

    print(
        "Sensitivity guardrail: these experiments quantify reconstruction uncertainty; "
        "do not select a configuration solely because it approaches the manuscript's reported Dice."
    )

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        name = str(experiment["name"])
        output_dir = args.output_root / name
        summary_path = output_dir / "summary_metrics.csv"
        print(f"\n[{index}/{len(EXPERIMENTS)}] {name}")

        if not (args.resume and summary_path.exists()):
            command = [
                sys.executable,
                str(runner),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(args.seed),
                "--protocol",
                "paper-like",
                "--target-per-class",
                str(args.target_per_class),
                "--preprocessing",
                str(experiment["preprocessing"]),
                "--run-mode",
                str(experiment["run_mode"]),
                "--run-threshold",
                str(experiment["threshold"]),
                "--models",
                *args.models,
            ]
            subprocess.run(command, check=True)
        else:
            print("Resuming: existing summary found, skipping training.")

        for row in read_summary(summary_path):
            combined.append(
                {
                    "experiment": name,
                    "preprocessing": experiment["preprocessing"],
                    "run_mode": experiment["run_mode"],
                    "run_threshold": experiment["threshold"],
                    "target_per_class": args.target_per_class,
                    "seed": args.seed,
                    **row,
                }
            )

    combined.sort(key=lambda row: (-float(row["dice_mean"]), str(row["experiment"])))
    aggregate_path = args.output_root / "sweep_summary.csv"
    write_summary(aggregate_path, combined)

    print("\nSensitivity sweep summary (sorted by Dice for inspection only):")
    for row in combined:
        print(
            f"{row['experiment']:>22} {row['model']:>3}: "
            f"Dice={float(row['dice_mean']):.3f} ± {float(row['dice_std']):.3f}; "
            f"AUC={float(row['auc_mean']):.3f}"
        )
    print(f"\nCombined results written to {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
