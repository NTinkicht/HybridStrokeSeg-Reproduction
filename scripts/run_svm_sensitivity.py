#!/usr/bin/env python3
"""Screen plausible RBF-SVM scaling and hyperparameter choices.

The manuscript reports an RBF-SVM but omits feature scaling, C and gamma. This
script evaluates a small predeclared grid on the same deterministic patient split
used by the clean-room reconstruction. It is an uncertainty analysis, not a
search for settings that reproduce the manuscript's reported Dice.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/svm_sensitivity"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-per-class", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == "svm"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one SVM row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slug_gamma(value: str) -> str:
    return value.replace(".", "p")


def main() -> int:
    args = parse_args()
    runner = Path(__file__).with_name("run_reproduction.py")
    args.output_root.mkdir(parents=True, exist_ok=True)

    scalers = ("standard", "minmax", "none")
    c_values = (0.1, 1.0, 10.0)
    gamma_values = ("scale", "0.01", "0.1")
    grid = list(itertools.product(scalers, c_values, gamma_values))

    print(
        "SVM sensitivity guardrail: this is a predeclared uncertainty screen for "
        "three omitted manuscript details (scaling, C, gamma). Do not choose a "
        "configuration solely because it is numerically closest to Dice 0.56."
    )

    rows: list[dict[str, object]] = []
    for index, (scaler, c_value, gamma) in enumerate(grid, start=1):
        name = f"{scaler}_C{str(c_value).replace('.', 'p')}_g{slug_gamma(gamma)}"
        output_dir = args.output_root / name
        summary_path = output_dir / "summary_metrics.csv"
        print(f"\n[{index}/{len(grid)}] {name}")

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
                "--preprocessing",
                "paper_minimal",
                "--target-per-class",
                str(args.target_per_class),
                "--run-mode",
                "threshold_count",
                "--run-threshold",
                "20",
                "--run-window",
                "25",
                "--coordinate-mode",
                "raw",
                "--close-radius",
                "1",
                "--models",
                "svm",
                "--svm-scaler",
                scaler,
                "--svm-c",
                str(c_value),
                "--svm-gamma",
                gamma,
            ]
            subprocess.run(command, check=True)
        else:
            print("Resuming: existing summary found, skipping training.")

        summary = read_summary(summary_path)
        rows.append(
            {
                "configuration": name,
                "scaler": scaler,
                "C": c_value,
                "gamma": gamma,
                "dice_mean": float(summary["dice_mean"]),
                "dice_std": float(summary["dice_std"]),
                "precision_mean": float(summary["precision_mean"]),
                "recall_mean": float(summary["recall_mean"]),
                "auc_mean": float(summary["auc_mean"]),
            }
        )

    combined_path = args.output_root / "svm_sensitivity.csv"
    write_csv(combined_path, rows)

    metadata = {
        "purpose": "SVM reconstruction uncertainty analysis, not score matching",
        "seed": args.seed,
        "target_per_class": args.target_per_class,
        "protocol": "paper-like oracle slice selection",
        "preprocessing": "paper_minimal",
        "run_mode": "threshold_count",
        "run_threshold": 20.0,
        "run_window": 25,
        "coordinate_mode": "raw",
        "close_radius": 1,
        "scalers": list(scalers),
        "C_values": list(c_values),
        "gamma_values": list(gamma_values),
        "num_conditions": len(grid),
        "manuscript_svm_dice": 0.56,
    }
    (args.output_root / "svm_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nSVM sensitivity summary (sorted by Dice for inspection only):")
    for row in sorted(rows, key=lambda item: float(item["dice_mean"]), reverse=True):
        print(
            f"{row['configuration']:>30}: Dice={float(row['dice_mean']):.3f} ± "
            f"{float(row['dice_std']):.3f}; AUC={float(row['auc_mean']):.3f}"
        )
    print(f"\nCombined results written to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
