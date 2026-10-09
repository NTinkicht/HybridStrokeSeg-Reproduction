#!/usr/bin/env python3
"""Plan, preprocess, and train the ISLES'24 NCCT nnU-Net v2 baseline.

This orchestrator is designed for persistent Google Drive storage and short-
lived Colab sessions. Re-running it:
- skips completed preprocessing;
- restores the deterministic patient folds;
- skips folds with checkpoint_final.pth;
- resumes interrupted folds from nnU-Net's latest checkpoint via --c.

The default experiment is Dataset504 / 3d_fullres / five folds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def dataset_dir(root: Path, dataset_id: int) -> Path:
    matches = sorted(root.glob(f"Dataset{dataset_id:03d}_*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Dataset{dataset_id:03d}_* below {root}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def fold_complete(results_root: Path, dataset_name: str, fold: int) -> bool:
    dataset_results = results_root / dataset_name
    if not dataset_results.is_dir():
        return False
    matches = list(dataset_results.glob(f"*/fold_{fold}/checkpoint_final.pth"))
    return bool(matches)


def write_status(
    path: Path,
    *,
    dataset_name: str,
    folds: list[int],
    results_root: Path,
    preprocessing_complete: bool,
) -> None:
    payload = {
        "dataset": dataset_name,
        "configuration": "3d_fullres",
        "preprocessing_complete": preprocessing_complete,
        "folds": {
            str(fold): {
                "checkpoint_final": fold_complete(
                    results_root,
                    dataset_name,
                    fold,
                )
            }
            for fold in folds
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, default=504)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Require existing preprocessing and proceed directly to training.",
    )
    args = parser.parse_args()

    for fold in args.folds:
        if fold not in range(5):
            parser.error("--folds values must be between 0 and 4")

    required_env = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required nnU-Net environment variables: " + ", ".join(missing)
        )

    raw_root = Path(os.environ["nnUNet_raw"])
    preprocessed_root = Path(os.environ["nnUNet_preprocessed"])
    results_root = Path(os.environ["nnUNet_results"])
    preprocessed_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    raw_dataset = dataset_dir(raw_root, args.dataset_id)
    dataset_name = raw_dataset.name
    preprocessed_dataset = preprocessed_root / dataset_name
    preprocess_marker = preprocessed_dataset / ".hybridstrokeseg_preprocess_complete"
    status_json = args.status_json or results_root / "isles2024_ncct_cv_status.json"

    env = os.environ.copy()

    if not preprocess_marker.exists():
        if args.skip_preprocessing:
            raise RuntimeError(
                f"Preprocessing marker is missing: {preprocess_marker}"
            )

        print("\n=== nnU-Net planning and preprocessing ===", flush=True)
        run(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(args.dataset_id),
                "--verify_dataset_integrity",
                "-c",
                args.configuration,
            ],
            env=env,
        )

        if not preprocessed_dataset.is_dir():
            raise RuntimeError(
                f"nnU-Net preprocessing did not create {preprocessed_dataset}"
            )

        split_source = raw_dataset / "splits_final.json"
        if not split_source.is_file():
            raise FileNotFoundError(split_source)
        shutil.copy2(split_source, preprocessed_dataset / "splits_final.json")
        preprocess_marker.write_text(
            "Planning/preprocessing completed and deterministic project folds installed.\n",
            encoding="utf-8",
        )
        print(f"Preprocessing marker written: {preprocess_marker}", flush=True)
    else:
        print(
            f"Preprocessing already complete: {preprocess_marker}",
            flush=True,
        )
        split_source = raw_dataset / "splits_final.json"
        if split_source.is_file():
            shutil.copy2(split_source, preprocessed_dataset / "splits_final.json")

    write_status(
        status_json,
        dataset_name=dataset_name,
        folds=args.folds,
        results_root=results_root,
        preprocessing_complete=True,
    )

    print("\n=== Five-fold nnU-Net training ===", flush=True)
    for fold in args.folds:
        if fold_complete(results_root, dataset_name, fold):
            print(f"Fold {fold}: checkpoint_final.pth already exists; skipping.", flush=True)
            continue

        print(
            f"Fold {fold}: starting or resuming training from the latest checkpoint.",
            flush=True,
        )
        run(
            [
                "nnUNetv2_train",
                str(args.dataset_id),
                args.configuration,
                str(fold),
                "--npz",
                "--c",
                "-device",
                "cuda",
            ],
            env=env,
        )
        write_status(
            status_json,
            dataset_name=dataset_name,
            folds=args.folds,
            results_root=results_root,
            preprocessing_complete=True,
        )

    write_status(
        status_json,
        dataset_name=dataset_name,
        folds=args.folds,
        results_root=results_root,
        preprocessing_complete=True,
    )
    print(f"\nCross-validation status: {status_json}", flush=True)
    print("All requested folds are complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
