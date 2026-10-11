#!/usr/bin/env python3
"""Plan, preprocess, and train the ISLES'24 NCCT nnU-Net v2 baseline.

This orchestrator is designed for persistent Google Drive storage and short-
lived Colab sessions. Re-running it:
- skips completed preprocessing;
- restores the deterministic patient folds;
- skips folds with checkpoint_final.pth;
- resumes interrupted folds from nnU-Net's latest checkpoint via --c;
- can cache preprocessed tensors on Colab's local SSD so dataloader workers do
  not depend on Google Drive FUSE during training.

The default experiment is Dataset504 / 3d_fullres / five folds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
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


def case_ids_from_splits(split_path: Path) -> list[str]:
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    case_ids: set[str] = set()
    for split in splits:
        case_ids.update(split.get("train", []))
        case_ids.update(split.get("val", []))
    if not case_ids:
        raise RuntimeError(f"No case IDs found in split file: {split_path}")
    return sorted(case_ids)


def configuration_dir(dataset_path: Path, configuration: str) -> Path:
    expected = dataset_path / f"nnUNetPlans_{configuration}"
    if expected.is_dir():
        return expected

    matches = sorted(
        path
        for path in dataset_path.iterdir()
        if path.is_dir() and path.name.endswith(f"_{configuration}")
    ) if dataset_path.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    return expected


def missing_preprocessed_files(
    dataset_path: Path,
    configuration: str,
) -> list[Path]:
    split_path = dataset_path / "splits_final.json"
    if not split_path.is_file():
        return [split_path]

    config_path = configuration_dir(dataset_path, configuration)
    if not config_path.is_dir():
        return [config_path]

    missing: list[Path] = []
    for case_id in case_ids_from_splits(split_path):
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            path = config_path / f"{case_id}{suffix}"
            try:
                if not path.is_file() or path.stat().st_size <= 0:
                    missing.append(path)
            except OSError:
                missing.append(path)
    return missing


def cache_preprocessed_dataset(
    source_dataset: Path,
    cache_root: Path,
    configuration: str,
    *,
    retries: int,
) -> Path:
    """Copy persistent preprocessing to local storage and verify every case."""
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dataset = cache_root / source_dataset.name
    retries = max(1, retries)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        print(
            f"Local preprocessing cache: copy attempt {attempt}/{retries} "
            f"from {source_dataset} to {cache_dataset}",
            flush=True,
        )
        try:
            shutil.copytree(
                source_dataset,
                cache_dataset,
                dirs_exist_ok=True,
                copy_function=shutil.copy2,
            )
            last_error = None
        except (OSError, shutil.Error) as exc:
            last_error = exc
            print(f"Local cache copy warning: {exc}", flush=True)

        missing = missing_preprocessed_files(cache_dataset, configuration)
        if not missing:
            print(
                "Local preprocessing cache verified for every case; "
                "training will read tensors from local SSD.",
                flush=True,
            )
            return cache_dataset

        preview = ", ".join(str(path.name) for path in missing[:8])
        print(
            f"Local cache verification found {len(missing)} missing/empty "
            f"preprocessed files: {preview}",
            flush=True,
        )
        if attempt < retries:
            time.sleep(3)

    detail = f" Last copy error: {last_error}" if last_error else ""
    raise RuntimeError(
        f"Could not build a complete local preprocessing cache after "
        f"{retries} attempt(s).{detail}"
    )


def preprocess_into_local_cache(
    *,
    raw_dataset: Path,
    cache_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    env: dict[str, str],
) -> Path:
    """Recreate preprocessing locally when the persistent Drive copy is incomplete."""
    cache_root.mkdir(parents=True, exist_ok=True)
    local_dataset = cache_root / dataset_name
    if local_dataset.exists():
        shutil.rmtree(local_dataset)

    local_env = env.copy()
    local_env["nnUNet_preprocessed"] = str(cache_root)

    print(
        "Persistent preprocessing could not be cached reliably. "
        "Rebuilding preprocessing on local SSD from the staged raw dataset.",
        flush=True,
    )
    run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(dataset_id),
            "--verify_dataset_integrity",
            "-c",
            configuration,
        ],
        env=local_env,
    )

    split_source = raw_dataset / "splits_final.json"
    if not split_source.is_file():
        raise FileNotFoundError(split_source)
    shutil.copy2(split_source, local_dataset / "splits_final.json")

    missing = missing_preprocessed_files(local_dataset, configuration)
    if missing:
        preview = ", ".join(str(path) for path in missing[:8])
        raise RuntimeError(
            f"Local preprocessing rebuild is incomplete: {len(missing)} "
            f"missing/empty files. First entries: {preview}"
        )

    print(
        "Local preprocessing rebuild verified; training can resume safely.",
        flush=True,
    )
    return local_dataset


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
    parser.add_argument(
        "--local-preprocessed-cache",
        type=Path,
        default=None,
        help=(
            "Optional local cache root for preprocessed tensors. Recommended "
            "for Colab so nnU-Net dataloader workers do not mmap Google Drive."
        ),
    )
    parser.add_argument(
        "--cache-copy-retries",
        type=int,
        default=3,
        help="Attempts to copy and verify the persistent preprocessing cache.",
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
    env["PYTHONUNBUFFERED"] = "1"

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

    training_env = env.copy()
    if args.local_preprocessed_cache is not None:
        try:
            cache_preprocessed_dataset(
                preprocessed_dataset,
                args.local_preprocessed_cache,
                args.configuration,
                retries=args.cache_copy_retries,
            )
        except RuntimeError as exc:
            print(f"Persistent-cache copy failed: {exc}", flush=True)
            preprocess_into_local_cache(
                raw_dataset=raw_dataset,
                cache_root=args.local_preprocessed_cache,
                dataset_id=args.dataset_id,
                dataset_name=dataset_name,
                configuration=args.configuration,
                env=env,
            )

        training_env["nnUNet_preprocessed"] = str(args.local_preprocessed_cache)
        print(
            f"Training nnUNet_preprocessed: "
            f"{training_env['nnUNet_preprocessed']}",
            flush=True,
        )

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
            env=training_env,
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
