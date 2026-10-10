#!/usr/bin/env python3
"""Build a minimal Kaggle handoff bundle for resuming ISLES'24 Fold 4.

The bundle contains only what nnU-Net needs to continue Fold 4:
- Dataset504 preprocessed data and deterministic splits;
- the trainer-level metadata from nnUNet_results;
- Fold 4 checkpoint/state files.

Folds 0-3 are intentionally excluded because they are already complete in
Google Drive. The resulting tar archive is suitable for upload as a private
Kaggle Dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        default="Dataset504_ISLES2024_NCCT",
    )
    parser.add_argument(
        "--trainer-name",
        default="nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument(
        "--archive-name",
        default="isles2024_fold4_resume.tar",
    )
    args = parser.parse_args()

    source_preprocessed = args.preprocessed_root / args.dataset_name
    source_trainer = args.results_root / args.dataset_name / args.trainer_name
    source_fold = source_trainer / f"fold_{args.fold}"

    if not source_preprocessed.is_dir():
        raise FileNotFoundError(source_preprocessed)
    if not source_fold.is_dir():
        raise FileNotFoundError(source_fold)

    checkpoint_latest = source_fold / "checkpoint_latest.pth"
    if not checkpoint_latest.is_file():
        raise FileNotFoundError(checkpoint_latest)

    bundle_root = args.output_dir / "isles2024_fold4_resume"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)

    target_preprocessed = (
        bundle_root / "nnUNet_preprocessed" / args.dataset_name
    )
    target_trainer = (
        bundle_root
        / "nnUNet_results"
        / args.dataset_name
        / args.trainer_name
    )
    target_fold = target_trainer / f"fold_{args.fold}"

    print(f"Copying preprocessed dataset: {source_preprocessed}")
    shutil.copytree(source_preprocessed, target_preprocessed)

    for name in (
        "dataset.json",
        "dataset_fingerprint.json",
        "plans.json",
    ):
        source = source_trainer / name
        if source.is_file():
            copy_file(source, target_trainer / name)

    print(f"Copying Fold {args.fold} resume state: {source_fold}")
    target_fold.mkdir(parents=True, exist_ok=True)
    for source in source_fold.iterdir():
        if not source.is_file():
            continue
        if source.name.startswith("checkpoint_") or source.name in {
            "debug.json",
            "progress.png",
        } or source.name.startswith("training_log_"):
            copy_file(source, target_fold / source.name)

    handoff = {
        "dataset": args.dataset_name,
        "trainer": args.trainer_name,
        "fold": args.fold,
        "checkpoint_latest_bytes": checkpoint_latest.stat().st_size,
        "purpose": "Resume ISLES'24 NCCT-only nnU-Net Fold 4 on Kaggle",
        "gpu_handoff": "Colab G4 -> Kaggle single T4",
        "note": (
            "GPU architecture changes can introduce small floating-point "
            "differences; configuration, fold, checkpoint, and software "
            "baseline remain unchanged."
        ),
    }
    (bundle_root / "handoff.json").write_text(
        json.dumps(handoff, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_path = args.output_dir / args.archive_name
    if archive_path.exists():
        archive_path.unlink()

    print(f"Creating archive: {archive_path}")
    with tarfile.open(archive_path, "w") as archive:
        archive.add(bundle_root, arcname=bundle_root.name)

    print(f"Archive ready: {archive_path}")
    print(f"Size: {archive_path.stat().st_size / (1024**3):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
