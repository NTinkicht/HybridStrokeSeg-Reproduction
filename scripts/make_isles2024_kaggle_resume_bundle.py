#!/usr/bin/env python3
"""Build a minimal Kaggle handoff archive for resuming ISLES'24 Fold 4.

The archive is streamed directly from the persistent nnU-Net files into a
single local tar file. It deliberately avoids first copying the 3+ GiB
preprocessed dataset into another Google Drive folder, which is slow and can
fail through the Drive FUSE mount.

The archive contains only what nnU-Net needs to continue Fold 4:
- Dataset504 preprocessed data and deterministic splits;
- trainer-level metadata from nnUNet_results;
- Fold 4 checkpoint/state files.

Folds 0-3 are intentionally excluded because they are already complete.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

BUNDLE_NAME = "isles2024_fold4_resume"


def add_tree(
    archive: tarfile.TarFile,
    source: Path,
    arcname: Path,
) -> None:
    """Add a directory tree directly to a tar archive."""
    if not source.is_dir():
        raise FileNotFoundError(source)
    archive.add(source, arcname=str(arcname), recursive=True)


def add_file(
    archive: tarfile.TarFile,
    source: Path,
    arcname: Path,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    archive.add(source, arcname=str(arcname), recursive=False)


def add_json(
    archive: tarfile.TarFile,
    payload: dict[str, object],
    arcname: Path,
) -> None:
    data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    info = tarfile.TarInfo(str(arcname))
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


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
    if not source_trainer.is_dir():
        raise FileNotFoundError(source_trainer)
    if not source_fold.is_dir():
        raise FileNotFoundError(source_fold)

    checkpoint_latest = source_fold / "checkpoint_latest.pth"
    if not checkpoint_latest.is_file():
        raise FileNotFoundError(checkpoint_latest)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / args.archive_name
    if archive_path.exists():
        archive_path.unlink()

    root = Path(BUNDLE_NAME)
    preprocessed_arc = root / "nnUNet_preprocessed" / args.dataset_name
    trainer_arc = (
        root
        / "nnUNet_results"
        / args.dataset_name
        / args.trainer_name
    )
    fold_arc = trainer_arc / f"fold_{args.fold}"

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

    print(
        "Creating tar directly from persistent source files "
        "(no intermediate Drive copy).",
        flush=True,
    )
    print(f"Preprocessed source: {source_preprocessed}", flush=True)
    print(f"Fold {args.fold} source: {source_fold}", flush=True)
    print(f"Output archive: {archive_path}", flush=True)

    with tarfile.open(archive_path, "w") as archive:
        add_tree(archive, source_preprocessed, preprocessed_arc)

        for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
            source = source_trainer / name
            if source.is_file():
                add_file(archive, source, trainer_arc / name)

        included_fold_files = 0
        for source in sorted(source_fold.iterdir()):
            if not source.is_file():
                continue
            keep = (
                source.name.startswith("checkpoint_")
                or source.name in {"debug.json", "progress.png"}
                or source.name.startswith("training_log_")
            )
            if keep:
                add_file(archive, source, fold_arc / source.name)
                included_fold_files += 1

        add_json(archive, handoff, root / "handoff.json")

    if included_fold_files == 0:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("No Fold 4 state files were added to the archive")

    print(f"Fold state files included: {included_fold_files}", flush=True)
    print(f"Archive ready: {archive_path}", flush=True)
    print(f"Size: {archive_path.stat().st_size / (1024**3):.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
