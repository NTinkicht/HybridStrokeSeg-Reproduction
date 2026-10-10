#!/usr/bin/env python3
"""Create resumable sharded Kaggle handoff archives for ISLES'24 Fold 4.

Why shards:
- reading 3+ GiB from a Google Drive FUSE mount into one tar proved fragile;
- this script builds small subject-group tar files locally, then copies each
  completed shard back to Drive;
- reruns skip shards that already have a matching SHA256 sidecar.

The resulting flat folder can be uploaded directly as a private Kaggle Dataset
with `kaggle datasets create -r skip`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

CASE_RE = re.compile(r"^(sub_stroke\d+)(?:_seg)?\.(?:b2nd|pkl)$")


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, source: Path, arcname: Path) -> None:
    archive.add(source, arcname=str(arcname), recursive=False)


def build_local_tar(
    output: Path,
    files: list[tuple[Path, Path]],
) -> None:
    if output.exists():
        output.unlink()
    with tarfile.open(output, "w") as archive:
        for source, arcname in files:
            if not source.is_file():
                raise FileNotFoundError(source)
            add_file(archive, source, arcname)

    with tarfile.open(output, "r") as archive:
        names = set(archive.getnames())
    expected = {str(arcname) for _, arcname in files}
    missing = sorted(expected - names)
    if missing:
        raise RuntimeError(f"Tar verification failed; missing entries: {missing[:10]}")


def persist_shard(
    local_tar: Path,
    destination: Path,
) -> None:
    checksum = sha256sum(local_tar)
    sha_path = destination.with_suffix(destination.suffix + ".sha256")

    if destination.is_file() and sha_path.is_file():
        saved = sha_path.read_text(encoding="utf-8").strip()
        if saved == checksum and destination.stat().st_size == local_tar.stat().st_size:
            print(f"Shard already persisted: {destination.name}", flush=True)
            return

    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    shutil.copy2(local_tar, partial)
    partial.replace(destination)
    sha_path.write_text(checksum + "\n", encoding="utf-8")
    print(
        f"Persisted {destination.name}: "
        f"{destination.stat().st_size / (1024**2):.1f} MiB",
        flush=True,
    )


def existing_shard_valid(destination: Path) -> bool:
    sha_path = destination.with_suffix(destination.suffix + ".sha256")
    if not destination.is_file() or not sha_path.is_file():
        return False
    expected = sha_path.read_text(encoding="utf-8").strip()
    return sha256sum(destination) == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        default="Dataset504_ISLES2024_NCCT",
    )
    parser.add_argument(
        "--trainer-name",
        default="nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--subjects-per-shard", type=int, default=10)
    args = parser.parse_args()

    source_preprocessed = args.preprocessed_root / args.dataset_name
    source_trainer = args.results_root / args.dataset_name / args.trainer_name
    source_fold = source_trainer / f"fold_{args.fold}"
    checkpoint_latest = source_fold / "checkpoint_latest.pth"

    if not source_preprocessed.is_dir():
        raise FileNotFoundError(source_preprocessed)
    if not source_fold.is_dir():
        raise FileNotFoundError(source_fold)
    if not checkpoint_latest.is_file():
        raise FileNotFoundError(checkpoint_latest)
    if args.subjects_per_shard < 1:
        parser.error("--subjects-per-shard must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    case_files: dict[str, list[Path]] = {}
    for path in source_preprocessed.iterdir():
        if not path.is_file():
            continue
        match = CASE_RE.match(path.name)
        if match:
            case_files.setdefault(match.group(1), []).append(path)

    subjects = sorted(case_files)
    if len(subjects) != 149:
        raise RuntimeError(
            f"Expected 149 preprocessed subjects, found {len(subjects)}"
        )

    gt_dir = source_preprocessed / "gt_segmentations"
    if not gt_dir.is_dir():
        raise FileNotFoundError(gt_dir)

    bundle_root = Path("isles2024_fold4_resume")
    pre_arc = bundle_root / "nnUNet_preprocessed" / args.dataset_name

    shard_manifest: list[dict[str, object]] = []

    batches = [
        subjects[index : index + args.subjects_per_shard]
        for index in range(0, len(subjects), args.subjects_per_shard)
    ]

    for shard_index, batch in enumerate(batches):
        shard_name = f"preprocessed_{shard_index:03d}.tar"
        destination = args.output_dir / shard_name

        if existing_shard_valid(destination):
            print(f"[{shard_index + 1}/{len(batches)}] skip {shard_name}", flush=True)
            shard_manifest.append(
                {
                    "file": shard_name,
                    "subjects": batch,
                    "sha256": destination.with_suffix(
                        destination.suffix + ".sha256"
                    ).read_text(encoding="utf-8").strip(),
                }
            )
            continue

        entries: list[tuple[Path, Path]] = []
        for subject in batch:
            for source in sorted(case_files[subject]):
                entries.append((source, pre_arc / source.name))
            gt = gt_dir / f"{subject}.nii.gz"
            entries.append((gt, pre_arc / "gt_segmentations" / gt.name))

        local = args.scratch_dir / shard_name
        print(
            f"[{shard_index + 1}/{len(batches)}] building {shard_name} "
            f"for {len(batch)} subjects",
            flush=True,
        )
        build_local_tar(local, entries)
        persist_shard(local, destination)
        checksum = sha256sum(local)
        shard_manifest.append(
            {"file": shard_name, "subjects": batch, "sha256": checksum}
        )
        local.unlink(missing_ok=True)

    metadata_entries: list[tuple[Path, Path]] = []
    for name in (
        "dataset.json",
        "dataset_fingerprint.json",
        "nnUNetPlans.json",
        "splits_final.json",
    ):
        source = source_preprocessed / name
        if source.is_file():
            metadata_entries.append((source, pre_arc / name))

    trainer_arc = (
        bundle_root / "nnUNet_results" / args.dataset_name / args.trainer_name
    )
    for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
        source = source_trainer / name
        if source.is_file():
            metadata_entries.append((source, trainer_arc / name))

    fold_arc = trainer_arc / f"fold_{args.fold}"
    for source in sorted(source_fold.iterdir()):
        if not source.is_file():
            continue
        keep = (
            source.name.startswith("checkpoint_")
            or source.name in {"debug.json", "progress.png"}
            or source.name.startswith("training_log_")
        )
        if keep:
            metadata_entries.append((source, fold_arc / source.name))

    metadata_name = "state_and_metadata.tar"
    metadata_destination = args.output_dir / metadata_name
    if not existing_shard_valid(metadata_destination):
        local = args.scratch_dir / metadata_name
        print("Building state_and_metadata.tar", flush=True)
        build_local_tar(local, metadata_entries)
        persist_shard(local, metadata_destination)
        local.unlink(missing_ok=True)

    state_checksum = metadata_destination.with_suffix(
        metadata_destination.suffix + ".sha256"
    ).read_text(encoding="utf-8").strip()

    manifest = {
        "dataset": args.dataset_name,
        "trainer": args.trainer_name,
        "fold": args.fold,
        "subjects": len(subjects),
        "subjects_per_shard": args.subjects_per_shard,
        "preprocessed_shards": shard_manifest,
        "state_archive": {
            "file": metadata_name,
            "sha256": state_checksum,
        },
        "checkpoint_latest_bytes": checkpoint_latest.stat().st_size,
        "gpu_handoff": "Colab G4 -> Kaggle single T4",
    }
    (args.output_dir / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Sharded handoff ready: {args.output_dir} "
        f"({len(batches)} preprocessed shards + 1 state shard)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
