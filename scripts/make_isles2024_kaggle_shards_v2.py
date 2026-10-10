#!/usr/bin/env python3
"""Create a resumable sharded Kaggle handoff for ISLES'24 Fold 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_tar(output: Path, entries: list[tuple[Path, Path]]) -> None:
    output.unlink(missing_ok=True)
    with tarfile.open(output, "w") as tar:
        for source, arcname in entries:
            if not source.is_file():
                raise FileNotFoundError(source)
            tar.add(source, arcname=str(arcname), recursive=False)


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def saved_info(path: Path) -> tuple[str, int] | None:
    meta = sidecar_path(path)
    if not path.is_file() or not meta.is_file():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        checksum = str(payload["sha256"])
        size = int(payload["size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if path.stat().st_size != size:
        return None
    return checksum, size


def persist(local: Path, destination: Path) -> tuple[str, int]:
    checksum = sha256sum(local)
    size = local.stat().st_size
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)

    with local.open("rb") as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

    if partial.stat().st_size != size:
        raise RuntimeError(
            f"Drive copy size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {size}"
        )
    partial.replace(destination)
    sidecar_path(destination).write_text(
        json.dumps({"sha256": checksum, "size": size}, indent=2) + "\n",
        encoding="utf-8",
    )
    return checksum, size


def subject_ids(split_path: Path) -> list[str]:
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    ids = sorted(
        {
            str(case_id)
            for fold in splits
            for key in ("train", "val")
            for case_id in fold[key]
        }
    )
    if len(ids) != 149:
        raise RuntimeError(f"Expected 149 subjects in splits, found {len(ids)}")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--subjects-per-shard", type=int, default=5)
    parser.add_argument("--fold", type=int, default=4)
    args = parser.parse_args()

    dataset = "Dataset504_ISLES2024_NCCT"
    trainer = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    pre = args.preprocessed_root / dataset
    cfg = pre / "nnUNetPlans_3d_fullres"
    gt = pre / "gt_segmentations"
    trainer_dir = args.results_root / dataset / trainer
    fold_dir = trainer_dir / f"fold_{args.fold}"
    checkpoint = fold_dir / "checkpoint_latest.pth"

    for path in (pre, cfg, gt, trainer_dir, fold_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.subjects_per_shard < 1:
        parser.error("--subjects-per-shard must be positive")

    subjects = subject_ids(pre / "splits_final.json")
    expected: dict[str, list[Path]] = {}
    missing: list[str] = []
    for sid in subjects:
        paths = [
            cfg / f"{sid}.b2nd",
            cfg / f"{sid}_seg.b2nd",
            cfg / f"{sid}.pkl",
            gt / f"{sid}.nii.gz",
        ]
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            missing.extend(absent)
        expected[sid] = paths
    if missing:
        raise RuntimeError(f"Missing expected files: {missing[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    bundle = Path("isles2024_fold4_resume")
    pre_arc = bundle / "nnUNet_preprocessed" / dataset
    cfg_arc = pre_arc / "nnUNetPlans_3d_fullres"
    gt_arc = pre_arc / "gt_segmentations"

    batches = [
        subjects[i : i + args.subjects_per_shard]
        for i in range(0, len(subjects), args.subjects_per_shard)
    ]
    manifest_shards: list[dict[str, object]] = []

    for index, batch in enumerate(batches):
        name = f"preprocessed_{index:03d}.tar"
        destination = args.output_dir / name
        existing = saved_info(destination)
        if existing is not None:
            checksum, size = existing
            print(
                f"[{index + 1}/{len(batches)}] skip {name} "
                f"({size / (1024**2):.1f} MiB)",
                flush=True,
            )
        else:
            entries: list[tuple[Path, Path]] = []
            for sid in batch:
                data, seg, pkl, label = expected[sid]
                entries.extend(
                    [
                        (data, cfg_arc / data.name),
                        (seg, cfg_arc / seg.name),
                        (pkl, cfg_arc / pkl.name),
                        (label, gt_arc / label.name),
                    ]
                )
            local = args.scratch_dir / name
            print(
                f"[{index + 1}/{len(batches)}] build {name} "
                f"for {len(batch)} subjects",
                flush=True,
            )
            add_tar(local, entries)
            checksum, size = persist(local, destination)
            local.unlink(missing_ok=True)
            print(
                f"[{index + 1}/{len(batches)}] persisted {name} "
                f"({size / (1024**2):.1f} MiB)",
                flush=True,
            )

        manifest_shards.append(
            {
                "file": name,
                "subjects": batch,
                "sha256": checksum,
                "size": size,
            }
        )

    state_name = "state_and_metadata.tar"
    state_dest = args.output_dir / state_name
    state = saved_info(state_dest)
    if state is None:
        entries: list[tuple[Path, Path]] = []
        for name in (
            "dataset.json",
            "dataset_fingerprint.json",
            "nnUNetPlans.json",
            "splits_final.json",
            ".hybridstrokeseg_preprocess_complete",
        ):
            source = pre / name
            if source.is_file():
                entries.append((source, pre_arc / name))

        trainer_arc = bundle / "nnUNet_results" / dataset / trainer
        for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
            source = trainer_dir / name
            if source.is_file():
                entries.append((source, trainer_arc / name))

        fold_arc = trainer_arc / f"fold_{args.fold}"
        for source in sorted(fold_dir.iterdir()):
            if not source.is_file():
                continue
            if (
                source.name.startswith("checkpoint_")
                or source.name.startswith("training_log_")
                or source.name in {"debug.json", "progress.png"}
            ):
                entries.append((source, fold_arc / source.name))

        local = args.scratch_dir / state_name
        print("Building state_and_metadata.tar", flush=True)
        add_tar(local, entries)
        state_checksum, state_size = persist(local, state_dest)
        local.unlink(missing_ok=True)
    else:
        state_checksum, state_size = state
        print(
            f"skip {state_name} ({state_size / (1024**2):.1f} MiB)",
            flush=True,
        )

    manifest = {
        "dataset": dataset,
        "trainer": trainer,
        "fold": args.fold,
        "subjects": len(subjects),
        "subjects_per_shard": args.subjects_per_shard,
        "preprocessed_shards": manifest_shards,
        "state_archive": {
            "file": state_name,
            "sha256": state_checksum,
            "size": state_size,
        },
        "checkpoint_latest_bytes": checkpoint.stat().st_size,
    }
    (args.output_dir / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Handoff ready: {len(batches)} data shards + 1 state shard",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
