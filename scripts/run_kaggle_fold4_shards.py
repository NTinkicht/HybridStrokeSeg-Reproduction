#!/usr/bin/env python3
"""Resume ISLES'24 NCCT Fold 4 from a sharded private Kaggle Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/kaggle/working/isles2024_runtime"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/isles2024_fold4_output"),
    )
    args = parser.parse_args()

    manifest_path = args.input_dir / "handoff_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.work_root.exists():
        shutil.rmtree(args.work_root)
    args.work_root.mkdir(parents=True)

    archives: list[tuple[str, str]] = []
    for item in manifest["preprocessed_shards"]:
        archives.append((item["file"], item["sha256"]))
    state = manifest["state_archive"]
    archives.append((state["file"], state["sha256"]))

    print(f"Verifying and extracting {len(archives)} shards.", flush=True)
    for index, (name, expected) in enumerate(archives, start=1):
        path = args.input_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256sum(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA256 mismatch for {name}: {actual} != {expected}"
            )
        print(f"[{index}/{len(archives)}] verified {name}", flush=True)
        with tarfile.open(path, "r") as archive:
            archive.extractall(args.work_root)

    bundle = args.work_root / "isles2024_fold4_resume"
    preprocessed_root = bundle / "nnUNet_preprocessed"
    results_root = bundle / "nnUNet_results"
    raw_root = args.work_root / "nnUNet_raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    dataset = manifest["dataset"]
    trainer = manifest["trainer"]
    fold = int(manifest["fold"])
    checkpoint = (
        results_root
        / dataset
        / trainer
        / f"fold_{fold}"
        / "checkpoint_latest.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw_root)
    env["nnUNet_preprocessed"] = str(preprocessed_root)
    env["nnUNet_results"] = str(results_root)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    run(
        [
            "python",
            "-c",
            (
                "import torch; "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('Visible GPUs:', torch.cuda.device_count()); "
                "print('GPU:', torch.cuda.get_device_name(0) if "
                "torch.cuda.is_available() else 'none')"
            ),
        ],
        env=env,
    )

    print(
        "Resuming Fold 4 from checkpoint_latest.pth on one visible GPU.",
        flush=True,
    )
    run(
        [
            "nnUNetv2_train",
            "504",
            "3d_fullres",
            str(fold),
            "--npz",
            "--c",
            "-device",
            "cuda",
        ],
        env=env,
    )

    fold_dir = results_root / dataset / trainer / f"fold_{fold}"
    final_checkpoint = fold_dir / "checkpoint_final.pth"
    if not final_checkpoint.is_file():
        raise RuntimeError("Fold 4 ended without checkpoint_final.pth")

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    shutil.copytree(
        fold_dir,
        args.output_root / f"fold_{fold}",
        dirs_exist_ok=True,
    )
    for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
        source = fold_dir.parent / name
        if source.is_file():
            shutil.copy2(source, args.output_root / name)
    shutil.copy2(manifest_path, args.output_root / "handoff_manifest.json")

    print(f"Fold 4 complete. Output: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
