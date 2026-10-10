#!/usr/bin/env python3
"""Resume ISLES'24 NCCT Fold 4 from a Kaggle input bundle.

Expected input:
  <input-root>/isles2024_fold4_resume.tar

The script extracts the read-only Kaggle input to /kaggle/working/runtime,
restores nnU-Net environment variables, resumes only Fold 4 from
checkpoint_latest.pth, and leaves the completed Fold 4 result tree under
/kaggle/working/isles2024_fold4_output for notebook output persistence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
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

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    runtime = args.work_root
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)

    print(f"Extracting resume bundle: {args.archive}", flush=True)
    with tarfile.open(args.archive, "r:*") as archive:
        archive.extractall(runtime)

    bundle = runtime / "isles2024_fold4_resume"
    preprocessed_root = bundle / "nnUNet_preprocessed"
    source_results = bundle / "nnUNet_results"
    raw_root = runtime / "nnUNet_raw"
    results_root = runtime / "nnUNet_results"

    raw_root.mkdir(parents=True, exist_ok=True)
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(preprocessed_root)
    if not source_results.is_dir():
        raise FileNotFoundError(source_results)

    shutil.copytree(source_results, results_root, dirs_exist_ok=True)

    handoff_path = bundle / "handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    print(json.dumps(handoff, indent=2), flush=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(raw_root)
    env["nnUNet_preprocessed"] = str(preprocessed_root)
    env["nnUNet_results"] = str(results_root)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    checkpoint = (
        results_root
        / handoff["dataset"]
        / handoff["trainer"]
        / f"fold_{handoff['fold']}"
        / "checkpoint_latest.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print("GPU visibility:", env["CUDA_VISIBLE_DEVICES"], flush=True)
    run(
        [
            "python",
            "-c",
            (
                "import torch; "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('GPU:', torch.cuda.get_device_name(0) if "
                "torch.cuda.is_available() else 'none')"
            ),
        ],
        env=env,
    )

    print(
        "Resuming Fold 4 from the transferred checkpoint. "
        "Only one T4 is exposed to preserve single-GPU continuation semantics.",
        flush=True,
    )
    run(
        [
            "nnUNetv2_train",
            "504",
            "3d_fullres",
            "4",
            "--npz",
            "--c",
            "-device",
            "cuda",
        ],
        env=env,
    )

    fold_dir = (
        results_root
        / handoff["dataset"]
        / handoff["trainer"]
        / "fold_4"
    )
    final_checkpoint = fold_dir / "checkpoint_final.pth"
    if not final_checkpoint.is_file():
        raise RuntimeError("Fold 4 training ended without checkpoint_final.pth")

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    shutil.copytree(
        fold_dir,
        args.output_root / "fold_4",
        dirs_exist_ok=True,
    )
    for name in ("dataset.json", "dataset_fingerprint.json", "plans.json"):
        source = fold_dir.parent / name
        if source.is_file():
            shutil.copy2(source, args.output_root / name)

    (args.output_root / "handoff.json").write_text(
        json.dumps(handoff, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Fold 4 complete. Kaggle output: {args.output_root}", flush=True)

    # Remove the 3+ GiB runtime copy so notebook output stays compact.
    shutil.rmtree(runtime)
    print("Temporary preprocessed runtime data removed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
