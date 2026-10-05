#!/usr/bin/env python3
"""Stage geometry-compatible ISLES 2022 data for nnU-Net v2.

This command is intentionally conservative: it never registers or resamples a
modality. If any requested channel or label does not already match the DWI grid,
the command stops and tells you to resolve registration first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hybridstrokeseg.data import discover_isles2022_cases
from hybridstrokeseg.nnunet import stage_isles2022_nnunet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Extracted ISLES 2022 root containing rawdata/ and derivatives/.",
    )
    parser.add_argument(
        "nnunet_raw",
        type=Path,
        help="Target nnUNet_raw directory.",
    )
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="ISLES2022")
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=("dwi", "adc", "flair"),
        default=("dwi", "adc", "flair"),
        help="Input channel order. Default: dwi adc flair.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Materialize images by copying or symlinking them.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    cases = discover_isles2022_cases(args.dataset_root)
    print(f"Discovered {len(cases)} labeled ISLES 2022 cases.")
    print("Checking all requested channels and masks against the DWI grid ...")

    dataset_dir = stage_isles2022_nnunet(
        cases,
        args.nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        channels=args.channels,
        mode=args.mode,
        overwrite=args.overwrite,
        split_seed=args.split_seed,
    )

    print(f"nnU-Net raw dataset staged at: {dataset_dir}")
    print(f"Deterministic 5-fold manifest: {dataset_dir / 'splits_final.json'}")
    print(
        "After nnUNetv2_plan_and_preprocess, copy that splits_final.json into the "
        "matching nnUNet_preprocessed/DatasetXXX_Name directory before training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
