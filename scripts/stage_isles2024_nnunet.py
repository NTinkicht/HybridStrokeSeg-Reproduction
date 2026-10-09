#!/usr/bin/env python3
"""Stage leakage-safe ISLES'24 data for nnU-Net v2."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybridstrokeseg.data.isles2024 import discover_isles2024_cases
from hybridstrokeseg.nnunet_isles2024 import stage_isles2024_nnunet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("nnunet_raw", type=Path)
    parser.add_argument("--dataset-id", type=int, default=504)
    parser.add_argument("--dataset-name", default="ISLES2024_NCCT")
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=("ncct", "cta", "tmax", "cbf", "cbv", "mtt"),
        default=("ncct",),
        help="Pre-interventional acute input channels only.",
    )
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially staged dataset by keeping complete copied files.",
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    cases = discover_isles2024_cases(
        args.dataset_root,
        required_channels=args.channels,
    )
    print(f"Discovered {len(cases)} labeled ISLES'24 training cases.")
    print(
        "ANTI-LEAKAGE: follow-up DWI/ADC, final masks, post-treatment variables, "
        "and outcome variables are not eligible model inputs."
    )

    dataset_dir = stage_isles2024_nnunet(
        cases,
        args.nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        channels=args.channels,
        mode=args.mode,
        overwrite=args.overwrite,
        resume=args.resume,
        split_seed=args.split_seed,
    )
    print(f"nnU-Net raw dataset staged at: {dataset_dir}")
    print(f"Deterministic patient folds: {dataset_dir / 'splits_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
