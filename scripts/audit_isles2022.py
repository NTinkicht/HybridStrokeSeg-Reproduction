#!/usr/bin/env python3
"""Audit an extracted ISLES 2022 training release before model development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybridstrokeseg.data import discover_isles2022_cases, summarize_geometry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data/raw/isles2022"),
        help="Extracted ISLES 2022 root containing rawdata/ and derivatives/",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable audit summary.",
    )
    args = parser.parse_args()

    cases = discover_isles2022_cases(args.root)
    summary = summarize_geometry(cases)
    summary["expected_public_training_cases"] = 250
    summary["case_count_matches_release"] = len(cases) == 250

    print(f"Complete labeled ISLES 2022 cases: {len(cases)}")
    print(f"ADC already matches DWI grid: {summary['adc_matches_dwi']}/{len(cases)}")
    print(f"FLAIR already matches DWI grid: {summary['flair_matches_dwi']}/{len(cases)}")
    print(f"Mask already matches DWI grid: {summary['mask_matches_dwi']}/{len(cases)}")

    if not summary["case_count_matches_release"]:
        print("WARNING: public training release is expected to contain 250 labeled cases.")

    if summary["flair_matches_dwi"] != len(cases):
        print(
            "NOTE: at least one FLAIR image is not on the DWI grid. Do not stack modalities "
            "for a 3-D network until registration/resampling is defined and validated."
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote audit summary: {args.json}")

    return 0 if summary["case_count_matches_release"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
