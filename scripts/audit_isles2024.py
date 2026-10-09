#!/usr/bin/env python3
"""Audit an extracted ISLES'24 public training release before modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybridstrokeseg.data.isles2024 import (
    discover_isles2024_cases,
    summarize_isles2024,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data/raw/isles2024"),
        help="Extracted ISLES'24 root containing raw_data/rawdata, derivatives and phenotype.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable audit summary.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=("ncct", "cta", "tmax", "cbf", "cbv", "mtt"),
        default=("ncct", "cta", "tmax", "cbf", "cbv", "mtt"),
        help="Acute imaging channels expected in this extracted subset.",
    )
    args = parser.parse_args()

    cases = discover_isles2024_cases(args.root, required_channels=args.channels)
    summary = summarize_isles2024(cases)
    summary["expected_public_training_cases"] = 149
    summary["case_count_matches_release"] = len(cases) == 149
    summary["task"] = (
        "predict final post-treatment infarct from pre-interventional acute CT "
        "and optional clinical data"
    )
    summary["required_channels"] = list(args.channels)
    summary["anti_leakage_rule"] = (
        "follow-up DWI/ADC and outcome variables are target-generation context only "
        "and must not be used as model inputs"
    )

    print(f"Complete labeled ISLES'24 cases: {len(cases)}")
    print(f"Expected acute channels: {', '.join(args.channels)}")
    for channel in args.channels:
        if channel == "ncct":
            print(f"ncct_present: {summary['ncct_present']}/{len(cases)}")
        else:
            print(f"{channel}_present: {summary[f'{channel}_present']}/{len(cases)}")
            print(
                f"{channel}_matches_ncct: "
                f"{summary[f'{channel}_matches_ncct']}/{len(cases)}"
            )
    print(
        f"lesion_mask_matches_ncct: "
        f"{summary['lesion_mask_matches_ncct']}/{len(cases)}"
    )

    print(f"Baseline clinical CSV present: {summary['baseline_csv_present']}/{len(cases)}")
    print(f"Outcome CSV present: {summary['outcome_csv_present']}/{len(cases)}")
    print(f"LVO mask present: {summary['lvo_mask_present']}/{len(cases)}")
    print(f"CoW mask present: {summary['cow_mask_present']}/{len(cases)}")
    print(f"Native 4-D CTP present: {summary['native_ctp_present']}/{len(cases)}")
    print(f"Registered CTP present: {summary['registered_ctp_present']}/{len(cases)}")

    if not summary["case_count_matches_release"]:
        print("WARNING: current public training release is expected to contain 149 cases.")

    geometry_keys = ["lesion_mask_matches_ncct"]
    geometry_keys.extend(
        f"{channel}_matches_ncct"
        for channel in args.channels
        if channel != "ncct"
    )
    if any(summary[key] != len(cases) for key in geometry_keys):
        print(
            "WARNING: at least one required derivative does not match the NCCT grid. "
            "Do not stage multimodal channels until the mismatch is understood."
        )

    print(
        "ANTI-LEAKAGE: follow-up DWI/ADC and post-treatment/outcome information "
        "must never be used as prediction inputs."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote audit summary: {args.json}")

    return 0 if summary["case_count_matches_release"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
