#!/usr/bin/env python3
"""Validate an extracted ISLES 2015 SISS dataset without training a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybridstrokeseg.data.isles2015 import assert_same_geometry, discover_siss_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data/raw/isles2015"),
        help="Extracted ISLES 2015 root",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Also load each case and verify all five images share geometry.",
    )
    args = parser.parse_args()

    cases = discover_siss_cases(args.root)
    print(f"Complete labeled SISS cases found: {len(cases)}")
    for case in cases:
        print(f"- {case.case_id}")
        if args.geometry:
            assert_same_geometry(case.as_dict().values())

    if len(cases) != 28:
        print(
            f"WARNING: expected 28 labeled SISS training cases from the challenge, found {len(cases)}"
        )
        return 2

    print("Dataset discovery check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
