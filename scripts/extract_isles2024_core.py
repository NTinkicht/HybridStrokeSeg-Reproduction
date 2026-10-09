#!/usr/bin/env python3
"""Selectively extract the ISLES'24 core modeling subset from train.7z.

The full archive is dominated by raw 4-D CTP. For the first modernization
experiments we only need:
- acute NCCT;
- registered CTA;
- registered perfusion maps (Tmax, CBF, CBV, MTT);
- final infarct labels from derivatives/ses-0002;
- baseline clinical CSV files.

Follow-up DWI/ADC, post-treatment outcomes, and raw 4-D CTP are deliberately
excluded from this extraction profile to reduce storage and enforce the
anti-leakage boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

EXPECTED_CASES = 149

CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "ncct": re.compile(r"/raw_?data/sub-[^/]+/ses-0001/[^/]*_ncct\.nii\.gz$", re.I),
    "cta": re.compile(
        r"/derivatives/sub-[^/]+/ses-0001/.*/?[^/]*space-ncct_cta\.nii\.gz$",
        re.I,
    ),
    "tmax": re.compile(
        r"/derivatives/sub-[^/]+/ses-0001/.*/?[^/]*space-ncct_tmax\.nii\.gz$",
        re.I,
    ),
    "cbf": re.compile(
        r"/derivatives/sub-[^/]+/ses-0001/.*/?[^/]*space-ncct_cbf\.nii\.gz$",
        re.I,
    ),
    "cbv": re.compile(
        r"/derivatives/sub-[^/]+/ses-0001/.*/?[^/]*space-ncct_cbv\.nii\.gz$",
        re.I,
    ),
    "mtt": re.compile(
        r"/derivatives/sub-[^/]+/ses-0001/.*/?[^/]*space-ncct_mtt\.nii\.gz$",
        re.I,
    ),
    "lesion_mask": re.compile(
        r"/derivatives/sub-[^/]+/ses-0002/[^/]*_lesion-msk\.nii\.gz$",
        re.I,
    ),
    "baseline_csv": re.compile(
        r"/phenotype/ses-0001/[^/]*_demographic_baseline\.csv$",
        re.I,
    ),
}

REQUIRED_IMAGING_CATEGORIES = (
    "ncct",
    "cta",
    "tmax",
    "cbf",
    "cbv",
    "mtt",
    "lesion_mask",
)


def find_7z() -> str:
    executable = shutil.which("7zz") or shutil.which("7z")
    if executable is None:
        raise RuntimeError(
            "7-Zip not found. In Colab run: apt-get update -qq && "
            "apt-get install -y -qq p7zip-full"
        )
    return executable


def list_archive_members(archive: Path, executable: str) -> list[str]:
    result = subprocess.run(
        [executable, "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    members: list[str] = []
    for line in result.stdout.splitlines():
        if not line.startswith("Path = "):
            continue
        value = line[len("Path = ") :].strip().replace("\\", "/")
        if value and value != str(archive):
            members.append(value)
    return members


def normalize_for_matching(member: str) -> str:
    value = member.replace("\\", "/").lstrip("./")
    return "/" + value


def classify_member(member: str) -> str | None:
    normalized = normalize_for_matching(member)
    for category, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(normalized):
            return category
    return None


def select_core_members(members: list[str]) -> tuple[list[str], Counter[str]]:
    selected: list[str] = []
    counts: Counter[str] = Counter()
    for member in members:
        category = classify_member(member)
        if category is None:
            continue
        selected.append(member)
        counts[category] += 1
    return selected, counts


def validate_counts(counts: Counter[str]) -> None:
    failures = {
        category: counts[category]
        for category in REQUIRED_IMAGING_CATEGORIES
        if counts[category] != EXPECTED_CASES
    }
    if failures:
        detail = ", ".join(f"{key}={value}" for key, value in failures.items())
        raise RuntimeError(
            "Core extraction selection does not match the expected 149-case release: "
            + detail
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults inside output_dir.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Inspect and validate archive contents without extracting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow extraction into a non-empty output directory.",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {args.output_dir}. "
            "Use --overwrite to resume/refresh extraction."
        )

    executable = find_7z()
    print(f"Listing archive with {executable}: {args.archive}")
    members = list_archive_members(args.archive, executable)
    print(f"Archive members discovered: {len(members)}")

    selected, counts = select_core_members(members)
    validate_counts(counts)

    print("Selected ISLES'24 core subset:")
    for key in CATEGORY_PATTERNS:
        print(f"  {key}: {counts[key]}")
    print(f"  total selected files: {len(selected)}")

    manifest_path = args.manifest or args.output_dir / "core_extraction_manifest.json"
    manifest = {
        "archive": str(args.archive),
        "expected_cases": EXPECTED_CASES,
        "profile": "core-no-raw-ctp-no-followup-mri-no-outcomes",
        "counts": dict(counts),
        "selected_files": len(selected),
        "anti_leakage_exclusions": [
            "raw 4-D CTP (deferred experiment)",
            "follow-up DWI",
            "follow-up ADC",
            "post-treatment/outcome CSV",
        ],
    }

    if args.list_only:
        print(json.dumps(manifest, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as handle:
        listfile = Path(handle.name)
        for member in selected:
            handle.write(member + "\n")

    try:
        print(f"Extracting {len(selected)} selected files to {args.output_dir} ...")
        subprocess.run(
            [
                executable,
                "x",
                str(args.archive),
                f"-o{args.output_dir}",
                "-y",
                "-scsUTF-8",
                f"-i@{listfile}",
            ],
            check=True,
        )
    finally:
        listfile.unlink(missing_ok=True)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Extraction manifest: {manifest_path}")
    print("Core extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
