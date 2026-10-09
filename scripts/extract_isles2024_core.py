#!/usr/bin/env python3
"""Selectively extract the ISLES'24 core modeling subset from train.7z.

The full archive is dominated by raw 4-D CTP. For the first modernization
experiments we only need:
- acute NCCT;
- registered CTA;
- registered perfusion maps (Tmax, CBF, CBV, MTT);
- final infarct labels from derivatives/ses-02;
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
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_CASES = 149

CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "ncct": re.compile(r"/raw_?data/sub-[^/]+/ses-0*1/[^/]*_ncct\.nii\.gz$", re.IGNORECASE),
    "cta": re.compile(
        r"/derivatives/sub-[^/]+/ses-0*1/.*/?[^/]*space-ncct_cta\.nii\.gz$",
        re.IGNORECASE,
    ),
    "tmax": re.compile(
        r"/derivatives/sub-[^/]+/ses-0*1/.*/?[^/]*space-ncct_tmax\.nii\.gz$",
        re.IGNORECASE,
    ),
    "cbf": re.compile(
        r"/derivatives/sub-[^/]+/ses-0*1/.*/?[^/]*space-ncct_cbf\.nii\.gz$",
        re.IGNORECASE,
    ),
    "cbv": re.compile(
        r"/derivatives/sub-[^/]+/ses-0*1/.*/?[^/]*space-ncct_cbv\.nii\.gz$",
        re.IGNORECASE,
    ),
    "mtt": re.compile(
        r"/derivatives/sub-[^/]+/ses-0*1/.*/?[^/]*space-ncct_mtt\.nii\.gz$",
        re.IGNORECASE,
    ),
    "lesion_mask": re.compile(
        r"/derivatives/sub-[^/]+/ses-0*2/[^/]*_lesion-msk\.nii\.gz$",
        re.IGNORECASE,
    ),
    "baseline_csv": re.compile(
        r"/phenotype/ses-0*1/[^/]*_demographic_baseline\.csv$",
        re.IGNORECASE,
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
SUBJECT_RE = re.compile(r"(sub-[^/]+)", re.IGNORECASE)


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
        help="Clear previous extraction state and rewrite selected files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Subjects per resumable extraction batch. Default: 10.",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

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
    state_dir = args.output_dir / ".extract_state"
    if args.overwrite and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    by_subject: dict[str, list[str]] = defaultdict(list)
    for member in selected:
        match = SUBJECT_RE.search(normalize_for_matching(member))
        if match is None:
            raise RuntimeError(f"Selected member has no subject identifier: {member}")
        by_subject[match.group(1).lower()].append(member)

    subject_ids = sorted(by_subject)
    if len(subject_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} unique subjects, found {len(subject_ids)}"
        )

    batches = [
        subject_ids[index : index + args.batch_size]
        for index in range(0, len(subject_ids), args.batch_size)
    ]
    for batch_index, batch_subjects in enumerate(batches):
        marker = state_dir / f"batch_{batch_index:03d}.done"
        if marker.exists() and not args.overwrite:
            print(
                f"[{batch_index + 1}/{len(batches)}] "
                f"Skipping completed batch ({len(batch_subjects)} subjects)"
            )
            continue

        batch_members = [
            member
            for subject_id in batch_subjects
            for member in by_subject[subject_id]
        ]
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as handle:
            listfile = Path(handle.name)
            for member in batch_members:
                handle.write(member + "\n")

        try:
            print(
                f"[{batch_index + 1}/{len(batches)}] Extracting "
                f"{len(batch_members)} files for {len(batch_subjects)} subjects ..."
            )
            subprocess.run(
                [
                    executable,
                    "x",
                    str(args.archive),
                    f"-o{args.output_dir}",
                    "-aoa",
                    "-scsUTF-8",
                    f"-i@{listfile}",
                ],
                check=True,
            )
        finally:
            listfile.unlink(missing_ok=True)

        marker.write_text(
            "\n".join(batch_subjects) + "\n",
            encoding="utf-8",
        )

    manifest["batch_size_subjects"] = args.batch_size
    manifest["batches"] = len(batches)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Extraction manifest: {manifest_path}")
    print("Core extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
