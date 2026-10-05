#!/usr/bin/env python3
"""Download the organizer-rearchived ISLES 2015 archive from Zenodo.

This script intentionally stores data under data/raw/, which is ignored by git.
It verifies the archive MD5 published on the Zenodo record before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/19135955/files/ISLES2015.zip?download=1"
EXPECTED_MD5 = "3b6a2226e3814faf272868a96e4837d1"
DEFAULT_ARCHIVE = Path("data/raw/ISLES2015.zip")
DEFAULT_EXTRACT_DIR = Path("data/raw/isles2015")


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - integrity check against published checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    print(f"Downloading {url}")
    print(f"Destination: {destination}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HybridStrokeSeg-Reproduction/1.0"},
    )
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if total_bytes:
                pct = 100.0 * copied / total_bytes
                print(f"\r{copied / 2**20:.1f} MiB / {total_bytes / 2**20:.1f} MiB ({pct:.1f}%)", end="")
        if total_bytes:
            print()

    shutil.move(str(partial), str(destination))


def verify(path: Path) -> None:
    actual = md5sum(path)
    if actual.lower() != EXPECTED_MD5.lower():
        raise RuntimeError(
            "Checksum mismatch. "
            f"Expected {EXPECTED_MD5}, got {actual}. "
            "Delete the archive and download it again."
        )
    print(f"MD5 verified: {actual}")


def extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        archive.extractall(destination)

    print(f"Extracted to: {destination}")

    lowered = [name.lower() for name in names]
    has_siss = any("siss" in name for name in lowered)
    has_training = any("train" in name for name in lowered)
    has_gt = any(
        token in name
        for name in lowered
        for token in ("ot", "lesion", "mask", "groundtruth", "ground_truth")
    )

    print("Archive sanity check:")
    print(f"  SISS-like paths found: {has_siss}")
    print(f"  training-like paths found: {has_training}")
    print(f"  mask/ground-truth-like paths found: {has_gt}")

    if not has_siss:
        print(
            "WARNING: no path containing 'SISS' was detected. "
            "Inspect the extracted directory before running reproduction experiments.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Archive path (default: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help=f"Extraction directory (default: {DEFAULT_EXTRACT_DIR})",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the verified ZIP after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the archive already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.force and args.archive.exists():
        args.archive.unlink()

    if not args.archive.exists():
        download(URL, args.archive)
    else:
        print(f"Using existing archive: {args.archive}")

    verify(args.archive)

    if args.extract:
        extract(args.archive, args.extract_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
