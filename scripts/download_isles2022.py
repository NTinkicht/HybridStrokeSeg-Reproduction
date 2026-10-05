#!/usr/bin/env python3
"""Download the public ISLES 2022 training release from Zenodo and verify MD5."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/7153326/files/ISLES-2022.zip?download=1"
EXPECTED_MD5 = "302ee280373cdd5c190ab763d72a7a50"
DEFAULT_ARCHIVE = Path("data/raw/ISLES-2022.zip")
DEFAULT_EXTRACT_DIR = Path("data/raw/isles2022")


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - published dataset integrity checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "HybridStrokeSeg-Reproduction/1.0"},
    )
    print(f"Downloading ISLES 2022 to {destination} ...")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.replace(destination)


def verify(path: Path) -> None:
    actual = md5sum(path)
    if actual.lower() != EXPECTED_MD5:
        raise RuntimeError(
            f"ISLES 2022 checksum mismatch: expected {EXPECTED_MD5}, got {actual}. "
            "Delete the archive and download it again."
        )
    print(f"MD5 verified: {actual}")


def _safe_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    root = destination.resolve()
    safe: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        safe.append(member)
    return safe


def extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination, members=_safe_members(archive, destination))
    print(f"Extracted ISLES 2022 to {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.archive.exists():
        args.archive.unlink()
    if not args.archive.exists():
        download(args.archive)
    else:
        print(f"Using existing archive: {args.archive}")

    verify(args.archive)
    if args.extract:
        extract(args.archive, args.extract_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
