#!/usr/bin/env python3
"""Resumably download the current public ISLES'24 training archive.

The archive is ~99 GB, so this downloader is designed for persistent storage
such as a mounted Google Drive. Partial downloads are resumed with HTTP Range
requests. The final file is verified against the published Zenodo MD5.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ZENODO_RECORD = "17652035"
ARCHIVE_NAME = "train.7z"
DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}?download=1"
)
EXPECTED_MD5 = "4959a5dd2438d53e3c86d6858484e781"
CHUNK_SIZE = 8 * 1024 * 1024


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _remote_size() -> int | None:
    request = urllib.request.Request(DOWNLOAD_URL, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def download_resumable(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = destination.stat().st_size if destination.exists() else 0
    remote_size = _remote_size()

    if remote_size is not None and current > remote_size:
        print(
            f"Local partial file is larger than remote object ({current} > {remote_size}); "
            "restarting."
        )
        destination.unlink()
        current = 0

    headers: dict[str, str] = {}
    mode = "wb"
    if current > 0:
        headers["Range"] = f"bytes={current}-"
        mode = "ab"
        print(f"Resuming at {current / (1024**3):.2f} GiB")
    else:
        print("Starting ISLES'24 archive download")

    request = urllib.request.Request(DOWNLOAD_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if current > 0 and status != 206:
            print("Server did not honor Range request; restarting from byte 0.")
            current = 0
            mode = "wb"
            request = urllib.request.Request(DOWNLOAD_URL)
            response.close()
            with urllib.request.urlopen(request, timeout=120) as fresh_response:
                _stream(fresh_response, destination, mode, current, remote_size)
            return
        _stream(response, destination, mode, current, remote_size)


def _stream(response, destination: Path, mode: str, current: int, remote_size: int | None) -> None:
    downloaded = current
    next_report = downloaded + 1024**3
    with destination.open(mode) as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                if remote_size:
                    pct = 100.0 * downloaded / remote_size
                    print(
                        f"Downloaded {downloaded / (1024**3):.1f} / "
                        f"{remote_size / (1024**3):.1f} GiB ({pct:.1f}%)"
                    )
                else:
                    print(f"Downloaded {downloaded / (1024**3):.1f} GiB")
                next_report += 1024**3


def extract_archive(archive: Path, output_dir: Path) -> None:
    executable = shutil.which("7zz") or shutil.which("7z")
    if executable is None:
        raise RuntimeError(
            "7-Zip executable not found. In Colab run: apt-get update -qq && "
            "apt-get install -y -qq p7zip-full"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [executable, "x", str(archive), f"-o{output_dir}", "-y"],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/isles2024/train.7z"),
    )
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip the expensive final MD5 verification (not recommended).",
    )
    args = parser.parse_args()

    if args.archive.exists() and not args.skip_checksum:
        print(f"Existing archive found: {args.archive}")
        checksum = md5sum(args.archive)
        if checksum == EXPECTED_MD5:
            print(f"MD5 verified: {checksum}")
        else:
            print(
                f"Existing file MD5 is {checksum}, expected {EXPECTED_MD5}. "
                "It may be a partial download; attempting resume."
            )
            download_resumable(args.archive)
            checksum = md5sum(args.archive)
            if checksum != EXPECTED_MD5:
                raise RuntimeError(
                    f"Checksum mismatch after download: {checksum} != {EXPECTED_MD5}"
                )
            print(f"MD5 verified: {checksum}")
    else:
        download_resumable(args.archive)
        if not args.skip_checksum:
            checksum = md5sum(args.archive)
            if checksum != EXPECTED_MD5:
                raise RuntimeError(
                    f"Checksum mismatch after download: {checksum} != {EXPECTED_MD5}"
                )
            print(f"MD5 verified: {checksum}")

    if args.extract:
        if args.extract_dir is None:
            parser.error("--extract requires --extract-dir")
        extract_archive(args.archive, args.extract_dir)
        print(f"Extracted to: {args.extract_dir}")

    print("ISLES'24 archive ready.")
    print(f"Zenodo record: {ZENODO_RECORD}")
    print(f"Archive: {args.archive}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Partial archive was kept and can be resumed.", file=sys.stderr)
        raise SystemExit(130)
