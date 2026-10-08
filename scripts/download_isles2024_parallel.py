#!/usr/bin/env python3
"""Download the ISLES'24 Zenodo archive in resumable parallel byte ranges.

This downloader is designed for mounted Google Drive in Colab:
- preserves any contiguous prefix previously downloaded by the older downloader;
- downloads the remaining file as independent HTTP Range chunks in parallel;
- keeps completed chunks in persistent storage across Colab disconnects;
- assembles the final archive only after every chunk is present;
- verifies the official Zenodo MD5 before deleting temporary chunks.

Each worker writes to an independent persistent chunk file, so concurrency does
not require random writes to one huge sparse archive. The Colab workflow uses
16 workers by request; retries and resumable chunk state handle transient
throttling or disconnects.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ZENODO_RECORD = "17652035"
ARCHIVE_NAME = "train.7z"
DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}?download=1"
)
EXPECTED_MD5 = "4959a5dd2438d53e3c86d6858484e781"
DEFAULT_PART_SIZE = 1024**3  # 1 GiB
COPY_BUFFER = 16 * 1024 * 1024


@dataclass(frozen=True)
class ByteRange:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def _status(response) -> int | None:
    return getattr(response, "status", None) or response.getcode()


def get_remote_size(url: str) -> int:
    """Return remote object size, requiring HTTP Range support."""
    request = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "HybridStrokeSeg-ISLES24/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = _status(response)
        content_range = response.headers.get("Content-Range")
        if status == 206 and content_range:
            match = re.fullmatch(r"bytes\s+0-0/(\d+)", content_range.strip())
            if not match:
                raise RuntimeError(f"Unexpected Content-Range: {content_range}")
            return int(match.group(1))

        if status == 200:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                raise RuntimeError(
                    "Server returned the whole object instead of honoring a Range request. "
                    "Parallel chunking is not safe for this endpoint."
                )

        raise RuntimeError(
            f"Zenodo endpoint did not confirm HTTP Range support: status={status}, "
            f"Content-Range={content_range!r}"
        )


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(COPY_BUFFER)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def plan_ranges(start: int, total_size: int, part_size: int) -> list[ByteRange]:
    if start < 0 or total_size < 1 or start > total_size:
        raise ValueError("invalid start/total_size")
    if part_size < 1:
        raise ValueError("part_size must be positive")

    ranges: list[ByteRange] = []
    cursor = start
    index = 0
    while cursor < total_size:
        end = min(cursor + part_size - 1, total_size - 1)
        ranges.append(ByteRange(index=index, start=cursor, end=end))
        cursor = end + 1
        index += 1
    return ranges


def _range_path(parts_dir: Path, item: ByteRange) -> Path:
    return parts_dir / f"part-{item.index:04d}-{item.start}-{item.end}.bin"


def _partial_path(final_path: Path) -> Path:
    return final_path.with_suffix(final_path.suffix + ".partial")


def _request_range(url: str, start: int, end: int):
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "HybridStrokeSeg-ISLES24/1.0",
        },
    )
    return urllib.request.urlopen(request, timeout=180)


def download_range(
    url: str,
    item: ByteRange,
    parts_dir: Path,
    *,
    retries: int,
) -> tuple[int, str]:
    """Download or resume one independent byte range."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    final_path = _range_path(parts_dir, item)

    partial = _partial_path(final_path)

    if final_path.exists():
        final_size = final_path.stat().st_size
        if final_size == item.size:
            return item.index, "existing"

        # Google Drive FUSE can occasionally persist a file rename before every
        # byte has reached Drive. Treat an undersized ".bin" as resumable data
        # instead of deleting and redownloading the whole range.
        if 0 < final_size < item.size:
            partial_size = partial.stat().st_size if partial.exists() else -1
            if final_size > partial_size:
                if partial.exists():
                    partial.unlink()
                final_path.replace(partial)
                print(
                    f"Part {item.index}: salvaged {final_size / (1024**2):.1f} MiB "
                    "from an undersized completed chunk.",
                    flush=True,
                )
            else:
                final_path.unlink()
        else:
            final_path.unlink()
    attempt = 0
    while True:
        existing = partial.stat().st_size if partial.exists() else 0
        if existing > item.size:
            partial.unlink()
            existing = 0
        if existing == item.size:
            partial.replace(final_path)
            return item.index, "completed"

        request_start = item.start + existing
        try:
            with _request_range(url, request_start, item.end) as response:
                status = _status(response)
                content_range = response.headers.get("Content-Range", "")
                if status != 206:
                    raise RuntimeError(
                        f"Range request {request_start}-{item.end} returned HTTP {status}"
                    )
                expected_prefix = f"bytes {request_start}-{item.end}/"
                if not content_range.startswith(expected_prefix):
                    raise RuntimeError(
                        f"Unexpected Content-Range for part {item.index}: {content_range}"
                    )

                mode = "ab" if existing else "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(COPY_BUFFER)
                        if not block:
                            break
                        handle.write(block)

            final_size = partial.stat().st_size
            if final_size != item.size:
                raise RuntimeError(
                    f"Part {item.index} incomplete: {final_size} != {item.size}"
                )
            partial.replace(final_path)
            return item.index, "downloaded"

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            RuntimeError,
            OSError,
        ) as exc:
            attempt += 1
            if retries >= 0 and attempt > retries:
                raise RuntimeError(
                    f"Part {item.index} failed after {attempt} attempts: {exc}"
                ) from exc
            wait = min(60, 3 * attempt)
            print(
                f"Part {item.index} retry {attempt} in {wait}s after: {exc}",
                flush=True,
            )
            time.sleep(wait)


def adopt_existing_prefix(archive: Path, prefix: Path, total_size: int) -> int:
    """Preserve a partial file produced by the previous sequential downloader."""
    if prefix.exists():
        size = prefix.stat().st_size
        if size >= total_size:
            raise RuntimeError(
                f"Saved prefix has invalid size {size}; expected less than {total_size}"
            )
        return size

    if archive.exists():
        size = archive.stat().st_size
        if 0 < size < total_size:
            print(
                f"Preserving existing {size / (1024**3):.2f} GiB partial download "
                "as the contiguous prefix.",
                flush=True,
            )
            archive.replace(prefix)
            return size

    return 0


def write_manifest(
    manifest_path: Path,
    *,
    total_size: int,
    prefix_size: int,
    ranges: list[ByteRange],
    workers: int,
    part_size: int,
) -> None:
    manifest = {
        "zenodo_record": ZENODO_RECORD,
        "url": DOWNLOAD_URL,
        "archive_name": ARCHIVE_NAME,
        "expected_md5": EXPECTED_MD5,
        "total_size_bytes": total_size,
        "prefix_size_bytes": prefix_size,
        "parallel_workers": workers,
        "part_size_bytes": part_size,
        "ranges": [asdict(item) | {"size": item.size} for item in ranges],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assemble(
    archive: Path,
    prefix: Path,
    parts_dir: Path,
    ranges: list[ByteRange],
    *,
    total_size: int,
) -> str:
    assembling = archive.with_suffix(archive.suffix + ".assembling")
    if assembling.exists():
        assembling.unlink()

    digest = hashlib.md5()
    written = 0

    with assembling.open("wb") as destination:
        if prefix.exists():
            with prefix.open("rb") as source:
                while True:
                    block = source.read(COPY_BUFFER)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
                    written += len(block)

        for item in ranges:
            part_path = _range_path(parts_dir, item)
            if not part_path.exists() or part_path.stat().st_size != item.size:
                raise RuntimeError(f"Missing or incomplete chunk: {part_path}")
            with part_path.open("rb") as source:
                while True:
                    block = source.read(COPY_BUFFER)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
                    written += len(block)

    if written != total_size:
        raise RuntimeError(
            f"Assembled size mismatch: wrote {written}, expected {total_size}"
        )

    checksum = digest.hexdigest()
    if checksum != EXPECTED_MD5:
        raise RuntimeError(
            f"Final MD5 mismatch: {checksum} != {EXPECTED_MD5}. "
            "Temporary prefix/chunks were kept for diagnosis."
        )

    assembling.replace(archive)
    return checksum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/isles2024/train.7z"),
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Persistent chunk directory. Default: <archive>.parts",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--part-size-gib",
        type=float,
        default=1.0,
        help="Independent range size in GiB. Default: 1.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=20,
        help="Retries per chunk. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep prefix/chunk files after successful checksum verification.",
    )
    args = parser.parse_args()

    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.part_size_gib <= 0:
        parser.error("--part-size-gib must be positive")

    archive = args.archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.parts_dir or Path(str(archive) + ".parts")
    prefix = Path(str(archive) + ".prefix")
    manifest_path = parts_dir / "manifest.json"
    part_size = max(1, int(args.part_size_gib * 1024**3))

    print("Checking Zenodo Range support and remote size ...", flush=True)
    total_size = get_remote_size(DOWNLOAD_URL)
    print(f"Remote size: {total_size / (1024**3):.2f} GiB", flush=True)

    if archive.exists() and archive.stat().st_size == total_size:
        print("A full-size archive already exists; verifying MD5 ...", flush=True)
        checksum = md5sum(archive)
        if checksum == EXPECTED_MD5:
            print(f"MD5 verified: {checksum}")
            return 0
        raise RuntimeError(
            f"Existing full-size archive has wrong MD5: {checksum}. "
            "Move/delete it before retrying."
        )

    prefix_size = adopt_existing_prefix(archive, prefix, total_size)
    ranges = plan_ranges(prefix_size, total_size, part_size)
    parts_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        total_size=total_size,
        prefix_size=prefix_size,
        ranges=ranges,
        workers=args.workers,
        part_size=part_size,
    )

    completed_bytes = prefix_size + sum(
        item.size
        for item in ranges
        if _range_path(parts_dir, item).exists()
        and _range_path(parts_dir, item).stat().st_size == item.size
    )
    print(
        f"Already persistent: {completed_bytes / (1024**3):.2f} / "
        f"{total_size / (1024**3):.2f} GiB",
        flush=True,
    )
    print(
        f"Downloading {len(ranges)} independent chunks with {args.workers} workers.",
        flush=True,
    )

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_range,
                DOWNLOAD_URL,
                item,
                parts_dir,
                retries=args.retries,
            ): item
            for item in ranges
        }
        done = 0
        persistent = completed_bytes
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                _, status = future.result()
                done += 1
                if status != "existing":
                    persistent += item.size
                print(
                    f"[{done}/{len(ranges)}] part {item.index:04d} {status}; "
                    f"persistent complete ranges {persistent / (1024**3):.1f} / "
                    f"{total_size / (1024**3):.1f} GiB",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"part {item.index}: {exc}")
                print(f"FAILED part {item.index}: {exc}", flush=True)

    if failures:
        raise RuntimeError(
            "Some chunks did not finish. Rerun the same command; completed chunks "
            "will be skipped and partial chunks resumed.\n" + "\n".join(failures)
        )

    print("All chunks are present. Assembling and verifying MD5 ...", flush=True)
    checksum = assemble(
        archive,
        prefix,
        parts_dir,
        ranges,
        total_size=total_size,
    )
    print(f"MD5 verified: {checksum}", flush=True)
    print(f"Final archive: {archive}", flush=True)

    if not args.keep_parts:
        if prefix.exists():
            prefix.unlink()
        shutil.rmtree(parts_dir)
        print("Temporary prefix/chunks removed after successful verification.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
