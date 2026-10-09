# ruff: noqa: I001
import hashlib
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_isles2024_parallel.py"
MODULE_NAME = "download_isles2024_parallel"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)
plan_ranges = MODULE.plan_ranges


def test_plan_ranges_covers_exact_interval_without_overlap():
    ranges = plan_ranges(start=100, total_size=350, part_size=100)

    assert [(item.start, item.end) for item in ranges] == [
        (100, 199),
        (200, 299),
        (300, 349),
    ]
    assert sum(item.size for item in ranges) == 250


def test_plan_ranges_empty_when_prefix_is_complete():
    assert plan_ranges(start=1024, total_size=1024, part_size=256) == []


def test_assemble_resumes_after_prefix_without_prefix_file(tmp_path):
    prefix_bytes = b"prefix-data"
    chunk0 = b"chunk-zero"
    chunk1 = b"chunk-one"
    final_bytes = prefix_bytes + chunk0 + chunk1

    archive = tmp_path / "train.7z"
    prefix = tmp_path / "train.7z.prefix"
    parts_dir = tmp_path / "train.7z.parts"
    parts_dir.mkdir()

    ranges = [
        MODULE.ByteRange(index=0, start=len(prefix_bytes), end=len(prefix_bytes) + len(chunk0) - 1),
        MODULE.ByteRange(
            index=1,
            start=len(prefix_bytes) + len(chunk0),
            end=len(final_bytes) - 1,
        ),
    ]

    (parts_dir / f"part-0000-{ranges[0].start}-{ranges[0].end}.bin").write_bytes(chunk0)
    (parts_dir / f"part-0001-{ranges[1].start}-{ranges[1].end}.bin").write_bytes(chunk1)

    assembling = tmp_path / "train.7z.assembling"
    assembling.write_bytes(prefix_bytes + chunk0)
    assert not prefix.exists()

    original_md5 = MODULE.EXPECTED_MD5
    MODULE.EXPECTED_MD5 = hashlib.md5(final_bytes).hexdigest()
    try:
        checksum = MODULE.assemble(
            archive,
            prefix,
            parts_dir,
            ranges,
            total_size=len(final_bytes),
        )
    finally:
        MODULE.EXPECTED_MD5 = original_md5

    assert checksum == hashlib.md5(final_bytes).hexdigest()
    assert archive.read_bytes() == final_bytes
    assert not assembling.exists()
