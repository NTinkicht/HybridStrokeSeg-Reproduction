# ruff: noqa: I001
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
