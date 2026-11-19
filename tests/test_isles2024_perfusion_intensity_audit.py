"""Tests for read-only raw perfusion-image intensity screening."""

import ast
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

_MODULE = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "audit_isles2024_perfusion_intensity.py")
)
summarize_pixels = _MODULE["summarize_pixels"]


def test_summarize_pixels_extremes_and_nan_do_not_modify_input():
    data = np.array([0, 0, 1, 2, -29691790, 15, np.nan, np.inf])
    original = data.copy()
    got = summarize_pixels(data)
    assert got["min"] == -29691790
    assert got["max"] == 15
    assert got["n_voxels"] == 8
    assert got["n_nonfinite"] == 2
    assert got["count_below_minus_10000"] == 1
    assert got["count_negative"] == 1
    np.testing.assert_array_equal(data, original)


def test_nonfinite_only_volume_fails():
    with pytest.raises(ValueError, match="no finite"):
        summarize_pixels(np.array([np.nan, np.inf]))


def test_notebook_code_parses_and_is_cpu_only():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "17_ISLES2024_PERFUSION_INTENSITY_AUDIT.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf8"))
    code = "".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "nnUNet_raw" in code
    assert "--images-tr" in code
    assert "intensity_audit_progress.json" in (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_isles2024_perfusion_intensity.py"
    ).read_text(encoding="utf8")
    assert "nnUNetv2_train" not in code
