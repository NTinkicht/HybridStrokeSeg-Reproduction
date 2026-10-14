"""Keep the CPU-only Colab recovery entrypoints syntactically valid."""

import ast
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    (
        "12_isles2024_NCCT_OOF_EVALUATION.ipynb",
        "15_isles2024_NCCT_OOF_RESUMABLE_CPU.ipynb",
    ),
)
def test_cpu_recovery_notebooks_are_valid_and_dont_install_editable_extras(name):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "ncct_oof_progress.json" in code
    assert "ncct_oof_resumable.log" in code
    assert "panoptica==0.5.22" in code
    assert "run_logged" in code
    assert "pip install -e" not in code
    assert '".[evaluation]"' not in code
