import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_nnunet_splits.py"
    spec = importlib.util.spec_from_file_location("verify_nnunet_splits", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_splits_pass(tmp_path: Path):
    module = _load_module()
    payload = (
        '[{"train":["a","b"],"val":["c"]},'
        '{"train":["a","c"],"val":["b"]}]'
    )
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(payload, encoding="utf-8")
    candidate.write_text(payload, encoding="utf-8")

    module.assert_identical_splits(reference, candidate)


def test_changed_patient_assignment_fails(tmp_path: Path):
    module = _load_module()
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        '[{"train":["a","b"],"val":["c"]}]',
        encoding="utf-8",
    )
    candidate.write_text(
        '[{"train":["a","c"],"val":["b"]}]',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="differs"):
        module.assert_identical_splits(reference, candidate)
