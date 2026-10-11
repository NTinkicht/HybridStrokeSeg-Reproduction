import importlib.util
from pathlib import Path


def _load_orchestrator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_isles2024_nnunet_cv.py"
    spec = importlib.util.spec_from_file_location("run_isles2024_nnunet_cv", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_case(config_dir: Path, case_id: str) -> None:
    (config_dir / f"{case_id}.b2nd").write_bytes(b"image")
    (config_dir / f"{case_id}_seg.b2nd").write_bytes(b"label")
    (config_dir / f"{case_id}.pkl").write_bytes(b"props")


def test_missing_preprocessed_files_detects_case_dropout(tmp_path: Path):
    module = _load_orchestrator_module()
    dataset = tmp_path / "Dataset504_ISLES2024_NCCT"
    config = dataset / "nnUNetPlans_3d_fullres"
    config.mkdir(parents=True)

    (dataset / "splits_final.json").write_text(
        '[{"train": ["sub_stroke0001"], "val": ["sub_stroke0002"]}]',
        encoding="utf-8",
    )
    _write_complete_case(config, "sub_stroke0001")
    _write_complete_case(config, "sub_stroke0002")

    assert module.missing_preprocessed_files(dataset, "3d_fullres") == []

    missing_path = config / "sub_stroke0002.b2nd"
    missing_path.unlink()

    missing = module.missing_preprocessed_files(dataset, "3d_fullres")
    assert missing == [missing_path]


def test_cache_preprocessed_dataset_builds_verified_local_copy(tmp_path: Path):
    module = _load_orchestrator_module()
    source = tmp_path / "persistent" / "Dataset504_ISLES2024_NCCT"
    config = source / "nnUNetPlans_3d_fullres"
    config.mkdir(parents=True)

    (source / "splits_final.json").write_text(
        '[{"train": ["sub_stroke0001"], "val": ["sub_stroke0002"]}]',
        encoding="utf-8",
    )
    _write_complete_case(config, "sub_stroke0001")
    _write_complete_case(config, "sub_stroke0002")

    cache_root = tmp_path / "cache"
    cached_dataset = module.cache_preprocessed_dataset(
        source,
        cache_root,
        "3d_fullres",
        retries=1,
    )

    assert cached_dataset == cache_root / source.name
    assert module.missing_preprocessed_files(cached_dataset, "3d_fullres") == []
