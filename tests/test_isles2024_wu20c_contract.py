"""WU20C prospective nnU-Net contract regression tests (no patient files)."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "qualify_isles2024_nnunet_runtime.py"
NOTEBOOK = ROOT / "notebooks" / "22_ISLES2024_WU20C_NNUNET_RUNTIME_QUALIFICATION.ipynb"
WU = run_path(str(SOURCE))


def source_plans():
    cfg = {
        "normalization_schemes": ["ZScoreNormalization"] * 5,
        "use_mask_for_norm": [False] * 5,
        "architecture": {"network_class_name": "PlainConvUNet", "arch_kwargs": {"n_stages": 6}},
        "spacing": [2.0, 0.41, 0.41], "patch_size": [32, 256, 224], "batch_size": 2,
        "preprocessor_name": "DefaultPreprocessor", "batch_dice": True,
        "median_image_size_in_voxels": [74, 562, 512],
        "resampling_fn_data": "resample_data_or_seg_to_shape",
        "resampling_fn_seg": "resample_data_or_seg_to_shape",
        "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
        "resampling_fn_data_kwargs": {"order": 3, "is_seg": False, "order_z": 0},
        "resampling_fn_seg_kwargs": {"order": 1, "is_seg": True, "order_z": 0},
        "resampling_fn_probabilities_kwargs": {"order": 1, "is_seg": False, "order_z": 0},
    }
    return {
        "dataset_name": "Dataset505_ISLES2024_NCCT_PERFUSION",
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
        "configurations": {"3d_fullres": cfg, "2d": copy.deepcopy(cfg)},
        "foreground_intensity_properties_per_channel": {str(i): {"mean": i} for i in range(5)},
    }


def test_locked_plans_copy_original_without_global_intensity_properties():
    old = source_plans()
    original = copy.deepcopy(old)
    for dataset_id, n in [(506, 1), (507, 5)]:
        new = WU["build_locked_plans"](old, dataset_id=dataset_id)
        cfg = new["configurations"]["3d_fullres"]
        assert cfg["normalization_schemes"] == ["NoNormalization"] * n
        assert cfg["use_mask_for_norm"] == [False] * n
        assert new["foreground_intensity_properties_per_channel"] == {str(i): {} for i in range(n)}
        assert new["image_reader_writer"] == "SimpleITKIO"
        assert len(new["configurations"]) == 1
        assert cfg["architecture"] == old["configurations"]["3d_fullres"]["architecture"]
    assert old == original


def test_locked_plan_rejects_silent_spacing_or_network_change():
    base = source_plans()
    p = WU["build_locked_plans"](base, dataset_id=506)
    p["configurations"]["3d_fullres"]["patch_size"][0] = 31
    with pytest.raises(ValueError, match="architecture/spacing"):
        WU["assert_locked_plans"](base, p, dataset_id=506)
    q = WU["build_locked_plans"](base, dataset_id=507)
    q["configurations"]["3d_fullres"]["normalization_schemes"][1] = "ZScoreNormalization"
    with pytest.raises(ValueError, match="normalize"):
        WU["assert_locked_plans"](base, q, dataset_id=507)


def fake_manifest(train, heldout):
    return {
        "fold": 2, "frozen_split_sha256": WU["FROZEN_SPLIT_SHA"],
        "n_training_patients": len(train), "n_heldout_patients": len(heldout),
        "heldout_ids": heldout,
        "provisional_fit": {
            ch: {
                "channel": ch, "n_training_cases": len(train),
                "training_ids": train.copy(), "validation_ids_excluded": heldout.copy(),
                "status": "candidate_not_authorized_for_training",
                "clip_low": -100, "clip_high": 100, "center": 0.0, "scale": 30.0,
            }
            for ch in WU["CHANNELS"]
        },
    }


def test_validate_frozen_train_only_119_cases_and_30_heldouts():
    train = [f"s{i:03}" for i in range(119)]
    heldout = [f"v{i:03}" for i in range(30)]
    m = fake_manifest(train, heldout)
    assert len(WU["validate_locked_fold_manifest"](m, train, heldout, 2)) == 5
    m["provisional_fit"]["MTT"]["training_ids"][-1] = heldout[0]
    with pytest.raises(ValueError, match="leakage"):
        WU["validate_locked_fold_manifest"](m, train, heldout, 2)
    m = fake_manifest(train, heldout)
    m["provisional_fit"]["NCCT"]["status"] = "training_approved"
    with pytest.raises(ValueError, match="status"):
        WU["validate_locked_fold_manifest"](m, train, heldout, 2)


def test_image_only_roi_deterministic_and_geometry_shift():
    mask = np.zeros((160, 150, 70), dtype=bool)
    mask[30:100, 20:140, 10:65] = True
    region = WU["image_only_roi"](mask, max_shape=(64, 64, 24))
    assert tuple(s.stop - s.start for s in region) == (64, 64, 24)
    assert WU["image_only_roi"](mask, max_shape=(64, 64, 24)) == region
    affine = np.array([[0.4, 0, 0, 2], [0, 0.4, 0, 3], [0, 0, 2, 4], [0, 0, 0, 1]], dtype=float)
    new = WU["shifted_affine"](affine, region)
    np.testing.assert_allclose(new @ [0, 0, 0, 1], affine @ ([s.start for s in region] + [1]))
    np.testing.assert_array_equal(new[:3, :3], affine[:3, :3])
    with pytest.raises(ValueError, match="support"):
        WU["image_only_roi"](np.zeros((20, 20, 20), dtype=bool))


def test_no_training_or_labels_in_qualification_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    assert "nnunetv2==2.8.1" in source
    assert "--fold-manifest" in source
    assert "nnUNetv2_train" not in source
    assert "labelsTr" not in source
    assert "--case" in source
    assert "training_ready" in source
