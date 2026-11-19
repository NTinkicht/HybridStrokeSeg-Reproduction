"""WU20 candidate fold-isolation and pilot staging safety tests (no patient data)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "hybridstrokeseg" / "wu20_fold_preprocessing.py"
STAGE = ROOT / "scripts" / "stage_isles2024_fold_pilot.py"
NOTEBOOK = ROOT / "notebooks" / "20_ISLES2024_FOLD_SAFE_NORMALIZATION_PILOT.ipynb"
if not SCRIPT.is_file():
    SCRIPT = ROOT / "wu20" / "fold_safe_preprocessing.py"
if not STAGE.is_file():
    STAGE = ROOT / "wu20" / "stage_isles2024_fold_pilot.py"
if not NOTEBOOK.is_file():
    NOTEBOOK = ROOT / "wu20" / "20_ISLES2024_FOLD_SAFE_NORMALIZATION_PILOT.ipynb"
WU = run_path(str(SCRIPT))
ST = run_path(str(STAGE))


def fake_ncct(shape=(16, 32, 32)):
    a = np.full(shape, -1000, dtype=np.float32)
    a[3:14, 5:27, 5:27] = 40
    return a


def fake_splits():
    ids = [f"sub_stroke{i:04d}" for i in range(1, 150)]
    folds = []
    off = 0
    for size in (30, 30, 30, 30, 29):
        val = ids[off:off+size]
        folds.append({"train": [p for p in ids if p not in val], "val": val})
        off += size
    return folds


def test_exact_149_frozen_split_and_train_validation_disjoint(tmp_path, monkeypatch):
    p = tmp_path / "splits.json"
    x = fake_splits()
    p.write_text(json.dumps(x))
    monkeypatch.setitem(WU["validated_split"].__globals__, "FROZEN_SPLIT_SHA", hashlib.sha256(p.read_bytes()).hexdigest())
    train, val = WU["validated_split"](p, 2)
    assert len(train) == 119 and len(val) == 30 and not set(train) & set(val)
    z = fake_splits()
    z[2]["train"].append(z[2]["val"][0])
    p.write_text(json.dumps(z))
    monkeypatch.setitem(WU["validated_split"].__globals__, "FROZEN_SPLIT_SHA", hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="leakage"):
        WU["validated_split"](p, 2)


def test_ncct_mask_from_image_only_has_provisional_qc():
    a = fake_ncct()
    support, summary = WU["ncct_head_tissue_support"](a)
    assert support.dtype == np.bool_
    assert support.sum() == (11*22*22)
    assert summary["clinical_brain_mask_validated"] is False
    assert not support[0, 0, 0]
    assert support[5, 10, 10]
    with pytest.raises(ValueError, match="Non-finite NCCT"):
        x = a.copy(); x[5, 9, 9] = np.nan
        WU["ncct_head_tissue_support"](x)
    with pytest.raises(ValueError, match="head foreground|coverage"):
        WU["ncct_head_tissue_support"](np.full((16,32,32), -1000, dtype=np.float32))


def test_deterministic_sampling_and_missing_modality_gate():
    a = fake_ncct()
    mask, _ = WU["ncct_head_tissue_support"](a)
    a[mask] += np.arange(mask.sum()) % 20
    x = WU["deterministic_training_sample"](a, mask, case_id="s1", channel="NCCT", max_samples=50)
    y = WU["deterministic_training_sample"](a, mask, case_id="s1", channel="NCCT", max_samples=50)
    np.testing.assert_array_equal(x,y)
    assert x.size == 50
    with pytest.raises(ValueError, match="Missing/extra"):
        WU["fit_fold_training_only"]({"NCCT": {}},training_ids=["s1"],excluded_ids=["s2"])


def fitted_params():
    train = ["s1", "s2", "s3"]
    samples = {ch: {case: np.linspace(0, 50, 80, dtype=np.float32) + i
                    for i, case in enumerate(train)} for ch in WU["CHANNELS"]}
    return WU["fit_fold_training_only"](samples, training_ids=train, excluded_ids=["s4"])


def test_validation_patient_never_used_to_fit_and_nonfinite_sanitized():
    p = fitted_params()
    assert set(p) == set(WU["CHANNELS"])
    assert all(x["training_ids"] == ["s1", "s2", "s3"] for x in p.values())
    assert all(x["validation_ids_excluded"] == ["s4"] for x in p.values())
    a = fake_ncct()
    support, _ = WU["ncct_head_tissue_support"](a)
    images = {ch: np.full_like(a, 10, dtype=np.float32) for ch in WU["CHANNELS"]}
    images["NCCT"] = a
    images["MTT"][5,10,10] = np.nan
    images["CBV"][5,10,10] = -1e11
    before = images["MTT"].copy()
    arm506, arm507, diag = WU["candidate_arms"](images,p,support=support)
    assert arm506["NCCT"] is arm507["NCCT"]
    assert diag["nonfinite_input_voxels"]["MTT"] == 1
    assert np.isfinite(arm507["MTT"]).all()
    assert np.max(np.abs(arm507["CBV"])) <= 8.000001
    assert np.isnan(images["MTT"][5,10,10]) and np.isnan(before[5,10,10])
    assert np.all(arm507["MTT"][~support] == 0)


def test_pilot_selection_rejects_nonheldout_case():
    train = [f"s{i:02d}" for i in range(119)]
    val = [f"v{i:02d}" for i in range(30)]
    x,y = ST["choose_pilot_ids"](train,val,n_train=6,validation_case="v17")
    assert len(x)==6 and y==["v17"] and not set(x)&set(y)
    with pytest.raises(ValueError,match="not held-out"):
        ST["choose_pilot_ids"](train,val,n_train=6,validation_case="s00")


def test_no_second_normalization_metadata_and_exact_inventory(tmp_path):
    train=["sub_stroke0001","sub_stroke0002"]
    val=["sub_stroke0003"]
    roots=[]
    for dataset,channels in ((506,1),(507,5)):
        path=tmp_path/f"Dataset{dataset}";roots.append(path)
        (path/"imagesTr").mkdir(parents=True)
        (path/"imagesTs").mkdir()
        (path/"labelsTr").mkdir()
        meta=WU["no_norm_dataset_json"](n_train=len(train),ncct_only=dataset==506)
        (path/"dataset.json").write_text(json.dumps(meta))
        assert set(meta["channel_names"].values())=={"noNorm"}
        for case in train:
            (path/"labelsTr"/f"{case}.nii.gz").write_bytes(b"trainlabel")
        for group,ids in (("imagesTr",train),("imagesTs",val)):
            for case in ids:
                for idx in range(channels):
                    (path/group/f"{case}_{idx:04d}.nii.gz").write_bytes(b"same" if idx==0 else b"extra")
    result=WU["validate_arm_workspaces"](*roots,train_ids=train,val_ids=val)
    assert result["507"]["n_train_images"]==10
    assert result["506"]["n_val_imagesTs"]==1
    (roots[1]/"imagesTr"/"sub_stroke0001_0000.nii.gz").write_bytes(b"different")
    with pytest.raises(ValueError,match="NCCT output differs"):
        WU["validate_arm_workspaces"](*roots,train_ids=train,val_ids=val)


def test_no_heldout_label_can_be_staged(tmp_path):
    train=["sub_stroke0001"]
    val=["sub_stroke0002"]
    roots=[]
    for ds,ch in [(506,1),(507,5)]:
        d=tmp_path/str(ds);roots.append(d)
        for directory in ("imagesTr","imagesTs","labelsTr"):
            (d/directory).mkdir(parents=True)
        (d/"dataset.json").write_text(json.dumps(WU["no_norm_dataset_json"](n_train=1,ncct_only=ds==506)))
        for group,ids in [("imagesTr",train),("imagesTs",val)]:
            for case in ids:
                for i in range(ch):(d/group/f"{case}_{i:04d}.nii.gz").write_bytes(b"raw")
        (d/"labelsTr"/"sub_stroke0001.nii.gz").write_bytes(b"label")
        (d/"labelsTr"/"sub_stroke0002.nii.gz").write_bytes(b"leak")
    with pytest.raises(ValueError,match="inventory mismatch"):
        WU["validate_arm_workspaces"](*roots,train_ids=train,val_ids=val)


def test_notebook_code_parses_and_blocks_gpu_training():
    notebook=json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":ast.parse("".join(cell["source"]))
    source="\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="code")
    assert "--pilot-train-count" in source
    assert "--pilot-validation-case" in source
    assert "nnUNetv2_train" not in source
    assert "nnUNetv2_plan_and_preprocess" not in source
    assert "noNorm" in json.dumps(notebook)
