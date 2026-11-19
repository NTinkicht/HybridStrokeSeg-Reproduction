"""Fail-closed WU-19 design regression and normalization candidate math tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_isles2024_wu19_ablation.py"
NORMALIZER = ROOT / "src" / "hybridstrokeseg" / "wu19_normalization_protocol.py"
# Allow the same tests to run against a local authoring directory.
if not SCRIPT.is_file():
    SCRIPT = ROOT / "plan_isles2024_wu19_ablation.py"
if not NORMALIZER.is_file():
    NORMALIZER = ROOT / "wu19" / "normalization_protocol.py"
D = run_path(str(SCRIPT))
NORMAL = run_path(str(NORMALIZER))


def make_folds():
    cases = [f'sub_stroke{i:04d}' for i in range(1, 150)]
    splits = []
    start = 0
    for size in [30,30,30,30,29]:
        validation = cases[start:start + size]
        splits.append({'train': [c for c in cases if c not in validation], 'val': validation})
        start += size
    return splits


def test_frozen_splits_hash_and_no_leakage(tmp_path, monkeypatch):
    path = tmp_path / 'splits.json'
    folds = make_folds()
    path.write_text(json.dumps(folds))
    monkeypatch.setitem(D['frozen_folds'].__globals__, 'FROZEN_SPLIT_SHA', hashlib.sha256(path.read_bytes()).hexdigest())
    assert D['frozen_folds'](path) == folds
    changed = json.loads(json.dumps(folds))
    changed[0]['train'][0] = changed[0]['val'][0]
    path.write_text(json.dumps(changed))
    monkeypatch.setitem(D['frozen_folds'].__globals__, 'FROZEN_SPLIT_SHA', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='leakage'):
        D['frozen_folds'](path)


def test_modified_split_bytes_fail_frozen_sha(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text('[]')
    with pytest.raises(ValueError, match='SHA256'):
        D['frozen_folds'](path)


def test_audit_requires_full_five_channel_inventory(tmp_path):
    p = tmp_path / 'audit.csv'
    cases = {f'sub_stroke{i:04d}' for i in range(1,150)}
    with p.open('w', newline='') as fp:
        writer=csv.DictWriter(fp, fieldnames=['case_id','channel','n_nonfinite','count_below_minus_10000','min','max'])
        writer.writeheader()
        for case in sorted(cases):
            for ch in D['CHANNELS']:
                writer.writerow({"case_id":case,"channel":ch,"n_nonfinite":0,"count_below_minus_10000":0,"min":0,"max":1})
    result=D['audited_csv'](p,D['RAW_FIELDS']|{'case_id','channel'},cases)
    assert len(result)==745
    with p.open('a') as fp: fp.write('sub_stroke0001,MTT,0,0,0,1\n')
    with pytest.raises(ValueError,match='745 unique'):
        D['audited_csv'](p,D['RAW_FIELDS']|{'case_id','channel'},cases)


def test_protocol_never_authorizes_training_and_uses_matching_folds():
    folds=make_folds()
    proto=D['plan_protocol'](folds,{'MTT':{}},{'plans505_sha256':'abc'})
    assert proto['training_ready'] is False
    assert len(proto['folds'])==5
    assert proto['targets']['Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST']['normalization'][0] == 'identical CT transform as Dataset506'
    assert len(proto['targets']['Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST']['normalization'])==5
    for f in proto['folds']:
        assert set(f['parameter_fit_allowed_cases']).isdisjoint(f['parameter_fit_disallowed_cases'])
        assert len(f['training_patient_ids']) in (119,120)


def make_samples(train_ids):
    return {c: np.linspace(-15, 125, 50, dtype=np.float32)+i*.01 for i,c in enumerate(train_ids)}


def test_train_only_fit_rejects_heldout_cases():
    train=['s1','s2','s3']; val=['s4']
    d=make_samples(train)
    good=NORMAL['fit_fold_channel'](d,train,val,channel='NCCT')
    assert good['n_training_cases']==3
    assert good['validation_ids_excluded']==val
    d['s4']=np.linspace(-1e9,1e9,50)
    with pytest.raises(ValueError,match='no validation'):
        NORMAL['fit_fold_channel'](d,train,val,channel='NCCT')
    d.pop('s4')
    d.pop('s3')
    with pytest.raises(ValueError,match='missing'):
        NORMAL['fit_fold_channel'](d,train,val,channel='MTT')


def test_fold_fit_refuses_degenerate_perfusion():
    train=['s1','s2'];val=['s3']
    d={k:np.zeros(50) for k in train}
    with pytest.raises(ValueError,match='Degenerate'):
        NORMAL['fit_fold_channel'](d,train,val,channel='MTT')


def test_candidate_sanitization_finite_and_non_destructive():
    train=['s1','s2']; val=['s3']
    d={k:np.linspace(0,200,100) for k in train}
    params=NORMAL['fit_fold_channel'](d,train,val,channel='MTT')
    arr=np.array([[[np.nan, np.inf, -np.inf, -9e9, 100., 0.]]],dtype=np.float32)
    saved=arr.copy()
    mask=np.array([[[True,True,True,True,True,False]]])
    processed=NORMAL['apply_candidate_transform'](arr,params,support=mask)
    assert np.isfinite(processed).all()
    assert processed.shape==arr.shape
    assert processed[0,0,-1]==0
    assert np.max(np.abs(processed)) <= 8
    np.testing.assert_array_equal(arr,saved)


def test_candidate_ct_rejects_nonfinite_and_bad_support():
    d=make_samples(['s1','s2'])
    params=NORMAL['fit_fold_channel'](d,['s1','s2'],['s3'],channel='NCCT')
    image=np.array([[[np.nan,45.]]])
    with pytest.raises(ValueError, match='Nonfinite NCCT'):
        NORMAL['apply_candidate_transform'](image,params,support=np.ones_like(image,dtype=bool))
    with pytest.raises(ValueError, match='matching 3D'):
        NORMAL['apply_candidate_transform'](image,params,support=np.ones((1,2),dtype=bool))


def test_colab_wu19_notebook_syntax_and_train_block():
    import ast
    notebook_path = ROOT / 'notebooks' / '19_ISLES2024_WU19_PREPROCESSING_PROTOCOL.ipynb'
    if not notebook_path.exists():
        notebook_path = ROOT / 'wu19' / '19_ISLES2024_WU19_PREPROCESSING_PROTOCOL.ipynb'
    nb = json.loads(notebook_path.read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    for c in nb['cells']:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
    assert 'training_ready' in code
    assert "'--raw-audit-csv'" in code
    assert 'nnUNetv2_train' not in code
    assert 'nnUNetv2_plan_and_preprocess' not in code
