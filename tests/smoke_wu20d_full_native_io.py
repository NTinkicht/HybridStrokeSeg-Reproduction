"""WU20D evaluator-runtime NIfTI full-native-grid smoke; no patient data."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from stage_isles2024_fold_pilot import _save_preprocessed
from hybridstrokeseg.wu20_fold_preprocessing import candidate_arms


def main() -> None:
    rng = np.random.default_rng(42)
    shape = (32, 48, 24)
    affine = np.diag((0.41015625, 0.41015625, 2.0, 1.0))
    ncct = np.full(shape, -1000.0, dtype=np.float32)
    ncct[4:28, 6:42, 3:21] = rng.normal(35, 6, size=(24, 36, 18))
    support = ncct > -100
    images = {"NCCT": ncct}
    for channel in ("TMAX", "CBF", "CBV", "MTT"):
        images[channel] = rng.normal(10, 2, size=shape).astype(np.float32)
    images["MTT"][15, 21, 9] = np.nan
    images["MTT"][16, 21, 9] = -788640128.0
    params = {}
    for channel in images:
        params[channel] = {
            "channel": channel, "status": "candidate_not_authorized_for_training",
            "clip_low": -100.0 if channel == "NCCT" else -50.0,
            "clip_high": 150.0 if channel == "NCCT" else 40.0,
            "center": 30.0 if channel == "NCCT" else 10.0,
            "scale": 20.0 if channel == "NCCT" else 4.0,
            "candidate_post_z_cap": None if channel == "NCCT" else 8.0,
        }
    arm506, arm507, audit = candidate_arms(images, params, support=support)
    assert audit["nonfinite_input_voxels"]["MTT"] == 1
    assert np.isfinite(arm507["MTT"]).all()
    assert np.max(np.abs(arm507["MTT"])) <= 8
    with tempfile.TemporaryDirectory(prefix="wu20d_nifti_smoke_") as td:
        root = Path(td)
        original = nib.Nifti1Image(ncct, affine)
        original.set_qform(affine, code=1)
        original.set_sform(affine, code=1)
        p506 = root / "506.nii.gz"
        p507 = root / "507.nii.gz"
        _save_preprocessed(p506, arm506["NCCT"], original)
        shutil.copy2(p506, p507)
        assert p506.read_bytes() == p507.read_bytes()
        reloaded = nib.load(p507)
        assert reloaded.shape == shape
        np.testing.assert_allclose(reloaded.affine, affine, atol=1e-5)
        np.testing.assert_array_equal(reloaded.get_fdata(dtype=np.float32), arm507["NCCT"])
        assert int(reloaded.header["qform_code"]) == 1
        assert int(reloaded.header["sform_code"]) == 1
    print("WU20D synthetic full-native-grid NIfTI + invalid perfusion smoke PASS")


if __name__ == "__main__":
    main()
