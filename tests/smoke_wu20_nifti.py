"""Nibabel-backed WU20 geometry/float32 I/O smoke for evaluation-runtime CI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from runpy import run_path

import nibabel as nib
import numpy as np

_STAGE = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "stage_isles2024_fold_pilot.py"))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.full((12, 18, 20), -1000.0, dtype=np.float32)
        image[2:11, 3:16, 3:17] = 37.0
        affine = np.array([[0.4, 0, 0, -10], [0, 0.4, 0, 6], [0, 0, 2, -20], [0, 0, 0, 1]], dtype=float)
        source = root / "input.nii.gz"
        nifti = nib.Nifti1Image(image, affine)
        nifti.set_qform(affine, code=1)
        nifti.set_sform(affine, code=1)
        nib.save(nifti, source)
        template = nib.load(source)
        transformed = np.where(image > -500, 2.25, 0.0).astype(np.float32)
        path = root / "output.nii.gz"
        _STAGE["_save_preprocessed"](path, transformed, template)
        after = nib.load(path)
        assert after.shape == template.shape
        assert after.get_data_dtype() == np.dtype(np.float32)
        np.testing.assert_allclose(after.affine, template.affine, atol=1e-6, rtol=0)
        np.testing.assert_allclose(after.get_qform(), template.get_qform(), atol=1e-6, rtol=0)
        np.testing.assert_allclose(after.get_sform(), template.get_sform(), atol=1e-6, rtol=0)
        np.testing.assert_array_equal(after.get_fdata(dtype=np.float32), transformed)
        print("WU20 nibabel NIfTI write/reload and shape/affine/float32: PASS")


if __name__ == "__main__":
    main()
