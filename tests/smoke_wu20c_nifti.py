"""Runtime smoke using real Nibabel NIfTI files (not patient data)."""

import tempfile
from pathlib import Path
from runpy import run_path

import nibabel as nib
import numpy as np

module = run_path(str(Path(__file__).resolve().parents[1] / "scripts" /
                      "qualify_isles2024_nnunet_runtime.py"))
with tempfile.TemporaryDirectory() as folder:
    raw = np.ones((24, 24, 24), dtype=np.float32) * 8
    affine = np.array([[0.5, 0, 0, 2], [0, 0.5, 0, 3], [0, 0, 2, 4], [0, 0, 0, 1.]])
    nii = nib.Nifti1Image(raw, affine)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=2)
    roi = (slice(4, 14), slice(6, 20), slice(5, 15))
    target = Path(folder) / "roi.nii.gz"
    module["safe_write_roi_nifti"](target, raw[roi], nii, roi)
    reloaded = nib.load(str(target))
    expected = module["shifted_affine"](affine, roi)
    np.testing.assert_allclose(reloaded.affine, expected)
    np.testing.assert_allclose(reloaded.get_qform(), expected)
    np.testing.assert_allclose(reloaded.get_sform(), expected)
    np.testing.assert_allclose(reloaded.get_fdata(dtype=np.float32), raw[roi])
    assert reloaded.shape == (10, 14, 10)
print("WU20C NIfTI ROI affine, sform/qform and voxel fidelity: PASS")
