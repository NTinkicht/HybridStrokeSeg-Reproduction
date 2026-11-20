"""Patient-free WU20G synthetic physical NIfTI staging smoke in evaluator CI."""
from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from hybridstrokeseg.wu20_fold_preprocessing import file_sha256
from stage_isles2024_wu20g_physical import (
    make_synthetic_inputs,
    stage_synthetic_case,
    synthetic_parameters,
    verify_written_case,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wu20g_CI_no_patient_data_") as tmp:
        p = Path(tmp)
        source, d506, d507 = p / "source", p / "506", p / "507"
        training, heldout = "sub_stroke9001", "sub_stroke9002"
        label = make_synthetic_inputs(source, training, label=True)
        assert label is not None
        input_hash = file_sha256(source / "imagesTr" / f"{training}_0000.nii.gz")
        train_receipt = stage_synthetic_case(
            source, d506, d507, training, "imagesTr", synthetic_parameters(), label_file=label
        )
        make_synthetic_inputs(source, heldout, label=False)
        held_receipt = stage_synthetic_case(
            source, d506, d507, heldout, "imagesTs", synthetic_parameters()
        )
        assert train_receipt["training_ready"] is False and held_receipt["training_ready"] is False
        assert train_receipt["synthetic_only"] and held_receipt["synthetic_only"]
        assert input_hash == file_sha256(source / "imagesTr" / f"{training}_0000.nii.gz")
        assert (d506 / "labelsTr" / f"{training}.nii.gz").exists()
        assert (d507 / "labelsTr" / f"{training}.nii.gz").exists()
        assert not (d506 / "labelsTr" / f"{heldout}.nii.gz").exists()
        assert not (d507 / "labelsTr" / f"{heldout}.nii.gz").exists()
        tr = verify_written_case(d506, d507, training, "imagesTr", has_label=True)
        va = verify_written_case(d506, d507, heldout, "imagesTs", has_label=False)
        assert tr["506_NCCT"] == tr["507_NCCT"]
        assert va["506_NCCT"] == va["507_NCCT"]
        assert train_receipt["candidate_transform_diagnostics"]["nonfinite_input_voxels"]["MTT"] == 1
        assert train_receipt["candidate_transform_diagnostics"]["abs_gt_8_output_voxels"]["MTT"] == 0
        # Both arms truly received byte-equal NCCT, not independent transforms.
        assert file_sha256(d506/"imagesTr"/f"{training}_0000.nii.gz") == file_sha256(
            d507/"imagesTr"/f"{training}_0000.nii.gz")
        # Tampering is never mistaken for a valid stage.
        bad = d507/"imagesTr"/f"{training}_0000.nii.gz"
        nifti = nib.load(str(bad))
        data = nifti.get_fdata(dtype=np.float32)
        data[14, 14, 7] += 0.125
        nib.save(nib.Nifti1Image(data, nifti.affine, header=nifti.header), str(bad))
        try:
            verify_written_case(d506, d507, training, "imagesTr", has_label=True)
        except ValueError as exc:
            assert "NCCT bytes" in str(exc)
        else:
            raise AssertionError("Tampered paired NCCT was silently accepted")
        # An existing staged case must not be overwritten on resume.
        try:
            stage_synthetic_case(source, d506, d507, heldout, "imagesTs", synthetic_parameters())
        except FileExistsError:
            pass
        else:
            raise AssertionError("Unreviewed overwrite permitted")
    print("WU20G: paired 506/507 native staging, heldout isolation, anomaly sanitation, "
          "byte hashes and overwrite guards PASS")


if __name__ == "__main__":
    main()
