# ruff: noqa: I001
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_isles2024_core.py"
MODULE_NAME = "extract_isles2024_core"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


def test_classify_official_core_members():
    samples = {
        "train/rawdata/sub-strokecase0001/ses-0001/"
        "sub-strokecase0001_ses-0001_ncct.nii.gz": "ncct",
        "train/derivatives/sub-strokecase0001/ses-0001/"
        "sub-strokecase0001_ses-0001_space-ncct_cta.nii.gz": "cta",
        "train/derivatives/sub-strokecase0001/ses-0001/perfusion-maps/"
        "sub-strokecase0001_ses-0001_space-ncct_tmax.nii.gz": "tmax",
        "train/derivatives/sub-strokecase0001/ses-0001/perfusion-maps/"
        "sub-strokecase0001_ses-0001_space-ncct_cbf.nii.gz": "cbf",
        "train/derivatives/sub-strokecase0001/ses-0001/perfusion-maps/"
        "sub-strokecase0001_ses-0001_space-ncct_cbv.nii.gz": "cbv",
        "train/derivatives/sub-strokecase0001/ses-0001/perfusion-maps/"
        "sub-strokecase0001_ses-0001_space-ncct_mtt.nii.gz": "mtt",
        "train/derivatives/sub-strokecase0001/ses-0002/"
        "sub-strokecase0001_ses-0002_lesion-msk.nii.gz": "lesion_mask",
        "train/phenotype/ses-0001/"
        "sub-strokecase0001_ses-0001_demographic_baseline.csv": "baseline_csv",
    }
    for path, expected in samples.items():
        assert MODULE.classify_member(path) == expected


def test_extractor_excludes_followup_mri_outcomes_and_raw_ctp():
    excluded = (
        (
            "train/rawdata/sub-strokecase0001/ses-0001/"
            "sub-strokecase0001_ses-0001_ctp.nii.gz"
        ),
        (
            "train/rawdata/sub-strokecase0001/ses-0002/"
            "sub-strokecase0001_ses-0002_dwi.nii.gz"
        ),
        (
            "train/rawdata/sub-strokecase0001/ses-0002/"
            "sub-strokecase0001_ses-0002_adc.nii.gz"
        ),
        (
            "train/phenotype/ses-0002/"
            "sub-strokecase0001_ses-0002_outcome.csv"
        ),
    )
    assert all(MODULE.classify_member(path) is None for path in excluded)
