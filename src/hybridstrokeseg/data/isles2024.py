"""ISLES 2024 dataset discovery and geometry-audit helpers.

ISLES'24 is a longitudinal infarct-prediction benchmark. Model inputs are
pre-interventional acute CT/CTA/CTP-derived data and optional clinical
variables. Follow-up MRI is label-generation context and must never be exposed
to the prediction model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .isles2022 import geometry_equal, geometry_signature


@dataclass(frozen=True)
class ISLES24Case:
    """Paths for one complete public ISLES'24 training case."""

    subject_id: str
    ncct: Path
    cta_ncct: Path
    tmax_ncct: Path
    cbf_ncct: Path
    cbv_ncct: Path
    mtt_ncct: Path
    lesion_mask_ncct: Path
    dwi_followup_ncct: Path
    adc_followup_ncct: Path
    baseline_csv: Path | None = None
    outcome_csv: Path | None = None
    lvo_mask_ncct: Path | None = None
    cow_mask_ncct: Path | None = None
    ctp_native: Path | None = None
    ctp_ncct: Path | None = None

    @property
    def case_id(self) -> str:
        return self.subject_id

    def acute_model_inputs(self) -> dict[str, Path]:
        """Return imaging channels that are valid pre-interventional inputs."""
        return {
            "ncct": self.ncct,
            "cta": self.cta_ncct,
            "tmax": self.tmax_ncct,
            "cbf": self.cbf_ncct,
            "cbv": self.cbv_ncct,
            "mtt": self.mtt_ncct,
        }


def _find_dataset_dir(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    joined = ", ".join(names)
    raise FileNotFoundError(f"Expected one of [{joined}] below {root}")


def _single_recursive_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{pattern}' below {directory}, found {len(matches)}"
        )
    return matches[0]


def _optional_recursive_match(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.rglob(pattern))
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one '{pattern}' below {directory}, found {len(matches)}"
        )
    return matches[0] if matches else None


def discover_isles2024_cases(root: str | Path) -> list[ISLES24Case]:
    """Discover complete labeled public ISLES'24 training cases.

    The current public release is BIDS-like and provides native acute images in
    raw_data plus derivatives co-registered to NCCT. Some historical package
    versions used rawdata; both spellings are accepted.
    """
    dataset_root = Path(root)
    raw_root = _find_dataset_dir(dataset_root, "raw_data", "rawdata")
    derivatives_root = _find_dataset_dir(dataset_root, "derivatives")
    phenotype_root = dataset_root / "phenotype"

    cases: list[ISLES24Case] = []
    subject_dirs = sorted(path for path in raw_root.glob("sub-*") if path.is_dir())
    if not subject_dirs:
        raise ValueError(f"No ISLES'24 subject directories found below {raw_root}")

    for raw_subject in subject_dirs:
        subject_id = raw_subject.name
        acute_raw = raw_subject / "ses-0001"
        derivative_subject = derivatives_root / subject_id
        acute_derivative = derivative_subject / "ses-0001"
        followup_derivative = derivative_subject / "ses-0002"

        if not acute_raw.is_dir():
            raise FileNotFoundError(f"Missing acute raw session: {acute_raw}")
        if not acute_derivative.is_dir():
            raise FileNotFoundError(f"Missing acute derivative session: {acute_derivative}")
        if not followup_derivative.is_dir():
            raise FileNotFoundError(
                f"Missing follow-up derivative session: {followup_derivative}"
            )

        baseline_csv = None
        outcome_csv = None
        if phenotype_root.is_dir():
            baseline_csv = _optional_recursive_match(
                phenotype_root / "ses-0001",
                f"*{subject_id}*demographic_baseline.csv",
            )
            outcome_csv = _optional_recursive_match(
                phenotype_root / "ses-0002",
                f"*{subject_id}*outcome.csv",
            )

        cases.append(
            ISLES24Case(
                subject_id=subject_id,
                ncct=_single_recursive_match(acute_raw, "*_ncct.nii.gz"),
                cta_ncct=_single_recursive_match(
                    acute_derivative, "*space-ncct_cta.nii.gz"
                ),
                tmax_ncct=_single_recursive_match(
                    acute_derivative, "*space-ncct_tmax.nii.gz"
                ),
                cbf_ncct=_single_recursive_match(
                    acute_derivative, "*space-ncct_cbf.nii.gz"
                ),
                cbv_ncct=_single_recursive_match(
                    acute_derivative, "*space-ncct_cbv.nii.gz"
                ),
                mtt_ncct=_single_recursive_match(
                    acute_derivative, "*space-ncct_mtt.nii.gz"
                ),
                lesion_mask_ncct=_single_recursive_match(
                    followup_derivative, "*space-ncct_lesion-msk.nii.gz"
                ),
                dwi_followup_ncct=_single_recursive_match(
                    followup_derivative, "*space-ncct_dwi.nii.gz"
                ),
                adc_followup_ncct=_single_recursive_match(
                    followup_derivative, "*space-ncct_adc.nii.gz"
                ),
                baseline_csv=baseline_csv,
                outcome_csv=outcome_csv,
                lvo_mask_ncct=_optional_recursive_match(
                    acute_derivative, "*space-ncct_lvo-msk.nii.gz"
                ),
                cow_mask_ncct=_optional_recursive_match(
                    acute_derivative, "*space-ncct_cow-msk.nii.gz"
                ),
                ctp_native=_optional_recursive_match(acute_raw, "*_ctp.nii.gz"),
                ctp_ncct=_optional_recursive_match(
                    acute_derivative, "*space-ncct_ctp.nii.gz"
                ),
            )
        )

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate ISLES'24 subject identifiers discovered")
    return cases


def case_geometry_report_isles2024(case: ISLES24Case) -> dict[str, object]:
    """Audit whether acute inputs and final-infarct mask share the NCCT grid."""
    reference = geometry_signature(case.ncct)
    inputs = case.acute_model_inputs()
    signatures = {name: geometry_signature(path) for name, path in inputs.items()}
    mask_signature = geometry_signature(case.lesion_mask_ncct)
    return {
        "case_id": case.case_id,
        "matches_ncct": {
            name: geometry_equal(reference, signature)
            for name, signature in signatures.items()
        }
        | {"lesion_mask": geometry_equal(reference, mask_signature)},
        "ncct_geometry": reference,
        "geometry": signatures | {"lesion_mask": mask_signature},
        "has_baseline_csv": case.baseline_csv is not None,
        "has_outcome_csv": case.outcome_csv is not None,
        "has_lvo_mask": case.lvo_mask_ncct is not None,
        "has_cow_mask": case.cow_mask_ncct is not None,
        "has_native_ctp": case.ctp_native is not None,
        "has_registered_ctp": case.ctp_ncct is not None,
    }


def summarize_isles2024(cases: list[ISLES24Case]) -> dict[str, int]:
    """Summarize completeness and NCCT-grid compatibility."""
    summary = {
        "cases": len(cases),
        "cta_matches_ncct": 0,
        "tmax_matches_ncct": 0,
        "cbf_matches_ncct": 0,
        "cbv_matches_ncct": 0,
        "mtt_matches_ncct": 0,
        "lesion_mask_matches_ncct": 0,
        "baseline_csv_present": 0,
        "outcome_csv_present": 0,
        "lvo_mask_present": 0,
        "cow_mask_present": 0,
        "native_ctp_present": 0,
        "registered_ctp_present": 0,
    }
    for case in cases:
        report = case_geometry_report_isles2024(case)
        matches = report["matches_ncct"]
        for name in ("cta", "tmax", "cbf", "cbv", "mtt"):
            summary[f"{name}_matches_ncct"] += int(matches[name])
        summary["lesion_mask_matches_ncct"] += int(matches["lesion_mask"])
        for field in (
            "baseline_csv",
            "outcome_csv",
            "lvo_mask",
            "cow_mask",
            "native_ctp",
            "registered_ctp",
        ):
            summary[f"{field}_present"] += int(report[f"has_{field}"])
    return summary
