"""ISLES'24 dataset discovery and geometry-audit helpers.

ISLES'24 is a longitudinal final-infarct prediction benchmark. Valid model
inputs are pre-interventional acute CT/CTA/CTP-derived images and, in separate
experiments, baseline clinical variables. Follow-up MRI is target-generation
context and must never be exposed to the prediction model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .isles2022 import geometry_equal, geometry_signature

SUPPORTED_ACUTE_CHANNELS = ("ncct", "cta", "tmax", "cbf", "cbv", "mtt")
ISLES2024_GEOMETRY_ATOL = 2e-5


@dataclass(frozen=True)
class ISLES24Case:
    """Paths for one public ISLES'24 training case."""

    subject_id: str
    ncct: Path
    lesion_mask_ncct: Path
    cta_ncct: Path | None = None
    tmax_ncct: Path | None = None
    cbf_ncct: Path | None = None
    cbv_ncct: Path | None = None
    mtt_ncct: Path | None = None
    baseline_csv: Path | None = None
    outcome_csv: Path | None = None
    dwi_followup: Path | None = None
    adc_followup: Path | None = None
    lvo_mask_ncct: Path | None = None
    cow_mask_ncct: Path | None = None
    ctp_native: Path | None = None
    ctp_ncct: Path | None = None

    @property
    def case_id(self) -> str:
        return self.subject_id

    def acute_model_inputs(self) -> dict[str, Path]:
        """Return only available pre-interventional acute imaging channels."""
        mapping = {
            "ncct": self.ncct,
            "cta": self.cta_ncct,
            "tmax": self.tmax_ncct,
            "cbf": self.cbf_ncct,
            "cbv": self.cbv_ncct,
            "mtt": self.mtt_ncct,
        }
        return {name: path for name, path in mapping.items() if path is not None}


def resolve_isles2024_root(root: str | Path) -> Path:
    """Resolve a dataset root even when the archive adds one wrapper directory."""
    candidate = Path(root)
    if not candidate.exists():
        raise FileNotFoundError(candidate)

    def has_core_dirs(path: Path) -> bool:
        has_raw = (path / "rawdata").is_dir() or (path / "raw_data").is_dir()
        return has_raw and (path / "derivatives").is_dir()

    if has_core_dirs(candidate):
        return candidate

    matches: list[Path] = []
    for path in candidate.rglob("*"):
        if path.is_dir() and has_core_dirs(path):
            matches.append(path)
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find an ISLES'24 root with rawdata/raw_data and derivatives below {candidate}"
        )
    raise ValueError(f"Multiple candidate ISLES'24 roots found below {candidate}: {matches}")


def _find_dataset_dir(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    joined = ", ".join(names)
    raise FileNotFoundError(f"Expected one of [{joined}] below {root}")


def _find_session_dir(root: Path, phase: int, *, required: bool = True) -> Path:
    """Resolve BIDS session labels used by ISLES'24 releases.

    The downloaded Zenodo v7 archive uses ses-01/ses-02. Older project
    assumptions used ses-0001/ses-0002, so both are accepted for robustness.
    """
    names = (
        f"ses-{phase:02d}",
        f"ses-{phase:04d}",
        f"ses-{phase}",
    )
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    if required:
        joined = ", ".join(names)
        raise FileNotFoundError(f"Expected one of [{joined}] below {root}")
    return root / names[0]


def _single_recursive_match(directory: Path, patterns: Iterable[str]) -> Path:
    matches: set[Path] = set()
    pattern_list = tuple(patterns)
    for pattern in pattern_list:
        matches.update(directory.rglob(pattern))
    ordered = sorted(matches)
    if len(ordered) != 1:
        raise ValueError(
            f"Expected exactly one of {pattern_list!r} below {directory}, found {len(ordered)}"
        )
    return ordered[0]


def _optional_recursive_match(directory: Path, patterns: Iterable[str]) -> Path | None:
    if not directory.is_dir():
        return None
    matches: set[Path] = set()
    pattern_list = tuple(patterns)
    for pattern in pattern_list:
        matches.update(directory.rglob(pattern))
    ordered = sorted(matches)
    if len(ordered) > 1:
        raise ValueError(
            f"Expected at most one of {pattern_list!r} below {directory}, found {len(ordered)}"
        )
    return ordered[0] if ordered else None


def _required_or_optional(
    directory: Path,
    patterns: tuple[str, ...],
    *,
    required: bool,
) -> Path | None:
    if required:
        return _single_recursive_match(directory, patterns)
    return _optional_recursive_match(directory, patterns)


def discover_isles2024_cases(
    root: str | Path,
    *,
    required_channels: Iterable[str] = SUPPORTED_ACUTE_CHANNELS,
) -> list[ISLES24Case]:
    """Discover public ISLES'24 training cases.

    The official BIDS-style structure uses rawdata plus derivatives.
    raw_data is accepted as a compatibility alias. required_channels allows
    selective extraction workflows such as ("ncct",) for the first baseline.
    """
    required = tuple(str(channel).lower() for channel in required_channels)
    unknown = sorted(set(required) - set(SUPPORTED_ACUTE_CHANNELS))
    if unknown:
        raise ValueError(f"Unsupported ISLES'24 acute channels: {unknown}")
    if "ncct" not in required:
        required = ("ncct",) + required

    dataset_root = resolve_isles2024_root(root)
    raw_root = _find_dataset_dir(dataset_root, "rawdata", "raw_data")
    derivatives_root = _find_dataset_dir(dataset_root, "derivatives")
    phenotype_root = dataset_root / "phenotype"

    cases: list[ISLES24Case] = []
    subject_dirs = sorted(path for path in raw_root.glob("sub-*") if path.is_dir())
    if not subject_dirs:
        raise ValueError(f"No ISLES'24 subject directories found below {raw_root}")

    for raw_subject in subject_dirs:
        subject_id = raw_subject.name
        acute_raw = _find_session_dir(raw_subject, 1)
        followup_raw = _find_session_dir(raw_subject, 2, required=False)
        derivative_subject = derivatives_root / subject_id
        acute_derivative = _find_session_dir(
            derivative_subject,
            1,
            required=any(channel != "ncct" for channel in required),
        )
        followup_derivative = _find_session_dir(derivative_subject, 2)

        if not followup_derivative.is_dir():
            raise FileNotFoundError(
                "Missing follow-up derivative session with final infarct mask: "
                f"{followup_derivative}"
            )

        baseline_csv = None
        outcome_csv = None
        if phenotype_root.is_dir():
            baseline_csv = _optional_recursive_match(
                _find_session_dir(phenotype_root, 1, required=False),
                (f"*{subject_id}*demographic_baseline.csv",),
            )
            outcome_csv = _optional_recursive_match(
                _find_session_dir(phenotype_root, 2, required=False),
                (f"*{subject_id}*outcome.csv",),
            )

        cases.append(
            ISLES24Case(
                subject_id=subject_id,
                ncct=_single_recursive_match(acute_raw, ("*_ncct.nii.gz",)),
                lesion_mask_ncct=_single_recursive_match(
                    followup_derivative,
                    ("*_lesion-msk.nii.gz", "*space-ncct*lesion-msk.nii.gz"),
                ),
                cta_ncct=_required_or_optional(
                    acute_derivative,
                    ("*space-ncct_cta.nii.gz",),
                    required="cta" in required,
                ),
                tmax_ncct=_required_or_optional(
                    acute_derivative,
                    ("*space-ncct_tmax.nii.gz",),
                    required="tmax" in required,
                ),
                cbf_ncct=_required_or_optional(
                    acute_derivative,
                    ("*space-ncct_cbf.nii.gz",),
                    required="cbf" in required,
                ),
                cbv_ncct=_required_or_optional(
                    acute_derivative,
                    ("*space-ncct_cbv.nii.gz",),
                    required="cbv" in required,
                ),
                mtt_ncct=_required_or_optional(
                    acute_derivative,
                    ("*space-ncct_mtt.nii.gz",),
                    required="mtt" in required,
                ),
                baseline_csv=baseline_csv,
                outcome_csv=outcome_csv,
                dwi_followup=_optional_recursive_match(
                    followup_raw,
                    ("*_dwi.nii.gz",),
                ),
                adc_followup=_optional_recursive_match(
                    followup_raw,
                    ("*_adc.nii.gz",),
                ),
                lvo_mask_ncct=_optional_recursive_match(
                    acute_derivative,
                    ("*space-ncct_lvo-msk.nii.gz",),
                ),
                cow_mask_ncct=_optional_recursive_match(
                    acute_derivative,
                    ("*space-ncct_cow-msk.nii.gz",),
                ),
                ctp_native=_optional_recursive_match(acute_raw, ("*_ctp.nii.gz",)),
                ctp_ncct=_optional_recursive_match(
                    acute_derivative,
                    ("*space-ncct_ctp.nii.gz",),
                ),
            )
        )

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate ISLES'24 subject identifiers discovered")
    return cases


def geometry_equal_isles2024(
    first: dict[str, object],
    second: dict[str, object],
) -> bool:
    """Compare ISLES'24 grids with tolerance for NIfTI qform/sform rounding.

    One released case stores the NCCT transform in the sform and the derived
    lesion-mask transform in the qform. Their physical grids are equivalent,
    but float32 header round-tripping produces a maximum direction-cosine
    difference of about 1.6e-5. A 2e-5 absolute tolerance accepts that benign
    representation difference while still rejecting meaningful grid changes.
    """
    return geometry_equal(first, second, atol=ISLES2024_GEOMETRY_ATOL)


def case_geometry_report_isles2024(case: ISLES24Case) -> dict[str, object]:
    """Audit available acute inputs and the final-infarct mask against NCCT."""
    reference = geometry_signature(case.ncct)
    inputs = case.acute_model_inputs()
    signatures = {name: geometry_signature(path) for name, path in inputs.items()}
    mask_signature = geometry_signature(case.lesion_mask_ncct)
    return {
        "case_id": case.case_id,
        "matches_ncct": {
            name: geometry_equal_isles2024(reference, signature)
            for name, signature in signatures.items()
        }
        | {"lesion_mask": geometry_equal_isles2024(reference, mask_signature)},
        "ncct_geometry": reference,
        "geometry": signatures | {"lesion_mask": mask_signature},
        "has_baseline_csv": case.baseline_csv is not None,
        "has_outcome_csv": case.outcome_csv is not None,
        "has_dwi_followup": case.dwi_followup is not None,
        "has_adc_followup": case.adc_followup is not None,
        "has_lvo_mask": case.lvo_mask_ncct is not None,
        "has_cow_mask": case.cow_mask_ncct is not None,
        "has_native_ctp": case.ctp_native is not None,
        "has_registered_ctp": case.ctp_ncct is not None,
    }


def summarize_isles2024(cases: list[ISLES24Case]) -> dict[str, int]:
    """Summarize available data and NCCT-grid compatibility."""
    summary = {
        "cases": len(cases),
        "ncct_present": len(cases),
        "cta_present": 0,
        "tmax_present": 0,
        "cbf_present": 0,
        "cbv_present": 0,
        "mtt_present": 0,
        "cta_matches_ncct": 0,
        "tmax_matches_ncct": 0,
        "cbf_matches_ncct": 0,
        "cbv_matches_ncct": 0,
        "mtt_matches_ncct": 0,
        "lesion_mask_matches_ncct": 0,
        "baseline_csv_present": 0,
        "outcome_csv_present": 0,
        "dwi_followup_present": 0,
        "adc_followup_present": 0,
        "lvo_mask_present": 0,
        "cow_mask_present": 0,
        "native_ctp_present": 0,
        "registered_ctp_present": 0,
    }
    for case in cases:
        report = case_geometry_report_isles2024(case)
        matches = report["matches_ncct"]
        inputs = case.acute_model_inputs()
        for name in ("cta", "tmax", "cbf", "cbv", "mtt"):
            present = name in inputs
            summary[f"{name}_present"] += int(present)
            if present:
                summary[f"{name}_matches_ncct"] += int(matches[name])
        summary["lesion_mask_matches_ncct"] += int(matches["lesion_mask"])
        for field in (
            "baseline_csv",
            "outcome_csv",
            "dwi_followup",
            "adc_followup",
            "lvo_mask",
            "cow_mask",
            "native_ctp",
            "registered_ctp",
        ):
            summary[f"{field}_present"] += int(report[f"has_{field}"])
    return summary
