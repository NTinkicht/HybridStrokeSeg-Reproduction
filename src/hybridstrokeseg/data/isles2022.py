"""ISLES 2022 BIDS-style dataset discovery and integrity helpers.

The public ISLES 2022 training release contains 250 expert-annotated cases with
DWI, ADC and FLAIR images in ``rawdata`` and lesion masks in ``derivatives``.
Images are released in native space, so this module deliberately *does not*
assume that all modalities are already registered to a common grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .isles2015 import load_volume


@dataclass(frozen=True)
class ISLES22Case:
    """Paths belonging to one labeled ISLES 2022 training case."""

    subject_id: str
    session_id: str
    adc: Path
    dwi: Path
    flair: Path
    mask: Path

    @property
    def case_id(self) -> str:
        return f"{self.subject_id}_{self.session_id}"

    def images(self) -> dict[str, Path]:
        return {"adc": self.adc, "dwi": self.dwi, "flair": self.flair, "mask": self.mask}


def _single_match(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '*{suffix}' in {directory}, found {len(matches)}"
        )
    return matches[0]


def discover_isles2022_cases(root: str | Path) -> list[ISLES22Case]:
    """Discover complete labeled ISLES 2022 cases below an extracted dataset root.

    Expected organizer layout::

        rawdata/sub-strokecaseXXXX/ses-0001/*_{adc,dwi,flair}.nii.gz
        derivatives/sub-strokecaseXXXX/ses-0001/*_msk.nii.gz
    """
    dataset_root = Path(root)
    raw_root = dataset_root / "rawdata"
    derivatives_root = dataset_root / "derivatives"

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing ISLES 2022 rawdata directory: {raw_root}")
    if not derivatives_root.is_dir():
        raise FileNotFoundError(f"Missing ISLES 2022 derivatives directory: {derivatives_root}")

    cases: list[ISLES22Case] = []
    for subject_dir in sorted(path for path in raw_root.glob("sub-strokecase*") if path.is_dir()):
        sessions = sorted(path for path in subject_dir.glob("ses-*") if path.is_dir())
        if not sessions:
            raise ValueError(f"No session directory found for {subject_dir.name}")

        for session_dir in sessions:
            subject_id = subject_dir.name
            session_id = session_dir.name
            derivative_session = derivatives_root / subject_id / session_id
            if not derivative_session.is_dir():
                raise FileNotFoundError(
                    f"Missing derivative session for {subject_id}/{session_id}: {derivative_session}"
                )

            cases.append(
                ISLES22Case(
                    subject_id=subject_id,
                    session_id=session_id,
                    adc=_single_match(session_dir, "_adc.nii.gz"),
                    dwi=_single_match(session_dir, "_dwi.nii.gz"),
                    flair=_single_match(session_dir, "_flair.nii.gz"),
                    mask=_single_match(derivative_session, "_msk.nii.gz"),
                )
            )

    if not cases:
        raise ValueError(f"No ISLES 2022 cases found below {dataset_root}")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate ISLES 2022 case identifiers discovered")
    return cases


def geometry_signature(path: str | Path) -> dict[str, object]:
    """Return spatial metadata used to compare modality grids."""
    _, metadata = load_volume(path)
    return {
        "size_xyz": metadata["size_xyz"],
        "spacing_xyz": metadata["spacing_xyz"],
        "origin_xyz": metadata["origin_xyz"],
        "direction": metadata["direction"],
    }


def geometry_equal(
    first: dict[str, object],
    second: dict[str, object],
    *,
    atol: float = 1e-5,
) -> bool:
    """Return whether two geometry signatures describe the same voxel grid."""
    if first["size_xyz"] != second["size_xyz"]:
        return False
    for key in ("spacing_xyz", "origin_xyz", "direction"):
        if not np.allclose(first[key], second[key], atol=atol, rtol=0):
            return False
    return True


def case_geometry_report(case: ISLES22Case) -> dict[str, object]:
    """Audit whether each modality/mask is already on the DWI grid."""
    signatures = {name: geometry_signature(path) for name, path in case.images().items()}
    reference = signatures["dwi"]
    return {
        "case_id": case.case_id,
        "matches_dwi": {
            name: geometry_equal(reference, signature)
            for name, signature in signatures.items()
        },
        "geometry": signatures,
    }


def summarize_geometry(cases: Iterable[ISLES22Case]) -> dict[str, int]:
    """Count how many cases have each image already matching the DWI grid."""
    totals = {"cases": 0, "adc_matches_dwi": 0, "flair_matches_dwi": 0, "mask_matches_dwi": 0}
    for case in cases:
        report = case_geometry_report(case)
        matches = report["matches_dwi"]
        totals["cases"] += 1
        totals["adc_matches_dwi"] += int(matches["adc"])
        totals["flair_matches_dwi"] += int(matches["flair"])
        totals["mask_matches_dwi"] += int(matches["mask"])
    return totals
