"""ISLES 2015 SISS dataset discovery and loading.

The historical challenge archive has appeared in more than one directory layout.
Discovery therefore relies on modality tokens in path components rather than on
one hard-coded folder hierarchy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd")
REQUIRED_MODALITIES = ("flair", "t1", "t2", "dwi", "ot")


@dataclass(frozen=True)
class SISSCase:
    """Paths for one labeled ISLES 2015 SISS training case."""

    case_id: str
    flair: Path
    t1: Path
    t2: Path
    dwi: Path
    ot: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "flair": self.flair,
            "t1": self.t1,
            "t2": self.t2,
            "dwi": self.dwi,
            "ot": self.ot,
        }


def _is_image(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in IMAGE_SUFFIXES)


def _is_archive_metadata(path: Path) -> bool:
    """Return whether ``path`` is packaging metadata rather than medical-image data.

    The organizer re-archive contains macOS AppleDouble/resource-fork entries
    below ``__MACOSX`` with filenames beginning ``._``. They can retain medical
    image suffixes such as ``.mha`` but are not readable medical images.
    """
    return any(part == "__MACOSX" for part in path.parts) or path.name.startswith("._")


def _token_text(path: Path) -> str:
    return "/".join(part.lower() for part in path.parts[-4:])


def infer_modality(path: Path) -> str | None:
    """Infer SISS modality from a challenge-style path."""
    text = _token_text(path)

    if (
        "groundtruth" in text
        or "ground_truth" in text
        or re.search(r"(^|[./_-])ot([./_-]|$)", text)
    ):
        return "ot"
    if "flair" in text:
        return "flair"
    if re.search(r"(^|[./_-])dwi([./_-]|$)", text) or "mr_dwi" in text:
        return "dwi"
    if re.search(r"(^|[./_-])t2([./_-]|$)", text) or "mr_t2" in text:
        return "t2"
    if re.search(r"(^|[./_-])t1([./_-]|$)", text) or "mr_t1" in text:
        return "t1"
    return None


def _case_root_for(path: Path, dataset_root: Path, modality: str) -> Path:
    """Infer case root from common ISLES layouts."""
    parent = path.parent
    parent_modality = infer_modality(parent / f"dummy_{modality}.mha")
    if parent_modality == modality and parent != dataset_root:
        return parent.parent
    return parent


def _case_id(case_root: Path, dataset_root: Path) -> str:
    try:
        rel = case_root.resolve().relative_to(dataset_root.resolve())
        text = "__".join(rel.parts)
        return text or case_root.name
    except ValueError:
        return case_root.name


def discover_siss_cases(
    root: str | Path,
    *,
    require_complete: bool = True,
) -> list[SISSCase]:
    """Discover labeled SISS cases below ``root``.

    Incomplete groups are ignored by default so the same function can scan a
    full historical archive that may also contain unlabeled challenge material.
    macOS resource-fork metadata from the historical re-archive is ignored.
    """
    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    grouped: dict[Path, dict[str, Path]] = {}

    for path in sorted(
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and _is_image(p) and not _is_archive_metadata(p)
    ):
        modality = infer_modality(path)
        if modality is None:
            continue

        case_root = _case_root_for(path, dataset_root, modality)
        bucket = grouped.setdefault(case_root, {})
        if modality in bucket and bucket[modality] != path:
            raise ValueError(
                f"Duplicate {modality!r} images for case root {case_root}: "
                f"{bucket[modality]} and {path}"
            )
        bucket[modality] = path

    cases: list[SISSCase] = []
    incomplete: list[str] = []

    for case_root, paths in sorted(grouped.items(), key=lambda item: str(item[0])):
        missing = [m for m in REQUIRED_MODALITIES if m not in paths]
        if missing:
            incomplete.append(f"{case_root}: missing {', '.join(missing)}")
            continue
        cases.append(
            SISSCase(
                case_id=_case_id(case_root, dataset_root),
                flair=paths["flair"],
                t1=paths["t1"],
                t2=paths["t2"],
                dwi=paths["dwi"],
                ot=paths["ot"],
            )
        )

    if not cases and incomplete and not require_complete:
        raise ValueError("No complete SISS cases found.\n" + "\n".join(incomplete))

    return cases


def load_volume(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load a NIfTI/MHA/MHD image using SimpleITK.

    Returns the array in SimpleITK's ``(z, y, x)`` indexing order and selected
    spatial metadata required for reproducible downstream checks.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SimpleITK is required to load medical images") from exc

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    metadata = {
        "size_xyz": tuple(int(v) for v in image.GetSize()),
        "spacing_xyz": tuple(float(v) for v in image.GetSpacing()),
        "origin_xyz": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
        "pixel_id": image.GetPixelIDTypeAsString(),
    }
    return np.asarray(array), metadata


def assert_same_geometry(paths: Iterable[str | Path], *, atol: float = 1e-5) -> None:
    """Raise if a set of images do not share size/spacing/origin/direction."""
    records = [load_volume(path)[1] for path in paths]
    if not records:
        raise ValueError("No image paths supplied")

    first = records[0]
    for idx, current in enumerate(records[1:], start=1):
        if current["size_xyz"] != first["size_xyz"]:
            raise ValueError(f"Geometry mismatch at item {idx}: size differs")
        for key in ("spacing_xyz", "origin_xyz", "direction"):
            if not np.allclose(current[key], first[key], atol=atol, rtol=0):
                raise ValueError(f"Geometry mismatch at item {idx}: {key} differs")
