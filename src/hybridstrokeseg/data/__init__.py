"""Dataset access helpers."""

from .isles2015 import SISSCase, discover_siss_cases, load_volume
from .isles2022 import (
    ISLES22Case,
    case_geometry_report,
    discover_isles2022_cases,
    geometry_equal,
    geometry_signature,
    summarize_geometry,
)

__all__ = [
    "SISSCase",
    "discover_siss_cases",
    "load_volume",
    "ISLES22Case",
    "discover_isles2022_cases",
    "geometry_signature",
    "geometry_equal",
    "case_geometry_report",
    "summarize_geometry",
]
