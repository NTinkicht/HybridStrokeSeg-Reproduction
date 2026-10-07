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
    "ISLES22Case",
    "ISLES24Case",
    "SISSCase",
    "case_geometry_report",
    "case_geometry_report_isles2024",
    "discover_isles2022_cases",
    "discover_isles2024_cases",
    "discover_siss_cases",
    "geometry_equal",
    "geometry_signature",
    "load_volume",
    "summarize_geometry",
    "summarize_isles2024",
]

from .isles2024 import (
    ISLES24Case,
    case_geometry_report_isles2024,
    discover_isles2024_cases,
    summarize_isles2024,
)
