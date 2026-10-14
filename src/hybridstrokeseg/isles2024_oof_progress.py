"""Resumable, fail-closed checkpoints for local ISLES'24 OOF evaluation.

All case-level data stay in the user-selected private Drive results directory.
Only Python standard-library dependencies are used here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "isles2024-oof-case-v1"


def atomic_json(path: Path, document: dict) -> None:
    """Write complete JSON through an atomic replace, never a partial checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def input_signature(
    case_id: str,
    fold: int,
    reference: Path,
    prediction: Path,
    brain_mask: Path | None = None,
) -> dict:
    """Invalidate cache if any input path, timestamp, or file size changes."""
    def describe(path: Path) -> dict:
        stat = path.stat()
        return {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        "case_id": case_id,
        "fold": fold,
        "reference": describe(reference),
        "prediction": describe(prediction),
        "brain_mask": describe(brain_mask) if brain_mask is not None else None,
    }


def load_case(path: Path, signature: dict) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_VERSION:
        return None
    if payload.get("inputs") != signature:
        return None
    row = payload.get("metrics")
    if not isinstance(row, dict):
        raise TypeError(f"Invalid case checkpoint (metrics missing): {path}")
    if row.get("case_id") != signature["case_id"] or row.get("fold") != signature["fold"]:
        raise ValueError(f"Case checkpoint identity mismatch: {path}")
    return row


def save_case(path: Path, signature: dict, row: dict) -> None:
    if row["case_id"] != signature["case_id"] or row["fold"] != signature["fold"]:
        raise ValueError("Cannot checkpoint mismatched patient or fold")
    atomic_json(
        path,
        {"schema": SCHEMA_VERSION, "inputs": signature, "metrics": row},
    )


def record_progress(
    path: Path,
    *,
    completed: int,
    total: int,
    fold: int | None,
    case_id: str | None,
    status: str,
    detail: str = "",
) -> dict:
    if not 0 <= completed <= total or total <= 0:
        raise ValueError("Progress must be between 0 and total")
    payload = {
        "status": status,
        "completed_cases": completed,
        "total_cases": total,
        "percent": round(100.0 * completed / total, 2),
        "fold": fold,
        "last_case": case_id,
        "detail": detail,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(path, payload)
    return payload
