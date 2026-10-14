#!/usr/bin/env python3
"""Verify that two nnU-Net experiments use exactly the same patient folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalized_splits(path: Path) -> list[dict[str, tuple[str, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty split list in {path}")

    normalized: list[dict[str, tuple[str, ...]]] = []
    for index, split in enumerate(payload):
        if not isinstance(split, dict):
            raise TypeError(f"Split {index} in {path} is not an object")
        train = tuple(str(case) for case in split.get("train", []))
        val = tuple(str(case) for case in split.get("val", []))
        if not train or not val:
            raise ValueError(f"Split {index} in {path} must contain train and val IDs")
        normalized.append({"train": train, "val": val})
    return normalized


def assert_identical_splits(reference: Path, candidate: Path) -> None:
    ref = normalized_splits(reference)
    cand = normalized_splits(candidate)
    if len(ref) != len(cand):
        raise RuntimeError(
            f"Fold-count mismatch: reference={len(ref)}, candidate={len(cand)}"
        )

    for fold, (ref_split, cand_split) in enumerate(zip(ref, cand, strict=True)):
        for partition in ("train", "val"):
            ref_ids = ref_split[partition]
            cand_ids = cand_split[partition]
            if ref_ids != cand_ids:
                ref_set = set(ref_ids)
                cand_set = set(cand_ids)
                missing = sorted(ref_set - cand_set)
                extra = sorted(cand_set - ref_set)
                order_only = not missing and not extra
                raise RuntimeError(
                    f"Fold {fold} {partition} differs from the reference split. "
                    f"missing={missing}, extra={extra}, order_only={order_only}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    assert_identical_splits(args.reference, args.candidate)
    print("Split identity verified: every fold and patient assignment is identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
