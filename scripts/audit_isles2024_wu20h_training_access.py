#!/usr/bin/env python3
"""WU20H: synthetic training-I/O trace contract. NO trainer or patient data access.

These checks are a rehearsal only. They cannot certify real nnU-Net trainer hooks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "WU20H-training-IO-synthetic-audit-v1"
CHECKPOINT_EPOCH = 1000
ALLOWED_EVENTS = {
    "train_image_open", "train_label_open", "train_step", "checkpoint_final_written",
    "outer_inference_image_open", "predictions_sealed", "outer_evaluation_label_open", "archive_receipt",
}


def inspect_trace(events: list[dict], train_ids: set[str], outer_ids: set[str]) -> dict:
    """Check no outside labels reachable BEFORE sealed final checkpoint and archive."""
    if not train_ids or not outer_ids or train_ids & outer_ids:
        raise ValueError("Invalid outer train/heldout partition")
    if len(train_ids | outer_ids) != 149:
        raise ValueError("Expected 149 frozen patient identities")
    if not isinstance(events, list) or not events:
        raise ValueError("Empty training trace cannot be audited")
    final_written = False
    archived = False
    train_seen: set[str] = set()
    inference_seen: set[str] = set()
    predictions_sealed = False
    last_epoch = 0
    for i, event in enumerate(events):
        kind = event.get("type")
        case = event.get("case_id")
        if kind not in ALLOWED_EVENTS:
            raise ValueError(f"Unexpected or prohibited trace event at {i}: {kind}")
        if kind in {"train_image_open", "train_label_open"}:
            if final_written or case not in train_ids:
                raise ValueError("Trainer opened case outside frozen training set")
            if kind == "train_label_open":
                train_seen.add(case)
        elif kind == "train_step":
            if final_written or type(event.get("epoch")) is not int:
                raise ValueError("Training step after checkpoint or malformed epoch")
            if event["epoch"] != last_epoch + 1 or event["epoch"] > CHECKPOINT_EPOCH:
                raise ValueError("Nonsequential or incomplete predetermined epoch trace")
            last_epoch = event["epoch"]
        elif kind == "checkpoint_final_written":
            if final_written or event.get("epoch") != CHECKPOINT_EPOCH:
                raise ValueError("Only one fixed final-epoch checkpoint permitted")
            if train_seen != train_ids or last_epoch != CHECKPOINT_EPOCH:
                raise ValueError("All train labels and 1000 complete epochs required")
            final_written = True
        elif kind == "archive_receipt":
            if not final_written or archived:
                raise ValueError("Archive must follow final checkpoint exactly once")
            archived = True
        elif kind == "outer_inference_image_open":
            if not final_written or not archived or predictions_sealed or case not in outer_ids:
                raise ValueError("Heldout inference before fixed checkpoint or after prediction seal")
            inference_seen.add(case)
        elif kind == "predictions_sealed":
            if not archived or predictions_sealed or inference_seen != outer_ids:
                raise ValueError("Cannot seal incomplete heldout predictions")
            predictions_sealed = True
        elif kind == "outer_evaluation_label_open" and (not predictions_sealed or case not in outer_ids):
            raise ValueError("Heldout lesion label accessed before predictions were sealed")
    if not final_written or not archived or not predictions_sealed:
        raise ValueError("Final epoch, archive and predictions seal required")
    return {
        "schema": SCHEMA, "status": "SYNTHETIC_TRAINING_TRACE_CONTRACT_PASS",
        "events_checked": len(events), "train_cases_covered": len(train_seen),
        "synthetic_training_steps": last_epoch, "synthetic_heldout_images": len(inference_seen),
        "outer_heldout_count": len(outer_ids),
        "fixed_checkpoint_epoch": CHECKPOINT_EPOCH,
        "trained_model_executed": False,
        "actual_nnunet_trainer_hook_verified": False,
        "patient_data_accessed": False,
        "training_ready": False,
    }


def make_synthetic_trace(train: list[str], heldout: list[str]) -> list[dict]:
    """IDs are fixture strings only; this never builds an executable trainer."""
    events = [{"type": "train_label_open", "case_id": case} for case in sorted(train)]
    events.extend({"type": "train_step", "epoch": epoch} for epoch in range(1, CHECKPOINT_EPOCH + 1))
    events.extend([{"type": "checkpoint_final_written", "epoch": CHECKPOINT_EPOCH},
                   {"type": "archive_receipt"}])
    events.extend({"type": "outer_inference_image_open", "case_id": c} for c in sorted(heldout))
    events.append({"type": "predictions_sealed"})
    return events


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("synthetic", "train", "release", "trace_real"), default="synthetic")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode != "synthetic":
        raise PermissionError("Real patient training access tracing requires independent audited release")
    train = [f"fixture_training_{n:03d}" for n in range(119)]
    heldout = [f"fixture_holdout_{n:03d}" for n in range(30)]
    result = inspect_trace(make_synthetic_trace(train, heldout), set(train), set(heldout))
    if args.output.is_symlink():
        raise ValueError("Output symlink is prohibited")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Synthetic audit is immutable; do not overwrite")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
