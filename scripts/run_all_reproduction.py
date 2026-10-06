#!/usr/bin/env python3
"""Run the full historical reproduction/diagnostic pipeline with resume support.

This orchestrator is designed for ephemeral environments such as Google Colab.
Every stage writes directly under a caller-provided persistent output root. On a
later rerun, completed stages are skipped and multi-condition diagnostics are
asked to resume from their existing outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isles2015"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/full_pipeline"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def mark(progress_path: Path, progress: dict[str, object], stage: str, status: str) -> None:
    stages = progress.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise TypeError("progress['stages'] must be a dictionary")
    stages[stage] = {"status": status, "timestamp_utc": now_iso()}
    progress["updated_utc"] = now_iso()
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")


def completed(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "pipeline_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "purpose": "persistent, resumable clean-room reproduction pipeline",
            "scientific_guardrail": (
                "Sensitivity stages quantify uncertainty in underspecified manuscript details. "
                "Do not choose settings solely because they approach the published Dice."
            ),
            "created_utc": now_iso(),
            "stages": {},
        }

    py = sys.executable
    here = Path(__file__).resolve().parent

    stages: list[tuple[str, Path, list[str]]] = [
        (
            "paper_size",
            args.output_root / "paper_like" / "summary_metrics.csv",
            [
                py,
                str(here / "run_reproduction.py"),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(args.output_root / "paper_like"),
                "--seed",
                str(args.seed),
                "--protocol",
                "paper-like",
                "--target-per-class",
                "15000",
                "--models",
                "mlp",
                "svm",
            ],
        ),
        (
            "leakage_free_slice",
            args.output_root / "leakage_free_slice" / "summary_metrics.csv",
            [
                py,
                str(here / "run_reproduction.py"),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(args.output_root / "leakage_free_slice"),
                "--seed",
                str(args.seed),
                "--protocol",
                "leakage-free-slice",
                "--target-per-class",
                "15000",
                "--models",
                "mlp",
                "svm",
            ],
        ),
        (
            "directional_sensitivity",
            args.output_root / "sensitivity_quick" / "sweep_summary.csv",
            [
                py,
                str(here / "run_sensitivity_sweep.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root / "sensitivity_quick"),
                "--seed",
                str(args.seed),
                "--resume",
            ],
        ),
        (
            "split_sensitivity",
            args.output_root / "split_sensitivity" / "split_sensitivity.csv",
            [
                py,
                str(here / "run_split_sensitivity.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root / "split_sensitivity"),
                "--num-seeds",
                "20",
                "--resume",
            ],
        ),
        (
            "svm_sensitivity",
            args.output_root / "svm_sensitivity" / "svm_sensitivity.csv",
            [
                py,
                str(here / "run_svm_sensitivity.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root / "svm_sensitivity"),
                "--seed",
                str(args.seed),
                "--resume",
            ],
        ),
        (
            "morphology_sensitivity",
            args.output_root / "morphology_sensitivity" / "morphology_sensitivity.csv",
            [
                py,
                str(here / "run_morphology_sensitivity.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root / "morphology_sensitivity"),
                "--seed",
                str(args.seed),
                "--resume",
            ],
        ),
        (
            "voxel_leakage_diagnostic",
            args.output_root / "voxel_leakage" / "voxel_leakage_summary.csv",
            [
                py,
                str(here / "run_voxel_leakage_diagnostic.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.output_root / "voxel_leakage"),
                "--seed",
                str(args.seed),
                "--resume",
            ],
        ),
    ]

    for index, (stage, final_artifact, command) in enumerate(stages, start=1):
        print(f"\n=== Stage {index}/{len(stages)}: {stage} ===", flush=True)
        if args.resume and completed(final_artifact):
            print(f"Skipping completed stage; found {final_artifact}", flush=True)
            mark(progress_path, progress, stage, "completed-existing")
            continue
        mark(progress_path, progress, stage, "running")
        try:
            run(command)
        except subprocess.CalledProcessError:
            mark(progress_path, progress, stage, "failed")
            raise
        if not completed(final_artifact):
            mark(progress_path, progress, stage, "failed-missing-final-artifact")
            raise RuntimeError(f"Stage {stage} finished without expected artifact: {final_artifact}")
        mark(progress_path, progress, stage, "completed")

    progress["status"] = "completed"
    progress["completed_utc"] = now_iso()
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nFull pipeline completed. Persistent outputs: {args.output_root}")
    print(f"Progress ledger: {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
