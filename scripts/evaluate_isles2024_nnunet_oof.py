#!/usr/bin/env python3
"""Evaluate the five-fold ISLES'24 NCCT nnU-Net OOF predictions.

This script is intentionally CPU-only. It verifies the deterministic OOF
cohort, checks prediction/reference geometry, computes the four public
ISLES'24 case-level metrics, writes patient-level results to the chosen output
directory, and emits aggregate/fold summaries with deterministic bootstrap CIs.

The challenge notes that official test evaluation additionally masks lesion
masks using an NCCT-derived brain mask. If organizer-compatible brain masks are
available, pass --brain-mask-root. Without them, outputs are labelled as local
OOF metrics using the official public metric implementation but without that
organizer masking step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hybridstrokeseg.isles2024_eval import (
    OFFICIAL_EVAL_SOURCE,
    bootstrap_mean_ci,
    evaluate_case,
    load_binary_mask,
    load_splits,
    sha256_file,
    validate_oof_splits,
)
from hybridstrokeseg.isles2024_oof_progress import (
    input_signature,
    load_case,
    record_progress,
    save_case,
)


METRIC_FIELDS = (
    "dice",
    "absolute_volume_difference_ml",
    "lesion_f1",
    "absolute_lesion_count_difference",
)


def dataset_dir(root: Path, dataset_id: int) -> Path:
    matches = sorted(root.glob(f"Dataset{dataset_id:03d}_*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Dataset{dataset_id:03d}_* below {root}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def trainer_dir(results_dataset: Path, configuration: str) -> Path:
    matches = sorted(results_dataset.glob(f"*__*__{configuration}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one trainer directory for {configuration} below "
            f"{results_dataset}, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_nifti(path: Path):
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "nibabel is required. Install with: pip install -e '.[evaluation]'"
        ) from exc
    return nib.load(str(path))


def assert_geometry(reference_img, prediction_img, *, case_id: str) -> None:
    if reference_img.shape != prediction_img.shape:
        raise ValueError(
            f"{case_id}: shape mismatch reference={reference_img.shape}, "
            f"prediction={prediction_img.shape}"
        )
    if not np.allclose(reference_img.affine, prediction_img.affine, atol=1e-4):
        max_delta = float(
            np.max(np.abs(np.asarray(reference_img.affine) - np.asarray(prediction_img.affine)))
        )
        raise ValueError(
            f"{case_id}: affine mismatch (max absolute delta {max_delta:.6g})"
        )


def find_brain_mask(root: Path | None, case_id: str) -> Path | None:
    if root is None:
        return None
    candidates = [
        root / f"{case_id}.nii.gz",
        root / f"{case_id}_brain-mask.nii.gz",
        root / f"{case_id}_brain_mask.nii.gz",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{case_id}: expected one brain mask in {root}; candidates={candidates}"
        )
    return matches[0]


def summary_stats(values: list[float], *, bootstrap_repetitions: int, seed: int) -> dict:
    array = np.asarray(values, dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(
        array,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "bootstrap_mean_95ci": [ci_low, ci_high],
    }


def read_nnunet_case_dice(summary_path: Path) -> dict[str, float]:
    if not summary_path.is_file():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: dict[str, float] = {}
    for item in payload.get("metric_per_case", []):
        prediction = Path(item["prediction_file"]).name
        case_id = prediction.removesuffix(".nii.gz")
        rows[case_id] = float(item["metrics"]["1"]["Dice"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, default=504)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brain-mask-root", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--hash-checkpoints", action="store_true")
    args = parser.parse_args()

    preprocessed_dataset = dataset_dir(args.preprocessed_root, args.dataset_id)
    results_dataset = dataset_dir(args.results_root, args.dataset_id)
    run_dir = trainer_dir(results_dataset, args.configuration)
    split_path = preprocessed_dataset / "splits_final.json"
    reference_root = preprocessed_dataset / "gt_segmentations"

    splits = load_splits(split_path)
    expected_cases = {case for split in splits for case in split["train"]}
    expected_cases.update(case for split in splits for case in split["val"])
    oof_cases = validate_oof_splits(splits, expected_cases)
    if len(oof_cases) != 149:
        raise RuntimeError(
            f"Expected the ISLES'24 public cohort to contain 149 OOF cases; "
            f"found {len(oof_cases)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    checkpoint_manifest: dict[str, dict] = {}
    nnunet_dice_deltas: list[float] = []

    progress_path = args.output_dir / "ncct_oof_progress.json"
    checkpoint_root = args.output_dir / "case_checkpoints"
    record_progress(
        progress_path, completed=0, total=len(oof_cases), fold=None,
        case_id=None, status="running", detail="Cohort preflight passed",
    )
    print("OOF preflight passed; beginning 5 folds", flush=True)
    for fold, split in enumerate(splits):
        print(f"Evaluating fold {fold}, {len(split['val'])} cases", flush=True)
        fold_dir = run_dir / f"fold_{fold}"
        checkpoint = fold_dir / "checkpoint_final.pth"
        validation_dir = fold_dir / "validation"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Fold {fold} is incomplete: {checkpoint}")
        if not validation_dir.is_dir():
            raise FileNotFoundError(validation_dir)

        checkpoint_entry = {
            "path": str(checkpoint),
            "size_bytes": int(checkpoint.stat().st_size),
        }
        if args.hash_checkpoints:
            checkpoint_entry["sha256"] = sha256_file(checkpoint)
        checkpoint_manifest[str(fold)] = checkpoint_entry

        nnunet_case_dice = read_nnunet_case_dice(validation_dir / "summary.json")
        expected_val = list(split["val"])
        prediction_cases = sorted(
            path.name.removesuffix(".nii.gz")
            for path in validation_dir.glob("*.nii.gz")
        )
        if sorted(expected_val) != prediction_cases:
            missing = sorted(set(expected_val) - set(prediction_cases))
            extra = sorted(set(prediction_cases) - set(expected_val))
            raise RuntimeError(
                f"Fold {fold} validation prediction mismatch. "
                f"Missing={missing}, extra={extra}"
            )

        for case_id in expected_val:
            prediction_path = validation_dir / f"{case_id}.nii.gz"
            reference_path = reference_root / f"{case_id}.nii.gz"
            cache_path = checkpoint_root / f"{case_id}.json"
            try:
                brain_mask_path = find_brain_mask(args.brain_mask_root, case_id)
                signature = input_signature(
                    case_id, fold, reference_path, prediction_path, brain_mask_path
                )
                row = load_case(cache_path, signature)
                reused = row is not None
                if row is None:
                    reference_img = load_nifti(reference_path)
                    prediction_img = load_nifti(prediction_path)
                    assert_geometry(reference_img, prediction_img, case_id=case_id)

                    reference = load_binary_mask(reference_img)
                    prediction = load_binary_mask(prediction_img)
                    zooms = reference_img.header.get_zooms()[:3]
                    voxel_volume_ml = float(np.prod(np.asarray(zooms, dtype=float)) / 1000.0)

                    brain_mask = None
                    if brain_mask_path is not None:
                        brain_img = load_nifti(brain_mask_path)
                        assert_geometry(
                            reference_img, brain_img, case_id=f"{case_id}/brain-mask"
                        )
                        brain_mask = load_binary_mask(brain_img)

                    metrics = evaluate_case(
                        case_id=case_id,
                        fold=fold,
                        reference=reference,
                        prediction=prediction,
                        voxel_volume_ml=voxel_volume_ml,
                        brain_mask=brain_mask,
                    )
                    row = asdict(metrics)
                    row["voxel_volume_ml"] = voxel_volume_ml
                    row["prediction_path"] = str(prediction_path)
                    row["reference_path"] = str(reference_path)
                    row["brain_mask_path"] = str(brain_mask_path) if brain_mask_path else ""

                if case_id in nnunet_case_dice:
                    delta = abs(float(row["dice"]) - nnunet_case_dice[case_id])
                    nnunet_dice_deltas.append(delta)
                    if delta > 1e-6:
                        raise RuntimeError(
                            f"{case_id}: public evaluator Dice {row['dice']} "
                            f"does not match nnU-Net Dice {nnunet_case_dice[case_id]} "
                            f"(delta={delta})"
                        )
                if not reused:
                    save_case(cache_path, signature, row)

                rows.append(row)
                done = len(rows)
                percent = 100.0 * done / len(oof_cases)
                record_progress(
                    progress_path, completed=done, total=len(oof_cases), fold=fold,
                    case_id=case_id, status="running",
                    detail="resumed" if reused else "computed",
                )
                print(
                    f"[{done:03d}/{len(oof_cases)}] {percent:5.1f}% "
                    f"fold={fold} case={case_id} "
                    f"{'RESUMED' if reused else 'COMPUTED'} "
                    f"Dice={float(row['dice']):.4f}",
                    flush=True,
                )
            except Exception as exc:
                record_progress(
                    progress_path, completed=len(rows), total=len(oof_cases),
                    fold=fold, case_id=case_id, status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                raise

    case_ids = [row["case_id"] for row in rows]
    if len(rows) != 149 or len(set(case_ids)) != 149:
        raise RuntimeError(
            f"Final OOF table must contain exactly 149 unique cases; "
            f"rows={len(rows)}, unique={len(set(case_ids))}"
        )

    csv_path = args.output_dir / "isles2024_ncct_oof_case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics_summary = {
        metric: summary_stats(
            [float(row[metric]) for row in rows],
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
        for metric in METRIC_FIELDS
    }

    fold_summary: dict[str, dict] = {}
    for fold in range(len(splits)):
        fold_rows = [row for row in rows if row["fold"] == fold]
        fold_summary[str(fold)] = {
            "n": len(fold_rows),
            "metrics": {
                metric: summary_stats(
                    [float(row[metric]) for row in fold_rows],
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    seed=args.bootstrap_seed + fold,
                )
                for metric in METRIC_FIELDS
            },
        }

    summary = {
        "dataset": preprocessed_dataset.name,
        "configuration": args.configuration,
        "n_oof_cases": len(rows),
        "unique_oof_cases": len(set(case_ids)),
        "fold_validation_counts": {
            str(fold): len(split["val"]) for fold, split in enumerate(splits)
        },
        "brain_mask_applied": args.brain_mask_root is not None,
        "challenge_metric_implementation": OFFICIAL_EVAL_SOURCE,
        "challenge_brain_mask_note": (
            "Organizer-compatible NCCT-derived brain masks were applied."
            if args.brain_mask_root is not None
            else (
                "No organizer-compatible NCCT-derived brain mask was supplied. "
                "Metrics use the public ISLES'24 metric implementation but are "
                "labelled local OOF metrics rather than exact hidden-test evaluation."
            )
        ),
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
        },
        "nnunet_dice_crosscheck": {
            "n_checked": len(nnunet_dice_deltas),
            "max_absolute_delta": (
                float(max(nnunet_dice_deltas)) if nnunet_dice_deltas else None
            ),
        },
        "metrics": metrics_summary,
        "folds": fold_summary,
        "checkpoints": checkpoint_manifest,
    }
    summary_path = args.output_dir / "isles2024_ncct_oof_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "dataset_id": args.dataset_id,
        "dataset_name": preprocessed_dataset.name,
        "configuration": args.configuration,
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
        "results_run_dir": str(run_dir),
        "output_csv": str(csv_path),
        "output_summary": str(summary_path),
        "brain_mask_root": str(args.brain_mask_root) if args.brain_mask_root else None,
        "official_eval_source": OFFICIAL_EVAL_SOURCE,
        "environment": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    manifest_path = args.output_dir / "isles2024_ncct_oof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"OOF cohort verified: {len(rows)} unique cases")
    print(f"Case-level metrics: {csv_path}")
    print(f"Aggregate summary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print(
        "Mean Dice: "
        f"{summary['metrics']['dice']['mean']:.6f} "
        f"(95% bootstrap CI "
        f"{summary['metrics']['dice']['bootstrap_mean_95ci'][0]:.6f}-"
        f"{summary['metrics']['dice']['bootstrap_mean_95ci'][1]:.6f})"
    )
    if args.brain_mask_root is None:
        print(
            "NOTE: organizer NCCT-derived brain masking was not applied; "
            "do not label these as exact hidden-test challenge metrics."
        )
    record_progress(
        progress_path, completed=len(rows), total=len(oof_cases), fold=None,
        case_id=None, status="complete", detail="All output files written",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
