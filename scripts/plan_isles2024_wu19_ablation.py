#!/usr/bin/env python3
"""WU-19: fail-closed, read-only protocol for ISLES2024 normalization ablations.

Derives immutable Dataset506/507 *design manifests*, NOT images or preprocessed
arrays. Reads frozen 149-case folds, per-channel Notebook17 and Notebook18
private audit CSVs, and Dataset505 plans. Never alters Dataset504/505.

IMPORTANT: This does NOT generate fold-calibrated training images. WU-20 must
implement label-free spatial support, train-only statistics, fold-specific
preprocessing AND matching inference before any GPU training can be authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

FROZEN_SPLIT_SHA = "664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027"
FOLD_SIZES = (30, 30, 30, 30, 29)
CHANNELS = ("NCCT", "TMAX", "CBF", "CBV", "MTT")
RAW_FIELDS = {"case_id", "channel", "n_nonfinite", "count_below_minus_10000", "min", "max"}
PRE_FIELDS = {"case_id", "channel", "pre_n_nonfinite", "pre_n_abs_gt_100", "pre_n_abs_gt_1000", "pre_near_constant"}
VERSION = "WU19-1.0"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_folds(path: Path) -> list[dict[str, list[str]]]:
    if digest_file(path) != FROZEN_SPLIT_SHA:
        raise ValueError("Frozen splits SHA256 mismatch; refusing to proceed")
    folds = json.loads(path.read_text(encoding="utf8"))
    if len(folds) != 5:
        raise ValueError("Expected exactly five frozen folds")
    ids = [case for fold in folds for case in fold["val"]]
    if len(ids) != 149 or len(set(ids)) != 149:
        raise ValueError("Expected 149 unique OOF cases")
    cases = set(ids)
    for i, fold in enumerate(folds):
        train, val = fold["train"], fold["val"]
        if len(val) != FOLD_SIZES[i]:
            raise ValueError(f"Fold {i} has incorrect validation count")
        if len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError(f"Duplicate patients in fold {i}")
        if set(train) & set(val) or set(train) != cases - set(val):
            raise ValueError(f"Train/validation leakage or missing patients in fold {i}")
    return folds


def audited_csv(path: Path, needed: set[str], cases: set[str]) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not needed <= set(reader.fieldnames or ()):
            raise ValueError(f"{path.name} missing required headers: {sorted(needed - set(reader.fieldnames or ())) }")
        rows = list(reader)
    keys = [(r["case_id"], r["channel"]) for r in rows]
    expected = {(case, channel) for case in cases for channel in CHANNELS}
    if len(rows) != 745 or len(set(keys)) != 745 or set(keys) != expected:
        raise ValueError(f"{path.name}: require 745 unique frozen patient-channel audit records")
    return dict(zip(keys, rows, strict=True))


def check_original_plans(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf8"))
    if obj.get("dataset_name") != "Dataset505_ISLES2024_NCCT_PERFUSION":
        raise ValueError("Wrong Dataset505 plans")
    config = obj["configurations"]["3d_fullres"]
    if config["normalization_schemes"] != ["ZScoreNormalization"] * 5:
        raise ValueError("Source experiment no longer matches frozen ZScoreNormalization plans")
    return {
        "original_configuration": "3d_fullres",
        "original_schemes": config["normalization_schemes"],
        "patch_size": config["patch_size"],
        "batch_size": config["batch_size"],
    }


def qc_summary(raw: dict, pre: dict, cases: set[str]) -> dict:
    result = {}
    for channel in CHANNELS:
        raw_nonfinite = sum(int(raw[(case, channel)]["n_nonfinite"]) > 0 for case in cases)
        raw_extreme = sum(int(raw[(case, channel)]["count_below_minus_10000"]) > 0 for case in cases)
        pre_nonfinite = sum(int(pre[(case, channel)]["pre_n_nonfinite"]) > 0 for case in cases)
        pre_gt100 = sum(int(pre[(case, channel)]["pre_n_abs_gt_100"]) > 0 for case in cases)
        pre_total_gt100 = sum(int(pre[(case, channel)]["pre_n_abs_gt_100"]) for case in cases)
        nearconstant = sum(pre[(case, channel)]["pre_near_constant"].lower() == "true" for case in cases)
        result[channel] = {
            "raw_patients_nonfinite": raw_nonfinite,
            "raw_patients_below_minus_10000": raw_extreme,
            "pre_patients_nonfinite": pre_nonfinite,
            "pre_patients_abs_z_gt_100": pre_gt100,
            "pre_voxels_abs_z_gt_100": pre_total_gt100,
            "pre_patients_near_constant": nearconstant,
        }
    return result


def plan_protocol(folds: list[dict], qc: dict, input_shas: dict) -> dict:
    return {
        "version": VERSION,
        "status": "DESIGN_ONLY_NOT_APPROVED_FOR_TRAINING",
        "training_ready": False,
        "reasons_not_training_ready": [
            "Training-only CT foreground statistics have not been fitted and frozen per fold",
            "Training-only perfusion robust statistics have not been fitted and frozen per fold",
            "Validated NCCT-derived, lesion-label-free support mask is not implemented",
            "Fold-isolated nnU-Net preprocessing, training and inference integration not qualified",
            "Policy for raw nonfinite perfusion voxels needs per-case anatomical QC before release",
            "Controlled pilot and clinical review of transformation bounds remain outstanding",
        ],
        "frozen_references": {"dataset504": "NCCT original; unchanged", "dataset505": "NCCT+perfusion original; unchanged",
                              "split_sha256": FROZEN_SPLIT_SHA, **input_shas},
        "targets": {
            "Dataset506_ISLES2024_NCCT_CTCTRL": {
                "channels": ["NCCT"], "nnunet_channel_names": ["CT"],
                "normalization": ["CT-style train-fold calibrated clipping + scaling"],
            },
            "Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST": {
                "channels": list(CHANNELS),
                "nnunet_channel_names": ["CT", "TMAX", "CBF", "CBV", "MTT"],
                "normalization": ["identical CT transform as Dataset506"] + ["robust train-fold calibrated"] * 4,
            },
        },
        "candidate_transform_policy": {
            "nonfinite": "For perfusion only: fill input NaN/Inf with fold-training median within image-only anatomical support, preserve background convention; never use lesion masks or validation labels. Investigational; release gate pending.",
            "CT": "nnU-Net CT-style: use fold-training NCCT support values for q0.5/q99.5, mean/std; clip to quantiles then scale by train mean/std; never global all-149 foreground statistics. Identical transform in 506 and 507.",
            "perfusion": "For each of TMAX, CBF, CBV, MTT, on training patients only: fit finite support q0.5/q99.5 and robust location/scale (median and IQR/1.349); replace invalids and winsorize before resampling; scale after; fail on degenerate scale. Apply identical stored fold params to validation/inference.",
            "candidate_z_safety_cap": 8,
            "cap_policy": "Cap |robust z| at 8 is a proposed training-only sensitivity candidate, NOT a validated clinical threshold and must not be finalized based on OOF results.",
            "support": "An image-only NCCT-derived anatomical support mask, with manual QC for edge cases, independently of target infarct labels or predictions.",
            "fit_scope": "Each model fold fits only on its own 119/120 training patient images; validation never contributes to fitted cohort statistics.",
            "testing": "Validate before and after spatial resampling, no NaN/Inf, finite consistent channel scales, label/affine/shape integrity, and reproducible folded inference.",
            "architectural_gate": "Default nnU-Net shared all-149 planning/preprocessing does NOT satisfy train-only fitted statistics for 5-fold OOF. WU20 must create fold-isolated preprocessing and inference, or switch to fully predetermined, fit-free transforms. Do not run unchanged nnUNetv2_plan_and_preprocess on all 149 and call it leak-free.",
        },
        "folds": [
            {"fold": i, "n_train": len(f["train"]), "n_validation": len(f["val"]),
             "training_patient_ids": list(f["train"]), "validation_patient_ids": list(f["val"]),
             "parameter_fit_allowed_cases": list(f["train"]),
             "parameter_fit_disallowed_cases": list(f["val"])}
            for i, f in enumerate(folds)
        ],
        "qcs_from_frozen_audits": qc,
        "evaluation": {
            "primary_comparison": "Dataset507 versus Dataset506, same frozen paired 149-case OOF design",
            "secondary": ["Dataset506 vs Dataset504", "Dataset507 vs Dataset505"],
            "primary_metrics": ["Dice", "lesion-wise F1", "absolute volume difference mL", "absolute lesion-count difference"],
            "inference": "Paired patient bootstrap, sign-flip permutation exploratory, Holm family-wise correction for four prespecified metrics; explicit fold sensitivity and failures.",
            "lock": "No test/validation outcome-guided normalizer fitting, clipping choice, or hyperparameter tuning."
        },
    }


def _atomically_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--raw-audit-csv", type=Path, required=True)
    parser.add_argument("--preprocessed-audit-csv", type=Path, required=True)
    parser.add_argument("--plans505", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folds = frozen_folds(args.splits)
    ids = {c for f in folds for c in f["val"]}
    raw = audited_csv(args.raw_audit_csv, RAW_FIELDS | {"case_id", "channel"}, ids)
    pre = audited_csv(args.preprocessed_audit_csv, PRE_FIELDS | {"case_id", "channel"}, ids)
    source_config = check_original_plans(args.plans505)
    qc = qc_summary(raw, pre, ids)
    if qc["MTT"]["raw_patients_nonfinite"] != 21 or qc["MTT"]["pre_patients_abs_z_gt_100"] != 125:
        raise ValueError("Frozen MTT audit controls changed: unexpectedly mismatched sources")
    input_hashes = {name: digest_file(path) for name, path in (
        ("raw_audit_sha256", args.raw_audit_csv),
        ("preprocessed_audit_sha256", args.preprocessed_audit_csv),
        ("plans505_sha256", args.plans505),
    )}
    plan = plan_protocol(folds, qc, input_hashes)
    plan["source_configuration"] = source_config
    plan["generated_utc"] = datetime.now(timezone.utc).isoformat()
    out = args.output_dir
    _atomically_write(out / "WU19_preprocessing_protocol_PRIVATE.json", plan)
    public = {k: v for k, v in plan.items() if k != "folds"}
    _atomically_write(out / "WU19_preflight_aggregate.json", public)
    lines = [
        "# WU-19: Controlled ISLES2024 preprocessing protocol (DESIGN ONLY)", "",
        "**Status: NOT APPROVED FOR TRAINING.** No Dataset506/507 images have been created.", "",
        "## Frozen sources and audit evidence", "",
        "All 149 OOF patients and five disjoint folds verified; split SHA256 " + FROZEN_SPLIT_SHA + ".", "",
        "| Channel | Raw patients with non-finite | Raw patients below −10000 | Pre patients with |z|>100 | Pre voxels |z|>100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for ch in CHANNELS:
        q = qc[ch]
        lines.append(f"| {ch} | {q['raw_patients_nonfinite']} | {q['raw_patients_below_minus_10000']} | {q['pre_patients_abs_z_gt_100']} | {q['pre_voxels_abs_z_gt_100']:,} |")
    lines += ["", "## Target contrasts", "", "- Dataset506: NCCT-only, fold-specific CT-style normalization.",
              "- Dataset507: exact same NCCT normalization plus robustly sanitized, training-fold calibrated four perfusion channels.",
              "- The primary comparison is 507 vs 506; Dataset504/505 are frozen secondary comparisons.",
              "", "## Release gates (must all be satisfied before GPU training)", ""]
    for s in plan["reasons_not_training_ready"]:
        lines.append("- [ ] " + s)
    lines += ["", "## Critical leakage safeguard", "",
              "nnU-Net's default CT scheme uses cohort foreground statistics in its plans. A single shared plan from all 149 patients leaks validation *distribution* into the preprocessing parameters even without using outcome labels. Train-only fitted statistics therefore require fold-specific preprocessing and matching validation inference. The notebook outputs only a protocol and will not launch training.",
              "", "## Audit limitations", "",
              "- Raw and preprocessed volumes cannot be matched voxel-for-voxel because preprocessing resamples/crops.",
              "- Extreme normalized intensities are not proof of lesion-prediction harm; Notebook18 reports a descriptive QC signal.",
              "- Proposed nonfinite filling, clipping and robust limits remain candidates requiring clinical review and training-only parameter fitting.",
              "- Per-patient manifests stay in private Drive. Do not commit any private data to public GitHub.", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "WU19_protocol_readme.md").write_text("\n".join(lines), encoding="utf8")
    print("WU-19 preflight complete: 149 paired patients, 745 raw and 745 preprocessed audit rows validated.")
    print("WU-19 training readiness: BLOCKED until independent fold-scoped transforms pass end-to-end qualification.")
    print("Outputs:", out)


if __name__ == "__main__":
    main()
