#!/usr/bin/env python3
"""Fail-closed paired ISLES'24 NCCT versus NCCT+perfusion local OOF analysis.

There are two modes:
  --perfusion-summaries <fold0.json> ... <fold4.json>
      Validates five nnU-Net summaries against the frozen NCCT case-level CSV;
      computes voxel Dice and absolute volume difference only. Lesion metrics
      are intentionally unavailable in this mode (need mask-level Panoptica).
  --perfusion-csv <Dataset505 OOF case metrics CSV>
      Validates 149 paired OOF cases, frozen folds and reference geometry;
      computes all four paired endpoints with bootstrap, permutation p-values,
      and Holm multiplicity correction.

Primary sign convention is perfusion minus NCCT; negative difference is better
for absolute errors, positive difference is better for Dice and lesion-F1.
These are *local out-of-fold*, not official hidden-test challenge metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = (
    ("dice", "Dice", "higher"),
    ("absolute_volume_difference_ml", "Absolute volume difference (mL)", "lower"),
    ("lesion_f1", "Lesion-wise F1", "higher"),
    ("absolute_lesion_count_difference", "Absolute lesion-count difference", "lower"),
)
REQUIRED = {"case_id", "fold", "dice", "absolute_volume_difference_ml", "reference_volume_ml", "voxel_volume_ml"}


def load_csv(path: Path, *, expected_n: int) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing fields {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    unique = {r["case_id"]: r for r in rows}
    if len(rows) != expected_n or len(unique) != expected_n:
        raise ValueError(f"{path}: require {expected_n} unique cases; found {len(rows)} rows, {len(unique)} unique")
    if any(not k.startswith("sub_stroke") for k in unique):
        raise ValueError(f"{path}: unexpected patient identifiers")
    if any((r.get("brain_mask_path") or "").strip() for r in rows):
        raise ValueError(f"{path}: brain-masked results cannot be compared to unmasked frozen baseline")
    return unique


def load_perfusion_summaries(paths: list[Path], baseline: dict[str, dict], expected_n: int) -> dict[str, dict]:
    if len(paths) != 5:
        raise ValueError("Provide exactly five nnU-Net fold validation summary.json paths in fold 0..4 order")
    out: dict[str, dict] = {}
    for fold, path in enumerate(paths):
        obj = json.loads(path.read_text(encoding="utf8"))
        for row in obj.get("metric_per_case", []):
            source = Path(row["prediction_file"])
            if source.suffixes[-2:] != [".nii", ".gz"]:
                raise ValueError(f"Malformed summary prediction path: {source}")
            case_id = source.name.removesuffix(".nii.gz")
            if case_id in out:
                raise ValueError(f"Duplicate Dataset505 prediction: {case_id}")
            if case_id not in baseline:
                raise ValueError(f"Unmatched Dataset505 case: {case_id}")
            old = baseline[case_id]
            voxel_ml = float(old["voxel_volume_ml"])
            m = row["metrics"]["1"]
            if not all(np.isfinite(float(m[k])) for k in ("Dice", "n_pred", "n_ref")):
                raise ValueError(f"Non-finite Dataset505 metrics: {case_id}")
            gt_505_volume_ml = float(m["n_ref"]) * voxel_ml
            old_gt_volume_ml = float(old["reference_volume_ml"])
            if not np.isclose(gt_505_volume_ml, old_gt_volume_ml, atol=1e-5, rtol=1e-6):
                raise ValueError(
                    f"{case_id}: reference volume differs between Dataset504/505: "
                    f"{old_gt_volume_ml} vs {gt_505_volume_ml} ml"
                )
            out[case_id] = {
                "case_id": case_id,
                "fold": str(fold),
                "dice": repr(float(m["Dice"])),
                "absolute_volume_difference_ml": repr(abs(float(m["n_pred"]) - float(m["n_ref"])) * voxel_ml),
                "reference_volume_ml": repr(gt_505_volume_ml),
                "voxel_volume_ml": repr(voxel_ml),
            }
    if len(out) != expected_n:
        raise ValueError(f"Expected {expected_n} unique Dataset505 cases, found {len(out)}")
    return out


def validate_pairs(baseline: dict[str, dict], perfusion: dict[str, dict], *, full: bool) -> list[dict]:
    if baseline.keys() != perfusion.keys():
        raise ValueError(f"Patient sets differ: missing={sorted(baseline.keys()-perfusion.keys())}, extra={sorted(perfusion.keys()-baseline.keys())}")
    paired = []
    for case_id in sorted(baseline):
        a, b = baseline[case_id], perfusion[case_id]
        if int(a["fold"]) != int(b["fold"]):
            raise ValueError(f"Patient-wise fold mismatch for {case_id}")
        for attr, atol in (("reference_volume_ml", 1e-5), ("voxel_volume_ml", 1e-9)):
            if not np.isclose(float(a[attr]), float(b[attr]), atol=atol, rtol=1e-6):
                raise ValueError(f"{case_id}: {attr} is different across modalities")
        if (
            full
            and "reference_lesion_count" in a
            and "reference_lesion_count" in b
            and int(a["reference_lesion_count"]) != int(b["reference_lesion_count"])
        ):
            raise ValueError(f"{case_id}: reference connected-component count differs")
        row = {"case_id": case_id, "fold": int(a["fold"]), "reference_volume_ml": float(a["reference_volume_ml"])}
        for key, _, _ in METRICS[:4 if full else 2]:
            if key not in b:
                raise ValueError(f"{case_id}: Dataset505 metric missing: {key}")
            old, new = float(a[key]), float(b[key])
            if not np.isfinite(old) or not np.isfinite(new):
                raise ValueError(f"Non-finite result for {case_id}/{key}")
            if key in ("dice", "lesion_f1") and not (0 <= old <= 1 and 0 <= new <= 1):
                raise ValueError(f"Invalid score outside [0,1]: {case_id}/{key}")
            if key not in ("dice", "lesion_f1") and (old < 0 or new < 0):
                raise ValueError(f"Negative absolute error for {case_id}/{key}")
            row[f"ncct_{key}"] = old
            row[f"perfusion_{key}"] = new
            row[f"delta_{key}"] = new - old
        paired.append(row)
    fold_sizes = {f: sum(x["fold"] == f for x in paired) for f in range(5)}
    if fold_sizes != {0: 30, 1: 30, 2: 30, 3: 30, 4: 29}:
        raise ValueError(f"Frozen folds do not match 30/30/30/30/29: {fold_sizes}")
    return paired


def percentile_ci(values: np.ndarray, *, n_reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    # Chunked to avoid >GB RAM allocations at large bootstrap counts.
    means = []
    for batch_start in range(0, n_reps, 2000):
        batch_n = min(2000, n_reps-batch_start)
        idx = rng.integers(0, len(values), size=(batch_n, len(values)))
        means.append(values[idx].mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(means), [0.025, 0.975])]


def paired_permutation_p(delta: np.ndarray, *, n_reps: int, seed: int) -> float:
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    tail = 0
    for batch_start in range(0, n_reps, 2000):
        batch_n = min(2000, n_reps - batch_start)
        flips = rng.integers(0, 2, size=(batch_n, len(delta)), dtype=np.int8) * 2 - 1
        dist = np.abs(flips @ delta / len(delta))
        tail += int(np.count_nonzero(dist >= observed - 1e-12))
    return float((tail + 1) / (n_reps + 1))


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(n)
    current = 0.0
    for rank, original_index in enumerate(order):
        current = max(current, (n-rank)*pvalues[int(original_index)])
        adj[int(original_index)] = min(current, 1.0)
    return [float(x) for x in adj]


def analyze(paired: list[dict], *, full: bool, n_boot: int, n_perm: int, seed: int) -> dict:
    n = len(paired)
    metrics = METRICS if full else METRICS[:2]
    result = {"n_patients": n, "fold_n": {str(f): sum(x["fold"] == f for x in paired) for f in range(5)},
              "mode": "four_metrics_full" if full else "dice_volume_interim", "metrics": {},
              "note": "Local matched out-of-fold validation, not the official ISLES'24 hidden-test evaluation.",
              "bootstrap_repetitions": n_boot, "permutation_repetitions": n_perm, "rng_seed": seed,
              "brain_mask_applied": False, "multiple_testing_method": "Holm family-wise correction (only in full four-metric mode)"}
    pvals = []
    for idx, (key, human, direction) in enumerate(metrics):
        base = np.asarray([r[f"ncct_{key}"] for r in paired], dtype=float)
        perf = np.asarray([r[f"perfusion_{key}"] for r in paired], dtype=float)
        delta = perf - base
        p = paired_permutation_p(delta, n_reps=n_perm, seed=seed+100+idx)
        pvals.append(p)
        sample_std = float(np.std(delta, ddof=1))
        effect_size = float(delta.mean() / sample_std) if sample_std > 0 else None
        n_equal = int(np.count_nonzero(np.isclose(perf, base, rtol=0, atol=1e-12)))
        quartile_bounds = np.quantile([r["reference_volume_ml"] for r in paired], [0.25, 0.5, 0.75])
        labels = np.searchsorted(quartile_bounds, [r["reference_volume_ml"] for r in paired], side="left")
        volume_quartiles = {}
        for bin_id in range(4):
            group = labels == bin_id
            volume_quartiles[str(bin_id + 1)] = {
                "n": int(group.sum()),
                "mean_reference_volume_ml": float(np.mean([r["reference_volume_ml"] for i, r in enumerate(paired) if group[i]])),
                "ncct_mean": float(base[group].mean()),
                "perfusion_mean": float(perf[group].mean()),
                "delta_mean": float(delta[group].mean()),
            }
        leave_one_fold_out = {
            str(fold): {
                "n": int(np.count_nonzero(np.asarray([r["fold"] != fold for r in paired]))),
                "delta_mean": float(delta[np.asarray([r["fold"] != fold for r in paired])].mean()),
            }
            for fold in range(5)
        }
        result["metrics"][key] = {
            "delta_median": float(np.median(delta)),
            "delta_q1": float(np.quantile(delta, 0.25)),
            "delta_q3": float(np.quantile(delta, 0.75)),
            "cohens_dz_paired": effect_size,
            "volume_quartile_sensitivity": volume_quartiles,
            "leave_one_fold_out_sensitivity": leave_one_fold_out,
            "name": human, "better": direction, "n": n,
            "ncct_mean": float(base.mean()), "perfusion_mean": float(perf.mean()),
            "perfusion_minus_ncct_mean": float(delta.mean()),
            "delta_95ci_paired_bootstrap": percentile_ci(delta, n_reps=n_boot, seed=seed+idx),
            "two_sided_sign_flip_permutation_p_nominal": p,
            "n_improved": int(((perf > base) if direction == "higher" else (perf < base)).sum()),
            "n_worsened": int(((perf < base) if direction == "higher" else (perf > base)).sum()),
            "n_unchanged": n_equal,
            "fold_means": {
                str(f): {"n": sum(r["fold"]==f for r in paired),
                         "ncct": float(np.mean([r[f"ncct_{key}"] for r in paired if r["fold"] == f])),
                         "perfusion": float(np.mean([r[f"perfusion_{key}"] for r in paired if r["fold"] == f]))}
                for f in range(5)
            },
        }
    if full:
        adjusted = holm_adjust(pvals)
        for i,(key,_,_) in enumerate(metrics):
            result["metrics"][key]["holm_adjusted_p_4_metrics"] = adjusted[i]
        result["four_metric_lesion_count_statistics_available"] = True
    else:
        result["four_metric_lesion_count_statistics_available"] = False
        result["pending_evaluation"] = ["lesion_f1", "absolute_lesion_count_difference"]
        result["caution"] = "Dice and volume p-values are nominal exploratory tests; four-endpoint Holm correction pending mask-level evaluation."
    result["reference_volume_quartiles_ml"] = [float(v) for v in np.quantile([r["reference_volume_ml"] for r in paired], [0.25,0.5,0.75])]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-csv", type=Path, required=True)
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--perfusion-csv", type=Path)
    mx.add_argument("--perfusion-summaries", type=Path, nargs=5, metavar="JSON")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    ap.add_argument("--permutation-repetitions", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--expected-n", type=int, default=149)
    args = ap.parse_args()
    if args.expected_n != 149:
        raise ValueError("Study protocol requires all 149 patients; --expected-n must remain 149")
    if args.bootstrap_repetitions < 100 or args.permutation_repetitions < 100:
        raise ValueError("At least 100 resamples required")
    baseline = load_csv(args.baseline_csv, expected_n=args.expected_n)
    full = args.perfusion_csv is not None
    if full:
        perfusion = load_csv(args.perfusion_csv, expected_n=args.expected_n)
        required_full = {"lesion_f1", "absolute_lesion_count_difference"}
        if any(not required_full.issubset(r) for r in perfusion.values()):
            raise ValueError("Dataset505 full OOF output lacks lesion-wise metrics")
    else:
        perfusion = load_perfusion_summaries(args.perfusion_summaries, baseline, args.expected_n)
    paired = validate_pairs(baseline, perfusion, full=full)
    summary = analyze(paired, full=full, n_boot=args.bootstrap_repetitions, n_perm=args.permutation_repetitions, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pathbase = "isles2024_paired_full" if full else "isles2024_paired_dice_volume"
    with (args.output_dir / f"{pathbase}_cases.csv").open("w", newline="", encoding="utf8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    (args.output_dir / f"{pathbase}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    lines = [
        "# ISLES2024: paired Dataset504 vs Dataset505 evaluation",
        "",
        "**Scope:** local five-fold OOF matched validation, 149 patients; not organizer hidden-test scores.",
        "",
        "**All four endpoints are exploratory unless their Holm-adjusted p-values survive 0.05.**",
        "",
        "| Metric | NCCT | NCCT + perfusion | Δ (perfusion − NCCT) | 95% paired CI | Nominal p | Holm-adjusted p (4 metrics) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for k, _, _ in (METRICS if full else METRICS[:2]):
        x = summary["metrics"][k]
        ci = x["delta_95ci_paired_bootstrap"]
        adjusted = x.get("holm_adjusted_p_4_metrics")
        adjusted_display = f"{adjusted:.5f}" if adjusted is not None else "Pending"
        lines.append(
            f"| {x['name']} | {x['ncct_mean']:.5f} | {x['perfusion_mean']:.5f} "
            f"| {x['perfusion_minus_ncct_mean']:+.5f} "
            f"| [{ci[0]:+.5f}, {ci[1]:+.5f}] "
            f"| {x['two_sided_sign_flip_permutation_p_nominal']:.5f} "
            f"| {adjusted_display} |"
        )
    lines.extend([
        "",
        "The reported paired patient bootstrap and sign-flip tests assume patient-level independence. "
        "Shared fold models and overlapping cross-validation training sets limit formal inference. "
        "The following analyses are *descriptive sensitivity checks*, not independent external validation.",
        "",
        "## Leave-one-fold-out sensitivity",
        "",
        "| Metric | Omitted fold | Remaining patients | Mean delta |",
        "|---|---:|---:|---:|",
    ])
    for k, _, _ in (METRICS if full else METRICS[:2]):
        m = summary["metrics"][k]
        for fold, x in m["leave_one_fold_out_sensitivity"].items():
            lines.append(f"| {m['name']} | {fold} | {x['n']} | {x['delta_mean']:+.5f} |")
    lines.extend([
        "",
        "## Lesion reference volume quartiles (descriptive)",
        "",
        "| Metric | Reference volume quartile | Cases | Mean ref volume (mL) | Mean delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for k, _, _ in (METRICS if full else METRICS[:2]):
        m = summary["metrics"][k]
        for q, x in m["volume_quartile_sensitivity"].items():
            lines.append(
                f"| {m['name']} | Q{q} | {x['n']} "
                f"| {x['mean_reference_volume_ml']:.3f} | {x['delta_mean']:+.5f} |"
            )
    lines.extend([
        "",
        "## Effect sizes and heterogeneity",
        "",
        "| Metric | Median delta | Q1 delta | Q3 delta | Paired Cohen dz |",
        "|---|---:|---:|---:|---:|",
    ])
    for k, _, _ in (METRICS if full else METRICS[:2]):
        x = summary["metrics"][k]
        dz = x["cohens_dz_paired"]
        lines.append(
            f"| {x['name']} | {x['delta_median']:+.5f} | {x['delta_q1']:+.5f} "
            f"| {x['delta_q3']:+.5f} | {dz:.4f} |" if dz is not None
            else f"| {x['name']} | {x['delta_median']:+.5f} | {x['delta_q1']:+.5f} "
                 f"| {x['delta_q3']:+.5f} | n/a |"
        )
    lines.extend([
        "",
        "## Scientific limitations",
        "",
        "- Dice and lesion-F1 higher is better; absolute lesion volume/count errors lower is better.",
        "- No organizer-compatible NCCT-derived brain mask was applied; these are local OOF metrics.",
        "- Patient scores are heterogeneous: mean gains do not imply a consistent individual benefit.",
        "- The current Dataset505 MTT fingerprint has an extreme negative outlier; this requires "
        "per-patient raw image analysis before claiming modality-specific improvement.",
        "- Both arms use nnU-Net ZScoreNormalization for the NCCT channel, not CTNormalization.",
        "- A clean separate preprocessing-controlled ablation is needed; retain Dataset504/505 unchanged.",
        "",
    ])
    (args.output_dir / f"{pathbase}_report.md").write_text("\n".join(lines), encoding="utf8")
    print("Verified", len(paired), "paired cases; all five frozen folds.")
    print("Mode:", summary["mode"])
    for k, obj in summary["metrics"].items():
        print(f"{k}: NCCT={obj['ncct_mean']:.6f}, NCCT+perfusion={obj['perfusion_mean']:.6f}, Δ={obj['perfusion_minus_ncct_mean']:+.6f}, 95% CI={obj['delta_95ci_paired_bootstrap']}, nominal p={obj['two_sided_sign_flip_permutation_p_nominal']:.5f}, adjusted p={obj.get('holm_adjusted_p_4_metrics')}")
    print("Outputs:", args.output_dir)


if __name__ == "__main__":
    main()
