"""Regression tests for the corrected ISLES'24 paired-statistics reporting."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from scripts.compare_isles2024_ncct_perfusion_paired import analyze, main


def _rows():
    rows = []
    for i in range(149):
        fold = min(i // 30, 4)
        volume = float(i + 1)
        rows.append(
            {
                "case_id": f"sub_stroke{i + 1:04d}",
                "fold": fold,
                "reference_volume_ml": volume,
                "ncct_dice": 0.15,
                "perfusion_dice": 0.20,
                "delta_dice": 0.05,
                "ncct_absolute_volume_difference_ml": 4.0,
                "perfusion_absolute_volume_difference_ml": 3.0,
                "delta_absolute_volume_difference_ml": -1.0,
                "ncct_lesion_f1": 0.1,
                "perfusion_lesion_f1": 0.12,
                "delta_lesion_f1": 0.02,
                "ncct_absolute_lesion_count_difference": 3.0,
                "perfusion_absolute_lesion_count_difference": 4.0,
                "delta_absolute_lesion_count_difference": 1.0,
            }
        )
    return rows


def test_all_four_metrics_expose_sensitivity_and_holm_values():
    summary = analyze(_rows(), full=True, n_boot=100, n_perm=100, seed=20261119)
    assert summary["n_patients"] == 149
    assert len(summary["metrics"]) == 4
    for metric in summary["metrics"].values():
        assert 0 <= metric["holm_adjusted_p_4_metrics"] <= 1
        assert set(metric["leave_one_fold_out_sensitivity"]) == set("01234")
        assert sum(x["n"] for x in metric["volume_quartile_sensitivity"].values()) == 149
        assert metric["delta_median"] is not None
        assert metric["delta_q1"] is not None
        assert metric["delta_q3"] is not None
    assert summary["metrics"]["dice"]["cohens_dz_paired"] is not None


def test_report_exposes_holm_adjustment_in_readable_table(tmp_path: Path, monkeypatch):
    inputs = _rows()
    columns = [
        "case_id", "fold", "reference_volume_ml", "voxel_volume_ml", "dice",
        "absolute_volume_difference_ml", "lesion_f1",
        "absolute_lesion_count_difference", "reference_lesion_count",
    ]
    baseline = tmp_path / "baseline.csv"
    perfusion = tmp_path / "perfusion.csv"
    for path, which in [(baseline, "ncct"), (perfusion, "perfusion")]:
        with path.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in inputs:
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "fold": row["fold"],
                        "reference_volume_ml": row["reference_volume_ml"],
                        "voxel_volume_ml": 0.001,
                        "dice": row[f"{which}_dice"],
                        "absolute_volume_difference_ml": row[
                            f"{which}_absolute_volume_difference_ml"
                        ],
                        "lesion_f1": row[f"{which}_lesion_f1"],
                        "absolute_lesion_count_difference": row[
                            f"{which}_absolute_lesion_count_difference"
                        ],
                        "reference_lesion_count": 3,
                    }
                )

    out = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paired", "--baseline-csv", str(baseline),
            "--perfusion-csv", str(perfusion),
            "--output-dir", str(out),
            "--bootstrap-repetitions", "100",
            "--permutation-repetitions", "100",
        ],
    )
    main()
    report = (out / "isles2024_paired_full_report.md").read_text()
    assert "Holm-adjusted p (4 metrics)" in report
    assert "Leave-one-fold-out sensitivity" in report
    assert "Lesion reference volume quartiles" in report
    assert "not organizer hidden-test" in report
    raw = json.loads((out / "isles2024_paired_full_summary.json").read_text())
    assert len(raw["metrics"]) == 4
