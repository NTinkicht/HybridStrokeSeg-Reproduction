"""WU-19 candidate train-only normalization mathematics (not nnU-Net integration).

Input maps MUST consist solely of deterministic training-fold, image-derived
anatomical-support samples. This module does not extract samples, determine
brain masks or write NIfTI files. Use for offline qualification only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def training_samples_only(
    samples_by_case: Mapping[str, np.ndarray], train_ids: Sequence[str], val_ids: Sequence[str]
) -> np.ndarray:
    """Reject train/validation leakage and concatenate finite training samples."""
    train, val = set(train_ids), set(val_ids)
    provided = set(samples_by_case)
    if not train or not val or train & val:
        raise ValueError("Missing or overlapping train/validation groups")
    if provided != train:
        raise ValueError("Fit input must contain exactly training cases: no validation or missing cases")
    values: list[np.ndarray] = []
    for case in sorted(train):
        arr = np.asarray(samples_by_case[case]).reshape(-1)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Non-numeric intensities for {case}")
        finite = arr[np.isfinite(arr)].astype(np.float64)
        if finite.size < 16:
            raise ValueError(f"Insufficient finite, anatomical-support training samples in {case}")
        values.append(finite)
    return np.concatenate(values)


def fit_fold_channel(
    samples_by_case: Mapping[str, np.ndarray], train_ids: Sequence[str],
    val_ids: Sequence[str], *, channel: str,
) -> dict:
    """Fit provisional CT-style or robust perfusion parameters on training IDs only.

    Deterministic q0.5/99.5 clipping; CT mean and std follow train-only
    foreground intensities, robust perfusion uses median/IQR/1.349.
    Anatomical support is a separate WU20 gate; this only accepts its samples.
    """
    if channel not in {"NCCT", "TMAX", "CBF", "CBV", "MTT"}:
        raise ValueError(f"Unsupported modality: {channel}")
    values = training_samples_only(samples_by_case, train_ids, val_ids)
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("Degenerate/invalid training-fold clipping bounds")
    if channel == "NCCT":
        mean, scale = float(values.mean()), float(values.std())
        family = "CT_style_fold_only"
    else:
        q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        mean, scale = float(med), float((q75 - q25) / 1.349)
        family = "perfusion_fold_robust"
    if scale < 1e-6 or not np.isfinite(scale):
        raise ValueError("Degenerate training-only normalization scale")
    return {
        "channel": channel, "scheme": family, "clip_low": float(lo),
        "clip_high": float(hi), "center": mean, "scale": scale,
        "n_training_cases": len(set(train_ids)), "n_samples": int(values.size),
        "training_ids": sorted(set(train_ids)), "validation_ids_excluded": sorted(set(val_ids)),
        "candidate_post_z_cap": 8.0 if channel != "NCCT" else None,
        "status": "candidate_not_authorized_for_training",
    }


def apply_candidate_transform(image: np.ndarray, params: dict, *, support: np.ndarray) -> np.ndarray:
    """Compute candidate image-only-support transform without changing inputs.

    Non-finite perfusion values are replaced with fold-trained *median* inside
    support, then clipped before scaling. Outside support is zero. This policy
    remains *candidate*, not clinically approved; WU20 must validate support.
    """
    raw = np.asarray(image)
    support = np.asarray(support)
    if raw.shape != support.shape or raw.ndim != 3 or support.dtype != np.bool_:
        raise ValueError("Expected matching 3D image + boolean anatomical support")
    if not support.any():
        raise ValueError("Empty image-only anatomical support")
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("Image needs numeric intensities")
    if params["status"] != "candidate_not_authorized_for_training":
        raise ValueError("Unexpected unreviewed normalization manifest")
    if not all(np.isfinite(params[k]) for k in ("center", "scale", "clip_low", "clip_high")):
        raise ValueError("Invalid normalization parameters")
    if params["scale"] < 1e-6 or params["clip_low"] >= params["clip_high"]:
        raise ValueError("Invalid training-fold scaling values")
    data = raw.astype(np.float64, copy=True)
    invalid = ~np.isfinite(data)
    if params["channel"] == "NCCT" and invalid.any():
        raise ValueError("Nonfinite NCCT input: manual QA required")
    data[invalid] = params["center"]
    np.clip(data, params["clip_low"], params["clip_high"], out=data)
    data -= params["center"]
    data /= params["scale"]
    if params["candidate_post_z_cap"] is not None:
        np.clip(data, -params["candidate_post_z_cap"], params["candidate_post_z_cap"], out=data)
    data[~support] = 0.0
    if not np.all(np.isfinite(data)):
        raise ValueError("Transform left nonfinite values")
    return data.astype(np.float32)
