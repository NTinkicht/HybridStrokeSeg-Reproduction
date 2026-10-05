"""Safe nnU-Net v2 staging for the ISLES 2022 modernization track.

The official nnU-Net v2 format requires every channel of a case and its label to
share geometry. ISLES 2022 is released in native space, so this module refuses
to stage a multimodal case when geometry does not already match the DWI grid.
Registration is intentionally a separate, auditable preprocessing step.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

from .data.isles2022 import ISLES22Case, geometry_equal, geometry_signature
from .splits import make_patient_kfolds

SUPPORTED_CHANNELS = ("dwi", "adc", "flair")


def nnunet_case_id(case: ISLES22Case) -> str:
    """Convert the BIDS-like case identifier to a stable nnU-Net identifier."""
    return case.case_id.replace("-", "_")


def build_dataset_json(
    *,
    channels: Sequence[str],
    num_training: int,
) -> dict[str, object]:
    """Build the minimal nnU-Net v2 ``dataset.json`` structure."""
    channel_list = tuple(str(channel).lower() for channel in channels)
    if not channel_list:
        raise ValueError("At least one input channel is required")
    if len(channel_list) != len(set(channel_list)):
        raise ValueError("Input channels must be unique")
    unsupported = [channel for channel in channel_list if channel not in SUPPORTED_CHANNELS]
    if unsupported:
        raise ValueError(f"Unsupported ISLES 2022 channels: {unsupported}")
    if num_training < 1:
        raise ValueError("num_training must be positive")

    return {
        "channel_names": {
            str(index): channel.upper()
            for index, channel in enumerate(channel_list)
        },
        "labels": {"background": 0, "stroke": 1},
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }


def build_splits_final(
    case_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 2026,
) -> list[dict[str, list[str]]]:
    """Build nnU-Net v2's custom ``splits_final.json`` data structure."""
    folds = make_patient_kfolds(tuple(case_ids), n_splits=n_splits, seed=seed)
    return [
        {"train": list(fold.train_ids), "val": list(fold.validation_ids)}
        for fold in folds
    ]


def _case_geometry_mismatches(
    case: ISLES22Case,
    channels: Sequence[str],
) -> list[str]:
    reference = geometry_signature(case.dwi)
    mismatches: list[str] = []
    for channel in channels:
        path = getattr(case, channel)
        if not geometry_equal(reference, geometry_signature(path)):
            mismatches.append(channel)
    if not geometry_equal(reference, geometry_signature(case.mask)):
        mismatches.append("mask")
    return mismatches


def audit_nnunet_geometry(
    cases: Iterable[ISLES22Case],
    *,
    channels: Sequence[str] = ("dwi", "adc", "flair"),
) -> dict[str, list[str]]:
    """Return case -> mismatching inputs for the requested nnU-Net channels."""
    channel_list = tuple(str(channel).lower() for channel in channels)
    build_dataset_json(channels=channel_list, num_training=1)
    failures: dict[str, list[str]] = {}
    for case in cases:
        mismatches = _case_geometry_mismatches(case, channel_list)
        if mismatches:
            failures[case.case_id] = mismatches
    return failures


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    raise ValueError("mode must be 'copy' or 'symlink'")


def stage_isles2022_nnunet(
    cases: Sequence[ISLES22Case],
    output_root: str | Path,
    *,
    dataset_id: int = 501,
    dataset_name: str = "ISLES2022",
    channels: Sequence[str] = ("dwi", "adc", "flair"),
    mode: str = "copy",
    overwrite: bool = False,
    split_seed: int = 2026,
) -> Path:
    """Stage geometry-compatible ISLES 2022 cases in nnU-Net v2 raw format.

    This function never registers or resamples images. If any requested channel
    or label differs from the DWI grid, staging stops before writing the dataset.
    """
    if not cases:
        raise ValueError("No ISLES 2022 cases supplied")
    if not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be between 1 and 999")
    if not dataset_name.replace("_", "").isalnum():
        raise ValueError("dataset_name must contain only letters, numbers, and underscores")

    channel_list = tuple(str(channel).lower() for channel in channels)
    metadata = build_dataset_json(channels=channel_list, num_training=len(cases))
    failures = audit_nnunet_geometry(cases, channels=channel_list)
    if failures:
        preview = "; ".join(
            f"{case_id}: {','.join(items)}"
            for case_id, items in sorted(failures.items())[:10]
        )
        suffix = "" if len(failures) <= 10 else f"; ... {len(failures) - 10} more"
        raise ValueError(
            "nnU-Net staging blocked because modalities/labels do not share the DWI grid. "
            f"Resolve registration first. {preview}{suffix}"
        )

    root = Path(output_root)
    dataset_dir = root / f"Dataset{dataset_id:03d}_{dataset_name}"
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty nnU-Net dataset directory: {dataset_dir}"
            )
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_map: dict[str, str] = {}
    nnunet_ids: list[str] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        target_id = nnunet_case_id(case)
        if target_id in nnunet_ids:
            raise ValueError(f"Duplicate normalized nnU-Net case identifier: {target_id}")
        nnunet_ids.append(target_id)
        case_map[case.case_id] = target_id

        for channel_index, channel in enumerate(channel_list):
            source = getattr(case, channel)
            destination = images_tr / f"{target_id}_{channel_index:04d}.nii.gz"
            _materialize(source, destination, mode)

        _materialize(case.mask, labels_tr / f"{target_id}.nii.gz", mode)

    (dataset_dir / "dataset.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "case_map.json").write_text(
        json.dumps(case_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "splits_final.json").write_text(
        json.dumps(build_splits_final(nnunet_ids, seed=split_seed), indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "STAGING_NOTE.txt").write_text(
        "splits_final.json is generated here for provenance. nnU-Net v2 expects custom "
        "splits_final.json in the corresponding nnUNet_preprocessed dataset directory after "
        "planning/preprocessing. Copy this file there before training if you want to force the "
        "repository's deterministic folds.\n",
        encoding="utf-8",
    )

    return dataset_dir


def copy_splits_to_preprocessed(dataset_dir: str | Path, preprocessed_dir: str | Path) -> Path:
    """Copy the staged deterministic folds to nnU-Net's preprocessed dataset folder."""
    source = Path(dataset_dir) / "splits_final.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_dir = Path(preprocessed_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "splits_final.json"
    shutil.copy2(source, destination)
    return destination


def configure_nnunet_environment(
    *,
    raw: str | Path,
    preprocessed: str | Path,
    results: str | Path,
) -> None:
    """Set nnU-Net v2 path variables for the current Python process."""
    os.environ["nnUNet_raw"] = str(Path(raw).resolve())
    os.environ["nnUNet_preprocessed"] = str(Path(preprocessed).resolve())
    os.environ["nnUNet_results"] = str(Path(results).resolve())
