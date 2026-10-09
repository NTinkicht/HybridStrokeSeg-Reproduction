"""Leakage-safe nnU-Net v2 staging for ISLES'24.

Only pre-interventional acute imaging channels may be staged as model inputs.
Follow-up DWI/ADC, outcome variables, and the final infarct mask are never
eligible input channels.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from .data.isles2022 import geometry_equal, geometry_signature
from .data.isles2024 import ISLES24Case, geometry_equal_isles2024
from .splits import make_patient_kfolds

SUPPORTED_ISLES24_CHANNELS = ("ncct", "cta", "tmax", "cbf", "cbv", "mtt")
_CHANNEL_PATHS = {
    "ncct": "ncct",
    "cta": "cta_ncct",
    "tmax": "tmax_ncct",
    "cbf": "cbf_ncct",
    "cbv": "cbv_ncct",
    "mtt": "mtt_ncct",
}


def nnunet_case_id_isles2024(case: ISLES24Case) -> str:
    """Convert an ISLES'24 subject identifier to a stable nnU-Net ID."""
    return case.case_id.replace("-", "_")


def build_isles2024_dataset_json(
    *,
    channels: Sequence[str],
    num_training: int,
) -> dict[str, object]:
    """Build nnU-Net v2 dataset metadata for valid acute channels."""
    channel_list = tuple(str(channel).lower() for channel in channels)
    if not channel_list:
        raise ValueError("At least one input channel is required")
    if len(channel_list) != len(set(channel_list)):
        raise ValueError("Input channels must be unique")
    unsupported = [
        channel for channel in channel_list if channel not in SUPPORTED_ISLES24_CHANNELS
    ]
    if unsupported:
        raise ValueError(f"Unsupported or leakage-prone ISLES'24 channels: {unsupported}")
    if num_training < 1:
        raise ValueError("num_training must be positive")

    return {
        "channel_names": {
            str(index): channel.upper()
            for index, channel in enumerate(channel_list)
        },
        "labels": {"background": 0, "final_infarct": 1},
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }


def audit_isles2024_nnunet_geometry(
    cases: Sequence[ISLES24Case],
    *,
    channels: Sequence[str],
) -> dict[str, list[str]]:
    """Return case -> channels/label that fail the NCCT-grid gate."""
    channel_list = tuple(str(channel).lower() for channel in channels)
    build_isles2024_dataset_json(channels=channel_list, num_training=1)
    failures: dict[str, list[str]] = {}

    for case in cases:
        reference = geometry_signature(case.ncct)
        mismatches: list[str] = []
        for channel in channel_list:
            path = getattr(case, _CHANNEL_PATHS[channel])
            if path is None:
                mismatches.append(f"{channel}:missing")
                continue
            if not geometry_equal_isles2024(reference, geometry_signature(path)):
                mismatches.append(channel)
        if not geometry_equal_isles2024(reference, geometry_signature(case.lesion_mask_ncct)):
            mismatches.append("lesion_mask")
        if mismatches:
            failures[case.case_id] = mismatches
    return failures


def _materialize_label_on_reference(
    source: Path,
    reference: Path,
    destination: Path,
    *,
    resume: bool = False,
) -> None:
    """Write a label with the NCCT header exactly, without resampling voxels.

    The public ISLES'24 release contains one benign qform/sform rounding
    discrepancy. nnU-Net's own integrity checker can be stricter than our
    dataset-specific tolerance, so staged labels are header-canonicalized to
    the corresponding NCCT grid. Voxel values and array indexing are unchanged.
    """
    if resume and destination.exists():
        if geometry_equal(
            geometry_signature(reference),
            geometry_signature(destination),
            atol=0.0,
        ):
            return

    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SimpleITK is required to stage ISLES'24 labels") from exc

    image = sitk.ReadImage(str(source))
    ref = sitk.ReadImage(str(reference))
    if image.GetSize() != ref.GetSize():
        raise ValueError(
            f"Cannot copy NCCT geometry onto label with different size: "
            f"{source} vs {reference}"
        )
    image.CopyInformation(ref)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(destination), useCompression=True)


def _materialize(
    source: Path,
    destination: Path,
    mode: str,
    *,
    resume: bool = False,
) -> None:
    if resume and destination.exists():
        if mode == "copy" and destination.stat().st_size == source.stat().st_size:
            return
        if mode == "symlink" and destination.is_symlink():
            return
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    raise ValueError("mode must be 'copy' or 'symlink'")


def stage_isles2024_nnunet(
    cases: Sequence[ISLES24Case],
    output_root: str | Path,
    *,
    dataset_id: int = 504,
    dataset_name: str = "ISLES2024_NCCT",
    channels: Sequence[str] = ("ncct",),
    mode: str = "copy",
    overwrite: bool = False,
    resume: bool = False,
    split_seed: int = 2026,
) -> Path:
    """Stage leakage-safe ISLES'24 images and final-infarct labels for nnU-Net."""
    if not cases:
        raise ValueError("No ISLES'24 cases supplied")
    if not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be between 1 and 999")
    if not dataset_name.replace("_", "").isalnum():
        raise ValueError("dataset_name must contain only letters, numbers, and underscores")

    channel_list = tuple(str(channel).lower() for channel in channels)
    metadata = build_isles2024_dataset_json(
        channels=channel_list,
        num_training=len(cases),
    )
    failures = audit_isles2024_nnunet_geometry(cases, channels=channel_list)
    if failures:
        preview = "; ".join(
            f"{case_id}: {','.join(items)}"
            for case_id, items in sorted(failures.items())[:10]
        )
        suffix = "" if len(failures) <= 10 else f"; ... {len(failures) - 10} more"
        raise ValueError(
            "nnU-Net staging blocked because acute channels or the label do not "
            f"share the NCCT grid. {preview}{suffix}"
        )

    dataset_dir = Path(output_root) / f"Dataset{dataset_id:03d}_{dataset_name}"
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if overwrite:
            shutil.rmtree(dataset_dir)
        elif not resume:
            raise FileExistsError(
                f"Refusing to overwrite non-empty nnU-Net dataset directory: {dataset_dir}"
            )

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_map: dict[str, str] = {}
    nnunet_ids: list[str] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        target_id = nnunet_case_id_isles2024(case)
        if target_id in nnunet_ids:
            raise ValueError(f"Duplicate normalized nnU-Net case identifier: {target_id}")
        nnunet_ids.append(target_id)
        case_map[case.case_id] = target_id

        for channel_index, channel in enumerate(channel_list):
            source = getattr(case, _CHANNEL_PATHS[channel])
            if source is None:
                raise ValueError(
                    f"Case {case.case_id} is missing requested channel {channel}"
                )
            destination = images_tr / f"{target_id}_{channel_index:04d}.nii.gz"
            _materialize(source, destination, mode, resume=resume)

        _materialize_label_on_reference(
            case.lesion_mask_ncct,
            case.ncct,
            labels_tr / f"{target_id}.nii.gz",
            resume=resume,
        )

    folds = make_patient_kfolds(tuple(nnunet_ids), n_splits=5, seed=split_seed)
    splits = [
        {"train": list(fold.train_ids), "val": list(fold.validation_ids)}
        for fold in folds
    ]

    (dataset_dir / "dataset.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "case_map.json").write_text(
        json.dumps(case_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "splits_final.json").write_text(
        json.dumps(splits, indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "STAGING_NOTE.txt").write_text(
        "ISLES'24 leakage gate: only pre-interventional acute imaging channels are "
        "eligible inputs. Follow-up DWI/ADC, outcome variables, and the final infarct "
        "mask must never be staged as input channels. Copy splits_final.json into the "
        "matching nnUNet_preprocessed dataset directory after planning/preprocessing. "
        "Staged labels copy the NCCT header exactly without voxel resampling so "
        "benign qform/sform rounding cannot trip nnU-Net integrity checks.\n",
        encoding="utf-8",
    )
    return dataset_dir
