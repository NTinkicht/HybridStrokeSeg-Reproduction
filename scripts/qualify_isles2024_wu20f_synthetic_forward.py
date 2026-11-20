#!/usr/bin/env python3
"""WU20F: CPU-only, exact nnU-Net 2.8.1 synthetic UNTRAINED model forward.

This is neither trained-model inference nor training authorization. Uses no patient data.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

from qualify_isles2024_nnunet_runtime import (
    PINNED_NNUNET,
    SOURCE_PLANS_SHA,
    assert_locked_plans,
    build_locked_plans,
)

from hybridstrokeseg.wu20_fold_preprocessing import file_sha256, no_norm_dataset_json

SCHEMA = "WU20F-synthetic-architecture-forward-v1"
SMALL_PATCH = (16, 64, 64)


def assert_no_training_release(result: dict) -> None:
    if result.get("training_ready") is not False:
        raise ValueError("Training readiness cannot be granted by synthetic smoke")
    if result.get("trained_model_inference_performed") is not False:
        raise ValueError("Untrained synthetic forward cannot count as model inference")


def run_forward(source_plans: Path, output: Path) -> dict:
    if file_sha256(source_plans) != SOURCE_PLANS_SHA:
        raise ValueError("Dataset505 frozen plans changed")
    if output.resolve() == source_plans.parent.resolve() or source_plans.parent.resolve() in output.resolve().parents:
        raise ValueError("Do not write into frozen nnU-Net plans directory")
    v = importlib.metadata.version("nnunetv2")
    if v != PINNED_NNUNET:
        raise RuntimeError(f"Expected nnunetv2=={PINNED_NNUNET}; got {v}")
    import torch
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    torch.manual_seed(202420)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    original = json.loads(source_plans.read_text(encoding="utf8"))
    # Identical synthetic NCCT supplied to both architectures; different randomly
    # initialized network weights mean model outputs are NOT comparable outcomes.
    ncct = torch.randn((1, 1, *SMALL_PATCH), dtype=torch.float32)
    additional = torch.randn((1, 4, *SMALL_PATCH), dtype=torch.float32)
    results = {}
    for dataset, n_inputs in ((506, 1), (507, 5)):
        plans = build_locked_plans(original, dataset_id=dataset)
        assert_locked_plans(original, plans, dataset_id=dataset)
        metadata = no_norm_dataset_json(n_train=119, ncct_only=(dataset == 506))
        pm = PlansManager(plans)
        cfg = pm.get_configuration("3d_fullres")
        n_heads = pm.get_label_manager(metadata).num_segmentation_heads
        if n_heads != 2 or cfg.normalization_schemes != ["NoNormalization"] * n_inputs:
            raise ValueError("Unexpected segmentation heads/normalization")
        network = get_network_from_plans(
            cfg.network_arch_class_name,
            cfg.network_arch_init_kwargs,
            cfg.network_arch_init_kwargs_req_import,
            n_inputs,
            n_heads,
            allow_init=True,
            deep_supervision=False,
        ).to("cpu")
        network.eval()
        tensor = ncct if dataset == 506 else torch.cat([ncct, additional], dim=1)
        with torch.inference_mode():
            logits = network(tensor)
        if isinstance(logits, (list, tuple)):
            raise ValueError("Deep supervision unexpectedly active in synthetic inference")
        expected = (1, 2, *SMALL_PATCH)
        if tuple(logits.shape) != expected or not torch.isfinite(logits).all().item():
            raise ValueError(f"Dataset{dataset} nonfinite or invalid forward logits")
        parameters = sum(int(p.numel()) for p in network.parameters())
        if parameters <= 0:
            raise ValueError("Empty synthetic architecture")
        results[str(dataset)] = {
            "input_channels": n_inputs, "segmentation_heads": n_heads,
            "input_shape": list(tensor.shape), "output_shape": list(logits.shape),
            "trainable_parameter_count": parameters,
            "locked_architecture": True, "mode": "untrained_cpu_forward",
        }
        del network, tensor, logits
        gc.collect()
    result = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "status": "SYNTHETIC_FORWARD_PASS_COMPLETE",
              "training_ready": False, "trained_model_inference_performed": False,
              "patient_image_or_label_read": False, "weights_loaded": False,
              "nnunet_version": v, "source_plans_sha256": SOURCE_PLANS_SHA,
              "synthetic_test_patch": list(SMALL_PATCH),
              "same_ncct_input_for_both_arms": True, "arms": results,
              "blocked_release_gates": [
                  "Synthetic untrained-model forward does not prove trained-model inference",
                  "Independent anatomical and preprocessing policy approvals remain unsigned",
                  "No five-fold transformed dataset has been staged",
                  "No training or performance comparison authorized",
              ]}
    assert_no_training_release(result)
    # Import local atomic writer only after the synthetic model has passed.
    from preflight_isles2024_wu20f_staging import atomic_json
    atomic_json(output / "WU20F_synthetic_forward_aggregate.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run_forward(a.source_plans, a.output), indent=2))


if __name__ == "__main__":
    main()
