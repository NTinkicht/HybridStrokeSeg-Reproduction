# WU-20F — 149-case virtual staging and synthetic nnU-Net forward qualification

**Status: technical engineering audit, NOT clinical approval or permission to train.**

## Why this work unit exists

WU20B fitted five 119/120-patient training-only normalizers, WU20C validated nnU-Net v2.8.1 `NoNormalization` on a held-out image region, WU20D passed four **full native-volume** technical tests (including `sub_stroke0113`'s 225,045 missing MTT values), and WU20E established an explicitly unapproved anatomy/clipping policy and review-gate evidence package. WU20E's 15 review decisions remain pending. The next engineering question is: **can the complete 149-image source inventory be prepared and can the locked 506/507 model architectures execute without mixing arms or requiring existing trained weights?**

## Preflight — no patient voxel or label content read

Run `scripts/preflight_isles2024_wu20f_staging.py` in a CPU Colab environment with mounted private Drive source images. It:

- Requires exact 149-patient split SHA256 `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027` and original Dataset505 plans SHA256 `6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca`.
- Requires original Dataset505 **five** source modalities (`NCCT`, `TMAX`, `CBF`, `CBV`, `MTT`), one declared lesion label per case, and no other label categories.
- Independently checks the five WU20B manifests against the exact frozen train/validation IDs, sample counts, and image-only normalization; checks WU20E's **unapproved policy** and canonical policy digest. The unapproved input is a prerequisite for an explicitly blocked output, not a training waiver.
- Opens NIfTI **headers only** for all 149 × 5 = **745 source images**, requiring matched shape/affine within each patient, finite nondegenerate geometry, valid 3D spacing. `labelsTr` files are checked **only for presence, size and file-system metadata**; no label headers/voxels, images' voxel arrays, or OOF/prediction files are accessed.
- Keeps one atomic per-case private JSON header checkpoint, rechecking all five image **and label** metadata signatures on resumed runs. Source signature is `filename, file size, mtime_ns`: **NOT an image-content SHA256** (large source files are not read). A changed signature fails closed rather than silently accepting a stale checkpoint.
- Produces five **virtual**, fold-isolated 506/507 inventories. For each fold, arm 506 declares 149 NCCT inputs and 119/120 training-label references, arm 507 declares 745 five-channel inputs and the **same training-label references**. All 29/30 held-out patient image files are declared as inference-only `imagesTs`. **No held-out labels are read or declared in `labelsTs`**. The future 507 NCCT `_0000` file must be byte-copied from its paired 506 input, never independently re-normalized.
- Emits no usable nnU-Net dataset, `nnUNet_raw` directory, `imagesTr` output, transformed NIfTI, training manifest, or trainable model. The output always has `staging_materialized=false` and `training_ready=false`. If even one case fails, no complete virtual inventory is released.

Expected virtual totals across **all five folds**, both arms: 745 arm-506 image declarations; 3,725 arm-507 image declarations; **596 training-label references per arm** (not 596 distinct labels; many cases are training in four folds). Patient-level details and manifest hashes stay private in Drive. A public-safe aggregate excludes identifiers and other sensitive metadata.

This is a **metadata-only** preflight, not a complete geometry/lesion-label compatibility test. Actual label geometry/semantic classes and all transformation outputs must be verified in a separately authorized staging run after independent approval.

## Synthetic forward — actual nnU-Net 2.8.1 network construction

Run `scripts/qualify_isles2024_wu20f_synthetic_forward.py` with exact nnU-Net v2.8.1 and PyTorch on CPU. It builds each candidate plan with the WU20C function, confirming frozen 3d_fullres geometry, spacing, architecture, `NoNormalization` and two segmentation heads. It instantiates each **real** nnU-Net `get_network_from_plans(...)` architecture from the frozen plans (one NCCT channel and five NCCT/perfusion channels), uses identical synthetic NCCT tensors, and checks finite output logits with two channels and the expected spatial dimensions.

- Uses only synthetic random arrays of shape **1 × channel × 16 × 64 × 64**, chosen to meet stride divisibility while limiting CPU RAM/time; the frozen **actual** patch size of **32 × 256 × 224** is not changed in model plans.
- Runs with `deep_supervision=False`, eval mode and `torch.inference_mode()`.
- Never loads trained model parameters or patient images; **model logits have no medical meaning** and are not evaluated against labels.
- Does not replace trained-model inference qualification. Actual OOF inference and image resampling/reconstruction remain **post-training** gates.

## Blocked research release decisions

1. Independent qualified reviewer assessment of all 15 anatomical overlays (plus a defensible review strategy for remaining 134 cases), resolving `sub_stroke0016`'s 7.38% support and extracranial coverage risks.
2. Independent scientific/clinical approval of train-only percentile clipping, median replacement for invalid perfusion, and final ±8 robust-z cap. Document alternative candidates **without looking at held-out clinical outcomes**.
3. Authentic signed version-controlled prospective protocol with hashes and reviewer credentials verified by people, never inferred from a CSV entry.
4. Separately approved *physical* full-cohort staging, actual label geometry/semantic validation, full preprocessing inventory and inference IO/segmentation reconstruction checks. **WU20F virtual inventories do not satisfy this.**
5. Only after all of the above: explicit training authorization, fivefold 1000-epoch paired 506/507 runs, and prespecified OOF evaluation. Keep Dataset504/505 unchanged.

## CLI example (CPU only)

```bash
PYTHONPATH=src:scripts python scripts/preflight_isles2024_wu20f_staging.py \
  --raw505 /private/nnUNet_raw/Dataset505_ISLES2024_NCCT_PERFUSION \
  --splits /private/nnUNet_preprocessed/Dataset505_ISLES2024_NCCT_PERFUSION/splits_final.json \
  --plans /private/nnUNet_preprocessed/Dataset505_ISLES2024_NCCT_PERFUSION/nnUNetPlans.json \
  --folds /private/results/wu20b_full_fold_calibration/fold_private \
  --wu20e-gate /private/results/wu20e_review_and_release_gate/WU20E_release_gate_aggregate.json \
  --wu20e-policy /private/results/wu20e_review_and_release_gate/WU20E_policy_PROPOSED_NOT_APPROVED_PRIVATE.json \
  --output /private/results/wu20f_virtual_staging_and_forward

PYTHONPATH=src:scripts python scripts/qualify_isles2024_wu20f_synthetic_forward.py \
  --source-plans /private/nnUNet_preprocessed/Dataset505_ISLES2024_NCCT_PERFUSION/nnUNetPlans.json \
  --output /private/results/wu20f_virtual_staging_and_forward
```

The CPU Colab **Notebook25** pins each source file to a specific Git commit. No user action is needed on GitHub after merge; run Notebook25 once and share its reports. Private patient-level files must never be committed publicly.
