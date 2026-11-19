# ISLES2024 WU-20D — full-native-volume CPU qualification

**Decision: technical exploration only. `training_ready=false`.** There are no trained 506/507 checkpoints and no clinically validated mask. No model inference, training, or performance-based selection is authorized.

## Motivation and baseline

Notebook 22 (WU-20C) verified nnU-Net 2.8.1's actual `noNorm` path, frozen 3d_fullres plans and paired NCCT equality for one *96 × 96 × 24* ROI in a held-out scan. That ROI did **not** contain any MTT NaN/Inf values. The original intensity audit identified 9 non-finite MTT voxels in `sub_stroke0016` and 225,045 in `sub_stroke0113`, and extreme negative outliers in additional cases. WU20D checks the **entire native NIfTI** and the full output of the official nnU-Net preprocessor, thereby addressing the missing data condition without using reference labels.

## Frozen design

- Original Dataset504 and Dataset505 remain read-only; original frozen patient fold hash **`664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027`**.
- Exact original Dataset505 nnU-Net plan hash **`6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca`**.
- Pinned nnU-Net **2.8.1** with actual `NoNormalization` mapping; no second normalization of pre-transformed inputs.
- Four prespecified audit cases: `sub_stroke0016` (fold 2), `sub_stroke0113` (fold 1), `sub_stroke0137` (fold 4), `sub_stroke0152` (fold 3). Each is **held out of its own fold** and transformed using its own 119-/120-case **training-only** WU20B parameter manifest.
- Candidate Dataset506 is NCCT only; 507 contains the same NCCT plus Tmax, CBF, CBV, MTT. NCCT NIfTI files are copied byte-for-byte from 506 to 507 before `DefaultPreprocessor.run_case` runs on each arm separately.
- The provisional NCCT image-only head-tissue support is **not a brain mask**. No validation lesion masks, DWI, ADC, clinical outcomes, OOF predictions, or pretrained checkpoints are used.

## Per-case checks

1. Verify frozen split, plans SHA and fold manifest identity, training IDs, excluded validation IDs, candidate parameters and source image geometry.
2. Cryptographically hash five native input NIfTIs and fold manifest. Resumable private results fail closed when source signatures change.
3. Measure **full-volume** MTT/other-map finite/non-finite and extreme raw voxels, distinguishing those **inside** versus **outside** the NCCT-derived provisional support. Independently check the expected raw nonfinite MTT counts of 9 and 225,045 for the two published high-priority cases.
4. Apply unchanged WU19/20 train-only candidate transform and record how many non-finite voxels were present; ensure all resulting maps are finite. This remains a **candidate** finite-value replacement/clipping policy, not an approved clinical policy.
5. Write temporary full-size NIfTI images to CPU local scratch, preserving original voxel geometry, qform and sform. No NIfTI inputs/results go to GitHub or to the persistent project results folder.
6. Run official nnU-Net **`DefaultPreprocessor.run_case`** on held-out full native images, `seg_file=None`, with separate locked 506/507 plans. Require finite output, exactly paired spatial extents and crop boxes, NCCT max difference ≤1e−5. Report any post-resampling perfusion values with |value| > 8.0001; these trigger `FULL_VOLUME_TECHNICAL_REVIEW_REQUIRED` rather than silently reclipping or treating as passed.
7. Persist only pseudonymized private per-patient JSON summaries to Google Drive. One process per patient limits memory accumulation and isolates failures. Re-run skips valid same-source completed checkpoints and refuses stale checkpoints for manual audit.

## Execution and limits

Notebook23 CPU `Run all` verifies source files, installs pinned nnU-Net 2.8.1, retrieves immutable repository code, invokes four cases **sequentially in distinct Python processes**, and validates four per-patient summaries. If Google Colab disconnects, rerun; completed source-hashed checkpoints are reused. Temporary images remain exclusively in `/tmp/` or Colab `/content/` and are deleted on normal completion or Python process exit.

**Outputs:** `MyDrive/HybridStrokeSeg/ISLES2024/results/wu20d_full_native_volume_qualification/private_cases/`, a private execution log and aggregate case-status summary. Raw scans remain untouched.

**This is not a full model inference smoke:** no 506/507 weights exist. It also cannot replace independent blinded anatomical-mask review of 15 overlays, justified clipping/finiteness policy approval, or prospective freeze/sign-off. Even if all four cases pass, no training is authorized until those independent gates are resolved and a no-label full-fold staging/inference integration is approved.
