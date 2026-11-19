# WU-20C — nnU-Net 2.8.1 technical qualification of Dataset506/507 preprocessing

**Status: TECHNICAL ROI PILOT ONLY; NOT APPROVED FOR TRAINING.** Original Dataset504/505, checkpoints and historical OOF results are unchanged.

## Origin and scope

WU20B completed all 149 image-only reads and created 5 provisional calibration manifests (119/120 train patients, 30/29 held out). WU20C checks whether the proposed paired preprocessing is compatible with **the actual nnU-Net 2.8.1 runtime**. It does not authorize GPU training.

The frozen Dataset505 plan SHA256 is `6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca`. Its `3d_fullres` settings include `SimpleITKIO`, 2.0 × 0.41015625 × 0.41015625 mm spacing, 32 × 256 × 224 patch, batch size 2, six-stage `PlainConvUNet`, and cubic data interpolation. The technical candidate plans copy the identical 3d_fullres architectural and geometric settings, replacing channel-specific normalization with `NoNormalization` (1 channel for 506; 5 for 507). Global 505 fingerprint intensity statistics are intentionally excluded from candidate plans because they are irrelevant to `NoNormalization` and must not leak training/validation information.

## Exact validation procedure

1. Verify original split/plan hashes, selected held-out ID, all five fold-trained coefficients and exact 119/120 training patient memberships.
2. Require installed nnU-Net **2.8.1** and resolve `noNorm` to its actual `NoNormalization` implementation; run `DefaultPreprocessor._normalize` on patient-free synthetic data and assert numerical identity.
3. Verify both locked 3d_fullres plans preserve SimpleITKIO, transpose, architecture, patch size, spacing, resampling and batch size.
4. Select the held-out patient's **acute imaging only**, with no infarct label access. Generate a deterministic, bounded 96 × 96 × 24 maximum image-only ROI around the NCCT support, with NIfTI affine translations correctly shifted on disk. This is a *technical test*, not full-volume inference or an anatomical selection policy.
5. Apply WU20B fold-trained normalization to candidate NCCT and perfusion ROI images; the NCCT inputs must be identical for 506 and 507.
6. Call nnU-Net's actual `DefaultPreprocessor.run_case` on the label-free held-out images for each arm. Check numeric finiteness, channel count, equal shapes and crop regions, and near-identical post-resampling NCCT values (max abs difference ≤ 1e-5).
7. Measure any perfusion interpolation overshoot beyond the provisional ±8 robust-z cap. Flag for review; do not silently alter results or treat the cap as an accepted clinical threshold.
8. Save **only private JSON QC results** in `results/wu20c_nnunet_runtime_qualification`; transformed NIfTI scratch files are deleted after the CPU test. Generate a private review template for all 15 WU20B mask overlays with initial status `PENDING` and preserve reviewer edits on reruns.

## Scientific limitations and release gates

The proposed anatomical support is a **head-tissue proxy, not a clinically approved brain mask**. The WU20B overlay for case `sub_stroke0016` suggests some extracranial tissue may be included. A medically qualified reviewer must review the 15 overlays while blinded to held-out results, decide whether brain tissue is sufficiently retained, record corrections, and explicitly sign off. The clipping q0.5–q99.5, robust IQR/1.349 scale, nonfinite replacement and optional ±8 cap remain provisional.

This test is limited to a bounded ROI and **does not test full native-volume preprocessing or actual prediction from a trained 506/507 model**. There are no candidate-trained checkpoints; `real_model_inference_performed=false` and `training_ready=false` are explicit in every output. Follow-up gates: signed anatomical review and fixed prospectively approved normalization policy; full-volume matching preprocessor geometry/spacing checks; synthetic prediction export round-trip and eventually real model inference and native-grid export; exact `fold=all` train-only workspace isolation; and evidence that the same frozen protocol is deployed in all five folds.

The user should only execute Notebook22 in CPU Colab after the GitHub PR has green tests and the immutable source commit has been pinned in its cells. **Never infer a trained-model quality improvement from this technical QC.**
