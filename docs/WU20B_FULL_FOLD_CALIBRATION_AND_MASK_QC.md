# ISLES2024 — WU-20B: Five-fold candidate calibration and anatomical support QC

**Scope:** CPU-only read-only-source research calibration, no model training. **Status: NOT APPROVED FOR GPU TRAINING.**

## Why this work is required

The frozen Dataset505 preprocessing audit counted 17,958 preprocessed MTT voxels with absolute z-score above 100 across 125/149 patients. Notebook20 demonstrated that a head-tissue proxy and proposed clipping/sanitation produce finite sample outputs for six training patients and held-out `sub_stroke0016`. Crucially, the pilot proxy covered only 7.38% of the held-out image, which requires independent review. Neither numerical cleanliness nor a successful limited pilot validates the anatomical support, normalizer policy, or segmentation benefit.

## Deliverable: Notebook21 + deterministic script

`calibrate_isles2024_all_folds.py` reads **only** Dataset505's acute five-channel `imagesTr/*.nii.gz` source, the frozen original `splits_final.json`, and Dataset505 `nnUNetPlans.json`. It does **not** read `labelsTr`, validation outcomes, DWI/ADC, existing model checkpoints, or segmentation predictions.

1. Frozen split SHA256 check: `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027`; original Dataset505 channel contract and 3D full-resolution normalization plans checked.
2. Each of the 149 patients is independently processed, with source path/size/mtime signatures and SHA256 of immutable per-case sampled checkpoint. Only finite image-only NCCT support voxels contribute. Sampling is deterministic: max 4,096 support voxels per modality per patient, identical selection procedure across all folds. A corrupt or changed checkpoint is recalculated, not silently resumed.
3. After **all** 149 cases successfully pass input/QC checks, fit candidate parameters for five folds separately: 119/120 training patients and exactly their held-out 30/29 patients explicitly excluded per fold. Existing WU19 `fit_fold_channel` enforces this patient manifest. No parameter values are copied from Notebook20's six-case pilot.
4. Save five private, SHA/provenance-bearing manifests with parameters for NCCT and perfusion, and matching training/excluded patient lists. Dataset506 and Dataset507 in the **same fold** must reuse *one identical NCCT parameter manifest and transformed NCCT image*. These are not released or used for training yet.
5. Output private image-only NIfTI mask overlays for `sub_stroke0016`, `sub_stroke0113`, `sub_stroke0137`, `sub_stroke0152`, the six lowest support-coverage cases and the six highest. Overlays use only native-axis NCCT sections (not anatomically reoriented) and the provisional support mask; **no labels**. Automatic coverage flags below 10% and above 40% are descriptive review triggers, not established clinical thresholds.
6. Propose a JSON architecture lock from the **original** Dataset505 3D full-resolution plans (spacing, patch size, batch size, architecture and interpolation). This is *not* an actual locked Dataset506/507 plan, and must be compared with real new nnU-Net plans in a later end-to-end qualification.

## Privacy and safety

All 149 five-channel samples, per-case metadata, case lists, NIfTI-derived PNG overlays, and full-fold candidate parameters are **private** and remain under `MyDrive/HybridStrokeSeg/ISLES2024/results/wu20b_full_fold_calibration/`.

Only source code, safe regression tests, and this method document are committed to the public GitHub repository. Patient data and image montages must never be uploaded to public GitHub. The script never writes into Dataset504, Dataset505 or any `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` model folder.

## Interpret results carefully

- Full-fold **parameter fitting** is **not** an approval of the mask. The proxy includes some extracranial tissue and may exclude pathology; an independent clinical/anatomical review is required, including low coverage `sub_stroke0016`.
- q0.5/q99.5 winsorization, robust IQR scale and ±8 cap remain *candidate values*, not verified clinical limits. The choice must be locked without access to held-out segmentation outcomes.
- Sample-based quantiles are 4,096 finite voxels per patient. They are reproducible estimates, not exhaustive voxel-level statistics.
- Patient images reused for other folds as training data are allowed; each fold's parameters are fitted only from that fold's 119/120 training patients. This is an out-of-fold evaluation, not an external test set.
- Per-file signature is path name, size, and modification timestamp; a separate SHA256 covers the derived per-case sample file. These are not full raw-image content hashes. Do not claim proof of raw-image byte immutability without a full input content hash audit.
- The normalizer is **not inference-ready** until the exact same fold manifest is loaded in held-out prediction. Inference cannot silently recompute parameters from held-out data.
- No nnU-Net planning/preprocessing, no model training, and no new Dataset506/507 full-size image staging are performed here.

## Result gates

`WU20B_progress.json` must say `COMPLETE` with 149 verified patients, and `WU20B_aggregate.json` must say `FULL_FOLD_PROVISIONAL_FIT_COMPLETE` with exactly five fold manifests. **Even when both pass, `training_ready=false`**.

Blockers before GPU work:

1. Independent clinical review of all flagged NCCT supports and normalization candidate bounds, including anatomically representative scans and site/coverage diversity.
2. Freeze a reviewed, label-free anatomical support and handling policy *without* optimizing against previous/held-out performance results.
3. Test pinned nnU-Net 2.8.1 `noNorm`, 3D architecture/spacing, preprocessing, and validation inference with the **same fold-specific manifest**. Check resampling overshoot/finite values and original output geometry.
4. Complete a no-leakage, on-disk fold-2 training/inference integration smoke with synthetic data before real GPU training.
5. Record and sign a final release QC manifest and version, then stage all five folds with verified inventories and immutable inputs.

**Default outcome:** even after all 149 cases have been processed, `training_ready=false` is correct and expected.
