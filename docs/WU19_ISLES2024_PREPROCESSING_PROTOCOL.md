# WU-19 — Controlled ISLES2024 preprocessing-correction design

**Protocol revision:** WU19-1.0 · **Status:** design/preflight qualified; **not yet authorized for new model training**.

## Motivation and verified evidence

The existing **149-patient, 5-fold** OOF experiments (Dataset504 NCCT; Dataset505 NCCT+Tmax/CBF/CBV/MTT) are frozen. Raw Dataset505 contains non-finite MTT voxels in **21/149** cases, MTT below −10,000 in **41/149**, and the persisted nnU-Net 3D full-resolution preprocessing contains normalized MTT `|z|>100` in **125/149** cases (17,958 voxels) and no non-finite voxels. These are quality indicators, not proof of a causal performance effect. The original channel metadata names NCCT as `NCCT`, and nnU-Net's 3D-fullres plan used **ZScoreNormalization for all five channels**.

Previous matched OOF mean Dice improved from 0.173835 to 0.220835 with perfusion, but its 4-endpoint Holm-adjusted p≈0.1225; lesion F1 also did not survive correction. A correction must not be picked based on this observed difference.

## Controlled experiments

| Dataset | Input | Treatment | Comparability |
|---|---|---|---|
| 504 | NCCT | Original ZScore pipeline | Frozen; no changes |
| 505 | NCCT + perfusion | Original ZScore pipeline | Frozen; no changes |
| 506 | NCCT | CT-style normalization with training-only calibration for each CV fold | New NCCT control |
| 507 | NCCT + perfusion | **Identical fold-specific NCCT transform as 506** + robust perfusion processing | New perfusion arm |

Primary contrast: **507 vs 506** on the same patients and frozen folds. Secondary contrasts: 506 vs 504; 507 vs 505. Never infer that original NCCT ZScore was *mathematically invalid*; it was a different, potentially suboptimal choice for quantitative CT intensities.

## Normalization protocol (candidate; requires pilot qualification)

**Critical leakage boundary:** for each fold, calculate cohort-level clipping/scale parameters using **only** its 119 or 120 training cases, excluding 29/30 held-out cases. Store a fold-specific manifest (fit-case IDs, code revision, sampled voxel support method, parameters and checksums). Validation/inference always load the corresponding fold's frozen training manifest.

1. **Image-only anatomical support:** derive a suitable support mask from NCCT anatomy, without lesion labels, final infarct masks, DWI/ADC, or validation outcomes. Verify masks independently for skull, air, acquisition artifacts, extreme truncation, registration and crop/affine consistency. This mask extraction is a WU20 release gate and must be consistent across 506 and 507.
2. **NCCT, both arms:** provisional nnU-Net CT-*style* window from train-only anatomical-support voxel percentiles 0.5/99.5. After clipping use mean/std fitted from training support. This is **not** the same as allowing nnU-Net to compute a single pooled plan across all 149 cases. The parameters (and support) are fold-specific. If matching nnU-Net's standard `CTNormalization` implementation is intended, explicitly validate the stat semantics against the **pinned nnU-Net release** before claiming equivalence.
3. **Perfusion maps (507 only):** training-only candidate q0.5/q99.5 winsorization; reference median and IQR/1.349 robust scale measured inside the same image-only support. Detect and replace NaN/Inf by fold-training median **inside support only**, and preserve a defined zero background outside support. Saturate extreme normalized values at a provisional absolute z of 8 **only as a candidate**; justify/qualify clinically and on training data, rather than tuning from OOF outcome performance. Apply a finite-preserving transform before any interpolation to prevent extreme/sentinel values propagating through resampling.
4. **Zero/degenerate handling:** refuse unusable support masks, too few finite training values, nonfinite fit coefficients, collapsed distribution/IQR, unexpected channel shape/affines and incomplete patient inventory; no silent fallback that changes different patient populations differently.
5. **nnU-Net integration:** avoid accidental second normalization; implement appropriate fold-specific preprocessing and inference, with matching normalization manifests, no pooled 149-case statistics. This may require custom preprocessor or separate fold-specific data identifiers/plans; **do not** stage 506/507 with a standard all-patients plan and call it leakage-free.

## Mandatory qualification gates before WU20 model training

- [ ] Choose and validate label-independent anatomical-support mask extraction on representative difficult cases, including `sub_stroke0016`, `sub_stroke0113`, `sub_stroke0137`, and `sub_stroke0152`.
- [ ] Fit all candidate transforms per fold using only the frozen training IDs; demonstrate validation perturbations do not change fitted parameters.
- [ ] Compare preserved original raw and candidate transformed data, both pre- and post-resampling, on sample cases; verify NIfTI labels, shape, affine, and orientation unchanged.
- [ ] Fully characterize nonfinite replacement and missingness patterns (especially high-volume NaN/Inf MTT) and create clinically reviewable before/after montages.
- [ ] Verify identical NCCT arrays/transform parameters for 506 and 507, per fold.
- [ ] Verify nnU-Net training **and inference** use corresponding fold-trained preprocessing; no default pooled channel normalization applied afterwards.
- [ ] Execute a no-GPU miniature synthetic/patient-free pipeline test including one train and one held-out case, then a bounded real-data pilot QC run (no outcome-guided selection).
- [ ] Freeze predeclared preprocessing, random seeds, checkpointing, architectures and evaluation plan before training.

## Future evaluation and reporting

Use five identical patient-disjoint folds and 1,000 epochs/fold (subject to documented compute reproducibility constraints). Four endpoints: Dice, lesion-wise F1, absolute lesion volume difference, absolute lesion-count difference; paired patient bootstrap CIs and four-endpoint Holm correction, plus fold and lesion-size sensitivities. Report training costs and test conditions. Do not interpret these local OOF metrics as official organizer hidden-test challenge scores. Historical model runs may have used temporary local preprocessing caches; the saved Drive tensor audit does not prove exact byte identity of historical training inputs.

## Implementation scope

WU19 provides executable **read-only preflight** and pure candidate transform functions, but does **not** generate Dataset506/507 arrays, choose final clinical clipping thresholds, alter input images, or train models. All private patient/manifests remain in the ISLES2024 Google Drive workspace.

## Sources and provenance

- Frozen split SHA256: `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027`.
- Private Drive raw audit: `results/perfusion_intensity_audit/`; preprocessed audit: `results/perfusion_preprocessed_tensor_audit/`.
- nnU-Net official reference: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation/intensity-normalization.md
- nnU-Net normalization source: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/normalization/default_normalization_schemes.py
