# ISLES2024 WU-20 — Fold-isolated preprocessing and staging pilot

**Scope:** CPU-only integration pilot; **status:** NOT APPROVED FOR MODEL TRAINING.  
**Baseline freezes:** Dataset504 and Dataset505 untouched.  
**Primary future comparison:** Dataset507 versus Dataset506, paired 149 OOF cases, exactly five original folds.

## Scientific rationale

Notebook 17 found extreme and non-finite native MTT map values. Notebook 18 detected 17,958 preprocessed MTT voxels at |z| > 100 across 125/149 patients. Neither finding by itself proves a causal effect on stroke-lesion segmentation, and the original four-endpoint adjusted comparison does not demonstrate a statistically significant perfusion advantage. Any choice of robust clipping/normalization must be decided without looking at validation labels or the observed OOF results.

WU-19 documented train-fold-specific CT-style/robust perfusion candidate policies. This WU-20 PR **implements** those policies in an executable, read-only-source staging pilot with safeguards. It does not stage all five folds or start training. All patient-level results are kept privately in Google Drive.

## Fold-isolated staging architecture

For each future CV fold `k`, create fully separate runtime roots:

```text
WU20_workspace/fold_k/
  nnUNet_raw/
    Dataset506_ISLES2024_NCCT_CTCTRL/
      dataset.json                   # one channel named 'noNorm'
      imagesTr/                      # 119 or 120 TRAINING patients, NCCT only
      imagesTs/                      # 30 or 29 HELD-OUT patients, NCCT only
      labelsTr/                      # TRAINING labels only; no held-out labels
    Dataset507_ISLES2024_NCCT_PERFUSION_ROBUST/
      dataset.json                   # five channels all named 'noNorm'
      imagesTr/                      # train NCCT/Tmax/CBF/CBV/MTT
      imagesTs/                      # held-out NCCT/Tmax/CBF/CBV/MTT
      labelsTr/                      # TRAINING labels only
  nnUNet_preprocessed/               # later; must remain fold-local
  nnUNet_results/                    # later; must remain fold-local
```

**Why this matters:** nnU-Net's standard global CT normalization may use pooled foreground intensity statistics from the whole dataset. Here candidate parameters are first calibrated **only from train patient images**, saved as a fold-specific manifest and applied to each arm's input NIfTIs before nnU-Net resampling. The `noNorm` channel identifier is an nnU-Net recognized mapping that ensures preprocessing and prediction do not repeat per-channel normalization. This *must* be pinned and verified in the actual installed nnU-Net version before any training. **Do not run one shared all-149 image dataset through the default planner and claim train-only calibration.**

For model training in a fold-local workspace, nnU-Net's `all` training selection on just those 119/120 images is the intended approach (rather than its default `fold=0` split, which would withhold another portion of the already training-only cohort). Its held-out predictions must come from the fold-local `imagesTs` and **that exact fold's** model/normalization manifest. This intended training/inference integration remains to be qualified; there is **no executable training script in this PR**.

## Implemented prototype normalization

- **Anatomical proxy:** a deterministic, lesion-label-free NCCT head envelope (`HU > -500`, largest connected foreground, 3D hole filling), then a soft-tissue support between `-100` and `150 HU`. This is a **provisional head-tissue proxy, not a validated brain mask**. It may include scalp and exclude portions of pathology or unusual acquisitions. Its shape, coverage, bounding box, and border contacts are recorded, and suspicious coverage is rejected. No validation labels or DWI/ADC are consulted.
- **Training-only parameter fit:** deterministic per-patient support sampling (default max 4096/channel/case) and existing WU-19 clip q0.5/q99.5, NCCT mean/std, perfusion median/(IQR/1.349) parameter computation. Calibration inputs must match an explicit list of train IDs exactly. Validation IDs are forbidden in the input samples.
- **Paired arms:** all raw NIfTI input channels must have matched shapes/affines. A single candidate NCCT transform is saved to Dataset506 and copied byte-for-byte into Dataset507, guaranteeing identical NCCT intensity inputs. Dataset507 additionally sanitizes non-finite perfusion voxels with training-median before clipping/robust scaling. The provisional robust-z cap of 8 is a **research candidate, not a clinical limit**.
- **Source/label integrity:** original NIfTI source files are opened read-only. Generated NIfTI files are float32 and checked for finite values and original shape/affine. Only *training* lesion labels are copied to the output; labels remain unchanged. Validation labels are neither loaded nor staged.
- **Partition integrity:** after staging, files are enumerated and checked against exact train/held-out IDs; the source split hash must match the frozen 149-case SHA256. Invalid or leaked inventories raise exceptions.

## Notebook 20 (bounded pilot)

Colab CPU example: 6 training cases from frozen fold 2 and one held-out case (`sub_stroke0016`). Fits on only the selected 6 training cases; all parameters are explicitly marked **PILOT ONLY**. This **cannot** be used as a full 119-patient fold calibration. It stages local scratch `.nii.gz` files under `/content/`, validates both arms and exports only the private JSON and execution log into `MyDrive/HybridStrokeSeg/ISLES2024/results/wu20_fold_isolated_pilot/`. No new training or model checkpoints are produced.

The only acceptable pilot outcome is `CPU_PILOT_COMPLETE_NOT_TRAINING_APPROVED` with `training_ready: false`, all staging inventories correct, no validation label, no non-finite processed inputs, and exactly identical NCCT output file checksums.

## Remaining release gates (before full five-fold CPU staging/GPU work)

1. Review and validate the NCCT head-tissue proxy on realistic acquisitions, including patients 0016, 0113, 0137, and 0152; have an independent clinical reviewer assess appropriateness for brain tissue statistics.
2. Establish predetermined, clinically defensible missingness and clipping/capping policy on training-only evidence. A high fraction of invalid voxels may indicate acquisition failure and require case QC/exclusion criteria **pre-specified independently of validation performance**.
3. Fit and freeze **all five** 119/120-case training-only parameter manifests, rather than pilot subset values. Produce per-fold source signatures, support QC and exact replayable coefficients.
4. Confirm nnU-Net `noNorm` is in the **pinned training version's** channel mapping. Verify no extra normalization and bounded interpolation overshoot after `nnUNetv2_plan_and_preprocess`. Do not introduce gradients/NaN from cubic interpolation.
5. Lock the `3d_fullres` network, target spacing, patch size, batch size, seeding and training schedule equally across 506 and 507 apart from input channel count. Model planner decisions must not silently differ by modality.
6. Test end-to-end fold-isolated preprocessing, training selection `all`, and inference on `imagesTs`, including exactly one held-out prediction, correct output geometry, and no validation labels entering the training pipeline. Only after passing this should models be trained.
7. Run a blind clinical QC on before/after maps and fix one method before OOF result inspection; do not pick thresholds from 504/505/506/507 validation scores.

## Reproducibility and source-of-truth

- Frozen split SHA256: `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027`.
- Frozen raw source: `nnUNet_raw/Dataset505_ISLES2024_NCCT_PERFUSION`.
- WU-19 source: `src/hybridstrokeseg/wu19_normalization_protocol.py`.
- Candidate WU-20 source: `src/hybridstrokeseg/wu20_fold_preprocessing.py` and `scripts/stage_isles2024_fold_pilot.py`.
- Private output: `results/wu20_fold_isolated_pilot/`.
- Official nnU-Net channel normalization documentation: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation/intensity-normalization.md
- Official normalization mapping: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py

**Do not present WU-20's pilot output as proof that the Dataset506/507 experiment is ready for training.**
