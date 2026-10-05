# Modernization protocol: ISLES 2022

This track is intentionally separate from the historical ISLES 2015 reproduction. It is a new, modern benchmark and must not be described as a reproduction of the manuscript's reported Dice values.

## Dataset

Use the public ISLES 2022 training release (250 labeled cases) and preserve the official hidden 150-case test set for challenge/external evaluation. The dataset contains DWI, ADC and FLAIR plus an expert lesion mask, follows a BIDS-style layout, is skull-stripped, and is released in native space without prior registration.

Primary references:

- Hernandez Petzsche MR et al. *ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.* Scientific Data 9, 762 (2022). DOI: 10.1038/s41597-022-01875-5. Dataset DOI: 10.5281/zenodo.7153326.
- Official challenge repository: https://github.com/ezequieldlrosa/isles22

## Mandatory data audit before training

Run:

```bash
python scripts/audit_isles2022.py /path/to/ISLES-2022 --json outputs/isles2022_geometry.json
```

The audit verifies the expected case layout and records whether ADC, FLAIR and the lesion mask already share the DWI voxel grid. Do **not** stack multimodal volumes simply because their array shapes happen to be similar. Registration/resampling must be explicit, deterministic and reported.

## Validation design

The primary development protocol is patient-level 5-fold cross-validation on the 250 public labeled cases. All preprocessing, registration fitting, intensity normalization choices, hyperparameter selection and threshold selection must be learned from training folds only. No voxel, slice or subject from the held-out fold may influence training.

Report fold-wise and pooled summaries with bootstrap 95% confidence intervals. Keep all split manifests under version control, but never commit patient images.

## Metrics

Use the official ISLES 2022 metrics as primary endpoints:

1. Dice score.
2. Absolute volume difference.
3. Absolute lesion-count difference.
4. Lesion-wise F1 score.

Also report precision, recall and HD95 as secondary descriptive metrics. Define connected-component connectivity, minimum-lesion filtering and empty-mask behavior explicitly.

## Baseline hierarchy

### Baseline A: nnU-Net v2

nnU-Net v2 is the primary modern baseline. It is maintained as a strong, self-configuring biomedical segmentation framework and provides a fairer reference point than a lightly tuned custom architecture.

Reference: Isensee F et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods 18, 203-211 (2021). DOI: 10.1038/s41592-020-01008-z.

For current validation practice also cite: Isensee F et al. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* MICCAI 2024, 488-498. DOI: 10.1007/978-3-031-72114-4_47.

### Baseline B: MedNeXt

Use MedNeXt only after the nnU-Net baseline is fully tuned and compute-matched. The comparison must use the same data folds, preprocessing, augmentation budget and evaluation code.

Reference: Roy S et al. *MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation.* MICCAI 2023, 405-415. DOI: 10.1007/978-3-031-43901-8_39.

### External reference: DeepISLES

DeepISLES is not a drop-in training baseline for this repository; it is a clinically validated ensemble derived from top ISLES'22 submissions and is useful as a contemporary reference point for generalizability and clinical utility.

Reference: de la Rosa E et al. *DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES'22 challenge.* Nature Communications 16, 7357 (2025). DOI: 10.1038/s41467-025-62373-x.

## Registration policy

The public release is in native space. Therefore registration is a first-class experimental variable, not a hidden preprocessing detail.

Before implementing the final multimodal training pipeline:

- audit the real release with `scripts/audit_isles2022.py`;
- determine which modalities already share the DWI/label grid;
- register only modalities that require it;
- use deterministic SimpleITK/ANTs parameters recorded in configuration;
- use linear interpolation for intensity images and nearest-neighbor interpolation for masks;
- visually inspect a stratified sample and record registration failures;
- include an ablation comparing DWI+ADC against DWI+ADC+FLAIR.

## Scientific comparison rule

Do not compare the ISLES 2015 handcrafted 2-D classifier and the ISLES 2022 3-D deep-learning models as though they were evaluated on the same benchmark. The data, modalities, disease-stage distribution, cohort size, challenge protocol and hidden-test construction are different. Historical results belong in the reproduction track; modern results belong here.
