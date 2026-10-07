# HybridStrokeSeg-Reproduction

Reproduction and modernization of **Deep Hybrid Learning for Ischemic Stroke Lesion Segmentation in MR Sequences**.

## Scope

This repository has three deliberately separated tracks:

1. **Historical reproduction - ISLES 2015**: reconstruct the paper's handcrafted 9-feature pixel-classification pipeline with an MLP and RBF-SVM.
2. **Bridge validation - real ISLES 2022**: retain a modern MRI benchmark that verifies the dataset correction and supports optional 3-D DWI/ADC/FLAIR experiments.
3. **Primary modernization - ISLES'24**: predict final post-treatment infarct from pre-interventional acute CT/CTA/CTP-derived imaging and baseline clinical data.

## Important dataset correction

The manuscript repeatedly refers to "ISLES 2022", but the reported dataset characteristics (28 training subjects; FLAIR, T1, T2 and DWI; co-registration to FLAIR) correspond to the **ISLES 2015 SISS** challenge, not the actual ISLES 2022 dataset.

Accordingly:

- the reproduction track targets **ISLES 2015 SISS**;
- real **ISLES 2022** is retained as an optional bridge MRI benchmark, not as the dataset used by the manuscript;
- **ISLES'24 is the primary forward-looking modernization benchmark**;
- historical, 2022, and 2024 results are never mixed as if they were the same task;
- reconstruction assumptions for underspecified manuscript details are explicitly documented and configurable.

## Run in Google Colab

You do not need Git or a local Python installation.

### Historical ISLES 2015 reproduction

[![Open historical reproduction in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTinkicht/HybridStrokeSeg-Reproduction/blob/main/notebooks/01_reproduce_isles2015.ipynb)

This notebook downloads and verifies the historical dataset, validates the 28 labeled SISS cases, exercises the reconstructed nine features, and can run both a quick smoke experiment and the full classical reproduction.

### ISLES 2022 modernization audit

[![Open ISLES 2022 audit in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTinkicht/HybridStrokeSeg-Reproduction/blob/main/notebooks/02_modernize_isles2022.ipynb)

This notebook downloads the public 250-case ISLES 2022 training release, verifies its published checksum, discovers the BIDS-style cases and audits whether ADC, FLAIR and lesion masks already share the DWI voxel grid before any multimodal 3-D training is attempted.

### First modern baseline: nnU-Net v2.8.1 on DWI + ADC

[![Open nnU-Net baseline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTinkicht/HybridStrokeSeg-Reproduction/blob/main/notebooks/03_nnunet_isles2022_dwi_adc.ipynb)

This notebook prepares a geometry-gated DWI+ADC nnU-Net v2 experiment, runs dataset integrity verification and preprocessing, installs the repository's deterministic patient-level 5-fold split, and provides fold-training/model-selection commands. Full five-fold training is intentionally not auto-started because it is an expensive GPU experiment.


### Primary modernization target: ISLES'24

ISLES'24 changes the problem from same-timepoint MRI lesion segmentation to **longitudinal final-infarct prediction** from pre-interventional acute data. The public training release contains 149 labeled cases and is approximately 99 GB.

The repository deliberately does not auto-download this archive. After obtaining and extracting it, audit the release with:

```bash
python scripts/audit_isles2024.py \
  /path/to/ISLES24 \
  --json outputs/isles2024_audit.json
```

The audit keeps valid acute model inputs separate from follow-up DWI/ADC and outcome information. Follow-up MRI and post-treatment/outcome variables are never permitted as prediction inputs.


After the audit passes, stage the first NCCT-only nnU-Net v2 baseline:

```bash
python scripts/stage_isles2024_nnunet.py \
  /path/to/ISLES24 \
  "$nnUNet_raw" \
  --channels ncct
```

The resulting deterministic five-fold split is generated from patient IDs only.

## Historical reproduction implementation

The repository contains:

- robust ISLES 2015 SISS discovery/loading for NIfTI/MHA/MHD files;
- the reconstructed 9-feature FLAIR descriptor;
- configurable FLAIR preprocessing alternatives for sensitivity analysis;
- deterministic patient-level splitting, using 19 train / 9 test cases when all 28 SISS cases are present;
- balanced lesion/non-lesion sampling, up to the paper's reported 15,000 examples per class;
- a 3x100 sigmoid MLP reproduction architecture and an RBF-SVM baseline;
- dilation-then-erosion morphological closing;
- per-case Dice, precision, recall and continuous-score ROC-AUC;
- CSV summaries plus JSON split and experiment metadata;
- unit tests and GitHub Actions CI;
- a browser-only Colab workflow.

See `docs/reconstruction_assumptions.md` before interpreting any numerical result. The original source code is unavailable, so ambiguous details are treated as reconstruction assumptions rather than silently attributed to the authors.

## Modernization implementation

The primary modernization target is now ISLES'24. ISLES 2022 remains available as a bridge benchmark.

The modernization track now contains:

- verified downloader and audit support for the public ISLES 2022 bridge release;
- ISLES'24 case discovery and NCCT-grid audit support with an explicit anti-leakage boundary between acute inputs and follow-up target-generation data;
- BIDS-style DWI/ADC/FLAIR/mask case discovery;
- voxel-grid geometry auditing for every modality against DWI;
- deterministic patient-level 5-fold cross-validation;
- the official ISLES 2022 primary metrics: Dice, absolute volume difference, absolute lesion-count difference and lesion-wise F1;
- geometry-gated nnU-Net v2 dataset staging and prediction evaluation;
- a pinned **nnU-Net v2.8.1** modern environment and a reproducible DWI+ADC baseline configuration;
- MedNeXt reserved for a later compute-matched secondary comparison;
- a 2026 reference-refresh map including ISLES 2015, ISLES 2022, nnU-Net, nnU-Net Revisited, MedNeXt, DeepISLES, WHO 2025 and the World Stroke Organization 2025 fact sheet.

Read `docs/research_strategy.md`, `docs/isles2024_protocol.md`, `docs/modernization_protocol.md`, and `docs/reference_refresh.md` before interpreting the modern experiments.

## Historical reproduction commands

After the ISLES 2015 dataset has been extracted, the paper-like reconstruction is:

```bash
python scripts/run_reproduction.py \
  --protocol paper-like \
  --target-per-class 15000 \
  --models mlp svm \
  --output-dir outputs/paper_like
```

A label-independent slice-selection sensitivity experiment is:

```bash
python scripts/run_reproduction.py \
  --protocol leakage-free-slice \
  --target-per-class 15000 \
  --models mlp svm \
  --output-dir outputs/leakage_free_slice
```

### Scientific warning

The manuscript says that one FLAIR slice per patient was used but does not state how it was chosen. The `paper-like` protocol therefore uses the slice with the largest ground-truth lesion area as an explicit reconstruction assumption. This is **oracle, label-informed slice selection** and must not be presented as an unbiased clinical evaluation. The alternative `leakage-free-slice` protocol selects the middle non-empty brain slice without reading the lesion mask, but it can miss lesions.

The paper also reports **scaled conjugate-gradient** MLP training. scikit-learn does not provide SCG, so the Python implementation currently uses LBFGS by default. The architecture and sigmoid activation match the manuscript, but the optimizer does not; this mismatch is automatically written to `experiment_metadata.json`.

## Historical outputs

Each run writes:

- `per_case_metrics.csv`
- `summary_metrics.csv`
- `split.json`
- `experiment_metadata.json`

The split is performed at patient level before any pixel sampling, preventing train/test voxel leakage.

## ISLES 2022 data audit

After extracting the public modern dataset:

```bash
python scripts/audit_isles2022.py \
  data/raw/isles2022 \
  --json outputs/isles2022_geometry.json
```

The public ISLES 2022 release is distributed in native space. A geometry mismatch is therefore not treated as a dataset error; it is a signal that registration/resampling must be explicit before a multimodal 3-D model is trained.

## First modern baseline commands

Install the pinned modern environment:

```bash
python -m pip install -e '.[modern]'
```

Stage DWI+ADC only if both channels and the label pass the DWI-grid geometry gate:

```bash
python scripts/stage_isles2022_nnunet.py \
  data/raw/isles2022 \
  "$nnUNet_raw" \
  --dataset-id 501 \
  --dataset-name ISLES2022_DWI_ADC \
  --channels dwi adc
```

Then use nnU-Net's integrity check and 3-D full-resolution planner:

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres
```

Train folds 0-4 with `--npz`; the exact protocol is recorded in `configs/isles2022_nnunet_dwi_adc.json` and demonstrated in `notebooks/03_nnunet_isles2022_dwi_adc.ipynb`.

## Data safety

Patient data, medical images, model checkpoints and local experiment outputs are ignored by Git and must not be committed to this public repository.

## Next milestone

Audit the real ISLES'24 public training release, then stage a leakage-safe **NCCT-only nnU-Net v2 baseline** on deterministic patient-level folds. After that, add registered perfusion maps, CTA, and finally baseline clinical fusion as controlled ablations. ISLES 2022 remains an optional bridge experiment; the historical ISLES 2015 reproduction is frozen as a separate evidence track.
