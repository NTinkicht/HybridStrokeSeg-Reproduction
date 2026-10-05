# HybridStrokeSeg-Reproduction

Reproduction and modernization of **Deep Hybrid Learning for Ischemic Stroke Lesion Segmentation in MR Sequences**.

## Scope

This repository has two deliberately separated tracks:

1. **Historical reproduction**: reconstruct the paper's handcrafted 9-feature pixel-classification pipeline with an MLP and RBF-SVM.
2. **Modernization**: evaluate contemporary 3D stroke-lesion segmentation baselines and updated datasets without mixing their results with the historical reproduction.

## Important dataset note

The manuscript repeatedly refers to "ISLES 2022", but the reported dataset characteristics (28 training subjects; FLAIR, T1, T2 and DWI; co-registration to FLAIR) correspond to the **ISLES 2015 SISS** challenge, not the actual ISLES 2022 dataset.

Accordingly:

- the reproduction track targets **ISLES 2015 SISS**;
- ISLES 2022 will be used only in the modernization track;
- reconstruction assumptions for underspecified paper details are explicitly documented and configurable.

## Run in Google Colab

You do not need Git or a local Python installation. Open the notebook in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NTinkicht/HybridStrokeSeg-Reproduction/blob/main/notebooks/01_reproduce_isles2015.ipynb)

The notebook downloads and verifies the historical dataset, validates the 28 labeled SISS cases, exercises the reconstructed nine features, and can run both a quick smoke experiment and the full classical reproduction.

## Current implementation

The repository now contains:

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

## Reproduction commands

After the dataset has been extracted, the paper-like reconstruction is:

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

## Outputs

Each run writes:

- `per_case_metrics.csv`
- `summary_metrics.csv`
- `split.json`
- `experiment_metadata.json`

The split is performed at patient level before any pixel sampling, preventing train/test voxel leakage.

## Data safety

Patient data, medical images, model checkpoints and local experiment outputs are ignored by Git and must not be committed to this public repository.

## Next milestone

Run the historical experiment in Colab, inspect the actual results, perform sensitivity analysis over the missing manuscript details, and only then decide whether the reported Dice values have been reproduced. The modernization track will remain separate and will use contemporary 3D models and datasets.
