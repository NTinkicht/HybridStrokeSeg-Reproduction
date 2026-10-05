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

The notebook downloads the organizer re-archive, verifies the archive checksum, discovers the labeled SISS cases, loads a FLAIR volume and lesion mask, and exercises the reconstructed nine-feature extractor.

## Current implementation

The repository now contains:

- a robust ISLES 2015 SISS dataset discovery/loading layer for NIfTI/MHA/MHD files;
- a clean-room 9-feature extractor using the paper-stated 3x3 neighborhood, four directions, 1x25 window and threshold 20;
- explicit configuration for ambiguous reconstruction choices;
- unit tests for feature geometry, directional edge behavior, modality discovery and complete-case grouping;
- GitHub Actions CI with `pytest` and `ruff`;
- a browser-only Colab entry point;
- documentation distinguishing manuscript statements from inferred implementation choices.

See `docs/reconstruction_assumptions.md` before interpreting any reproduction result. The original source code is unavailable, so ambiguous details are treated as reconstruction assumptions rather than silently attributed to the authors.

## Data safety

Patient data, medical images, model checkpoints and local experiment outputs are ignored by Git and must not be committed to this public repository.

## Next milestone

Implement the paper-reproduction preprocessing alternatives, deterministic patient-level splits, balanced lesion/non-lesion sampling, MLP and RBF-SVM training, morphology, Dice/precision/recall/AUC evaluation, and sensitivity analysis over the underspecified choices.
