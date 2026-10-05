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
- reconstruction assumptions for underspecified paper details will be explicitly documented and configurable.

## Status

Repository initialization is in progress. Patient data will never be committed to GitHub.
