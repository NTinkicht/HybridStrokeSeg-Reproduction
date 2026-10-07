# Research strategy: 2015 reproduction to ISLES'24 modernization

## Scientific position

The original manuscript must be interpreted as an ISLES 2015 SISS study, despite repeatedly naming the dataset as "ISLES 2022". Its cohort size, modalities, challenge history, and comparison table correspond to ISLES 2015.

The project therefore separates historical reproducibility from modern clinical benchmarking.

## Track A: historical reproduction - ISLES 2015

Purpose: determine how much of the reported handcrafted MLP/SVM pipeline can be reconstructed from the manuscript.

Status: mature enough to freeze as the historical evidence layer.

The reproduction has already quantified sensitivity to patient split, run-length interpretation, intensity scaling, SVM scaling/hyperparameters, morphology, and deliberately invalid within-patient pixel leakage. These results should be reported as reproducibility evidence, not optimized until the paper's published Dice is matched.

Further historical diagnostics remain available in the repository, but they are no longer the primary research direction.

## Track B: bridge validation - real ISLES 2022

Purpose: provide a clean intermediate MRI benchmark between the historical 2015 SISS setting and the 2024 longitudinal CT prediction task.

ISLES 2022 is not claimed to be the dataset used by the original manuscript.

The existing repository support for ISLES 2022 remains useful for:
- verifying the correction from 2015 to real 2022;
- evaluating modern 3-D MRI segmentation on DWI/ADC/FLAIR;
- providing an optional bridge experiment when compute and time permit.

It is secondary to the new primary modernization target.

## Track C: primary modernization - ISLES'24

ISLES'24 becomes the main forward-looking benchmark.

This is not simply a newer version of the same segmentation task. It predicts the final post-treatment infarct from pre-interventional acute information.

Primary inputs:
- NCCT;
- CTA;
- CT perfusion and/or perfusion maps;
- baseline demographic and clinical variables where permitted by the experiment.

Target:
- binary final infarct mask registered to NCCT space and derived from follow-up imaging.

Strict anti-leakage rule:
- follow-up DWI and ADC are target-generation context only;
- post-treatment variables and 3-month outcomes must never enter the prediction model;
- the final lesion mask must never influence preprocessing, channel selection, patient splitting, or hyperparameter selection for held-out cases.

## Core research questions

RQ1. Historical reproducibility:
How closely can the original ISLES 2015 handcrafted MLP/SVM method be reconstructed from the published description, and which underspecified choices materially affect performance?

RQ2. Temporal and modality shift:
How does the problem change when moving from subacute MRI lesion segmentation to hyperacute CT-based final-infarct prediction?

RQ3. Modern baseline:
How strong is a rigorously validated 3-D nnU-Net v2 baseline on ISLES'24 using only pre-interventional imaging?

RQ4. Multimodal contribution:
What is the incremental value of CTA and perfusion maps over NCCT alone?

RQ5. Clinical fusion:
Do baseline clinical variables improve final-infarct prediction beyond imaging alone under patient-level cross-validation?

RQ6. Reproducibility lessons:
Which methodological choices from the historical paper remain useful, and which fail under a modern multicenter longitudinal benchmark?

## Predeclared ISLES'24 experiment ladder

1. Data audit and leakage gate.
2. NCCT-only nnU-Net v2 baseline.
3. NCCT + perfusion maps.
4. NCCT + CTA + perfusion maps.
5. Imaging + baseline clinical feature fusion.
6. Optional raw 4-D CTP experiment only after the registered-map baselines are stable.
7. Compute-matched secondary architecture only after the nnU-Net baseline is complete.

Ablations must be compared on identical patient folds and with the same evaluation code.

## Validation

Use patient-level cross-validation on the 149 public training cases.

All preprocessing decisions, normalization statistics, thresholding, model selection, and clinical-variable handling must be learned inside the training folds.

No slice-level or voxel-level random splitting is permitted in the primary modern results.

Where possible, preserve the official hidden test set as an external benchmark. If official challenge submission is no longer available, report cross-validation transparently and do not imply challenge-test performance.

## Paper framing

The paper should not be framed as "we updated an ISLES 2022 method to 2024."

The defensible framing is:

"From ISLES 2015 to ISLES'24: Reproducibility, Dataset Shift, and Modernization of Ischemic Stroke Lesion Modeling."

The historical and modern tracks answer different questions and must not be collapsed into a single directly comparable leaderboard table.
