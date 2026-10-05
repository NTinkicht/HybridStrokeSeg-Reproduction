# Reconstruction assumptions

This repository is a **clean-room reconstruction from the manuscript** because the original source code is no longer available. A manuscript statement is never silently replaced by an implementation choice: missing details are recorded as assumptions and exposed through configuration where practical.

| Component | Manuscript support | Current reconstruction choice | Status |
|---|---|---|---|
| Dataset | 28 subacute cases with FLAIR/T1/T2/DWI; Table 2 says ISLES 2015 | ISLES 2015 SISS labeled training cases | corrected identity |
| Input sequence | FLAIR is explicitly used for the pixel classifier | one axial FLAIR slice per patient | stated/underspecified |
| Input dimensionality | MLP is reported with 9 input features | exactly 9 features | stated |
| Spatial location | 2-D locations are listed | raw row and column indices | inferred, configurable |
| Intensity | intensity feature is listed | preprocessed FLAIR intensity | stated/inferred |
| Neighbourhood | 3x3 box neighbourhood is described, dimensionality unclear | one 3x3 box-mean scalar | inferred |
| Weighted local mean | listed without weights | normalized 3x3 binomial kernel | inferred |
| Directional descriptor | horizontal, vertical, 45°, 135° | four threshold-count features | stated/inferred |
| Directional window | `1x25` | centered window of length 25 | stated/inferred |
| Threshold | 20 | `>= 20` | stated; intensity scale ambiguous |
| Preprocessing | filtering, histogram specification and brain extraction are named | `paper_minimal`: 3x3 median filter, preserve challenge scale, non-zero FLAIR brain support | inferred approximation |
| Alternative normalization | histogram target/range are absent | `robust_uint8` percentile mapping and `zscore_brain` are sensitivity modes | reconstruction sensitivity |
| Train/test split | 19 train / 9 test is stated, but IDs and seed are absent | deterministic 19/9 patient split, seed 2026 | inferred |
| Class balancing | about 15k lesion and 15k randomly selected non-lesion pixels | equal sampling without replacement, up to 15k per class | stated/inferred |
| Slice selection | one FLAIR slice per case is mentioned but rule is absent | `paper-like`: maximum-GT-lesion slice; `leakage-free-slice`: middle-brain slice | unresolved, explicit alternatives |
| MLP architecture | 9 inputs, 3 hidden layers of 100, sigmoid, 2 outputs | 9 -> 100 -> 100 -> 100 -> 2 with sigmoid activations | stated |
| MLP output encoding | lesion `10`, non-lesion `01` | lesion `[1,0]`, non-lesion `[0,1]` | stated |
| MLP optimizer | scaled conjugate-gradient backpropagation | NumPy implementation of Møller SCG; `sigma=5e-5`, `lambda=5e-7` defaults | named algorithm matched; parameters inferred from legacy `trainscg` defaults |
| MLP loss | not reported | two-output sigmoid cross-entropy by default; MSE available as sensitivity | unresolved, configurable |
| MLP initialization | not reported | Xavier-uniform weights, zero biases | inferred |
| MLP input scaling | not reported | `StandardScaler` fit on training pixels only | inferred |
| SVM | RBF kernel and SMO | scaled RBF SVM; `C=1`, `gamma=scale` | stated/inferred |
| Post-processing | dilation followed by erosion | binary closing with a radius-1 disk | stated/inferred |
| Dice | Dice coefficient is reported | both-empty masks score 1.0 | inferred edge case |
| ROC/AUC | ROC is reported | AUC from continuous decision scores, never hard labels | methodological clarification |

## Why the feature count is nine

The interpretation consistent with the reported **9-input MLP** is:

`2 location + 1 intensity + 1 neighbourhood + 1 weighted local mean + 4 directional = 9`.

This does not prove that the unpublished implementation used these exact scalar definitions. It is the smallest reproducible interpretation consistent with the manuscript.

## Critical slice-selection warning

The `paper-like` protocol selects the slice with the largest ground-truth lesion area because the manuscript does not say how its single slice was selected. This is an **oracle, label-informed rule**. It is useful only as a diagnostic reconstruction assumption for comparison with the paper's single-slice numbers. It must not be presented as an unbiased clinical evaluation.

The `leakage-free-slice` protocol instead selects the middle non-empty brain slice without using the lesion mask. It removes that label leakage but can miss lesions, so it is a sensitivity analysis rather than a claim about the unknown original implementation.

## Scaled conjugate-gradient reconstruction

The paper explicitly names scaled conjugate-gradient backpropagation. The reproduction now implements Møller's SCG recurrence directly instead of substituting LBFGS. The defaults `sigma=5e-5`, `lambda=5e-7`, and `min_grad=1e-6` match the documented defaults of MATLAB's legacy `trainscg` function; the manuscript itself does not state whether those defaults were used.

This closes **one** reproducibility gap, but not all MLP ambiguity. The manuscript does not report its loss function, initialization procedure, feature normalization, epoch/stopping settings, or enough code to establish whether a MATLAB network constructor silently added its own preprocessing/transfer-function defaults. For that reason, SCG is now an optimizer match, not a claim of byte-for-byte recovery of the original network.

The command-line interface keeps `lbfgs`, `adam`, and `sgd` as explicit sensitivity alternatives. The default is now `--mlp-solver scg`. For SCG, `--mlp-loss cross_entropy` is the default and `--mlp-loss mse` can be run as a loss-function sensitivity experiment.

## What a numerical reproduction claim still requires

Before saying that the published Dice values have been reproduced, we must run the full dataset and sensitivity analyses over preprocessing, slice choice, threshold/intensity scaling, MLP loss and initialization assumptions, SVM hyperparameters, morphology and random seed. A result close to the paper under one guessed configuration is not, by itself, proof that the original unpublished implementation has been recovered.
