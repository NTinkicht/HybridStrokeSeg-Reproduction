# Historical reproduction status

## First real-data run

A Google Colab execution on 2026-10-06 validated all **28 labeled ISLES 2015 SISS training cases** and completed the clean-room single-slice reproduction on the deterministic 19/9 patient split.

The current `paper-like` protocol uses the maximum-ground-truth-lesion FLAIR slice for each subject because the manuscript does not state how its single slice was selected. This is an oracle reconstruction assumption and is not a leakage-free clinical protocol.

### Paper-size clean-room result

The run used 15,000 lesion and 15,000 non-lesion training pixels, the reconstructed nine-feature descriptor, a 3x100 sigmoid MLP trained with LBFGS, and an RBF-SVM. The MLP hit its 500-iteration limit and did not converge.

| Model | Manuscript Dice | Current reproduction Dice | Difference | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.59 ± 0.27 | 0.237 ± 0.254 | -0.353 | 0.186 | 0.542 | 0.804 |
| RBF-SVM | 0.56 ± 0.26 | 0.269 ± 0.289 | -0.291 | 0.206 | 0.518 | 0.827 |

These values are **not a successful numerical reproduction** of the manuscript. They are evidence that one or more unavailable implementation details materially affect performance.

The largest unresolved candidates are:

- the manuscript's histogram specification / intensity scale before applying the reported threshold of 20;
- the exact mathematical definition of the four-direction "run-length" feature;
- the exact 3x3 neighborhood and weighted-local-mean definitions;
- the original single-slice selection rule;
- the unavailable scaled-conjugate-gradient MLP training behavior;
- SVM hyperparameters and feature scaling;
- morphology structuring element / radius;
- the original patient split and any voxel-level split behavior.

### Leakage-free slice sensitivity

Selecting the middle non-empty brain slice without reading the lesion mask reduced performance further:

| Model | Dice | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.135 ± 0.245 | 0.131 | 0.198 | 0.791 |
| RBF-SVM | 0.139 ± 0.264 | 0.106 | 0.271 | 0.836 |

Several selected middle-brain slices contained no lesion, producing undefined AUC values. This confirms that the unknown original slice-selection rule is a major reproducibility issue.

## Next experiment: controlled sensitivity, not score matching

The next step is a preregistered-style diagnostic sweep over two especially important ambiguities:

1. intensity preprocessing (`paper_minimal` versus a robust 0-255 mapping, which makes the manuscript's threshold of 20 interpretable on a conventional 8-bit scale);
2. three explicit directional-feature interpretations: threshold count, contiguous run through the center, and maximum contiguous run within the 1x25 support.

The sweep is deliberately framed as uncertainty analysis. A configuration must not be selected merely because its Dice happens to approach the manuscript's reported number.

Run the fast SVM screen with:

```bash
python scripts/run_sensitivity_sweep.py
```

It writes each experiment separately and creates:

```text
outputs/sensitivity_quick/sweep_summary.csv
```

The most scientifically plausible variants can then be repeated at the full 15,000-per-class setting with both MLP and SVM, followed by morphology and optimizer sensitivity.
