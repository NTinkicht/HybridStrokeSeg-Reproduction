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

- the original single-slice selection rule;
- true histogram specification rather than simple intensity rescaling;
- SVM hyperparameters and feature scaling;
- the exact 3x3 neighborhood and weighted-local-mean definitions;
- the unavailable scaled-conjugate-gradient MLP training behavior;
- morphology structuring element / radius;
- possible voxel-level split behavior implied by the manuscript's contradictory 70/15/15 wording;
- the exact mathematical definition of the four-direction "run-length" feature.

### Leakage-free slice sensitivity

Selecting the middle non-empty brain slice without reading the lesion mask reduced performance further:

| Model | Dice | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.135 ± 0.245 | 0.131 | 0.198 | 0.791 |
| RBF-SVM | 0.139 ± 0.264 | 0.106 | 0.271 | 0.836 |

Several selected middle-brain slices contained no lesion, producing undefined AUC values. This confirms that the unknown original slice-selection rule is a major reproducibility issue.

## Completed diagnostic: intensity scale and directional-feature interpretation

A 12-condition, 2,000-per-class SVM screen tested two preprocessing families and three explicit interpretations of the paper's underspecified four-direction descriptor. This was uncertainty analysis, not parameter tuning.

| Configuration | Dice mean ± SD | AUC |
| --- | ---: | ---: |
| paper-minimal + maximum contiguous run, threshold 20 | **0.244 ± 0.298** | 0.788 |
| paper-minimal + centered contiguous run, threshold 20 | 0.243 ± 0.302 | 0.796 |
| paper-minimal + threshold count, threshold 20 | 0.227 ± 0.302 | 0.798 |
| robust 0-255 + best tested variant | 0.213 ± 0.326 | 0.786 |
| robust 0-255 + worst tested variant | 0.204 ± 0.304 | 0.778 |

The main conclusion is negative but useful: **simple 0-255 rescaling and the three tested run-length interpretations do not explain the manuscript's reported Dice of 0.56 for the SVM.** The paper-minimal variants were actually better than the robust-uint8 variants in this screen. The two contiguous-run interpretations only improved the quick baseline by about 0.016-0.017 Dice.

## Completed diagnostic: undisclosed 19/9 patient split

A predeclared 20-seed screen repeated the same leakage-safe 19-train / 9-test SVM reconstruction with 2,000 lesion and 2,000 non-lesion training pixels per run. No seed was selected after inspection.

The distribution of mean Dice across the 20 patient splits was:

| Statistic | Mean Dice |
| --- | ---: |
| Mean ± SD across splits | **0.237 ± 0.077** |
| Median | **0.238** |
| 2.5th-97.5th percentile | **0.107-0.373** |
| Observed range | **0.087-0.388** |

The manuscript reports SVM Dice **0.56 ± 0.26**. Under the current reconstruction, even the best of the 20 predeclared split seeds reached only **0.388**, so the undisclosed patient split by itself does not explain the reported score. Because this was the quick 2,000-per-class screen rather than the paper-size 15,000-per-class run, this is evidence against split composition as the sole explanation, not a mathematical proof that split composition has no effect.

The experiment writes:

```text
outputs/split_sensitivity/split_sensitivity.csv
outputs/split_sensitivity/split_sensitivity_summary.json
```

## Next diagnostic: SVM scaling, C and gamma

The manuscript says that an RBF-SVM was trained with SMO but does not report feature scaling, the penalty parameter `C`, or the RBF width / `gamma`. These choices can materially alter an RBF-SVM when the nine handcrafted inputs live on very different numerical scales.

A predeclared 27-condition screen is now available. It evaluates:

- scaling: `standard`, `minmax`, `none`;
- `C`: `0.1`, `1`, `10`;
- `gamma`: `scale`, `0.01`, `0.1`.

All other reconstruction choices stay fixed, including the deterministic 2026 patient split, paper-minimal preprocessing, threshold-count directional feature, raw spatial coordinates, radius-1 morphological closing and 2,000 pixels per class.

Run:

```bash
python scripts/run_svm_sensitivity.py
```

It writes:

```text
outputs/svm_sensitivity/svm_sensitivity.csv
outputs/svm_sensitivity/svm_sensitivity_metadata.json
```

This is an uncertainty diagnostic, not score matching. A high-scoring configuration is informative only if the improvement is broad and scientifically plausible, not merely because one grid point happens to approach 0.56.

After this diagnostic, the remaining priorities are true histogram specification, morphology sensitivity, a deliberately labeled leakage diagnostic for the contradictory 70/15/15 voxel-split wording, exact neighborhood/weighted-mean reconstruction, and an exact scaled-conjugate-gradient MLP implementation.
