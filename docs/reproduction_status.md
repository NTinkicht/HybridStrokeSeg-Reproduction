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

- the exact patient split and any voxel-level split behavior;
- the original single-slice selection rule;
- true histogram specification rather than simple intensity rescaling;
- the exact 3x3 neighborhood and weighted-local-mean definitions;
- the unavailable scaled-conjugate-gradient MLP training behavior;
- SVM hyperparameters and feature scaling;
- morphology structuring element / radius;
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

Therefore the next investigation should move away from trying more threshold values and focus on larger protocol uncertainties.

## Next diagnostic: undisclosed 19/9 patient split

The manuscript gives a 19/9 case-level split but does not identify the patient IDs or random seed. With only 28 cases and very heterogeneous lesion sizes, a single arbitrary split can have a large effect on mean Dice.

Run the predeclared patient-split sensitivity study with:

```bash
python scripts/run_split_sensitivity.py --num-seeds 20
```

Every run remains a leakage-safe 19-train / 9-test patient holdout. The script does **not** select a favorable split. It reports the distribution of mean SVM Dice across seeds and writes:

```text
outputs/split_sensitivity/split_sensitivity.csv
outputs/split_sensitivity/split_sensitivity_summary.json
```

Interpretation rule:

- if the manuscript's 0.56 Dice lies far above the observed split-sensitivity range, the missing patient split cannot plausibly explain the gap by itself;
- if the distribution is extremely wide and approaches the manuscript value naturally across multiple seeds, split composition becomes a major explanatory factor;
- regardless of the result, selecting the best seed after inspection is prohibited because that would be score matching rather than reproduction.

After this diagnostic, the remaining priorities are true histogram specification, SVM scaling/hyperparameter sensitivity, morphology, a leakage diagnostic for the contradictory 70/15/15 voxel-split wording, and an exact scaled-conjugate-gradient MLP implementation.
