# Historical reproduction status

## First real-data run

A Google Colab execution on 2026-10-06 validated all **28 labeled ISLES 2015 SISS training cases** and completed the clean-room single-slice reproduction on the deterministic 19/9 patient split.

The current `paper-like` protocol uses the maximum-ground-truth-lesion FLAIR slice for each subject because the manuscript does not state how its single slice was selected. This is an oracle reconstruction assumption and is not a leakage-free clinical protocol.

### Paper-size clean-room result

The run used 15,000 lesion and 15,000 non-lesion training pixels, the reconstructed nine-feature descriptor, a 3x100 sigmoid MLP trained with LBFGS, and an RBF-SVM. The MLP hit its 500-iteration limit and did not converge.

| Model | Manuscript Dice | Current reproduction Dice | Difference | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.59 ± 0.27 | 0.252 ± 0.260 | -0.338 | 0.197 | 0.585 | 0.830 |
| RBF-SVM | 0.56 ± 0.26 | 0.269 ± 0.289 | -0.291 | 0.206 | 0.518 | 0.827 |

These values are **not a successful numerical reproduction** of the manuscript. They are evidence that one or more unavailable implementation details materially affect performance.

The largest unresolved candidates are:

- the original single-slice selection rule;
- true histogram specification rather than simple intensity rescaling;
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

## Completed diagnostic: SVM scaling, C and gamma

The manuscript says that an RBF-SVM was trained with SMO but does not report feature scaling, the penalty parameter `C`, or the RBF width / `gamma`. A predeclared 27-condition screen tested three scaling modes, three `C` values and three gamma choices while keeping the deterministic 2026 patient split and all other reconstruction settings fixed.

The strongest conditions were:

| Configuration | Dice mean ± SD | AUC |
| --- | ---: | ---: |
| no feature scaling, C=1, gamma=scale | **0.322 ± 0.391** | 0.857 |
| no feature scaling, C=10, gamma=scale | **0.320 ± 0.323** | 0.891 |
| min-max scaling, C=10, gamma=scale | 0.272 ± 0.304 | 0.828 |
| standard scaling, C=10, gamma=0.1 | 0.258 ± 0.291 | 0.813 |
| standard scaling, C=10, gamma=scale | 0.256 ± 0.292 | 0.813 |
| standard scaling, C=1, gamma=scale | 0.227 ± 0.302 | 0.798 |

The screen therefore shows that **feature scaling is materially important**, with the two best no-scaling configurations improving the quick baseline by about 0.09 Dice. However, even the strongest predeclared condition reached only **0.322 mean Dice**, still well below the manuscript's reported **0.56**. Ordinary choices of scaling, `C` and `gamma` therefore do not explain the reproduction gap by themselves.

This result must not be interpreted as recovery of the original SVM settings. The grid was an uncertainty analysis and the top condition was observed after running the full predeclared screen.

The experiment writes:

```text
outputs/svm_sensitivity/svm_sensitivity.csv
outputs/svm_sensitivity/svm_sensitivity_metadata.json
```

## Completed diagnostic: morphology scale

The manuscript specifies dilation followed by erosion but does not report the structuring-element size. A 10-condition screen tested disk radii 0, 1, 2, 3 and 5 under both standard scaling and no SVM scaling.

| Configuration | Dice mean ± SD | AUC |
| --- | ---: | ---: |
| no scaling, radius 3 | **0.324 ± 0.395** | 0.857 |
| no scaling, radius 5 | 0.324 ± 0.395 | 0.857 |
| no scaling, radius 2 | 0.323 ± 0.392 | 0.857 |
| no scaling, radius 1 | 0.322 ± 0.391 | 0.857 |
| no scaling, radius 0 | 0.322 ± 0.391 | 0.857 |
| standard scaling, radius 0 | 0.228 ± 0.302 | 0.798 |
| standard scaling, radius 5 | 0.218 ± 0.301 | 0.798 |

The no-postprocessing control and the strongest radius differ by only about **0.002 Dice**, so morphology scale is not a plausible explanation for the manuscript's reported 0.56 SVM Dice. Under standard scaling, larger closing radii actually reduce Dice.

## Completed diagnostic: voxel-level leakage implied by contradictory split wording

The manuscript contains conflicting 70/30 and 70/15/15 random-split language. To test whether this could correspond to pixel-level rather than patient-level splitting, a deliberately invalid leakage diagnostic placed pixels from the same 28 patients on both sides of the split.

This is **not** a clinically valid evaluation protocol and must never be reported as generalization performance.

| Configuration | Dice mean ± SD | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: |
| pixel 70/15/15, no scaling | **0.443 ± 0.304** | 0.478 | 0.679 | 0.929 |
| pixel 70/30, no scaling | **0.442 ± 0.308** | 0.477 | 0.668 | 0.927 |
| pixel 70/30, standard scaling | 0.302 ± 0.320 | 0.263 | 0.480 | 0.886 |
| pixel 70/15/15, standard scaling | 0.297 ± 0.319 | 0.256 | 0.479 | 0.897 |

This is the largest upward shift observed so far. Within-patient pixel leakage plus no scaling raises the quick SVM result into the **0.44 Dice** range and AUC to about **0.93**. That makes leakage a credible contributor to the manuscript's reported score, but it still does not fully explain the reported **0.56 ± 0.26** Dice at the 2,000-per-class diagnostic sample size.

## Completed confirmation: paper-size voxel leakage

The leakage diagnostic was repeated with the manuscript-scale **15,000 lesion + 15,000 non-lesion training pixels** per condition.

| Configuration | Dice mean ± SD | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: |
| pixel 70/30, no scaling | **0.441 ± 0.272** | 0.404 | 0.800 | 0.925 |
| pixel 70/15/15, no scaling | **0.438 ± 0.269** | 0.404 | 0.804 | 0.927 |
| pixel 70/30, standard scaling | 0.425 ± 0.296 | 0.396 | 0.701 | 0.922 |
| pixel 70/15/15, standard scaling | 0.422 ± 0.291 | 0.389 | 0.714 | **0.934** |

Increasing the balanced sample from 2,000 to 15,000 pixels per class **does not move the no-scaling leakage Dice toward 0.56**. The strongest mean Dice remains about 0.44. Interestingly, the larger sample substantially improves the standard-scaled leakage conditions from about 0.30 to about 0.42, but the best result is still materially below the manuscript's **0.56 ± 0.26**.

The combined evidence therefore supports a narrower conclusion: within-patient pixel leakage is a credible contributor to inflation, but it is **not sufficient by itself** to reproduce the published score.

## Next diagnostic: true histogram specification

The manuscript explicitly mentions histogram specification but gives neither the reference image nor the algorithm. The autonomous pipeline now includes a patient-level 19/9, 15,000-per-class diagnostic comparing:

- no histogram matching;
- empirical-CDF histogram matching to the first training reference slice;
- empirical-CDF histogram matching to a deterministic median-intensity training reference slice;

under both standard scaling and no SVM scaling.

Reference images are selected from training patients only. This is an uncertainty analysis, not recovery of the original unknown preprocessing.

The stage writes:

```text
outputs/histogram_specification/histogram_specification_summary.csv
outputs/histogram_specification/histogram_specification_metadata.json
```

If histogram specification still does not close the gap, the remaining high-priority fidelity work is the exact neighborhood/weighted-local-mean reconstruction and an exact scaled-conjugate-gradient MLP with the paper's reported two-output encoding.
