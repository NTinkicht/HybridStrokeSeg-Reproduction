# Morphology result and voxel-leakage diagnostic

## Completed morphology sensitivity

A real-data Colab run on 2026-10-06 completed the predeclared 10-condition morphology screen on the 28 labeled ISLES 2015 SISS cases. The experiment used the paper-like oracle slice-selection reconstruction, paper-minimal preprocessing, threshold-count directional features, 2,000 lesion + 2,000 non-lesion training pixels, SVM C=1 and gamma=scale.

| SVM scaling | Closing radius | Dice mean ± SD | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 3 | **0.324 ± 0.395** | 0.295 | 0.405 | 0.857 |
| none | 5 | 0.324 ± 0.395 | 0.293 | 0.407 | 0.857 |
| none | 2 | 0.323 ± 0.392 | 0.297 | 0.399 | 0.857 |
| none | 1 | 0.322 ± 0.391 | 0.298 | 0.396 | 0.857 |
| none | 0 | 0.322 ± 0.391 | 0.298 | 0.395 | 0.857 |
| standard | 0 | 0.228 ± 0.302 | 0.195 | 0.342 | 0.798 |
| standard | 1 | 0.227 ± 0.302 | 0.193 | 0.342 | 0.798 |
| standard | 2 | 0.224 ± 0.302 | 0.191 | 0.343 | 0.798 |
| standard | 3 | 0.223 ± 0.302 | 0.188 | 0.344 | 0.798 |
| standard | 5 | 0.218 ± 0.301 | 0.184 | 0.347 | 0.798 |

The morphology radius has a negligible effect under the stronger no-scaling SVM configuration: the full tested range changes mean Dice by only about 0.002. Under standard scaling, larger closing radii slightly reduce Dice. Therefore the manuscript's omitted morphology scale does **not** explain the gap between the current reconstruction and the reported SVM Dice of 0.56.

The strongest tested morphology setting, no scaling with radius 3, reaches only 0.324 mean Dice. It should not be treated as a recovered original parameter because the condition was observed after the full predeclared screen.

## Next diagnostic: deliberately test voxel-level leakage

The manuscript contains contradictory splitting language: it describes a 19/9 patient holdout (approximately 70/30 by cases), but also refers to random 70/15/15 partitioning. The next experiment therefore tests a deliberately invalid interpretation in which pixels from every selected patient slice are randomly divided between training and testing.

This is **not** a clinically valid evaluation. It is a diagnostic designed only to answer whether within-patient voxel leakage could materially inflate the reported score.

The diagnostic evaluates four predeclared conditions:

- random pixel 70/30 split with standard SVM scaling;
- random pixel 70/30 split with no SVM scaling;
- random pixel 70/15/15 split with standard SVM scaling;
- random pixel 70/15/15 split with no SVM scaling.

Every patient intentionally contributes pixels to both training and test sets. The same label-informed maximum-lesion slice reconstruction is used. Training is balanced to 2,000 lesion + 2,000 non-lesion pixels. Test Dice is summarized across the 28 patients using each patient's held-out pixels. Morphological post-processing is not applicable because the test set is a random subset of pixels rather than a contiguous image mask.

Run:

```bash
python scripts/run_voxel_leakage_diagnostic.py
```

Persistent Colab usage can specify:

```bash
python scripts/run_voxel_leakage_diagnostic.py \
  --data-root data/raw/isles2015 \
  --output-root /content/drive/MyDrive/HybridStrokeSeg/results/voxel_leakage \
  --resume
```

Outputs:

```text
voxel_leakage_summary.csv
voxel_leakage_metadata.json
pixel_70_30_standard/per_case_metrics.csv
pixel_70_30_none/per_case_metrics.csv
pixel_70_15_15_standard/per_case_metrics.csv
pixel_70_15_15_none/per_case_metrics.csv
```

Any performance increase from this diagnostic must be described as evidence of leakage sensitivity, never as improved reproduction performance.
