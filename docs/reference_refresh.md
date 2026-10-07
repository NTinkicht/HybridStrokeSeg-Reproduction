# Reference modernization map

This document separates references that are historically necessary for reproducing the manuscript from references that should be updated when preparing a revised 2026 paper.

## Dataset identity correction

The manuscript's reference [15] is bibliographically and chronologically inconsistent: it labels the benchmark "ISLES 2022" while describing the 2015 challenge paper and a 28-case SISS training cohort. The historically correct source is:

- Maier O et al. **ISLES 2015 - A public evaluation benchmark for ischemic stroke lesion segmentation from multispectral MRI.** *Medical Image Analysis*. 2017;35:250-269. DOI: 10.1016/j.media.2016.07.009.

For modern benchmark sections, add rather than substitute:

- Hernandez Petzsche MR et al. **ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.** *Scientific Data*. 2022;9:762. DOI: 10.1038/s41597-022-01875-5. Dataset DOI: 10.5281/zenodo.7153326.
- Riedel EO et al. **The Ischemic Stroke Lesion Segmentation Challenge (ISLES)'24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes.** *Radiology: Artificial Intelligence*. Dataset DOI: 10.5281/zenodo.16813698.
- de la Rosa E et al. **ISLES'24: Improving final infarct prediction in ischemic stroke using multimodal imaging and clinical data.** arXiv:2408.10966.

These are different datasets and, for ISLES'24, a different longitudinal prediction task. They must not be conflated.

## Epidemiology and clinical background

Replace old web fact sheets and decade-old global-burden statistics with current authoritative sources:

- World Health Organization. **Stroke.** Fact sheet updated 19 December 2025. https://www.who.int/news-room/fact-sheets/detail/stroke
- Feigin VL et al. **World Stroke Organization: Global Stroke Fact Sheet 2025.** *International Journal of Stroke*. 2025;20(2):132-144. DOI: 10.1177/17474930241308142.

The revised introduction should avoid carrying forward the manuscript's old "15 million strokes / 6 million deaths" wording without rechecking the denominator and year. Use the contemporary GBD 2021-based figures reported by WHO/WSO instead.

## Historical-method references to retain

The following families remain useful only when discussing the historical development of handcrafted or classical machine-learning stroke segmentation:

- Maier et al. Extra Trees / classical subacute stroke segmentation.
- Mitra et al. random-forest multimodal MRI segmentation.
- Maier et al. SVM/classifier comparisons.
- The ISLES 2015 challenge submissions cited in the manuscript (Kamnitsas, Feng, Halme, Reza and related entries).

Do not present these as current state of the art. Their role in a 2026 manuscript is historical context and reproducibility comparison.

## Modern segmentation baselines

Add strong, reproducible contemporary baselines before claiming novelty over deep segmentation methods:

- Isensee F et al. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** *Nature Methods*. 2021;18:203-211. DOI: 10.1038/s41592-020-01008-z.
- Isensee F et al. **nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.** *MICCAI 2024*. pp. 488-498. DOI: 10.1007/978-3-031-72114-4_47.
- Roy S et al. **MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation.** *MICCAI 2023*. pp. 405-415. DOI: 10.1007/978-3-031-43901-8_39.

nnU-Net v2 should be the primary implementation baseline. MedNeXt is a secondary architecture comparison only after compute and validation are matched.

## Stroke-specific modern references

Add current benchmark and translation references:

- de la Rosa E et al. **DeepISLES: a clinically validated ischemic stroke segmentation model from the ISLES'22 challenge.** *Nature Communications*. 2025;16:7357. DOI: 10.1038/s41467-025-62373-x.
- Ren T et al. **How We Won the ISLES'24 Challenge by Preprocessing.** arXiv:2505.18424. Use this as a challenge-method reference when discussing the importance of CT preprocessing, not as the sole definition of state of the art.
- Current 2026 reviews of acute ischemic stroke segmentation should be used to distinguish same-modality lesion segmentation from cross-modality or longitudinal tissue-outcome prediction.

DeepISLES remains important for MRI generalizability, while ISLES'24 references are the correct context for the new primary modernization task.

## References to demote or remove

- Vendor marketing/news pages should not be used to support clinical or technical MRI claims when peer-reviewed or standards-based sources are available.
- Generic WHO URLs from old snapshots should be replaced by the current WHO stroke fact sheet or the WSO fact sheet above.
- The manuscript's 2023/2024 application papers can remain as examples of recent work, but they should not anchor the state-of-the-art claim once ISLES 2022, DeepISLES, nnU-Net v2 validation literature and stronger reproducible baselines are included.
- Any statement that FLAIR "inherently encapsulates information from other sequences" should be removed or rewritten; the modern ISLES benchmark explicitly treats DWI, ADC and FLAIR as distinct complementary inputs.

## Citation policy for the revised paper

The revised manuscript should maintain two explicit layers:

1. **Reproduction layer:** cite the sources and benchmark papers that were available to the original method and preserve the historical ISLES 2015 context.
2. **Bridge layer:** cite real ISLES 2022 and DeepISLES when discussing modern MRI segmentation.
3. **Primary modernization layer:** cite ISLES'24 dataset/challenge papers, current CT final-infarct prediction work, nnU-Net/nnU-Net Revisited, and any compute-matched architecture actually tested.

Never replace an old source in a way that makes the historical experiment appear to have used a dataset, method, metric or clinical claim that did not exist at the time.
