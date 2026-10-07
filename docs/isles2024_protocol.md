# ISLES'24 modernization protocol

## Dataset and task

ISLES'24 is a multicenter longitudinal stroke dataset built for prediction of the final post-treatment infarct from pre-interventional acute stroke data.

The current public training release contains 149 labeled patients. The challenge test cohort contains 96 hidden patients.

The public dataset includes:
- acute NCCT;
- acute CTA;
- 4-D CTP;
- derived perfusion maps including Tmax, CBF, CBV and MTT;
- baseline demographic and clinical variables;
- follow-up DWI and ADC;
- a final infarct mask registered to NCCT space;
- optional vessel-related annotations such as LVO and Circle of Willis masks.

The current Zenodo release is approximately 99 GB, so the repository must not silently auto-download it in Colab.

Primary dataset reference:
Riedel EO et al. The Ischemic Stroke Lesion Segmentation Challenge (ISLES)'24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes. Radiology: Artificial Intelligence.

Dataset record:
https://zenodo.org/records/16813698

Official challenge:
https://isles-24.grand-challenge.org/

## Critical task distinction

ISLES'24 is not ordinary same-timepoint lesion segmentation.

The model sees pre-interventional acute information and predicts a biologically evolving final infarct that is defined later. This is a longitudinal tissue-outcome prediction problem.

Therefore:
- follow-up DWI and ADC are never input channels;
- the final infarct mask is a label only;
- post-treatment and 3-month outcome variables are excluded from model inputs;
- only baseline clinical variables may be used in clinical-fusion experiments.

Any experiment that violates this rule is considered leakage and must not appear in the primary results.

## Phase 0: dataset audit

After the user obtains and extracts the public dataset, run:

python scripts/audit_isles2024.py /path/to/ISLES24 --json outputs/isles2024_audit.json

The audit checks:
- expected 149-case count;
- required acute modalities;
- NCCT-grid consistency of registered CTA and perfusion maps;
- NCCT-grid consistency of the final infarct mask;
- presence of baseline and outcome clinical files;
- presence of optional LVO/CoW masks and raw/registered CTP.

The audit deliberately does not treat follow-up MRI as a model input.

## Phase 1: first baseline

Primary baseline:
3-D nnU-Net v2 using NCCT only.

Why NCCT first:
- it provides the cleanest imaging-only baseline;
- it tests how much final infarct information is recoverable from the standard structural CT;
- it gives a reference before adding perfusion or angiographic information.

Use deterministic patient-level folds and official challenge-style metrics.

## Phase 2: perfusion contribution

Add registered perfusion maps:
- Tmax;
- CBF;
- CBV;
- MTT.

Compare against the exact same NCCT-only folds.

The main comparison is not whether the multimodal model achieves a larger raw score after arbitrary tuning. It is the paired incremental value of perfusion information under a controlled protocol.

## Phase 3: CTA contribution

Add CTA as an additional registered imaging channel.

Predeclare the comparison:
- NCCT;
- NCCT + perfusion;
- NCCT + CTA + perfusion.

If CTA-specific preprocessing such as vessel windowing or skull stripping is added, it must be validated inside training data and documented.

## Phase 4: baseline clinical fusion

Only baseline variables available at prediction time may be considered.

Clinical fusion must:
- handle missingness within training folds;
- standardize numeric variables using training-fold statistics;
- encode categorical variables without using held-out information;
- report an imaging-only comparator on identical folds.

Post-treatment variables and 3-month outcomes are prohibited as inputs.

## Phase 5: raw 4-D CTP

Raw dynamic CTP is deliberately postponed.

The archive is large and the modeling problem is substantially more expensive. It should be attempted only after registered perfusion-map baselines are complete and reproducible.

## Evaluation

Primary metrics should mirror the challenge family:
- Dice;
- absolute volume difference;
- absolute lesion-count difference;
- lesion-wise F1.

Also report:
- recall;
- precision;
- HD95 when meaningful;
- lesion-volume-stratified performance;
- center-stratified performance if center labels are available for the public cohort.

Report fold-wise results and confidence intervals, not just a single pooled mean.

## Data and licensing

The public training data are distributed for non-commercial research use under the dataset terms reported by the official publication/record. Do not redistribute the dataset in this public repository.

Git must continue to ignore:
- patient images;
- clinical records;
- model checkpoints;
- extracted archives;
- predictions containing patient identifiers.

## Immediate engineering milestone

The repository now supports:
- ISLES'24 BIDS-style case discovery;
- a strict separation between valid acute inputs and follow-up target-generation data;
- an NCCT-grid audit command.

The next implementation milestone is nnU-Net staging for the NCCT-only baseline, followed by deterministic cross-validation manifests and official-style evaluation.
