# ISLES'24 modernization protocol

## Dataset and task

ISLES'24 is a multicenter longitudinal stroke dataset built for prediction of the final post-treatment infarct from pre-interventional acute stroke data.

The current public training release contains 149 labeled patients. The challenge test cohort contains 96 hidden patients. The current Zenodo v7 archive is `train.7z` (~99 GB), published MD5 `4959a5dd2438d53e3c86d6858484e781`.

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
https://zenodo.org/records/17652035

Official challenge:
https://isles-24.grand-challenge.org/


## Persistent parallel download

Because the current training archive is about 99 GB, store it on persistent storage rather than the ephemeral Colab filesystem.

For unstable Colab sessions, the preferred workflow is the parallel chunked downloader:

```bash
python scripts/download_isles2024_parallel.py \
  --archive "/content/drive/MyDrive/HybridStrokeSeg/ISLES2024/raw/train.7z" \
  --parts-dir "/content/drive/MyDrive/HybridStrokeSeg/ISLES2024/raw/train.7z.parts" \
  --workers 4 \
  --part-size-gib 1
```

The downloader first verifies that the Zenodo endpoint supports HTTP Range requests. It then stores independent 1 GiB byte ranges as persistent files and fetches four ranges concurrently. Completed chunks are skipped on rerun and partially downloaded chunks resume from their saved byte count.

If the older sequential downloader already created a partial `train.7z`, the parallel downloader renames that contiguous partial file to `train.7z.prefix` and keeps it as already-downloaded data. It downloads only the remaining ranges.

After all ranges are present, the script sequentially assembles the final archive, verifies the current Zenodo v7 MD5, and only then removes the temporary prefix/chunk files.

The single-stream `scripts/download_isles2024.py` downloader remains available as a fallback.

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
- an NCCT-grid audit command;
- leakage-safe nnU-Net v2 staging for acute channels;
- a deterministic five-fold patient split;
- a predeclared NCCT-only baseline configuration.

After audit, stage the first baseline with:

    python scripts/stage_isles2024_nnunet.py /path/to/ISLES24 "$nnUNet_raw" --channels ncct

The next implementation milestone is official-style ISLES'24 evaluation plus a reproducible Colab/HPC training workflow for the NCCT-only baseline.


## Core selective extraction

The completed 99 GB archive is intentionally not expanded in full for the
first experiments. The repository now extracts a leakage-safe core subset:

- acute NCCT;
- CTA registered to NCCT;
- registered Tmax, CBF, CBV and MTT maps;
- final-infarct lesion masks from derivatives/ses-0002;
- baseline demographic/clinical CSV files.

Raw 4-D CTP, follow-up DWI/ADC and outcome CSV files are excluded from this
first extraction profile. This reduces storage and makes the temporal
anti-leakage boundary explicit.

The extraction is resumable in subject batches:

    python scripts/extract_isles2024_core.py train.7z /path/to/core --batch-size 10

After extraction:

    python scripts/audit_isles2024.py /path/to/core \
      --channels ncct cta tmax cbf cbv mtt \
      --json outputs/isles2024_core_audit.json

Then stage the first predeclared baseline:

    python scripts/stage_isles2024_nnunet.py \
      /path/to/core "$nnUNet_raw" \
      --dataset-id 504 \
      --dataset-name ISLES2024_NCCT \
      --channels ncct \
      --resume

The one-click persistent Colab workflow is
`notebooks/08_prepare_isles2024_core_and_ncct_baseline.ipynb`.


## NCCT-only nnU-Net v2 training workflow

The first predeclared modern baseline is now staged as
`Dataset504_ISLES2024_NCCT` with 149 patient-level training cases and a fixed
five-fold split (seed 2026).

Before nnU-Net preprocessing, staged lesion masks copy the corresponding NCCT
header exactly **without resampling voxels**. This handles the one benign
qform/sform rounding discrepancy observed in the public release while
preserving the mask array unchanged.

Persistent Colab workflow:

`notebooks/11_isles2024_NCCT_NNUNET_CV.ipynb`

The workflow pins `nnunetv2==2.8.1`, runs:

    nnUNetv2_plan_and_preprocess -d 504 --verify_dataset_integrity -c 3d_fullres

then restores the deterministic `splits_final.json` and trains folds 0--4
with validation probabilities enabled:

    nnUNetv2_train 504 3d_fullres FOLD --npz --c -device cuda

The orchestrator skips folds with `checkpoint_final.pth` and resumes
interrupted folds from nnU-Net checkpoints. Persistent outputs are written to:

- `ISLES2024/nnUNet_preprocessed/`
- `ISLES2024/nnUNet_results/`
- `ISLES2024/results/nnunet_ncct_cv.log`
- `ISLES2024/results/nnunet_ncct_cv_status.json`

After all five folds complete, the next step is out-of-fold evaluation using
the official ISLES'24 metric family: Dice, absolute volume difference,
absolute lesion-count difference, and lesion-wise F1.
