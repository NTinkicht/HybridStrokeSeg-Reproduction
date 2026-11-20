# WU-20H — Sequential-fold staging capacity and held-out-label trainer-access hold

**Engineering status: implemented for read-only qualification only. Scientific release status: BLOCKED.**

WU20B–WU20G established five training-only normalization fits, official nnU-Net 2.8.1 input compatibility, four complete native-volume anomaly tests, 149 source-case NIfTI header inventory, a two-architecture *untrained* forward, synthetic NIfTI staging, and a five-fold physical-staging storage estimate. These facts do **not** authorize clinical-mask selection, real Dataset506/507 materialization, model training, checkpoint selection, or out-of-fold evaluation.

## Why sequential staging is necessary

WU20G's actual, verified cohort-size estimate:

| Quantity | Bytes | Decimal GB |
|---|---:|---:|
| Five-fold uncompressed 506+507 image arrays | 339,171,655,680 | 339.17 |
| Eight copies of 149 compressed label files across all folds | 189,978,112 | 0.19 |
| Five-fold combined uncompressed budget | 339,361,633,792 | 339.36 |
| Conservative simultaneous 2× reservation | 678,723,267,584 | 678.72 |
| Scratch free at 2026-11-20 WU20G run | 241,870,721,024 | 241.87 |

Because every image is represented **once per fold in both arms** (1+5 float32 channels), the per-fold image budget is **67,834,331,136 bytes**. Since each source lesion label occurs in four training folds and two arms, a single fold's label budget is **at most 47,494,528 bytes**, yielding a **conservative per-fold upper-bound output 67,881,825,664 bytes** and a 2× staging-only reservation of **135,763,651,328 bytes** (~126.44 GiB). These are source-array/label estimates, not guaranteed .nii.gz sizes, nnU-Net preprocessed tensors, dataloader cache, checkpoints, log archives or GPU working memory. **Actual training capacity remains unqualified**. Colab scratch space changes between runtime instances; Notebook27 queries its current free bytes.

A future approved orchestration must materialize only one `outer_fold_0..4` at a time, verify 506/507 NCCT byte parity and valid training labels, separately preprocess/train both arms against the same frozen plan and fixed epoch target, then archive signed provenance and model/inference receipts. **Never implement automatic deletion of unarchived tensors or checkpoints.** After a user-authorized retention and storage review, clean intermediates explicitly, then move to next fold. WU20H emits this *plan*, not a runnable orchestration.

## WU20H read-only evidence validation

`scripts/plan_isles2024_wu20h_sequential.py --mode plan` checks:

1. Frozen source `splits_final.json` SHA-256 `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027` and plans `nnUNetPlans.json` SHA-256 `6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca`.
2. WU20G immutable private plan, 149 patient / 745 header checks and 149 label-stat checks, `training_ready=false`, `staging_materialized=false`, 1,000 fixed-epoch hold, and the source WU20F/WU20E evidence-file SHA-256 digests.
3. All **five actual WU20B fold-manifest file SHA-256 values** match WU20G's source manifest hashes and the WU20F private virtual file; each train/held-out case ID exactly matches the frozen 149-case split, with no duplicate IDs or leakage.
4. Both 506 and 507 virtual expected filenames are **exact** for `imagesTr`, `imagesTs` and `labelsTr`; `labelsTs` is empty, and 507 NCCT must be an eventual byte-copy of the matched 506 NCCT. No hidden default nnU-Net normalization may be introduced.
5. The existing 15-case review CSV is reviewed for **status only**. A manually typed `APPROVED` or reviewer name is not a cryptographic signature, so the resulting gate **never** grants clinical authorization.
6. The live temporary scratch free space is compared against the one-fold conservative staging reservation; the result is **provisional staging capacity only**. Failures may trigger `provisional_staging_space_meets_2x=false` but the report remains a completed, **blocked** engineering plan.

The generated private report includes original case identities and hash evidence and must live in a private Drive directory. The aggregate report removes patient IDs, fold case rosters and detailed evidence. No image pixels, label files, predictions or model weights are accessed.

**Denied modes:** `stage`, `train`, `cleanup`, `approve`, `release` fail immediately with `PermissionError`. There is no CLI flag, fake approval JSON, `--force`, automatic cleanup, or training command that can bypass this hard block.

## nnU-Net trainer and checkpoint selection audit: preparation, NOT certification

The standard nnU-Net runner was not modified or instrumented in WU20H. With physical datasets not approved or materialized, **no evidence exists that a real nnU-Net training process honors the proposed outer-fold label-access policy**. WU20H instead implements a strict **synthetic event contract** in `scripts/audit_isles2024_wu20h_training_access.py` for future trainer instrumentation:

- The exact outer 119/120 training IDs are the only permitted cases for `train_label_open`; the 29/30 outer held-out labels must remain unavailable to training and checkpoint selection. If an internal validation split reduces the number of training cases, this deviates from the prespecified all-outer-training design and must be audited/rejected.
- A complete sequence of 1 through 1,000 epoch-completion receipts must precede the **single fixed final epoch-1000 checkpoint**. No `checkpoint_best`, metrics-selected model or held-out-performance-based early stopping.
- The final checkpoint must be archived before any outer held-out image inference. All held-out predictions must then be **sealed**, before a held-out lesion label may be opened for the prespecified OOF evaluation.
- Any unknown event, missing epoch, early held-out label access, unsigned/unverified archive or missing case causes the event-trace contract to fail.
- The 1,000 events in Notebook27 are fake synthetic events (fixture IDs); they are **not** evidence of running 1,000 real training epochs. Both the training trace and its report set `trained_model_executed=false`, `actual_nnunet_trainer_hook_verified=false`, `training_ready=false`.

The subsequent release candidate needs actual pinned nnU-Net 2.8.1 trainer hooks and instrumented synthetic integration before training. Runtime enforcement must *block filesystem access* to outer held-out labels until predictions are sealed (not merely log an after-the-fact trace), and must guarantee identical optimizer, seed, 1,000 epochs, checkpoint and augmentation policies for 506/507.

## Required independent human and technical gates

1. Independently review all 15 original anatomical masks, including low coverage `sub_stroke0016`, and approve a statistically defensible review protocol for the 134 unreviewed cases.
2. Expert review and sign an **immutable prospective protocol** for head-tissue masking, train-only percentiles, nonfinite MTT imputation and final ±8 clipping. A passed mathematical sanitizer does not demonstrate preservation of medically meaningful extrema.
3. Implement independently reviewed, **durable per-case transactions** for real 149-case physical staging, fully validate label geometry and labels only for each outer fold's train cases, and validate real full-volume nnU-Net preprocessing on all cases. The current WU20G `stage` entrypoint must stay disabled until the new gated PR is approved.
4. Audit actual trainer code/runtime isolation and prespecified fixed-epoch checkpoint; estimate and provision full scratch+preprocessing+checkpoint+archive storage for each sequential fold, with immutable retention receipts.
5. Only then may an explicitly approved separate release PR enable actual 506/507 training. Use out-of-fold labels **after** model predictions are fixed for prespecified paired Dice, lesion F1, absolute volume difference, lesion-count and corrected statistical analysis.

## Notebook 27

`notebooks/27_ISLES2024_WU20H_SEQUENTIAL_RELEASE_HOLD.ipynb` runs on **Colab CPU**. It mounts private Drive, checks original splits and plans hashes, downloads the exact SHA-pinned two WU20H scripts, reads **only private JSON/CSV technical metadata**, produces a private one-fold storage/sequencing plan, and runs a synthetic **non-executable trainer trace**. It produces no transformed scans, training commands, checkpoint files or automated cleanup; `training_ready=false` is enforced.

Private results: `MyDrive/HybridStrokeSeg/ISLES2024/results/wu20h_sequential_release_hold/`.

**Interpretation:** WU20H can be engineering-complete with a passing CPU notebook *while the scientific release remains blocked*. Do not equate these states.
