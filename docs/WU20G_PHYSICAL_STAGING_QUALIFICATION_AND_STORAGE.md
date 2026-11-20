# WU-20G — guarded physical Dataset506/507 staging implementation and storage qualification

**Scientific release status: BLOCKED.** WU-20G is an engineering implementation and qualification, **not** approval to materialize actual 149-patient training datasets, not clinical mask validation, and not authorization to run GPUs. The production CLI intentionally refuses `--mode stage` unconditionally. A subsequent reviewed release change (WU-20H) must enable real materialization **only after authentic independent review and sign-off**; a user-supplied JSON flag cannot bypass this lock.

## Existing evidence inherited and checked

- Frozen original Dataset505 plan SHA256 `6203521136de2f57bbe3895a639863112a5869697de9b752ade42dc1a23e4dca`, split SHA256 `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027`, **149 cases**, validation folds **30/30/30/30/29**.
- WU-20B: five prospective 119/120-training-case calibrations from image-derived support, no heldout outcome/label use. These remain **provisional**.
- WU-20C/D: actual pinned nnU-Net v2.8.1 official preprocessor passed one bounded heldout ROI and four full native volumes with anomalous MTT, paired NCCT equality and 0 perfusion overshoot in that tested set. Neither test establishes validation of the entire 149-case transformed cohort.
- WU-20E: release gate is explicitly BLOCKED; **15 priority anatomical mask overlays pending independent assessment**, and percentile clipping, imputation and robust Z-cap unsigned. The separate 134-patient review strategy is not yet agreed.
- WU-20F: all 745 source image NIfTI headers, 149 training-label file-presence checks and five *virtual* inventories passed; actual untrained nnU-Net model forward passed both architectures. It did not read reference lesion label headers or voxels. WU-20F results are prerequisites to WU-20G storage planning.

## WU-20G implementation

`scripts/stage_isles2024_wu20g_physical.py` implements three modes:

1. `--mode plan` (default): independently cross-checks the WU-20F completed report, each of 149 currently signed-by-metadata header checkpoints against source file stat signatures, and the WU-20E unapproved gate and five WU-20B fold-specific train-only fits against the immutable actual cohort. It opens **no** image or label voxels, and writes only private storage and planned workspace manifests, with `training_ready=false`, `staging_materialized=false`. Returns an explicitly numeric *uncompressed* storage estimate based on actual 149 NIfTI shapes: 5 folds × **(1 NCCT + 5 multimodal)** float32 volumes per patient, plus eight compressed label-file copies per patient (four training folds × both arms). Gzip size is unknown. Reserves a recommended free-space budget ≥2× the predicted uncompressed bytes to cover temporary staging and retries; a low-space result is reported and would block any subsequent release, not silently assumed acceptable. File content signatures are not produced here; previously verified headers rely on file size and nanosecond mtime.
2. `--mode synthetic`: **patient-free synthetic physical staging** using case IDs `sub_stroke9001/9002` only. Builds fake 32×32×16 NIfTI inputs, including synthetic NaN and extreme negative MTT voxels. Transforms one synthetic training case and one synthetic held-out case using the existing WU-19/WU-20 candidate policy; generates real 506/507 NIfTI volumes into ephemeral scratch; makes 507 NCCT a **byte copy** of 506 NCCT; copies only the synthetic *training* label into paired `labelsTr`. Confirms input and output SHA256, native affine/shape, finite output, support integrity and cap; refuses partial/dirty outputs. Requires actual nnU-Net v2.8.1 and runs `DefaultPreprocessor.run_case(...,seg_file=None)` with both locked plans, verifying finite arrays, matching crop boxes and exact NCCT equality. All temporary volumes are deleted on exit. Only JSON technical receipts persist.
3. `--mode stage`: **ALWAYS DENIED** with `PermissionError`, before reading inputs or creating any outputs. No `--force`, environment toggle or JSON approval stub is accepted. Clinical labels are never read by the production CLI. This lock is deliberate until a new audited prospective release is signed and integrated.

The staging core performs source 5-channel voxel geometry checks, rejected NaN NCCT, training-label geometry and **binary lesion value** integrity, pairwise NCCT SHA equality and on-disk sanitization. All per-case files stage into temporary transaction folders, verify their hashes before and after transfer, and reject preexisting output paths. A failed multi-file transaction is **not silently retried**; any partial output requires independent reconciliation before resume. A future production implementation must add durable transaction receipts across interruptions and verify the complete five-fold physical inventory with audited `validate_arm_workspaces` before enabling materialization.

## Frozen geometry and training method caveat

- Target architecture/spacing, `3d_fullres` patch size **32×256×224**, model and reader remain those of frozen Dataset505. Channel metadata is exactly `noNorm`/`NoNormalization` for the 506 one-channel and 507 five-channel experiments. WU-20F's smaller synthetic patch was only a CPU smoke; it did not alter model training plans.
- The correct outer heldout cohort is **not** a permissible internal validation dataset for training/checkpoint selection. Default `nnUNetv2_train` generally assumes validation labels and may build a different split. Therefore **NO executable training commands** are emitted. A separately reviewed custom trainer must train on the outer 119/120 cases for a prespecified 1,000 epochs and select a fixed checkpoint without accessing 29/30 heldout labels, with identical settings between both arms.
- Final out-of-fold lesion evaluation, adjudication of mask/clip clinical validity, and model inference/reconstruction remain later work units; synthetic logits have no medical meaning.

## Colab execution and private files

Run CPU-only `notebooks/26_ISLES2024_WU20G_PHYSICAL_STAGING_QUALIFICATION.ipynb` after the PR merges. It uses a pinned commit, validates actual frozen hashes, performs **plan then synthetic** mode only. It writes private reports to `MyDrive/HybridStrokeSeg/ISLES2024/results/wu20g_physical_staging_qualification/`:

- `WU20G_physical_staging_plan_PRIVATE.json` and redacted `WU20G_physical_staging_plan_aggregate.json`;
- `WU20G_synthetic_staging_aggregate.json`;
- `WU20G_CPU_qualification.log` and `WU20G_qualification_summary.json`.

No transformed patient images, private original labels, patient-overlays or signatures go into public GitHub or a persistent training directory. **Do not run the user-specific staging code on patient inputs until the independent release.**

## Approval required before WU-20H

1. Genuine independent review of the 15 priority overlays (especially `sub_stroke0016`) and defensible plan for all remaining patients; correct any anatomical-support defects without inspecting heldout outcome metrics.
2. Prospectively signed normalization/clipping/finite-imputation method and reviewer credentials; keep candidate alternatives blinded to heldout labels and performance.
3. Disk/workspace provisioning using the WU-20G actual-volume estimate, physical 149-patient per-fold transaction/restart qualification, label-only-training geometry and SHA checks, and final full-cohort nnU-Net preprocessing.
4. Reviewed training strategy that fixes 1,000 epochs without ever using the outer heldout labels for checkpoint selection. Only then consider a separately authorized GPU run.
