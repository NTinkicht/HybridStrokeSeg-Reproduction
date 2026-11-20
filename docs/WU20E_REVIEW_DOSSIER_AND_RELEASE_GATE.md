# WU-20E — Independent anatomical-support review and fail-closed release gate

**Status:** engineering and evidence-preparation deliverable only. **No clinical approval and no Dataset506/507 training authorization.**

## Source-evidence chain (read-only)

1. WU20B: all 149 NCCT patient image-derived support samples audited, five training-only calibrations fitted from their own 119/120 patients, 15 priority overlay PNGs generated; source split SHA256 `664ed51e15374f8f9568d586fba2cb87d5157dc90986e5eca82c9c8440bf6027`.
2. WU20C: pinned nnU-Net 2.8.1, locked architecture and `NoNormalization`, matched paired 506/507 NCCT in held-out ROI. ROI test did not include raw anomaly voxels.
3. WU20D: pinned nnU-Net 2.8.1, four complete native image volumes with real MTT anomalies technically passed, exactly matching postresampling NCCT and zero perfusion cap overshoot. **No trained model inference was performed.**
4. WU20E: recomputes dataset/fold QC against the **actual frozen splits file**, original summaries, five private 119/120-case normalizer manifests, the complete 15-overlay inventory, and the existing private reviewer CSV. It writes a candidate-policy digest and a **non-approving**, cryptographically traceable release report.

Private screenshots, patient-specific CSV/JSON, PDF atlas and SHA evidence manifest live in Google Drive only. Do not commit these artifacts to the public GitHub repo.

## Prospective preprocessing policy under review (NOT approved)

- Cohort: frozen 149 cases and exact 5-fold train/held-out split, 1000-epoch model target unchanged if a later release authorizes training.
- Training-only calibration: deterministic 4,096 voxels **per training case per channel** from image-only provisional head-tissue support. The normalizer is fit separately for each fold. Validation images and lesion labels must not influence fit statistics or threshold selection.
- NCCT: q0.005 and q0.995 clipping limits from train-only support; original training-sample mean and standard deviation applied after clipping; non-finite NCCT fails closed.
- TMAX/CBF/CBV/MTT: q0.005/q0.995 train-only clipping, median center, scale = (q0.75-q0.25)/1.349; finite-limit replacement using the **training-only median**, followed by candidate post-z cap ±8. Outside provisional head-tissue support, zero.
- Masks: `provisional_head_tissue_v1`. This is **not** a validated intracranial brain mask. An overlay may include extra-cranial tissue or exclude brain; visually inspect and require clinical/technical decision before locking any use.
- The 15 prioritized overlays cover selection extremes, but **do not prove anatomical acceptability of all 149 scans**. A prospective sampling/review standard for the remaining cohort is still required.
- nnU-Net `3d_fullres` architecture/spacing exactly copied from frozen Dataset505, `noNorm`/`NoNormalization` only; Original 504/505 are immutable.
- Evaluate prespecified paired OOF Dice, lesion F1, AVD mL, lesion count difference; perform family-wise error correction. Neither old OOF results nor lesions may tune preprocessing.

### The clinical reviewer must decide

- For **each of the 15 overlay cases**, is relevant intracranial brain tissue preserved? Is the level of skull, soft-tissue or other extracranial inclusion acceptable? Is correction required? Consider using source native volumes beyond the three shown slices.
- Is the 4,096-voxels-per-case sample strategy appropriate for the intended comparison?
- Are the fixed q0.005/q0.995 clip limits, train-only median replacement for non-finite perfusion, and final ±8 cap scientifically/clinically acceptable? Consider preserving valid perfusion extremes and whether non-finite regions should instead be masked/flagged.
- How must the project audit the remaining 134 patients and rare scan geometries so the 15 selected cases are not treated as universal evidence?

Record decisions in `results/wu20c_nnunet_runtime_qualification/WU20C_mask_review_checklist_PRIVATE.csv`; **preserve the original case identity/overlay file mappings**. Values in `decision`: `APPROVED` only after professional review, otherwise `REJECTED` or `PENDING`. Every approval requires `brain_tissue_preserved=YES`, `correction_required=NO`, `extracranial_tissue_included=NO` or independently justified `ACCEPTABLE`, a reviewer name, and ISO-8601 date. A separate actual signed protocol and reviewer credentials must be verified manually; inserting text into CSV is not an authentic signature.

## Release algorithm (intentionally does NOT issue approval)

`python scripts/audit_isles2024_wu20e_release_gate.py --wu20b <WU20B_aggregate.json> --wu20c <WU20C_probe_aggregate.json> --wu20d <WU20D_four_case_aggregate_PRIVATE.json> --folds <fold_private> --splits <splits_final.json> --review <WU20C_mask_review_checklist_PRIVATE.csv> --overlays <private_mask_overlays> --output <private_review_folder>`.

The program refuses changed frozen split checksum, incorrect or leaky 5-fold calibrations, missing source reports, unexpected four-case findings, mismatch of the 15 original overlays, incomplete reviewer fields and unsafe output locations. It computes SHA256 of source files and images. **It always sets `training_ready=false`**, including after an apparently complete CSV, because independent signed scientific review, full-cohort staging and model forward/inference-path qualification remain unfulfilled.

## Next WU after human review

**WU-20F — full-cohort staging dry-run + prospective method freeze + synthetic forward pass:** generate five isolated, metadata-only staging inventories for both arms; verify all raw NCCT/perfusion/label presence and geometry without leaking validation labels or touching original models. Independently qualify a CPU synthetic forward pass with the pinned nnU-Net architecture to remove the circular requirement for an already-trained 506/507 model. Once independent mask/transform reviewers have signed the prospective methods and WU20F passes, seek an explicit training-release decision. Run actual trained-model held-out inference only *after* training; never claim it was performed beforehand.

**Privacy:** No raw case images, private patient manifests, review decisions, or atlas PDF in the public repository. The repo contains only the generic release-gate utilities, a reusable atlas builder, and synthetic tests.
