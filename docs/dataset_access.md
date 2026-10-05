# Dataset access

## Historical reproduction dataset: ISLES 2015 SISS

The paper's reported cohort is best matched by the **ISLES 2015 Sub-Acute Ischemic Stroke Lesion Segmentation (SISS)** task:

- 28 public training cases
- modalities: FLAIR, T1, T2, DWI
- expert lesion masks for training cases
- 1 mm isotropic, co-registered/skull-stripped challenge data

The original SMIR host is no longer active. In 2026, the challenge organizers re-archived the historical ISLES datasets on Zenodo.

### Preferred source

Zenodo record (organizer re-archive):

- https://zenodo.org/records/19135955
- DOI: `10.5281/zenodo.19135955`
- file: `ISLES2015.zip`
- recorded size: 763.4 MB
- MD5: `3b6a2226e3814faf272868a96e4837d1`

Direct file URL used by the helper script:

`https://zenodo.org/records/19135955/files/ISLES2015.zip?download=1`

Official historical challenge page:

- https://www.isles-challenge.org/ISLES2015/

### Download

From the repository root:

```bash
python scripts/download_isles2015.py --extract
```

By default the archive is stored under `data/raw/` and extracted under `data/raw/isles2015/`.

## Important reproducibility note

Do **not** upload the MRI data or derived patient images to GitHub. The repository tracks code, configuration, checksums, split manifests, and aggregate outputs only.

The public SISS training set should be used to recreate the paper. Historical challenge test labels were not released publicly, so our reproducible evaluation will use deterministic patient-level splits or cross-validation over the 28 labeled training cases and will report the exact split IDs.

## Modernization dataset

ISLES 2022 is a separate benchmark and must not be mixed with the historical reproduction. It contains DWI, ADC and FLAIR and will be integrated in a separate modern experiment track.
