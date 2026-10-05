# Reconstruction assumptions

This repository is a **clean-room reconstruction from the manuscript**, because the
original source code is no longer available.

The rule for this project is simple: a paper statement is never silently replaced
by an implementation choice. Every missing detail is documented and configurable.

| Component | Manuscript support | Current reconstruction choice | Status |
|---|---|---|---|
| Input sequence | FLAIR is explicitly used for the pixel classifier | FLAIR slice is the feature extractor input | stated |
| Input dimensionality | MLP is reported with 9 input features | exactly 9 features are emitted | stated |
| Spatial location | 2-D locations are listed | raw row and column indices | inferred, configurable |
| Intensity | intensity feature is listed | raw preprocessed FLAIR intensity | stated/inferred |
| Neighbourhood | 3x3 box neighbourhood is described, but dimensionality is unclear | one 3x3 box-mean scalar | inferred |
| Weighted local mean | listed without weights | 3x3 binomial kernel `[1 2 1; 2 4 2; 1 2 1]/16` | inferred |
| Directional descriptor | horizontal, vertical, 45°, 135° | four separate features | stated |
| Directional window | `1x25` | centered odd window of length 25 | stated/inferred |
| Threshold | 20 | `>= 20` | stated; intensity scale remains ambiguous |
| "Run length" operation | prose is not sufficient for exact reconstruction | count of values meeting threshold in each centered directional window | inferred |
| Image boundary handling | not reported | out-of-image directional samples count as below threshold | inferred |
| Preprocessing | filtering, histogram specification, brain extraction are mentioned | intentionally not guessed in the feature module | unresolved |

## Why the feature count is nine

The only interpretation currently consistent with the paper's reported **9-input MLP** is:

`2 location + 1 intensity + 1 neighbourhood + 1 weighted local mean + 4 directional = 9`.

This does **not** prove that the original unpublished implementation used these exact scalar definitions. It is the smallest reproducible interpretation consistent with the manuscript.

## Directional feature terminology

The manuscript calls the four directional descriptors "run length". The current implementation does not claim to be a conventional gray-level run-length matrix (GLRLM) feature. It is named `directional_threshold_count` in code to make the reconstruction assumption explicit.

## Next unresolved reproduction decisions

Before claiming numerical reproduction of the published Dice scores, the project must still define and sensitivity-test:

- the filtering operation;
- histogram-specification target and intensity range;
- exact brain extraction method;
- how the single FLAIR slice per patient was selected;
- patient-level train/test split and random seed;
- MLP scaled-conjugate-gradient implementation details;
- SVM `C`, `gamma`, feature scaling, and SMO settings;
- morphological structuring element and iterations;
- empty-mask Dice handling.

These will be implemented behind configuration switches and reported as reconstruction assumptions rather than retroactively attributed to the paper.
