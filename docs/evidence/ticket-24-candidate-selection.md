# Ticket 24 candidate selection evidence

Recorded 2026-09-23. This public-safe record contains aggregate execution
measurements and artifact checksums only; it contains no study identifiers,
source rows, reports, private paths, or per-study failure details.

## Owner selection

The owner selected the Ticket 23 recommendation: preserve and select the exact
existing three-fold gold-only candidate
`ticket-11-local-candidate-20260911-v1`, manifest SHA-256
`cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`.
The frozen candidate was not modified. The selection is recorded here and in
Ticket 24, separately from its immutable manifest.

Ticket 23 found no validated development result that justified replacing this
candidate. The gold-only fold OOF pooled macro AUC was 0.464313 (study-bootstrap
95% interval 0.404–0.524), compared with 0.474353 for the fold-prevalence-only
comparator; the predeclared within-fold permutation test gave p=0.918. The
OOF predictions do not directly measure the deployed three-model average, and
no all-data refit was run. The choice preserves the existing tested delivery
path; it is not a claim of high predictive quality or superiority to an
unmeasured refit.

## Identity and prior execution

The approved Ticket 12 v13 representative execution was bound to the selected
candidate and exact delivery artifacts:

| Artifact | Identity |
| --- | --- |
| Candidate | `ticket-11-local-candidate-20260911-v1`; manifest SHA-256 `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc` |
| Kaggle request | `ticket-12-ticket-11-local-candidate-20260912-v13`; SHA-256 `16bf6733c586bacaa46248ecb444f756c0521891ebe1f66de38190836db91a2b` |
| Owner approval record | SHA-256 `f34ed2bad3187099b9ec9d0e2aabeb6c4de05d77d6250954482aa6a927a8589b`; scope `ticket-12-exact-bundle-upload-and-execution` |
| Three fold checkpoints | SHA-256 `5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76`, `4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d`, `808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8` |
| Delivery notebook | SHA-256 `1e958fbd0c789825b30a583cdf70224a8599ee7412f8d86764af8d9b2e28e54f` |
| Kaggle adapter | SHA-256 `0af614463e0cc97c497fabe6c531206a06097ff81d2196837ff39ffba5e647d7` |
| Kaggle execution evidence | SHA-256 `bf334023a63dd817af24d44fb7d39dd7870c5243f92da4ad19275cdd2dbf6a59` |
| Representative output | 158 rows × 13 columns; SHA-256 `66761efb64a74b7c4d9e426861799622b711247ae2e896f42908833fde62d586` |
| Recovery evidence | SHA-256 `c2268ab54142d6f5c31674817b066d6c777152b7e943ee16ea8d7dfe74788c53` |

The execution used an internet-disabled CPU environment with the pinned NumPy
2.2.6, pydicom 3.0.1 and torch 2.7.1+cpu runtime. The owner approval records
that the exact v13 request was approved for upload and execution, with rules
acceptance and account eligibility checked on 2026-09-12. The v13 request
itself had `upload.submission_allowed=false`, and no leaderboard submission
was performed. This evidence is specific to the v13 request and its
representative workload; a changed bundle, input, or workload needs its own
approval and execution evidence.

## Measured scope

The v13 run processed 158 studies, 965 series and 31,486 DICOM files/slices.
Its output passed the local contract: exact header and UID order, finite scores
in `[0, 1]`, zero missing or duplicate rows, and zero decode failures. The
complete investigation measured fallback at 0/158. Save/reload and intentional
interruption recovery resumed from 17 checkpoint rows, lost zero rows, and
produced byte-identical output.

Whole-notebook runtime was 2,932.068918 seconds. The recorded fixed-plus-
measured-scaling projection at approximately 1,300 studies was 9,462.946510
seconds, below the 25,920-second engineering target and 32,400-second hard
limit. Peak RAM was 26,716,372,992 bytes and peak VRAM was zero. The execution
bundle was 940,350 bytes and the three model checkpoints totaled 314,944
bytes.

These figures describe the 158-study representative workload. They do not
establish hidden-set runtime, output completeness, or fallback behavior.
`accepted_by_kaggle=false` remains accurate: no leaderboard submission was
performed. The saved file passed the local output contract only.

## Current competition requirements

The [official evaluation overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
was checked on 2026-09-23. It lists macro-averaged ROC AUC over twelve targets,
an internet-disabled CPU or GPU notebook limit of nine hours, the required
`submission.csv` name, the October 15, 2026 entry/team deadline, and the
October 22, 2026 final submission deadline (both at 11:59 PM UTC). The
[competition rules page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
returned no readable page body through the available public fetch. The v13
approval's rules and eligibility check is dated 2026-09-12; current account
eligibility and the complete current rules text were not independently
reverified for this selection record.

## Selection boundary

This records the owner's candidate selection and preserves prior evidence. It
does not create a new Kaggle request, run the hidden workload, generate a full
competition-test output, or submit to the leaderboard. Hidden-set runtime and
fallback rates remain unknown. Ticket 24 therefore has a selected candidate
but does not claim full hidden-test delivery or platform acceptance. The
candidate manifest, checkpoints, delivery notebook, adapter, and prior output
remain unchanged.

## Source records

- Ticket 23 decision: `.scratch/rsna-2026-fresh/issues/23-compare-fold-ensembles-with-all-data-refitting.md`
- Ticket 23 comparison: `docs/evidence/ticket-23-delivery-strategy.md`
- Ticket 12 Kaggle record: `docs/evidence/ticket-12-kaggle.md`
- Ticket 24 selection and remaining gates: `.scratch/rsna-2026-fresh/issues/24-select-and-deliver-the-final-candidate.md`
