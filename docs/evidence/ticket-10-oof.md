# Ticket 10 complete three-fold OOF evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation,
synthetic-fixture, and real-corpus execution evidence; it contains no study
identifiers, reports, raw paths, private manifests, or per-study prediction
rows.

## Implemented seam

`rsna_knee.evaluation.evaluate_oof_predictions` accepts a mapping of study
identifiers to twelve scores and a checksum-covered frozen fold manifest. It
rejects missing or extra gold studies, validates finite bounded scores, joins
labels from the manifest, recomputes pooled target AUCs, and evaluates every
held-out fold independently. The primary pooled macro is defined only when all
twelve target AUCs are defined. One-class targets remain `null`; the separate
defined-target summary is labeled and never substituted silently.

`bootstrap_oof_predictions` samples studies, or reliable groups when the
grouping assessment is explicitly reliable. It uses a fixed seed, records the
resample count and percentile interval method, and preserves the number of
undefined target and macro resamples rather than dropping them. Additional
fixed-seed fold repeats carry a separate manifest/repeat identity and must be
aggregated separately.

`rsna_knee.oof` loads one completed reference-only `run.json` per fold plus
its configuration, metrics and final/latest checkpoint evidence, reconciles
recorded membership and configuration identity with the frozen manifest,
requires one run for each of folds 0, 1 and 2 with one common configuration
fingerprint, and recomputes metrics instead of trusting stored metric fields.
When the private ticket-03 inventory is supplied, the source cohort and all
twelve reference-label joins are independently validated. The aggregation
writer requires source run paths and rejects report destinations inside them;
it only creates regenerated views and does not mutate source run directories.

## Tests

The public tests cover perfect, reversed and tied ranking behavior, complete
OOF identity, pooled and fold metrics, explicit undefined support,
reproducible study bootstrap, reliable-group bootstrap, artifact-contract and
configuration mismatch rejection, wrong-fold prediction rejection, and
read-only aggregation output behavior.

The execution adapters are
`slurm/run_ticket10_fold.sbatch` for one independent fold and
`slurm/run_ticket10_aggregate.sbatch` for the read-only comparison view. Fold
jobs request one non-exclusive `cluster1-cpu` node, 16 CPUs, 64 GB and six
hours; a shared lock and Slurm-state check cap active fold jobs at two.
Aggregation requests one node, two CPUs, 8 GB and one hour.

## Real three-fold execution (superseded, historical)

> **Superseded for comparison (2026-09-11).** This execution used the
> version-1 fold manifest `ae81f327…d51ad3e7`, whose defective gold allocation
> (26 / 4 / 28 studies) let fold-level base rates dominate the pooled metric.
> Per-fold mean scores tracked training prevalence (Pearson 0.985 over the 36
> fold × target cells), and a training-prevalence-only predictor reached a
> pooled macro of 0.382. The records and metrics below are retained unchanged
> as historical evidence and must not be compared with any version-2 result.
> The corrected rerun is recorded at the end of this document.

The real run used the pinned Python 3.11 environment, the existing verified
ticket-03 inventory, the existing verified ticket-04 fold manifest, the
existing ticket-05 training integration manifest, and the source archive. The
inventory-to-fold reconciliation passed for 4,407 training studies and 58
complete gold studies. The frozen manifest assigned 26, 4 and 28 gold studies
to folds 0, 1 and 2, respectively. Fold jobs 6930, 6931 and 6932 completed
folds 1, 2 and 0 under one common configuration fingerprint. The read-only
aggregation job 6934 regenerated the public and private OOF views with 2,000
study bootstrap resamples using seed 20260911.

The public aggregate records exactly 58 held-out predictions, twelve pooled
target AUCs, pooled macro AUC `0.38417970047680566`, fold macro AUCs
`0.43171391044456064`, `0.4930555555555555` and `0.4942663572196966`, and
population fold-macro standard deviation `0.02920630081021206`. All twelve
pooled target AUCs are defined. The study-bootstrap macro interval is
`0.33494579762627497` to `0.43939929879272754` with median
`0.38428722266773246`; 1 of 2,000 macro resamples is undefined, as is 1 MCL
target resample. All other target resamples are defined. The aggregation
verified that source run records were unchanged and that the public views
contain no study identifiers.

Its study-level bootstrap unit is intentional: the grouping audit status is
`unavailable`, so repeated-patient leakage cannot be ruled out and no
unattested patient key is used.

## Interpretation and limits

This is repeated development cross-validation used for model selection. It is
not an untouched internal test, and it does not establish clinical
generalization. The 58 reference studies remain sparse, several target/fold
supports are small, study-level splitting cannot rule out repeated-patient
leakage, and bootstrap intervals describe estimator uncertainty rather than
external validity. The compute-node environment record has an explicit but
unavailable Git revision; its environment fingerprint and scientific artifact
identities are retained privately, so this is not clean-release revision
provenance. No competitive-performance, GPU-compatibility, Kaggle, promotion,
upload, or final-selection claim is made.

## Corrected three-fold rerun on the version-2 manifest (2026-09-11)

The rerun started only after V14 passed on the corrected manifest
`8460234d…2f287fb8` (ticket 04, jobs 6941 and 6942).

- **Training runs.** Fold jobs **6943**, **6944** and **6945**
  (cluster1-cpu, `slurm/run_ticket10_fold.sbatch`) trained folds 0, 1 and 2.
  They used the unchanged mechanics configuration, fingerprint
  `aa83200904398e7ffb07ba9f1f0c2f1fbb5be4d2b0dfcd885d1008cf9b5c49ea`, and the
  100-successful-step budget.
  - Each fold passed its CPU preflight and completed 100 optimizer and
    scheduler steps with zero all-masked skips.
  - Training/held-out study counts were 38/20, 39/19 and 39/19.
- **Provenance.** The compute-node environment records the code identity
  `git-1fdbc63+uncommitted;src-sha256=43ac40f9…3d64`.
- **Aggregation.** Read-only aggregation job **6947** recomputed all metrics
  from the three atomic run records. It used 2,000 study-bootstrap resamples
  and 10,000 within-fold label permutations, both with seed `20260911`.

| Metric | Version-2 rerun | Superseded v1 run (historical) |
| --- | ---: | ---: |
| Pooled twelve-target macro AUC | **0.464313** | 0.384180 |
| Study-bootstrap 95% interval (pooled macro) | 0.404–0.524 | 0.335–0.439 |
| Fold macro AUCs | 0.492283 / 0.421119 / 0.488201 | 0.431714 / 0.493056 / 0.494266 |
| Pair-weighted within-fold macro AUC | **0.467378** (95% 0.411–0.523) | 0.489341 |
| Cross-fold macro AUC | 0.462880 | 0.310796 |
| Mean within-fold pair share | 0.332 | 0.418 |
| Training-prevalence-only pooled macro (no images) | **0.474353** | 0.382168 |
| Fold mean score vs training prevalence (Pearson, 36 cells) | 0.979 | 0.985 |
| Within-fold permutation p (one-sided) | **0.918** | 0.672 |

Pooled and within-fold target AUCs for the version-2 rerun:

| Target | Pooled | Within-fold | Cross-fold | Prevalence baseline |
| --- | ---: | ---: | ---: | ---: |
| ACL | 0.462010 | 0.553903 | 0.416819 | 0.443015 |
| MCL | 0.351474 | 0.360544 | 0.346939 | 0.493197 |
| Medial Meniscus | 0.443510 | 0.393502 | 0.468468 | 0.476562 |
| Lateral Meniscus | 0.427329 | 0.467925 | 0.407407 | 0.437888 |
| Medial OA | 0.480620 | 0.446512 | 0.497674 | 0.492248 |
| Lateral OA | 0.411992 | 0.331395 | 0.452174 | 0.462282 |
| PF OA | 0.465894 | 0.471042 | 0.463320 | 0.490991 |
| Effusion | 0.577640 | 0.638060 | 0.547486 | 0.475776 |
| Synovitis | 0.445639 | 0.476534 | 0.430357 | 0.453405 |
| Baker's | 0.467391 | 0.413043 | 0.494565 | 0.492754 |
| Contusion | 0.497976 | 0.506073 | 0.493927 | 0.482456 |
| Fracture | 0.540278 | 0.550000 | 0.535417 | 0.491667 |

All twelve pooled and all 36 fold-target AUCs are defined. One of 2,000
bootstrap resamples had an undefined macro (one MCL draw), and it is counted
rather than dropped.

**Interpretation.** The correction removed the fold-imbalance artifact: cross-
and within-fold AUCs now agree (0.463 vs 0.467), where the v1 values were 0.311
vs 0.489. It did not reveal image signal.

- Every fold model still outputs essentially its training prevalence. Fold
  mean scores track training prevalence (r = 0.979), and within-fold score SDs
  are 0.0001–0.010.
- The pre-specified test is a one-sided within-fold permutation test of the
  pair-weighted within-fold macro, with measurable signal only if p < 0.05. It
  gives p = 0.918 (null mean 0.500, SD 0.024), so **no measurable image signal
  is claimed**.
- The pooled macro's 95% interval contains 0.5, and the pooled macro sits
  below the no-image prevalence baseline.
- Single-target values such as Effusion (within-fold 0.638) are
  unadjusted and are not evidence of signal on their own.

This remains mechanics-baseline development CV for model selection. It is not
an untouched internal test and establishes no clinical or competitive claim.

**Cross-check against the audit.** Historical aggregation job **6946** re-read
the superseded v1 runs with `RSNA_TICKET10_ALLOW_SUPERSEDED_FOLD_MANIFEST=1`
and wrote to a separate directory. The new diagnostics reproduced the fold
audit's independent computations exactly:

- pooled 0.384180;
- within-fold 0.489341 and cross-fold 0.310796;
- prevalence baseline 0.382168;
- Pearson 0.985055;
- fold-standardized 0.493776.

All 76 superseded ticket 04/08/10 evidence files were verified byte-identical
after the rerun.
