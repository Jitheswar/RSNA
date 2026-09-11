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

## Real three-fold execution

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
