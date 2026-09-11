# 10: Produce complete three-fold OOF evidence

**What to build:** Complete gold-only development CV and produce trustworthy pooled OOF metrics and regenerated comparison views.

**Blocked by:** 09: Resume an interrupted experiment safely

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Produce exactly one prediction from the held-out fold for each of the 58 expected reference studies, reconciling actual verified cohort membership and target label joins.
- [ ] Compute pooled per-target ROC-AUC and its twelve-target macro average; separately report fold metrics, variation and class counts, leaving one-class AUC undefined without silent reduced-target averaging.
- [ ] Verify metric behavior against independently known ranking cases; report study/group bootstrap uncertainty and undefined resamples, evaluating additional fixed repeats separately.
- [ ] Make clear that repeated development CV supports model selection and is not an untouched internal test or clinical-generalization proof.
- [ ] Each job writes only its own atomic records; a read-only aggregation step generates comparison views without mutating source runs.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Real three-fold OOF execution

The explicit implementation request authorized execution. The bounded
cluster1-cpu fold jobs 6930, 6931 and 6932 completed all three frozen folds
under one common configuration fingerprint, with 100 successful optimizer
steps per fold and held-out counts 26, 4 and 28. The private inventory and
fold-manifest reconciliation passed for 4,407 training studies and 58 gold
reference studies. The aggregation job 6934 recomputed the metrics from the
three atomic run records and wrote regenerated public/private views without
mutating those source records.

The pooled OOF result contains exactly 58 predictions and all twelve target
AUCs are defined. The pooled twelve-target macro ROC-AUC is
`0.38417970047680566`. Fold macro AUCs are
`0.43171391044456064`, `0.4930555555555555` and `0.4942663572196966`, with
population standard deviation `0.02920630081021206`. A deterministic
study-level bootstrap used 2,000 resamples and seed 20260911; one macro and
one MCL resample were undefined and were reported rather than imputed or
dropped from the audit.

Independent ranking-fixture tests cover perfect, reversed and tied rankings,
and the public result contains no study identifiers. The grouping audit remains
unavailable, so the uncertainty unit is study rather than an unattested
patient/group key. This is development CV for model selection, not an
untouched internal test or clinical-generalization proof. V09 shared-storage
evidence and any GPU, Kaggle, promotion, upload or final-selection claims
remain unresolved and are not asserted here.

The compute-node environment record contains an explicit unavailable Git
revision; environment and scientific identity fingerprints are retained, but
this execution is not clean-release revision provenance.
