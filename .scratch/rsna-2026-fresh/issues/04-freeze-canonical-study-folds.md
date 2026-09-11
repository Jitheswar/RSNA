# 04: Freeze canonical study folds

**What to build:** Publish immutable private three-fold membership and public-safe generation evidence for every eligible training study.

**Blocked by:** 03: Inventory and safely admit competition data

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Use multilabel-aware allocation for gold studies and deterministic allocation for report-only studies, respecting trustworthy groups across both cohorts.
- [ ] Record generation method, named seed, manifest version/checksum and all target class counts; no permanent holdout and no seed search against model scores.
- [ ] Demonstrate complete unique membership, consistent label joins and exclusion of every held-out study's images, metadata and reports from fold adaptation.
- [ ] If no reliable patient key exists, record that repeated-patient leakage cannot be ruled out; later reliable grouping invalidates affected comparisons and requires new folds.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V02 — Patient-key scope and reliability

- [ ] Inspect the legitimate competition grouping fields across indexed studies, including missingness, within-study consistency, cross-study reuse and issuer/site scope. A patient key is reliable only when its scope is supported by competition documentation or equivalent authoritative provenance and there are zero unexplained identity collisions/contradictions. Otherwise record the examined fields, why linkage is unavailable or untrustworthy, and the explicit study-level-separation limitation. Do not infer identity from report similarity or anatomy.

**Blocking consequence:** Block patient-grouped claims and use of an untrustworthy key. Unresolved contradictory grouping evidence blocks fold freeze for affected data until resolved. A documented conclusion that no reliable patient key exists permits the already-agreed StudyInstanceUID-based baseline; it does not permit patient-independent generalization claims. A subsequently reliable key requires new folds and invalidation of materially affected comparisons.

### V14 — Frozen fold manifest and reference evaluation support

- [ ] Verify exactly one membership record for every eligible study per repeat, valid declared fold IDs, immutable manifest checksum, reproducible generation metadata and zero overlap of reliable groups across folds. Record all twelve gold-target positive/negative counts in every fold. Demonstrate that the proposed three-fold allocation has both classes for every target where feasible under mandatory grouping; do not search seeds for favorable model scores. Any mathematically unavoidable one-class fold-target is documented as undefined, with pooled class support and limitations explicit.

**Blocking consequence:** Block report-derived adaptation and serious CV until membership, grouping and label joins are consistent. Do not split a reliable group or substitute 0.5 to manufacture support. If grouping prevents the planned allocation, revise the fold design before freezing it; affected scores cannot be presented as comparable to a previous manifest.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Implementation evidence

Implemented the deterministic three-fold freeze seam in
`src/rsna_knee/folds.py`. It consumes ticket-03's private source inventory,
rejects unresolved partial labels, assigns gold studies with a deterministic
multilabel-aware greedy allocator, assigns report-only studies with stable
identifier hashing, and keeps explicitly reliable grouping keys unsplit across
both cohorts. The default path records study-level separation and the explicit
limitation that repeated-patient leakage cannot be ruled out.

The private manifest records every membership row, labels, grouping audit,
generation method, named seed, source identity, adaptation exclusion policy,
per-fold positive/negative counts for all twelve targets, and a checksum. A
separate immutable public evidence file omits study, series, report, and group
identifiers. Reload verifies the checksum and source membership/label joins;
held-out studies are rejected from adaptation inputs.

The runner is `slurm/run_ticket04_freeze_folds.sbatch`, bounded to one
`cluster1-cpu` node, two CPUs, 8 GB, and one hour without exclusive allocation.
Synthetic tests and public-safe evidence are recorded in
`docs/evidence/ticket-04-folds.md`.

No private competition fold manifest was generated. V02 patient-key evidence,
real-corpus membership/label-join verification, and V14 completion remain
unresolved; this implementation does not claim those gates passed.

### 2026-09-11 — Owner-approved version-2 fold-generation correction

The owner approved the correction proposed by the fold audit. The version-1
phase-2 greedy scored candidate folds by absolute post-add imbalance, so real
fold 1 kept only its four support-seed studies (gold folds 26 / 4 / 28).

The correction is versioned as generation v2 (`rsna-knee-fold-manifest-v2`):

- **Scoring.** Phase 2 scores the change in total imbalance.
- **Size cap.** Each fold holds `floor(N/3)` or `ceil(N/3)` gold studies under
  study-level grouping.
- **Gates.** Generation refuses to publish unless gold fold-size balance,
  per-target class balance (`max(3, 0.15 × expected)`), both-class support and
  deterministic regeneration all pass.
- **Tests.** Regression tests reproduce the seed-only fold under v1, and random
  multilabel tests at n = 36, 58, 90 and 300 pass every v2 gate.

The corrected canonical manifest `8460234d…2f287fb8` was generated exactly once
in Slurm job 6941. It reuses the superseded manifest's source, grouping
evidence, named seed and repeat; no seed search was performed and no model
score was consulted. Gold folds are 20 / 19 / 19, the worst class-count
deviation is 1.53 studies, and all twelve targets have both classes in every
fold.

V14 was re-run in job 6942 and **passed**. The checks cover exact membership
and label joins, the checksum, independent regeneration from the inventory,
all balance/support gates and single-correction seed provenance; reliable-group
overlap is not applicable because no reliable patient key exists. V02
therefore remains the documented study-level limitation: repeated-patient
leakage cannot be ruled out.

The v1 manifest `ae81f327…d51ad3e7` is retained unchanged as superseded
historical evidence. Evidence: `docs/evidence/ticket-04-folds.md`.
