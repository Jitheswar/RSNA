# 18: Measure the controlled benefit of report supervision

**What to build:** Produce a controlled OOF comparison of gold-only and gold-plus-clean-report supervision on the same image baseline.

**Blocked by:** 16: Compare one approved local language-model extractor; 17: Train one fold with admissible weak labels

**Status:** resolved

**Authorization:** Direct user authorization to implement this ticket was received on 2026-09-23; it superseded the original publication-only authorization.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Fix folds, preprocessing, architecture, initialization strategy/seeds, optimizer/schedule, augmentation, batch size, successful-step budget, BCE and evaluation across arms.
- [x] Record mixture policy, gold/weak study exposures, supervised entries and compute; do not conceal unequal data exposure under an epoch count.
- [x] Evaluate the selected admissible extractor's labels with pooled/per-target OOF AUC, fold consistency and uncertainty; avoid attributing tiny noisy differences to a real benefit.
- [x] A negative or inconclusive result completes the experiment and preserves the gold-only path; do not add focal/class-weighted/soft-label losses to rescue this controlled comparison.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V15 — Derived-supervision provenance and isolation

- [x] For each extractor/rule/teacher/pseudo-label/student artifact, resolve versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set identities and all upstream adaptation inputs. Verify zero prohibited held-out input influence for the consuming fold and no unresolved lineage. Enforce reference precedence and zero-mask unknown/uncertain/failed states. Frozen evaluation-only outputs must be distinguishable from admitted training labels.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-23 — controlled three-fold comparison completed

Implemented the paired OOF comparator and Slurm fold adapter, then completed
all six matched training runs with code revision
`034a4d1c5e36c24c8a8382740a2c04f9694ff960`. Each arm used the same frozen
folds and training configuration, 100 successful optimizer steps per fold,
and final-checkpoint evaluation. Run/configuration fold partitions,
supervision checksums, held-out exclusion, report admission, source lineage,
compute limits, and pinned code revision were validated for every run.

The pooled gold-only macro AUC was 0.4643; gold plus deterministic Ticket 15
report supervision was 0.4772 (delta +0.0129). The paired 95% study-bootstrap
interval was [-0.0301, 0.0568], and fold deltas were positive in one fold and
negative in two. The result is inconclusive; no report-supervision benefit is
claimed. Focal, class-weighted, and soft-label losses were not used. Per-target
metrics, fold mean/spread/range, source-stratified training exposures, Slurm
requests, observed CPU time, and peak RSS are in the external reports.

Ticket 16 Astra outputs were excluded because no training-admission rule had
been predeclared; only the frozen deterministic Ticket 15 labels were used.
The report output paths are
[`ticket-18-report.md`](/home/ch.jitheswar/ticket-18-report.md)
and
[`ticket-18-report.json`](/home/ch.jitheswar/ticket-18-report.json),
both outside the RSNA folder and free of study/source-row identifiers. Two
setup attempts against occupied preflight paths stopped before training; the
report records them as excluded from the completed-run comparison.

Verified scope: these three development folds and the frozen deterministic
report artifact. Limitations remain: there is no trusted patient-group key,
these are development-CV results rather than an untouched internal test, and
they support no clinical-generalization claim. Any future extractor or
teacher/student artifact needs its own admission and lineage checks.
