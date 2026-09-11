# 18: Measure the controlled benefit of report supervision

**What to build:** Produce a controlled OOF comparison of gold-only and gold-plus-clean-report supervision on the same image baseline.

**Blocked by:** 16: Compare one approved local language-model extractor; 17: Train one fold with admissible weak labels

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Fix folds, preprocessing, architecture, initialization strategy/seeds, optimizer/schedule, augmentation, batch size, successful-step budget, BCE and evaluation across arms.
- [ ] Record mixture policy, gold/weak study exposures, supervised entries and compute; do not conceal unequal data exposure under an epoch count.
- [ ] Evaluate the selected admissible extractor's labels with pooled/per-target OOF AUC, fold consistency and uncertainty; avoid attributing tiny noisy differences to a real benefit.
- [ ] A negative or inconclusive result completes the experiment and preserves the gold-only path; do not add focal/class-weighted/soft-label losses to rescue this controlled comparison.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V15 — Derived-supervision provenance and isolation

- [ ] For each extractor/rule/teacher/pseudo-label/student artifact, resolve versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set identities and all upstream adaptation inputs. Verify zero prohibited held-out input influence for the consuming fold and no unresolved lineage. Enforce reference precedence and zero-mask unknown/uncertain/failed states. Frozen evaluation-only outputs must be distinguishable from admitted training labels.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.
