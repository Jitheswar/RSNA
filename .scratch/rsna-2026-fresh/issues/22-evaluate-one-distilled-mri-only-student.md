# 22: Evaluate one distilled MRI-only student

**What to build:** Optionally compare one fold-isolated distilled student with its teacher and an appropriate undistilled baseline.

**Blocked by:** 20: Evaluate the strongest image/supervision combination

**Status:** resolved

**Authorization:** Initially planning-only. The owner's direct instruction to implement Ticket 22 on 2026-09-23 authorized this implementation and the bounded CPU experiment below. GPU execution and the other approval-gated actions remain unauthorized unless separately approved.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Define the fixed three-arm teacher/compact-student/soft-target experiment, 100 student updates per arm per fold, deterministic gold and soft-target exposures, CPU resource ceiling and complete run lineage. Hard-label and soft-target semantics have separate provenance.
- [x] Verify fold-specific teacher/student isolation: held-out studies did not influence the consuming fold's teacher adaptation or student targets; inference remains MRI-only.
- [ ] Partial: report paired OOF/per-target quality, uncertainty, CPU training-source inference timing and checkpoint sizes. Full notebook-to-CSV runtime, hidden-scale projection, peak VRAM and complete delivery-package size remain unmeasured under V16.
- [x] Retain the Ticket 19 teacher checkpoint identities and accept the inconclusive result; no prerequisite pseudo-label experiment or automatic final-model replacement was implied.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V15 — Derived-supervision provenance and isolation

- [x] For the teacher/student chain in this experiment, record versioned code/configuration, source and checkpoint hashes, fold/repeat, eligible/excluded study-set hashes and adaptation lineage. Verify no prohibited held-out influence for the consuming fold. Teacher outputs for held-out studies remain distinguishable as evaluation-only outputs, separate from admitted training targets. Broader project-wide V15 remains open for artifacts outside this ticket.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [x] Record the completed CPU experiment, verified scope, unresolved gates and limitations in the [external report](../../../../ticket-22-report.md), without study, series, or report identifiers. GPU, full-pipeline and future hidden-scale checks are not marked passed.

## Answer

The three-fold CPU experiment completed successfully in Slurm job 7121. The distilled compact student scored 0.4698 pooled OOF macro-AUC versus 0.4606 for the matched gold-only compact student, a +0.0092 delta with paired 95% study-bootstrap interval −0.0272 to +0.0503. This is inconclusive development-CV evidence, not a generalization claim. No model was promoted; the established gold-only active system remains unchanged. CPU inference timing and checkpoint bytes are recorded, while full notebook-to-CSV/hidden-scale runtime, peak VRAM, and full delivery-package size remain open under V16. See the [external report](../../../../ticket-22-report.md) for detailed results, lineage, runtime, checks, and limitations.

## Comments

- 2026-09-23 — Owner directly instructed implementation of Ticket 22. This authorized the bounded CPU experiment; GPU use remains separately unauthorized.
- 2026-09-23 — Three-fold OOF comparison completed in Slurm job 7121 on `cluster1-cpu` (one node, 16 CPUs, 64 GB requested, six-hour limit, nonexclusive). Distilled-vs-undistilled pooled macro-AUC delta was +0.0092 (paired 95% interval −0.0272 to +0.0503); teacher-vs-distilled delta was −0.0011 (interval −0.0402 to +0.0388). The result is inconclusive; the gold-only active system remains unchanged.
- 2026-09-23 — Earlier jobs 7117 and 7119 stopped during preflight/output validation before model training. The runner fixes were applied and job 7121 passed preflight and completed. Full notebook-to-CSV runtime, hidden-scale projection, VRAM and delivery-package size remain partial under V16. See the [external report](../../../../ticket-22-report.md).
