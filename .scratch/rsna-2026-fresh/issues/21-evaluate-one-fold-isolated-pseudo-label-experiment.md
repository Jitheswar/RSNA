# 21: Evaluate one fold-isolated pseudo-label experiment

**What to build:** Optionally test one fully traceable teacher-to-pseudo-label-to-student hypothesis and its OOF result.

**Blocked by:** 20: Evaluate the strongest image/supervision combination

**Status:** resolved

**Type:** task

**Authorization:** Planning-only note superseded by the owner's direct instruction to implement Ticket 21 on 2026-09-23. GPU training, model/data upload, competition submission and other separately approval-gated actions remain unauthorized.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Defined and wrote the fold-0 comparator, teacher cohort, budget, eligible/excluded cohorts and frozen MRI-only confidence policy before teacher fitting.
- [ ] Partial: teacher, design and pseudo-score artifacts record fold/repeat, code/config/checkpoint hashes, candidate identities, settings and output hashes. No student was trained because zero pseudo-label studies passed admission, so student lineage is not applicable and remains undemonstrated.
- [x] Held-out studies/reports were excluded from teacher fitting and candidate generation. Admission rejects held-out and contaminated/unresolved lineage; tests cover contamination, checkpoint identity and uncertain-score masking.
- [x] Reported one-fold teacher OOF uncertainty and the full observed Slurm training/inference cost. No student OOF score or paired delta is estimable because no student data were admitted.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V15 — Derived-supervision provenance and isolation

- [ ] Partial for this ticket: teacher and pseudo-label artifacts resolve versioned code/configuration, source/checkpoint hashes, fold/repeat, eligible/excluded identities and upstream adaptation inputs. Held-out influence is excluded; uncertain labels are masked; the tests reject contaminated/unresolved lineage. No student artifact exists because admission returned zero studies, and the broader V15 gate remains open.

**Blocking consequence:** Reject artifacts with ambiguous or contaminated lineage from that fold's training and model-selection evidence. Invalidate and rerun materially affected comparisons after repair; an apparent AUC gain cannot override the gate. Clean reference-only work remains possible.

## Completion evidence

- [x] Recorded the real fold-0 no-label result, teacher-only OOF uncertainty, measured job cost, verified scope and remaining limits without study identifiers or report text. No student comparison, complete-repeat OOF, model promotion or approval-dependent action is claimed.

## Answer

The predeclared fold-0 MRI-only experiment completed within the bounded CPU allocation. All 65 eligible report-only training studies abstained under the frozen 0.10/0.90 policy, so no student or paired effect was produced. The gold-only teacher's macro AUC was 0.4923 on 20 held-out gold studies, with a 95% study-bootstrap interval of 0.4281–0.5662. The active gold-only mechanics baseline remains unchanged. Student lineage and the broader V15 gate remain partial because no pseudo-labels were admitted. The private run artifacts retain detailed lineage, uncertainty and compute records.

## Comments

- 2026-09-23 — Owner explicitly instructed Ticket 21 implementation. Ticket 20 was resolved as a no-run adoption-gate outcome, so the established gold-only mechanics baseline remained the fixed comparator.
- 2026-09-23 — Slurm job 7109 completed the predeclared fold-0 experiment. Of 65 eligible report-only training studies, zero passed the 0.10/0.90 confidence rule; all abstained. The gold-only teacher's 20 held-out predictions had macro AUC 0.4923 (95% study-bootstrap interval 0.4281–0.5662). No student or paired effect was produced. Detailed lineage and cost records remain in the private run artifacts.
- 2026-09-23 — The first Slurm attempt exposed an unset `SLURM_TIMELIMIT`; the strict preflight remained in place and the adapter now resolves the active limit from `scontrol`. The corrected run passed resource preflight and completed within the six-hour, non-exclusive CPU allocation.
