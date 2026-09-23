# 20: Evaluate the strongest image/supervision combination

**What to build:** Test whether individually supported image and supervision choices improve the combined MRI-only system.

**Blocked by:** 19: Evaluate one stronger image-model variant

**Status:** resolved

**Authorization:** Originally planning-only. The owner's direct instruction to implement Ticket 20 on 2026-09-23 authorized this evaluation. The prerequisite adoption gates failed, so no combined model run was started. GPU training and other approval-gated actions remain separately unauthorized.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Named the proposed fixed comparison and recorded the eligible Ticket 15 supervision lineage on the frozen folds. The joint arm was not run, and no gain was inferred from independent results.
- [ ] Partial: pooled/per-target OOF evidence, fold variation, uncertainty, and available training costs for the eligible prerequisite baselines are reported. Full-pipeline and combination costs remain unmeasured; the whole-pipeline cost requirement is not passed and remains an open V16 gate.
- [x] Neither the report-supervision nor wider-encoder change earned adoption; the established gold-only mechanics system remains the active path.
- [x] No result was promoted or frozen from Ticket 20; the working competition path was not overwritten or blocked.

## Completion evidence

- [x] Recorded prerequisite research evidence, the no-run decision, verified scope, unresolved gates and limitations in the [external report](../../../../ticket-20-report.md), without study, series, report or source-row identifiers. No future or approval-dependent check is marked passed.

## Comments

- 2026-09-23 — Owner explicitly instructed implementation of Ticket 20.
- 2026-09-23 — Ticket 18's report-supervision delta was −0.0009 (paired 95% study-bootstrap interval −0.0276 to +0.0267); Ticket 19's wider-encoder delta was +0.0066 (interval −0.0295 to +0.0442). Neither earned adoption. No combined run was started, and no combined gain is claimed. The established gold-only system remains active. The full-pipeline cost item remains open under V16. See the [external report](../../../../ticket-20-report.md) for per-target evidence, fold variation, costs, lineage, scope, and limitations.
