# rsna-2026-fresh wayfinding map

## Notes

The map records concise decisions and context pointers for the local issue
tracker. Restricted report text and identifiers remain outside public
repository artifacts.

## Decisions-so-far

- 2026-09-14 — Ticket 15 implementation is complete for the deterministic,
  synthetic-testable fold boundary. The ticket remains explicit about pending
  private-report and report-derived-training evidence: [ticket 15](issues/15-evaluate-fold-specific-deterministic-report-extraction.md)
  · [detailed external report](../../../ticket-15-report.md).
- 2026-09-21 — Ticket 16 is resolved/closed with a mixed/review-required
  scientific outcome after the corrected Astra aggregation review: [ticket 16](issues/16-compare-one-approved-local-language-model-extractor.md)
  · [corrected final report](../../../tsk2-report.md).
- 2026-09-21 — Ticket 17 is implemented for the synthetic, locally testable
  admission and training scope; no private run or benefit claim is made:
  [ticket 17](issues/17-train-one-fold-with-admissible-weak-labels.md)
  · [public evidence](../../docs/evidence/ticket-17-weak-supervision.md)
  · [external report](../../../ticket-17-report.md).
- 2026-09-21 — Ticket 17 completed exactly one real fold-0 run using the
  pre-existing frozen Ticket 15 labels; Astra remained excluded and no benefit
  claim was made:
  [public evidence](../../docs/evidence/ticket-17-weak-supervision.md)
  · [external report beside tsk.txt](../../../tsk-report.md).
- 2026-09-23 — Ticket 18 is closed after the corrected additive-supervision
  comparison completed with an inconclusive result; paired gold exposure was
  identical in all folds:
  [ticket 18](issues/18-measure-the-controlled-benefit-of-report-supervision.md)
  · [detailed external report](../../../tsk-report.md)
  · [comparison artifacts](../../../ticket-18-additive-v2/comparison/ticket-18-additive-v2.md).
- 2026-09-23 — Ticket 19 implementation and CPU OOF comparison are complete with
  an inconclusive +0.0066 pooled macro-AUC delta (paired 95% interval −0.0295 to
  +0.0442). Full inference runtime, VRAM and complete package size remain open;
  VRAM requires separate GPU approval:
  [ticket 19](issues/19-evaluate-one-stronger-image-model-variant.md)
  · [external report](../../../ticket-19-report.md).
- 2026-09-23 — Ticket 21's bounded fold-0 run completed with no pseudo-labels
  admitted: all 65 eligible candidates abstained under the predeclared policy.
  The gold-only teacher's one-fold macro AUC was 0.4923 (95% study-bootstrap
  interval 0.4281–0.5662); no student effect is estimable. Student lineage remains
  partial, and the active gold-only baseline was not replaced:
  [ticket 21](issues/21-evaluate-one-fold-isolated-pseudo-label-experiment.md).

## Fog

- Real-report accuracy, coverage, latency, clinical review and report-derived
  training benefit remain outside Ticket 17; Slurm execution is evidenced for
  the one bounded fold above.
