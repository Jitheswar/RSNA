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

## Fog

- Real-report accuracy, coverage, latency, clinical review and report-derived
  training benefit remain outside Ticket 17; Slurm execution is evidenced for
  the one bounded fold above.
