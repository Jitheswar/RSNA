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
- 2026-09-23 — Ticket 22's three-fold CPU distillation comparison completed with
  an inconclusive +0.0092 pooled macro-AUC delta over the matched compact
  gold-only student (paired 95% study-bootstrap interval −0.0272 to +0.0503).
  No model was promoted; full-pipeline/hidden-scale runtime, VRAM and complete
  package size remain open under V16:
  [ticket 22](issues/22-evaluate-one-distilled-mri-only-student.md)
  · [external report](../../../ticket-22-report.md).
- 2026-09-23 — Ticket 23 is resolved as a no-refit delivery recommendation.
  The existing three-fold gold-only candidate remains the verified path for
  Ticket 24 to consider; the exact all-data refit's quality and runtime remain
  unknown, and all-reference fallback priors are recorded without changing the
  frozen candidate: [ticket 23](issues/23-compare-fold-ensembles-with-all-data-refitting.md)
  · [public evidence](../../docs/evidence/ticket-23-delivery-strategy.md)
  · [external report](../../../ticket-23-report.md).
- 2026-09-23 — The owner selected Ticket 23's recommended Ticket 11 candidate
  for Ticket 24, then requested one full hidden-test submission without any
  model or pipeline changes. At that point the frozen v13 notebook was bound to
  the 158-study representative input, and no submission had been attempted:
  [ticket 24](issues/24-select-and-deliver-the-final-candidate.md)
  · [selection evidence](../../docs/evidence/ticket-24-candidate-selection.md).
- 2026-09-24 — Ticket 24 is resolved for the owner-approved v6 delivery. The
  exact frozen candidate completed Kaggle code-submission ref `56515942` with
  public score `0.498`; the private score is not present in the current status
  listing. The v3 OOM and v4/v5 layout failures remain historical, and v6's
  hidden-run interruption recovery is explicitly unverified:
  [ticket 24](issues/24-select-and-deliver-the-final-candidate.md)
  · [selection evidence](../../docs/evidence/ticket-24-candidate-selection.md)
  · [machine-readable status](../../artifacts/private/ticket-24-v6-remote-upload-status.json).
- 2026-09-25 — Ticket 25 measured namespaced input-mount discovery against the
  selected Ticket 11 candidate. The 158-study CPU result was byte-identical to
  Ticket 24; no new candidate was selected or uploaded. The synthetic lookup
  probe fell from 32.924 to 0.682 seconds, while the projected 1,300-study
  runtime remains unverified on Kaggle:
  [ticket 25](issues/25-evaluate-one-late-efficiency-optimization.md)
  · [external report](../../../ticket-25-efficiency-report.md).
- 2026-09-25 — Ticket 26 is complete for one generated synthetic study through
  the selected frozen candidate. The report records slice sampling, localizer
  exclusion, all twelve scores, provenance, and limits; no patient data or
  clinical claim is included:
  [ticket 26](issues/26-demonstrate-one-frozen-research-example.md)
  · [external report](../../../ticket-26-report.md).
- 2026-09-25 — Ticket 27 is resolved with a local research/reproduction report
  outside the repository. It records the gold-only OOF uncertainty, actual
  controlled results and unrun experiments, selected candidate provenance,
  resource/recovery evidence, and unresolved limits without identifiers or
  report text:
  [ticket 27](issues/27-produce-the-research-and-reproduction-report.md)
  · [report](../../../RSNA-ticket-27-research-and-reproduction-report.md)
  (SHA-256 prefix `91363b5e…cfb913a`).

## Fog

- Real-report accuracy, coverage, latency, clinical review and report-derived
  training benefit remain outside Ticket 17; Slurm execution is evidenced for
  the one bounded fold above.
