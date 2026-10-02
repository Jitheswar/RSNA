# 71: Compare medical heads with the exact scored anchor

**Type:** task
**Status:** claimed
**Assignee:** campaign agent
**Blocked by:** 68, 70

## Fixed protocol before medical results

Declare this protocol while Ticket68's live controller is still waiting for
complete bank proof, with no fitted medical head or predictive result. Use the
verified exact0.936 source/asset/runtime Gold58 predictions from Ticket70 and
the verified medical **held-out-fold** report-supervision predictions from
Ticket68. Never average all three medical heads on Gold58: two saw each study.

For all twelve targets, the single primary candidate is
`rankpct(0.90 * rankpct(scored_anchor) + 0.10 * rankpct(medical_report_OOF))`,
using average ties and percentiles over the complete fixed58-study cohort.
No per-target weight, vote-weight grid, threshold, feature, fold, layer, seed,
checkpoint or training-budget selection is introduced. The small uniform vote
tests complementarity to the already strong anchor rather than choosing a
weight from medical results. This is an agent scientific decision under the
standing campaign authorization, not a new exact owner instruction.

Primary statistic: candidate minus exact-anchor twelve-target macro ROC-AUC,
2000 paired study bootstrap resamples, seed20261002, two-sided95% percentile
interval, original three-fold direction and per-target support/deltas. Use the
existing strict metric/support conventions; no patient-group independence is
claimed because reliable patient groups are missing. Compare both predictions
on identical membership/order/known-target masks. No smaller/reselected cohort.

Fixed diagnostics, with the same10% vote and no selection among them:

- Reference-only medical OOF vote versus anchor and versus primary candidate.
- Report-supervision zero-image OOF vote versus anchor and primary candidate.
- Standalone medical report/reference/zero-image metrics for interpretation.

Provisional delivery review requires positive primary paired95% lower bound,
positive direction on all three original folds, plus positive report-image
evidence against its zero-image control under the fixed Ticket68 gates and
positive candidate-minus-zero-image-vote paired lower bound. Diagnostics do
not become alternate candidates after inspecting scores. Failed/unsupported
gates reject this fixed medical overlay; they do not trigger another vote sweep.

Gold58 and public parent selection are adaptive. These checks cannot establish
clean OOF for the public anchor, untouched generalization or a competition
score. They only admit provisional delivery review. No automatic promotion,
upload, scoring request or final selection follows. Original-MRI adapter,
fixed deployed fold-probability mean, exact package/runtime, full-scale timing,
offline competition output and current official requirements must pass first.

## Acceptance

- [x] One exact scientific protocol declared before medical predictive results.
- [ ] Exact anchor and medical native/terminal/source/asset receipts verified.
- [ ] Immutable analysis implementation and full input/resource admission frozen.
- [ ] Independent fixed paired analysis and diagnostics completed.
- [ ] Provisional disposition recorded without relabeling validation as rank proof.

## Comments

- October2: protocol fixed with Ticket68 at `wait_for_complete_bank`, fitting
  plan admittedFalse/fitting launchedFalse; Ticket70 is prepared and unlaunched.
  No predictions inspected or labels scored for this decision.
