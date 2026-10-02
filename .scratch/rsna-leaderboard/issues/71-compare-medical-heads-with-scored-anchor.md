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
- [x] Exact anchor and medical native/terminal/source/asset receipts verified.
- [x] Immutable analysis implementation and full input/resource admission frozen.
- [ ] Independent fixed paired analysis and diagnostics completed.
- [ ] Provisional disposition recorded without relabeling validation as rank proof.

## Comments

- October2: protocol fixed with Ticket68 at `wait_for_complete_bank`, fitting
  plan admittedFalse/fitting launchedFalse; Ticket70 is prepared and unlaunched.
  No predictions inspected or labels scored for this decision.

October2,16:20UTC — fixed analysis and separate independent replay implemented;
103 combined Ticket70/71 guards pass and analysis module typing passes. Prepared
immutable protocol planf90bb2ad7a216eb3a235595775ae5d559569b707a0b8b6b5b20369b57cb2934d
under artifacts/private/ticket71-scored-anchor-analysis-v1. Forty-five static
local dependency modules are frozen; portable medical adapter remains editable.
Final input/native/terminal receipts and CPU resource admission remain absent;
no prediction/label analysis has run. No candidate/scoring/promotion follows.

October2,16:47UTC: v1 remains frozen against the failed v3 anchor and cannot
advance. Mechanically superseding v2 plan25fa04a170d1683cbc47c1bbaf573a64399b6302b08b3d9a2f4f6e44db84e560
binds the separately admitted v4 metadata recovery and its independent verifier.
Every scientific protocol field is identical to v1; no retuning follows the
now verified inconclusive standalone medical result. Final anchor/input proof,
analysis CPU admission, primary terminal and independent replay still required.

October2,17:19UTC: exact medical and anchor native/source/terminal/runtime
chains pass complete final input admission. Original v2 plan and its10% vote
remain unchanged. Dedicated lifecycle/admission plus two sole1h CPU components
(3540s watchdog each) reuse the previously protected7200s comparison/replay
allowance; zero spend, no model fitting/scoring/promotion. Protected full route
and review projectOctober10,19:49UTC and1968671467335bytes<2TB. Five lifecycle
guards pass. Primary analysis and independent replay remain unlaunched at this
admission; every metric/support/vote/gate must reproduce after actual terminals.
