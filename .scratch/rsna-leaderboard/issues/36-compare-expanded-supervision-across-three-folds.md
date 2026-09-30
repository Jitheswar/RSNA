# 36: Compare expanded supervision across all three folds

**What to build:** The expanded admissible image cohort produces a complete matched three-fold comparison against reference-only training, with a promotion or rejection decision.

**Blocked by:** 35: [Expand eligible image preparation in restartable batches](35-expand-eligible-image-preparation-in-restartable-batches.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** ready-for-human

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Freeze the expanded cohort, admitted label policy, training configuration and budget before evaluating the comparison.
- [x] Complete all three folds with strict adaptation isolation and matched reference exposure; record additional weak exposure separately.
- [x] Reconcile every held-out prediction and source/run/label identity through the existing OOF and supervision-comparison interfaces.
- [x] Report pooled and within-fold AUC, per-target support, fold variation, paired uncertainty, undefined resamples and observed compute. Repeated validation remains development evidence.
- [x] Reject an incompatible or partially completed arm rather than comparing incomparable totals or silently dropping a fold.
- [x] Produce reloadable checkpoints and candidate-compatible metadata for the selected arm; preserve source runs and the reference-only comparator.
- [x] Record an explicit promotion/rejection decision with uncertainty and resource cost, and update the final-freeze register. An inconclusive result is not renamed a demonstrated gain.

## Completion evidence

Ticket 36 completed the approved CPU-only comparison. The final frozen plan is `artifacts/private/ticket36-expanded-supervision/ticket36-run-plan.json` (SHA-256 `c13f2fd421477cdb0e4dd8d6ccd72c9c3e02ad9c2bf4cfb13bb5f97d11f190f8`), with source manifest `bf055f303bcc00154d2d76a6ad3720cda5c7bfc356a4c68bdcb2172505ff6789`; the approval still matches that plan and retains zero spend, zero GPU, no network, and no promotion or release authority.

The sealed feature manifest contains 4,228 studies: 58 gold reference studies and 4,170 image-ready report studies. Eight incomplete report inputs were excluded. The comparison has 58 held-out predictions (fold sizes 20/19/19), all 12 targets defined, matched gold exposure, and report-only exposure kept separate. The reference pooled macro AUC is `0.518125`; report-supervised is `0.528005`; pooled delta is `+0.009880`. Fold deltas are `+0.015521`, `+0.044826`, and `+0.020188`. The paired 2,000-resample study bootstrap 95% interval is `[-0.035188, +0.055273]` with one undefined resample. This is inconclusive development evidence, not a demonstrated gain.

The predeclared rule advances report supervision to a separate candidate review because the pooled estimate and all three fold deltas are positive. No candidate was promoted or frozen. Both arms retain three reloadable fold checkpoints and candidate-compatible review metadata; final-candidate freeze and release remain blocked. Detailed pooled and per-target support/AUC results are in `artifacts/private/ticket36-expanded-supervision/comparison.md` and `.json`; compute, decision, and review artifacts are in the same private run root. The standalone implementation report is `/home/ch.jitheswar/ticket-36-implementation-report.md`.

Observed feature compute covers 179 of 183 recorded allocations; 44 measured attempts exited nonzero, four allocations lack usable timer output, and five training allocations have only one usable outer timer. Six fold-run counters and the finalization allocation are separately recorded. External spend and GPU use were zero. The storage projection was 466,010,251,658 bytes against a 2 TB soft limit. Failed and partial attempts, including jobs 7776–7779 finalization evidence, are archived under `artifacts/private/ticket36-expanded-supervision/private/attempts/`. Static syntax and whitespace checks passed; no tests were run.
