# 32: Complete the three-fold ResNet-18 comparison

**What to build:** The selected ResNet-18 configuration yields a complete reference OOF comparison and a documented decision about whether it warrants promotion or further investigation.

**Blocked by:** 31: [Run one ResNet-18 fold through the complete lifecycle](31-run-one-resnet18-fold-through-complete-lifecycle.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Disposition:** completed — provisional/inconclusive; not promoted to final candidate.

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Freeze one scientific configuration and complete all three corrected folds on the same reference cohort; reject missing, duplicate, contaminated or mismatched predictions.
- [x] Reuse only externally frozen features across folds; train all fitted heads/transforms independently with exact fold exclusions.
- [x] Report pooled twelve-target macro AUC, per-target support, fold results, pair-weighted within-fold AUC, score-offset diagnostics and paired uncertainty against the preserved baseline.
- [x] Include constant and training-prevalence controls and a declared image-ablation/shuffle diagnostic; training-only fitting cannot establish image generalization.
- [x] Use 2,000 paired bootstrap resamples for the real comparison, counting undefined outcomes and resampling reliable groups where available.
- [x] Verify the comparison boundary using controlled metric/cohort fixtures and preserve all source run artifacts; fixtures do not stand in for real performance.
- [x] Record a promotion or rejection decision for final-candidate consideration with effect, uncertainty, cost and rationale; a provisional candidate remains explicitly inconclusive.
- [x] Identify any triggered supervision, alternative-representation or fine-tuning branch in the final-freeze register before it begins. If a branch is not selected, say so explicitly.
- [x] Reserve remaining compute for the complete comparison; no extra GPU allowance is implied and no partial set of folds is reported as complete.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.



## Comments

### 2026-09-25T12:32:32Z — TSK7 gate satisfied; comparison preregistered

Ticket 31 closed as a completed one-fold non-promotion. The predeclared validation policy explicitly requires completing planned folds before interpreting an improvement, and this ticket is the declared corrected three-fold confirmation. This satisfies TSK7's conditional path.

Created the separate Ticket 32 approval at [`../resource-approvals/ticket32-resnet18-imagenet1k-v1.json`](../resource-approvals/ticket32-resnet18-imagenet1k-v1.json); the Ticket 31 approval and outputs remain unchanged. Frozen the scientific and analysis plan at `artifacts/private/ticket32-resnet18/ticket32-run-plan.json`. The plan uses the corrected 58-study reference cohort, the preserved corrected v2 baseline, the same frozen ResNet-18 configuration across folds, 2,000 paired study-bootstrap resamples, training-prevalence and constant controls, a zero-frozen-image-feature ablation, and within-fold label permutation diagnostics. Reliable patient grouping is unavailable in the admitted manifest, so study-level resampling is used and repeated-patient leakage remains a limitation.

A focused controlled comparison-boundary check passed: `tests/test_image_model_comparison.py` — 2 passed in 2.43s. This is software evidence only. A sequential three-fold job is prepared at `artifacts/private/ticket32-resnet18/run-three-fold-cpu.sbatch`, scoped to `cluster1-cpu`, 16 CPUs, 64 GiB, six hours, no GPU and zero external spend. No fold outputs are yet claimed here.


### 2026-09-25T13:11:19Z — Three-fold comparison complete; provisional/inconclusive

All three corrected reference folds completed in Slurm job `7169` on `cluster1-cpu`, using 16 CPUs and 64 GiB. Each completed 100/100 optimizer steps, with no skipped batches. Held-out coverage was 20/19/19 studies and all 58 predictions matched the corrected fold manifest. The exact run IDs and checkpoint digests are recorded in `artifacts/private/ticket32-resnet18/ticket32-run-plan.json`; source run artifacts remain under `artifacts/private/ticket32-resnet18/fold-0/`, `fold-1/` and `fold-2/`.

Candidate pooled macro AUC was `0.5181249302` versus `0.4643127624` for the corrected v2 baseline (`+0.0538121678`). The paired 2,000-study-bootstrap 95% interval was `-0.0045235889` to `0.1153942616` (1 undefined resample). Candidate pair-weighted within-fold macro AUC was `0.5461538465`; the predeclared 10,000-resample within-fold label-permutation test gave `p=0.039296`, indicating measurable signal by the existing rule. The candidate OOF bootstrap interval (`0.471360` to `0.560698`) includes `0.5`. Full-image minus zero-image-feature ablation delta was `+0.0157481`, with 95% interval `-0.045430` to `0.077447` (1 undefined). Per-target results are mixed.

**Decision:** provisional/inconclusive; not promoted to final-candidate consideration and not rejected as unhelpful. This applies the existing policy without adding a numeric threshold. No supervision, alternative-representation or fine-tuning branch was started. Ticket 38 DINOv2 remains a possible separately scoped follow-up; no resource approval or run was created for it. Final-candidate freeze and all release actions remain blocked.

The focused controlled boundary check passed: `tests/test_image_model_comparison.py` — 2 passed in 2.43s. Analysis job `7173` completed checkpoint replay, candidate/baseline OOF evaluation, paired uncertainty, controls and the frozen-feature zero-ablation; replay error was at most `5.96e-8`, with held-out cache hits 20/19/19 and no materialization. The first two analysis attempts stopped before image loading/metric computation and are documented in the private run plan. Full comparison: `artifacts/private/ticket32-resnet18/public/comparison.md`.

### 2026-09-25 — Closed by owner request

The owner requested closure after completion of the three-fold comparison. The result remains provisional/inconclusive and was not promoted; the separate final-candidate, publication, redistribution, and release gate remains blocked.
