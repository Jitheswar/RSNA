# 43: Complete final selection and reproduction handoff

**What to build:** By October 14 IST, the owner receives and, under the applicable authorization, completes evidence-based final selection with the exact reproducible artifacts and an honest campaign report.

**Blocked by:** 42: [Verify the exact final candidate on Kaggle](42-verify-exact-final-candidate-on-kaggle.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** ready-for-agent

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [ ] Reconcile the final candidate and scoring identities with the completed branch-decision snapshot; final selection cannot bypass an unresolved selected/triggered experiment.
- [ ] Prepare the strongest eligible single-model choice and a complementary simple ensemble only when its completed evidence justifies it, within the freshly verified official final-selection allowance.
- [ ] Document close/inconclusive tradeoffs using local evidence, target/fold consistency, robustness and runtime headroom; public/local disagreement is investigated rather than optimized away.
- [ ] Present exact final-selection choices and evidence before any required account-action approval. Verify actual final-selection state after authorized action; a recommendation alone is not a completed selection.
- [ ] Retain the immutable scored fallback and recovery instructions alongside the selected artifact identities and offline reproduction requirements.
- [ ] Account for all allocated GPU attempts, CPU/resource limitations, data/label scope, actual experiment outcomes and submission attempts; no project-imposed candidate-count cap or hidden budget extension is introduced.
- [ ] Produce a public-safe report distinguishing completed, rejected, inconclusive, failed and unrun work, with private evidence pointers that do not expose patient data or credentials.
- [ ] Complete delivery and explicit final selection by October 14 at 23:59 IST. Missed milestones reduce scope or use the eligible fallback; the internal deadline is not silently moved.
- [ ] Report the actual achieved state if authorization, scoring or required evidence is still missing. Do not claim private first place, scientific benefit or final delivery without its corresponding evidence.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.
