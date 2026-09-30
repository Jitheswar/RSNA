# 42: Verify the exact final candidate on Kaggle

**What to build:** The exact final-frozen candidate passes the current Kaggle delivery contract and has its actual execution and scoring state recorded without confusing example runs with hidden scoring.

**Blocked by:** 41: [Maintain the deployable fallback and freeze the final candidate after branch decisions](41-maintain-fallback-and-freeze-final-candidate-after-branch-decisions.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** ready-for-agent

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Dependency meaning:** Ticket 41 must complete phase B. Its parallel fallback-readiness phase alone does not unblock this ticket's final-candidate verification.

## Acceptance criteria

- [ ] Require the final candidate's immutable identity and branch-decision snapshot from the completed final-freeze gate; reject a candidate whose selected branches remain undecided.
- [ ] Refresh the applicable official rules, submission/final-selection allowances, offline runtime limits and account delivery state. Preserve inaccessible facts as unresolved rather than treating the old research as fresh verification.
- [ ] Prepare the exact offline notebook, weights, runtime dependencies and input/output contract with local evidence for that same bundle before requesting any required external-action approval.
- [ ] Use the code-competition submission route and exact version/output binding after applicable authorization; preserve any already authorized exact scope and do not introduce an artificial submission-count cap.
- [ ] Verify remote bundle identity and record execution status, exposed runtime/resource/fallback diagnostics and score status. A successful request or example notebook output is not a completed hidden score.
- [ ] Keep hidden diagnostics that are not exposed explicitly unknown; a projection or the historical three-study execution cannot fill that gap.
- [ ] If delivery fails, retain the fallback and log the cause. Operational fixes preserve scientific identity and repeat affected checks; scientific changes return to the final-freeze gate.
- [ ] Use the established offline adapter/layout/streaming tests for changed packaging and keep real remote execution outside automated tests.
- [ ] Complete intended final-candidate delivery checks by October 13 IST or report the actual limitation and preserve the verified fallback without fabricating readiness.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.
