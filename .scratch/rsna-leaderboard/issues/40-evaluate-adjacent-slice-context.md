# 40: Evaluate adjacent-slice context

**What to build:** A controlled 2.5D input comparison measures whether geometrically adjacent slices add useful context while preserving the study-level prediction and delivery contract.

**Blocked by:** 32: [Complete the three-fold ResNet-18 comparison](32-complete-three-fold-resnet18-comparison.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** ready-for-agent

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Execution trigger:** Select only after the simpler representation has a clear diagnostic outcome and a bounded context experiment is justified before the modeling cutoff.

## Acceptance criteria

- [ ] Freeze the hypothesis, same-series adjacent-slice construction, physical-spacing and edge policies before comparison; never combine arbitrary sampled slices or different series as adjacent context.
- [ ] Retain the admitted backbone and matched supervision/training policy where possible; identify input/sampling changes as a new scientific identity.
- [ ] Version preprocessing and feature caches separately from the 2D route, preserving anatomical and missingness semantics.
- [ ] Complete the same three-fold reference cohort and paired OOF evidence, including uncertainty, target support and an accounting of extra decoding/cache work.
- [ ] Exercise actual adjacency, boundary and invalid-series cases through the input-to-study-output interface rather than private helper assertions.
- [ ] Produce reloadable checkpoints and measure complete inference cost; no automatic GPU execution permission or budget extension is implied.
- [ ] If GPU or another optional artifact is required, satisfy its exact readiness/approval gate and record the actual supplying dependency before execution.
- [ ] Record the branch's promotion/rejection decision in the final-freeze register; a sampling change after the cutoff cannot be treated as an operational fix.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.
