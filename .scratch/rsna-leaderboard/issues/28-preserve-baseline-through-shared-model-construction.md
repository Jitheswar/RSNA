# 28: Preserve the baseline through shared model construction

**What to build:** The existing compact model completes training, checkpoint reload, candidate freezing and offline study prediction through shared versioned model construction, with its predictions and historical candidates preserved.

**Blocked by:** None (can start immediately).

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Use additive versioned model configuration and shared reconstruction across training, checkpoints, candidate loading and offline inference; existing compact-model artifacts remain readable.
- [x] Demonstrate the existing study-to-training-to-OOF-to-frozen-candidate-to-submission lifecycle with a small fixture, preserving study identity, target order and expected probabilities.
- [x] Compare the same frozen baseline before and after the prefactor; preserve its original artifacts and require unchanged predictions for this behavior-preserving change.
- [x] Reject unknown model identities, incompatible configuration and altered checkpoints at the public artifact boundary, leaving prior complete artifacts intact.
- [x] Keep the public CPU training and separately authorized GPU paths intact; do not add a shortcut around resource approval.
- [x] Provide an independently usable prefactor and its focused lifecycle evidence; do not introduce external weights or a new experiment platform.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

### 2026-09-25 — Resolved after integration review

Shared versioned model construction, historical checkpoint/candidate compatibility, the complete synthetic lifecycle, artifact-boundary rejection, and the CPU/GPU authorization seam were verified. The same frozen candidate produced byte-identical before/after 396-byte CSVs, SHA-256 `181d9e23e972b352b2557e08ad8b0b9640b051e7b8e0dc6532a799551c8759e5`. Both outputs and a checksum manifest are committed under `docs/evidence/ticket-28-historical-parity/`. The integrated branch passed 377 tests and mypy. Historical parity proves behavior preservation on the synthetic input, not real-image discrimination. See `/home/ch.jitheswar/RSNA-tickets-28-29-30-report.md` and main commit `7a742799`.
