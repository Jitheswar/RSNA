# 30: Audit existing report-derived labels

**What to build:** Existing learned and deterministic report outputs receive a reproducible fold-specific admission or rejection decision, with target-level reliability and usable coverage.

**Blocked by:** None (can start immediately).

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

## Acceptance criteria

- [x] Reconstruct source, prompt/extractor, output, adaptation and fold provenance from existing artifacts; reject ambiguous lineage and unverified usage eligibility.
- [x] Freeze a prospective admission policy using eligible training-side evidence only; held-out report evaluation cannot feed back into that fold's prompts, rules or thresholds.
- [x] Report support, confidence intervals where estimable, positive/negative reliability, abstention/failure coverage and newly covered calls separately from shared calls.
- [x] Preserve positive, negative, not-mentioned, uncertain, unknown and extraction-failed states; mask unsupported supervision and retain reference-label precedence.
- [x] Without a clinical reviewer, distinguish observable report/reference disagreement from any unverified explanation of its cause; do not invent diagnostic labels.
- [x] Exercise the existing supervision-admission boundary with leakage, malformed-output and conflicting-reference cases, then produce private immutable admitted artifacts and public-safe aggregate evidence.
- [x] Record which targets and cohorts are admissible, which are rejected and whether the local-extractor branch is warranted. No new remote report processing or image training is included.
- [x] If the local extractor is selected later, apply this same admission contract to its outputs and bind the replacement identity rather than silently rewriting the original decision.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.

## Comments

### 2026-09-25 — Resolved after subset interval review

The fold-specific prospective admission policy, source and usage provenance, private admitted labels, public aggregates, training-side gates, reference precedence, state preservation, and leakage/malformed/conflict boundaries were verified. The isolated follow-up added Wilson 95% confidence intervals for newly covered and shared gold calls without changing the frozen admission decision or private artifact. The revised public JSON SHA-256 is `fc13a7c0080c931da4ffd66961a36f58972c4260bd90ff79878b9e34cbcbca0a`; privacy and count checks passed. The integrated branch passed 377 tests and mypy. The 58-study reference cohort and lack of a clinical reviewer limit claims of reliability and clinical meaning. See `docs/evidence/ticket-30-report-label-audit.json` and `/home/ch.jitheswar/RSNA-tickets-28-29-30-report.md`.
