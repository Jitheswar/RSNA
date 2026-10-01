# 41: Maintain the deployable fallback and freeze the final candidate after branch decisions

**What to build:** A verified fallback stays deployable while experiments proceed, and the final candidate is frozen only after every selected or triggered experiment has a recorded promotion or rejection decision.

**Blocked by:** 28: [Preserve the baseline through shared model construction](28-preserve-baseline-through-shared-model-construction.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** claimed

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Final-freeze blockers:** Every selected or triggered experiment branch and every scientific rerun registered below, including the selected ResNet route. These are additional phase-specific blockers, not prerequisites for parallel fallback maintenance. No unresolved selected branch may be bypassed by pointing at the fallback.

**Initial branch register:** The ResNet route through tickets 31 and 32 is selected. The GPU, report-supervision/expansion and optional extractor, alternative-representation, fine-tuning and context branches are activated when their stated conditions and resource decisions select them. Register each activation by ticket number, title, exact run/cohort/configuration identity and evidence pointer before it executes; downstream label, feature or ensemble changes are separate registered scientific variants. The diagnostic and label-admission outcomes supplying a selected branch must also be resolved.

## Acceptance criteria

- [x] Phase A: preserve the existing scored fallback as an immutable candidate, verify its exact artifact/runtime identity and complete offline submission contract, and keep it usable while experimental branches remain open.
- [x] Maintain separate states for fallback readiness and final-freeze readiness. Phase A completion does not resolve this whole ticket or unblock final delivery.
- [ ] Track every selected/triggered branch and scientific variant with selection time, status, supplying dependencies, evidence, decision owner and explicit promotion/rejection rationale. A previously activated branch cannot disappear or be relabeled never-selected to clear the gate.
- [ ] Before phase B, require all registered selected/triggered branches to have terminal promotion/rejection decisions. Ticket closure, a provisional promising result, a running experiment or pending OOF analysis is not a substitute for its decision.
- [ ] At the October 11 modeling cutoff, incomplete or unavailable work requires an explicit rejection from this delivery with a not-evaluated/incomplete reason; do not fabricate a scientific failure or promote partial evidence. If any branch still lacks a decision, final freeze remains blocked.
- [ ] Select from eligible complete evidence using the agreed local-validation policy. A simple equal-weight ensemble is itself a selected branch and must have a completed paired OOF/runtime promotion or rejection decision before final freeze.
- [ ] Freeze the final scientific configuration and exact checkpoint, preprocessing, model, dependency and evidence identities only after the gate passes. Record an immutable snapshot of the complete branch decisions alongside the candidate.
- [ ] Check complete MRI-only offline output, expected study coverage, target order, finite probabilities and explicit decode/fallback accounting. Measure discovery through CSV cost and preserve representative-versus-hidden evidence distinctions.
- [ ] Optimize only operational work with prediction-equivalence evidence after freeze. A new scientific branch or changed model/sampling reopens the final-freeze gate and requires reevaluation and a new candidate identity; late branches cannot silently enter delivery.
- [ ] Verify gate behavior with selected-running, undecided, promoted, rejected, never-selected and late-added branch cases: open selected branches block final freeze but do not disable the verified fallback.
- [ ] Record phase B completion separately; resolve this ticket only when final freeze and its verification are complete. No project candidate/submission-count cap is added and no extra compute is granted.

## Completion evidence

Record the demonstration, focused boundary checks, exact artifact identities, measured scope, unresolved limitations and any required branch disposition. A ready-for-agent label authorizes preparation within the spec; it does not establish resource approval or successful scientific performance.


## Comments

### September 30 — phase A integrity/readiness complete; phase B open

Fresh authenticated read-only observations confirm the 0.498 historical fallback
and 0.936 public baseline remain COMPLETE at their exact saved source versions.
Current remote source code still matches both local snapshots. `load_candidate`
verified all 40 historical candidate artifacts; the exact 75-file delivery
payload digest and offline wheel/runtime pins passed a current integrity scan.
The retained CSV's schema, bounds and archived input-order/hash proof remain
valid. No inference, upload, scoring, cap change, final freeze or selection was
performed by this audit. Exact evidence is
`artifacts/private/ticket41-readiness-audit-v1/audit.json` and
`historical-output-contract-check.json`.

The final-freeze register now explicitly consolidates selected public baseline,
frontier/scoring and Ticket 39 branches. Ticket 39's actual preserved state has
all three folds complete but final analysis unverified, rather than its older
queued-job comments. Its resource-scope replay amendment remains pending;
Ticket 40 remains unselected under the owner's explicit instruction. Prior
inconclusive comparisons are preserved without inventing final dispositions.
Phase B and its boundary tests remain outstanding; this ticket is not resolved.

2026-09-30: The newly registered private CPU duration branch, Ticket 52, is now terminal: primary advancement gate failed, inconclusive/no promotion, all saved outputs independently replayed. Its closure is recorded in the final-freeze register. Outstanding Ticket 39 analysis/resource scope and Ticket 51 scoring/budget decisions still prevent Phase B; no final selection occurred.

### October 1 readiness update

Ticket 39 now has a verified complete analysis and terminal **inconclusive/no-promotion** disposition; its hidden-scale delivery gate remains unverified and no candidate is admitted. The owner approved Ticket 51 and standing autonomous campaign decisions. Exact frontier submission 56739912/source 354194017 is accepted and initially PENDING under a persistent observer. Its result, remaining candidate review and delivery/rights checks still precede phase B. The 0.936 and 0.498 scored fallbacks remain preserved. No first-place or final-freeze claim.
