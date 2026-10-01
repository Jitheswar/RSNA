# 51: Score the verified public frontier recipe

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 47

## Exact prepared action

Issue one code-competition submission of
`jitheswar/rsna-public-frontier-gpu-preflight-attributed` **version 3**, output
`submission.csv`, against `rsna-knee-abnormality-detection`. The saved version
completed; exact remote source, all 30 prediction artifacts, all four CoAt
receipts and zero reported fallback passed independent validation. Its visible
CSV is byte-identical to the corrected preflight. This evaluates an unchanged
public recipe against our **0.936** submission 56705536. The author's **0.943**
is not our measured score.

Artifacts: `artifacts/private/public-frontier-adoption/scored-baseline-v1/`.
The immutable `competition-submission-proposal.json` checksum is
`3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c`.
`scoring-resource-proposal.json` binds the budget decision; source, validation
and accounting hashes identify the exact reviewed saved version.

## Required owner decision

The agreed campaign has a **20 GPU-hour / 72,000-second cap**. It has charged
**69,348 seconds**, no active/reserved request, and **2,652 seconds left**.
The specification requires a new concrete budget decision before expansion.
Earlier hidden allocation remains unknown; the read-only telemetry audit
does not permit reducing its conservative charge.

Proposed new total cap: **36 GPU hours / 129,600 seconds**, adding **16 GPU
hours / 57,600 seconds**, exclusively for one exact hidden scoring attempt.
It uses two free T4s, 28,800 wall seconds reserved, a 28,000-second execution
watchdog, zero external spend, paid scaling disabled, one concurrent GPU
request and no training. Current charges plus the reservation would be
**126,948 seconds**, below the proposed cap. No automatic extension or retry.
This is a campaign resource decision; final candidate selection is separate.

## Requirements

- [x] Verify full-workload saved version 3 terminal source/output and reconcile
  its measured visible allocation before the budget decision.
- [x] Preserve exact scoring/resource proposals and prior scored fallback.
- [x] Prepare and test a guarded exact-source submission command, accepted-reference observer and conservative accounting/recovery path.
- [x] Record the owner's explicit budget-extension decision against this exact
  proposal; preserve the 20-hour cap until that decision is received.
- [x] Refresh free quota, paid scaling, official submission capacity and exact
  saved-version identity; admit one 57,600-second reservation after all pass.
- [x] Submit the exact verified saved version through the code API; preserve
  accepted reference and script-version identity. Never submit the visible CSV.
- [ ] Poll only the accepted reference; retain terminal score/error evidence
  and account for hidden exposure conservatively when allocation is unknown.
- [ ] Record the measured comparison and disposition without equating public
  score to private rank or bypassing final-candidate selection gates.

## Comments

September 30: preparation is complete. No scoring request or cap change has
been issued. Ticket 47 is resolved; the remaining prerequisite is the separate
resource decision above. First private place is not achieved or verified.

### September 30 — guarded execution prepared; approval remains pending

`scripts/score_public_frontier.py` now performs a read-only authenticated preflight by default. Submission mode requires the exact new owner-approval artifact, corresponding approved ledger cap, fresh source/attachment/container checks, enough free quota, paid scaling disabled, official capacity, no duplicate intent/source and no live scoring request. It preserves intent before one exact saved-version code API call and retains the GPU reservation on an ambiguous response. `scripts/monitor_public_frontier_scoring.py` observes only the accepted reference and reconciles terminal COMPLETE/ERROR with the existing full-reservation accounting policy. No automatic retry or final selection is added.

**45 focused tests passed**, including quota day retention, owner/cap/resource gates, stale preflight, exact version/output selection, timeouts, saved response recovery, duplicate prevention, source identity and terminal accounting. A fresh real read-only check at **17:35 UTC** confirmed exact version 3 COMPLETE, four submissions available, sufficient free quota and paid scaling off. The sole remaining condition is the pending resource decision: existing 20-hour cap, 2,652 GPU seconds left, one scoring reservation requires 57,600 seconds. Actual ledger checksum is identical before and after the check. No approval, cap amendment, scoring intent, competition submission or new GPU allocation was created.

[Execution/recovery instructions](../../../docs/evidence/ticket-51-scoring-execution.md); exact private preparation receipt: `artifacts/private/public-frontier-adoption/scored-baseline-v1/scoring-execution-preparation-v1.json`. This operational preparation does not establish a frontier score or first place.

### September 30 — observer endpoint mismatch repaired before execution

A fresh authenticated completion audit exposed an operational defect in the prepared observer: `GetSubmission.url` is the raw CSV download URL, while the exact matching `ListSubmissions` row carries the saved-code URL. Primary raw responses prove the same reference and no SDK rewrite. An integration test reproduced the original rejection before the fix. The observer now preserves exact status/download and listing source separately, with matching references and explicit field provenance; terminal admission checks both records.

**53 focused tests pass**, including the actual CLI success path, asynchronous endpoint status responses, missing/duplicate/wrong references, source drift and full-reservation accounting exactly once in a temporary ledger. Live primary replay after the fix verifies existing submission 56705536 COMPLETE at 0.936 and script version 354125287; actual campaign ledger remains unchanged. Private diagnosis `observer-url-diagnosis-v1/` and the new `scoring-execution-preparation-v2.json` supersede the earlier implementation receipt without overwriting it. The budget approval is still pending; no frontier scoring attempt was made.

### October 1 — approved and submitted

The owner answered **yes** to this exact 20→36 GPU-hour proposal and the separate Ticket 39 replay, then authorized Kaggle-compliant campaign decisions without further questions. Immutable [authorization](../resource-approvals/campaign-owner-autonomy-20261001.json), [exact approval](../resource-approvals/ticket51-frontier-scoring-20261001.json) and [cap amendment](../resource-approvals/cap-amendment-20261001.json) retain the decisions. Job 7966 completed and its 161 GPU seconds were reconciled first.

Fresh preflight at **00:56:35 UTC** verified exact source/version 3, unchanged attachments/container, internet disabled, five official slots, free quota 58,442.016 wall seconds, paid scaling false and no competing request. One code API call accepted **submission 56739912**, saved script **354194017**. No visible CSV upload, retry or final selection occurred. Current campaign charge **69,509 seconds**, active scoring reservation **57,600**, cap **129,600**, unreserved remainder **2,491**. Unknown hidden timing will retain the full conservative charge.

Exact-reference observer `rsna-frontier-score56739912.service` is active. First observation is **PENDING**, empty public/private scores, with independently matched source/download records. Accepted response, intent, source identity and subsequent observations are retained under the existing private artifact root. Best confirmed score remains 0.936; no new score or first-place evidence is claimed.

October 1 continuation: `rsna-frontier-continuation56739912.service` is active alongside the exact-score observer. It resumes the existing campaign thread once, after matching terminal evidence/full reservation accounting and the parent turn completes. Existing ChatGPT authentication only; no API-key billing. Wrong references/source, unreconciled charges, concurrent turns and duplicate intent are covered by the 62 passing focused tests. No continuation has launched while scoring remains PENDING. [Mechanism and recovery](../../../docs/evidence/ticket-51-scoring-execution.md).
