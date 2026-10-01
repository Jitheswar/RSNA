# Ticket 51 scoring execution and recovery

One exact code-scoring attempt is prepared for `jitheswar/rsna-public-frontier-gpu-preflight-attributed`, saved version **3**, output `submission.csv`. [Issue 51](../../.scratch/rsna-leaderboard/issues/51-score-verified-public-frontier.md) binds the existing source/evidence proposal and the requested **20→36 GPU-hour** campaign extension. The October 1 owner-approved extension is applied and exact submission **56739912** is accepted (script **354194017**); scoring is pending under the persistent observer. No retry or final selection has been performed.

The [submission command](../../scripts/score_public_frontier.py) defaults to a read-only check. It rehashes the reviewed local source/output evidence and checks the authenticated current notebook version, whole source, attachments, pinned container, private setting, internet disabled and T4 machine. It also refreshes free quota, paid scaling, official capacity, duplicate-source/intention evidence and live competition submissions. SDK duration strings lose whole days, so resource admission uses the typed durations' `total_seconds()`.

Current read-only evidence at September 30, 17:35 UTC confirms exact version 3 COMPLETE, no duplicate frontier submission or active scoring request, paid scaling disabled, 58,442.016 free wall seconds and four official submissions available. The campaign still has **2,652 GPU seconds**; the **57,600 GPU-second** scoring reservation does not fit. Those gates were satisfied on October 1 before the exact submission. These transient observations are refreshed by every invocation.

## Before execution

After an actual owner decision, record its text and receipt time in an immutable approval file. The command requires schema `rsna-ticket51-scoring-approval-v1`, `decision=approved`, exact proposal SHA-256 `3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c`, exact resource-proposal SHA-256 `bc56fdbfb597f5dc7d3109780844d2f57e0b1eda9d7e620f85aeffd6da6b809d`, approved total cap `129600`, request `57600`, exact handle/version/output, one attempt, zero spend, paid scaling false and final selection false. It also requires nonempty `owner_decision_text` and a timezone-aware, nonfuture `owner_decision_received_at_utc`.

The approved budget amendment must separately be applied to the authoritative campaign ledger with its decision/source evidence retained. The command checks that the ledger cap matches the exact approved cap; it never changes that cap or writes an approval. The actual October 1 **yes** decision binds both reviewed resource proposals; the subsequent standing autonomy instruction governs later justified campaign decisions.

Run with `/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/bin/python`:

```bash
python scripts/score_public_frontier.py --approval /absolute/path/to/actual-owner-approval.json
python scripts/score_public_frontier.py --approval /absolute/path/to/actual-owner-approval.json --submit
```

The second invocation refreshes all gates again and requires a preflight less than 120 seconds old. It exclusively creates an immutable intent, reserves both GPUs through the existing locked ledger functions, and makes one `competition_submit_code` request specifying saved version 3. It does not upload the visible CSV. The accepted API receipt and exact submission reference are preserved before follow-up observation.

## Observation and recovery

The [observer](../../scripts/monitor_public_frontier_scoring.py) requires the existing accepted response and exact proposal identity. Run it in the same SDK environment:

```bash
python scripts/monitor_public_frontier_scoring.py
python scripts/monitor_public_frontier_scoring.py --watch
```

It gets status and scores for the accepted reference and locates that same reference in the submission listing for its saved-code URL/script-version ID. It rejects mismatched references, missing/duplicate matches and source drift. Watch mode polls every 60 seconds. COMPLETE or ERROR produces immutable terminal evidence and calls the existing accounting helper in the repository environment. Unknown hidden timing charges the full **57,600 GPU seconds**. Repeated terminal observation is idempotent against the same exact evidence. An observation error does not establish termination or justify retry; a cancelled state does not manufacture missing allocation evidence.

If the submission call times out, preserve the intent and reservation and inspect authoritative submissions before any further action. The command refuses another create-code-submission call once an intent exists. If the accepted response was saved but local ledger recording failed, reconcile that exact reference using the saved receipt and the repository environment: `.venv/bin/python scripts/score_public_frontier.py --ledger-action accepted --approval /absolute/path/to/actual-owner-approval.json --submission-ref EXACT_REFERENCE`; then observe it. If no accepted response was received, recover the authoritative reference from the actual service response/listing and bind it to the preserved request before observation. Never infer zero GPU use or restart from a missing local status file.

53 focused tests pass. Verification exercises exact source/version selection, approval/cap/quota/concurrency boundaries, stale checks, ambiguous timeouts, retained accepted receipts, duplicate prevention, source identity rejection and existing conservative terminal accounting. Tests use mocked account calls and temporary ledgers. The real read-only check leaves the campaign ledger unchanged and makes no submission.

## Observer repair verified before scoring

The original observer incorrectly treated `GetSubmission.url` as the saved-code URL. A September 30 read-only audit found that this endpoint returns `/submissions/REF/REF.raw`, while `ListSubmissions` returns `/code/OWNER/NOTEBOOK?scriptVersionId=ID` for the same accepted code submission. Raw responses and parsed SDK objects agree; the endpoint meanings differ. The original 45-test suite did not exercise this successful-response pattern.

A regression test first failed in the actual observer CLI with the captured endpoint pattern. The corrected observer keeps exact status/score/download evidence and listing source evidence separate, with explicit field provenance and identical accepted references. Tests cover independently timed status/listing responses, wrong references, missing and duplicate source matches, source drift and full two-GPU terminal accounting exactly once in a temporary ledger. A fresh authenticated replay through the repaired helper for the existing scored submission **56705536** passes: COMPLETE, score **0.936**, saved script version **354125287**, with the distinct download/code URLs preserved and no campaign ledger change.

Private primary evidence and before/after diagnosis are in `observer-url-diagnosis-v1/` under the Ticket 51 artifact root. `scoring-execution-preparation-v2.json` supersedes the earlier implementation receipt. No frontier scoring request, budget amendment or owner approval was created by this repair.

## October 1 live execution

Actual approval and cap amendment are retained in `.scratch/rsna-leaderboard/resource-approvals/`. Ticket 39 job 7966 was reconciled first (161 GPU seconds). Submission 56739912 accepted exact frontier version 3/source 354194017 after fresh source/settings checks, five official slots, sufficient typed free quota, paid scaling disabled and no competing request. `rsna-frontier-score56739912.service` is active with a 60-second poll. Initial exact state PENDING has no score; the best confirmed public result remains 0.936. Campaign charge 69,509 and active reservation 57,600 of 129,600 GPU seconds preserve all prior conservative charges.

### Continuation after reconciled scoring

The [continuation coordinator](../../scripts/continue_after_frontier_scoring.py) waits for exact submission **56739912**, script **354194017**, COMPLETE/ERROR evidence and the matching **57,600-second** terminal ledger charge. It also waits for the current parent turn to finish before resuming the existing campaign thread. An exclusive lock and immutable launch intent permit one continuation; it makes no competition submission itself. Existing ChatGPT account authentication is required and API-key billing is refused. Its prompt carries the owner's standing authorization, frozen experiment identities, remaining resource boundaries and unresolved scientific/rights checks.

`rsna-frontier-continuation56739912.service` is active. Private arm/prompt/launch receipts are retained under `artifacts/private/campaign-continuation-frontier56739912/`. The mechanism uses the documented [non-interactive session resume](https://learn.chatgpt.com/docs/non-interactive-mode) command. **62 focused tests pass** across scoring, observation, accounting and continuation: wrong references/source, missing or modified evidence, unreconciled reservations, concurrent parent turns and duplicate launch intent are rejected. This is an armed future continuation; no continuation has launched while scoring remains pending.
