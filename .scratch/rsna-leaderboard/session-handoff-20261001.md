# Campaign handoff — October 1, 2026

## Active-session update — Ticket 54 complete

One separate CPU duration experiment completed while exact scoring was pending.
Job **7975**, 16 CPUs/64 GiB, completed `0:0` in **256 allocated node seconds**,
zero GPU/spend. All six 7,500-step heads reproduce their original 1,500-step
prefixes exactly; all 116 final predictions replay exactly with frozen 16-thread
settings. Preserve the failed four-thread probe; the replay tolerance did not
change. [Ticket 54](issues/54-test-longer-convnext-report-head.md) is resolved.

Report OOF development AUC **0.691065** vs **0.659947**, delta **+0.031119**,
interval **[-0.004120,+0.067897]**, one negative fold: **inconclusive/no promotion**.
Report-supervision and image controls are positive, but no complementarity to
D4, competition gain or final candidate is established. All admitted reports
were exposed, with 15,000 draws per fold and matching gold exposure.
Private evidence: `artifacts/private/ticket54-convnext-long-report-head/`.
Plan SHA-256 `1b77479d88d1a3686a14fc292449456c522dd689b579fb07a9ddda5c38abb593`.
[Evidence](../../docs/evidence/ticket-54-long-convnext-report-head.md).

The next GPU action remains the immutable Ticket 53 paired request **v4** after
exact scoring **56739912** is terminal/reconciled and live resource gates pass.
`scripts/analyze_public_meniscus_development.py` is ready to analyze the fixed
pair after independent inference validation; its fold summaries are development
subgroups, with no fold-specific retraining or clean OOF claim. No GPU cap or
reservation changed during these CPU steps. Leave unrelated concurrent
`mise.toml` edits intact.

## Active-session update, 04:35 UTC

Actual private CPU MRI preflight completed and was independently verified:
58 real studies, 336 header series, 273 selected slot-series, 72.203 seconds,
zero decode failures/fallbacks and zero GPUs. The 471,453,696-byte uint8 native
cache passes the specialist contract and remains identical after unlink.
Evidence: `artifacts/private/public-meniscus-adoption/cpu-mri-preflight-v2/`.

The current immutable paired development request is **v4**, checksum
`0e490a080a5c51c1aeaaab4e67da5673129d47ebec22a48173eecf0e3fa78ed6`.
It binds the successful CPU pixel/mask hashes, while retaining the same model,
23 anchor cells and single 10% vote. Draft v3 was superseded before launch.
No GPU parity, predictive gain, reservation or cap amendment is claimed.
Exact accepted frontier **56739912** remains PENDING at **04:33:19 UTC**;
finish its terminal reconciliation before launching the next GPU request.
[Evidence](../../docs/evidence/ticket-53-public-meniscus.md).

## Active-session update, 03:53 UTC

The primary campaign session has taken over continued work. Exact scoring
**56739912** remains **PENDING**, independently observed at **03:53:23 UTC**.
The exact-scoring observer remains active. The waiting original-thread
continuation coordinator was stopped, with a private takeover receipt retained,
to prevent overlapping agents from changing campaign state. No scoring request
was cancelled, restarted or retried; the goal remains active.

[Ticket 53](issues/53-admit-public-meniscus-specialist.md) completed CPU admission
of the public0033 specialist and froze one paired Gold58 inference request at
`artifacts/private/public-meniscus-adoption/paired-development-v3/`, checksum
`1f3473712bb55df8d16de94cbe2dbc3b62f2b3f81345df130dd7d87df2ae1519`.
Twelve cache/overlay/random-state tests pass. Preparation v1 was superseded before
any launch after an actual CSV precision regression; v2 was superseded to restore
all anchor random streams after specialist inference. No vote variants were
evaluated or selected. No predictive gain is measured yet.

The next GPU request proposes two free T4s, 2,700-second session limit,
2,400-second watchdog, 5,400 GPU seconds, one attempt and no training/scoring.
It proposes a 36→38-hour campaign cap under standing authorization; no cap or
reservation changed. Finish/reconcile the current accepted scoring reference
before applying the justified resource decision and launching the exact pair.
Reused Gold58 and upstream reference-set selection remain development limits.
The campaign objective includes both public and final private first place;
neither is established. [Evidence](../../docs/evidence/ticket-53-public-meniscus.md).

The owner is closing the session and requested committing all uncommitted repository changes. Closing the session does not cancel the standing campaign authorization or the accepted scoring request.

## Standing authorization

The goal is first place on the **final private** RSNA Knee Abnormality Detection leaderboard, with **$0 external spend**. The owner approved the exact Ticket 39 replay and Ticket 51 cap-extension proposals, then instructed: **“approve anything as long as we win 1st place and fits kaggle comp dont ask me”**. Handle justified campaign decisions autonomously within Kaggle rules; preserve concrete scientific plans, resource decisions and actual evidence. Do not fabricate a numerical owner approval or claim first place from a public score. See [authorization](resource-approvals/campaign-owner-autonomy-20261001.json) and [current spec](spec.md).

## Current execution

- **56739912:** accepted exact frontier notebook version **3**, script **354194017**; latest observation at **03:20:30 UTC / 08:50:30 IST** is **PENDING**, with empty public/private scores. Inspect the actual latest evidence before acting; this timestamp is a handoff snapshot.
- **56705536:** scored provisional baseline **COMPLETE / 0.936**, script **354125287**, notebook version **2**.
- **56515942:** historical scored fallback **COMPLETE / 0.498**, script **352323130**, notebook version **1**. Keep both fallbacks and their artifacts.
- Campaign ledger: cap **129,600 GPU seconds**, charged **69,509**, frontier reservation **57,600**, unreserved remainder **2,491**. Earlier hidden allocation timing is unknown and its conservative charge must remain. The new scoring run also charges its entire reservation if hidden timing is unavailable.

Scoring artifacts are private and retained at `artifacts/private/public-frontier-adoption/scored-baseline-v1/`. The immutable submit intent and accepted response already exist. **Do not issue a second create-code-submission call for this one-attempt proposal.** Monitor only the accepted reference, check exact saved-source identity, and reconcile terminal evidence through the existing locked ledger helper. An observation failure does not establish termination or authorize reclaiming its reservation.

## Background services and recovery

Both services were independently verified **active/running** at handoff:

```bash
systemctl --user status rsna-frontier-score56739912.service
systemctl --user status rsna-frontier-continuation56739912.service
```

The scoring observer polls every 60 seconds, preserves exact-reference status and separate matching source evidence, and reconciles COMPLETE/ERROR conservatively. The continuation waits for matching terminal accounting and for the parent turn to finish, then resumes the existing thread **01a0f30c-4ad6-7830-836e-352e2c70b261** once. It refuses API-key billing and uses existing ChatGPT CLI authentication. Its exclusive launch intent prevents duplicate continuation. Arm, prompt, lock and eventual launch/result receipts are private at `artifacts/private/campaign-continuation-frontier56739912/`.

These are transient user services on the server. `loginctl` currently reports **State=active, Linger=no**. Closing the client session alone need not stop them, but a server reboot or termination of the last server login can stop the user manager. No logout/reboot persistence is claimed. On return, inspect the exact services, accepted scoring state and launch receipts before restarting anything. The observer can be run from the existing Kaggle SDK environment; restarting observation never creates a submission. [Execution and recovery instructions](../../docs/evidence/ticket-51-scoring-execution.md).

## Scientific disposition and next decision

Ticket 39 analysis job **7966** completed `0:0`; its full **161-second** GPU allocation is reconciled. All 58 OOF outputs, six candidate/baseline checkpoint identities and both pooled metrics passed independent checks; GPU replay is exact for all three folds. Fine-tuned AUC **0.518355** versus short frozen **0.464313**, but image-signal controls remain inconclusive. Disposition is **inconclusive/no promotion**. Full hidden-scale delivery cost is unverified; do not rerun its frozen experiment or admit a final delivery candidate from this result. [Evidence](../../docs/evidence/ticket-39-fine-tuning.md).

Ticket 52's longer DINO head comparison is complete and **inconclusive/no promotion** against the matched longer ResNet control. Ticket 40 remains unselected. Remaining candidate review, component rights, delivery checks and final selections are substantive decisions; the owner authorization permits handling them without another permission question.

After scoring completes, record the actual result and disposition in Ticket 51 and the final register before choosing another experiment. Fresh [primary-source research](../../docs/research/competitive-next-route-2026-10-01.md) establishes no verified accessible public recipe above author score **0.943**. Roman's specialist-containing v24 also shows **0.943**; higher-scoring PJ/Aastik routes require inaccessible checkpoints. Do not spend another full hidden reservation merely to copy a recipe without evidence of useful complementarity. Register and justify any next experiment/resource amendment before execution.

## Attribution, schedule and verification

The latest two submissions are attributed reproductions of public solutions and publicly released weights. The frontier retains **23 upstream code cells unchanged**; the model choices and blend weights are upstream contributions. Our changes cover execution compatibility, validation, watchdogs and packaging. Credits, Apache license text and modification notices are retained. Component-specific model/data terms and a winner handoff remain separate checks. Do not describe this reproduction as our original modeling innovation or suggest copying alone establishes a route to first place.

No reliable scoring ETA is exposed. The earlier result was confirmed about 4 hours 40 minutes after submission; the latest chat estimate of roughly 11:10 a.m. IST is conditional on similar timing, not a deadline or proof of allocation.

Internal delivery remains before **October 15**, with scientific freeze **October 11**. Official final submission deadline is **October 22, 23:59 UTC**; five submissions/day, two final selections, nine-hour notebook runtime, internet disabled and `submission.csv`. One pending/active GPU request across backends; CPU ceilings 16 CPUs, 64 GB, six hours, two concurrent jobs; storage soft ceiling 2 TB. Paid scaling remains disabled.

Verification: **62 focused scoring/observer/accounting/continuation tests passed**, plus **14 source-execution/validation tests passed** before commit. All 12 changed Python files parsed; `git diff --check` passed. Private patient data, caches, weights, notebook execution outputs and credentials remain outside Git under the repository's existing ignore rules. No final private rank is available and the campaign goal is not complete.
