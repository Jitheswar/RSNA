# RSNA campaign session handoff — October 1, 2026

Read this first when resuming in a new session. The owner requested this handoff
and a GitHub commit before switching sessions. No experiment was launched during
handoff preparation. The goal is first place on **both the public and final
private leaderboards**; neither has been achieved. Best confirmed account score:
**0.936**. Leader in the independent API check at **05:13 UTC**: **rhoskeri / 0.961**.
Refresh live observations before acting.

## Resumed continuation — October 1, 06:21 UTC

Ticket 55 is now **resolved, inconclusive/no promotion**. Training recovery
**7978** completed all six 7,500-step runs, all original prefixes match exactly,
and independent verifier **7981** reproduced **116** final predictions and all
seven paired comparisons. Report ResNet AUC **0.686730**; fixed half blend
**0.705751**, improvement interval **[-0.009373,+0.050577]**, one negative fold.
New CPU allocation **356 node seconds**, including verifier path failure **7980**;
zero GPU/spend. Preserve both original failures and completed recovery evidence.
Root: `artifacts/private/ticket55-resnet-long-report-head-v2/`.
[Completed evidence](../../docs/evidence/ticket-55-resnet-recovery.md).

Exact frontier **56739912 / 354194017** remains **PENDING at 06:18:35 UTC**.
Fresh authenticated leaderboard observation at **06:12:27 UTC** retains leader
**rhoskeri / 0.961**. Best confirmed account score remains **0.936**. No CPU job
is currently live; the exact-scoring observer is running and the old continuation
coordinator stays stopped. Ticket 53 v4 still awaits terminal scoring reconciliation.
These observations supersede the operational snapshots below; refresh before use.

## Resume in this order

1. Read `AGENTS.md`, `CONTEXT.md`, this handoff and the relevant ticket below.
   Inspect Git, live jobs/services and campaign-controller ownership. Keep one
   controller; the old continuation coordinator is deliberately stopped.
2. Refresh accepted submission **56739912**, exact saved source **354194017 /
   notebook v3**. If terminal, verify source identity and reconcile through the
   existing observer/accounting helpers before another GPU request. If pending,
   retain its reservation and use admitted CPU work.
3. [Ticket 55](issues/55-test-matched-long-resnet-report-head.md) is now complete.
   Preserve its six final runs and independently verified inconclusive disposition.
   Do not repeat its consumed training or validation requests. Further CPU work
   needs a new prospective hypothesis and resource decision; the fixed blend does
   not pass its declared improvement gate.
4. Next GPU work: [Ticket 53](issues/53-admit-public-meniscus-specialist.md), immutable
   paired request **v4**. Launch after scoring is terminal/reconciled, all GPU
   backends are clear, and fresh quota/scaling/resource-decision gates pass.
5. Independently validate actual results, update the ticket, [map](map.md) and
   [freeze register](final-freeze-register.md), then check, commit and push.
   Close the campaign goal only when its leaderboard objective is actually met.

## Standing authorization and resources

Owner instruction: **“approve anything as long as we win 1st place and fits kaggle
comp dont ask me”**. Proceed with justified, Kaggle-compliant decisions and **$0
external spend**. [Standing authorization](resource-approvals/campaign-owner-autonomy-20261001.json)
SHA-256: `634562f8d5d89016e625ecb6befb57ed410b281a1f70762f3b4391d8ed7103ca`.
Record concrete agent decisions; distinguish them from exact numerical owner approvals.

- GPU ledger: `.scratch/rsna-leaderboard/gpu-campaign-ledger.json`; cap **129,600 s
  / 36 h**, charged **69,509 s**, outstanding **57,600 s**, unreserved **2,491 s**.
  Preserve unknown-allocation charges and pending reservations.
- One pending/active GPU request across backends. At most two nonexclusive CPU
  jobs, each one node, at most 16 CPUs / 64 GB / 6 h. Use Slurm for substantial
  cluster work. Ticket 55 is narrower: 16 CPUs / 64 GiB / 30 min.
- Paid scaling stays off; storage soft ceiling **2 TB**. Science freeze **October
  11**, internal delivery before **October 15**. Recorded official final deadline:
  **October 22, 23:59 UTC**; five submissions/day, two final selections, nine-hour
  notebook limit, internet off, `submission.csv` delivery.
- Standing GitHub authorization: check, commit and push completed work to private
  **https://github.com/Jitheswar/RSNA**, branch `master`, configured `origin`; verify
  remote/local equality and preserve unrelated work/history.

## Accepted scoring and background services

| Submission | Saved script/version | Last verified state |
| --- | --- | --- |
| **56739912** | **354194017 / v3** | **PENDING**, blank scores/errors; observer **2026-10-01 05:56:06 UTC** |
| 56705536 | 354125287 / v2 | COMPLETE, public **0.936** |
| 56515942 | 352323130 / v1 | COMPLETE, public **0.498** |

56739912 was accepted **00:56:50 UTC** for
`jitheswar/rsna-public-frontier-gpu-preflight-attributed`. Proposal SHA-256:
`3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c`.
Its single submission attempt is consumed. **Monitor this reference; do not
submit it again.** Preserve both fallbacks. No final selections have been made.

Private root: `artifacts/private/public-frontier-adoption/scored-baseline-v1/`.
Read `scoring-status.json`, `scoring-observations.jsonl`,
`accepted-scoring-source.json`, immutable `scoring-submit-intent.json` and
`scoring-submit-response.json`. Terminal observation writes
`terminal-scoring-evidence.json` and charges the full **57,600 GPU seconds** if
hidden timing is unknown. A pending scoring request may be absent from notebook
quota reservations; that does not clear campaign concurrency.

`rsna-frontier-score56739912.service` is **active/running**, PID **2617889** at
handoff; it polls every 60 seconds. The old
`rsna-frontier-continuation56739912.service` is **inactive/dead**, deliberately
stopped to prevent competing agents. Keep it stopped. Takeover receipt:
`artifacts/private/campaign-continuation-frontier56739912/active-session-takeover-20261001.json`.
Prior campaign goal thread: `01a0f57e-d652-7443-a987-909c8e6956a5`; inspect goal
and controller ownership when transferring work to a new thread. No CPU Slurm job
is queued/running at handoff. User services are transient; inspect actual state
after reconnecting. Observation recovery command:

```bash
/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/bin/python scripts/monitor_public_frontier_scoring.py
```

This command can reconcile terminal accounting. [Execution/recovery details](../../docs/evidence/ticket-51-scoring-execution.md).
Repository `.venv/bin/python`: Python 3.11.16, torch 2.7.1+cpu, NumPy 2.2.6,
pydicom 3.0.1, pandas 2.2.3, pytest 8.3.5. Kaggle authentication already exists
in the separate SDK environment; keep credentials out of outputs.

## Work completed and actual disposition

- Earlier tickets established DICOM preparation, three development folds,
  report-label auditing, feature extraction, matched supervision experiments,
  Kaggle delivery, attributed public ensemble reproduction and exact scoring
  safeguards. Read the [ticket index](issues/) and [freeze register](final-freeze-register.md)
  for full history; retained summaries are under `docs/evidence/`.
- [Ticket 39](../../docs/evidence/ticket-39-fine-tuning.md): final analysis and exact
  three-fold GPU replay complete; job **7966**, **161 GPU seconds** reconciled.
  Fine-tuned development AUC **0.518355** versus short frozen **0.464313**;
  image controls inconclusive, **no promotion**. Its fixed experiment is complete.
- [Ticket 52](../../docs/evidence/ticket-52-dinov2-duration.md): longer DINO head
  comparison complete, **inconclusive/no promotion** against matched ResNet.
  Ticket 40 remains unselected.
- [Ticket 53](../../docs/evidence/ticket-53-public-meniscus.md): actual CPU bundle
  admission verified eight payload files and strictly loaded 233 FP32 tensors in
  **42.493 s**. Real MRI preprocessing passed **58 studies / 336 series / 273
  selected slot-series** in **72.203 s**, no decode failures/fallbacks, zero GPU/spend.
  `(58,6,12,336,336)` uint8 cache: **471,453,696 bytes**, identical after unlink.
  Predictive benefit remains unmeasured.
- [Ticket 54](../../docs/evidence/ticket-54-long-convnext-report-head.md): job **7975**
  completed six 7,500-step ConvNeXt heads, **256 allocated CPU-node seconds**,
  zero GPU/spend. All original prefixes and 116 final predictions reproduce exactly
  at frozen **16-thread** settings. Report AUC **0.691065** versus short **0.659947**,
  delta **+0.031119**, paired interval **[-0.004120,+0.067897]**, folds **+ / − / +**:
  **inconclusive/no promotion**. Long reference AUC **0.601455**; report-minus-reference
  and image controls positive. Preserve the failed four-thread replay probe;
  tolerance stayed **1e-6**. Private root: `artifacts/private/ticket54-convnext-long-report-head/`.
- [Ticket 55](../../docs/evidence/ticket-55-resnet-recovery.md): original job **7977**
  failed on absent-report provenance, **32 CPU-node seconds**. Diagnosed recovery
  **7978** completed, **290 seconds**; independent verifier **7981** completed,
  **60 seconds**, after path-only failure **7980**, **6 seconds**. Every original
  prefix and all 116 final predictions replay exactly. Report ResNet **0.686730**;
  fixed half blend **0.705751**, paired interval crosses zero and one fold declines.
  The branch is **inconclusive/no promotion**. No GPU/spend or scoring action.

## Prepared GPU comparison: Ticket 53 v4

Root: `artifacts/private/public-meniscus-adoption/paired-development-v4/`.
Request SHA-256: `0e490a080a5c51c1aeaaab4e67da5673129d47ebec22a48173eecf0e3fa78ed6`.
Notebook SHA-256: `df6b288b6b6b8abcc90c0ad5287c42554727e8344a0b9cf5e4d0face2d4a7686`.
**Prepared, unlaunched, unreserved**; cap amendment unapplied. Older drafts were
superseded before launch. V4 binds CPU cache/mask hashes and preserves anchor RNG.

Keep all **23 upstream anchor cells** and attribution/license notices. Exactly one
fixed Medial Meniscus rank vote: **10% Transformer + 80% CoAt/Raptor + 10%
specialist**; eleven other targets unchanged. Specialist:
`renta0426/rsna-knee-public0033-meniscus-bag-v1/2`; checkpoint SHA-256:
`e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7`.
Twelve internal heads do not establish useful supervision for every target;
inspect the publisher's training configuration before widening its use.

Prepared resources: two free T4s, **2,700 s** session, **2,400 s** watchdog,
**5,400 GPU seconds**, one attempt, no fitting/scoring. Proposed cap: **36→38 h /
136,800 s** under standing authorization. After terminal scoring reconciliation,
register/apply the concrete agent decision, refresh gates, then reserve/launch once.
No further owner permission is needed. Analyze with
`scripts/analyze_public_meniscus_development.py` after independent matching-request
validation. Gold58 fold summaries are development subgroups, without fold-specific
retraining or a clean OOF claim for the full-fit public anchor.

## Interpretation and unfinished delivery

Small-cohort head AUCs are not directly comparable to public D4 scores. Reused
reference studies, public recipe selection and missing attested patient keys limit
conclusions. Measure same-cohort complementarity before full hidden scoring or
promotion. [Competitive-route research](../../docs/research/competitive-next-route-2026-10-01.md)
and [accessibility audit](../../docs/research/public-frontier-accessibility-update-2026-09-30.md)
found a verified accessible author recipe at **0.943**; higher PJ/Aastik scores
require inaccessible checkpoints. Roman's specialist v24 was also **0.943** and
changed another route weight, so it is not an isolated specialist gain.

Candidate selection, component rights, hidden-scale cost, full delivery checks
and two final selections remain unfinished; consult Tickets 41–43 in the
[freeze register](final-freeze-register.md). Final private rank is unavailable until
competition end. Neither code preparation nor a provisional AUC completes the goal.

## Git and handoff verification

Earlier work was committed/pushed in **572f8fb**. A separate workflow then committed
**642c787** (static typing/runtime boundaries); preserve it. This handoff commit
adds Ticket 55's source, original CPU decision, regression tests and failure/recovery
evidence, and refreshes campaign pointers. Private MRI/reports/checkpoints/caches,
execution outputs and credentials stay outside Git and remain on this server.

Handoff checks: **21 focused tests passed** across provenance, specialist overlay
and frozen-head helpers; both new Python files parsed. See the finishing message
or `git log -1` for the handoff commit and verified push result.
