# RSNA campaign session handoff — October 1, 2026

Authoritative update after completed work at **08:40 UTC**. Goal: first place
on **both public and final private leaderboards**; neither is established.
Best confirmed account result **0.936**, submission **56705536**, script
**354125287 / v2**. Previous **0.498** fallback **56515942** is preserved.
Last authenticated public leader at **08:00:31 UTC**: **rhoskeri / 0.961**.
No final selection has been performed. Refresh live observations before acting.

## Resume order

1. Read `AGENTS.md`, `CONTEXT.md`, this handoff and relevant tickets. Inspect
   Git, Slurm and campaign services. Keep one controller; the old continuation
   coordinator stays deliberately stopped.
2. Preserve exact terminal frontier failure **56739912 / 354194017 / v3**:
   generic hidden-rerun error, no public/private score. Latest saved notebook
   logs contain only its successful three-study visible run. Its unknown
   **57,600 GPU seconds** are fully charged. Do not repeat the consumed scoring
   request or infer a root cause from status `COMPLETE`.
3. Continue [Ticket 59](issues/59-diagnose-hidden-frontier-execution-failure.md).
   Phase 1 still lacks a red-capable hidden-failure loop. The unchanged-anchor
   Ticket 53 Gold58 run passed, so it is not a reproduced hidden error. A
   bounded broader real-MRI CPU input probe is the next useful diagnostic
   preparation; freeze a concrete request/resource decision before execution.
   Do not rank causes, relax guards or claim a repair before reproducing the
   actual symptom. A recovered runtime needs a distinct validation/scoring plan.
4. Preserve completed CPU experiments 55–58 and their no-promotion results.
   Do not repeat consumed requests or select a new budget/checkpoint from their
   outcomes. Any new scientific comparison must have a separate prospective
   hypothesis, source identities, exclusions and concrete resource decision.
5. Independently verify new results, update ticket/map/freeze register/handoff,
   run relevant checks, commit and push private `origin/master`. Close the
   campaign goal only when both leaderboard ranks are actually verified.

## Completed continuation

| Ticket | Exact jobs / request | Result |
| --- | --- | --- |
| 51 | Submission 56739912, script 354194017, v3 | Terminal hidden-rerun error, no score; full 57,600 GPU seconds charged |
| 53 | Private paired notebook v1; analysis 7990 | Reject fixed specialist overlay: macro AUC 0.951479→0.951179, interval crosses zero; no promotion |
| 57 | Extract 7983 / 7984; verify 7985 | All 4,228 studies / 187,272 slices verified, frozen feature error 0.0; 2,558 CPU-node seconds |
| 58 | Cost 7986; fit 7989; native verify 7991 | Inconclusive adaptation vs matched head: 0.689232 vs 0.691387, interval crosses zero, one negative fold; 2,483 CPU-node seconds |

Ticket 53 independently verifies all 27 finite prediction artifacts, all four
CoAt readers, zero fallback events, exact CPU-admitted native pixel/mask hashes
and eleven unchanged columns. Medial Meniscus AUC also worsened. Only one fixed
vote ran; no weight search or competition submission. Its source metadata
canonicalizes dataset `/2`; actual mounted payload hashes establish contents,
without separate numeric mounted-version telemetry. [Evidence](../../docs/evidence/ticket-53-public-meniscus.md).

Ticket 57's unfitted prefix manifest SHA-256:
`21bfeb43e24e8e55797961aeb6f0ad6b313cafad4cdb22ee2aeb2f2870d84e40`.
Payload 37,586,239,488 bytes, all source/packing/order and reconstructed frozen
features independently checked. [Evidence](../../docs/evidence/ticket-57-frozen-resnet-prefix-cohort.md).

Ticket 58's six fixed final runs pass independent original-DICOM replay:
116 final predictions, 58 warm-parent predictions and all zero-image controls
match exactly at 16 threads / batch two; five paired comparisons are identical.
Fit plan `81da4c9de7b400ab232df49166ba2d668167e806e40502b76d0123036cf6f388`;
validation `57a44ce3b24dccd7841d2b062ae1dc0a012e3a0408786f3d1fccdbfaa11502df`.
No D4 complementarity or leaderboard improvement. [Evidence](../../docs/evidence/ticket-58-report-layer4-adaptation.md).

## Standing authorization and resource state

Owner: **“approve anything as long as we win 1st place and fits kaggle comp dont
ask me”**. Handle justified Kaggle-compliant decisions autonomously with **$0
external spend**. [Authorization](resource-approvals/campaign-owner-autonomy-20261001.json)
SHA-256 `634562f8d5d89016e625ecb6befb57ed410b281a1f70762f3b4391d8ed7103ca`.
Record concrete agent decisions; do not label them new exact numeric owner approvals.

- GPU ledger `.scratch/rsna-leaderboard/gpu-campaign-ledger.json`: cap **136,800 s
  / 38 h**, charged **129,637 s**, reserved **0**, remaining **7,163 s**. Ticket 53's
  standing-authorized 36→38-hour amendment is concrete and retained. Its sole
  free-quota increment 1,263.767 wall seconds is rounded to 1,264 wall / 2,528
  GPU seconds; allocation interval unknown. Preserve all prior unknown charges.
- No current campaign CPU or GPU allocation. One pending/active GPU request
  across backends; at most two nonexclusive Slurm CPU jobs, each <=16 CPUs /
  64 GiB / six hours. Use narrower exact decisions where recorded. No paid scaling.
- Storage soft ceiling 2 TB. Science freeze October 11, internal delivery before
  October 15. Recorded official final deadline October 22, 23:59 UTC; five daily
  submissions, two final selections, nine-hour notebook limit, internet off.
- Check/commit/push completed work to private **https://github.com/Jitheswar/RSNA**,
  configured `origin`, branch `master`; verify remote/local commit equality.

## Immutable evidence and operational corrections

Private roots: `public-frontier-adoption/scoring-failure-diagnosis-v1/`,
`public-meniscus-adoption/paired-development-v4/`, `ticket57-resnet-prefix-cohort/`,
`ticket58-prefix-cost-v1/`, `ticket58-report-layer4-adaptation-v1/`, all under
`artifacts/private/`. Retain consumed attempts and snapshots; do not overwrite.

Ticket 57's extraction plan freezes the original preparation/extraction scripts;
keep those bytes unchanged. Ticket 58's cost plan froze its original mutable
issue: all 104 dependencies were rechecked, then the exact original prospectus
was archived in `prospective-spec-archive.md` with `consumed-spec-archive.json`
before the tracker outcome update. The fit has a separate immutable prospective
spec/source snapshot. Historical cost replay must use the archived original
spec identity rather than assuming today's resolved issue has its old bytes.

Fit observer initially used guessed job 7987; it was stopped and rebound to
actual accepted 7989. Verifier observer initially used unsupported flags; a
correct positional launch retained actual terminal 7991. No model job was
relaunched. Correction receipts are preserved. The generic observer now rejects
IDs differing from the accepted launch and foreign controller IDs. Parse actual
submission responses; never guess the next job number. Its CLI takes positional
`ROOT JOB_ID`, not `--root/--job-id`.

Latest focused checks: 28 prefix/cache/observer/overlay tests pass, static typing
passes 82 source files, actual Ticket 53 validation and Ticket 58 complete native
integration verification pass. No hidden-runtime fix or first-place claim.
