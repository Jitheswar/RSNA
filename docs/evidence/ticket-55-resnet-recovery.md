# Ticket 55 — failed original request and prepared metadata repair

No 7,500-step ResNet or fixed-blend result is available. The original one-attempt
CPU request failed on an absence-metadata check after its first 1,500-step weights
and predictions reproduced the historical run exactly. Ticket 55 remains claimed.
Repair is tested; recovery v2 is unprepared and unlaunched at session transfer.

## Original execution and diagnosis

Root: `artifacts/private/ticket55-resnet-long-report-head/`. Original plan SHA-256:
`49c1e99220ff4df1b8bcc23253114caa6354b8d9d0c7cf749afaa483d214fedd`.
Launch accepted **2026-10-01 05:25:41 UTC**, job **7977**. Controller reported
**FAILED / 1:0**, start **05:25:42 UTC**, end **05:26:14 UTC**, **32 allocated
CPU-node seconds**, `KLEF-C-NODE1`, 16 CPUs / 64 GiB, zero GPU/spend, no requeue
or restarts. The terminal response was observed in the prior session before Slurm
purged the job; no raw terminal file was saved. Retained `failure-diagnosis.json`
records the outcome/duration. A missing controller record is not a live-job signal.

Original preparation independently verified 109 frozen files and pinned 4,228
feature entries. `pooled-cohort-evidence.json` records all 4,228 verified actual
entries and the `(4228,2076)` bank. Preserve original snapshot, plan, decision,
launch intent/response, accepted launch, running allocation, pooled evidence and
`slurm-7977.err`.

First reference-only prefix: **0.0 maximum head-state error**, **0.0 prediction
error**. Independent diagnosis reproduced full ResNet initialization identity and
ordered gold-exposure digest. Both runs had zero report sample/supervised-entry
exposures. Historical absent weak sequence is `null`; the newer helper hashes the
empty sequence as `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
The literal comparison stopped fitting. Only the first prefix was reached; all
six original-prefix checks remain required.

`scripts/ticket55_resnet_long_report_head.py::weak_sequence_matches` accepts the
historical `null` only with zero exposures on both sides and the current canonical
empty hash. Actual report sequence digests remain strict. Initialization, gold
sequence, weights, optimization, prediction gate and **1e-6 tolerance** are unchanged.
Four regression tests cover equivalent absence, hiding report draws, corrupt or
missing empty hashes, and strict ordering of report sequences.

## Next CPU action

1. Inspect owned CPU jobs and confirm no recovery request/decision already exists.
   Preserve the consumed original one-attempt identity without relaunching it.
2. Prepare the separate diagnosed recovery:

   ```bash
   RSNA_TICKET55_REVISION=2 .venv/bin/python scripts/ticket55_resnet_long_report_head.py prepare
   ```

   This creates `artifacts/private/ticket55-resnet-long-report-head-v2/` and decision
   `.scratch/rsna-leaderboard/resource-approvals/ticket55-resnet-long-report-head-20261001-v2.json`.
   It binds the original plan/diagnosis, permits stage attempt 2 of 2, and writes a
   new immutable snapshot and plan identity. It does not submit a job.
3. Independently verify all frozen files, compile the snapshot, run `bash -n` on
   `run.sbatch`, and compare optimization, seeds, folds, admission, fixed **50/50**
   blend and **1e-6** gate with the original scientific choices. Intervening commit
   `642c787` added helper type annotations; retain the actual recovery snapshot
   and require all six prefix gates before accepting its results.
4. Record an exclusive immutable intent, submit the new launcher once with
   `sbatch --parsable`, and immediately save the actual response/job ID. One launch
   of this identity, no requeue. Resources: one nonexclusive node, **16 CPUs /
   64 GiB / 30 min**, **1,200 s** watchdog, zero GPU/network/spend.
5. Require six 1,500-step prefix gates and six final 7,500-step runs. Save terminal
   controller evidence before purge. Verify checkpoint hashes and independently
   replay **116** final predictions using frozen **16-thread**, batch-two settings.
   Recompute AUCs and seven fixed comparisons with **2,000 study resamples**,
   seed **20261001**.
6. Record support, folds, uncertainty and image controls. A positive fixed-blend
   comparison permits provisional review under the declared gate, without automatic
   promotion. Update ticket/map/register from independently verified results,
   then check, commit and push.

Standing authorization covers the concrete recovery decision; no numerical owner
approval or GPU-cap change is claimed. Same-cohort D4 comparison, development
limitations and final delivery remain unfinished.
