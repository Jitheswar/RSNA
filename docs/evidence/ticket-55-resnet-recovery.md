# Ticket 55 — completed matched long ResNet and fixed blend

The recovered 7,500-step ResNet report head reached **0.686730** development AUC;
the prospectively fixed 50/50 ResNet/ConvNeXt report-head blend reached **0.705751**.
The blend's paired improvement interval crosses zero and one fold declines, so
the experiment closes as **inconclusive/no promotion**. Neither result establishes
a competition gain or first place on either leaderboard.

## Completed recovery and independent verification

Diagnosed recovery **v2** was frozen before launch, with SHA-256
`8e6899d7af4688e1512686559ff9c38bc94673cc2364d8264fb7e95588928e49`.
Independent preparation verified **111 frozen files**, pinned **4,228 feature
files**, compiled 82 Python files and checked the launcher's Bash syntax. The
optimization, seeds, folds, report admission, fixed half blend and **1e-6** gate
match the original plan. The separate recovery preserves the consumed original
attempt and binds its operational failure diagnosis.

Slurm CPU job **7978** completed `0:0` on **KLEF-C-NODE1**, **16 CPUs / 64 GiB**,
one nonexclusive node, no requeue or restarts. Start **06:11:49 UTC**, end
**06:16:39 UTC**: **290 allocated CPU-node seconds**. Code elapsed **285.601 s**;
peak process RSS **750,997,504 bytes**. All six runs completed **7,500/7,500**
successful steps, without skipped batches. All six original 1,500-step prefixes
matched initialization, ordered reference/report exposures, weights and predictions
exactly. The complete pooled bank is byte-identical to the original request's
unfitted bank, with shape **(4228,2076)**.

Independent verifier **7981** completed `0:0` in **60 allocated CPU-node seconds**
on the same CPU resource shape. It rehashed all 4,228 feature files and 111 frozen
inputs, checked fold membership and ordered exposure digests, strictly loaded six
final checkpoints and independently replayed all **116** held-out predictions.
Every observed replay error was **0.0**, using frozen **16-thread**, batch-two
settings. It recomputed all seven comparisons, including all **2,000** paired
study resamples per comparison, seed **20261001**, and the declared disposition.
Terminal controller responses were retained before purge.

Verifier attempt **7980** failed `1:0` after **6 CPU-node seconds** because its
relocated source snapshot inferred the wrong project root. This stopped before
prediction replay or metric recomputation. The separately recorded verifier
recovery resolves the exact hash-verified fold path from the frozen plan and sets
the project root in its launcher. Original failure evidence remains intact;
no training was rerun, scientific setting changed or tolerance relaxed.

This continuation used **356 CPU-node seconds** in total, including the failed
verification, **zero GPU seconds** and **$0 external spend**. The original failed
training request's 32 CPU-node seconds remain separately recorded below.

## Fixed comparison results

| Comparison | Macro AUC delta | Paired 95% interval | Fold directions |
| --- | ---: | --- | --- |
| Report 7,500 vs 1,500 steps | +0.025993 | [-0.011717,+0.062958] | + / + / + |
| Reference 7,500 vs 1,500 steps | +0.003631 | [-0.036508,+0.043914] | − / − / + |
| Long report vs matched long reference | +0.054709 | [+0.006222,+0.104346] | + / + / + |
| Long report vs zero image features | +0.151880 | [+0.051783,+0.242587] | + / + / + |
| Long reference vs zero image features | +0.102663 | [+0.001956,+0.198201] | − / + / + |
| Long ConvNeXt report vs long ResNet report | +0.004335 | [-0.038930,+0.054519] | + / − / − |
| Fixed half blend vs long ResNet report | +0.019021 | [-0.009373,+0.050577] | + / − / + |

Long reference-only ResNet AUC is **0.632021**. Short report ResNet is **0.660737**;
long report ConvNeXt is **0.691065**. All seven comparisons have 2,000 defined
macro resamples and zero undefined outcomes. Complete private evidence retains
per-target support, estimates and uncertainty. Every fold exposed all
**2,731 / 2,846 / 2,749** admitted reports, with **15,000** report draws and
matched **15,000** reference draws.

Report supervision and image-feature controls establish useful signal on this
development cohort. Duration improvement and architecture/blend advantage remain
uncertain. No additional budget, weight or checkpoint was selected from these
results. This reused 58-study development set has no attested patient key, and
study bootstrap does not remove repeated-development or patient-group limitations.
No same-cohort comparison against D4, final candidate or scoring request is claimed.

## Reproduction and disposition

Private recovery root: `artifacts/private/ticket55-resnet-long-report-head-v2/`.
`analysis.json`, `validation.json`, original/final checkpoints and both verifier
attempts retain exact identities. The [independent verifier](../../scripts/validate_ticket55_resnet_long_report_head.py)
can be run through the preserved bounded Slurm launcher at
`independent-validation-v2/run.sbatch`; its immutable output already exists, so
a later audit must use a fresh output path rather than overwrite it.

[Training recovery decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket55-resnet-long-report-head-20261001-v2.json)
and [verifier recovery decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket55-independent-validation-20261001-v2.json)
use standing owner authorization without claiming new numerical owner approvals.
[Ticket 55](../../.scratch/rsna-leaderboard/issues/55-test-matched-long-resnet-report-head.md)
is resolved as **inconclusive/no promotion**. GPU accounting, pending submission
56739912 and both scored fallbacks are preserved. Final delivery remains unfinished.

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

## Historical recovery procedure (now completed)

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

The procedure above was completed by jobs 7978 and 7981, with the separately
diagnosed verifier failure preserved. Same-cohort D4 comparison and final delivery
remain unfinished.
