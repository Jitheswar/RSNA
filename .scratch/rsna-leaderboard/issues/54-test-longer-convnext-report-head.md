# 54: Test a fixed longer ConvNeXt report-supervised head budget

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)

## Hypothesis

Ticket 50's 1,500-step report head saw 3,000 report draws from each fold's
2,731–2,846 admitted report studies: roughly one pass. A fixed **7,500-step**
run may improve image learning without new encoder extraction or GPU use.
This tests duration, not architecture or a blend-weight search. The existing
report AUC 0.659947 is development evidence. Performance on the same cohort
as the scored frontier and useful complementarity remain separate questions.

## Frozen scope

- Reuse the verified 4,228-study ConvNeXt pooled bank and corrected three folds.
- Preserve the exact MLP32, optimizer, loss, learning rate, seeds, initialization,
  masks, report admission and ordered gold/report sampling from Ticket 50.
- Train three matched reference-only/report pairs for 7,500 successful steps.
  Require each 1,500-step prefix to reproduce its original exposure digests,
  checkpoint state and held-out predictions before continuing.
- Select only the final checkpoint. No intermediate metric selects a budget,
  weight or checkpoint. Strictly reload and replay all 116 final predictions.
- Compare each arm with its own 1,500-step baseline; report-minus-gold,
  descriptor-preserving image ablations, target support, fold directions and
  2,000 paired study resamples. Retain undefined outcomes and reused-cohort limits.

## Resource decision

One nonexclusive Slurm CPU job, **16 CPUs / 64 GiB / 30 minutes**, zero GPUs,
1,200-second code watchdog, one launch attempt, no requeue, no downloads,
network, publication, final selection or external spend. The standing October 1
owner authorization permits this concrete CPU experiment without another
question. It does not change the GPU cap or reservation for accepted scoring
**56739912**, and it does not launch the prepared meniscus GPU experiment.

## Requirements

- [x] Pin inputs, old final checkpoints, source snapshot, runtime and one exact plan.
- [x] Independently check live allocation and original 1,500-step replay.
- [x] Complete all six final runs and checkpoint/prediction replay.
- [x] Analyze fixed comparisons and record disposition in the final register.

## Comments

Selection is made before observing any 7,500-step predictions. A positive
paired interval and positive directions in all three folds permit provisional
review only; they do not establish leaderboard improvement or first place.

### October 1 — one CPU launch accepted, 04:47 UTC

The immutable plan is
`artifacts/private/ticket54-convnext-long-report-head/plan.json`, SHA-256
`1b77479d88d1a3686a14fc292449456c522dd689b579fb07a9ddda5c38abb593`.
Independent prelaunch verification checked **100 frozen files**, all three
partition boundaries, unchanged optimization settings and the 30-minute CPU
resource envelope. Shell syntax and source compilation passed.

Slurm job **7975** was accepted once and is RUNNING on **KLEF-C-NODE2**,
16 CPUs, 64 GiB, zero GPUs, no requeue/restarts. Live controller allocation is
retained. No completed prediction or predictive improvement is claimed yet.
[Concrete autonomous CPU decision](../resource-approvals/ticket54-convnext-long-report-head-20261001.json).

## Answer — October 1

Job **7975** completed `0:0` in **256 allocated CPU-node seconds**, no GPUs
or spend. All six 7,500-step runs completed without skipped batches. Every
1,500-step prefix reproduced the original weights, initialization, ordered
exposures and held-out predictions exactly. All **116 final predictions**
replayed exactly after strict checkpoint reload and independently with the
frozen **16-thread** CPU setting. A four-thread probe exceeded the unchanged
1e-6 gate at **1.3113e-6**; that negative probe is retained. Matching the
frozen threading setting restored exact replay; no tolerance was relaxed.

Report-supervised OOF development AUC **0.691065** versus its 1,500-step
**0.659947**, delta **+0.031119**, paired 95% interval
**[-0.004120,+0.067897]**; two fold directions improve and one declines.
Reference-only duration delta **-0.001701** is also inconclusive. The long
report arm beats its matched long reference-only arm by **+0.089610**,
interval **[+0.044308,+0.138018]**, all three folds positive. Report
full-minus-zero-image delta **+0.174286**, interval
**[+0.110341,+0.234113]**, all three folds positive. These controls establish
development image signal and report-supervision benefit for this matched
budget, but not a reliable longer-versus-shorter gain or a competition gain.

All **2,731 / 2,846 / 2,749** admitted report studies were exposed in their
fold, with **15,000 report draws per fold**, matched 15,000 reference draws,
and held-out target entries masked. All five comparisons retain twelve-target
support, fold directions and 2,000 defined paired resamples with zero undefined.
Disposition: **inconclusive/no promotion**. No blend, public submission,
final selection or first place follows from this result. The reused reference
development cohort remains a limitation. [Evidence](../../../docs/evidence/ticket-54-long-convnext-report-head.md).
