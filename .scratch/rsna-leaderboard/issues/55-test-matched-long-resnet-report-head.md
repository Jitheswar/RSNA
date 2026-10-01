# 55: Complete a matched long ResNet report-head comparison

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)

## Hypothesis

Ticket 54's fixed longer ConvNeXt report head reached development AUC 0.691065,
with an inconclusive duration gain. Measure ResNet with the same 7,500-step
budget to distinguish duration from representation, then evaluate **one fixed
50/50 report-head blend**. No blend weights or training budgets are swept.
Neither representation has yet been measured against D4 on the same cohort.

## Frozen scope

- Use the existing 4,228 verified ResNet feature entries, original anatomical
  pooling, MLP32, exact three folds, report admission, seeds, optimizer and loss.
- Pool once without fitting, retaining hashes of every source entry and pooled
  array. Require all six 1,500-step prefixes to reproduce their Ticket 44
  initialization, exposure digests, head state and predictions within the
  existing 1e-6 replay gate before continuing.
- Train three matched reference-only/report pairs to 7,500 successful steps.
  Final checkpoint only; no intermediate metric chooses a checkpoint or budget.
- Compare duration per arm, matched report-minus-reference, long ConvNeXt
  versus long ResNet, and the fixed half blend versus long ResNet. Preserve
  image ablations, per-target support, fold directions and 2,000 paired study
  resamples, with reused-development and missing patient-key limitations.

## Resource decision

One nonexclusive CPU Slurm job, **16 CPUs / 64 GiB / 30 minutes**, zero GPUs,
1,200-second code watchdog, one attempt, no requeue or new downloads/network.
Existing features and private experimental checkpoint admissions are reused.
Standing October 1 owner authorization applies; no new numerical owner approval
is claimed. No GPU cap change, competition submission, publication or final
selection. Accepted scoring **56739912** remains the sole GPU request.

## Requirements

- [x] Freeze the original inputs, source snapshot, runtime and resource decision.
- [x] Verify source/pooling identities and all six original prefixes.
- [x] Complete all six fixed final runs and independently replay predictions.
- [x] Record the fixed comparisons and disposition in the final register.

## Comments

All choices above precede 7,500-step ResNet predictions. A paired lower bound
above zero and positive directions in every fold permit provisional review only,
with image-signal controls. No OOF development result establishes first place.

October 1 session transfer: original job **7977** failed `1:0` after **32
allocated CPU-node seconds**, zero GPU/spend. Its first reference-only 1,500-step
weights and predictions reproduced exactly. The guard rejected historical
absent-report `null` versus the canonical empty hash despite zero report exposures.
The comparison is repaired with four regression tests; report-bearing sequences,
scientific settings and the 1e-6 gate remain strict. The original attempt is consumed.
Separate diagnosed **v2** recovery is supported in source but **not frozen or
launched**. All six prefixes, long results, independent final replay and comparisons
remain unfinished. See [failure evidence and recovery steps](../../../docs/evidence/ticket-55-resnet-recovery.md)
and [current handoff](../session-handoff-20261001.md).

## Answer — October 1 resumed execution

Separate diagnosed recovery v2, plan
`8e6899d7af4688e1512686559ff9c38bc94673cc2364d8264fb7e95588928e49`,
completed in Slurm job **7978**, `0:0`, **290 CPU-node seconds**, zero GPU/spend.
Independent prelaunch verification checked 111 frozen files and pinned 4,228
features. The recovered pooled bank is byte-identical to the original unfitted
bank; all six original 1,500-step prefixes and final 7,500-step budgets passed.

Independent verifier **7981**, `0:0`, **60 CPU-node seconds**, checked all source
and feature hashes, six final checkpoints, fold exclusion and ordered exposures.
All **116** final predictions replay exactly with frozen 16-thread/batch-two
settings. Seven full comparisons and 2,000 paired resamples each reproduce.
Verifier **7980**'s diagnosed snapshot-path failure cost **6 CPU-node seconds**;
its evidence is retained and no training was repeated.

Report ResNet AUC **0.686730** versus short **0.660737**: delta **+0.025993**,
interval **[-0.011717,+0.062958]**, all folds positive. Fixed **50/50** report
blend AUC **0.705751**: delta versus long ResNet **+0.019021**, interval
**[-0.009373,+0.050577]**, folds **+ / − / +**. Report-minus-reference and report
image-ablation controls are positive, but the declared blend gate fails.
Disposition: **inconclusive/no promotion**. Same-cohort D4 complementarity and
final delivery are unverified. [Full results and retained failures](../../../docs/evidence/ticket-55-resnet-recovery.md).
