# Ticket 54 — fixed longer ConvNeXt report-head experiment

The fixed 7,500-step report-supervised head reached **0.691065** OOF development
AUC, up from **0.659947** at 1,500 steps. Its paired interval crosses zero and
one fold declines, so the duration branch closes as **inconclusive/no promotion**.
The campaign objective remains first place on both public and final private
leaderboards; this experiment establishes neither rank nor a competition score.

## Actual execution and reproduction

One Slurm CPU job **7975** completed `0:0` on **KLEF-C-NODE2**, 16 CPUs,
64 GiB, zero GPUs, no requeue or restarts. Start **04:47:13 UTC**, end
**04:51:29 UTC**: **256 allocated CPU-node seconds**. Code elapsed **248.320 s**,
peak process RSS **606,846,976 bytes**, external spend **$0**. The terminal
controller response was retained before Slurm purged the completed job.

The experiment reuses the verified 4,228-study ConvNeXt pooled bank and exact
Ticket 50 MLP32, learning rate, loss, seeds, masks, samplers and fold admission.
Only the successful-step budget changes from 1,500 to 7,500. Saving every 1,500
steps replaces the earlier 15-step save interval without changing optimization.
No new weight download, encoder fitting, network access or blend selection ran.

All six runs completed **7,500/7,500** successful steps with no skipped batches.
Each 1,500-step prefix reproduced its original initialization, ordered gold/report
exposures, complete head state and held-out predictions exactly. Final
checkpoint reload and independent matching-thread CPU replay reproduced all
**116 predictions exactly**. Held-out target entries are masked from the fitting
tensor; held-out studies never enter the sampled training partitions.

A separate four-thread replay probe exceeded the unchanged **1e-6** tolerance,
observed error **1.3113021850585938e-6**. Repeating with the frozen **16-thread**
setting produced **zero error for all six checkpoints**. Both observations are
retained; no tolerance was relaxed and no training rerun occurred. Reproduction
must retain the frozen threading setting.

## Fixed comparison results

| Comparison | Macro AUC delta | Paired 95% interval | Fold directions |
| --- | ---: | --- | --- |
| Report 7,500 vs 1,500 steps | +0.031119 | [-0.004120,+0.067897] | + / − / + |
| Reference 7,500 vs 1,500 steps | −0.001701 | [-0.024058,+0.020103] | − / − / + |
| Long report vs matched long reference | +0.089610 | [+0.044308,+0.138018] | + / + / + |
| Long report vs zero image features | +0.174286 | [+0.110341,+0.234113] | + / + / + |
| Long reference vs zero image features | +0.073646 | [+0.005342,+0.141688] | + / + / + |

The long reference-only AUC is **0.601455**. All five comparisons use **2,000
paired study resamples**, seed **20261001**, with all 2,000 defined and zero
undefined macro outcomes. Full evidence retains target support and intervals.
The zero-image control preserves descriptors and masks while removing image
features from the same fitted head.

All **2,731 / 2,846 / 2,749** fold-admitted report studies were visited; each
fold received **15,000 report draws** and a matching **15,000 reference draws**.
Report-supervision and image-feature controls are positive for this development
budget. Longer report training has an encouraging point estimate, but fails the
predeclared provisional-review gate because its paired lower bound is negative
and one fold declines. No further duration or blend variants were selected.

These are reused OOF development estimates, with study bootstrap because no
attested patient key is available. They are not an untouched generalization
estimate. They have not been evaluated on the same cohort as the scored D4
frontier; raw AUC numbers across those cohorts do not establish complementarity.
No final candidate, public submission or leaderboard improvement is claimed.

## Provenance and disposition

Private root: `artifacts/private/ticket54-convnext-long-report-head/`.
Exact plan SHA-256
`1b77479d88d1a3686a14fc292449456c522dd689b579fb07a9ddda5c38abb593`.
Independent preparation verified **100 frozen files**. Result validation checked
all six checkpoint/prediction identities, exposure counts, original prefix
replay, recomputed pooled AUCs and persisted controller evidence.

[Concrete agent resource decision under standing authorization](../../.scratch/rsna-leaderboard/resource-approvals/ticket54-convnext-long-report-head-20261001.json)
records a private, zero-spend CPU experiment, without claiming a new numeric owner
approval. No GPU cap or reservation was changed by this experiment.
[Resolved ticket](../../.scratch/rsna-leaderboard/issues/54-test-longer-convnext-report-head.md).
