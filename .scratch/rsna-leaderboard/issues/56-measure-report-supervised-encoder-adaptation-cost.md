# 56: Measure CPU cost of report-supervised last-block adaptation

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)

## Question and prospective scope

The frozen ResNet/ConvNeXt heads show development image signal, but their long
budgets and fixed blend do not establish a competitive gain. Before proposing
larger training, measure the cost of adapting ResNet-18 layer4 and its MLP32
using existing admitted report supervision. Ticket 39's 100-step reference-only
experiment did not test this broad report-training route.

Use four report studies admitted for training in corrected fold 0, selected
without labels by actual valid-slice count at the 10th/40th/70th/100th order
statistics, SHA-256(uid) tie-break. Exclude every gold study and fold-0 held-out
study. Reuse exact 224px whole-field preprocessing, series/slice sampling,
anatomical pooling, ImageNet weights and the verified long fold-0 report head.
No adjacent-slice channel context or Ticket 40 experiment is included.

Measure two equivalent implementations with fresh identical initialization:
native frozen-prefix encoding on every update, and precomputed unfitted
layer3 activations with adaptation only through layer4/head. Each performs
one warmup plus four timed updates, batch four, masked admitted report BCE,
Adam, layer4 learning rate 1e-5 and head rate 1e-4. Require native/cached score
and final-state parity within 1e-6, finite gradients/losses, actual layer4/head
parameter changes, unchanged earlier weights and normalization buffers.
All fitted probe weights are discarded; no candidate checkpoint or AUC is made.

Measure prefix extraction, temporary activation write/read with hash parity,
peak memory and valid-slice counts. Project one possible three-fold 1,500-update
adaptation using two reference plus two report draws per update and actual
fold-specific slice counts, with a 2x cost allowance. This is a cost scenario,
not a selected training budget or evidence of model benefit. Full extraction,
training and promotion require separately specified decisions. Reject CPU
follow-up if a fold exceeds six hours, total projected training/extraction exceeds
24 CPU-node hours or projected memory/storage exceed existing limits.

## Resource decision

One nonexclusive Slurm CPU node, **16 CPUs / 64 GiB / 15 minutes**, a **780-second**
watchdog, one launch attempt, no requeue, zero GPU/network/spend or new downloads.
Standing owner authorization applies. Exact scoring **56739912** stays the sole
GPU request; no cap amendment or submission is made.

## Requirements

- [x] Freeze source, inputs, deterministic membership and concrete resource decision.
- [x] Verify actual MRI/cache identity and exclude held-out fitting inputs.
- [x] Measure both implementations and verify equivalent real updates.
- [x] Record costs, projection assumptions and disposition; preserve failed attempts.

## Comments

Prepared after Ticket 55's completed, independently verified result. Only
training-side images and admitted report targets are used. Throughput measured
on four reports cannot prove generalization, full-cohort I/O or a leaderboard gain.

## Answer — October 1, 06:45 UTC

One launch, job **7982**, completed `0:0`, **28 CPU-node seconds**, zero GPU/spend.
All **232** real slices preserve prior source/mask/descriptor identities; all four
reports are training-side, with 14 admitted entries per update. Native median
**1.315271 s**, cached-prefix median **0.508122 s**. Both logits and the final
weights after five real updates match exactly; only layer4/head changed and all
normalization buffers stayed fixed. No fitted checkpoint was retained.

Independent checks verified 95 frozen files and recalculated the registered
three-fold scenario: **2.305 node hours** with 2x allowance and **37.586 GB** of
activation payload. This is **eligible for separate full-route planning**, subject
to full-cohort I/O/extraction, storage, training and final-evaluation checks.
No predictive gain or final candidate is claimed. [Evidence and limits](../../../docs/evidence/ticket-56-cpu-adaptation-cost.md).
