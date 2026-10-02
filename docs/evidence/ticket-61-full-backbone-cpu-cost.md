# Ticket 61 — whole-backbone implementation and real-MRI CPU cost

The separate full-backbone MRI variant is implemented and a bounded CPU pilot
completed **8016 /COMPLETED /0:0**, **127 allocated CPU-node seconds**. All
encoder stages changed and normalization running buffers stayed fixed. A fresh
process independently admitted report targets/masks, strictly reloaded the
diagnostic checkpoint and reproduced all logits **exactly**. This establishes
an executable learning and checkpoint path, not predictive improvement.

## Implemented route

`resnet18-full-backbone-static-v1` is a separate model identity reconstructed by
the shared `ModelConfig`/model factory/checkpoint loader. It preserves admitted
ResNet tensor names, grayscale/ImageNet normalization and anatomical pooling.
Every encoder parameter is trainable; encoder normalization uses its admitted
running statistics. Fixed-size slice chunks encode the entire padded bag;
mask changes do not produce variable-length slice tensors. Padded pixels are
replaced before encoding, including NaNs, and padded features are excluded.
Chunking bounds individual operation workspace; backward retains activations
across all chunks, so the whole batch still requires measured memory admission.

The new fixed-shape masked BCE normalizes over supervised entries, excludes
absent NaN logits/targets and contributes zero gradient from missing targets.
It avoids host scalar reads and variable-length target indexing. Empty masks
produce differentiable zero; data admission must separately reject empty
supervision before fitting. Prior frozen/layer4 variants retain their identities.

**39 focused tests pass**, including early/late encoder gradients and updates,
unchanged normalization buffers, padding isolation, invariant encoder chunk
shapes across mask changes, masked BCE value/gradient equivalence, strict
checkpoint reconstruction and exact CPU replay. Typing passes **83 source
files**. These small fixtures establish implementation boundaries only.

## Exact independent CPU experiment

Private root: `artifacts/private/ticket61-cpu-full-backbone-pilot-v1/`.
Plan SHA-256:
`43adab68e2113441d5bccb04273a935bd43ab8039cbbd0a841f4758b4364bfaa`.
Result SHA-256:
`0937052b0dbf7da3e66f4a1c2aa89e1038c83424a5f82627ed04c17c37ecff6b`.
Diagnostic checkpoint:
`d8e876ccda8f701f3c917cf0ec929acc428b29fc40b2a392a6c9e3006da49b05`.

The [agent decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket61-cpu-full-backbone-pilot-20261002.json)
uses standing owner authorization. One nonexclusive CPU job, **16 CPUs /64 GiB
/900 seconds**, **780-second** code watchdog, one attempt/no requeue, zero
GPU/external spend. Its launcher runs the independent verifier in a fresh
process within the same allocation. Exact accepted job, complete terminal
resources/source/plan, exit code and independent verification are reconciled.

Four label-blind slice-count order statistics reuse Ticket 56's admitted fold-0
training reports. Held-out and reference-gold studies are excluded. Original
prepared series are read-only and their MRI source/packing/mask/descriptor
identities are checked. The complete packed input SHA matches the previous
actual pixel admission; **232 valid slices /512 padded slots** and **14
supervised entries per update**. Initialization uses admitted public ResNet
weights and the prior fold-0 long report head. Native frozen and static initial
logits match **exactly**.

One warmup plus three timed Adam updates, encoder **1e-5**, head **1e-4**.
Timed updates **5.955313 /6.150059 /5.628644 seconds**, median **5.955313**.
Every encoder parameter has a finite gradient. Stem/layer1/2/3/4 and head
weights change; normalization buffers change **zero**. Peak producer RSS
**17,880,358,912 bytes**. Fresh-process replay verifies exact retained MRI array
identity, independently reconstructs fold-admitted targets/masks, strictly
reloads the full checkpoint and matches all four study logits with max error
**0**. The four fitting losses decline; no held-out metric was computed.

## Measured feasibility and unresolved work

At a **2×** training-time allowance:

| Fixed duration | Per fold | Three folds plus a prospective final fit |
| --- | ---: | ---: |
| 1,500 updates | 17,865.939 s /4.963 h | 71,463.755 s /19.851 node h |
| 7,500 updates | 89,329.694 s /24.814 h | 357,318.774 s /99.255 node h |

The short training portion fits one six-hour CPU job. The long duration needs
staged checkpoints and multiple bounded jobs. These are warm-cache projections
from four reports and static 512-slot bags; random full-cohort I/O, compact pixel
materialization, checkpoint boundaries, validation, final fit overhead and
offline delivery remain unmeasured. Neither duration is a complete admitted
training route. No full campaign, candidate promotion or scoring request was
launched. Diagnostic fitting is not a provisional candidate.

The measured CPU alternative can progress independently of the queued TPU
environment probe. The next substantive step is a complete fixed comparison
and source-cache/I/O/checkpoint cost plan, using all prescribed folds and matched
reference exposures. TPU availability and the full CoAt scan remain separate
live questions. Best confirmed public **0.936**; both first-place ranks remain
unproven.
