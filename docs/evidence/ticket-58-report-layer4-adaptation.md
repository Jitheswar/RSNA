# Ticket 58: matched report-supervised adaptation remains inconclusive

The fixed layer4 adaptation did not improve on matched head continuation.
**No promotion or competition submission.** The independently replayed primary
development result is **0.689232 vs 0.691387**, paired delta **−0.002155**,
95% interval **[−0.022304,+0.019717]**, with one negative fold.

## Fixed comparison and independent evidence

Both arms start from each fold's independently admitted Ticket 55 long report
head and the original ImageNet ResNet-18 encoder. Each receives the same 1,500
successful Adam updates, two reference plus two report draws per update, seed
20260912 and report offset 34034034. Head rate 1e-4; adaptation additionally
trains layer4 at 1e-5. Earlier encoder parameters and all normalization buffers
stay fixed. No augmentation, sweep, changed packing, selected intermediate
checkpoint or new representation. All 4,228 studies use the complete independently
verified Ticket 57 unfitted prefix bank; no held-out input or label enters fitting.

| Registered development comparison | Paired macro AUC delta | 95% interval |
| --- | ---: | --- |
| Adaptation − matched head continuation (primary) | −0.002155 | [−0.022304, +0.019717] |
| Adaptation − warm report parent | +0.002502 | [−0.018120, +0.025671] |
| Head continuation − warm report parent | +0.004657 | [−0.005013, +0.014625] |
| Adaptation − zero encoded image features | +0.153733 | [+0.050706, +0.246726] |
| Head continuation − zero encoded image features | +0.153662 | [+0.051591, +0.247388] |

Each comparison uses the same 58 held-out predictions, twelve target assessments,
2,000 paired study resamples, seed 20261001; all resamples are defined. The primary
fold deltas are **+0.052518 / +0.001877 / −0.018148**. Both image controls are
positive in every fold, preserving the original masks/descriptors; this establishes
development image signal, not an adaptation advantage. Per-target class support
and all fold metrics remain in the immutable analysis artifact.

Fit **7989** and independent verifier **7991** completed `0:0`. The verifier
strictly reloaded all six final checkpoints and checked initial state and warm
parent identity, fold membership, report admission, complete target masks,
ordered source/entry exposures, 1,500-step Adam moments/rates, finite losses and
state tensors, changed treatment layer4/head and unchanged earlier parameters
and normalization buffers. All **116 final predictions**, **58 warm-parent
predictions**, and zero-image controls agree **exactly** with inference on
original DICOMs at the frozen **16-thread / batch-two** settings. Selected slice
order, source identity, masks and descriptors also agree. Maximum prediction
error **0.0**; all five comparisons recompute identically from both routes.

## Measured cost

Full-bank cost probe **7986** completed `0:0`, **180 CPU-node seconds**. Loading
and verifying all 4,228 entries took **140.500 s**, payload **37,586,239,488 bytes**,
peak process RSS **38,800,142,336 bytes**. A fixed 128-study random-read probe
read **1,132,928,436 bytes** in **4.539 s** on the existing warm/shared cache;
this is not a cold-cache measurement. Six real batches per fold established
finite gradients, intended parameter changes, fixed buffers and exact frozen
logits; their fitted states were discarded and no held-out prediction was computed.
The prospective conservative fit projection was **5,676.858 seconds**, including
2x measured load/training allowances and 1,800 seconds for final checks.

Actual six-run fit allocation **1,982 node seconds**, peak process RSS
**38,959,521,792 bytes**. Independent actual-MRI verification **321 node seconds**,
peak RSS **3,725,078,528 bytes**. Total cost probe + fit + verification **2,483
CPU-node seconds**, zero GPU and $0 spend. Ticket 57's separate extraction and
complete activation verification cost **2,558 node seconds**; these allocations
are measured node time, not elapsed campaign time.

## Identity and limitations

Private fit root: `artifacts/private/ticket58-report-layer4-adaptation-v1/`.
Fit plan SHA-256:
`81da4c9de7b400ab232df49166ba2d668167e806e40502b76d0123036cf6f388`.
Independent replay plan SHA-256:
`748e4e318f1dfd9d48b99748bac06c2512b94a62464802080fc6c6250ed72ae4`.
Validation SHA-256:
`57a44ce3b24dccd7841d2b062ae1dc0a012e3a0408786f3d1fccdbfaa11502df`.
Exact successful controller records and source snapshots are retained. Observer
binding/CLI corrections affected evidence collection only; each model job ran
once. The consumed cost prospectus was archived with its original hash before
the mutable issue was updated; the fit freezes its own private prospective spec.

The 58-study cohort has been reused, and patient groups are not independently
attested. No local result establishes D4 complementarity, generalization to the
private leaderboard, or first place. The primary advancement gate fails, so
native replay confirms reproducibility without selecting a candidate.
