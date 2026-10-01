# 58: Test report-supervised layer4 adaptation against matched head continuation

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 57

## Prospective hypothesis

Adapting the last ResNet-18 stage with broad admitted report supervision adds
useful image discrimination beyond another 1,500 updates of the same head.
Ticket 39's short reference-only adaptation did not test this supervision or
initialization. Ticket 55 supplies independently verified warm head states;
Ticket 56 establishes exact native/cached adaptation parity and plausible CPU cost.

## Fixed scientific choices

- Reuse all 4,228 admitted image studies and exact corrected three folds. Use
  only each fold's admitted reference/report training membership and masks;
  no held-out study or target entry enters an optimizer batch. Keep report labels
  distinct from the twelve supplied reference assessments.
- Require independently complete Ticket 57 activation and source/packing
  verification before freezing executable fitting requests. Cache generation
  fits nothing; adapted layer4 weights and heads are separate for every fold.
- Both arms start from the same fold's **Ticket 55 long report head** and the
  original ImageNet encoder. The warm head was fitted without that fold's
  held-out inputs. Record the shared initial head and encoder state digests.
- Control: frozen full encoder, head learning rate **1e-4**. Treatment: only
  layer4 and head train, layer4 rate **1e-5**, head rate **1e-4**. Adam, constant
  rates, no augmentation, all normalization buffers remain frozen in eval mode.
  Keep 224px whole-field inputs, single-slice grayscale replicated to RGB,
  fixed eight-slice/16-series packing and anatomical pooling. No Ticket 40
  adjacent-slice channels or new representation is selected.
- Each arm performs **1,500 successful updates**, each with **two reference plus
  two report draws**, existing deterministic batch seed **20260912** and report
  seed offset **34034034**. Use masked BCE mean over the combined supervised
  entries. Both arms receive identical ordered batches and target masks.
  Initialization seed **20261001** applies before strict parent-weight loading.
- Checkpoint every 500 updates for operational recovery; select only the fixed
  1,500-update final checkpoint. Do not evaluate or select intermediate checkpoints.
- The control may use the original unfitted pooled bank for speed only after
  verifying agreement with cached-prefix/frozen-layer4 reconstruction. Keep
  identical loss, optimizer, batch shape and head precision. Treatment consumes
  complete verified prefixes and the shared native layer4/pooling/head code.

## Evidence and disposition

Primary comparison: treatment minus matched head continuation on complete
**58-study** held-out predictions, macro ROC AUC across all twelve targets,
**2,000 paired study resamples**, seed **20261001**, all fold/target directions
and class support. Secondary fixed comparisons: each arm versus the Ticket 55
long report parent, and full image features versus zero encoded image features
with descriptors/masks preserved. No ensemble weights or budgets are swept.

Require all six successful final runs, source/fold/exposure identities, finite
losses/gradients, changed treatment layer4/head, unchanged earlier parameters and
normalization buffers, reload and independent replay within **1e-6** with frozen
16-thread/batch-two inference. Retain CPU/runtime identities. Verify actual native
MRI inference against the final adapted cached-prefix outputs before admitting a
delivery candidate; prefix-only replay cannot prove the full offline adapter.

A positive primary paired lower bound and positive directions in every fold,
with image-signal controls, permit provisional candidate review only. Otherwise
record inconclusive/rejected evidence without automatic promotion. The reused
development set and missing attested patient groups limit generalization claims.
No local result establishes first place or D4 complementarity.

## Resources and prerequisites

Use standing zero-spend authorization, with an exact separately frozen resource
decision after Ticket 57 completes. One nonexclusive CPU job, at most
**16 CPUs / 64 GiB / 6 hours**, explicit watchdog and no automatic requeue. Reserve
and measure complete six-run, checkpoint/replay and evaluation cost; partition
only if measured memory/runtime requires it. A prospective random-read/cache
load benchmark must establish execution cost and peak memory. Preserve consumed
attempts; failures need diagnosis and a separate recovery identity.

This CPU experiment adds no GPU request, model download, competition scoring,
publication, cap change or final selection. The separate exact D4 scoring request
56739912 failed with no score; its full unknown GPU exposure was reconciled.

## Requirements

- [x] Admit the complete verified activation bank and benchmark actual random
  reads/full-bank loading, memory and native reconstruction.
- [x] Freeze the exact executable experiment and concrete resource decision.
- [x] Complete six matched runs and independent final prediction/native replay.
- [x] Record primary/secondary comparisons, cost and candidate disposition.

## Comments

Choices registered October 1 before any adapted held-out predictions. Ticket 56's
four-report projection omitted full-cohort random reads, final evaluation and
checkpoint overhead. Actual costs must replace that projection before fitting.

## Answer — October 1

**Inconclusive/no promotion.** Adaptation macro AUC **0.689232** versus matched
head continuation **0.691387**, delta **−0.002155**, paired 95% interval
**[−0.022304,+0.019717]**, with fold directions **+0.052518 / +0.001877 /
−0.018148**. The primary gate fails. Both image-feature controls have positive
lower bounds, but another duration/adaptation gain is unestablished. No budget,
checkpoint, representation or ensemble weight was selected after these results.

Full-bank cost job **7986** completed `0:0`, **180 node seconds**. All 4,228
entries loaded in **140.500 s**, peak **38,800,142,336 bytes**; 128 random reads
took **4.539 s** on the warm/shared cache. Each fold's real gradient/scope probe
had exact frozen native logits; probe states were discarded. Conservative full
fit projection **5,676.858 seconds** supported one bounded 3-hour allocation.

Fit **7989** completed `0:0`, **1,982 node seconds**, all six fixed 1,500-update
runs. Verifier **7991** completed `0:0`, **321 node seconds**: all final
checkpoint states, target/source masks, training exclusions, ordered exposures,
optimizer budget, frozen earlier parameters and normalization buffers checked.
All **116 final predictions** and **58 warm-parent predictions** replay exactly
from original DICOMs with 16 threads / batch two; zero-image controls and all
five 2,000-resample comparisons also agree. Maximum prediction discrepancy **0.0**.
Total new CPU allocation **2,483 node seconds**, zero GPU/spend.

Fit plan SHA-256 `81da4c9de7b400ab232df49166ba2d668167e806e40502b76d0123036cf6f388`;
independent validation SHA-256
`57a44ce3b24dccd7841d2b062ae1dc0a012e3a0408786f3d1fccdbfaa11502df`.
The initial observer for fitting was bound to a guessed job number; it was
stopped and rebound to actual accepted 7989 without duplicate fitting. The
verifier observer initially used unsupported flags; corrected positional
launch retained actual successful 7991 terminal evidence. Both correction
receipts remain private. A prelaunch verifier contract check also corrected
its assumptions to the actual generic observer schema before freezing/launch.

The cost request had frozen this mutable issue. Its complete 104 input hashes
were rechecked and the original prospectus archived before this result update;
fit and verification already freeze separate immutable source/spec snapshots.
The development cohort remains reused, with no attested patient-group isolation.
No D4 complementarity, leaderboard gain, candidate promotion or final selection.
[Complete evidence](../../../docs/evidence/ticket-58-report-layer4-adaptation.md).
