# Ticket 69: portable medical inference mechanics

The additive CPU inference adapter is implemented and **26 synthetic tests plus
its source type check pass**. It is not yet a verified real-MRI delivery model.
Ticket68's actual fitted heads, native receipts, predictive disposition and
fresh package/runtime admission remain prerequisites. No new model forward on
real MRI, fitting, GPU/TPU job, upload, submission or promotion occurred here.

Implementation: `src/rsna_knee/biomedclip_inference.py`. Existing frozen Ticket
64/66/67/68 sources and snapshots remain unchanged. Current metadata shows the
first three feature partitions completed successfully; the fourth is running.

## Fixed prediction contract

The production vision loader requires the exact admitted checkpoint SHA
`52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`
and configuration SHA
`9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0`.
Restricted `torch.load(weights_only=True)` follows hash verification, strict
151 finite FP32 vision-tensor coverage and the official vision-only factory.
Exact admitted CPU Torch/TorchVision/timm/OpenCLIP/Transformers/NumPy/Pillow
versions are checked. No text tower or pretrained download is requested.

Every valid whole-field224 grayscale slice is repeated RGB, normalized with
pinned CLIP mean/std and encoded separately in FP32, with explicit unit512
projection. Invalid slots stay zero. Original boolean16×8 masks and ten
anatomical descriptors remain explicit; no D4 cropped336 cache substitution.

Three exact final2076→32→12 fold heads are loaded through independent SHA,
identity, feature-fingerprint, final-budget and restricted-load gates. Loaded
heads retain their fold and supervision arm; swapped folds and mixed arms are
rejected. CPU
constructor RNG is preserved; head dtype remains FP32 even if the caller changes
Torch's global default dtype. Head rows use the audited batch-two grouping,
including the last singleton. A selected held-out fold and deployed three-fold
**probability mean** are separate API modes. An average of logits is not used.
The deployed mean's development performance cannot be inferred by averaging
heads on Gold58, because two heads trained on each such study.

All-invalid studies retain their row and receive mandatory supplied eligible-
reference priors, with explicit fallback flags. The helper computes Laplace
alpha1 priors from binary/missing trusted reference rows supplied by the caller;
actual bundle construction must verify those rows belong to the eligible fold,
without substituting weak labels. No implicit arbitrary fallback is introduced.
The existing strict submission writer remains responsible for exact UID/target
order, complete membership and twelve finite bounded probabilities.

## Meaningful checks and remaining delivery gates

Tests exercise exact RGB normalization, one call per valid slice, invalid zeros,
unit vectors, bad/zero/nonfinite outputs, held-out-fold dispatch, probability
mean, fixed head grouping, all-invalid row preservation and priors, malformed
input/metadata/masks, missing or swapped heads, mixed supervision arms, bad
priors/architecture, SHA-before-load,
restricted loading, final identity/budget, RNG preservation and global dtype.
Typing passes for the new source module. These are mechanics tests using
synthetic fixtures; they do not establish actual medical checkpoint or MRI
adapter parity, runtime on Kaggle or competitive performance.

After successful Ticket68 native verification and provisional predictive gate,
separately freeze an all58 original-MRI adapter-equivalence request, exact
production head/prior/source/runtime bundle and fresh complete storage/calendar
cost. Original-MRI replay, representative/full-scale timing and the actual
Kaggle offline runtime must pass before any candidate use. The current Kaggle
container differs from the admitted CPU environment; compatibility is unproven.
No package or trained candidate was created by this ticket.

## Exact scored-anchor distinction

A read-only audit verifies all **five embedded prediction/audit cells** in the
local baseline against the exact saved source for **submission56705536 /
script354125287 /notebookv2**, whose retained terminal is COMPLETE /0.936.
The D4 notebook has **23** embedded cells; none of those complete source cells
is byte-identical to the scored baseline's complete cells. This alone does not
rule out reuse of individual matching fragments, but it cannot prove full-model
prediction parity. The existing58-study D4 cache is different lineage and is
not proof of the exact0.936 baseline's Gold58 predictions.

Private source-audit receipt `artifacts/private/ticket69-scored-anchor-distinction-v1.json`,
SHA `d01bc8f53dd58767b9b0e6b7adad9750415134222622b510d0b1d7d383580be5`.
No new inference or AUC was computed. A later paired candidate comparison must
obtain verified same-cohort predictions from the exact scored notebook, or
independently prove an exact matching component reconstruction. Its GPU launch
requires the existing accelerator request to be terminal, sufficient real free
quota and a complete separately frozen resource decision. No cached D4 result
is relabeled as scored-baseline evidence.
