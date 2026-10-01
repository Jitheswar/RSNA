# Ticket 53 public meniscus: reject the fixed development overlay

The single registered 10% Medial Meniscus vote lowered development macro AUC.
**Reject this overlay; no promotion, weight search or scoring request.** The
campaign objective remains first
place on both public and final private leaderboards; neither rank is established.

## Completed paired result — October 1

| Development comparison | Anchor | Fixed overlay | Paired delta | 95% study-bootstrap interval |
| --- | ---: | ---: | ---: | --- |
| Twelve-target macro AUC | 0.951479 | 0.951179 | −0.000300 | [−0.001226, +0.000348] |
| Medial Meniscus AUC (26 positive / 32 negative) | 0.974760 | 0.971154 | −0.003606 | [−0.014709, +0.004181] |

All 2,000 resamples are defined, seed 20261001. All three subgroup fold macro
deltas are zero; these are subgroup evaluations, not fold-specific retraining.
Eleven other columns are exactly unchanged. The 58 reference studies and public
parent selection have been reused: these results are development evidence,
not clean OOF, an untouched generalization estimate or a competition score.

Private offline notebook `jitheswar/rsna-d4-meniscus-paired-development`, saved
version **1**, completed at **07:57:52 UTC**, code time **1,244.793 seconds**
on two T4s, **21.462 seconds per development study** including shared setup.
All 23 anchor cells remain byte-identical. Independent validation confirmed the
actual CPU-admitted pixel/mask hashes, the disk-cache branch, strict specialist
weights/runtime, all four CoAt readers and **27 finite component prediction
artifacts**. There were **zero fallback events**. Specialist inference itself
took **72.669 seconds**. Peak reserved parent GPU memory was **7.288 GB** and
**7.074 GB**; parent peak RSS **8,372,340 KiB**. These measurements do not
establish hidden-scale throughput or whole-VM memory.

The exact accepted source is private, offline and pinned to the reviewed
container. The metadata API canonicalizes explicit dataset `/2` to its slug;
actual mounted manifest/payload hashes verify version-2 contents. Its numeric
mounted-version association was not separately observed. Raw source captures
and the corrected interpretation remain retained. A first validator assumed
twelve raw specialist columns; the actual contract has two meniscus columns.
That local validator correction did not change the executed GPU recipe.

Before this launch, exact frontier submission **56739912** was terminal with
a hidden-rerun error and no score; its full unknown **57,600 GPU seconds**
were conservatively charged. Standing owner authorization supported the concrete
**36→38-hour** cap amendment and one **5,400-GPU-second** reservation. The sole
paired run's normalized free-quota increment was **1,263.767 wall seconds**:
rounded **1,264 wall / 2,528 GPU seconds** charged, zero reserved afterward.
No controller epoch interval is invented. Ledger: **129,637 charged / 136,800
cap**, **7,163 seconds** unreserved, $0 external spend.

Independent analysis job **7990** completed `0:0`, **10 CPU-node seconds**,
and recomputed the one fixed vote and registered paired comparisons. Private
root: `artifacts/private/public-meniscus-adoption/paired-development-v4/`.
`final-disposition.json` SHA-256:
`ae504b28fc25291b94b342fa3b19282dcc1a9d1a68e20c3c3dc4050291960232`.
The 0.936 and 0.498 scored fallbacks remain preserved. This successful accessible
run does **not** reproduce the hidden scoring failure in Ticket 59.

## Historical admission and prospective preparation

The following sections preserve what was known before prediction execution.

The private notebook `jitheswar/rsna-public-meniscus-cpu-preflight`, saved version
**1**, completed at **2026-10-01 03:35:40 UTC**. The accepted request pins
`renta0426/rsna-knee-public0033-meniscus-bag-v1/2`. All eight prediction-bearing
files match the reviewed manifest, including the **347,965,736-byte** checkpoint
SHA-256 `e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7`.
The complete **233-tensor FP32** state strictly loaded with the publisher's model
definition. Elapsed code time **42.4929 s**, peak RSS **1,611,292 KiB**, assigned
GPUs **0**, external spend **$0**. No forward inference, fitting or submission ran.

Independent verification checked the terminal response, exact saved source,
private/internet-off/CPU settings, every retained file hash, all state shapes and
dtypes, twelve output heads, and declared 4,349-study report-training cohort with
all 58 reference studies excluded. The manifest retains `fold_safe_oof=false`.
Dataset source ID **19356288** is the expected version association observed in
the public Roman attachment; this CPU check establishes the accepted `/2` request
and actual matching payload bytes, rather than separately observing its numeric
mount source ID. [Publisher and primary-source provenance](../research/public-frontier-accessibility-update-2026-09-30.md).

Private evidence: `artifacts/private/public-meniscus-adoption/cpu-asset-preflight-v1/`.
Request SHA-256 `4d8bb0998db39516dd665da68a0883aec5b21ef026d4d6c4bfa33a37c86de28f`;
output SHA-256 `2efa788b0d0454811b103da8e3da2008e439c046be24cce5b49e9db9c44f2948`.
The observed container uses torch 2.10.0+cu128, Transformers 5.0.0, safetensors
0.7.0 and NumPy 2.0.2. CPU loading alone does not measure two-T4 activation memory,
hidden execution cost, model parity or predictive benefit.

## Actual real-MRI preprocessing

Private offline CPU notebook `jitheswar/rsna-meniscus-gold58-cpu-preflight`,
saved version **1**, completed at **2026-10-01 04:15:33 UTC**. It executes the
frozen anchor's preprocessing definitions and uses the one native pixel group
shared by all 20 DINO members. The downloaded public manifest matches the
earlier admitted manifest SHA-256
`496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844`.
No model forward pass, fitting or competition submission ran.

All **58 real studies**, **336 header series** and **273 selected slot-series**
completed in **72.2027 s**, with **0 decode failures and 0 fallback events**.
Every study has **3–6 present slots**. The `(58,6,12,336,336)` uint8 disk cache
contains **471,453,696 bytes** and passes the publisher's exact specialist input
contract. Its bytes and the float32 `(58,6)` mask remain unchanged after the
anchor unlinks the cache path. Peak RSS **2,486,636 KiB**, assigned GPUs **0**,
external spend **$0**. Terminal status, exact saved code, private/offline/CPU
settings, cohort/metadata identity, coverage and retained hashes were
independently validated.

Private evidence: `artifacts/private/public-meniscus-adoption/cpu-mri-preflight-v2/`.
Request SHA-256 `6379a8e185ab2328471fdec9dab1aacc660537ce7104f654a9e1b4e453a8fbbc`;
output SHA-256 `73ef3f9646756245273a37c5e9f0534996e9c0846e6dc01cfa2b824044284694`.
Pixel SHA-256 `05487fb92372ee865188c2eb96805a4b6c4f9bc6ff40df174fe6e5d584446ece`;
mask SHA-256 `a2f7cf59ca798a60761d63a4bf3db2b317bc6647999699402873e3559bb0fdd0`.
The v1 preparation missed an annotation import; local contract execution caught
and corrected it before any remote launch. Only draft v2 was executed, as saved
remote version 1. [Autonomous CPU decision](../../.scratch/rsna-leaderboard/resource-approvals/ticket53-real-mri-cpu-admission-20261001.json).

## Historical frozen experiment

One paired Gold58 inference request is prepared, without launch, at
`artifacts/private/public-meniscus-adoption/paired-development-v4/`. Request
SHA-256 `0e490a080a5c51c1aeaaab4e67da5673129d47ebec22a48173eecf0e3fa78ed6`.
All **23 existing D4 anchor cells** retain their exact hashes. An explicit private
input view maps only the 58 reference training studies into the test interface;
labels and reports do not enter the prediction view. Original training CSV bytes
and ordered reference membership are pinned before inference.

The cache adapter retains the completed `build_cache` return, including a memmap
whose directory entry the anchor unlinks. It passes the exact arrays and masks
to the publisher's immutable specialist runtime after DINO completes, then
releases them. This avoids the published NumPy-allocation hook, which cannot
observe our anchor's allocator seam. The experiment deliberately forces the disk
branch. It requires one complete matching cache and makes specialist failure fatal.
It also requires the actual CPU-admitted native pixel and mask hashes before
specialist inference. Draft v3 was superseded before launch to bind this new
real-MRI evidence; all 23 source cells, the model and scientific vote remain
unchanged. Actual GPU pixel parity is still unmeasured.

The fixed Medial Meniscus vote is **0.10 Transformer + 0.80 CoAt/Raptor + 0.10
specialist**, with the anchor's average-tie percentile ranks. Eleven other columns
and study order remain exactly equal to the loaded anchor. Route verification
allows only `2e-15` absolute CSV roundoff: an actual 58-row pandas roundtrip changed
21 ranks by at most `1.11e-16`. The first exact-equality guard rejected this known
anchor behavior, its regression failed before the correction, and preparation v1
was preserved as superseded. No scientific blend variant was evaluated or chosen.

**12 focused tests passed**. They exercise exact unchanged columns and inputs,
known fixed-vote outputs, malformed/nonfinite/missing/reordered predictions,
dataframe-index independence, actual CSV roundoff, ambiguous caches, and a sparse
**>1 GiB** cache that remains readable after the anchor unlinks it. Successful
and failed specialist calls preserve the tested Python/NumPy/PyTorch CPU random
streams; the wrapper also scopes both CUDA random states for the eventual GPU
run. Its model initialization therefore cannot advance the anchor's random
streams. Draft v2 was superseded before launch to add this isolation; the
scientific vote never changed. Actual CUDA parity remains unmeasured. The frozen
development notebook compiled and all 23 source-cell identities were independently
rechecked. No GPU or prediction result is claimed by these tests.

The resource proposal is two free T4s, **2,700 s** server timeout, **2,400 s**
watchdog, **5,400 GPU seconds**, one attempt, $0 spend and no fitting or scoring.
A concrete **36→38 GPU-hour** amendment is proposed under the standing owner
authorization. No cap amendment, GPU reservation or launch has occurred. The
existing accepted scoring request **56739912** was independently observed
**PENDING**, without scores, at **04:33:19 UTC**; its observer remains running
and its 57,600-second reservation remains intact.

Before launch, reconcile that exact reference, verify no GPU request across
backends, refresh free quota and paid-scaling settings, and record/apply the
concrete resource decision. The standing October 1 authorization permits this
decision without another approval question. Analyze the one fixed pair with
2,000 study bootstrap resamples, per-target support and fold directions. Gold58
has been reused and public parents were selected on it; this is development
evidence, not clean OOF or an untouched estimate. No public-score improvement,
final selection or first place is established.
