# 47: Evaluate complete public frontier inference

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 45

## Purpose

Reproduce the accessible public four-reader CoAt ensemble before preparing any scored comparison. Preserve its exact source recipe, attribution and component identities; reject a superficially complete CSV when a reader or study failed. This is a separate public-baseline route and does not change Ticket 39's frozen scientific choices or select a final candidate.

## Requirements

- [x] Resolve source-code license with primary evidence and retain attribution, license text and changed-file notice.
- [x] Prepare immutable visible-only two-T4 request, exact 23 upstream code cells, strict receipt/output validation, zero spend and bounded execution.
- [x] Verify rejection of omitted readers, fallback/neutral predictions, changed reduction/checkpoint epochs and missing prediction artifacts with focused tests.
- [x] Wait for exact submission 56705536 to become terminal and reconcile its GPU exposure. Validate current free quota, disabled paid scaling, one active GPU request at a time and campaign capacity before launch.
- [x] Launch the prepared request once, preserve canonical handle/version and validate exact remote code/metadata.
- [x] Inspect terminal output, all four reader receipts, complete component prediction artifacts and exact ordered finite submission. Charge both assigned GPUs, retaining unknown VM exposure conservatively.
- [x] Record disposition, runtime/memory limitations and a separate full-workload scoring proposal only if full inference passes.

## Prepared request

`artifacts/private/public-frontier-adoption/gpu-inference-preflight-v2/request.json` freezes **two T4s**, **1,800 wall seconds / 3,600 GPU seconds** reserved, a **900-second code watchdog**, **at most three visible studies**, **no training**, **no hidden scoring**, and **zero external spend**. CLI `kernels push` must include `--timeout 1800 --accelerator NvidiaTeslaT4`; the upstream pinned container digest and input attachments are retained. Retries may only address diagnosed operational failures, at most three launches, each separately admitted into the remaining campaign capacity. No GPU reservation or launch has been issued at this checkpoint.

`scripts/prepare_public_frontier_gpu_preflight.py` embeds all 23 source code cells unchanged and separately compiles/executes them, retaining each original future import. It adds visible-cohort/device checks, execution limits and `scripts/public_frontier_validation.py`. Any validation failure removes `submission.csv` and fails the run. The complete upstream markdown credits are retained alongside Apache text and the adaptation notice.

The validator requires twenty DINO member prediction artifacts, five A5 folds, four Raptor views, all four CoAt family members, their exact epoch groups, equal 25% within-family weights, the original 40% CoAt / 60% Raptor inner mixture and probability-mean reduction. All reader receipts must match the exact publisher success-status constant and report zero fallback studies; recorded failed, neutral-filled or dropped predictions are rejected. It verifies component prediction hashes, finite arrays and exact UID/label order, child logs for dense preparation fallbacks, and final output schema/probability bounds. Successful precision retries that retain complete finite predictions are allowed. Memory sampling uses device-wide `nvidia-smi` and the exposed cgroup every two seconds, covering child readers where the platform exposes these measurements; parent PyTorch/RSS metrics and sampling limitations are separately named.

## Evidence

The prior CPU check is [Ticket 46](46-validate-public-frontier-assets.md), resolved: four pinned manifests, 107 files and eleven checkpoints verified. The exact [D4 source page](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946?scriptVersionId=354057270) renders **Apache 2.0**, **Version 2 of 2**, **Best Score 0.943 V1** and **332.1 seconds on two T4s**. This is source/author provenance, not this account's score or hidden-runtime evidence.

Rendered source, Kaggle terms, current competition rules and official Apache license are retained at `artifacts/private/external-notebook-audit/frontier-2026-09-30/license-use-audit/`; its `license-use-evidence.json` pins every file hash. Prepared bundle verification checks these hashes, official license text, exact upstream code cells and immutable request/artifact digests. See [frontier research](../../../docs/research/public-frontier-options-2026-09-30.md).

2026-09-30: **12 focused tests passed** in `tests/test_public_frontier_validation.py`; generated notebook compiles, all 23 embedded cells match upstream source exactly, and `git diff --check` passes. Scoring submission **56705536** is still authenticated **PENDING** with an active monitoring service. Do not treat waiting time as terminal allocation evidence or launch a competing GPU request.

The initial prepared request v1 is preserved and superseded by v2 before any launch. V2 strengthens child-status validation against three downloaded publisher runtime files and the D4 manifest (all hashes match the CPU-verified inputs), and adds memory sampling for child processes. The model recipe, workload limit and resource reservation are unchanged. Preparation revisions consumed no GPU execution attempts. Source-status evidence is `reader-runtime-audit/status-identity-evidence.json` under the frontier audit.

Primary Kaggle version/source APIs independently connect author submission **56696090**, score **0.943**, to v1 script version **353578144**. The scored v1 and frozen v2 have identical source strings for all 23 code cells and identical eighteen attachment version IDs; v2 adds only its first attribution/correction markdown cell. Exact old source and comparison hashes are retained in `d4-public0946-v1/code-version-comparison.json`. This resolves the version-to-score provenance gap while leaving our own performance unmeasured.


### 2026-09-30 — first GPU execution failure and diagnosed repair

Submission 56705536 completed with public score **0.936** and its exact terminal accounting was reconciled. Prepared request v2 launched once as canonical notebook version **1**. Remote code and all attachments matched; private, internet-off, two T4s, pinned Docker identity. It ended **ERROR** with `OSError: source code not available` in `inspect.getsource` while exporting CoAt input helpers. DINO/A5/Rad/Raptor ran; CoAt did not. No submission CSV survived the strict failure handler. The request cannot be considered complete inference.

Normalized authenticated quota increased **190.753 wall seconds** with zero outstanding quota reservation and unchanged refresh identity. Sequential request accounting rounded up for both assigned GPUs: **382 GPU seconds** charged. `gpu-inference-preflight-v2/failure-accounting.json` binds exact request, terminal state, quota snapshots and failure. Remaining campaign capacity after reconciliation: **3,892 seconds**.

`tests/test_public_frontier_source_execution.py` executes the generator's actual cell loop and reproduces the exact `inspect.getsource` failure without model execution. Both cases failed before the repair. Registering exact cell text in Python's `linecache` under lookup-compatible virtual filenames fixes source export; all **14** focused source/receipt tests pass. Model code, recipe, checkpoint identities, strict validation and workload limits are unchanged. Original failed request and logs remain preserved.

A separate immutable **v3** request, SHA-256 `7a24c3010d53b233c12c85656aece36c27c4cf5e5dcbd5cdd010e440036c5adb`, was admitted under the owner's existing broad zero-spend authorization after diagnosis. It reserves **3,600 GPU seconds**, exactly two T4s, 900-second code watchdog, 1,800-second platform limit, at most three visible studies and no scoring. It was pushed once as canonical notebook version **2**; source and terminal validation remain required. This is a diagnosed operational repair, not a blind retry or new scientific recipe. Campaign headroom while reserved: **292 seconds**.


### Corrected dry run completed and independently verified

Canonical notebook version **2** completed. Request v3 retains exact remote code, attachments, private internet-off two-T4 environment and pinned container. All four readers executed with the exact publisher success statuses, epoch groups, equal within-family weights and unchanged 40% CoAt / 60% Raptor mix. All **30** saved prediction artifacts were independently hash-checked: twenty DINO members, A5, parent after Rad, four CoAt readers, final predictions and three five-head Rad arrays. Exact study/label order, shapes, finite values, CSV bounds and final prediction/CSV equivalence passed. Child logs contain no dense-preparation fallback. This is three-study visible evidence, not hidden-runtime or performance evidence.

Code elapsed **340.948 seconds**. Whole-device two-second sampling observed peak GPU usage **4,663,017,472 / 4,962,910,208 bytes**, cgroup peak **22,391,545,856 bytes**, 164 samples, no sampler errors. Sampling may miss peaks and container hierarchy limits host-memory scope. Parent-only measurements are separately preserved.

Normalized authenticated quota increased **362.004 wall seconds** during the sole active campaign request; charged conservatively as **726 GPU seconds**. No VM epoch interval was inferred. `gpu-inference-preflight-v3/validation.json` and `visible-run-validation-and-accounting.json` preserve independent checks and quota/terminal/source hashes. Campaign capacity: **68,834 charged, zero reserved, 3,166 remaining** of 72,000 GPU seconds.

A separate full-workload bundle is prepared at `artifacts/private/public-frontier-adoption/scored-baseline-v1/`. It preserves the exact embedded upstream model cells and strict checks, replacing only the visible guard and execution envelope. One visible save reserves **2,400 GPU seconds** (two T4s, 1,200-second platform limit, 900-second code watchdog), fitting current capacity. Hidden scoring would require **57,600 GPU seconds** separately; it is not launched or authorized by that save. Stronger exact-submission allocation telemetry was not recovered for the earlier baseline, so its conservative charge remains unchanged.

Full-workload visible save accepted as canonical notebook version **3**, with exact remote code/input/environment matching. Read-only service `rsna-public-frontier-scored-save-v1.service` monitors only that saved version. No competition submission was issued. Source/output validation and terminal charge remain required before scoring.


## Answer

The corrected complete visible inference passed, and the separately saved full-workload **version 3** is **COMPLETE** and independently verified. Its **30** prediction artifacts and all strict reader/output checks passed. The visible `submission.csv` is **byte-identical** to the passed preflight; model recipe, checkpoints and blend remain unchanged. Saved-version code elapsed **238.288 seconds**, with an isolated normalized quota increment of **256.763 wall seconds**, charged as **514 GPU seconds**. No hidden scoring was issued.

Campaign terminal capacity is now **69,348 charged**, **zero reserved**, **2,652 remaining** of **72,000 GPU seconds**. Exact saved version, source/metadata proof, complete output, immutable accounting and `competition-submission-proposal.json` reside under `artifacts/private/public-frontier-adoption/scored-baseline-v1/`. Scoring proposal SHA-256: `3490b9fcba9381e4778c70f4e2a4597ff660c7eb7e74dc3f8b0f446ea763317c`. A single 57,600-second hidden reservation would bring charges plus reservation to **126,948 seconds**. The proposed total cap is **129,600 seconds / 36 GPU hours**, adding **16 GPU hours**, with zero added spend and paid scaling disabled.

**Disposition: complete visible reproduction; exact scoring comparison prepared, blocked on a separate campaign-budget decision.** The author-public 0.943 remains source provenance, not our score. All source and license notices remain intact. The prior scored baseline **0.936** remains available. [Ticket 51](51-score-verified-public-frontier.md) owns the exact resource decision and subsequent one-attempt scoring evaluation; this ticket resolves only the inference and scoring-proposal gates, with no final selection or first-place claim.
