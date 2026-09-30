# 47: Evaluate complete public frontier inference

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 45

## Purpose

Reproduce the accessible public four-reader CoAt ensemble before preparing any scored comparison. Preserve its exact source recipe, attribution and component identities; reject a superficially complete CSV when a reader or study failed. This is a separate public-baseline route and does not change Ticket 39's frozen scientific choices or select a final candidate.

## Requirements

- [x] Resolve source-code license with primary evidence and retain attribution, license text and changed-file notice.
- [x] Prepare immutable visible-only two-T4 request, exact 23 upstream code cells, strict receipt/output validation, zero spend and bounded execution.
- [x] Verify rejection of omitted readers, fallback/neutral predictions, changed reduction/checkpoint epochs and missing prediction artifacts with focused tests.
- [ ] Wait for exact submission 56705536 to become terminal and reconcile its GPU exposure. Validate current free quota, disabled paid scaling, one active GPU request at a time and campaign capacity before launch.
- [ ] Launch the prepared request once, preserve canonical handle/version and validate exact remote code/metadata.
- [ ] Inspect terminal output, all four reader receipts, complete component prediction artifacts and exact ordered finite submission. Charge both assigned GPUs, retaining unknown VM exposure conservatively.
- [ ] Record disposition, runtime/memory limitations and a separate full-workload scoring proposal only if full inference passes.

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
