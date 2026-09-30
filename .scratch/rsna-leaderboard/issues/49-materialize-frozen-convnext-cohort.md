# 49: Materialize frozen ConvNeXt cohort

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)
Blocked by: 48

## Purpose

Extract one frozen ConvNeXt Tiny representation for the exact 4,228-study Ticket 36 image cohort, enabling a separately planned matched head experiment. The 32-study pilot passes the predefined CPU throughput gate; it does not establish predictive improvement.

## Requirements

- [x] Freeze exact checkpoint/runtime/source/input identities and two disjoint balanced study partitions before execution.
- [x] Admit two CPU allocations of at most three hours each, one node / sixteen CPUs / 64 GiB each, no more than two concurrent jobs, no GPU, no new checkpoint download and no execution-time network; zero spend and no training.
- [x] Read the existing report series cache with checksum validation; decode only the existing 58 gold images against the admitted archive. Preserve deterministic 224-pixel whole-field packing, eight slices per series, masks and descriptors.
- [x] Produce complete finite float32 768-dimensional features and retain source provenance and exact prior mask/descriptor parity for every study. Reject missing, duplicate or changed membership and preserve terminal controller evidence.
- [x] Publish a complete immutable feature manifest, actual allocation evidence and measured cost. No candidate promotion; matched gold/report head training requires a separate frozen plan.

## Evidence

[Ticket 48](48-measure-frozen-convnext-cpu-feasibility.md): 32 studies / 1,464 slices completed in 34.6 seconds; twofold prepared-cohort estimate 1.4532 node hours. Gold decoding and queue delay are excluded from that estimate. The two three-hour requests reserve at most six CPU node hours, below the proposed 24-node-hour extraction ceiling, and must finish by October 3. Apply the existing owner's zero-spend campaign authorization only after the exact new plan is frozen. This does not alter Ticket 39 or expand the 72,000-GPU-second cap.

## Comments

2026-09-30: Exact extraction plan SHA-256 `ca2adc77d18f78652f640fd08adced9476321870a6304159d61e071470877f62` is frozen under `artifacts/private/ticket49-convnext-feature-cohort/`. Two disjoint partitions contain 2,114 studies each and 93,640 / 93,632 selected slices. Jobs **7935** (partition 0) and **7936** (partition 1) were submitted once under the existing owner's zero-spend authorization; each requests 16 CPUs, 64 GiB and at most three hours, with no requeue. Maximum reservation is six CPU node hours, no GPU seconds. Read-only controller monitoring services retain terminal evidence independently of this conversation.

The isolated admitted CPU runtime and checkpoint are reused from Ticket 48. The source snapshot is frozen; checksum checks cover the checkpoint, runtime package files, prior manifests and extraction source. Every study must reproduce the prior source identity, slice/series masks and descriptors. Features publish atomically and cannot overwrite existing artifacts. No head training is authorized by this extraction request.

Fresh `du -sb` measurement of the artifact root plus source archive totals **529,589,614,556 bytes**; the estimated additional feature footprint is under 2 GB and remains below the 2 TB storage cap. Additional existing campaign roots remain recorded in the campaign storage ledger. Python compilation, Bash parsing and `git diff --check` passed before launch. Exact scoring submission **56705536** remains PENDING; the active GPU reservation and cap remain unchanged.

## Answer

Both extraction jobs completed successfully: partition 0 job **7935** produced 2,114 studies and partition 1 job **7936** produced 2,114 studies. Code elapsed **1,527.19 / 1,497.56 seconds**, including all 58 gold studies decoded from the admitted archive. Zero GPUs, no training, no network during execution and zero spend.

Independent CPU verification job **7937** completed in **37.80 code seconds**, hash checking every new feature file and every corresponding prior representation file. All **4,228 studies / 187,272 selected slices** have finite float32 `(16,8,768)` features, zero padding, complete source identity and exact prior slice-mask / series-mask / anatomical-descriptor parity. The manifest is `features/feature-manifest.json`, SHA-256 `e0cb3d624458a0f339c0c962b88dc670b3ecb6bf2b935fabb5342829a34b3d45`.

The unfitted anatomical pooled bank `(4228,3100)` is `verification-v1/pooled-cohort.npz`, SHA-256 `12fca5ca4af6277ff3e6f64fe147f10e2aaec355d297a2e8793d61507c7ea97e`. It preserves the same plane grouping and metadata; prepooling equivalence and descriptor-preserving ablation passed focused tests. Verification result is `verification-v1/verification-result.json`; terminal records for all three jobs are preserved by controller monitors.

Independent verification had a separate exact 15-minute CPU admission, request SHA-256 `eb32a3596b3aadd966a6ec988f007f00642abd9be4786c0cf569b5a0a30567f5`; combined extraction/verification worst-case reservation 6.25 CPU node hours remains below the 24-node-hour extraction ceiling. Original frozen extraction plans and feature outputs were preserved. This completes feature preparation and supports separately planned head training in Ticket 50; it does not establish predictive improvement or competition delivery.
