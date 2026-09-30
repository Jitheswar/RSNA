# Compute and reusable-artifact inventory

Recorded 2026-09-25, Asia/Kolkata. Read-only inventory for **Inventory usable compute and existing reusable artifacts**. Scheduler state is a momentary observation, not a reservation. No jobs, model downloads, resource changes, patient-level output, or tests were performed. This document contains aggregate facts only.

## Planning conclusions

CPU diagnostics can use the existing authorized ceiling: one nonexclusive `cluster1-cpu` node, at most 16 CPUs, 64 GB requested memory and six hours, with at most two concurrent autonomous CPU jobs. The scheduler displayed idle CPU nodes, but this is not a guaranteed start time. GPU experiments need a concrete approved resource request; no GPU-hour entitlement or queue duration can be inferred. Prefer one small experiment at a time and retain a CPU/frozen-feature fallback.

The usable historical image-training scope is **158 studies**, including all 58 reference studies and 100 report-only studies. Complete metadata and report extraction do not establish complete decoded image caches for all 4,407 studies. The current 128-pixel representation cannot establish readiness for a 224-pixel pretrained experiment. Plan a bounded new preprocessing/feature cache with explicit membership and checksums before a larger weak-supervision campaign.

## Scheduler observations

Queries used: `sinfo -h -o '%P|%a|%l|%D|%t|%G'`, `scontrol show partition` for the three relevant partitions, `scontrol show node 'KLEF-G-NODE[1-3]'`, and `squeue` aggregate partition/state/reason/GRES fields. Other users' names and job identifiers were not requested or retained.

| Resource | Observed state | What the proposal can assume |
| --- | --- | --- |
| `cluster1-cpu` | UP; ten nodes, 640 configured CPUs; eight nodes idle, one mixed, one allocated | CPU route exists. Repository ceilings remain binding even though partition MaxTime/MaxMem are unlimited. |
| `cluster1-gpu` | UP; one 64-CPU node with `gpu:4`, all CPUs allocated; two running jobs, three pending | Main GPU capacity was occupied. Pending reasons included resources, priority, and unavailable/reserved nodes. No start-time promise. |
| `cluster1-gpu-add` | UP; two 64-CPU nodes, both mixed; one exposes `gpu:1`, the other custom `vllm_gpu1:1,vllm_gpu2:1`; one running job and two pending in this partition | Custom GRES is not interchangeable with a whole GPU. Verify allocation/device/memory separately. |
| Owner's queue | No rows from `squeue -u ch.jitheswar` | No active/pending owner job was observed; this grants no priority. |

Each GPU node reported 257,000 MB configured host memory. The main four-GPU node reported only 1,790 MB instantaneous free host memory; do not use free memory as a granted limit. All three relevant partitions report unlimited maximum time and memory, `OverSubscribe=NO`, and `PreemptMode=OFF`. These settings do not establish whole-device isolation, actual VRAM, quota, or freedom from contention. The scheduler GRES strings do not identify physical GPU models.

`sacct` historical accounting was attempted for the owner from September 10 onward, but the Slurm accounting database connection was refused. Accordingly this inventory cannot calculate recent queue wait distributions, successful GPU-hour totals, failure rates, or forecast allocation availability.

## Actual prior GPU evidence

The [bounded GPU experiment evidence](../evidence/ticket-13-gpu-experiment.md) records one approved smoke run on September 14: `cluster1-gpu-add`, one generic GPU on KLEF-G-NODE2, **NVIDIA A30X**, 25,337,004,032 bytes assigned VRAM, driver 580.126.20 / CUDA 12.6, sixteen CPUs, 64 GB host RAM, a thirty-minute limit. The job completed in **7 minutes 13 seconds**, with peak VRAM **487,037,440 bytes** and zero retries. It loaded 38 training and 20 held-out studies but performed only **one optimizer step** of the mechanics model. This is proof of that exact environment/allocation, not full model training or an A40 guarantee.

The repository CPU environment and separate Python 3.11 CUDA environment are documented in that evidence. Revalidate dependencies and device facts for a new architecture. A single-GPU proposal within sixteen CPUs/64 GB is plausible as an initial shape, but the duration, batch size, and memory fit require the selected model's bounded preflight. Shared/custom GRES cannot silently substitute for the verified whole-device shape.

## Retained inputs and supervision

| Artifact evidence | Observed aggregate | Reuse boundary |
| --- | --- | --- |
| `artifacts/private/ticket-03-inventory-run/public/inventory.json` | 4,407 studies, 24,371 series, 819,078 readable DICOM headers, 58 reference-labeled and 4,349 report-only studies; zero missing/extra series or identity conflicts; inventory records full-corpus reconciliation | Header/source reconciliation, not pixel decoding or preprocessed-cache completeness. Archive identity metadata records 265,018,885,676 bytes; this inventory did not rehash/read the archive. |
| `artifacts/private/ticket-03-inventory-run/public/selection.json` | 58 gold plus 100 report-only studies, 965 series, 31,486 selected files | Representative selection, not full corpus. Historical unsupported-syntax counters are not a substitute for later decoder admission. |
| `artifacts/private/ticket-05-real-training-input/private/integration-manifest.json` | 158 study entries, 965 series; existing policy uses 128×128 float32, nearest resize, min/max normalization and up to 16 series × eight sampled slices | Membership and configuration were read; raw pixels and existing caches were not revalidated. This representation is not a 224-pixel pretrained cache. |
| Historical `artifacts/private/ticket-04-real/private/fold-manifest.json` and corrected `artifacts/private/ticket-04-real-v2/private/fold-manifest.json` | Root follow-up verified schema v1 and v2 respectively; the historical inventory records 4,407 memberships and three folds | Use the corrected v2 manifest for new comparisons and verify its exact checksum against the run design. Existence of the older manifest does not make it the active split. |
| Retained deterministic extraction public evidence and private `report-labels.json` under the existing Ticket 15 run directory | Private label artifact exists (145,697,605 bytes); public evidence reports all 58 held-out gold reports, zero missing; report-only training fold occurrences total 8,698, with zero missing training reports in each fold | Complete extraction coverage means every eligible report was processed, not every target was labeled confidently or correctly. No report text was printed or transmitted. |
| Ticket 15 report-only aggregate | Eligible report-only counts 2,847 / 2,965 / 2,886 across folds; supervised target-entry counts 3,333 / 3,396 / 3,337 | These are fold occurrences/target entries, not unique fully labeled studies. Abstention masks must be retained. |
| [Actual weak-supervision run](../evidence/ticket-17-weak-supervision.md) | Fold 0 admission had 2,885 training studies, but actual image scope had 38 reference plus 65 report-only studies; 2,782 report-only studies excluded by image scope | The prior weak training experiment did not exercise the full eligible report corpus. |
| Ticket 16 corrected aggregation and Ticket 18 additive design retained outside repo | Corrected learned-extractor aggregate exists; the frozen additive design explicitly sets Astra admission false and binds Ticket 15 label/source digests | Presence of learned outputs does not authorize their training use or establish a complete admitted replacement. |

The public-safe [research/reproduction report](../../../RSNA-ticket-27-research-and-reproduction-report.md) records deterministic extractor mean target coverage 0.1121 on gold reports, with selective accuracy defined on only eleven targets. Labels therefore need target-level admission; there is no basis for treating all 4,349 reports as fully trusted twelve-label training rows.

The immutable candidate manifest and prior real CPU training/representative inference evidence remain available in `artifacts/private`. This inventory did not rehash weights or recursively validate all protected derivatives. There is no fresh claim of full-corpus extraction/cache completeness, checksum validity, or new-resolution readiness.

## External weights and storage

The conventional `~/.cache/huggingface/hub` and `~/.cache/torch/hub/checkpoints` directories do not exist in this account at inspection time. No already-admitted ResNet-18 or DINOv2 weight artifact was identified in the bounded inspected locations. This is not proof that no such weights exist anywhere; no whole-home or shared-filesystem recursive search was performed. Existing mechanics candidate weights are separate from external pretrained weights. Frozen-feature fallback therefore still needs the chosen weights' admission and an initial extraction pass; it is not ready merely because CPU training is available.

`statvfs` reported **92,157,849,370,624 bytes total** and **24,480,788,209,664 bytes available** on the filesystem hosting the project (approximately 92.16 TB and 24.48 TB decimal); `df -h` rounded this to 84T total/23T available. This is shared free space, not reserved project headroom. The project soft budget remains **2 TB decimal**. No current complete project-usage ledger was established by the bounded metadata inspection; a heavy recursive disk scan was deliberately not run on the login node. Consequently **remaining room within the project's 2 TB limit is unknown**, even with substantial shared free space.

Before cache expansion, account for retained archive/raw files, current derivatives, temporary atomic-write duplication, environments and candidates. A later bounded compute-node usage/accounting pass can close this operational gap. Protect originals and candidates; no deletion is implied. For scale context only, the separate [image-options report](leaderboard-image-learning-options.md) estimates approximately 325 MB for one hypothetical 4,407-study, 48-slice, 384-feature float32 embedding set; actual cache size and preprocessing cost still depend on selected membership and representation.

## Enough to formulate the initial proposal

1. Keep the first real learning diagnostic within the already exercised 58-reference/158-study image scope, existing CPU ceilings, frozen folds and protected candidate. Confirm raw/cache checksums for the exact selected input before running.
2. Specify a bounded single-GPU smoke request for the chosen admitted model, with an explicit maximum time and actual device/isolation checks. Use the A30X smoke only as historical compatibility evidence; never budget nominal 45 GB A40 memory as guaranteed.
3. If the queue prevents that request from starting, retain a CPU diagnostic/frozen-head route. Feature extraction still needs measured CPU/GPU throughput; no promise of completing it under six hours follows from the existing mechanics runs.
4. Treat full report-only image admission and a storage ledger as prerequisites for full-corpus training. Process in bounded, restartable batches, preserving source/fold/label/cache lineage; a preprocessed subset should be reported as a subset.
5. Requery scheduler state immediately before scheduling. Queue duration, GPU model for the next allocation, full project storage headroom, full-corpus cache readiness and new-model throughput remain unknown rather than assumed.
