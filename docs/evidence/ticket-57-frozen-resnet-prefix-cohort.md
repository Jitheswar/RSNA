# Ticket 57 complete frozen ResNet prefix cohort

All **4,228 studies / 187,272 valid slices** completed extraction and independent
verification. Every reconstructed frozen ResNet feature matches the original
admitted feature cache exactly: maximum absolute discrepancy **0.0**. No labels,
fitted statistics, model adaptation or predictions entered this extraction.

| Stage | Exact Slurm job | Terminal | Allocated node seconds | Peak process RSS |
| --- | --- | --- | ---: | ---: |
| Partition 0: 2,114 studies / 93,640 slices | 7983 | COMPLETED, 0:0 | 949 | 2,064,228,352 bytes |
| Partition 1: 2,114 studies / 93,632 slices | 7984 | COMPLETED, 0:0 | 908 | 2,405,412,864 bytes |
| Independent complete-cohort verification | 7985 | COMPLETED, 0:0 | 701 | 383,496,192 bytes |

Total allocation is **2,558 CPU-node seconds**, zero GPU use and $0 external
spend. These are wall seconds of each one-node allocation, not CPU-core seconds.
The two nonexclusive extraction jobs each used 16 CPUs / 64 GiB / three hours;
the independent verifier used 16 CPUs / 64 GiB / thirty minutes. All attempts
had watchdogs, no requeue and no restarts. Exact controller terminal records
were captured before scheduler purge. No failed extraction or retry occurred.

The sealed activation directory contains **37,617,892,848 bytes**, including
31,653,360 bytes of NPZ/identity overhead above the **37,586,239,488-byte** FP32
prefix payload. Every valid slice is stored at shape `(256,14,14)` before layer4,
with explicit ascending packed-slice indices, exact masks and anatomical
descriptors. The prelaunch campaign storage observation remained below the 2 TB
soft ceiling, with room for payload, metadata and temporary writes.

Summed per-study extraction measurements: encoding **980.059 s**, writing and
hashing **271.469 s**, whole-study processing **1,829.952 s**. The remaining
whole-study time includes source reads, packing and reference validation;
these measurements overlap between the two jobs and must not be called elapsed
campaign wall time or a separate pure-I/O benchmark. Random reads and full-bank
training memory are measured separately before fitting in Ticket 58.

The independent verifier checked all file and array hashes, dtype/shape,
explicit indices, exact source identities, masks/descriptors against Ticket 36,
complete/disjoint accepted producer membership, successful terminal allocations,
frozen source/runtime and unchanged encoder state. It actually reconstructed
all **187,272 slices** through the original layer4 and pooling; completion is
supported by this full replay, not directory counts or an inventory.

Private root: `artifacts/private/ticket57-resnet-prefix-cohort/`.

- Extraction plan SHA-256: `5c2755f4bcc47e1d36851d8032dc78fa3830cf92e4acabac491c6f58d6f4f8f5`.
- Verification request: `f815fdec419d4f74a1ec79d8137eae0d783019b317fb8850c98c76c3b0ea5a8c`.
- Sealed feature manifest: `21bfeb43e24e8e55797961aeb6f0ad6b313cafad4cdb22ee2aeb2f2870d84e40`.
- Verification result: `8a4dc36097e234d0e8ae696a947c61b45f773f46dbceaaab4d871a73c36186bd`.

The typed cache loader rejects changed files, arrays, packing order and
source/plan identities. Eleven cache tests and three existing native layer4
lifecycle tests passed; static typing passed before extraction. The separately
added adaptation seam has three further tests demonstrating native/cached logits
and gradient agreement, fixed normalization and descriptor-preserving image
ablation; static typing passes across 82 source files. These are mechanics
checks, not evidence of predictive gain.

[Ticket 58](../../.scratch/rsna-leaderboard/issues/58-test-report-supervised-layer4-adaptation.md)
prospectively fixes three fold-specific paired arms, identical warm report heads
and batches, 1,500 final updates, learning rates, checkpoints, image controls and
paired uncertainty before adaptation predictions exist. Actual complete-cache
cost admission and a distinct executable fitting decision remain required.
No new candidate, competition score, final selection or first place is claimed.
