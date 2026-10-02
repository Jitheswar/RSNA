# Ticket 62 — exact original MRI pixel cohort

All **4,228 admitted studies /187,272 valid selected slices** are retained as
exact FP32 pixels and match fresh source packing. Slurm **8017 completed
`0:0`**, **2,080 allocated CPU-node seconds**, one nonexclusive 16-CPU/64-GiB
job with a three-hour limit. No labels, model, fitting, GPU, network, new
weights or external spend. This establishes usable inputs, not predictive gain.

Plan SHA-256:
`718e2c127bef8ea1135c7e76b653b2fee6313a6d69dc89219e8f78e5b645b581`.
Private root: `artifacts/private/ticket62-mri-pixel-cohort-v1/`.

Only valid slices are stored, with explicit ascending indices into the same
16-series ×8-slice packed bag. Padded reconstruction is exactly zero. Each
entry binds complete file/array hashes, original study/source identity,
representation, materialization plan, masks, anatomical descriptors and sampled
source indices. Source series caches and original archive remained read-only.
Exact pixel payload **37,586,239,488 bytes**; measured projected campaign peak
**1,229,684,476,792 bytes**, under the two-terabyte soft cap.

Producer elapsed **1,112.019 seconds**, peak RSS **2,273,918,976 bytes**. A fresh
process in the same allocation re-read each original source, reapplied the
frozen native packing algorithm, and compared every reconstructed pixel,
mask, descriptor and source mapping. It uses the same algorithm, rather than
an independently implemented transform. All arrays match exactly; the prior
four-report full-pixel SHA also replays exactly. Verifier elapsed **964.378
seconds**, native source reading/packing **704.605 seconds**, verified cache
reading/reconstruction **189.317 seconds** across all entries, maximum single
entry **0.121137 seconds**. These observations are from one warm shared CPU
allocation; future startup still needs its own measurement.

Manifest SHA-256:
`b5a88da82ea1448cb7b97618e964462dadffe1db69d89d86c7f819e68541c173`.
Independent validation SHA-256:
`80cc8d20359291ebf94fea8c80302bdb77054fc27b605a5f5c91032c8836af38`.
Accepted job/plan, retained Slurm terminal, frozen dependencies, resources and
successful complete verification agree in `terminal-verification.json`.

The shared trainer accepts file-backed pixel arrays so retaining the complete
cohort does not retain all padded images in memory. Every later materialization
rehashes and checks its entry. Full encoder/head learning-rate groups and
descriptor-aware held-out/native/candidate prediction use the registered model
identity. Focused checks cover corruption/wrong membership, exact reconstruction,
actual encoder updates, interrupted/resumed versus continuous model/loss/exposure
equality, CPU execution scope and native prediction parity: **57 tests passed**,
typing **84 source files**. No fitted candidate is created by these tests.

[Ticket 63](../../.scratch/rsna-leaderboard/issues/63-measure-shared-full-backbone-training-delivery-cost.md)
measures actual all-fold input admission and mixed-source staged trainer costs.
A complete matched three-fold comparison, final-fit reserve and offline delivery
plan must be separately frozen before longer fitting. Public best remains
**0.936**; neither first-place rank is established.
