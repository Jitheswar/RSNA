# 19: Evaluate one stronger image-model variant

**What to build:** Evaluate one narrowly scoped image architecture or input-policy change against the established supervision-controlled baseline.

**Blocked by:** 18: Measure the controlled benefit of report supervision

**Status:** resolved

**Authorization:** Initially planning-only. The owner's direct instruction to implement Ticket 19 on 2026-09-23 separately authorized this implementation and the bounded CPU experiment below. GPU execution and the other approval-gated actions remain unauthorized unless separately approved.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Define one proposal: widen the shared slice encoder from 8/16/32 to 16/32/64 while keeping supervision, folds and other comparison settings fixed.
- [x] No numerical preprocessing or sampling change was made; cache semantics remain unchanged.
- [x] OOF/per-target evidence, uncertainty and training cost are recorded. Full end-to-end inference runtime, VRAM and complete delivery-package size remain unmeasured and are non-blocking for Ticket 19 closure; the parent V16 gate applies before local candidate promotion.
- [x] No pretrained resource was downloaded or used, and no GPU job was executed. V13/V10 execution gates remain unclaimed.

## Completion evidence

- [x] Record the CPU demonstration, verified scope, unresolved gates and limitations in the external report without exposing restricted data. Future full-inference, VRAM and package checks are not marked passed.

## Comments

- 2026-09-23 — Owner explicitly instructed implementation. A CPU-only three-fold run was submitted to `cluster1-cpu` as Slurm array `7101`, with two concurrent tasks maximum. No pretrained resource or GPU is used.
- 2026-09-23 — All three folds completed and the paired comparison aggregated successfully in job `7105`. The pooled macro-AUC delta was +0.0066 with a paired 95% study-bootstrap interval of −0.0295 to +0.0442; the result is inconclusive. Implementation and CPU evaluation are complete. Status is `needs-info` for any follow-up that would close the full-inference, VRAM and package-size evidence gaps; VRAM requires separate GPU approval. See [external report](../../../../ticket-19-report.md).
- 2026-09-23 — Closure verification passed: full pytest 306 passed; focused Ticket 19 tests 4 passed; mypy, `git diff --check`, both Ticket 19 Slurm syntax checks, 3/3 candidate checkpoint reloads, canonical 58-study OOF integrity (20/19/19, no duplicates or overlap), comparison controls, and identifier scan all passed. Full end-to-end inference runtime, VRAM and complete delivery-package size remain unmeasured and non-blocking for this experiment; they remain later V16 promotion evidence. The 16/32/64 candidate was not promoted. See [external closure report](../../../../ticket-19-report.md).
