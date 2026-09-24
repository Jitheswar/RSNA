# 25: Evaluate one late efficiency optimization

**What to build:** Optionally measure one efficiency improvement after a valid final delivery path exists.

**Blocked by:** 24: Select and deliver the final candidate

**Status:** ready-for-human

**Authorization:** The owner explicitly authorized implementation with “implement ticket 25.” The authorized experiment is limited to this ticket and its stated CPU boundary; it does not authorize GPU work, heavier compute, Kaggle uploads, kernel execution, or competition submissions.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Target the measured input-mount discovery bottleneck and compare against the exact Ticket 11 candidate. The report includes startup, loading, DICOM decoding, preprocessing, ensemble forward passes and CSV output costs.
- [x] Report the inherited OOF macro AUC and zero expected quality delta from the routing-only change; report the observed local output equivalence, measured and projected runtimes, peak VRAM, RAM and package sizes. OOF was not recomputed.
- [x] Freeze the experiment variant separately by the Ticket 11 manifest and Ticket 25 adapter SHA-256. Ticket 24 remains selected; no replacement or owner selection was made. Obtain owner selection before any future replacement.
- [x] Keep Ticket 24's valid delivery unchanged. This experiment made no Kaggle upload, kernel execution or competition submission.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V16 — Whole-notebook runtime and inference cost

- [x] Record the Ticket 24 actual Kaggle baseline: 3 studies, 15 series, 557 slices, 1,247.609 seconds, 879,849,472-byte peak RAM and zero VRAM. Its input-discovery stage took 821.485 seconds. Record the Ticket 25 cluster1-cpu run on 158 representative studies, 965 series and 31,486 slices, including the 26-study long tail; it took 1,094.782 seconds, peaked at 1,127,796,736 bytes RAM and zero VRAM, had no fallbacks or decode failures, and produced a byte-identical CSV. The allocated job used 4 CPUs and requested 64 GB; its measured process RSS remained below the 8 GiB bound.
- [x] Record the Ticket 24 v6 delivery package (196,517,501 bytes), candidate bundle (940,350 bytes), three checkpoints/model weights (314,944 bytes total), and unchanged OOF macro AUC 0.464313. The namespaced discovery probe reduced adapter lookup from 32.924 seconds to 0.682 seconds on a Slurm CPU node with 31,486 synthetic empty files; this is not Kaggle-host runtime evidence.
- [x] Recompute the projection without overlapping the outer study loop with nested decoding, preprocessing and forward-pass timings. The Ticket 24 18,078.937-second projection corrects to 12,755.104 seconds at 1,300 studies; replacing only its observed 821.485-second discovery stage with the measured namespaced lookup gives a projected 11,934.301 seconds. Both are projections from a three-study Kaggle run, not a hidden-scale execution.
- [ ] Run the Ticket 25 adapter/notebook bundle on Kaggle and verify hidden-scale runtime. This remains unverified; the Ticket 24 v6 submission remains the only selected and Kaggle-scored candidate.

**Blocking consequence:** Block normal promotion above the 7h12m engineering target until optimized or the specific margin risk is approved. Execution beyond nine hours is ineligible and cannot be waived by internal approval. Block full-scale compliance claims when evidence is only a tiny example; keep unknown hidden runtime explicit and preserve the previous verified candidate.

## Completion evidence

- [x] Record the demonstration, verified scope, unresolved gates and limitations without exposing restricted data. No future execution or approval-dependent check is marked passed.

## Answer

Ticket 25 measured namespace-scoped input discovery against the selected Ticket 11 candidate. The adapter now searches `datasets/` for delivery and approval metadata and `competitions/` for competition input, so it does not traverse the image payload while looking for metadata. The model, preprocessing, decoding, forward passes, ensemble and CSV output are unchanged. The 158-study output was byte-identical to Ticket 24 v6.

The Slurm CPU run measured 1,094.782 seconds for 158 studies, with 1.128 GB peak RSS and zero VRAM. In the synthetic mount probe with 31,486 empty files, adapter discovery fell from 32.924 to 0.682 seconds. The Ticket 24 Kaggle baseline's measured input-discovery stage was 821.485 seconds. Replacing that one fixed stage yields a projection of 11,934 seconds at 1,300 studies, versus a corrected 12,755-second baseline projection. This projected saving is not an observed Kaggle improvement.

The Ticket 24 candidate remains selected and unchanged. The measured Ticket 25 adapter SHA-256 was `3d3b928ef6769f0eddd22f8bba02b82664e26742a848c68418472c741d19c6a0`; after the run, the profile schema was advanced to v2 to distinguish the corrected non-overlapping timing semantics. The final source SHA-256 is `2755f40c1d19ed93940dee131ba602183282e31b51b86b7c921caa380be44164`; the schema label change does not affect prediction behavior. No replacement, Kaggle upload, new kernel execution or submission occurred. A new exact delivery request and owner selection are required before any future promotion.

Full report: [`/home/ch.jitheswar/ticket-25-efficiency-report.md`](../../../../ticket-25-efficiency-report.md). The aggregate machine-readable run record remains private at `artifacts/private/ticket-25-efficiency/run-7147/public-summary.json`.

## Comments

- 2026-09-24 IST — The owner authorized Ticket 25 implementation and directed the agent to use its best recommendation without asking for routine choices. Ticket 24 was committed first as `3367e9f`. The selected final candidate and its Kaggle delivery remain unchanged.
- 2026-09-25 IST — The bounded cluster1-cpu benchmark completed its inference and wrote the aggregate summary; its final console status initially failed on a relative-path formatting error. The path formatter was corrected, and `--verify-existing` validated the saved summary and matched submission checksum. The source fix does not rerun inference. No Kaggle action was performed.
