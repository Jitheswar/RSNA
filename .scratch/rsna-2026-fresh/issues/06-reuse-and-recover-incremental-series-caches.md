# 06: Reuse and recover incremental series caches

**What to build:** Reuse deterministic series preprocessing across sampling choices and recover safely from interrupted cache work.

**Blocked by:** 05: Run a real MRI study through inference

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Store all successfully processed accepted slices with source identity/content hash, preprocessing version/configuration, schema, shape, order, warnings and checksum.
- [ ] Prove eight-versus-sixteen-slice sampling reuses the same cache while source or preprocessing changes rebuild only affected series.
- [ ] Test interrupted publication, checksum mismatch and conflicting writers; incomplete entries cannot become cache hits.
- [ ] Benchmark bounded NFS reads/writes and concurrency, including temporary atomic-write space; expand only after the mechanics baseline and resource evidence justify it.
- [ ] Allow cleanup only for explicitly disposable reproducible unreferenced cache material; preserve raw archive, reports, checkpoints, folds, OOF and experiment/candidate records.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V09 — Shared-storage throughput, concurrency and peak use

- [ ] Measure representative indexing/extraction, cache writes and loader reads with declared worker counts and resources. Account for raw/archive data, caches, environments, checkpoints/candidates, concurrent writers and temporary atomic-write copies. Project total peak use at no more than 2 TB unless an expansion is explicitly approved. Demonstrate that test jobs stay within requested CPU/memory/wall-time limits, publish caches atomically, reject partial output after interruption and do not allow conflicting writers for one cache key.

**Blocking consequence:** Block cache expansion or concurrency/resource settings whose projected space or measured resource use exceeds the authorized envelope. Reduce batch/worker/output scope or request a documented boundary change. Block readers of unsafe/partial cache artifacts until atomicity is fixed; never delete protected artifacts to force the budget calculation to pass.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Incremental series-cache implementation

Implemented the private `SeriesCache` seam in `src/rsna_knee/series_cache.py`.
It stores all accepted materialized slices with per-slice source identity and
content SHA-256, preprocessing/schema configuration, geometry order, shape,
warnings, array checksum, and metadata checksum. Sampling controls remain
outside the key, so the cache benchmark demonstrates eight-versus-sixteen
slice reuse while source or materialization changes invalidate only the
affected series.

Publication uses per-key advisory locks, same-directory staging, fsyncs, and
atomic directory rename. Incomplete, mismatched, corrupt, or policy-incompatible
entries cannot become hits. Cleanup is explicitly scoped to a marked private
mode-0700 disposable cache root and an explicit referenced-key set; raw data,
reports, checkpoints, folds, OOF, and candidate artifacts are outside the
cleanup boundary.

`InferenceEngine` and the real-DICOM runner accept an optional private cache
root. The bounded `slurm/run_ticket06_cache_benchmark.sbatch` adapter measures
cold materialization, warm sampling reuse, bounded worker concurrency, cache
I/O, and conservative concurrent temporary-write space in aggregate-only
evidence. Focused tests cover the mechanics and the real source reader records
exact source-entry hashes.

V09 remains unresolved: no private-corpus NFS run, cluster throughput/concurrency
measurement, actual RSS measurement, or whole-project 2 TB peak-storage
reconciliation was performed. Cache expansion and higher concurrency remain
blocked until that authorized Slurm evidence is recorded.
