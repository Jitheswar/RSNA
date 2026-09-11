# Ticket 06 series-cache implementation evidence

Recorded 2026-09-11 (Asia/Kolkata). This note describes repository-level
mechanics only. It contains no competition identifiers or private paths.

## Implemented boundary

`rsna_knee.series_cache.SeriesCache` is the public cache seam. It accepts a
`SeriesRecord` and a model-independent `PreprocessConfig`, derives a key from
the source series and `PreprocessConfig.cache_config()`, and returns a
`SeriesCacheResult` containing the prepared series, key, hit/miss reason, and
I/O measurements.

Each complete entry contains:

- all accepted float32 materialized slices in `images.npy`;
- per-slice SOP identity, source-entry identity, and SHA-256 content hash;
- preprocessing and prepared-output schema versions;
- materialization shape, dtype, geometry order, and warnings;
- an array checksum, byte count, and checksum over the metadata itself.

The source hash is taken from the raw source reader for real DICOM input. When
callers construct a decoder-independent `SeriesRecord` without a raw hash,
the cache derives a deterministic hash from the decoded slice record and
pixels. The optional `source_content_hashes` argument allows an upstream
archive adapter to provide exact source hashes.

## Recovery and cleanup behavior

Per-key advisory locks serialize writers. A writer completes the array and
metadata in a same-directory staging directory, fsyncs them, then publishes
the directory with an atomic rename. A missing, incomplete, checksum-invalid,
policy-mismatched, or shape-invalid entry is never a hit and is rebuilt under
the same key. Staging directories left by an interruption are ignored by
readers and are removed only when
`cleanup_disposable(..., remove_partial=True)` is explicitly requested.

Cleanup accepts an explicit set of referenced cache keys and operates only
below the configured cache root. It cannot select raw archives, reports,
checkpoints, folds, OOF artifacts, or candidate records.

The cache root is required to be mode 0700 and carries a role marker. A
populated directory without that marker is rejected, which prevents an
operator from accidentally treating a raw-data or protected-artifact directory
as disposable cache storage.

## Repository verification

The focused tests cover:

- eight-versus-sixteen-slice reuse;
- source and preprocessing invalidation at series granularity;
- persisted provenance, shape/order/warnings, and checksums;
- checksum repair and incomplete-publication rejection;
- same-key concurrent writers;
- bounded benchmark accounting; and
- explicit unreferenced-cache cleanup.

The real-DICOM integration path accepts `--cache-root` and attaches a source
content hash while streaming each selected entry. The existing real integration
tests exercise this path with a private-safe fixture.

## Unresolved V09 evidence

No NFS throughput, cluster concurrency, or full private-corpus peak-storage
benchmark is claimed by this implementation. Those measurements still need an
authorized `cluster1-cpu` Slurm run with declared worker counts and the actual
raw/archive, cache, environment, checkpoint/candidate, and temporary-write
budgets. Until that run is recorded, cache expansion and larger concurrency
remain blocked by V09.
