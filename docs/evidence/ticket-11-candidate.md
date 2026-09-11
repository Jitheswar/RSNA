# Ticket 11 locally validated offline candidate evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation and
real-corpus execution evidence; it contains no study identifiers, reports, raw
paths, or per-study prediction rows.

## Frozen candidate

`rsna_knee.candidate` freezes a complete inference bundle without overwriting
an existing candidate. The bundle contains three `final_selected` fold
checkpoints, the shared tested inference code, the inference configuration,
separate cluster/offline requirement locks, the offline notebook, corrected
public OOF evidence, provenance, and a checksum-covered manifest. The copied
code snapshot has 30 Python files, excludes `training.py` and
`training_cli.py`, has no network imports or training/report dependency, and
uses the same decoder, geometry, preprocessing, packing, model, aggregation,
submission, and recovery implementation in both environments.

The immutable candidate is
`ticket-11-local-candidate-20260911-v1`. Its manifest checksum is
`cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`; the
shared code digest is
`9676e0fde04df53de42ea0f3d7f820e9ac46f3442778841fada397c537f5a27b`, and the
source code revision recorded in the bundle is `54869dc`. The candidate is
intentionally `locally_validated_only`: Kaggle verification is false, owner
selection is required, and export is disabled.

The candidate validator independently checked the corrected three-fold OOF
evidence, the fold checksum, model/configuration compatibility, pinned runtime
versions (Python 3.11.16, NumPy 2.2.6, pydicom 3.0.1 and PyTorch 2.7.1+cpu),
notebook format and network-free instructions, artifact sizes/checksums, and
read-only reload. The candidate metadata, notebook and public OOF evidence
contain no identifiers.

## Real execution

The final execution was Slurm job `6950` through
`slurm/run_ticket11_candidate.sbatch`, on the `cluster1-cpu` partition with
one non-exclusive node, 16 CPUs, 64 GB requested memory and a six-hour wall
limit. The adapter placed the frozen candidate's `code/` directory first on
`PYTHONPATH`, so the real run used the immutable code snapshot rather than a
mutable working-tree import. It processed the authorized representative
private DICOM manifest and included 100 declared long-tail studies in the
measured scope.

Measured input and output contract:

| Quantity | Result |
| --- | ---: |
| Studies | 158 |
| Series | 965 admitted / 965 usable / 965 complete |
| Expected and decoded slices | 31,486 / 31,486 |
| Submission rows | 158 |
| Submission columns | 13 |
| Full-study fallback | 0 / 158 (0.0%) |
| Lost series / slices | 0 / 0 |
| Decode failures / diagnostic warnings | 0 / 0 |

`submission.csv` was independently validated with the exact header: first
`StudyInstanceUID`, followed in order by `ACL`, `MCL`, `Medial Meniscus`,
`Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`,
`Synovitis`, `Baker's`, `Contusion`, and `Fracture`. UID order matched the
manifest exactly; all twelve scores per row were finite and bounded in
`[0, 1]`. Private evidence retained one per-study record for all 158 studies,
and every private record used normal model inference with no fallback,
rejection, warning, or series/slice loss.

## Runtime evidence

The complete measured pipeline, from candidate/model startup through CSV
creation, took 604.504 seconds. Model loading took 0.034 seconds. Peak
resident memory was 2,071,003,136 bytes; the CPU run has no VRAM measurement.
The candidate bundle was 953,991 bytes. Linear projection from the measured
158-study scope to 1,300 studies is 4,973.767 seconds, below the 25,920-second
engineering target and 32,400-second hard limit. The projection is not a
full-scale claim: the measured workload is representative only, the hidden
execution environment is unknown, and the candidate has not run in Kaggle.

## Verification and limitations

- The candidate-focused tests, exact submission validation, immutable/tamper
  detection, real fixture execution, benchmark projection, and frozen-snapshot
  imports passed. The final full suite passed 151 tests, and mypy reported no
  issues in 32 source files.
- An earlier job (`6948`) failed before study processing because it used a
  stale integration manifest missing the current decoder-version field. It
  produced no candidate output and is not counted as execution evidence.
- Job `6949` completed the same representative workload but used the mutable
  repository source before the snapshot-path correction; it is retained as
  historical evidence, not the final provenance run.
- Actual approved Kaggle execution, hidden full-workload timing, owner final
  selection, export, upload, and competition submission remain pending. No
  approval-dependent gate is marked passed by this document.
- Candidate save/reload and immutable publication were verified, and the real
  runner supports damaged-input recovery, but no interrupted-process resume
  demonstration or executed offline notebook session was run. The metadata
  checks are structural checks of the frozen manifest, OOF record and
  notebook; they do not replace a future owner audit of all private delivery
  metadata and notebook outputs.

The machine-readable public/private evidence and submission are ignored
execution artifacts under `artifacts/private/`; the private record remains
separate from this public-safe document.
