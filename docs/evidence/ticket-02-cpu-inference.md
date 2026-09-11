# Ticket 02 CPU inference evidence

Recorded 2026-09-11 (Asia/Kolkata) after the final implementation was
submitted through Slurm job `6896`.

## Verified scope

The job ran on `KLEF-C-NODE2` in the `cluster1-cpu` partition with four CPUs,
8 GB requested memory, one node, and a 30-minute wall-time limit. It used the
user-space virtual environment at
`/home/ch.jitheswar/RSNA/.venv/bin/python`, based on Python 3.11.16 from
`mise`. The recorded package identity was NumPy 2.2.6, PyTorch 2.7.1+cpu, and
pydicom 3.0.1.

The synthetic run processed two studies and three valid series in 0.223
seconds. It produced two rows with exactly thirteen columns, the required
`StudyInstanceUID` plus the twelve targets in contract order. All scores were
finite and within `[0, 1]`; no study fallback was used. The generated
submission and machine-readable job record are intentionally ignored under
`artifacts/` because they are execution outputs rather than source artifacts.

## Tests and interfaces

The public test suite covers geometry ordering and rejection, deterministic
shape/mask packing, masked hierarchical pooling, finite gradients, twelve
outputs, UID alignment, exact CSV schema, bounded scores, all-invalid study
fallback, and the complete synthetic inference seam. The final run passed 19
tests; mypy reported no issues for the eleven source modules.

## Explicit limitations

- The fixture creates in-memory explicit-little-endian pydicom MR datasets and
  maps them through the decoder seam. Real-corpus DICOM compatibility,
  transfer-syntax coverage, pixel semantics, and corpus coverage remain
  unverified for later tickets.
- No training loop, real-label evaluation, full training preflight, Kaggle
  execution, upload, or competition submission was performed.
- The cluster's `python3.11` module was inspected but rejected for this run
  because its interpreter could not import `_ctypes`; the verified environment
  was the separate user-space `mise` Python 3.11.16 environment.
