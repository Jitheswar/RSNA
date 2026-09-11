# Ticket 05 real-DICOM integration evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe evidence; it contains
no study, series, SOP, report, or source-path identifiers. The identifier-bearing
manifest, submission, and detailed diagnostics remain in ignored private
artifacts.

## Demonstrated scope

Slurm job 6913 ran the private ticket-03-selected representative data through
the real integration boundary on `cluster1-cpu`. The job built a checksum-
covered private integration manifest, verified the source archive identity,
decoded every admitted entry, and sent each study through the shared
preprocessing, hierarchical model, and submission writer.

The aggregate result was:

- 158 input studies and 965 admitted series;
- 31,486 expected and decoded slices, with zero decode failures or rejected
  series;
- zero non-diagnostic exclusions, zero fallback studies, and zero diagnostic
  warnings in this selected batch;
- 158 output rows with the exact 13-column submission schema; and
- finite scores bounded to the model's `[0, 1]` contract.

The run took 551.6 seconds. The adapter requested one node, 16 CPUs, 64 GB,
and six hours without exclusive allocation, and completed without a resource
limit failure. The execution used the pinned Python 3.11.16,
PyTorch 2.7.1+cpu, NumPy 2.2.6, and pydicom 3.0.1 environment.

## Frozen policies exercised

- Geometry `rsna-knee-geometry-v2`: position tolerance `1e-4`, orientation
  tolerance `1e-3`, spacing tolerance `1e-3`, consistent dimensions and
  orientation required, and at least three usable slices.
- Decoder `rsna-knee-dicom-v1`: pydicom-only attempt order, explicit VR little
  endian admission, required file metadata and MR modality, single-frame
  grayscale pixels, and typed damaged/unsupported/infrastructure outcomes.
- Preprocessing `rsna-knee-preprocess-v1`: one grayscale channel, deterministic
  aspect-preserving nearest-neighbor resize/pad to 128 by 128, valid-slice
  0th–100th percentile normalization, stored padding removed before modality
  transforms, no VOI/window transform, float32 output, and explicit masks for
  short series. Packing samples eight evenly spaced slices and up to sixteen
  series after materialization.
- Cache boundary: preprocessing configuration excludes sampling controls and
  model architecture; inference imports neither report extraction nor training
  orchestration.
- Source integrity: production verification requires the checksum-identified
  source ZIP; unverified extracted directories are reserved for local fixture
  smoke tests.

## Verification

The repository suite passes with 69 tests, including synthetic decoder,
geometry, padding/intensity, source-mapping, localizer-exclusion, partial-file,
manifest-integrity, and public-evidence cases. Mypy reports no issues across
the 18 source modules.

## Unresolved gates and limitations

- This is representative selected-subset evidence, not a claim of blanket
  pixel compatibility for every class in the full competition corpus. The run
  exercised the observed explicit-VR-little-endian subset; compressed or other
  unsupported syntaxes remain explicitly rejected and require separate V06
  verification.
- The real run used a deterministic seeded untrained hierarchical model to
  verify mechanics and output contracts. It makes no diagnostic-performance,
  fold, training, or candidate-quality claim.
- V04 and V05 are demonstrated for the admitted selected series and synthetic
  edge cases, but repeated real-series agreement and later study-recovery
  policy gates remain pending.
- The current V04 policy freezes projected inter-slice spacing and declared
  pixel dimensions; in-plane DICOM `PixelSpacing` consistency is intentionally
  outside this version and remains a follow-up gate.
- No report adaptation, training, upload, competition submission, archive
  deletion, or protected data movement was performed.
