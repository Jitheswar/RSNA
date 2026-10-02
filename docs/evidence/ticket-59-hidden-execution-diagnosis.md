# Ticket 59: hidden frontier error remains unreproduced

Exact submission **56739912 / script 354194017 / v3** is terminal with a generic
hidden-rerun error and no public/private score. Authenticated **08:56:47 UTC**
refresh confirms that result, both scored fallbacks **0.936 / 0.498**, and public
leader **rhoskeri / 0.961**. Saved notebook logs belong to the successful visible
session; no hidden traceback was returned. The official owned-submission download
endpoint also did not expose either checked output file. First place is unverified.

The exact hidden failure's full unknown **57,600 GPU seconds** remain charged.
The later fixed Ticket 53 development run passed all 58 real studies, all 27
anchor prediction artifacts and zero fallbacks. That success does not reproduce
the hidden error. No duplicate scoring, speculative guard change or root-cause
claim has been made.

## Completed broader input probe

A label-blind SHA-256 ordering selected 256 of the 4,170 admitted report-source
MRI studies. Every original acquisition for each selected study was read from
the archive, including series excluded from the learning cache. No reference
label values, reports, fitted model or GPU entered this probe.

| Measured field | Result |
| --- | --- |
| Exact CPU job | 7992, COMPLETED, 0:0 |
| Allocated CPU-node seconds | 264 |
| Original DICOMs / raw bytes | 48,870 / 31,604,773,004 |
| Header series / selected slot-series | 1,446 / 1,195 |
| Native disk cache bytes | 2,080,899,072 |
| Failed decode / failed-fallback events | 0 / 0 |
| Present slots per study | 3–6 |
| Peak process RSS | 5,391,650,816 bytes |
| GPU / external spend | 0 / $0 |

The executed fragments are unchanged definitions from the exact attributed D4
source, with its admitted native pixel group. Execution excludes GPU/model,
CoAt and hidden-input branches. Exact successful terminal controller, runtime,
source fragments and source-entry SHA records are retained. Independent summary
verification checks all 30 frozen inputs, study membership/counts and all 48,870
persisted source-file sizes; source bytes were hashed by the producer and were
not independently rehashed. Temporary pixels were released; original extracted
MRI inputs remain private for further diagnostic replay.

Private root: `artifacts/private/public-frontier-adoption/broader-cpu-input-probe-v1/`.
Plan SHA-256:
`dc7834a855ffc29112b9fb3b1f65d0c28f649dee29f6f68ad1b387f9b8981a8e`.
Result SHA-256:
`14bf8e3c7adc62851051c2b721c2ae8e5c167399ed56b2a4d95360c109562197`.
Disposition: **hidden failure unreproduced**. This passes a broader accessible
input seam without establishing the hidden run's cause.

## Full original input scan: complete

The measured 264-second probe supports one separate bounded scan of **all 4,407
original training-series metadata study UIDs**, including reference and previously
excluded source studies, with no label values read. Sorted alternating membership
produces disjoint **2,204 / 2,203** partitions, accepted jobs **7995 / 7996**.
Both run with unchanged native pixel rules/source fragments. The worker's only
mechanical changes generalize the cohort dimension and enforce larger exact
CPU resource checks. No model forward pass, fitting or GPU request.

Each job is nonexclusive, **16 CPUs / 64 GiB / three hours**, watchdog **10,560
seconds**, one attempt, no requeue. A per-partition forecast uses 2x measured
time per study plus 900 seconds and fits that bound. Raw-data allowance **384
GiB per partition**, pixel-bank allowance **32 GiB per partition**, aggregate
**832 GiB** temporary ceiling. Direct observed project storage before freezing
was **603,970,359,296 bytes**, keeping the total below the 2 TiB soft ceiling.

Private root: `artifacts/private/public-frontier-adoption/full-cpu-input-probe-v2/`.
Campaign plan SHA-256:
`a2d7f3624f1798fc3ca591e9db6f83ca171bb97793cca43d19e2ca4c0c84c1b8`.
All source hashes, disjoint membership, exact decisions, worker parsing and
launchers passed prelaunch checks. Observers take each actual accepted job ID.
An earlier unlaunched v1 draft retained an incorrectly transcribed, larger storage
figure labelled measured; it was preserved and superseded before any job launch.
The preparer now measures and records storage directly. Source rules and resource
limits did not change during that correction.

Jobs **7995 / 7996** both completed `0:0`, **2,485 / 2,489 allocated CPU-node
seconds**. Independent coverage verification on October 2 proves complete,
disjoint **4,407-study / 819,078-DICOM** membership against the original archive
central directory and `train_series.csv`. Raw bytes total **569,154,778,424**.
All **24,371** original header series are represented, **20,204** selected
slot-series completed, and actual cache events agree with per-study slot counts.
Both partitions have zero decode failures and zero failed/fallback events.
All exact executed fragments, source records, decisions and terminal identities
pass checks. Peak process RSS is **21,154,127,872 / 21,165,096,960 bytes**.

Summary validation: `full-cpu-input-probe-v2/independent-summary-validation.json`.
This checks original source-entry membership, archive CRC/size records and
producer SHA records; it does not itself reread every retained DICOM. A separate
one-attempt CPU integrity job **8012**, 16 CPUs / 32 GiB / one hour, independently
rehashes every retained source payload and computes CRC against the original
archive records. It completed `0:0`, **779 allocated verifier CPU-node seconds**;
all **819,078 payloads / 569,154,778,424 bytes** match independent SHA and CRC
checks. Exact accepted request, immutable worker/plan, terminal allocation and
independently verified partition byte totals also pass terminal reconciliation.
Its frozen plan is
`8838254220079e484e16f90aa87ef066afc412ef467e45bad3c6606cd5c51237`.
Payload validation SHA-256:
`bcc777b9193de735cd7ac7e76d76b8ce2558468a222052e53631e42ceba50844`.
Actual verifier terminal and reconciliation are retained under
`full-input-independent-verification-v1/`; complete original scan plus independent
integrity verification totals **5,753 CPU-node seconds**.

A red-capable hidden-error loop and any repair remain pending. The
[diagnosis skill](../../.agents/skills/diagnosing-bugs/SKILL.md) requires
“No red-capable command, no Phase 2.” No hypotheses are ranked while the actual
hidden symptom lacks a matching reproduction. Retain successful/failed outputs
and matching terminal evidence before considering a subsequent action. GPU ledger
stays **129,637 charged / 136,800 cap / zero reserved / 7,163 remaining**.

## October 2 — exact residual CoAt preparation seam

Fresh authenticated **01:08:08 UTC** leaderboard snapshot: **Elyra rank 2,005
of 4,808, 0.936**; **rhoskeri rank 1, 0.961**. Exact submission 56739912 still
has the generic hidden-rerun error and no score; private rank remains unavailable.
The downloaded official snapshot and submission observations are private under
`scoring-failure-diagnosis-v1/live-refresh-20261002/`. No submission was issued.

One fixed **64-study** CPU probe **8013** completed `0:0`, **148 allocated
CPU-node seconds**, with zero preparation failures, warnings, sparse fallbacks
or dense-sampler fallbacks. The label-blind selection uses evenly spaced order
statistics of original study slice count over all 4,407 studies, SHA256(UID)
breaking ties. It includes the **67 / 632** original-slice extremes.
Actual rendered bags have **61–94 windows**, uint8 **384×384** triplets and aligned
slot/geometry sidecars. Peak worker RSS **1,702,268,928 bytes**.

This executes the actual D4 residual CoAt `_prepare_bag_task`, including the
notebook's unchanged dense sampler, renderer, native physical grid and right-knee
canonicalization. CPU runtime sources and the OpenCV 4.12.0.88 wheel match the
already verified Kaggle mounted hashes. No checkpoint, model forward pass,
labels/reports, GPU or execution-time network are used. Independent summary
verification checks actual terminal/resources, frozen code/metadata identities,
study/order coverage, window/schema and producer pixel/sidecar hash records.
Images are not independently replayed. This success does not reproduce or
identify the hidden error.

Private root: `coat-cpu-input-probe-v1/`; plan SHA-256
`9ea8ee92ccd79764220f7daab596a8b5c32233e8cdad80c7b17db01f1fec725f`;
result SHA-256
`5f2abec231694050de913227ae56c9ea41274ce658f2f3b1d33e74ec6c7161bf`.

The measured pilot supports a distinct **all-4,407-study** CoAt preparation scan,
job **8014**, private root `full-coat-cpu-input-probe-v1/`, plan SHA-256
`abea8f3b42312c3fba1a43a15ea732af256a19fdaa22c59e5998538da83dc5fa`.
One nonexclusive **16-CPU / 64-GiB / six-hour** job, **21,540-second watchdog**,
two preparation workers, one attempt/no requeue/GPU/spend. Twice measured pilot
cost scaled to full membership plus 900 seconds forecasts **21,282.375 seconds**,
inside that bound. Original MRI files are reused read-only, and bags are released
after recording hashes. Only the worker's resource guard/watchdog change; the
scientific input functions are unchanged. An exact-job controller observer
retains terminal evidence. Full outcome is pending; do not infer it from the pilot.
