# Ticket 12 Kaggle baseline delivery evidence

Recorded 2026-09-11 (Asia/Kolkata). This public-safe record contains no
competition study identifiers, reports, raw paths, or prediction rows.

## Implemented delivery seam

`rsna_knee.kaggle` now provides four immutable stages:

1. `prepare_kaggle_delivery` reloads the ticket-11 candidate and validates its
   manifest, selected checkpoint hashes, inference configuration, local exact
   submission evidence, runtime projection, zero-fallback evidence, delivery
   notebook, network-free `kaggle.py` adapter, and pinned offline environment.
   It writes an `owner_approval_required` request containing the exact hashes
   and an explicit delivery layout. It does not contact Kaggle or enable
   upload/submission.
2. `discover_kaggle_inputs` resolves Kaggle's namespaced dataset and
   competition mounts by validating the exact request checksum, candidate
   manifest checksum, owner approval, required competition files, and
   immutable candidate-code mirror before the adapter is imported. This keeps
   discovery independent of the hidden DICOM count and supports paths such as
   `/kaggle/input/datasets/<owner>/<slug>` and
   `/kaggle/input/competitions/<slug>`.
3. `run_kaggle_baseline` is the notebook adapter for Kaggle's
   `test.csv`/`test_series.csv`/`test_series/` layout. It reuses the candidate
   decoder, preprocessing, hierarchical model, probability ensemble and
   strict submission writer. Non-smoke execution verifies the mounted request,
   owner approval, copied notebook/adapter hashes, and installed pins before
   inference. It records resource sizes, peak VRAM, notebook-stage timings,
   projection, actual workload, output checksum, and measured fallback
   diagnostics; detailed per-study failures are emitted only to the private
   evidence file.
4. `record_kaggle_verification` accepts only an owner approval for the exact
   request and externally supplied accepted Kaggle evidence. It rejects
   mismatched hashes, example-only runs, network/report dependencies, malformed
   output, hard-limit violations, missing package/model/VRAM measurements,
   missing observed-versus-projected timing, missing recovery evidence, and
   fallback evidence that invents zero for an unknown hidden rate. Promotion
   additionally requires preservation of the accepted CSV and private
   per-study failure report.
   `promote_kaggle_candidate` writes a separate `kaggle_verified_only` pointer
   only when fallback is measured at zero, recovery is demonstrated, and the
   runtime gate is eligible; it never edits the candidate or request.

The approval file must confirm current rule recheck, owner rule acceptance,
account eligibility, and authorization for the exact notebook/model upload and
competition submission. It must also list dated official rule sources and state
whether material changes were reconciled. No approval file is created by the
implementation. Final selection remains outside this interface.

## Local evidence carried forward

The exact ticket-11 candidate remains the input candidate:

- candidate: `ticket-11-local-candidate-20260911-v1`
- candidate manifest checksum:
  `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`
- source revision recorded by the candidate: `54869dc`
- local representative scope: 158 studies, 965 series, and 31,486 decoded
  slices
- local output: 158 rows, thirteen columns, exact header/order, finite scores
  bounded to `[0, 1]`
- local complete-study fallback: 0/158
- projected runtime at approximately 1,300 studies: 4,973.767 seconds

This is local candidate evidence only. The projection is not a hidden-scale
runtime claim, and the hidden fallback rate remains unknown.

The local preparation command produced the immutable request
`ticket-12-ticket-11-local-candidate-20260911-v3` after the final contract
checks were added. Its request checksum is
`c1351d4b405d29e17547bb3d385646235d1c9845ee15790644d6a04d66453ac7`, its
delivery notebook checksum is
`20ad131cbf8aa988004fa1bf46cf07253453466edc9dda27ef460e078eced01e`, and
its adapter checksum is
`f158cc31d305001d6bde1a6c82dd800afed8fbc9d612b1bd1c0cedf91f1fde97`.
The request is private ignored execution metadata; its read-only JSON was
reloaded with the candidate, notebook and adapter checks and checked for
identifier-bearing fields.

The v3 request and its failed v1/v2 execution records were preserved. Under
the owner-authorized path-compatibility fix, a new immutable request
`ticket-12-ticket-11-local-candidate-20260912-v4` was prepared with request
checksum
`a32718182d733d00e81383aedead5a2073fb8bcaf906688529dfecc28310f942`, the same
candidate manifest checksum, delivery notebook checksum
`5b3516dd70575adb41b4411d8e84828b5b49d3d4298771383e5d731897cf1f64`, and
adapter checksum
`adc65d83f31041f6242c8d52c3902f5f1c5215a1e2be64e98f3eb43144d6fe59`.
The v4 private delivery and approval datasets were ready, and the
internet-disabled CPU kernel reached the candidate runtime after resolving
the namespaced mounts. It then stopped before inference because Kaggle's
runtime provided `pydicom 3.0.2` while the immutable candidate requires
`pydicom 3.0.1`; no `submission.csv` or execution evidence was produced. The
preserved v4 log is private execution evidence, not a verification pass.

## Runtime mismatch resolution and do3 execution history

The owner-authorized `do3.txt` instruction explicitly preserved the frozen
candidate, its manifest, checkpoints, and environment contract. The candidate
has no formal Python-version pin; its frozen offline runtime pins are exactly
NumPy 2.2.6, pydicom 3.0.1, and torch 2.7.1+cpu. A separate internet-disabled
Kaggle diagnostic recorded the preinstalled runtime as Python 3.12.13, NumPy
2.0.2, pydicom 3.0.2, and torch 2.10.0+cpu.

The v5 request preserved the first pydicom-only attempt:

- request `ticket-12-ticket-11-local-candidate-20260912-v5`, checksum
  `06fec9c9af3251ac62cb34ed5c8dbaf01dcbd330f0b19928eda177dac187bf4`
- notebook checksum `0e055d8f0c890c15e9f6d6eb4d2dd39cebc92ec8ba4309a3b3875d9d005cb3f5`
- adapter checksum `8af7a8888c90f1abe6964aab609966385e07201636d992e8f98e186d5b4c0f3b`
- failed before DICOM at approximately 346.55 seconds because NumPy also
  differed from its frozen pin

The v6 request added exact wheels for all three frozen packages. Their
filename and SHA-256 records are:

- `numpy-2.2.6-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` —
  `fd83c01228a688733f1ded5201c678f0c53ecc1006ffbc404db9f7a899ac6249`
- `pydicom-3.0.1-py3-none-any.whl` —
  `db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41`
- `torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` —
  `8f8b3cfc53010a4b4a3c7ecb88c212e9decc4f5eeb6af75c3c803937d2d60947`

The v6 request checksum was
`d346bf9dbe0b2e52d31457511ae1a396f7d1e47769df4673e20720c24229c373`, with
notebook checksum
`a014403f76219868401cf8cc62f26fa61775ef51a8474b4cc694e3b97c8bf7a1` and
adapter checksum
`07b4328659f0c14f5e6ebf93338541a6c4fb026045629a0d5698b61b53952684`. Kaggle
installed the wheelhouse, but v6 preserved a NumPy import failure caused by
reloading compiled packages in the live notebook process:
`AttributeError: 'numpy.ufunc' object has no attribute '__module__'`.

The v7 fresh-process delivery isolated the wheelhouse and ran the candidate in
a child interpreter. Its request checksum was
`4e2d9cfb0c16a329cf5a6c4b95cca0e27f788df8145aadf4288ac9fd6a8588d`, but the
child initially lacked the mounted delivery code path and failed at
approximately 283.71 seconds with `ModuleNotFoundError: No module named
'rsna_knee'`. The full v7 log, runtime evidence, and generated child script
are preserved.

The corrected v8 request and private delivery were then executed with internet
disabled and no leaderboard action:

- request `ticket-12-ticket-11-local-candidate-20260912-v8`, checksum
  `5ffe03d59a397185bb016c29a281f88028226723b63a025909d24da25e12005b`
- frozen candidate manifest checksum:
  `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`
- frozen checkpoint SHA-256 values:
  `5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76`,
  `4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d`,
  `808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8`
- notebook checksum `fa4dd69a50b5083d5d6a108f6437297bcef6ef390d33babf5349941557845c45`
- adapter checksum
  `07b4328659f0c14f5e6ebf93338541a6c4fb026045629a0d5698b61b53952684`
- Kaggle kernel: `jitheswar/ticket-12-kaggle-baseline-v8`
- runtime before installation: Python 3.12.13, NumPy 2.0.2, pydicom 3.0.2,
  torch 2.10.0+cpu
- runtime after isolated offline installation: Python 3.12.13, NumPy 2.2.6,
  pydicom 3.0.1, torch 2.7.1+cpu
- installation evidence: `--no-index`, `--no-deps`, local-wheel-only, and
  internet disabled
- actual hidden workload: 3 studies, 15 series, 557 slices, all 557 decoded
- whole-notebook runtime: 1,643.330900231 seconds
- linear projection at 1,300 studies: 712,110.0567667667 seconds; this is
  above the hard runtime limit and is not a hidden-scale performance pass
- decode/fallback failures: 0; fallback measured 0/3 for this run, with no
  claim about unobserved studies
- `submission.csv`: 3 rows, 13 columns, exact header and UID order, finite
  scores bounded to `[0, 1]`; SHA-256
  `e7a4dac875ce39e1c5945258e4341b64b47dbff62f01ec8fe36fd84dc2c13fb3`
- Kaggle acceptance: not claimed (`accepted_by_kaggle` is false)

The v8 execution JSON, private study diagnostics, runtime-install audit,
submission, and kernel log are retained under the private
`artifacts/private/ticket-12-kaggle-output-20260912-v8` directory. The v5,
v6, and v7 failed attempts remain retained as historical evidence.

## do4 runtime profile

The successful v8 run was profiled with a new v9 delivery/request and the
same frozen candidate. The v9 request checksum is
`2c02a4cff475a90bd6824e6af6cf7fcc4c0e75aca961b1ef3ba2b6a35b0b6ddc`, its
notebook checksum is
`bfe09da59f7618ffd52de84541399d8b59263a6d56b8a4cfae460ba7e22ffeb8`, and its
adapter checksum is
`b0772728e05b71a31c19860ed91f5cff9a41c25ec3189ddd7a2328b986760ece`. The
candidate manifest and all three checkpoint references are unchanged. v9
completed with internet disabled and produced the same `submission.csv`
SHA-256 as v8:
`e7a4dac875ce39e1c5945258e4341b64b47dbff62f01ec8fe36fd84dc2c13fb3`.

### v9 measured wall-clock profile

The wheel-verification, wheel-install and runtime-probe rows are sub-timers
of notebook startup/preflight. Adapter request validation, candidate loading,
and the runtime pin gate are sub-timers of adapter preflight; they are shown
separately to expose the repeated validation cost.

| Stage | Seconds |
| --- | ---: |
| Notebook startup/preflight | 321.909461893 |
| Offline wheel SHA verification | 3.311456167 |
| Offline wheel installation | 23.671774630 |
| Offline runtime probe | 2.704222118 |
| Child imports | 2.398322663 |
| Kaggle input discovery | 908.021352274 |
| Adapter preflight | 1.081053687 |
| — delivery request/approval/hash validation | 1.044922124 |
| — candidate loading/hash verification | 0.028172022 |
| — runtime pin gate | 0.006682184 |
| DICOM directory/index discovery | 0.300780700 |
| DICOM file read/decode | 10.004591415 |
| — DICOM read/parse and validation | 9.149307636 |
| — pixel decoding | 0.855283779 |
| Preprocessing/resize | 10.868352275 |
| Packing | 0.004594595 |
| Model inference/ensembling | 2.231096130 |
| CSV generation/validation | 0.005310799 |
| Total notebook runtime | 1,257.155148467 |

The v9 linear projection at 1,300 studies is 544,767.231002367 seconds,
also above both the engineering target and the hard limit; the profiling run
does not change the V16 gate state.

The run indexed 15 series and opened 557 DICOM files exactly once: 557 open
operations, 557 decode calls, and zero reopens. The private profile records
the exact per-study and per-series timings without exposing them here. Across
the three studies, DICOM read/decode time ranged from 1.461683252 to
7.051957565 seconds (median 1.567702761); across 15 series it ranged from
0.237414068 to 3.320441497 seconds (median 0.343043618). Series index scans
ranged from 0.006036744 to 0.133563898 seconds (median 0.010300643).

The v9 process saw four logical CPUs in its affinity mask, used two PyTorch threads and two
inter-op threads, and consumed 18.78 CPU seconds over 24.697 wall seconds for
the adapter process (76.0% process CPU utilization). Notebook startup used
27.70 recorded CPU seconds over 321.909 wall seconds (8.6%); these are process
measurements, not host-wide utilization.

The old v8 schema explicitly accounted for 326.241198836 seconds and left
1,317.089701395 seconds (80.15%) outside its stage breakdown. The v9 profile
assigns 1,233.437973680 seconds (98.11%) to fixed setup categories:
notebook startup, imports, input discovery, adapter preflight, and model
metadata loading. Workload-dependent directory indexing, DICOM access,
preprocessing, inference and CSV work total 23.414725914 seconds (1.86%), or
0.042037210 seconds per decoded slice and 7.804908638 seconds per study,
excluding fixed setup. The remaining 0.302448873 seconds is timer/loop and
unattributed process overhead.

### Direct comparison with Ticket 11 local execution

Ticket 11 measured 604.504043193 seconds for 158 studies, 965 series, and
31,486 decoded slices. Its runner timer starts after manifest/archive/model
setup; model loading itself was 0.034 seconds. That gives approximately
0.019199137 seconds per slice and 3.825974957 seconds per study. Thus the v9
Kaggle workload path is about 2.19x slower per slice and 2.04x slower per
study, but the whole-notebook v8 ratio of 153.7x per slice is dominated by
fixed Kaggle delivery/setup overhead rather than inference.

The local path opens one ZIP handle and processes the checksum-covered
manifest serially. For each slice, `_SourceReader.decode` opens the ZIP member
once to hash it and reopens it once to decode it, so local access is two serial
member reads per slice. The Kaggle path is directory-backed rather than ZIP-
backed: it recursively indexes each series once, then opens each DICOM once
and decodes serially. No repeated DICOM rescan or reopen was observed in v9.
Within each study, preprocessing packs the selected slices/series into one
padded tensor and the three-model ensemble runs once; studies are processed
sequentially. Inference is therefore batched within a study but not across
studies, and its 2.231-second total is not the bottleneck. No reusable decoded
or preprocessed-image cache is enabled in `/kaggle/working`; the working
directory is used only for the isolated runtime site and evidence/output.

The exact bottleneck is the first access to the mounted delivery wheelhouse
during `discover_kaggle_inputs`: that path performs full SHA-256 reads of the
three-wheel, approximately 185.7 MiB wheelhouse. v9 then repeated those reads
through `load_kaggle_delivery`, and the run path repeated request/runtime and
candidate checks again. The candidate checkpoint/code hashes are small and
measured at 0.028 seconds, while DICOM, pixel decoding, preprocessing and
inference together are only a few tens of seconds. The v9 input-discovery timer
of 908.021 seconds is therefore the dominant measured delay; the exact split
between the remote-mounted wheel reads is not separately exposed by v9, but
the reads are explicit in the checksum-validation call path.

### Applied narrow optimization

The approved v10 change carries one checksum-verified validation result from
`discover_kaggle_inputs` into `run_kaggle_baseline`. Discovery now performs the
request, full candidate, notebook, adapter, wheelhouse, code-mirror, and owner
approval checks once, then the run reuses that immutable in-process result only
when every candidate, competition, request, approval, notebook, adapter, and
wheel path matches exactly. The run records the reuse mode and preserves the
one-time candidate and approval hashes in its detailed profile.

Direct callers that do not pass a discovered layout retain full request,
approval, artifact, and candidate revalidation. The installed-runtime pin gate
still runs immediately before inference in both paths. No frozen candidate,
checkpoint, preprocessing, model call, prediction, output contract, or
leaderboard behavior changed. A private v10 delivery and kernel were launched
with request checksum
`4d5ea3b739721406b7f3b4f91f508caa18b898b7e1b5f4932d52fda2f3a12a70`, notebook
checksum
`8f6ffe6e9a6b8a7020757310d3b8e40a6eec2dc77746fcce32d4ea8bc7f79e8c`, adapter
checksum
`cd4e4f0459cd424575f1681534194528da388ac825d1efccc9ba77a13b2e3d40`, and the
same frozen candidate/checkpoint set. Its Kaggle run remains private and
submission-free. The v10 kernel completed with status
`completed_kaggle_notebook`, no failure reasons, and the reuse flags
`delivery_validation_reused=true`, `candidate_hashes_reused=true`,
`runtime_wheel_hashes_reused=true`, and `approval_hash_reused=true`. Its output
passed the local contract and is byte-identical to v8/v9. The complete v10
notebook took 2,427.300282192 seconds for the same 3-study workload; this is
not an end-to-end speedup because Kaggle startup and input discovery were
slower in this sample.

The v9 profile artifacts are retained under
`artifacts/private/ticket-12-kaggle-output-20260912-v9`. Their SHA-256 values
are: `kaggle-execution.json`
`cbc9db5f18abe3dff2119e7b4873165ceafceb3403578d3df4d06e0259d117fb`,
`private-execution.json`
`43ce4d6e4e9c58e43a84af7a8dd1f25e176a7a3af390bae8a69f4628a13af86c`,
`runtime-install.json`
`934137baba2e62a118afcaf1f6afc9f455c877d033f82f96a87ddd7e093e4a5d`,
`submission.csv`
`e7a4dac875ce39e1c5945258e4341b64b47dbff62f01ec8fe36fd84dc2c13fb3`, and
the v9 kernel log
`d007775572a4fa0d888b69b2ad6d93810f9f9e33e69d4814b9a01a9e9328fe13`. The v10
artifacts are retained under
`artifacts/private/ticket-12-kaggle-output-20260912-v10`; their key SHA-256
values are `kaggle-execution.json`
`5a13e7c9bf0f26ffba1f81a0994d70e24295f5d0990dc5b593d1695c84c19881`,
`private-execution.json`
`9a8e663336f80220c57312fadb465f34a3088c87c3bb3ee9dffa4fd5df2832d3`,
`runtime-install.json`
`934137baba2e62a118afcaf1f6afc9f455c877d033f82f96a87ddd7e093e4a5d`,
`submission.csv`
`e7a4dac875ce39e1c5945258e4341b64b47dbff62f01ec8fe36fd84dc2c13fb3`, and
the v10 kernel log
`c7b613c9ade74818734f5c67e1607aa0abae556acb862cabb24f9b3025408266`.

## v12 larger representative-workload preparation

The competition metadata available locally exposes only three official test
studies, so the next verification step uses a separate private representative
input rather than claiming hidden-scale access. The new input is bound by
`input-manifest.json` and contains:

- 158 studies, 965 series, and 31,486 DICOM files/slices;
- 6.107594937 average and 14 maximum series per study;
- 199.278481013 average and 601 maximum slices per study;
- a reproducible nearest-rank p90 long-tail rule: 9 or more series or 326 or
  more slices, covering 26 studies. The carried-forward Ticket 11 scope also
  records its separate 100-study long-tail declaration; this execution reports
  the computed rule explicitly rather than conflating the two definitions.

The input manifest checksum is
`1f579e432fc67b9fa69cf5386fed7d512b768e3a71811a11a4c1f0ecc609aae5`.
The fresh request is `ticket-12-ticket-11-local-candidate-20260912-v13` with
request checksum
`16bf6733c586bacaa46248ecb444f756c0521891ebe1f66de38190836db91a2b`. It is
bound to the unchanged Ticket 11 candidate manifest
`cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`, the v13
notebook checksum
`1e958fbd0c789825b30a583cdf70224a8599ee7412f8d86764af8d9b2e28e54f`, and the
v13 adapter checksum
`0af614463e0cc97c497fabe6c531206a06097ff81d2196837ff39ffba5e647d7`. The
offline wheelhouse remains pinned to NumPy 2.2.6, pydicom 3.0.1, and torch
2.7.1+cpu with the previously audited wheel checksums. The delivery mirror
reloads successfully against the exact request.

The adapter now records the full startup-to-CSV stage breakdown, CPU/GPU and
memory evidence, model/checkpoint/delivery/runtime sizes, actual workload
distribution, output missing/duplicate/decode counts, and a fixed-plus-
measured-scaling projection. The delivery notebook also contains the same-path
save/reload, intentional interruption, resume, lost-work, rollback, and final
output comparison probe. The 172-test repository suite passes; targeted mypy
for the delivery adapter passes.

Before the gated Kaggle run, the exact directory-backed adapter was exercised
locally on this representative input. It decoded all 31,486 files with zero
decode failures and produced 158 valid thirteen-column rows with exact UID
order, finite/bounded scores, zero missing rows, zero duplicate rows, and
fallback 0/158. Observed wall time was 388.982 seconds; the corrected
fixed-plus-measured-scaling projection at 1,300 studies was 3,199.987 seconds,
leaving 22,720.013 seconds to the 25,920-second engineering target and
29,200.013 seconds to the 32,400-second hard limit. This is a local
directory-path preflight only: it is not Kaggle execution, and its fallback
rate remains explicitly unknown for hidden data.

For context, the prior three-study v10 linear projection was
1,051,830.122 seconds; the representative preflight reports the fixed/scaling components
separately rather than treating its fixed setup as study-proportional. This
comparison is diagnostic only because the v10 run was on Kaggle and this
preflight was local.

The preflight was followed by the explicitly approved private execution. The
owner approval artifact is
`artifacts/private/ticket-12-kaggle-approval-dataset-20260912-v13/owner-approval.json`,
with SHA-256
`f34ed2bad3187099b9ec9d0e2aabeb6c4de05d77d6250954482aa6a927a8589b`. It names
request checksum
`16bf6733c586bacaa46248ecb444f756c0521891ebe1f66de38190836db91a2b`, records
the rules and overview recheck on 2026-09-12, and authorizes the exact private
bundle/input execution. The request itself remains submission-disabled
(`upload.submission_allowed=false`); no leaderboard submission was performed.

The exact private Kaggle datasets were uploaded and became ready:

- delivery: `jitheswar/rsna-knee-ticket12-delivery-v13`;
- approval: `jitheswar/rsna-knee-ticket12-approval-v13`;
- representative input: `jitheswar/rsna-knee-ticket12-representative-v12`,
  version 1, status `Ready`, 26,202,240,381 bytes.

The private kernel is
`jitheswar/ticket-12-kaggle-v13-representative-workload`. Versions 1--3 were
created while Kaggle was still processing the large input and did not provide
qualifying full-workload evidence because Kaggle omitted that unresolved source.
Version 4 attached all three sources and completed. Its saved execution
evidence is
`artifacts/private/ticket-12-kaggle-output-20260912-v13/kaggle-execution.json`
with SHA-256
`bf334023a63dd817af24d44fb7d39dd7870c5243f92da4ad19275cdd2dbf6a59` and
status `completed_kaggle_notebook`.

The completed v13 run used the exact request, candidate, notebook, adapter,
checkpoint set, and offline runtime pins. Kaggle reported Python 3.12.13 on
four logical CPUs, with internet disabled and no GPU. The installed/imported
runtime was NumPy 2.2.6, pydicom 3.0.1, and torch 2.7.1+cpu. It processed 158
studies, 965 series, and 31,486 DICOM slices/files. The p90 long-tail rule
remained 9 series or 326 slices, covering 26 studies.

The output contract passed locally after download: 158 rows, 13 columns, exact
header, exact UID order, finite and bounded scores, zero missing rows, zero
duplicate rows, and zero DICOM decode failures. `submission.csv` is 47,078
bytes with SHA-256
`66761efb64a74b7c4d9e426861799622b711247ae2e896f42908833fde62d586`.
The private per-study evidence is preserved in
`private-execution.json` (SHA-256
`c13c0bc06406dbcd7f8c59687197012d642145ceba7644ec03cc3436ef3a9444`) and
reports measured fallback 0/158 with a complete investigation and no unknown
hidden-failure flag.

The whole notebook wall time was 2,932.068918 seconds. The fixed-plus-
measured-scaling projection at 1,300 studies was 9,462.946510 seconds,
leaving 16,457.053490 seconds to the 25,920-second engineering target and
22,937.053490 seconds to the 32,400-second hard limit. The run's peak RAM was
26,716,372,992 bytes and peak VRAM was zero. The v17 recovery probe saved and
reloaded a checkpoint, intentionally exited with code 75 after 17 rows, resumed
to 158 rows, lost zero rows, and produced byte-identical source and recovered
outputs. Its evidence SHA-256 is
`c2268ab54142d6f5c31674817b066d6c777152b7e943ee16ea8d7dfe74788c53`.

The current official sources used for the owner recheck were the [competition
rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
and [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview).
These facts establish the approved private representative run. Formal Kaggle
platform acceptance is intentionally not claimed because the request forbids a
leaderboard submission; the execution evidence therefore retains
`accepted_by_kaggle=false` while recording `accepted_by_local_contract=true`.

## Verification state

| Gate | State | Reason |
| --- | --- | --- |
| V08 exact Kaggle bundle/output | private execution/output contract passed; platform acceptance intentionally not claimed | v13 completed with the exact checksum-bound bundle and produced a locally validated 158-row output. `accepted_by_kaggle=false` is retained because no leaderboard action was authorized or performed. |
| V12 rules/account/eligibility | passed for the approved private v13 execution | The exact v13 request checksum, current rule sources, account eligibility, upload permission, and private execution scope are preserved in the owner approval artifact. |
| V16 whole-notebook runtime | passed for the representative projection | v13 measured 2,932.068918 seconds whole-notebook time and projected 9,462.946510 seconds at 1,300 studies, below both the 25,920-second engineering target and 32,400-second hard limit. |
| V17 rows/fallback/recovery | passed for the representative workload | v13 produced 158 exact rows, measured 0/158 fallback, zero decode failures, and verified save/reload plus interruption/recovery with 17 checkpoint rows, zero lost rows, and byte-identical output. |

The ticket-12 test suite covers request immutability, exact candidate matching,
owner-approval scope, accepted-output verification, unknown-fallback
preservation, Kaggle-layout smoke inference, pre-run approval/hash gates,
private-report validation, multi-wheel runtime binding, fresh-process notebook
execution, discovered-validation reuse, fail-closed layout binding, and
promotion immutability. The local suite passes 172 tests; targeted mypy for the
delivery adapter passes. The smoke run is synthetic and is not Kaggle
verification.

## Closure state

The v13 owner-approved private execution is complete, and the representative
evidence gates V08 (execution/output contract), V12, V16, and V17 are satisfied
as described above. No competition submission, formal leaderboard acceptance,
promotion, final selection, or Ticket 13 work was performed. The ticket is not
fully closeable as a formally Kaggle-accepted competition delivery while the
owner's explicit no-leaderboard constraint remains in force; the remaining
state is an intentional platform-acceptance limitation, not a failed runtime,
output, rules, resource, fallback, or recovery gate.

The paragraph above is the pre-closure interpretation recorded immediately
after execution. The exact Ticket 12/V08 wording was then audited on
2026-09-12. It requires an explicitly approved Kaggle execution to complete
and generate a contract-valid `submission.csv`; it does not explicitly require
that the file be submitted to the competition leaderboard. Its wording that
“any new upload/submission remains subject to approval” makes submission an
approval-controlled action, not a stated closure prerequisite. The parent
specification uses the same execution-in-environment distinction and keeps
final status separate through owner selection.

## Ticket 12 closure audit — 2026-09-12

Private v13 execution is sufficient to close Ticket 12 at its execution-
verification level. The exact approved request, immutable candidate bundle,
offline runtime, Kaggle execution, representative workload, output contract,
runtime projection, fallback accounting, and interruption recovery are
recorded above and bound by the following final evidence:

| Evidence | Identity/result |
| --- | --- |
| Request | `ticket-12-ticket-11-local-candidate-20260912-v13`; SHA-256 `16bf6733c586bacaa46248ecb444f756c0521891ebe1f66de38190836db91a2b` |
| Owner approval | SHA-256 `f34ed2bad3187099b9ec9d0e2aabeb6c4de05d77d6250954482aa6a927a8589b`; exact private execution scope approved |
| Kaggle execution | Kernel `jitheswar/ticket-12-kaggle-v13-representative-workload`, version 4; status `completed_kaggle_notebook`; evidence SHA-256 `bf334023a63dd817af24d44fb7d39dd7870c5243f92da4ad19275cdd2dbf6a59` |
| Output | `submission.csv`; 158 rows × 13 columns; exact header/UID order; finite/bounded scores; zero missing/duplicate rows; SHA-256 `66761efb64a74b7c4d9e426861799622b711247ae2e896f42908833fde62d586` |
| Runtime | 2,932.068918 seconds observed; 9,462.946510 seconds projected at approximately 1,300 studies; below 25,920-second target and 32,400-second hard limit |
| Fallback/recovery | Measured 0/158, complete investigation; save/reload and interruption recovery verified; zero lost rows; recovery evidence SHA-256 `c2268ab54142d6f5c31674817b066d6c777152b7e943ee16ea8d7dfe74788c53` |

### Promotion boundary

The ticket-level promotion is recorded as `private-kaggle-execution-verified`
with the exact bundle preserved and no mutation of the Ticket 11 candidate or
request. It is not a final candidate: `final_candidate=false`, owner selection
is still required, and no export or competition submission is authorized by
this closure record. `accepted_by_kaggle=false` remains an accurate historical
platform field because no leaderboard submission occurred; it is not treated
as a failed V08 closure gate because V08 does not explicitly require that
action. Hidden competition runtime and hidden fallback behavior remain
unknown, as required by V16/V17.
