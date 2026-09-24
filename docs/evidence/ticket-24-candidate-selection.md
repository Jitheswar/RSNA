# Ticket 24 candidate selection evidence

Recorded 2026-09-23. This public-safe record contains aggregate execution
measurements and artifact checksums only; it contains no study identifiers,
source rows, reports, private paths, or per-study failure details.

## Owner selection

The owner selected the Ticket 23 recommendation: preserve and select the exact
existing three-fold gold-only candidate
`ticket-11-local-candidate-20260911-v1`, manifest SHA-256
`cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`.
The frozen candidate was not modified. The selection is recorded here and in
Ticket 24, separately from its immutable manifest.

Ticket 23 found no validated development result that justified replacing this
candidate. The gold-only fold OOF pooled macro AUC was 0.464313 (study-bootstrap
95% interval 0.404–0.524), compared with 0.474353 for the fold-prevalence-only
comparator; the predeclared within-fold permutation test gave p=0.918. The
OOF predictions do not directly measure the deployed three-model average, and
no all-data refit was run. The choice preserves the existing tested delivery
path; it is not a claim of high predictive quality or superiority to an
unmeasured refit.

## Identity and prior execution

The approved Ticket 12 v13 representative execution was bound to the selected
candidate and exact delivery artifacts:

| Artifact | Identity |
| --- | --- |
| Candidate | `ticket-11-local-candidate-20260911-v1`; manifest SHA-256 `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc` |
| Kaggle request | `ticket-12-ticket-11-local-candidate-20260912-v13`; SHA-256 `16bf6733c586bacaa46248ecb444f756c0521891ebe1f66de38190836db91a2b` |
| Owner approval record | SHA-256 `f34ed2bad3187099b9ec9d0e2aabeb6c4de05d77d6250954482aa6a927a8589b`; scope `ticket-12-exact-bundle-upload-and-execution` |
| Three fold checkpoints | SHA-256 `5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76`, `4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d`, `808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8` |
| Delivery notebook | SHA-256 `1e958fbd0c789825b30a583cdf70224a8599ee7412f8d86764af8d9b2e28e54f` |
| Kaggle adapter | SHA-256 `0af614463e0cc97c497fabe6c531206a06097ff81d2196837ff39ffba5e647d7` |
| Kaggle execution evidence | SHA-256 `bf334023a63dd817af24d44fb7d39dd7870c5243f92da4ad19275cdd2dbf6a59` |
| Representative output | 158 rows × 13 columns; SHA-256 `66761efb64a74b7c4d9e426861799622b711247ae2e896f42908833fde62d586` |
| Recovery evidence | SHA-256 `c2268ab54142d6f5c31674817b066d6c777152b7e943ee16ea8d7dfe74788c53` |

The execution used an internet-disabled CPU environment with the pinned NumPy
2.2.6, pydicom 3.0.1 and torch 2.7.1+cpu runtime. The owner approval records
that the exact v13 request was approved for upload and execution, with rules
acceptance and account eligibility checked on 2026-09-12. The v13 request
itself had `upload.submission_allowed=false`, and no leaderboard submission
was performed. This evidence is specific to the v13 request and its
representative workload; a changed bundle, input, or workload needs its own
approval and execution evidence.

## Measured scope

The v13 run processed 158 studies, 965 series and 31,486 DICOM files/slices.
Its output passed the local contract: exact header and UID order, finite scores
in `[0, 1]`, zero missing or duplicate rows, and zero decode failures. The
complete investigation measured fallback at 0/158. The notebook's recovery
probe serialized 17 rows from an already-completed output and appended the
remaining rows from that same output; it reproduced the CSV byte-for-byte.
This verifies file reconstruction, not interruption and resumption of
inference, so inference-level recovery remains unverified.

Whole-notebook runtime was 2,932.068918 seconds. The recorded fixed-plus-
measured-scaling projection at approximately 1,300 studies was 9,462.946510
seconds, below the 25,920-second engineering target and 32,400-second hard
limit. Peak RAM was 26,716,372,992 bytes and peak VRAM was zero. The execution
bundle was 940,350 bytes and the three model checkpoints totaled 314,944
bytes.

These figures describe the 158-study representative workload. They do not
establish hidden-set runtime, output completeness, or fallback behavior.
`accepted_by_kaggle=false` remains accurate: no leaderboard submission was
performed. The saved file passed the local output contract only.

## Current competition requirements

The [official evaluation overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
was checked on 2026-09-23. It lists macro-averaged ROC AUC over twelve targets,
an internet-disabled CPU or GPU notebook limit of nine hours, the required
`submission.csv` name, the October 15, 2026 entry/team deadline, and the
October 22, 2026 final submission deadline (both at 11:59 PM UTC). The
[competition rules page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
returned no readable page body through the available public fetch. The v13
approval's rules and eligibility check is dated 2026-09-12; current account
eligibility and the complete current rules text were not independently
reverified for this selection record.

## Selection boundary

This section records the owner's candidate selection and the state before the
later delivery-only authorization below. It does not claim a full hidden-test
output or platform acceptance. Hidden-set runtime and fallback rates remain
unknown.

## Hidden-test submission request — 2026-09-23

The owner requested exactly one full hidden-test Kaggle submission and
explicitly prohibited model or pipeline changes. Preflight found that the only
frozen delivery notebook is the Ticket 12 v13 notebook. It hard-codes request
`ticket-12-ticket-11-local-candidate-20260912-v13`; that request declares
`workload_kind=representative`, selects mount
`rsna-knee-ticket12-representative-v12`, and pins the representative manifest
and CSV hashes. The notebook passes that mount slug to input discovery, which
validates those hashes before inference. Reusing this version for a hidden
submission would use the attached 158-study representative input or fail its
input checks instead of running the full hidden set.

At this preflight stage no new binding had been made. No Kaggle kernel push or
competition submission was made. The account's
read-only submission listing returned no prior submissions. Kaggle's [code
competition debugging guide](https://www.kaggle.com/code-competition-debugging)
states that each submission reruns the notebook against a hidden version of
the competition data and that failed submissions count toward the team's
daily limit. The single requested attempt was preserved. A successful full
hidden run requires a new hidden-test request/notebook binding. Consequently no full
hidden-test acceptance, leaderboard score/runtime, or hidden fallback evidence
exists. The only fallback evidence remains the prior representative 0/158.

## Delivery-only hidden-test preflight — 2026-09-23

The owner then explicitly authorized local preparation of a new Ticket 24
delivery-only request and notebook, with the selected model, checkpoints,
preprocessing, and inference behavior unchanged. The request is bound to
`workload_kind=hidden_test` and the official `rsna-knee-abnormality-detection`
competition mount. It intentionally does not pin a test manifest or test CSV
hash because Kaggle selects the hidden competition version at scoring time.

| Artifact | Identity |
| --- | --- |
| Request | `ticket-24-ticket-11-local-candidate-20260923-hidden-v2`; checksum `663e3da7034c8158bb3953366e373735a524fa7ad1f68bf11f909e89f15df802` |
| Candidate | `ticket-11-local-candidate-20260911-v1`; manifest SHA-256 `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc` |
| Fold checkpoints | `5d71865ee26f36d6d447bafe28d8c6c6b9d7a18f63aad31c7df8f9532f530c76`, `4b91ab25ef3d3e7ba45dd54c56c30122b0196f6e07b64cc40ebba34df7f4095d`, `808813e7ac465aa2f02c4115145f9928a56bd7a2cd502ab3486989f53a7592e8` |
| Delivery notebook | `ticket24_hidden_test_delivery.ipynb`; SHA-256 `1ef62c08556962305153a6028c195ca68bffa8abf8fe7a74f822abbbc259d04a` |
| Delivery adapter | SHA-256 `c672f553d37f737f3894e1ccc3006f0490786cbe60e68ba07113f82edd691bc1` |
| Offline runtime | NumPy `2.2.6`, pydicom `3.0.1`, torch `2.7.1+cpu`; all three pinned wheel hashes were rechecked in the bundle |
| Delivery dataset draft | `jitheswar/rsna-knee-ticket24-hidden-delivery-v2`; private local metadata SHA-256 `44e6db581939b536611478e8e4540260266116435cc477333b8d1305efe669cb` |
| Kernel metadata draft | `jitheswar/ticket-24-hidden-test-delivery-v2`; SHA-256 `d220a2317a8894329fb9c95aa9e899820dfc9702e40263cf1f964eb0f9828bdc` |

The delivery adapter changes only the request-bound approval scope and input
binding: hidden-test requests identify the official competition mount but do
not pin replaceable hidden files. Candidate model, checkpoints, decoder,
preprocessor, model inference, probability aggregation, and strict submission
writer remain the frozen Ticket 11 versions. The runtime remains CPU-only and
internet-disabled. Request, notebook, adapter, candidate mirror, and offline
wheel checks passed locally.

The final notebook does not reproduce a recovery probe by copying rows from an
already completed CSV. Its adapter evidence leaves inference-level save/reload
and interruption recovery unverified; V17 recovery remains open. The first
local v1 draft was retained as superseded and was never uploaded.

The existing private local candidate output was revalidated against its
expected study UID order: 158 rows, 13 columns, exact header and UID order,
finite scores within `[0, 1]`; output SHA-256
`26dffbdb06f8d2d6b7410597067fafff299010d6d554b0535cf8f79e953bee1d`. This is
local contract evidence only. The hidden output, hidden UID set, full-set
runtime, fallback rate, and Kaggle acceptance remain unknown until the exact
approved notebook is executed.

The prepared request has `approved=false`, `upload.allowed=false`, and
`submission_allowed=false`. No approval artifact was created. The local
kernel metadata references the exact-request approval dataset that must be
created after owner approval; it is not present in the delivery bundle. No
Kaggle push or competition submission was performed. Stop at this boundary
until the owner explicitly approves the exact request and current rules and
eligibility are rechecked.

## Source records

- Ticket 23 decision: `.scratch/rsna-2026-fresh/issues/23-compare-fold-ensembles-with-all-data-refitting.md`
- Ticket 23 comparison: `docs/evidence/ticket-23-delivery-strategy.md`
- Ticket 12 Kaggle record: `docs/evidence/ticket-12-kaggle.md`
- Ticket 24 selection and remaining gates: `.scratch/rsna-2026-fresh/issues/24-select-and-deliver-the-final-candidate.md`

## Exact-request approval preflight — 2026-09-23

The owner approved the exact v2 request checksum
`663e3da7034c8158bb3953366e373735a524fa7ad1f68bf11f909e89f15df802` and
authorized a private Kaggle push, execution, and submission after a current
rules/account check. The canonical request checksum revalidated. Before any
upload, inspection found that the request-bound notebook SHA-256
`1ef62c08556962305153a6028c195ca68bffa8abf8fe7a74f822abbbc259d04a`
hard-codes request ID `ticket-24-ticket-11-local-candidate-20260923-hidden-v1`,
while the approved request ID is `...hidden-v2`. The notebook bootstrap
requires an exact ID match and would stop before inference. The v2 request was
therefore not uploaded or submitted.

The current official [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
still specifies a CPU/GPU notebook runtime of at most nine hours, internet
disabled, and an output named `submission.csv`; it lists the October 22, 2026
final submission deadline. The official [rules page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
returned an HTML shell with no readable rules body. Authenticated
`kaggle competitions files` access succeeded for this account, and the
read-only submission listing returned no prior submissions. The account can
currently access the competition data; the unavailable full rules text is not
treated as independently reconciled.

A separate local v3 draft was staged at
`artifacts/private/ticket-24-kaggle-delivery-20260923-v3/` and
`artifacts/private/ticket-24-kaggle-kernel-20260923-v3/`. It changes the
notebook's request ID constant/heading and the corresponding request and
Kaggle metadata bindings only. The corrected notebook SHA-256 is
`6a4249075d75eb8cefcc88abf9bdbd7268b69f597d8ded7d8dd27cec5c02b927`; request
ID `ticket-24-ticket-11-local-candidate-20260923-hidden-v3` has canonical
checksum `f95b22401d553e4b3fff73f1282faf8d046584098eea23a9ebe87a907597bff5`.
The frozen candidate manifest, all three checkpoints, and adapter hashes are
unchanged. The local v3 request remains approval-gated (`approved=false`,
`upload.allowed=false`, `submission_allowed=false`) and has not been uploaded.
It is not covered by the owner's v2 approval. No kernel push, hidden-test
execution, or competition submission has occurred.

## V3 approval and current preflight — 2026-09-23

The owner explicitly approved request checksum
`f95b22401d553e4b3fff73f1282faf8d046584098eea23a9ebe87a907597bff5` for a
private upload, Kaggle kernel execution, and one competition submission. The
approval is recorded in `/home/ch.jitheswar/tsk.txt`. Local
`load_kaggle_delivery` validation passed for the exact v3 request and these
bindings:

- Candidate: `ticket-11-local-candidate-20260911-v1`, manifest SHA-256
  `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc`.
- All three checkpoint hashes match the request and frozen candidate.
- Delivery notebook SHA-256
  `6a4249075d75eb8cefcc88abf9bdbd7268b69f597d8ded7d8dd27cec5c02b927`.
- Adapter SHA-256
  `c672f553d37f737f3894e1ccc3006f0490786cbe60e68ba07113f82edd691bc1`.
- All three pinned offline runtime wheel hashes match the request.
- The request selects `workload_kind=hidden_test` on the official competition
  mount, with CPU execution and internet disabled.

The current official [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
still specifies notebook-only submissions, a nine-hour CPU/GPU notebook cap,
internet disabled, the `submission.csv` name, and an October 22, 2026 final
deadline. The current Kaggle competition API reports the same deadline and
submission constraints, five submissions per day, and
`requiresIdentityVerification=True`. Authenticated read-only Kaggle CLI checks
show the account has entered the competition (`userHasEntered=True`) and has
no prior submissions, but the CLI does not expose whether the account has
completed identity verification. The live
[rules route](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
returned an HTML shell without readable rules text. The previous Ticket 12
owner approval records rules acceptance and eligibility checked on
2026-09-12; today's full rules changes and identity-verification status could
not be reconciled.

The Ticket 24 approval dataset was not created. No private upload, Kaggle
kernel push/execution, or competition submission was made. The exact local
binding passed, but the V12 rules/eligibility and identity-verification gate
remains incomplete. Hidden runtime, hidden fallback rate, Kaggle acceptance,
and leaderboard score remain unknown; no hidden-test evidence is inferred
from the representative run.

## V3 private upload, kernel run, and one submission — 2026-09-24 IST

This post-action record supersedes the preceding v3 preflight status; the
preflight notes describe the state before the owner's current-rules and
eligibility attestation and before remote actions.

The owner stated in `/home/ch.jitheswar/tsk.txt` that the Kaggle account was
identity-verified and eligible and that the current rules had been manually
reviewed and accepted. The exact v3 checksum
`f95b22401d553e4b3fff73f1282faf8d046584098eea23a9ebe87a907597bff5` was
approved for a private upload, kernel execution, and one submission. A
separate exact-request owner approval was created at
`artifacts/private/ticket-24-kaggle-approval-dataset-20260923-v3/` with
approval-file SHA-256
`e91097463de8cc2450088f40e353146ead479ba72d86ec7b2305d23617be138a`.

Both Kaggle datasets were created privately and reached `ready`:

- Delivery: `jitheswar/rsna-knee-ticket24-hidden-delivery-v3`.
- Owner approval: `jitheswar/rsna-knee-ticket24-approval-v3`.

The delivery dataset was downloaded back from Kaggle and all 77 uploaded files
matched the local v3 bundle byte-for-byte. The kernel
`jitheswar/ticket-24-hidden-test-delivery-v3` was pushed as private version 1,
with internet and GPU disabled and the competition mount attached. Kaggle
reported the kernel run as `COMPLETE`.

The kernel's aggregate evidence, preserved under
`artifacts/private/ticket-24-kaggle-output-20260923-v3/`, records 3 studies,
15 series, and 557 slices from the competition mount. The output contract
passed with 3 rows and 13 columns, exact header, finite bounded scores, zero
missing or duplicate rows, and zero decode failures. The CSV SHA-256 is
`e7a4dac875ce39e1c5945258e4341b64b47dbff62f01ec8fe36fd84dc2c13fb3`. Fallback
was measured at 0/3 for this run. Whole-notebook time was 2,796.861 seconds,
peak RAM was 1,047,113,728 bytes, and VRAM use was zero. The notebook projects
12,984.607 seconds at 1,300 studies, below the 25,920-second engineering
target and 32,400-second hard limit. This 3-study run does not provide actual
full scoring-set runtime or fallback evidence.

After confirming there were no prior submissions, exactly one code-competition
submission was sent using kernel version 1 and `submission.csv`. Kaggle assigned
submission ref `56500900`; its status is `SubmissionStatus.COMPLETE`. The API
returned blank public and private score fields. The submission record does not
expose the full scoring CSV, per-study execution diagnostics, full-set runtime,
or full-set fallback rate. No second attempt was made. The status response is
preserved at `artifacts/private/ticket-24-kaggle-output-20260923-v3/submission-status.json`
with SHA-256 `c2f844dfa4e809bf9f79d2b2f33a293fffdcaf44831696cba16c6ca2914fa012`. Kaggle's
[code-competition debugging guide](https://www.kaggle.com/code-competition-debugging)
describes the private hidden-data rerun and output validation used for code
competition submissions; the local execution evidence above is kept separate
from the completed submission result.

The frozen Ticket 11 candidate manifest and all three checkpoint hashes are
unchanged. No model, preprocessing, or inference behavior was modified.
Inference recovery remains unverified and no leaderboard score is claimed.

## V3 scoring failure correction and v4 memory validation — 2026-09-24

The owner clarified that the v3 full hidden scoring attempt failed with Kaggle
`Notebook Out of Memory`. The preserved submissions API status
`SubmissionStatus.COMPLETE` is a submission workflow status and does not prove
that Kaggle successfully scored the notebook. The locally archived v3 execution
log and `kaggle-execution.json` describe a separate 3-study dry run; they do not
contain the full scoring OOM trace. The v3 request, notebook, kernel metadata,
logs, submission reference `56500900`, hashes, and existing failure status were
left unchanged. A private aggregate status record is at
`artifacts/private/ticket-24-v4-validation/v3-failure-status.json`.

### Diagnosis

The previous adapter's `_load_kaggle_studies` decoded the complete cohort into
`StudyRecord` objects before inference. Each record kept its decoded slice
pixel arrays alive, so resident memory grew with the total cohort. A same-host
Slurm rerun of the archived v3 adapter on the exact 158-study representative
input peaked at 26,773,291,008 bytes RSS, close to the archived v13 peak of
26,716,372,992 bytes. The previous archived v13 evidence did not include
stage-level RSS samples; the same-host rerun reproduced the cohort-scale peak.
Model weights total 314,944 bytes in the prior record and are not a plausible
source of a 26 GB peak.

### Delivery-only correction

V4 reads study and series metadata once, decodes one study, predicts it,
commits the scores to a checksum-bound private checkpoint, releases the study
and its arrays, then decodes the next. It retains only small prediction rows
and aggregate diagnostics. The active-study bound is one. Candidate weights,
architecture, preprocessing, slice and series selection, probability
aggregation, fallback values, target order, and CSV meaning remain frozen.

Exact files changed for the v4 work:

- `src/rsna_knee/kaggle.py` — stream one decoded study at a time; collect
  per-stage and per-study RSS; add candidate/input-bound, row-checksummed
  prediction checkpoints with safe truncation of an incomplete trailing row;
  and preserve exact scores through save/load and resume. Also preserve the
  total series count in execution evidence instead of overwriting it with the
  final study's count.
- `tests/test_kaggle.py` — cover study release, bounded active studies,
  checksum-bound interruption/resume with byte-identical CSV, and correct
  aggregate series counts.
- `notebooks/ticket24_hidden_test_delivery_v4.ipynb` — copy the delivery-only
  v3 flow with a new v4 request ID and matching title; no model logic changes.
- `scripts/ticket24_v4_representative_validation.py` — reproduce and validate
  the full local workload, same-host output comparison, aggregate memory
  report, and recovery evidence.
- `scripts/ticket24_v4_previous_adapter_comparison.py` — run the preserved v3
  adapter locally on the same input/runtime to compare predictions and measure
  its cohort-retention peak.
- `scripts/ticket24_v4_artifact_preflight.py` — verify request, candidate,
  notebook, adapter, code mirror, wheel hashes, hidden-input binding, and
  private/disabled kernel settings without requiring or creating approval.
- `slurm/run_ticket24_v4_validation.sbatch` — run the full repository tests,
  mypy, compilation, shell syntax checks, and representative v4 inference on
  `cluster1-cpu` within the ticket's CPU and memory limits.
- `slurm/run_ticket24_v4_artifact_preflight.sbatch` — compile the final helper
  scripts, check Slurm shell syntax, and run request/notebook preflight on
  `cluster1-cpu`.
- `.scratch/rsna-2026-fresh/issues/24-select-and-deliver-the-final-candidate.md`
  and this evidence file — correct the v3 outcome and record the v4 proof while
  keeping Ticket 24 open.

The frozen candidate directory and Ticket 25 were not modified.

### Representative v4 results

The local v4 adapter processed 158 studies, 965 series, and 31,486 slices. All
31,486 expected slices decoded; there were zero decode failures and zero
fallback studies. RSS was sampled after every decode, prediction, and study
release. The per-study active bound was one. The process peaked at
1,121,271,808 bytes RSS, a 95.8% reduction from the archived v13 peak. Current
RSS samples ranged from 249,503,744 to 896,593,920 bytes after study release;
the first and last release samples were 261,107,712 and 408,936,448 bytes.
This small-work-unit RSS variation is retained in the private profile rather
than described as constant RSS. The 158-sample profile and phase memory values
are recorded in
`artifacts/private/ticket-24-v4-validation/representative-158/kaggle-execution.json`.

The whole local adapter run took 879.552 seconds. Peak RSS was
1,121,271,808 bytes; the same-host archived v3 adapter run peaked at
26,773,291,008 bytes. The v4 output CSV SHA-256 is
`26dffbdb06f8d2d6b7410597067fafff299010d6d554b0535cf8f79e953bee1d`.
It has 158 rows and 13 columns, the expected UID order and target order, finite
scores bounded by `[0, 1]`, and no missing or duplicate rows.

V4 output was byte-identical to the archived v3 adapter's output when both
were run on the same Slurm host against the same 158-study input and candidate.
The v4 run reported two intra-op threads and 64 inter-op threads; the archived
adapter comparison explicitly set two of each. Both CSV checksums are
`26dffbdb06f8d2d6b7410597067fafff299010d6d554b0535cf8f79e953bee1d`.
Compared with the historical Kaggle v13 output, the local CSV differs in 395
of 1,896 score cells, with maximum absolute difference
`1.1920928955078125e-07` (one float32 epsilon), and two pairwise score-order
relations differ. The archived Kaggle run recorded two intra-op and two
inter-op threads; its CPU model/kernel details were not captured. A one-study
local 2/2-thread probe still showed small score differences. This is reported
as cross-runtime float32 variation, not serialization-only variance. The
same-host old-adapter comparison proves the v4 streaming change itself did not
change output bytes on that host, despite the inter-op setting difference.
The cross-runtime differences remain a limitation for future platform
execution.

The previous archived v13 run and the local old-adapter rerun recorded process
peak RSS only; the latter grew to 26.1 GB while studies were being loaded. V4
stage RSS and after-study RSS are sampled at runtime. The local run recorded
host memory values through `/proc`; no v4 Kaggle run was made, so Kaggle's
available RAM or memory limit is not claimed.

Checkpoint save/load was verified during the complete v4 run. The regression
test simulated interruption after the first study, resumed the remaining
studies from the checksum-bound checkpoint, and produced a byte-identical CSV
to the uninterrupted run. The complete repository suite passed (326 tests),
mypy reported no issues in 58 source files, Python compilation passed, all
Slurm scripts passed `bash -n`, final notebook cells parsed, request/artifact
preflight passed, and `git diff --check` passed.

### V4 request and stop point

| Artifact | Identity |
| --- | --- |
| Request ID | `ticket-24-ticket-11-local-candidate-20260924-hidden-v4` |
| Request SHA-256 | `dc96d83447d70a32f9e6165c5777f00ba88028e611d84203f5dfdab726138cb5` |
| Candidate manifest SHA-256 | `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc` |
| Notebook SHA-256 | `df99abaf68de32f50a9d2b1b80a76109c97fd25be484a9d808e11ac33eef44e6` |
| Adapter SHA-256 | `5e452686911f9c66c4aa1872ebd7f06131784cb1f43ab58011a33528771c6ab9` |
| Delivery dataset metadata SHA-256 | `6a4060d303fe18f3c51b6c7210ab8e2793e75ca61a516d0e079bdfbc7ec94804` |
| Kernel metadata SHA-256 | `b77d4f7b797e13d626791fc1e2fbf1a012f57a0f3c81aa06a8091cba310c970d` |

The request is bound to `workload_kind=hidden_test` and only the official
`rsna-knee-abnormality-detection` mount; it does not pin replaceable hidden
CSV or manifest hashes. It remains `owner_approval_required`, with
`upload.allowed=false` and `submission_allowed=false`. Kernel metadata sets
private visibility, internet off, GPU off, and the exact v4 notebook. The
approval dataset is only a planned metadata source; no v4 approval artifact
was created. `load_kaggle_delivery`, the notebook/request-ID audit, candidate
manifest and checkpoint hashes, the delivery code mirror, and all three
offline wheel hashes passed local preflight. No v4 dataset upload, kernel
push/execution, or competition submission occurred. Ticket 24 remains
`ready-for-human`; the next gate is owner approval of the exact v4 checksum.

## V4 approval request validation — 2026-09-24 IST

The owner supplied `/home/ch.jitheswar/tsk2.txt` authorizing the v4 request,
private upload, kernel execution, and one competition submission. The request
ID matches the prepared v4 request and the notebook SHA-256 matches the local
notebook. The supplied request checksum is 65 hexadecimal characters
(`dc96d83447d70a32f9e6165c5777f00ba88028e611d84203f5ddfdab726138cb5`), so it
cannot be a SHA-256 digest. The prepared request checksum is
`dc96d83447d70a32f9e6165c5777f00ba88028e611d84203f5dfdab726138cb5`; the
supplied string has an extra `d` in its final 16-character segment.

The exact-checksum approval gate therefore did not pass. No v4 approval
artifact was created and no Kaggle API/CLI request, private upload, kernel
push or execution, or competition submission was initiated. V4 Kaggle status
is not started; v4 runtime, hidden fallback count, and score are unavailable.
The recorded local 158-study runtime and zero fallbacks are representative
validation only, not Kaggle execution evidence. A repeat of the local
preflight in the current shell also stopped at import because this Python
environment lacks NumPy; the prior Slurm preflight result remains recorded
above. Ticket 24 remains `ready-for-human` with the v4 approval gate pending.

## V4 authorized run stopped at approval validation — 2026-09-24 IST

The owner supplied `/home/ch.jitheswar/tsk3.txt` with the prepared request's
exact 64-character SHA-256
(`dc96d83447d70a32f9e6165c5777f00ba88028e611d84203f5dfdab726138cb5`) and the
matching notebook SHA-256. Slurm artifact preflight job 7134 passed. Read-only
Kaggle checks found only the prior v3 delivery/approval datasets and v3
kernel; the competition submissions list contained the single prior v3
submission.

An exact owner-approval artifact was prepared locally for the v4 request. Its
local `_approval_payload` check returned
`owner approval scope does not match the exact delivery request`. That check
was invoked without an explicit `expected_scope`; following the owner's
instruction to stop on any new validation failure, no retry or correction was
made. The v4 approval dataset was not uploaded. No v4 delivery upload, kernel
push or execution, or competition submission occurred. Kaggle completion
status is not started; v4 runtime, hidden fallback count, peak memory, and
score are unavailable. The local representative workload metrics above are
not Kaggle execution evidence. Ticket 24 remains `ready-for-human`.

## V4 private upload stopped before kernel execution — 2026-09-24 IST

The requested retry of `_approval_payload` passed when called with the exact
`expected_scope` from the v4 request. The private approval dataset upload was
accepted for `jitheswar/rsna-knee-ticket24-approval-v4`. The v4 delivery
bundle upload was also accepted for
`jitheswar/rsna-knee-ticket24-hidden-delivery-v4`, using Kaggle CLI
`--dir-mode zip`. The CLI confirmed the uploaded files include
`candidate.zip`, `code.zip`, and `runtime.zip`; the v4 notebook expects the
`candidate/`, `code/`, and `runtime/` directory trees directly under the
mounted dataset. This package layout does not satisfy the notebook's input
layout, so it is a validation failure. No kernel was pushed or executed and
no competition submission was made. Following the stop condition, no upload
correction or retry was attempted. Kaggle runtime, fallback count, peak
memory, and score remain unavailable. Ticket 24 remains `ready-for-human`.

## V5 archive-layout repair — 2026-09-24 IST

`tsk5.txt` authorized a local fix for the exact Kaggle mount mismatch while
preserving Ticket 24 v4 and the frozen Ticket 11 candidate. V5 keeps the
top-level `candidate.zip`, `code.zip`, and `runtime.zip` artifacts and embeds
a standard-library bootstrap in the request-bound notebook. The bootstrap
extracts into a temporary directory with per-archive compressed-size,
entry-count, expanded-size, path-depth, and compression-ratio bounds. It
rejects traversal paths, duplicate/case-colliding members, symlinks, special
files, encrypted members, and unsupported compression. It verifies every
candidate manifest artifact, shared code file, adapter hash, runtime wheel
hash, mounted notebook hash, and canonical request checksum before those
files are used. The adapter's extracted-root option continues to resolve the
approval and competition inputs from the Kaggle mount.

The flat upload root is
`artifacts/private/ticket-24-kaggle-upload-20260924-v5/`. It contains only
`dataset-metadata.json`, `request.json`, the v5 notebook, and the three ZIP
files; it has no candidate/code/runtime directories for the CLI to repackage.
The matching private kernel metadata is in
`artifacts/private/ticket-24-kaggle-kernel-20260924-v5/` with internet and GPU
disabled. The exact local mount simulation contained `candidate.zip`,
`code.zip`, `runtime.zip`, `request.json`, and
`ticket24_hidden_test_delivery_v5.ipynb` at its top level.

The archive layout preflight passed. It verified 41 candidate files
(953,991 expanded bytes), 31 delivery-code files (748,970 bytes), and all
three runtime wheels (194,737,431 bytes), then compared extracted trees with
the prepared sources. Traversal, duplicate-member, and symbolic-link ZIPs
were rejected and failed extraction left no temporary tree. `load_kaggle_delivery`
accepted the staged request, notebook, adapter, frozen candidate, and runtime
wheels. The extracted-root discovery regression test passed. Notebook code
cells compile; `run_kaggle_baseline` is AST-identical to v4, and the complete
candidate tree is byte-identical to v4. All three checkpoint hashes remain
the frozen values recorded above. This is static and binding evidence; no v5
inference run was started.

V5 request ID is
`ticket-24-ticket-11-local-candidate-20260924-hidden-v5`, canonical request
checksum `d91d5a520e22b0d8f206e4f2929fc4978b81986f038fa0fc54d20f356062808a`,
notebook SHA-256
`3c54a47cdc4b6e6634de577dcadcbbd759e111a1d8ff771a20a06e744682aecb`, and
adapter SHA-256
`8dd46cf3fea1509303376f1e10f8fee8990a932dce7c4c6da2a45c0188a60043`. The
request remains owner-approval-required, with upload and submission disabled.
No v5 approval artifact or Kaggle upload was created; no kernel was pushed or
executed and no competition submission was made. V4 remains preserved with its
previously uploaded private datasets and was not executed. Hidden runtime,
fallback, score, and platform acceptance remain unknown. Machine-readable
preflight evidence is at
`artifacts/private/ticket-24-v5-layout-preflight.json`; the local task report
is `/home/ch.jitheswar/tsk5.md`.

## V5 exact approval and private upload stopped at status validation — 2026-09-24 IST

The owner approved the exact v5 request checksum
`d91d5a520e22b0d8f206e4f2929fc4978b81986f038fa0fc54d20f356062808a` and
notebook SHA-256
`3c54a47cdc4b6e6634de577dcadcbbd759e111a1d8ff771a20a06e744682aecb` in
`/home/ch.jitheswar/tsk6.txt`. The request, notebook, candidate, adapter,
runtime wheels, and approval scope passed local validation. The exact-request
approval artifact SHA-256 is
`654988ce502b09351c514d6817e9a8ef9a62616ce8701c27979026302fe8beee`; its
authorization evidence binds the task-file SHA-256
`4c6cd18cc19efd6615ddf91febcdf909306f797ada637aed6eed8b18d43cc99c`.

The private v5 approval dataset
`jitheswar/rsna-knee-ticket24-approval-v5` reached `ready`; its remote listing
contained only `owner-approval.json`. The private delivery dataset create
request for `jitheswar/rsna-knee-ticket24-hidden-delivery-v5` reported
successful upload of `request.json`,
`ticket24_hidden_test_delivery_v5.ipynb`, `candidate.zip`, `code.zip`, and
`runtime.zip`, then reported that the dataset was being created. The following
read-only status query returned HTTP 403 from
`GetDatasetStatus`; because the delivery dataset's readiness and remote file
listing could not be verified, work stopped at that point. No kernel was
pushed or executed and no competition submission was made. The owner's
stop-on-validation-failure instruction was followed. The v5 request,
notebook, candidate, and v4 artifacts were not changed.

The locally installed Kaggle CLI 1.6.17 did not expose the code-kernel
submission flags, so an isolated Kaggle CLI 2.2.4 environment was prepared and
used for the read-only checks and private dataset creation. This did not alter
the project environment. The approval dataset is confirmed ready. At that
point the delivery dataset was not yet verifiable; the subsequent authenticated
file listing and exact mismatch are recorded below. The v5 task report is
`/home/ch.jitheswar/tsk6.md`.

## V5 remote contents verification — 2026-09-24 IST

The owner instructed read-only verification through Kaggle's authenticated
dataset list and file-list interfaces, treating the earlier
`GetDatasetStatus` HTTP 403 as inconclusive. Kaggle CLI 2.2.4, using the
existing account credentials, returned the private dataset in
`kaggle datasets list -m --search rsna-knee-ticket24-hidden-delivery-v5
--format json`: `jitheswar/rsna-knee-ticket24-hidden-delivery-v5`, total size
191,107,374 bytes. The complete `kaggle datasets files
jitheswar/rsna-knee-ticket24-hidden-delivery-v5 --page-size 200 --format json`
response succeeded with 77 entries and no next-page token.

Only two of the five expected top-level files are present, both at the expected
sizes: `request.json` (7,071 bytes) and
`ticket24_hidden_test_delivery_v5.ipynb` (46,103 bytes). The other expected
files are missing: `candidate.zip` (387,870 bytes), `code.zip` (168,633 bytes),
and `runtime.zip` (190,519,576 bytes). Instead, the file listing exposes 41
files under `candidate/`, 31 under `code/`, and three wheel files under
`runtime/`. Since the remote listing itself fails the exact-name requirement,
no downloads were attempted and the stop condition was applied before kernel
push, execution, or competition submission. No correction or re-upload was
attempted. The earlier HTTP 403 is a non-blocking status-endpoint limitation;
the separate file listing provides conclusive evidence of a delivery-content
mismatch.

Machine-readable evidence, including the full returned file-name/size listing,
local SHA-256 values, expected sizes, and comparison result, is at
`artifacts/private/ticket-24-kaggle-v5-remote-verification.json`. The v5 task
report is `/home/ch.jitheswar/tsk6.md`.

## V6 dual-layout delivery and local inference — 2026-09-24 IST

V5 remains a failed packaging-contract attempt. The authenticated v5 listing
confirmed that Kaggle represented its candidate, code, and runtime archives as
expanded files and directories. V6 changes only the delivery notebook and its
package-consumption layer. The model, checkpoints, preprocessing, inference,
aggregation, fallback behavior, and output predictions are unchanged.

| Artifact | Identity |
| --- | --- |
| Request | `ticket-24-ticket-11-local-candidate-20260924-hidden-v6`; canonical SHA-256 `1a89b6e780e688f5b8d30dd264231ef920b3e07efe2bc99cfa7a77a6efc439f8`; raw `request.json` SHA-256 `19dabfaa543b4d2cc8535e74ae323b4b0550805084186d84f7cb25c2504ae162` |
| Delivery notebook | `ticket24_hidden_test_delivery_v6.ipynb`; SHA-256 `9dd87c8059a4e17909d98aa1bdb65c706c216554de1ec843df2d2afbe0133c13` |
| Delivery adapter | SHA-256 `8dd46cf3fea1509303376f1e10f8fee8990a932dce7c4c6da2a45c0188a60043` |
| Candidate | `ticket-11-local-candidate-20260911-v1`; manifest SHA-256 `cd4e9d45832844211c6ccb77f794b5241e164a4da5860a67193ea8a4deec95bc` |

The checksum-bound layout contract accepts exactly two forms: the three ZIP
archives plus request and notebook, or the candidate/code/runtime trees plus
request and notebook. For ZIPs, v6 safely extracts to temporary storage and
verifies all bound files. For expanded trees, it requires the exact expected
paths and hashes, rejects missing/extra/duplicate/hard-linked/symlink/path-
traversal entries and mixed forms, and uses the verified trees in place.
Regression tests cover these cases and preserve the archive safety checks.

The locally simulated Kaggle-expanded mount has exactly 77 files: 75 payload
files plus `request.json` and the notebook. Its sorted path/size/SHA-256 tree
digest is
`ca5a373063fea93011e6832ec15afecceb6c61e41e445e76bb0ac163bbbcb846`. The
payload-only digest bound in the request is
`b78c5f717832b0f4712a8528bfb6695c74cf2d33403b06d5060277d4b4aef2d5`. A
separate five-entry ZIP mount passed safe extraction and yielded the identical
75-file payload tree. Both request/notebook copies and their checksum bindings
passed preflight. The earlier v5 remote tree digest remains historical evidence
(`e711a989076b39bf2c128a116379269fc4ee5f48ec55b6f7eba540af54812b8d`); v6
has not been uploaded or remotely listed.

Slurm job 7135 ran the 158-study representative workload through the exact
expanded v6 mount. It processed 965 series and all 31,486 slices with zero
decode failures and zero fallbacks. Active-study limit was one. Peak RSS was
1,125,679,104 bytes (1.048 GiB); wall time was 456.794 seconds. The 158 × 12
scores passed the submission contract with exact UID and target order and were
byte-identical to the validated v4 output and same-runtime previous-adapter
output, SHA-256
`26dffbdb06f8d2d6b7410597067fafff299010d6d554b0535cf8f79e953bee1d`.

Focused tests passed 13/13; the full repository suite passed 340/340; mypy
reported no issues in 58 source files. Notebook code cells compiled, the
At the time this local-preparation record was written, owner approval was pending and no v6 Kaggle upload, kernel execution, or competition submission had yet occurred. The later exact approval and private uploads, followed by the remote-download stop, are documented in the section below. Ticket 24 remained open; Ticket 25 remained paused.


Machine-readable preflight and inference evidence are at
`artifacts/private/ticket-24-v6-artifact-preflight.json` and
`artifacts/private/ticket-24-v6-validation/representative-158/validation-summary.json`.

## V6 private upload stopped before kernel execution — 2026-09-24 IST

The owner approved the exact v6 request and notebook hashes in /home/ch.jitheswar/tsk.txt (SHA-256 42120b665fff2cb98b4e092589506499603c33b1ff36a45ac448e60152d24ab9). The request ID is ticket-24-ticket-11-local-candidate-20260924-hidden-v6; its canonical SHA-256 is 1a89b6e780e688f5b8d30dd264231ef920b3e07efe2bc99cfa7a77a6efc439f8, raw request.json SHA-256 is 19dabfaa543b4d2cc8535e74ae323b4b0550805084186d84f7cb25c2504ae162, and notebook SHA-256 is 9dd87c8059a4e17909d98aa1bdb65c706c216554de1ec843df2d2afbe0133c13. The private approval artifact SHA-256 is 96d7c267f2eacd9fa17b02301f28f14cbaf9ebcbf98d597481c540b151d26fed. Kaggle accepted the approval dataset jitheswar/rsna-knee-ticket24-approval-v6, whose remote listing contained only owner-approval.json, and accepted the private delivery upload jitheswar/rsna-knee-ticket24-hidden-delivery-v6.

The authenticated delivery file listing contained 77 paths in the expanded-tree form: 75 payload files, request.json, and the v6 notebook. Every listed path and size matched the prepared expected tree. At that time Kaggle still reported the dataset size as 0 and that it was being created. A read-only dataset download then exited with status 1. The verifier captured stderr in the subprocess exception, but the displayed traceback did not expose or preserve its text. The process stopped before comparing remote file SHA-256 values or staging the downloaded mount; therefore the remote byte contents are unverified even though the listing paths and sizes matched.

Following the task's stop condition, no v6 kernel was pushed or executed and no competition submission was made. Kaggle runtime, peak memory, fallback count, and score are unavailable because hidden inference did not run. The earlier Slurm representative run remains local-only evidence: 456.794 seconds wall time, peak RSS 1,125,679,104 bytes, and zero fallbacks on 158 studies; it is not a Kaggle result. The machine-readable stop status is at artifacts/private/ticket-24-v6-remote-upload-status.json.

## V6 resumed remote verification, kernel execution, and submission response — 2026-09-24 IST

The owner updated /home/ch.jitheswar/tsk.txt (SHA-256 a4054523e8f165f5c92282d0caf90f6964abc89286287f70e93fc776cc7e2cdb) to keep v6 unchanged, retry remote readiness/content verification, and proceed with the already-approved kernel and one competition submission if the existing dataset could be verified. No delivery bundle or notebook was rebuilt, re-uploaded, or changed.

Kaggle now lists jitheswar/rsna-knee-ticket24-hidden-delivery-v6 at 191,111,219 bytes. The authenticated file listing contains exactly 77 expanded-tree paths. The downloaded files match the listing and prepared v6 hashes; the full remote tree digest is ca5a373063fea93011e6832ec15afecceb6c61e41e445e76bb0ac163bbbcb846, and the 75-file payload digest is b78c5f717832b0f4712a8528bfb6695c74cf2d33403b06d5060277d4b4aef2d5. Safe staging verified 41 candidate files, 31 code files, and 3 runtime wheels. The detailed preflight is artifacts/private/ticket-24-v6-remote-mount-preflight.json.

Private kernel jitheswar/ticket-24-hidden-test-delivery-v6 version 1 completed. The hidden input contained 3 studies, 15 series, and 557 slices. Kaggle execution evidence reports 1,247.608946 seconds notebook runtime, 879,849,472 bytes peak RAM, 0/3 fallbacks, and zero decode failures. The kernel used CPU with 4 available logical CPUs, Python 3.12.13, and the approved offline runtime pins. The retrieved submission.csv is 3 rows by 13 columns, has the exact header and UID order, contains finite scores in [0, 1], has no duplicate rows, and matches the execution-record SHA-256 e7a4dac875ce39e1c5945258e4341b64b47dbff62f01ec8fe36fd84dc2c13fb3. Runtime, private case-level execution, and install evidence are under artifacts/private/ticket-24-v6-kernel-evidence/.

Exactly one competition submission request was issued for this CSV. Kaggle's API returned HTTP 400 from CreateSubmission and did not return a submission reference, accepted status, or score. No retry or second submission was made. It remains unverified whether Kaggle recorded the request; the score is unavailable. The earlier v3 submission remains a separate historical attempt. Ticket 24 remains ready-for-human and Ticket 25 remains paused.


## V6 code-competition submission correction — 2026-09-24 IST

The previous v6 submission command was recovered from the local execution log:

~~~bash
/home/ch.jitheswar/.venvs/kaggle-cli-2.2.4/bin/kaggle competitions submit -c rsna-knee-abnormality-detection -f artifacts/private/ticket-24-v6-kernel-evidence/submission.csv -m "Ticket 24 v6 hidden test delivery"
~~~

It uploaded the local 1.01 kB file and omitted both code-kernel arguments. Its HTTP 400 came from competitions.CompetitionApiService/CreateSubmission, the file-only endpoint. The captured CLI output was exactly 400 Client Error: Bad Request for url: https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateSubmission. The CLI did not expose or preserve the HTTP response body, and the saved run artifacts contain no body to recover. Kaggle’s current [code-competition submission contract](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) uses the kernel owner/slug, exact version, and the output filename.

Read-only Kaggle checks identified the kernel as jitheswar/ticket-24-hidden-test-delivery-v6, current saved version 1. The remote kernel metadata reports version 1, and its five remote cell source strings match the approved local v6 notebook (SHA-256 9dd87c8059a4e17909d98aa1bdb65c706c216554de1ec843df2d2afbe0133c13). Kaggle reports the session status KernelWorkerStatus.COMPLETE. The kernels list endpoint reported lastRunTime 2026-09-24T06:19:29.470000, while GetKernel metadata returned earlier 2026-09-21 timestamps; these endpoint timestamps are recorded separately. A paginated listing of 14,863 current-session output files across 75 pages includes submission.csv. The remote serialized notebook hash was 2ad6bfd116ad3f21a2cec94d89c28e0710df088add3d7dbacf8e1d23ec439daf; while that full-document hash differs from the approved local notebook hash, all five cell source strings match. The current session output listing API did not accept an explicit numeric version label, so version association is recorded through current-version metadata, matching cell source, and the completed current session.

Before correction, Kaggle’s submissions listing contained only the earlier v3 ref 56500900, so the old 400 had not appeared as a v6 submission. One corrected code-submission request was then made with the same message and these arguments: competition rsna-knee-abnormality-detection, kernel jitheswar/ticket-24-hidden-test-delivery-v6, kernel version 1, output filename submission.csv, message Ticket 24 v6 hidden test delivery. This used CreateCodeSubmission; submission.csv was passed as the kernel output name, and no local CSV was uploaded. Kaggle returned submission ref 56515942. The subsequent submissions listing reports status SubmissionStatus.PENDING; public and private score fields are blank. No second corrected request was made.

## V6 submission scoring complete — 2026-09-24 IST

A read-only refresh of Kaggle’s submissions listing reports ref 56515942 as SubmissionStatus.COMPLETE with public score 0.498 and no private score. The corrected code-competition submission has finished scoring; no additional submission was made.

Ticket 24 is resolved for this owner-approved delivery. The v6 kernel ran the exact frozen candidate against Kaggle's `competition_hidden_test` input (3 studies, 15 series, 557 slices), produced all three expected rows with the exact 13-column contract, and recorded zero fallback and decode failures. Kaggle completed scoring within 1,247.609 seconds of notebook runtime. The local 158-study representative run and prior adapter comparison remain separate evidence; the hidden run's interruption recovery was not tested, and no 1,300-study Kaggle runtime or private score is claimed. The v3 OOM and v4/v5 packaging failures are preserved as historical attempts and do not describe the final v6 delivery.
