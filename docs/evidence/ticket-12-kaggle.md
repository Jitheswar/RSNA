# Ticket 12 Kaggle baseline delivery evidence

Recorded 2026-09-11 (Asia/Kolkata). This public-safe record contains no
competition study identifiers, reports, raw paths, or prediction rows.

## Implemented delivery seam

`rsna_knee.kaggle` now provides three immutable stages:

1. `prepare_kaggle_delivery` reloads the ticket-11 candidate and validates its
   manifest, selected checkpoint hashes, inference configuration, local exact
   submission evidence, runtime projection, zero-fallback evidence, delivery
   notebook, network-free `kaggle.py` adapter, and pinned offline environment.
   It writes an `owner_approval_required` request containing the exact hashes
   and an explicit delivery layout. It does not contact Kaggle or enable
   upload/submission.
2. `run_kaggle_baseline` is the notebook adapter for Kaggle's
   `test.csv`/`test_series.csv`/`test_series/` layout. It reuses the candidate
   decoder, preprocessing, hierarchical model, probability ensemble and
   strict submission writer. Non-smoke execution verifies the mounted request,
   owner approval, copied notebook/adapter hashes, and installed pins before
   inference. It records resource sizes, peak VRAM, notebook-stage timings,
   projection, actual workload, output checksum, and measured fallback
   diagnostics; detailed per-study failures are emitted only to the private
   evidence file.
3. `record_kaggle_verification` accepts only an owner approval for the exact
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

## Verification state

| Gate | State | Reason |
| --- | --- | --- |
| V08 exact Kaggle bundle/output | locally prepared | Request, notebook and adapter hashes are checked; actual Kaggle execution is pending. |
| V12 rules/account/eligibility | blocked pending owner | The owner must personally recheck rules, eligibility and exact upload/submission authority. |
| V16 whole-notebook runtime | locally projected only | Ticket 11 measured a representative CPU workload; hidden Kaggle runtime has not run. |
| V17 rows/fallback/recovery | locally evidenced only | Local fallback is zero; hidden failure details must be measured or explicitly retained as unknown. |

The targeted ticket-12 test suite covers request immutability, exact candidate
matching, owner-approval scope, accepted-output verification, unknown-fallback
preservation, Kaggle-layout smoke inference, pre-run approval/hash gates,
private-report validation, and promotion immutability. The smoke run is
synthetic and is not Kaggle verification.

## Pending owner action

Before any upload or submission, the owner must approve the exact prepared
request. After the approved notebook runs on the actual Kaggle workload, keep
the accepted `submission.csv`, private diagnostics, and recovery evidence,
annotate the execution evidence with the platform-accepted state, and run the
verifier with both preserved artifacts. No Kaggle upload, submission, account
action, or final selection was performed for this ticket.
