# RSNA campaign handoff — October 2, 2026

Objective remains first place on **both public and final private leaderboards**.
Neither is achieved. Latest authenticated official snapshot **01:08:08 UTC**:
**Elyra rank 2,005 / 4,808, 0.936**; public leader **rhoskeri / 0.961**.
Best exact submission **56705536 / script 354125287 / v2**. Earlier **0.498**
fallback preserved. Private score/rank and final selections remain unavailable.

## Resume order

1. Read root instructions and [previous handoff](session-handoff-20261001.md)
   for earlier training/comparison dispositions and owner authorization. Inspect
   current Git, Slurm and service state; keep one campaign controller.
2. Preserve completed exact integrity job **8012**, private root
   `artifacts/private/public-frontier-adoption/full-input-independent-verification-v1/`.
   It hashes all **819,078 / 569,154,778,424-byte** original retained DICOM payloads
   and independently computes archive CRC. Output belongs at
   `full-cpu-input-probe-v2/independent-payload-validation.json`.
   Actual terminal `0:0`, **779 allocated CPU-node seconds**, exact accepted
   job/source/plan and complete independent byte totals pass terminal
   reconciliation. Validation SHA-256
   `bcc777b9193de735cd7ac7e76d76b8ce2558468a222052e53631e42ceba50844`.
   One consumed attempt; do not rerun it.
3. Continue the all-source residual CoAt preparation probe **8014**, private root
   `artifacts/private/public-frontier-adoption/full-coat-cpu-input-probe-v1/`.
   Inspect actual job handle, journal, exception/failed rows and exact terminal
   observer. It processes **all 4,407 studies**, using the exact patched notebook
   input function with original MRI reused read-only. One nonexclusive CPU job,
   16 CPUs / 64 GiB / six hours, two preparation workers, 21,540-second watchdog,
   one attempt, no requeue/GPU/spend. Forecast **21,282.375 s** includes 2x pilot
   cost +900 s. No model forward pass, labels/reports or hidden input.
4. On terminal success, run the lightweight verifier:
   `.venv/bin/python scripts/validate_ticket59_coat_input_probe.py ROOT`.
   It verifies actual execution records and producer image-hash records;
   it does not independently replay pixels. If accessible input fails, preserve
   exact source/trace, minimize that input and test the actual call-site guard.
   Do not call a nearby failure the hidden cause without matching evidence.
5. [Ticket 59](issues/59-diagnose-hidden-frontier-execution-failure.md) still lacks
   a matching red-capable hidden-error loop. No ranked cause, recipe repair,
   rescoring, candidate promotion or final selection has occurred. Preserve the
   exact failed **56739912 / 354194017 / v3** request and full unknown charge.
   Another scoring request requires a distinct justified plan.
6. Observe the separate bounded TPU environment request, kernel **136722939 /v1**,
   `jitheswar/rsna-tpu-environment-admission-v1`, root
   `artifacts/private/ticket60-tpu-environment-v1/`, latest **QUEUED** at
   **03:11:29 UTC**. Exact requested source/private/offline settings pass;
   actual hardware remains unverified. One 600-second server bound, 540-second
   code watchdog, GPU off, no attachments/models/MRI/training, zero spend.
   Existing launch intent is consumed: never push it again. Observer
   `rsna-ticket60-tpu-environment-v1-observer.service` preserves exact terminal.
   On terminal, run Kaggle SDK Python with
   `scripts/verify_ticket60_tpu_environment.py`, reconcile outputs/quota and
   conservatively charge 600 seconds in the new TPU ledger if unresolved.
   [Ticket 60](issues/60-establish-actual-tpu-training-capacity.md) and
   [evidence](../../docs/evidence/ticket-60-tpu-environment.md). A passing
   allocation only enables a separately frozen real-MRI full-backbone pilot.
7. Full-backbone code and CPU cost pilot are verified independently of TPU:
   **8016 /COMPLETED /0:0**, **127 CPU-node seconds**, root
   `artifacts/private/ticket61-cpu-full-backbone-pilot-v1/`. Plan
   `43adab68e2113441d5bccb04273a935bd43ab8039cbbd0a841f4758b4364bfaa`.
   Four admitted training reports, 232 valid/512 static padded slices; native
   initial logits and fresh-process diagnostic checkpoint replay match exactly.
   Every encoder stage updates, fixed normalization buffers; median **5.955313
   s/update**, peak producer RSS **17.88 GB**. Actual terminal/accepted plan and
   independent validation pass. One consumed attempt, do not relaunch.
   [Ticket 61](issues/61-prepare-fixed-shape-full-backbone-report-training.md),
   [evidence](../../docs/evidence/ticket-61-full-backbone-cpu-cost.md). The new
   versioned model/loss preserves prior variants. Shared CPU optimizer/resume,
   file-backed pixels and native/evaluation/candidate prediction now pass
   **57 focused tests**, typing **84 source files**.
   A 2× training-only projection gives 1,500 steps/fold 4.963 h, 7,500 steps
   24.814 h. Neither is a complete admitted route. CPU work can advance while
   TPU queues; no dependency on TPU for this alternative.
8. Preserve completed exact pixel cohort **8017 /COMPLETED /0:0**, **2,080
   CPU-node seconds**, root `artifacts/private/ticket62-mri-pixel-cohort-v1/`.
   Plan `718e2c127bef8ea1135c7e76b653b2fee6313a6d69dc89219e8f78e5b645b581`.
   Every pixel/mask/descriptor/source mapping from 4,228 studies /187,272 valid
   slices matches fresh native source packing; the prior four-report full pixel
   hash also replays exactly. Accepted plan/job/resources/frozen source/terminal
   reconciliation passes. Read/reconstruction 189.317 s, native replay 704.605 s.
   One consumed attempt; do not rerun. [Evidence](../../docs/evidence/ticket-62-exact-mri-pixel-cohort.md).
9. Observe new bounded mixed-source shared-trainer cost job **8018**, root
   `artifacts/private/ticket63-full-backbone-training-cost-v1/`, plan
   `40f5268a019a9d247cb3ced8d0d7a62c419bd7b1e544a1370af7f884d65f7e39`.
   One nonexclusive 16-CPU/64-GiB/90-minute attempt, GPU/spend zero, forecast
   3,629.069 seconds. Five fresh processes admit the complete pixel cohort and
   original three-fold supervision, interrupt at one mixed-source update,
   resume to three, run three continuously and compare complete states. No
   held-out AUC, full fit or candidate promotion. Persistent observer
   `rsna-ticket63-cpu8018-observer.service`; do not relaunch. On success,
   independently bind `cost-result.json`, stage metrics, exact source/accepted
   job/resources and retained terminal. Then separately freeze complete
   7,500-update three-fold matched-control regime, measured staged startup,
   checkpoint/storage/evaluation/conditional final-fit/offline delivery cost.
   Choose stage size from actual complete costs, each job <=six hours. No
   intermediate AUC selection. [Ticket 63](issues/63-measure-shared-full-backbone-training-delivery-cost.md).
10. Independently verify results, update evidence/tracker/freeze register, run
   relevant checks, commit/push private `origin/master` and verify equal heads.
   Mark the goal achieved only after both actual first-place ranks are proven.

## Results verified this continuation

- Full original DINO input scan **7995 / 7996**, both `COMPLETED 0:0`.
  **4,407 studies / 819,078 DICOMs / 24,371 header series / 20,204 selected
  slot-series**. Disjoint original-source coverage, actual fragment/source
  hashes, events, decisions and terminal identities pass independent summary
  checks. **Zero decode failures/fallback events**, **4,974 CPU-node seconds**.
  Peak producer RSS ~21.17 GB. Images were released; originals remain private.
  This tests one DINO pixel group, excluding other layouts/GPU/model/CoAt paths.
- Exact residual CoAt preparation pilot **8013**, `COMPLETED 0:0`, **148
  CPU-node seconds**. Fixed label-blind **64** original-slice-count order
  statistics over the complete corpus, includes **67–632** slice extremes.
  Bags **61–94 windows**; **zero failures/warnings/sampler/sparse fallbacks**.
  Peak worker RSS **1,702,268,928 bytes**. Exact mounted source/OpenCV hashes
  match prior admission. Independent summary and actual generated runtime pass.
  This tests CPU native bags; no model/GPU/other-reader/hidden path.

Full CoAt plan SHA-256:
`abea8f3b42312c3fba1a43a15ea732af256a19fdaa22c59e5998538da83dc5fa`.
CoAt pilot plan:
`9ea8ee92ccd79764220f7daab596a8b5c32233e8cdad80c7b17db01f1fec725f`.
Integrity plan:
`8838254220079e484e16f90aa87ef066afc412ef467e45bad3c6606cd5c51237`.
Observers: `rsna-ticket59-independent8012-observer.service` and
`rsna-ticket59-fullcoat8014-observer.service`. Actual terminal records remain
under each root's `controller-JOB/` even if transient units disappear.

## Resources and delivery

Standing owner authorization remains **“approve anything as long as we win
1st place and fits kaggle comp dont ask me”**, zero external spend. Agent
decisions are recorded as such, not claimed as new numeric owner approvals.
GPU ledger unchanged: **129,637 charged / 136,800 cap / zero reserved / 7,163
remaining seconds**. Full CoAt **8014** remains active, CPU-only, and one TPU
environment request is queued with **600 TPU seconds reserved**. Shared-trainer
cost job **8018** runs in the second CPU slot. At most two concurrent
nonexclusive CPU jobs, each <=16 CPUs /64 GiB /six hours. No additional MRI
extraction or large retained pixel banks in the CoAt probe. Preserve all prior
unknown GPU charges. Old continuation coordinator remains stopped.

Science freeze October 11, internal delivery before October 15; recorded official
final deadline October 22 at 23:59 UTC. Existing five-submission/day, two-final
selection, nine-hour offline notebook constraints remain. Refresh live official
state before a delivery decision. GitHub repository must remain private:
**https://github.com/Jitheswar/RSNA**.

Current focused tests: **57 passed**; typing **84 source files**; syntax checks
and generated launcher checks pass. Main evidence:
[Ticket 59](../../docs/evidence/ticket-59-hidden-execution-diagnosis.md).

Medical encoder [primary-source research](../../docs/research/medical-image-encoder-options-2026-10-02.md)
found one conditional BiomedCLIP admission route: ungated MIT, 224-pixel,
vision-only projected 512-D output. Its MRI evidence concerns scientific-article
figures, not proven knee transfer. MedSigLIP is gated; MRI-CORE release rights,
digest and strict loading remain unresolved. Metadata/source only, no weights
downloaded. Keep current whole-backbone experiment; no new encoder launch from
the note alone.
