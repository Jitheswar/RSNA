# 27: Produce the research and reproduction report

**What to build:** Produce a local research report and reproduction guide grounded in preserved experiment and candidate evidence.

**Blocked by:** 24: Select and deliver the final candidate

**Status:** resolved

**Authorization:** Originally planning-only. On 2026-09-25 the owner directly instructed: “implement ticket 27 and generate report of it outside RSNA folder as .md file.” This authorizes the local report and tracker updates; it does not authorize further experiments, external publication, artifact transfer, uploads, or submissions.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Describe domain/task, image-only baseline, actual controlled experiments, selection strategy and separately reproducible cluster/inference environments.
- [x] Report pooled/per-target OOF, class counts, variation and uncertainty; disclose development-selection bias, lack of untouched internal test and any unverified patient separation.
- [x] Tie reported results to source, configuration, fold/data/supervision identities and checkpoints while keeping identifiers, reports and restricted artifacts private.
- [x] Explain resource costs, failure/recovery policies and evidence scope; do not claim clinical superiority from small CV or leaderboard gains.
- [x] Deliver local documentation only; external publication and artifact transfer remain subject to rights and approval, and omitted optional experiments are not reported as executed.

## Completion evidence

- [x] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Answer

Ticket 27 is resolved with the updated local research and reproduction report
at [`/home/ch.jitheswar/RSNA-ticket-27-research-and-reproduction-report.md`](../../../../RSNA-ticket-27-research-and-reproduction-report.md),
outside the RSNA folder. Its SHA-256 is
`d4777ef147ee2c9a4d314a98f43d64c2ec4e56d64ae3cc8de2de13da910d2cec`.

The report summarizes the corrected three-fold gold-only
OOF results, per-target support and uncertainty, controlled report/image-model/student
experiments, candidate selection and the small Kaggle execution, CPU resource
and recovery evidence, and reproducible cluster/offline setup. It distinguishes
completed, inconclusive, unrun, and unverified work. No study identifiers,
source rows, medical report text, per-study predictions, or restricted
artifacts were included. The preserved Ticket 26 Slurm completion summary
records the generated report path `/home/ch.jitheswar/ticket-26-report.md` and
its SHA-256 `5a2bf90e079d71c127ce48852c3f281bde04b03421f10a31be4b4639b23891e5`.
The hash matches the Ticket 26 issue. The summary records 1 study, 3 series,
36 decoded slices, 2 usable series, 16 selected model slices, zero fallback,
and finite bounded scores. The Ticket 26 report body was not available in this
workspace, so the hash and summary counts were checked against the issue and
preserved Slurm output; per-target scores and report-only provenance rows were
not independently compared. The report does not assert clinical performance,
full-scale runtime, or patient separation.

## Comments

- 2026-09-25 — The owner explicitly authorized Ticket 27 implementation and
  required the Markdown report to be outside the RSNA folder. The report was
  assembled from preserved evidence; no experiment, upload, submission, or
  external publication was performed.
- 2026-09-25 — Follow-up from `/home/ch.jitheswar/tsk.txt`: restored the exact
  previous Ticket 27 report from local session output and verified its original
  recorded SHA before updating its Ticket 26 section. The updated Ticket 27
  report SHA-256 is `d4777ef147ee2c9a4d314a98f43d64c2ec4e56d64ae3cc8de2de13da910d2cec`.
  Ticket 26's recorded hash and headline counts match the issue and Slurm
  completion output. Its report body remains absent from the workspace, so
  direct comparison of individual scores and report-only provenance remains
  unverified. No experiment or model artifact was changed; Ticket 27 remains
  resolved.
