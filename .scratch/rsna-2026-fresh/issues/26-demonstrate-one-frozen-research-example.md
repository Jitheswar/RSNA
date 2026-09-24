# 26: Demonstrate one frozen research example

**What to build:** Create a small later academic demonstration of one frozen example and model with visible processing and provenance.

**Blocked by:** 24: Select and deliver the final candidate

**Status:** resolved

**Resolution:** Implemented and ran one public-safe synthetic example through the selected frozen Ticket 11 candidate. The report records its processing decisions, scores, provenance, and limits.

**Authorization:** Initially a planning artifact only. The owner separately authorized implementation with “implement ticket 26” on 2026-09-25. Work remained within the synthetic, local research-demo scope; no patient data, external deployment, or model/data upload was used.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [x] Use a generated synthetic example; no patient MRI or reports were uploaded or published.
- [x] Show selected series/slices, preprocessing/rejections, twelve model scores, model/configuration/candidate provenance and known limitations.
- [x] Reuse the checksum-verified Python candidate inference interface through a small CLI and Slurm adapter.
- [x] Describe scores as uncalibrated model confidence and the demonstration as research/education; make no clinical diagnostic or validated-risk claims.
- [x] Keep external deployment and restricted-data publication behind applicable approval and eligibility gates; neither was performed.

## Completion evidence

- [x] Record the executed synthetic demonstration, verified scope, unresolved gates and limitations without exposing restricted data. No future execution or approval-dependent check is marked passed based solely on prepared code or documentation.

## Answer

The selected Ticket 11 candidate processed one generated study containing three series and 36 slices. All slices decoded; two diagnostic series contributed 16 geometry-sampled slices, and one synthetic localizer was excluded. Inference used all three frozen fold checkpoints, produced twelve finite bounded scores, and used no fallback. The complete report is [`/home/ch.jitheswar/ticket-26-report.md`](../../../../ticket-26-report.md).

This verifies only the synthetic example and its output contract. The report labels scores as uncalibrated confidence, records development-selection and patient-grouping limitations, and does not claim clinical performance or full-scale runtime compliance.

## Comments

- 2026-09-25 IST — The first bounded Slurm attempt stopped before inference because the synthetic archive fingerprint included a ZIP checksum not used by the shared identity validator. The fingerprint construction was aligned with that interface; the failed attempt produced no report or model output.
- 2026-09-25 IST — Final Slurm job 7154 completed on `cluster1-cpu` with 4 CPUs and 8 GB requested memory. It decoded 36/36 synthetic slices, retained 2 series, excluded 1 marked localizer, selected 16 slices, used zero fallback, and returned finite bounded scores. No real data, external upload, or deployment was used. The report SHA-256 is `5a2bf90e079d71c127ce48852c3f281bde04b03421f10a31be4b4639b23891e5`.
