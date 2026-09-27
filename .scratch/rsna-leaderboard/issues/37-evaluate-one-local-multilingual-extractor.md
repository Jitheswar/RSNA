# 37: Evaluate one local multilingual extractor

**What to build:** If existing outputs are inadequate, one reviewed local multilingual extractor produces reproducible, fold-isolated report-label evidence and an admission or rejection outcome.

**Blocked by:** 29: [Diagnose real-image learning within campaign limits](29-diagnose-real-image-learning-within-campaign-limits.md); 30: [Audit existing report-derived labels](30-audit-existing-report-derived-labels.md)

**Parent:** [RSNA leaderboard campaign with zero external spend](../spec.md)

**Status:** resolved

**Execution constraints:** Follow the approved spec: zero external spend; significant work through Slurm; existing CPU ceilings; one active or pending GPU request and 20 allocated GPU-hours across the first campaign, including failures and retries; 2 TB soft storage budget; exact resource and delivery approvals where required. Questions go in visible chat. Synthetic checks establish software behavior only. No project-imposed candidate or submission-count cap.

**Execution trigger:** Only select this branch when the existing-label audit shows inadequate eligible coverage or reliability and a bounded local extraction proposal fits the remaining resources.

## Acceptance criteria

- [x] Review the exact candidate model/revision, rights, competition eligibility, local runtime and source identity before use; the researched Qwen3-8B option is not automatically admitted.
- [x] Freeze prompt, output schema, language-handling policy, decoding and evidence-span requirements with training-side development only; keep original report text local.
- [x] Measure a bounded eligible sample for real memory, throughput, malformed outputs and abstention before any broader extraction request.
- [x] Use the existing exact GPU approval boundary if GPU execution is chosen, count all attempts against the campaign cap and retain a measured CPU option where feasible. No new remote report transfer is included.
- [x] Compare target-level reliability and coverage with existing deterministic and learned outputs, including new calls, language support and unidentifiable rates.
- [x] Apply the same six-state and reference-precedence admission contract as the existing audit; uncertain or unsupported targets remain masked.
- [x] Exercise schema failure, fold contamination and immutable output behavior through the public extraction-to-admission boundary.
- [x] Produce admitted artifacts or a documented rejection; downstream training that selects these artifacts adds this ticket as a real blocker and creates a new run identity.
- [x] Register selection before extraction and record a terminal promotion/rejection disposition for final-freeze readiness, distinguishing not evaluated from measured failure.

## Completion evidence

The implementation is in `src/rsna_knee/ticket37.py` and `src/rsna_knee/ticket37_cli.py`, with a CPU Slurm entry point at `slurm/run_ticket37_local_qwen_pilot.sbatch`. Focused boundary tests cover malformed schema, fold isolation, six-state preservation, and immutable private outputs. The selected Qwen3-8B snapshot was pinned by immutable revision and manifest; selection, prompt, and sample plan were frozen before inference.

The complete bounded pilot processed 58 held-out gold reports and 6 report-only sample reports with 65 generation calls. Sixty-one unique reports failed output validation (46 invalid JSON and 15 invalid target sets); every invalid-JSON output reached the frozen 384-token maximum, consistent with truncation. Three valid gold outputs supplied one binary call each for Medial Meniscus, Effusion, and Baker's. All three matched the available reference, but per-target support was one, no reliability claim is supported, and no training-side gate admitted any call. All report-only sample outputs failed parsing, so the branch is rejected under the frozen policy. No downstream training or broader extraction was run. Any future attempt should use a separately registered prompt/schema/output budget and run identity.

Aggregate and private artifact identities, runtime, target-level coverage, language counts, limitations, and the terminal rejection are in `/home/ch.jitheswar/ticket-37-implementation-report.md` and the Ticket 37 section of `.scratch/rsna-leaderboard/final-freeze-register.md`. Public-safe summary: `/home/ch.jitheswar/ticket-37-qwen3-8b/public/pilot-summary.json`; restricted results remain under `/home/ch.jitheswar/ticket-37-qwen3-8b/private/`. The public summary persists neither report text nor evidence spans. Test and typecheck evidence: 11 focused tests passed; mypy passed. Two-axis review found no remaining actionable issues after permission and manifest fixes.

## Comments

### 2026-09-27 — Bounded CPU pilot complete; candidate rejected

See the completion evidence above and standalone report. The frozen policy admitted zero calls, and candidate promotion remains false. Release gates are unchanged.
