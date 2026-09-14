# Ticket 16 — Approved local language-model comparison

Status: implementation prepared; model approval and private execution remain pending.

Date: 2026-09-14

## Scope

Ticket 16 adds the second report-extraction boundary. It does not download a
checkpoint, inspect a competition report, submit a Slurm job, or claim a
learned extractor result. The implementation is deliberately useful before
approval: it records a concrete proposal, validates the exact local artifact
when approval exists, freezes the prompt, and makes a controlled comparison
against ticket 15 possible.

The implementation is in
[`src/rsna_knee/local_report_extraction.py`](../../src/rsna_knee/local_report_extraction.py),
with its command adapter in
[`src/rsna_knee/local_report_extraction_cli.py`](../../src/rsna_knee/local_report_extraction_cli.py)
and bounded Slurm wrapper in
[`slurm/run_ticket16_local_report_extraction.sbatch`](../../slurm/run_ticket16_local_report_extraction.sbatch).

## Exact proposal and approval boundary

The committed factory `proposed_flan_t5_small_approval()` names one resource:

| Field | Proposed value |
| --- | --- |
| Model | `google/flan-t5-small` / `FLAN-T5-small` |
| Source | [Hugging Face model card](https://huggingface.co/google/flan-t5-small) |
| Immutable revision | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| Weight file | `model.safetensors` |
| Weight SHA-256 | `495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749` |
| Supporting loader files | `config.json`, `generation_config.json`, `special_tokens_map.json`, `spiece.model`, `tokenizer.json`, `tokenizer_config.json`; each exact SHA-256 is carried in `supporting_artifact_hashes` |
| License | Apache-2.0 as shown on the model card |
| Availability record | Public revision assessed against the `2026-09-14` cutoff; this remains an owner eligibility review item |
| Intended purpose | Local research extraction of conservative report states; not clinical use or competition inference |
| Expected benefit | Quantify any change in coverage, accuracy, auditability or latency versus ticket 15; no benefit is assumed |
| Leakage assessment | No competition-report fine-tuning is claimed; training-data overlap is not independently established |
| Reproducibility method | Repeat the exact local artifact, frozen prompt, fold manifest and private report-record digest; compare text-free semantic digests |
| Deterministic comparator | Ticket-15 frozen deterministic extractor on identical fold and report cohorts |
| Execution | `cluster1-cpu`, CPU only, at most 16 CPUs / 64 GB / six hours |
| Transfer | `local-only-no-report-transfer` |

Supporting loader-file digests pinned by the proposal:

```text
config.json             439aa0fecf5a5546a1def68b1fc45e538e2c94528ce805378daf091e2bf6e4de
generation_config.json  f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a
special_tokens_map.json 5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f
spiece.model            d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86
tokenizer.json          fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a
tokenizer_config.json   fcde0f79bffda3688119c94330866a8fbf8de20ae65a8c492c9bd47c704655a0
```

The proposal is not an approval. `LocalModelApproval.require_approved()`
rejects it, and `verify_local_model_artifact()` checks approval before it
reads the model directory. An approved manifest must retain the exact model
identity, primary weight hash, all supporting loader-file hashes, an owner
approval reference, approver and date. The loader verifies every declared file
before importing the optional runtime.
Changing the revision or weights changes the approval fingerprint and therefore
requires a new proposal. GPU devices are rejected by the ticket-16 loader and
retain their separate approval/preflight gate.

The optional `transformers` adapter receives a local directory only and passes
`local_files_only=True` and `trust_remote_code=False` to both tokenizer and
model loading. The Slurm wrapper additionally sets hub/transformer offline
flags. No report text is sent to a model hub, API or inference provider.

## Frozen prompt and output contract

`freeze_local_prompt()` returns an immutable prompt configuration. Its
fingerprint covers the prompt text, the official target criteria, target order,
greedy generation settings, input/output limits and JSON schema. The prompt
requires all twelve targets and one of the preserved states:

- `positive`
- `negative`
- `not-mentioned`
- `uncertain`
- `unknown`
- `extraction-failed`

The six-state representation retains the ticket's five requested outcomes and
also keeps unsupported-but-readable `unknown` distinct from software/model
failure. Positive and negative outputs require a finite confidence in `[0, 1]`
and receive supervision mask one. All abstained states receive mask zero.
Evidence is an optional single-line model snippet capped at 512 characters;
raw model output is represented only by a digest and size in private audit
records. Invalid JSON, missing/duplicate targets, invalid confidence, and
invalid evidence produce explicit extraction-failed records instead of
silently becoming negatives.

## Controlled comparison protocol

`run_local_report_extraction()` reproduces ticket 15's partition policy:

1. report-only studies from the fold's training side become training-label
   records;
2. gold reports from that fold's held-out side become evaluation-only records;
3. held-out reports are excluded from local model adaptation, prompt
   development and label training; and
4. report identifiers, source references and raw report text remain private.

`compare_report_extractors()` refuses a comparison unless the deterministic and
local runs have the same fold-manifest checksum, repeat, report-source version,
report record-set digest, held-out evaluation UIDs and training eligibility.
It reports per-target accuracy where confident predictions exist, coverage,
not-mentioned/uncertain/unknown/failure counts, confidence, evidence-based
auditability, latency, token/artifact cost, repeatability and deltas from the
deterministic baseline. The public artifact is aggregate-only; the private
artifact carries fold identifiers, report digests, response digests and model
evidence without copying source reports.

## Verification performed

The synthetic suite in `tests/test_local_report_extraction.py` covers:

- exact proposal metadata and fail-closed unapproved use;
- local weight hash verification and changed-weight rejection;
- stable prompt identity and report-text privacy;
- positive, negative, not-mentioned, uncertain, unknown and
  extraction-failed parsing with masks;
- malformed model output as explicit extraction failure;
- identical fold/cohort comparison and measured token/character cost;
- source-version mismatch rejection; and
- public/private comparison artifact separation.

Observed focused result: `9 passed`. Mypy passes for the new module, CLI,
package exports and the updated report-label provenance enum.

No owner-approved manifest, local checkpoint, optional model runtime, private
report corpus, Slurm job, real-model metric, or learned-supervision benefit
claim exists from this change. V13 is pending approval/eligibility review;
V15 is implemented as a fail-closed lineage and fold-isolation contract but
has not been demonstrated on a learned artifact. The detailed implementation
handoff is outside the repository at
`/home/ch.jitheswar/ticket-16-report.md`.
