# Ticket 16 — Astra Medium Codex handoff and closure evidence

Status: resolved/closed — TICKET 16 SCIENTIFIC EXPERIMENT COMPLETE — MIXED / REVIEW REQUIRED.

Date: 2026-09-21 closure; handoff preparation began 2026-09-14.

## Final closure evidence

The authoritative corrected scientific report is outside the repository at
[`tsk2-report.md`](../../../tsk2-report.md). The corrected metric definitions
and regression tests are in
[`astra_aggregation.py`](../../src/rsna_knee/astra_aggregation.py) and
[`test_astra_aggregation.py`](../../tests/test_astra_aggregation.py).

The completed Astra Medium run validated 276/276 batches covering 4,407
studies, with zero validation failures and zero retries. Astra materially
increased weak-label coverage for all 12 targets, while held-out behavior was
target-dependent. The common-target macro selective accuracy was Astra 0.8614
versus Ticket 15 0.8961; micro selective accuracy was Astra 325/370 (87.8%)
versus Ticket 15 68/78 (87.2%). Where both extractors were supervised, they
agreed on all 75 held-out target calls and disagreed on none.

The corrected aggregation distinguishes selective TP/TN/FP/FN metrics from
full-reference positive/negative capture, excludes abstentions from selective
rates, reports exact per-extractor and common macro target sets, and records
micro numerators and denominators. Report-only accuracy, precision, recall,
specificity and AUC remain undefined.

Astra labels are not generally approved for training by Ticket 16 because no
predeclared admission rule exists. The frozen Astra outputs, prompt,
fingerprints, Ticket 15 artifacts and gold labels were preserved unchanged.
No new scientific decision, admission policy, Astra rerun, report
reinterpretation, training run or Ticket 17 work was performed.

## Historical scope and seam

The intended Ticket 16 learned extractor is Astra Medium used by manually
switching the model in a Codex subscription. The repository stops at a
controlled handoff seam:

1. an owner supplies and approves the final medical extraction prompt;
2. the local code binds that prompt to the fixed experiment, canonical folds,
   report digests and immutable private batches;
3. the operator switches Codex to Astra Medium with reasoning and effort both
   fixed to Medium, attaches one batch, and saves Astra's response verbatim;
4. local validation accepts an exact response or records all-target
   `extraction-failed` output; and
5. Luna later performs aggregation, comparison, metrics and repository
   bookkeeping.

There is no model API, API-key handling, provider SDK, billing/token API,
network model loader, or external model wrapper. The implementation is
`src/rsna_knee/astra_handoff.py`; its CLI is
`src/rsna_knee/astra_handoff_cli.py`.

## Experiment identity and approval record

The prepared identity record is produced by
`astra_medium_experiment_approval()` and can be written without private data:
the committed public copy is
[`ticket-16-astra-medium-approval.json`](ticket-16-astra-medium-approval.json).

| Field | Fixed value |
| --- | --- |
| Experiment ID | `ticket16-astra-medium-codex-subscription-v1` |
| Model name | `Astra Medium` |
| Model surface | `Codex subscription model switch` |
| Model revision identity | `subscription-selected-astra-medium` |
| Reasoning effort | `medium` |
| Effort | `medium` |
| Execution mode | `manual-sequential-codex-handoff` |
| Data transfer | `private-reports-remain-in-user-controlled-codex-session` |
| Model identity approval reference | `tsk3.txt` |
| Approval scope | model identity, fixed settings and manual-handoff design |
| Prompt approval | separate and required |
| API / key / SDK / wrapper | explicitly prohibited and represented as false/absent |

The approval fingerprint covers the identity and policy record. The model
identity fingerprint excludes changing approval prose but covers the exact
experiment/model/surface/revision/settings/policy used to prevent mixing.
There is no provider checkpoint hash because Astra is selected through the
Codex subscription rather than loaded as a local artifact.

## Prompt artifact gate

There is intentionally no medical prompt in this commit. The owner later
creates an unapproved candidate with `make_astra_prompt_artifact()`, reviews
it, and freezes it by calling `prompt.approve(...)`. The serialized artifact
contains:

- prompt ID and version;
- exact prompt text and its text SHA-256;
- canonical target order and `ASTRA_OUTPUT_SCHEMA_VERSION`;
- approval reference, approver and date; and
- the computed prompt fingerprint.

`build_astra_batch_plan()` refuses an unapproved prompt. Loading the prompt
checks both the text hash and the fingerprint, so changing one character after
approval cannot reuse a prior plan. The prompt is the only source of medical
labeling policy: Astra is instructed to apply it, not design or revise it.

## Exact output contract

The top-level JSON object has exactly these keys:

```json
{
  "schema_version": "rsna-knee-ticket16-astra-output-v1",
  "experiment_id": "ticket16-astra-medium-codex-subscription-v1",
  "model_identity_fingerprint": "<64 lowercase hex>",
  "reasoning_effort": "medium",
  "effort": "medium",
  "prompt_fingerprint": "<64 lowercase hex>",
  "batch_id": "t16-astra-b0001-<16 lowercase hex>",
  "input_digest": "<64 lowercase hex>",
  "studies": [
    {
      "study_instance_uid": "<private study ID>",
      "targets": [
        {"target": "ACL", "state": "positive"},
        {"target": "MCL", "state": "negative"},
        {"target": "Medial Meniscus", "state": "not-mentioned"},
        {"target": "Lateral Meniscus", "state": "uncertain"},
        {"target": "Medial OA", "state": "unknown"},
        {"target": "Lateral OA", "state": "unknown"},
        {"target": "PF OA", "state": "not-mentioned"},
        {"target": "Effusion", "state": "unknown"},
        {"target": "Synovitis", "state": "unknown"},
        {"target": "Baker's", "state": "not-mentioned"},
        {"target": "Contusion", "state": "unknown"},
        {"target": "Fracture", "state": "negative"}
      ]
    }
  ]
}
```

The example states are structural only; they are not a medical policy or a
real report result. The target list must contain every canonical target once,
in order. No confidence, evidence, value, commentary, markdown or
`extraction-failed` target state is accepted. Semantic states map as follows:

| Astra state | Value | Mask |
| --- | ---: | ---: |
| `positive` | 1 | 1 |
| `negative` | 0 | 1 |
| `not-mentioned` | null | 0 |
| `uncertain` | null | 0 |
| `unknown` | null | 0 |
| `extraction-failed` | null | 0 |

The final row is pipeline-only. Invalid JSON/schema, malformed or incomplete
study/target sets, duplicates, missing studies, unknown studies, invalid
states, and output-order violations produce one all-target failure record for
every expected study. No label is guessed. A response whose experiment,
model identity, prompt fingerprint, reasoning, effort, batch ID or input
digest differs from the plan is rejected as a context mismatch and cannot
enter the completion ledger.

## Batching and fold isolation

Available reports are sorted by `StudyInstanceUID` and placed in batches of at
most 16. For `N` unique available reports, the expected number is
`ceil(N / 16)`. Batch IDs are stable hashes of the experiment, model identity,
prompt, fold manifest, source version, full report-set digest, batch position,
and ordered private report metadata/digests. A plan refuses duplicate studies
or a study appearing in two batches.

The batch input contains only the prompt artifact, output contract, stable
batch metadata, study IDs needed to map responses, and report text. It never
contains reference labels, target values, or held-out labels. The private
cohort contract mirrors Ticket 15:

- a gold report is consumed only by its assigned held-out evaluation fold;
- a report-only report is consumed only by the other two folds as a potential
  training-side report label;
- held-out reports are never admitted to a training-side partition; and
- the frozen fold manifest is the sole source of gold labels, with model
  states never overriding those labels.

`assert_astra_cohorts_match_ticket15()` compares manifest checksum, repeat,
report source/set digest, fold study sets, eligible sets, missing sets and
held-out exclusions against a Ticket 15 run. The Astra seam performs no
report interpretation and no learned-label training, so there is no
report-only accuracy claim in its public artifacts.

## Artifact and recovery design

Preparation writes, without overwriting:

```text
public/plan.json                              # text-free metadata
private/plan.json                             # identifier-bearing plan, no raw text
private/batches/<batch-id>/input.json         # private Codex input, includes report text
private/batches/<batch-id>/attempt-0001.raw.json
private/batches/<batch-id>/attempt-0001.audit.json
```

Raw response files preserve the exact returned text. Audit files preserve
input/model/prompt/effort digests, response digest, semantic output digest,
failure code, fold metadata and validated states, but do not duplicate raw
report text. Public plan/status files contain only counts, statuses and
digests; no report text, source references or study identifiers. The semantic
digest is computed from canonical states, values, masks and bound batch
context, not from response formatting.

Resume scans immutable audit files. A batch with a validated or
`extraction-failed` audit is complete and is not selected again. A rejected
context-mismatch attempt is retained but leaves the batch pending. Any retry
must be explicit, uses a new attempt number, and cannot overwrite the first
raw response or audit.

## Operator workflow after prompt review

```bash
# 1. Write the identity record; this performs no model action.
PYTHONPATH=src .venv/bin/python -m rsna_knee.astra_handoff_cli \
  write-approval --output /private/ticket-16/astra-approval.json

# 2. After the owner supplies the final prompt, create and explicitly approve
#    its immutable artifact (the command reads only that prompt file).
PYTHONPATH=src .venv/bin/python -m rsna_knee.astra_handoff_cli \
  write-prompt --prompt-file /private/ticket-16/final-prompt.txt \
  --approval-reference owner-prompt-review-<date> \
  --approved-by <owner> --approved-on <YYYY-MM-DD> \
  --output /private/ticket-16/approved-prompt.json

# 3. Prepare the private plan and inputs. This is the first command that reads
#    the private report corpus, and it still does not invoke Astra.
PYTHONPATH=src .venv/bin/python -m rsna_knee.astra_handoff_cli prepare \
  --fold-manifest /private/ticket-04/private/fold-manifest.json \
  --reports /private/reports/private/report-corpus.json \
  --approval /private/ticket-16/astra-approval.json \
  --prompt /private/ticket-16/approved-prompt.json \
  --public-plan /private/ticket-16/public/plan.json \
  --private-dir /private/ticket-16/private

# 4. For each ID from `pending`, manually switch the Codex model to Astra
#    Medium, set reasoning/effort to Medium, attach one input.json, and save
#    the response verbatim outside the repository.
PYTHONPATH=src .venv/bin/python -m rsna_knee.astra_handoff_cli pending \
  --private-plan /private/ticket-16/private/plan.json \
  --private-dir /private/ticket-16/private

# 5. Validate and persist one response; repeat only for explicitly pending IDs.
PYTHONPATH=src .venv/bin/python -m rsna_knee.astra_handoff_cli record \
  --private-plan /private/ticket-16/private/plan.json \
  --private-dir /private/ticket-16/private \
  --batch-id <stable-batch-id> \
  --raw-output /private/ticket-16/responses/<stable-batch-id>.json

# 6. Publish text-free status for Luna; this is not a metrics report.
PYTHONPATH=src .venv/bin/python -m rsna_knee.astra_handoff_cli publish \
  --private-plan /private/ticket-16/private/plan.json \
  --private-dir /private/ticket-16/private \
  --public-output /private/ticket-16/public/status.json
```

No command in this workflow invokes Astra automatically. No prompt tuning,
held-out adaptation, weak-label training, aggregation, comparison metric or
repository publication is performed by this handoff module.

## Historical verification and original gate

Synthetic tests cover identity/effort locking, prompt gating, stable batching,
fold isolation, exact output validation, state masks, malformed/incomplete
failure handling, context mismatch rejection, privacy, immutable artifacts and
resume behavior. Those statements describe the historical readiness gate
before the run; the final private execution and corrected aggregation are
recorded in the Final closure evidence section above.
