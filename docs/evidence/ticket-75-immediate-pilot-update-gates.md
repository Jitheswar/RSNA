# Ticket75 immediate pilot update gates

The fixed telemetry gates previously ran only on the complete 256-observation
pilot. The editable XLA adapter now validates each rank immediately after its
actual update, before permitting another update. The second update establishes
initialized ordinary Adam's graph baseline; the third and later updates in each
of the three fresh worker generations reject compilation or graph churn against
that exact same-rank, same-generation baseline. Missing compile/execute evidence,
invalid graph hashes, unsupported training CPU fallback and cold intervals over
600 seconds fail the same shared validator used by the complete final summary.
The original thresholds and complete final acceptance gates remain unchanged.

A fresh adapter accepts only its original consecutive worker prefix: continuous
1–16, staged 1–8, or the fresh staged generation 9–16. Skipped/repeated steps,
path/generation reuse and a consumed failed or in-flight update are rejected
before backend work. Partial raw before/training/guard metrics and the current
observation remain available in last_update_evidence for the eventual worker's
failure recorder. An exception after backend work leaves the adapter consumed;
there is no hidden retry. This evidence remains in memory until that future
worker explicitly persists it; this module does not claim an on-disk failure
receipt or whole-pilot supervision.

The supervisor-facing validate_fixed_pilot_prefix gate checks a nonempty unique
consecutive prefix with every actual rank present at each completed update. It
uses the maximum synchronized interval across all eight ranks for each update,
rejects cumulative remaining-update time over the fixed 480-second envelope,
and can be applied before scheduling the next update. A passing prefix explicitly
does not establish actual TPU execution or permit a competition submission.
The future supervisor must collect/persist these observations and invoke this
gate; that supervisor and global barrier are still pending.

The original optimizer guard additionally rejects decoupled_weight_decay=True
when that newer-runtime field is present, while preserving compatibility with
the original CPU runtime that omits it. Ordinary Adam's original rates, groups,
parameter order, precision and update semantics remain unchanged.

Validation:129 pilot tests pass, including all 32 complete prefixes, immediate
rank7 compilation failures in each fresh generation, absent/foreign initialized
baselines, missing ranks, duplicate/skipped observations, unknown execution,
cold and cumulative time overruns, and pre-backend worker-prefix/consumed-failure
rejection. Existing complete-summary, loss, input and portable-state tests also
pass. Mypy passes all93 source modules; updated files compile and diff checks
pass. These are CPU contract tests, not actual XLA or real224-MRI TPU execution.

At the latest observation on October3 IST, actual job8128 was COMPLETED and its
fold0 stage6 result recorded7000 successful updates. Jobs8132 (fold1 stage6) and
8133 (fold2 stage1) were RUNNING. The sole T73 scheduler, T74v3 analysis waiter
and corrected CPU-reference waiter remain active. CPU-reference v2 has not yet
created a launch intent or accepted job. No live/frozen scientific source,
request, checkpoint or reserve owner was edited. No TPU allocation, upload or
daily scoring submission was made. The overall first-place goal is active.
