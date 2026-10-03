# Ticket75 source-gated input-bank sidecar

Added an explicit hash-bound external manifest to the editable frozen-bank
loader. The default manifest must still resolve inside the bank; selection and
all384 canonical array paths remain confined to the original array directory.
The sidecar is hashed through the same descriptor used for JSON parsing, retained
in metadata change detection, and reverified at completion. Original arrays are
never copied or modified by sidecar preparation.

The new preparation script first checks the actual CPU request, sole accepted
job, successful terminal, result/source closure, checkpoints and full original
attachment inventory through the existing verifier. Only after those gates and
all384 original array SHA/size bindings pass can it create an immutable metadata
directory. Its final receipt rechecks arrays and source bindings. Partial output
is retained and cannot be retried. This remains preparation only: no completed
private bundle, upload, CPU2.8/TPU proof, accelerator admission or scoring.

The actual script was executed while job8139 was running. It failed at the
required completed-CPU gate before creating its output directory. Job8139 later
failed during the first selected cache call, so this source remains ineligible;
the helper must be rebound to a separately admitted successful recovery before
it can produce genuine MRI metadata. No successful native manifest is claimed.

Validation:16 loader/preparation checks plus243 existing pilot checks pass
(259total). Three added loader checks cover an external sidecar using the
original sparse synthetic224 arrays, wrong/post-admission sidecar bytes, and
default-manifest symlink escape. Two preparation guards prove rejected source
chains create no output and partial preparations cannot consume sources again.
Synthetic inputs establish mechanics only. Mypy passes97 modules; compilation
and diff checks pass. No daily competition submission was used.
