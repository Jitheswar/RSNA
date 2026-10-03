# Ticket77 original cache path diagnosis

The separately admitted corrected-root CPU reference8139 failed after374
CPU-node seconds, actual controller FAILED/1:0, before the first array export,
model construction or optimizer update. Its reference-started record, immutable
request45f34a3ff1de8570b71bbae25695e3237013834da5c5e369a2a40a0242f8c35d,
accepted handle, stderr and terminal remain preserved. The one-attempt lifecycle
stopped; no requeue/relaunch occurred. Previous8131 failure88 seconds and both
conservative1800-second request charges remain retained.

The terminal traceback is load_pixel_entry(binding['pixel_path']) followed by
AttributeError: 'str' object has no attribute 'is_file'. The actual frozen loader
expects pathlib.Path; JSON carries a string. An AST-call harness isolates the
exact worker expression and original hash-bound selected cache entry without
native decoding, model construction, training, allocation or scoring.

Feedback loop run before changing the worker:

    .venv/bin/python scripts/check_ticket77_cache_path.py --worker artifacts/private/ticket75-cpu-reference-v2/source-snapshot/ticket75_cpu_reference.py --expect pass

This failed with the same frozen-worker/loader traceback and exact AttributeError.
Ranked hypotheses were path conversion, copied-loader API mismatch, and invalid
cache bytes. Both loader files are identical SHA
72075e3f8d180f060a4b35e1b44c8ef6b7471193ef3754d83d06d8dbc0e0a78a.

Only a new editable worker was changed: ticket77_cpu_reference.py wraps the
existing JSON path in Path. Every other byte matches the original worker.
The same call harness passes using the genuine selected cache, whose SHA is
7609abb08a22f633b02c238078e1129a8402fb3ed5bdc844c54160c0ddad26ca.
The corrected worker SHA is
dee537e07bc1b135a71516d69b8afa55a5b61dbed4cf28fd1e403377afdbd97a.
Rerunning the frozen worker with --expect legacy-path-failure reproduces the old
failure. This establishes the path boundary correction, not successful native
MRI replay, complete reference execution or a new resource admission.

Two regression checks pass: executing the actual old/new call preserves every
identity argument while correcting the path type; byte comparison and normalized
full AST comparison prove no other worker logic changed. Combined with259 pilot
checks,261 checks pass. Mypy97/compilation/diff checks pass. No debug logging was
added to the scientific worker.

Remaining work: separately account another bounded preparation/history reserve,
retain8131/8139 charges and every original final-fit/delivery/review reserve,
prepare a new unused request and copied corrected worker, verify copied-root and
cache-call preflights, and start a sole one-attempt controller under the shared
max-two CPU lock. Update the unused T74 reserve owner through a new frozen
lifecycle; do not edit existing frozen owners, requests or scientific closures.
Portable reference/manifest preparation must require the new actual successful
terminal. No such request has been launched in this change.

At verification, T73 owner897091 and T74v3 owner1261544 are active. Fold1 job8132
completed7000 updates with actual COMPLETE/0:0 terminal. T73 automatically
launched final fold0 stage7 job8140; it and fold2 stage1 job8133 are running.
No TPU/GPU allocation, asset upload or daily scoring submission occurred.
