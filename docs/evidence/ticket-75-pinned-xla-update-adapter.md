# Ticket75 pinned XLA update adapter preparation

The preceding goal turn added the original replica input mapping and complete
fixed telemetry gates. This turn implements one update inside a future admitted
fresh PJRT worker. No actual accelerator/backend update, upload or scored
competition submission was performed. First place remains unverified.

`scripts/ticket75_xla_update.py` has no launcher or CLI and acquires no device at
import. Backend initialization requires explicit admitted-worker context,
complete exact installed API source hashes, pinned original CPU-wheel runtime,
Python3.12.13/Torch2.8.0+cpu/XLA2.8.0/NumPy2.5.0/libtpu0.0.17, no CUDA, replicated
mode, actual eight-rank TPU topology and verified highest backend precision.
Reduced-precision environment settings are rejected before importing XLA.
The eventual accepted-resource/input/runtime worker and supervisor are still
required; merely importing this module does not establish that admission.

The update keeps the original full-backbone model, every FP32 encoder/head
parameter, exact original parameter ordering, ordinary Adam defaults and rates
1e-5/1e-4, and constant scheduler. Exact CPU optimizer scalar steps and complete
moment shapes/dtypes are checked before and after each update. Encoder
normalization stays fixed. The original mapped replica loss is backpropagated,
then the public pinned mean gradient reduction executes **once**, followed by
ordinary Adam and scheduler, then synchronous completion. It never calls a
second gradient-reducing optimizer helper or silently changes optimizer.

The direct reduction/step ordering matches the public optimizer helper's body
in the pinned source. Actual supported counter/metric data use the pinned
[metrics API](https://github.com/pytorch/xla/blob/d517649bdef6ab0519c30c704bde8779c8216502/torch_xla/debug/metrics.py).
Both ExecuteTime and ExecuteReplicatedTime samples are retained; missing
execution/compile evidence is unknown rather than zero. Raw samples, all counters
and executed fallback inventory are recorded before training, after training
and after finite/readback work. Counter reset, unsupported training fallback,
and fallback operations without corresponding counter evidence fail.
Forward/backward and optimizer phase graph hashes use the actual pinned API
also used by the official
[Dynamo bridge](https://github.com/pytorch/xla/blob/d517649bdef6ab0519c30c704bde8779c8216502/torch_xla/_dynamo/dynamo_bridge.py);
missing hashes cannot become synthetic success.

A separate, timed collective finite guard checks loss, every gradient/model
parameter/moment, nonnegative second moments and unchanged normalization
buffers across all eight ranks. Training-study loss is reduced by the same world
mean for diagnosis. The prescribed first3 reduced gradients are exported on CPU;
there is no option to omit that reference gate. Later updates avoid full
parameter/gradient readback. All readback/finite-guard costs remain included in
the synchronized interval, with training compilation/fallback phase separately
observed. The existing complete telemetry and state-parity gates must still
be applied by the future worker/supervisor; one update is not a completed pilot.

Pinned additional primary-source captures are ignored under
`artifacts/private/ticket75-adapter-api-source-v1/`. Metrics source SHA256 is
5b516613ec8781c44c0f22178048ac39ad8ea25bf248e94a6651e8ca1fe01475; bridge source SHA256 is
d4566564fef3f92847e0d80ffe0834a96a66bbfb6a8e036a72249ac702a9f79f. Existing pinned
XLA model/reduction/precision source captures are unchanged. These captures
prove referenced source identity, not the future installed wheel identity.

Validation:16 new CPU checks exercise ordinary Adam's exact full-encoder scope,
one actual synthetic32-pixel full-backbone gradient/update with unchanged
buffers, altered rates/groups/optimizer settings/frozen parameters, rejected
unadmitted/low-precision/missing-source initialization before XLA import,
retained raw metrics and both execution metric names, absent evidence,
counter reset/unobserved fallback and missing graph hash. With existing pilot
input/telemetry/state/loss/selection tests,108 pass. Mypy remains clean across93
source modules; adapter compiles. These tests do not execute XLA or pretend that
synthetic32-pixel inputs are the real224-MRI pilot.

The adapter is not frozen into an accepted TPU request and remains editable.
The real CPU reference still awaits a slot; jobs8128 fold0 stage6 and8127 fold1
stage5 were observed RUNNING, CPU preparation PID1203487 genuinely live and no
accepted preparation job yet. All frozen T73/T74v2/CPU sources and histories are
unchanged. Original input assembly, actual hostCPU2.8 reference/gradient/state
comparison, fresh-process16/staged8+8 supervision and RNG/checkpoint/export
handling, complete private attachment, quota/vacancy admission, actual TPU run
and independent original CPU delivery remain pending. No daily scoring slot
was used.
