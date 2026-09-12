# Ticket 13 GPU experiment evidence

Recorded 2026-09-12 (Asia/Kolkata). This is public-safe implementation and
synthetic-fixture evidence. No GPU allocation, owner approval artifact,
competition data, private identifier, or model upload is included.

## Implemented seam

`rsna_knee.gpu` provides the approval-gated execution boundary:

1. `prepare_gpu_experiment` reloads a passed, checksum-covered CPU preflight,
   freezes the CUDA training configuration and exact single-GPU Slurm request,
   freezes a canonical scientific identity for the bound inputs, and writes an
   immutable `owner_approval_required` request. Preparing a request does not
   submit work.
2. `load_gpu_approval` requires an approval for the exact request checksum,
   the ticket-13 scope, and the request's declared retry policy.
3. `submit_gpu_experiment` builds one `sbatch` command with the exact
   partition, GRES, node/task, CPU, memory, wall-time and exclusivity fields.
   It never invokes `srun`. A scheduler retry is allowed only when the exact
   approval permits it and the failure is classified as transient. Durable
   in-job failure evidence is consulted first, so an in-job retry cannot be
   followed by an additional wrapper retry. A checksum-keyed private
   submission record consumes the approval after a successful `sbatch` and
   prevents duplicate initial submissions.
4. `run_approved_gpu_experiment` verifies the live Slurm allocation, records
   actual CUDA/nvidia-smi device, driver, runtime and memory facts, delegates
   training to the existing checkpointed fold trainer, reloads the final
   checkpoint, and writes execution evidence. It stops when assigned GPU
   memory is not observable, compares peak VRAM to the verified assignment,
   and keeps a retry on the same request/configuration using the durable latest
   checkpoint when available. Production study loading is deferred until
   these allocation checks complete. The execution authorization carries the
   request, approval, preflight and allocation checksums into the private
   trainer seam, which independently reloads and verifies those artifacts and
   the live allocation; the recomputed fold identity must match the approved
   identity. The execution record is written before lazy loading so a loader
   failure consumes the approval fail-closed.
5. `run_gpu_smoke_test` is explicitly restricted to a smoke-scoped request and
   the `cluster1-gpu-add` partition. It requires at least one example and uses
   the same forward, backward, checkpoint and recovery path as the full seam.

The GPU adapter is
`slurm/run_ticket13_gpu_experiment.sbatch`. Its default request is one
non-exclusive `cluster1-gpu` node/task with one GPU, 16 CPUs, 64 GB and six
hours. The adapter and Python layer both verify the allocation; nominal A40
capacity is never treated as assigned memory, and an unverified shared-memory
allocation is stopped before training. The bounded CPU preflight
adapter is `slurm/run_ticket13_cpu_preflight.sbatch`.

The repository `.venv` is CPU-only and is used only for the prerequisite
preflight. GPU execution requires a separate Python 3.11 environment described
by `environments/cluster-gpu/requirements.txt`; the adapter rejects the CPU
interpreter, requires Python 3.11, and checks CUDA availability before
private-data loading. The shared trainer also refuses CUDA execution outside
an active Slurm job. The
CUDA wheel/driver combination remains unverified until the approved cluster
smoke run. Smoke requests are capped at 60 minutes; live Slurm CPU, memory,
wall-time, GPU-count and sharing fields are required before training starts.

## Demonstrated scope

Synthetic tests cover immutable request and approval binding, CPU-preflight
gating, exact Slurm/GRES admission, hardware and assigned-memory recording,
forward/backward/checkpoint evidence, final-checkpoint reload, transient
retry, durable retry coordination, and immediate stop for cancellation, OOM,
configuration, empty-input, malformed-submission-output, and lazy-loader
failures. The shared training module records the approved operational resource
request while retaining its existing scientific identity and checkpoint
semantics; public retry evidence redacts exception text while private failure
evidence retains diagnostic detail.

The final checks run on 2026-09-13 are:

- focused GPU and training tests: 35 passed;
- full repository test suite: 192 passed;
- mypy for `src/rsna_knee`: passed;
- Bash syntax checks for both ticket-13 adapters: passed.

## Verification limits

No real `cluster1-cpu` preflight, owner-approved GPU request, `sbatch` GPU
submission, `cluster1-gpu` or `cluster1-gpu-add` allocation, A40 driver/memory
measurement, real-data forward/backward pass, or GPU checkpoint recovery was
run. Consequently V10 and every approval- or hardware-dependent acceptance
criterion remain open. The implementation does not mark those future checks
as passed and does not claim GPU compatibility, full-training results, runtime,
or scientific validity from synthetic tests.
