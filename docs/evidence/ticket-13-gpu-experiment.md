# Ticket 13 GPU experiment evidence

Recorded 2026-09-14 (Asia/Kolkata). This public document contains safe
implementation and bounded verification evidence. No competition data,
private identifier, or model upload is included.

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
preflight. GPU execution uses a separate Python 3.11 environment described by
`environments/cluster-gpu/requirements.txt`; the adapter rejects the CPU
interpreter, requires Python 3.11, and checks CUDA availability before
private-data loading. The shared trainer also refuses CUDA execution outside
an active Slurm job. The pinned CUDA wheel/driver combination was verified by
the bounded smoke below. Smoke requests are capped at 60 minutes; live Slurm
CPU, memory, wall-time, GPU-count and sharing fields are required before
training starts.

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

## Approved bounded smoke verification — 2026-09-14

One fresh, exact request was approved and submitted once through `sbatch` after
the passed CPU preflight. The scientific configuration and resource shape were
unchanged from the prior failed attempts; commit
`98fe68529cdb928c53c7989445c3ad9e8df84738` supplied the training-artifact
isolation fix.

| Item | Observed value |
| --- | --- |
| Request checksum | `6e80763415f8bd37cfaf73d0a2fcdea33af6786e9053ea5b5429d3f651fd5864` |
| Approval checksum | `2dcdeaac28c8f0ed5936a7fbdabe48e4d60060dfd826adef6c02261521e7fed9` |
| CPU preflight checksum | `33455298896a53ec634f4e425838e5292bfd2bd7a4882e1b3cc9d8a55f7ce50f` |
| Slurm job / node | `6979` / `KLEF-G-NODE2` |
| Partition / GRES | `cluster1-gpu-add` / `gpu:1(IDX:0)` |
| Nodes / tasks / CPUs | `1 / 1 / 16` |
| Host memory / wall time | `64 GB / 30 minutes` |
| Sharing / isolation | non-fractional, `none`, device isolation `yes` |
| GPU | NVIDIA A30X, compute capability `8.0` |
| Driver / CUDA | `580.126.20` / `12.6` |
| Assigned GPU memory | `25,337,004,032` bytes |
| Free GPU memory before / after | `25,090,785,280` / `23,975,100,416` bytes |
| Peak VRAM / assigned-bound check | `487,037,440` bytes / passed |
| Runtime | `00:07:13`, exit `0:0` |
| Retries | `0` |

The live allocation proof had every required whole-device check true,
including the concrete `gpu:1(IDX:0)` allocation, `/dev/nvidia0` mapping,
node mapping, generic GRES device file, non-fractional mode and Slurm
`select/cons_res`/`CR_CPU` configuration. The observed device is recorded
honestly as an A30X; no nominal A40 capacity was used.

The production private-data loader completed for the checksum-bound fold 0
inputs, yielding 38 training studies and 20 held-out studies. The approved
model shape completed one forward/backward pass and one successful Adam
optimizer step with batch size 2. Loss was finite at
`0.6996539831161499`; the trainer's finite-gradient guard passed. Three
checkpoint events were recorded (initial, target-reached latest, and final),
the final checkpoint reload passed, and the final checkpoint SHA-256 was
`b5ead4b2e7bdc1081529cd4d8f684a25501f7ff672d9c0e7740855176f3ce907`.

The execution root retained its evidence files, while all trainer outputs
were written only below its `training/` child. This was one bounded smoke
step, not full training, an A40 equivalence claim, a competition submission,
or a Ticket 14 experiment.

## Verification limits

V10 is verified only for the exact one-step `cluster1-gpu-add` allocation
recorded above. The evidence does not establish full-training performance,
capacity for a nominal A40, behavior on another GPU/resource shape, or
scientific validity beyond the bounded mechanics smoke. No full training,
retry, Kaggle submission, or Ticket 14 work was performed.
