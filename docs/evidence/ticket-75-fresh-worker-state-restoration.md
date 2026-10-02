# Ticket75 fresh-worker state restoration

Previous goal turn classification: progress. Commitf37a21b introduced the complete
separately identified CPU-converted checkpoint contract and passed173 relevant
checks. This turn began from the clean worktree and polled actual jobs8132/8133
and the active T73 scheduler, T74v3 analysis waiter and corrected CPU-reference
waiter. All remain live. No timeout or missing completion receipt triggered a
retry; the real CPU reference still has no accepted job.

The editable XlaPilotUpdate adapter now restores only the exact staged step8
checkpoint into an unused fresh generation1 worker. It validates request/input/
initial/scientific identities, full checkpoint state, exact expected config,
original CPU full-backbone model configuration, complete tensor keys/shapes/
dtypes/metadata and encoder/head parameter order before mutating anything.
A previously trained or restore-consumed adapter cannot reuse its runtime.

After preflight, the single restore becomes consumed. The model loads strictly
on CPU and moves to the already admitted device before original Adam/scheduler
construction, so optimizer groups refer to current parameters. Complete optimizer
and scheduler states load from copies: later updates cannot mutate the checkpoint
through CPU tensor aliasing. The optimizer guard additionally requires both Adam
moments on their corresponding parameter's device while scalar steps stay on CPU.

This ordering follows the pinned
[Torch2.8 optimizer loader](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py):
ordinary noncapturable/nonfused step tensors retain their current device, whereas
floating moments follow parameter device/dtype. The installed optimizer loader
source is now required alongside the four pinned XLA API sources before backend
initialization. A primary-source capture under ignored
artifacts/private/ticket75-restore-api-source-v1 records optimizer.py SHA256
22e21ad10aa7cd994189a9fa7d19c169df4dc9d0c4f4a8dc3fca6aa4568fcabe,49829bytes.
This capture proves referenced source identity; it does not prove that a future
offline installed wheel matches. That installed-byte check remains mandatory.

Model/device/state/normalization initialization completes synchronously before
setting and verifying the actual current rank's device RNG and restoring all
host RNG families. No further initialization is queued after RNG restoration.
The method reports next batch index8 and fresh generation1, with no whole-pilot
or actual TPU proof. Generation1 training is rejected without successful exact
restore; a partially failed device restore remains consumed and cannot be retried
or treated as success. The future supervisor must still prove a genuinely fresh
PJRT process, actual per-rank device RNG behavior and exact continuous replay.

Validation:9 new restore checks and the existing173 pilot checks pass together
(182total). They use actual original full-backbone CPU model/Adam/scheduler state
with synthetic fixed gradients, replacing only device synchronization/RNG
boundaries. Successful rank7 restoration retains every model tensor and CPU
scalar step, avoids checkpoint moment aliasing, restores host RNG after fresh
model construction, and advances restored Adam/scheduler8→16. Used workers,
wrong path/source/config/architecture/buffer shape, failed device RNG verification,
second restore and unrestored generation1 training are rejected. Bad preflight
leaves the target model unchanged; partial device failure remains consumed.
Mypy passes94 source modules, modified files compile and diff checks pass.
These checks are CPU restore behavior, not fresh TPU execution, device migration
or same-backend TPU RNG/collective proof.

Actual rank capture/CPU conversion and global writer, model-only export, fresh
PJRT supervisor, hostCPU2.8 reference, complete bounded private attachment,
quota/vacancy admission, actual TPU pilot and independent native delivery remain
pending. No frozen live scientific source or resource owner was edited. No new
CPU/TPU allocation, upload or competition scoring submission occurred. First
place on either leaderboard remains unproven and the original goal stays active.
