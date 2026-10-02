# Ticket75 complete disposable-pilot checkpoint state

Previous goal turn classification: progress. Commit59607ef corrected the editable
adapter's required before-update collective finite admission and passed142
pilot checks. This turn starts from a clean synchronized worktree and re-polls
actual training jobs8132/8133 and live T73/T74v3/corrected-reference controllers.
They remain active; no expired observation or missing receipt was treated as
permission to relaunch. The real CPU reference still waits for a free slot.

The new rsna_knee.tpu_pilot_checkpoint module implements the separately
identified disposable pilot's complete CPU-converted checkpoint contract. It
requires exact request, input, initial-state and registered scientific-plan SHA
identities; continuous/staged path, fixed0/3/8/16 boundaries and declared fresh
worker generation; model/config metadata, complete ordinary Adam/scheduler state,
fixed original batch cursor and all real-source exposure counts. The original
full parameter ordering, FP32 model/moment tensors, integer normalization
buffers, exact scalar Adam step, default rates/settings and nonnegative second
moments are enforced. No model parameter can silently disappear from Adam.

Every packet retains eight distinct rank entries, each with the complete encoded
Python/NumPy/Torch host RNG and the pinned XLA API's integer running RNG state.
The cached pinned API source was inspected for get_rng_state/set_rng_state and
CPU-converting save semantics; this turn neither installed nor invoked XLA.
The future worker must capture the actual synchronized rank states and use the
CPU-converting XLA save path before entering this CPU-only contract. Merely
providing synthetic integer RNG fields cannot prove device execution.

Atomic immutable publication uses an exclusive temporary file, flush/fsync,
actual byte-bound validation, no-overwrite hard-link publication and directory
fsync. An overbound or partial temporary remains preserved and prevents retry or
overwrite. Loading hashes and restricted-loads the same open file with
weights_only=True. All tensors must already be ordinary CPU tensors; NumPy RNG
pickle globals and direct accelerator tensor serialization are unnecessary.

The final replay check independently validates both exact declared step16 paths,
then requires type/tensor-exact model, Adam, scheduler, configuration, parameter
ordering, progress/exposures, batch cursor and all eight rank RNG states, including
ordered model metadata. There is no cross-backend tolerance applied to this
same-backend gate. Packet and replay receipts explicitly leave actual TPU
execution unproven, credit no original fitting updates and permit no competition
submission or candidate promotion.

Validation:31 new checkpoint checks pass, with142 existing pilot checks also
passing (173 distinct checks). They cover actual fresh CPU step8→16 model/Adam/
scheduler and host-RNG restoration, restricted serialization, original bytes and
metadata, missing rank7 or device RNG, invalid host RNG, wrong source/generation,
changed Adam step/moments/dtypes/order, omitted parameters, padding exposure,
unsupported objects/non-CPU tensors, atomic overwrite/overbound rejection, and
exact replay rejecting model/Adam or rank7 RNG changes. An additional original
full-backbone architecture test publishes and restricted-loads all parameters,
buffers and initialized ordinary Adam state using synthetic fixed gradients.
The smaller CPU replay fixture and synthetic gradients do not establish real
MRI fitting, fresh PJRT execution or actual eight-rank collectives. Mypy passes94
source modules; new files compile and diff checks pass.

The new module remains editable, not frozen into an accepted accelerator
request. Worker-side CPU conversion/rank-state collection, fresh PJRT supervisor,
actual device RNG restoration, model-only export, CPU2.8 reference, bounded
private attachment, fresh quota/vacancy admission and actual TPU/independent
native delivery remain pending. No frozen live CPU route was changed, no new
allocation/upload/scoring attempt occurred, and the original first-place goal
remains active and unproven.
