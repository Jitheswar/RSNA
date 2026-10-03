# Ticket75 all-rank checkpoint capture

Previous goal turn classification: progress. Commit7d79251 admitted a separately
bounded no-update CPU cache-path recovery, froze its copied-source preflights and
started an actual waiting one-attempt controller. This turn verifies actual
training jobs8140/8141 and T73/T74v4/recovery process identities. Both CPU slots
remain occupied; no observation expiry authorizes retry or a third allocation.

The editable pilot adapter now has a concrete capture_checkpoint operation. It
requires the original full-model/config/device/optimizer/scheduler and exact
completed path/step/generation prefix, four admitted request/input/initial/science
SHA identities and an unused declared boundary. It consumes the boundary before
backend conversion; any later conversion/collective/publication/RNG failure
leaves the worker failed/in-flight and blocks capture/training retries. Partial
packet publications remain retained under the existing atomic no-overwrite
writer. Partial stage evidence remains in memory for the future worker to persist.

Every rank synchronizes and captures its host RNG plus the pinned integer device
RNG API state. The new CPU conversion helper flattens all tensor leaves for the
existing backend converter while rebuilding original mapping/list/tuple types
and OrderedDict metadata itself. Converted tensors must match original shapes/
dtypes, be ordinary CPU tensors and finite; CPU copies have no source aliases.
It retains full model/normalization buffers, Adam moments/step/groups, scheduler,
parameter order, complete config and all16 original supervision masks. Original
normalization values are checked against the adapter baseline before capture.

A deterministic content SHA covers types, order, shapes, dtypes, all tensor bytes,
scalars and ordered metadata. The checkpoint mesh exchanges small CPU records
containing each rank's complete-state SHA, exact identity/path/boundary/generation/
destination/byte-limit context and its RNG state. It rejects missing, duplicate,
boolean/out-of-range ranks, different state/config/masks/context and malformed RNG.
All eight states must match; no master-only state assertion can substitute.

Only global ordinal0 builds and atomically publishes the complete CPU packet.
All eight publication records must acknowledge the identical context, with exactly
one valid master SHA and no peer writer. The master also checks the acknowledged
SHA equals its actual publication. The packet remains restricted-load compatible,
with all eight rank RNG records. Checkpoint conversion/hash/publication time and
raw before/after metrics belong to the checkpoint-export phase, including all
readbacks; they are not excluded as untimed work or relabeled training fallback.

After publication each rank checks its own complete host/device RNG is unchanged.
The result claims all eight RNG records captured and LOCAL RNG preserved only;
global post-capture preservation remains false until the future supervisor checks
all eight completed receipts. Fresh-PJRT replay and complete-pilot proof likewise
remain false. A captured packet or mocked collective cannot prove those facts.
The future supervisor must enforce the shared600-second checkpoint/export phase
and3540-second whole-process deadlines; this method cannot interrupt a hung
collective and provides no independent timeout proof.

API evidence uses the pinned XLA2.8 source d517649bdef6ab0519c30c704bde8779c8216502:
_maybe_convert_to_cpu synchronizes/converts selected device tensors; mesh_reduce
transports CPU serialized records through rendezvous; get_rng_state returns the
integer device state. The official immutable source was fetched and confirmed
byte-identical to the cached source SHA
e0738da80ef043be25e010ead99eada39849db2c596a7b37bfa530aa9783262a.
Installed offline wheel identity and actual hardware behavior remain unverified.
[Official pinned XLA source](https://github.com/pytorch/xla/blob/d517649bdef6ab0519c30c704bde8779c8216502/torch_xla/core/xla_model.py).

Validation:36 new capture guards plus261 existing pilot checks pass together
(297total). Tests exercise the original full architecture with synthetic CPU
constant-gradient Adam step3, actual bounded serialization/restricted reload,
ordered model metadata and all tensor equality, RNG invariance, non-master no
writes, conversion/rank-divergence/missing-rank failures, overbound retained
partial publication, changed device RNG after publication, consumed boundary/
failed-worker rejection and pre-backend wrong-prefix rejection. Device conversion/
mesh/RNG APIs are mocked; pure CPU tree/collector checks also cover lossy copies,
unsupported objects, every fingerprint detail and exactly-one-master transport.
After narrowing the result to local versus global RNG proof, all8 adapter capture
checks pass again. Mypy98/compilation/diff checks pass. No real MRI/CPU2.8/TPU or
fresh runtime evidence is claimed from these checks.

The new helper/adapter remain outside every accepted TPU request. Frozen T73,
T74v4 and CPUv3 sources/closures are unchanged and reverified. T74v4 PID1407768
and corrected reference controllerPID1409117 are genuinely live; the reference
has no consumed launch intent or accepted CPU job. Training8140 finalfold0 and
8141 finalfold1 remain genuinely running. Actual CPU reference completion,
source-bound portable/manifest derivation, complete private attachment/rank
worker/fresh-PJRT supervisor, runtime/quota admission, actual TPU stability/parity/
exact replay and independent native delivery remain pending. No daily scored
submission, upload or accelerator allocation occurred.
