# Ticket75 restricted-load reference state

The original CPU request remains immutable and its actual controller is live
waiting for a free owner CPU slot. This turn adds a derived portable-reference
state contract and converter. No real-MRI reference packet, final attachment,
upload, accelerator request or scored competition submission was produced.

Inspection of the original trainer confirms its checkpoints retain a NumPy
MT19937 ndarray in RNG state and use trusted `weights_only=False` loading.
The frozen CPU reference worker preserves that original format. The new local
converter must not change the worker or rewrite its eventual checkpoints.
A synthetic original checkpoint reproduces the restricted-loader failure; the
converted packet loads with `weights_only=True` without NumPy safe-global
registration. Original checkpoint bytes and their SHA remain unchanged.

`src/rsna_knee/tpu_pilot_state.py` losslessly encodes the complete Python random
state, MT19937's624 uint32 words/cursor/Gaussian cache and CPU Torch RNG bytes.
The words use int64 tensors with checked unsigned32-bit support; this is a
serialization representation, not a changed RNG algorithm. Validation uses
private generators and does not consume global RNG. It rejects incomplete
schemas, invalid cursors, out-of-range words, unknown pickle objects and non-CPU
tensors. Model and Adam tensor dtypes/values, ordered state metadata, original
configuration, identity, progress, sampler and scheduler remain preserved.

The disposable CPU reference envelope retains the exact original parent file
SHA and0/3/8/16 boundary. Step0 cannot claim learned optimizer state. Later
boundaries require exactly four real-study exposures/update and the original
successful optimizer/scheduler/sampler counters. No packet can claim TPU
execution, original fit update credit, candidate promotion or scoring.
The strict loader checks exact packet SHA and source/boundary identity before
restricted loading.

`prepare_ticket75_portable_reference.py` only accepts the original CPU plan
f9977c944803733e6565cc5b23c30b4d27b04128ff7ef51c335d3d6a65be4428 after its actual
successful accepted Slurm handle, terminal/observation and hash-bound completion
receipt. It also rechecks all original MRI/target/mask attachment and state bytes.
Its source-host conversion may read the exact trusted legacy pickle, then must
strictly re-load every derived packet using restricted loading. It uses an
exclusive output directory and atomic publication, preserves all originals,
binds the complete original success/source chain, and checks the8GB retained
source/packet/serialization-temporary envelope. Only0/3 are prospective attached
states;8/16 remain local. Producing these reference packets still does not produce
a final pinned TPU attachment or admit accelerator execution.

Validation:18 new state tests pass. A synthetic ordinary Adam run restores all
state from an8-update restricted packet and continues to16 with exact model,
Adam, scheduler, sampler and all three RNG families, including cached Gaussian
draws. Ordered model metadata is also preserved. Tests reproduce original
restricted-load failure and prove converted restricted loading without altering
the original bytes; reject changed identities, unearned steps/exposures, missing
Adam, invalid RNG words/schema and executable/non-CPU values. With existing
replicated loss and original-pool input guards,54 tests pass. Mypy passes91 source
files; converter compilation passes.

The actual current-state converter guard was exercised before any conversion:
`Actual completed CPU preparation required before conversion`, and no portable
output directory was created. This is a useful verified absence, not actual
MRI/CPU2.8/TPU proof. Current jobs8128 fold0 stage6 and8127 fold1 stage5 are RUNNING;
CPU preparation controller PID1203487 is genuinely live. The existing T73 and
T74v2 controllers and all frozen sources/plans remain unchanged. No daily scored
submission was used. Actual TPU adapter, host CPU2.8 reference, complete pinned
attachment, fresh quota/vacancy and one real TPU execution remain pending.
