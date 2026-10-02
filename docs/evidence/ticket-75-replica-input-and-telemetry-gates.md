# Ticket75 replica inputs and fixed backend telemetry gates

The previous goal turn advanced portable checkpoint/RNG conversion and proved
synthetic exact fresh replay. This turn adds the prospective input mapping and
backend-observation gates needed by the one fixed TPU pilot. No actual MRI
export, backend observation, accelerator request, upload or score was created.
The public and final private first-place objective remains unverified.

`src/rsna_knee/tpu_pilot_batches.py` admits exactly the original four-study CPU
batch: FP32 images[4,16,8,1,224,224], boolean slice/series/supervision masks,
FP32 ten-column descriptors and12 targets. Each real draw must retain valid MRI
and nonempty finite bounded original supervised entries. Absent NaN targets are
preserved. The mapping keeps reference/reference/weak/weak on ranks0–3 with the
original values and shape[1,16,8,1,224,224]. Ranks4–7 receive zero pixels and
descriptors with every image and supervision mask false, and zero sample/target
exposure. Every rank receives the same original four-study supervised-entry
denominator for the already tested8*N_rank/D_global mean-reduced loss.

All admission/scalar/exposure accounting happens on CPU before transfer. No
rank is resampled and padding cannot replace an original real draw. Source
exposure totals use the complete16-mask sequence and exact original prefix at
0/3/8/16, with two reference plus two weak samples/update and zero mixed/pseudo
exposure. This is metadata/tensor structure proof, not native source MRI proof.

`src/rsna_knee/tpu_pilot_telemetry.py` checks all256 per-rank observations for
32 physical updates across continuous16 and staged8+fresh8. These paths repeat
16 diagnostic learning updates; they are not32 new updates or original T64
fit credit. The staged resume retains steps9–16 and its distinct declared
worker generation. Missing or duplicate observations, absent execute/compile
metrics or phase graph hashes, and unobserved fallback counters cannot pass.

Training-phase unsupported CPU fallback fails, including scalar fallback. The
worker must capture separately timed finite/readback/checkpoint phases; none can
conceal compilation or fallback belonging to prediction/backpropagation.
The first update includes cold execution and the second establishes initialized
ordinary Adam. Subsequent training steps must retain both forward/backward and
optimizer graph hashes, with zero additional compilation. Ordinary Adam remains
required by the fixed pilot; graph churn fails rather than changing optimizer or
widening the gate. No learning update is an untimed/discarded warmup.

Timing uses maximum across the eight ranks. A cold update exceeding600 seconds
or the sum of the remaining29 synchronized global update intervals exceeding
480 seconds fails the fixed design. These necessary update checks do not prove
the whole startup/cold/host-reference/checkpoint/global3540-second budget: that
requires the separately recorded actual phases. Results explicitly leave actual
backend identity, real MRI identity, checkpoint parity and TPU execution unproven;
synthetic telemetry is never an actual accelerator success claim.

Validation:16 input/exposure tests and22 telemetry tests pass. They exercise
all eight original roles, retained absent targets, common denominator, masked
loss/gradient equivalence, exact source-prefix counts, changed packing/dtypes,
empty real draws, subset resampling, missing ranks/generations/metrics/hashes,
unsupported fallback on rank7, repeated steady compilation/graph changes,
nonfinite timing and both individual/aggregate timing overruns. Together with
18 portable-state,19 loss and17 sampler/input tests,92 pass. Mypy passes93 source
files; both new modules compile. These modules are not yet frozen into an actual
TPU request and remain editable during adapter preparation.

At the actual observation jobs8128 fold0 stage6 and8127 fold1 stage5 are RUNNING.
T73 controller PID897091, amended T74 waiter PID1199029 and original CPU preparation
controller PID1203487 are genuinely live. Original frozen workers, source plans,
resource decisions and histories remain unchanged. The real CPU reference still
awaits a free slot. The actual TPU worker/supervisor, host CPU2.8 comparison,
complete pinned attachment, fresh quota/vacancy, real execution and independent
CPU delivery remain pending. Daily scored submissions used in this work: zero.
