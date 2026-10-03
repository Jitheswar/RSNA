# Ticket75 model-only export and CPU replay preparation

Previous goal turn classification: progress. Commite6bd2ae implemented strict
fresh-worker staged checkpoint restoration and passed182 pilot checks. This
turn re-polls live training jobs8132/8133 and current T73/T74v3/corrected-reference
controllers. All remain active; the real CPU reference is still waiting for a
free slot. No observation expiry caused relaunch or retry.

The new CPU-only tpu_pilot_export module derives a model-only packet by loading
and hash-verifying the exact complete pilot checkpoint with weights_only=True.
Only a fixed final step16 checkpoint with the exact expected config is accepted.
The packet retains original model/config/parameter ordering/ordered metadata,
request/input/initial/scientific identities, parent checkpoint hash and frozen
source/license binding inventory. It includes no optimizer, scheduler, rank RNG
or training-progress state. Separate fields explicitly credit no registered
training updates and claim no actual TPU or original-MRI replay proof.

Source/license inventory requires nonempty unique named SHA bindings and exact
agreement with the expected frozen inventory. This is metadata identity
validation; actual source/license file bytes and license/provenance admission
must still be checked by the future bundle/native delivery verifier. Synthetic
inventory entries in tests are not actual model license evidence.

Model-only publication shares the existing exclusive temporary-file, byte-bound,
flush/fsync, no-overwrite hard-link and directory-fsync writer. This refactor
preserves complete-checkpoint publication behavior. Restricted CPU loading
hashes and deserializes the same open export file. Original architecture,
parameter ordering, buffers, tensor shapes/dtypes and metadata are checked
before a CPU model is mutated by strict restore.

The replay helper accepts exactly the eight prescribed study slots, original
16-series×8-slice geometry at the model's declared image size, FP32 CPU pixels/
descriptors and boolean validity masks. Every study needs valid MRI; nonfinite,
wrong-shaped or missing inputs fail without fallback. It evaluates the original
model path in fixed batch-two calls and returns complete finite8×12 logits.
The caller must independently decode/hash-check the prescribed original MRI and
record16-thread runtime/native source provenance. Input geometry alone is not
native replay proof and this helper does not pretend to decode DICOM.

Validation:21 new export checks plus182 existing pilot checks pass together
(203total). Tests use the actual original full-backbone architecture, synthetic
fixed-gradient initialized Adam checkpoint and synthetic32-pixel packed inputs.
Restricted model-only round-trip reproduces fixed batch-two logits within the
predeclared1e-6 CPU tolerance and retains no training state. Tests reject changed
parent/input/config/license identity, missing/duplicate inventory, injected
optimizer state, altered dtype/parameter order, nonfinite model tensors,
promotion/native-proof claims, wrong parent file hash and a nonfinal checkpoint.
Architecture mismatch is rejected before CPU model mutation; corrupt/missing
input studies fail before inference. Existing atomic checkpoint and all pilot
contracts pass after shared-writer refactoring. Mypy passes95 source modules;
new/modified files compile and diff checks pass. Synthetic32-pixel replay is
not real224-MRI, actual fresh PJRT, CPU2.8 or cross-backend inference proof.

The new module remains editable until an independently admitted request freezes
it. Real reference preparation, actual rank-state capture/CPU conversion and
global export writer, fresh PJRT supervisor, hostCPU2.8 reference, complete
private attachment, fresh quota/vacancy admission, TPU pilot execution and
independent original-MRI/native-license/runtime delivery remain pending. No
frozen live source, original science or resource owner was edited. No new
allocation/upload/daily scoring attempt occurred. First place remains unproven
and the original campaign goal stays active.
