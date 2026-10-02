# Ticket76 copied-helper root diagnosis and separate recovery

Actual CPU reference8131 was accepted once under the immutable Ticket75 plan
f9977c944803733e6565cc5b23c30b4d27b04128ff7ef51c335d3d6a65be4428. Its actual
Slurm terminal is FAILED/1:0,16 CPUs/64GiB/nonexclusive/no requeue,88 seconds
from start to end. The observation command's returncode0 proves retrieval,
not a successful job. The original consumed intent and all sources, raw receipt
and stderr are preserved. There is no attachment/reference directory, optimizer
history or successful result; failure occurred in the original fold-manifest
call before original label examples, source replay/export or optimizer updates.

I omitted RSNA_PROJECT_ROOT from the CPU launcher. The unchanged copied
`rsna_knee.ticket36` helper consequently used its snapshot location as project
root, looking for the original fold manifest inside that snapshot. Existing
full-training launchers already supply the variable. This is an operational
launch bug; the original manifest and scientific model/labels were not changed.

The diagnosis workflow uses a fast actual-import repro:
`check_ticket76_snapshot_paths.py <actual-source-snapshot>`. With the variable
removed it reproduces the exact FoldManifestIntegrityError and missing snapshot
path. Ranked possibilities were omitted root, missing/changed original manifest,
and incompatible copied path constants. With only the root variable corrected,
the exact call passes, verifies original manifest SHA
6db019d04ca94a5af688f860db2a46becc1b432972fdf925642254e2cb5a6bb9/corrected fold
identity, and checks original model/supervision path presence. Wrong root fails
again. This reads metadata only, with no MRI or optimizer work. Source/label
admission separately rechecks the original complete T73 input closure.
An initial overly broad harness path restriction was removed: the report-label
artifact legitimately lives outside the project; the bug test now measures the
actual failing manifest call rather than inventing a new directory policy.

The new corrected request is not a repeat of8131's consumed intent. It has its
own run root `artifacts/private/ticket75-cpu-reference-v2`, immutable plan SHA
45f34a3ff1de8570b71bbae25695e3237013834da5c5e369a2a40a0242f8c35d, source bindings,
launcher and one-attempt controller. The original entire CPU reference worker
is byte-identical; only operational root/path/request context is changed.
Its actual launcher exports RSNA_PROJECT_ROOT. The copied helpers also pass the
same metadata repro against this newly frozen snapshot before any allocation.
No original training update is repeated or credited.

New resource decision
`.scratch/rsna-leaderboard/resource-approvals/ticket76-cpu-root-recovery-20261003.json`,
SHA13435027cfd592f5b4e440546335eae13d081d0995912864a8ccec58703022c5, preserves
actual failed88 seconds and the full1800-second failed bound conservatively;
adds a distinct1800-second CPU preparation and100MB source/history allowance.
The entire original T73/T74/T75 future calendar and storage protections stay
intact, including original7200-second additional T74 analysis, full28800-second
pilot route, all final-fit/delivery/queue/review terms and8GB retained pilot assets.
Past failure charges are separate from future calendar. No old consumed budget
or launch intent is reused. Standing owner authorization covers this agent
admission; no new exact numeric owner approval is claimed. TPU quota/execution,
conditional final fit and scoring remain unadmitted.

Fresh complete-route calendar sums563400 seconds, finishOctober9,11:47:02 UTC
before the October11,00:00 UTC conservative science admission cutoff.
Fresh existing apparent storage1,383,306,165,792B plus protected399,521,980,644B,
8GB pilot/100MB analysis/100MB recovery projects1,791,028,146,436B. Only previously
proved completed-control duplicate future storage allowances are credited;
all physical files and every outstanding treatment/final-fit allowance remain.
The new launch repeats fresh entire-route calendar/storage admission.

Unused T74v2 waiter was actually stopped; its frozen source and start receipt
remain. Separately frozen T74v3 now owns both the original pilot reserve and
new recovery reserve. Original scientific plan/protocol and terminal, launch,
watch and completed-input reserve-credit function ASTs stay identical.
Its real unit/PID1261544 runs the frozen v3 source. Shared final-resource lock
prevents admitting the recovery after analysis input/resource admission.

Recovery controller `rsna-ticket76-cpu-reference-v1.service` is actually live
PID1265703; `/proc` command identifies the exact new source and plan. At observation
there is no new launch intent or accepted job: owner slots contain RUNNING
8128 fold0/stage6 and8132 fold1/stage6. Original8127 fold1/stage5 completed
successfully. The new controller preserves the two pending/running CPU job
limit and one-attempt/no-ambiguous-repeat behavior.

The separately prepared `prepare_ticket76_portable_reference.py` uses the new
request identity and retains its operational recovery history. It still refuses
conversion without genuine CPU success and preserves all original checkpoints.
No actual portable output, full64-MRI reference, TPU pilot or predictive gain is
claimed. Both this issue and the public/private first-place goal remain active.

Validation:17 focused actual-path, T74v3 reserve and new CPU-lifecycle tests pass;
mypy remains clean across93 source modules and all new scripts compile.
The actual red and corrected green helper loops both ran; the new frozen
snapshot also passed. No debug instrumentation remains. Original frozen sources
were not patched, histories were not rewritten, and daily scored submissions
used in this work are zero.
