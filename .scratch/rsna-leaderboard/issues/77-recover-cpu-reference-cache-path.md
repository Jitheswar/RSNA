# 77: Recover the original CPU reference cache-path interface

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

Job8139 failed1:0 after374 CPU-node seconds before array export/model construction
or updates: JSON pixel_path was passed as a string to the original Path-only
cache loader. Preserve both8131/8139 failures, raw terminals, consumed requests
and conservative charges. No same-plan retry or frozen source mutation.

## Acceptance

- [x] Isolate and reproduce the actual frozen cache call with original selected binding.
- [x] Prove a separate worker differs only by Path conversion; actual hash-bound cache call passes.
- [x] Preserve old failure and add regression checks at the exact call seam.
- [x] Admit separate bounded CPU/history reserve retaining all existing delivery and review reserves.
- [x] Prepare a new unused request and exact copied-source/root/cache preflights.
- [x] Start separately frozen reserve owner/controller under the shared max-two CPU lock.
- [ ] Verify actual64-MRI reference completion and original boundaries/terminal.
- [ ] Derive restricted portable packets and source-gated manifest under the successful new identity.

## Comments

October3 IST: actual red/green/red call harness and two regression checks pass.
Corrected worker remains editable/unlaunched. Full evidence:
[cache-path diagnosis](../../../docs/evidence/ticket-77-cpu-reference-cache-path.md).
No daily scoring submission used; no additional resource admission or retry.

October3 IST: separate resource decision6b0e77...bc39 adds1800CPU seconds/100MB retaining both prior failed bounds, full pilot28800s/8GB and all final-fit/delivery/review reserves. Fresh conservative finishOct9 14:18UTC, projected1805460173590B. T74v4 actual ownerPID1407768 live, oldv3 source/PID retained/stopped; no scientific AST changes. New frozenCPUv3 requestd4551d20a9d380d811fe94f1ca64d16556fc11b1f0853a0032f32ec7d592a46c passes copied-root and actual cache-call checks before/after snapshot. Exact frozen one-attempt controllerPID1409117 genuinely live waiting for max-two slot; no new intent/acceptedCPU handle yet. Both new source-gated portable/manifest helpers reject pending source before output.39focused checks/mypy97/compile/diff PASS. Jobs8133 and8140 genuinely live; no daily scoring, upload or accelerator allocation. Evidence docs/evidence/ticket-77-separately-bounded-cpu-recovery.md.
