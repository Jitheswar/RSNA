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
- [ ] Admit separate bounded CPU/history reserve retaining all existing delivery and review reserves.
- [ ] Prepare a new unused request and exact copied-source/root/cache preflights.
- [ ] Start separately frozen reserve owner/controller under the shared max-two CPU lock.
- [ ] Verify actual64-MRI reference completion and original boundaries/terminal.
- [ ] Derive restricted portable packets and source-gated manifest under the successful new identity.

## Comments

October3 IST: actual red/green/red call harness and two regression checks pass.
Corrected worker remains editable/unlaunched. Full evidence:
[cache-path diagnosis](../../../docs/evidence/ticket-77-cpu-reference-cache-path.md).
No daily scoring submission used; no additional resource admission or retry.
