# 03: Inventory and safely admit competition data

**What to build:** Produce a reconciled private corpus inventory and integrity-checked representative extraction batch ready for bounded real-input verification.

**Blocked by:** 02: Run a synthetic study through CPU inference

**Status:** ready-for-agent

**Authorization:** Planning artifact only. The owner has approved ticket publication, not implementation. Do not start work until separately instructed; ready-for-agent describes specification readiness, not execution permission.

**Governing constraints:** Follow the approved parent specification. Significant compute uses Slurm, never KLEF-MASTER. Once implementation is authorized, autonomous CPU work is limited to cluster1-cpu, one node, at most 16 CPUs, 64 GB requested memory, six hours, two concurrent jobs and no exclusive allocation; actual memory use must also be bounded. Every GPU training job, heavier CPU request, protected deletion, paid resource, model/data upload and competition submission retains its specified approval boundary. Preserve private data, the 2 TB soft budget and immutable working candidates. No gate below is marked verified by ticket publication.

## Acceptance criteria

- [ ] Account for the complete expected corpus, including unreadable headers; reconcile the reported 4,407 studies and 58 reference-labeled studies without treating historical counts as new audit evidence.
- [ ] Inventory transfer syntax, geometry, dimensions, slice counts and descriptors; preserve anatomical plane, fluid sensitivity and fat suppression as separate descriptors.
- [ ] Select all gold studies plus 50–200 representative report-only studies for initial pixel-cache work; metadata inventory covers the whole corpus without requiring full pixel preprocessing.
- [ ] Preserve source/archive identity and original data. Keep identifiers and manifests private; establish private-path ignore rules and public-safe schemas/checksums.
- [ ] Estimate peak extraction/storage use before admitting each bounded batch; exceeding the 2 TB soft budget requires approval.

## Verification gates

These retain the parent specification's acceptance criteria and blocking consequences. Where a gate spans stages, this ticket supplies only its demonstrated stage; later execution evidence remains pending.

### V01 — Archive and extraction integrity

- [ ] Record competition/source version and archive identity. Every file admitted to a selected extraction batch must pass the archive's integrity check and have a preserved content checksum; extraction paths must remain inside the intended destination. Reconcile expected and actual files and study/series membership for that batch with zero unexplained missing, extra, truncated or identity-conflicting entries. Full-corpus completeness may be claimed only after the same reconciliation covers the full corpus. Completion records must distinguish partial from verified batches.

**Blocking consequence:** Do not admit affected raw files or batches to caches, training or candidate evidence; quarantine and repair/re-extract them. Independently verified batches can proceed. Do not claim complete extraction or propose archive deletion until full required integrity, structure, source-version and usability checks pass; deletion still needs explicit approval.

### V03 — Transfer-syntax, geometry and series inventory

- [ ] Reconcile the corpus source inventory against study/series metadata and account for every expected instance as inspected or explicitly unreadable. Record counts for transfer syntax, frame count, dimensions, spatial-tag availability/consistency and series descriptors, with every unknown/error counted. Select representative complete integration series covering each observed supported syntax and relevant dimensional/geometry/series class; identify unsupported and unrepresented classes explicitly. A small header sample cannot satisfy corpus coverage.

**Blocking consequence:** Block claims of full-corpus compatibility and blanket cache expansion until coverage is established. Only explicitly inventoried, supported subsets may enter bounded verification. Unknown or unsupported classes remain quarantined/rejected and must contribute to coverage and failure reports rather than being silently dropped.

## Completion evidence

- [ ] Record the demonstration or research evidence, verified scope, unresolved gates and limitations without exposing restricted data. Do not mark a future execution or approval-dependent check as passed based solely on prepared code or documentation.

## Comments

### 2026-09-11 — Implementation evidence

Implemented the ticket-03 inventory and bounded-admission seams in
`src/rsna_knee/inventory.py` and `src/rsna_knee/archive.py`. The implementation
loads the source CSVs without retaining report text, distinguishes all
reference-labelled studies from report-only studies, inventories every archive
member's DICOM header without pixel preprocessing, records transfer syntax,
frame/dimension and geometry classes, and preserves the three source series
descriptors independently.

Partial reference-label rows are rejected from both the gold and report-only
populations. Geometry admission also records irregular slice spacing, and
reliably named scouts/localizers are excluded while absent or ambiguous series
descriptions remain explicit unknowns. Extraction rechecks the inventory's
archive identity and writes a failed completion record for rejected attempts.

It deterministically selects every gold study plus a 50–200 report-only sample
using coverage-seeking strata. It estimates retained archive, extraction,
cache, and atomic temporary usage against the decimal 2 TB soft budget. The
selection plan blocks incomplete series and transfer syntaxes without explicit
decoder support. `extract_batch` validates safe paths, reads selected members
to EOF so ZIP integrity is checked, records content SHA-256 checksums, and
publishes a fresh destination atomically. Private and aggregate public JSON
reports are written separately, and identifier-bearing paths are ignored by
Git.

The Slurm adapter is
`slurm/run_ticket03_inventory.sbatch`, bounded to one `cluster1-cpu` node,
eight CPUs, 32 GB, and six hours without exclusive allocation. Synthetic
archive tests cover source reconciliation, unreadable headers, geometry and
dimension rejection, deterministic selection, budget gating, public/private
schemas, safe extraction, and the complete report/extraction seam.

The private full-corpus header scan, selected real-data extraction, real pixel
decoder verification, and V01/V03 completion claims remain pending. Detailed
public-safe evidence is recorded in
`docs/evidence/ticket-03-inventory.md`.
