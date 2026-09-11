# Ticket 03 inventory and admission evidence

Recorded 2026-09-11 (Asia/Kolkata). This is public-safe implementation
evidence; it contains no competition identifiers, reports, raw paths, or
private manifest contents.

## Demonstrated scope

The public synthetic archive seam demonstrates:

- source CSV parsing with separate reference-labelled and report-only study
  populations, while retaining anatomical plane, fluid sensitivity, and fat
  suppression as independent series descriptors;
- header-only DICOM inspection, including transfer syntax, frame count,
  dimensions, geometry status, spacing consistency, diagnostic-series
  classification, and explicit unreadable-header accounting;
- deterministic coverage-seeking selection of all gold studies plus a bounded
  report-only sample;
- storage peak estimation with a hard 2 TB soft-budget admission gate;
- fresh-destination extraction that verifies ZIP member size/CRC while reading
  to EOF, records SHA-256 content checksums, and publishes only after atomic
  reconciliation; and
- identifier-free public summaries alongside private UID/path-bearing audit
  records.

The bounded Slurm adapter requests one `cluster1-cpu` node, eight CPUs, 32 GB,
and six hours without exclusive allocation. The full archive scan is
header-only and the optional extraction is explicitly gated on reviewed
transfer-syntax support and storage evidence.

## Tests

The ticket-specific inventory and admission tests pass in the pinned Python
3.11 environment. The existing mechanics tests continue to run against the
same package and no new runtime dependency was added.

## Unresolved gates

- The private competition archive has not been claimed as fully inventoried by
  this document. Run `slurm/run_ticket03_inventory.sbatch` and preserve its
  private completion records before treating V01 or V03 as verified.
- Real transfer-syntax pixel decoding, real preprocessing, cache usability,
  and representative-series integration remain pending later verification
  gates. Header readability is not pixel-decoder compatibility evidence.
- No archive deletion, upload, submission, report transmission, or external
  resource use was performed.
