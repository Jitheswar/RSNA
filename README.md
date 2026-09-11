# RSNA knee mechanics baseline

This repository contains the ticket 2 tracer bullet: a small, image-only
inference core that turns in-memory synthetic pydicom studies into the exact
twelve-target competition schema. It is intentionally not a real-DICOM
compatibility or training-preflight claim.

## Reproducible CPU environment

The project is pinned to Python 3.11.16 and a CPU PyTorch wheel. `mise` keeps
the interpreter in the user account; the project virtual environment must be
created from that interpreter:

```bash
mise install
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
```

The lock contains the verified runtime and test dependencies. The source tree
can be used without installation by setting `PYTHONPATH=src`.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m mypy src/rsna_knee
```

## Synthetic CPU demonstration

Significant computation belongs in Slurm, not on `KLEF-MASTER`. After the
user-space environment is prepared, submit the bounded adapter:

```bash
sbatch --wait slurm/run_cpu_inference.sbatch
```

The adapter requests one `cluster1-cpu` node, four CPUs, 8 GB, and 30 minutes;
it has no exclusive allocation. It writes the ignored demonstration output
and JSON evidence below `artifacts/`. The CLI refuses to run outside Slurm
unless `--allow-local-smoke` is explicitly supplied for a tiny development
check.

The public interfaces are:

- `decode_dataset` and `read_dicom_slice`: map a pydicom dataset or file to a
  decoder-independent core record. Synthetic coverage is intentionally narrow;
  real-corpus compatibility remains a later gate.
- `order_slices`: validates geometry and orders by spatial position, never by
  filename or instance number.
- `prepare_series` and `pack_study`: apply fixed 128×128 preprocessing,
  retain all accepted slices, sample eight only at packing time, and expose
  slice/series masks.
- `HierarchicalStudyModel`: shared 2D encoding, masked slice-to-series and
  series-to-study mean pooling, and a twelve-logit head.
- `InferenceEngine` and `write_submission`: MRI-only inference, explicit
  all-invalid study fallback, finite bounded scores, exact UID order, and the
  thirteen-column output contract.
