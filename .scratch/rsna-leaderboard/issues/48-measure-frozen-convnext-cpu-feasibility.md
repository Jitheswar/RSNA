# 48: Measure frozen ConvNeXt CPU feasibility

Type: task
Status: resolved
Labels: ready-for-agent
Parent: [RSNA leaderboard campaign](../spec.md)

## Purpose

Measure whether one independent frozen ConvNeXt Tiny representation can be extracted from the existing admitted image cohort within the zero-spend CPU campaign and calendar. This ticket authorizes only a 32-study throughput and integrity pilot; any cohort extraction and matched head experiment needs a separately frozen plan after the pilot.

## Requirements

- [x] Admit exact public pretrained checkpoint, isolated CPU runtime, source snapshot, immutable image identities and resource plan under the owner's existing campaign authorization.
- [x] Run one CPU Slurm allocation, one node, sixteen CPUs, 64 GiB, at most ten minutes, no GPU and no network during execution. No training or reference-score selection.
- [x] Preserve features, masks, descriptors, source hashes, elapsed time, memory and terminal allocation evidence; reject failures or nonfinite features.
- [x] Project full-cohort extraction with a twofold time allowance against at most 24 CPU node hours, six hours per job, at most two concurrent jobs, and completion by October 3.
- [x] Record proceed/stop disposition without candidate promotion or a first-place claim.

## Context

The exact upstream model is `timm/convnext_tiny.fb_in22k_ft_in1k` revision `bc48a87f119bae1b06418f5340bfa97ef75d4609`, Apache-2.0 model card. Safetensors: 114,374,272 bytes, SHA-256 `77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a`. The model's reported license does not grant additional underlying dataset rights. Private experiment use follows the existing campaign authorization; final release remains separately gated.

Use the existing deterministic 224-pixel whole-field image preparation, eight slices per series, masks and anatomical descriptors. Expand grayscale to RGB and apply the published ImageNet mean/std. Keep all backbone parameters frozen and return 768-dimensional pre-classifier features. Pilot studies are report-only and chosen without label outcomes. Ticket 39 and Ticket 40 remain governed by their existing plans.

See [primary feasibility research](../../../docs/research/public-complement-training-feasibility-2026-09-30.md).

## Answer

Exact CPU pilot job **7934** completed with `ExitCode=0:0`, **36 allocated wall seconds**, zero GPUs and zero spend. All **32 feature files / 1,464 valid slices** were independently hash checked: finite float32 `(16, 8, 768)` arrays with zero padded features, exact existing slice/series masks and anatomical descriptors. The frozen encoder's strict full-checkpoint load matched 28,589,128 parameters; classifier removal produced 768-dimensional features without training.

Measured code elapsed **34.5901 s**, encoding **16.1340 s**, peak process RSS **1,388,871,680 bytes**. Full 4,228-study / 187,272-slice projection with the fixed twofold allowance is **5,231.46 s / 1.4532 node hours**. This estimate excludes gold archive decoding and queue delays. It passes the 24-node-hour gate; a separate bounded extraction request is justified. No cohort extraction, head training or candidate selection was authorized by the pilot itself.

Plan SHA-256 `503aa1f1315c68855c0e611e82a4888f7f3e4a0ed59b2119a6ceaa7af8e87391`; exact checkpoint admitted under the owner's existing zero-spend authorization. Source/runtime/image identities and checkpoint admission are retained under `artifacts/private/ticket48-convnext-cpu/`. Live controller and terminal evidence are in `controller-7934/`; immutable results are in `job-7934/`; independent validation is `independent-output-validation.json`.

## Comments

2026-09-30: One attempt submitted and completed. The saved actual allocation has `TRES=cpu=16,node=1,billing=16` and `OverSubscribe=OK`. The pilot script checked a newer Slurm `AllocTRES` field that this cluster omits; independent terminal inspection confirms no GPU allocation. Future extraction must inspect this cluster's actual `TRES` field as well. No change to Ticket 39, no Ticket 40 execution and no claimed prediction improvement.
