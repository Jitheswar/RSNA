# Ticket 66: BiomedCLIP MRI parity and cost admission

The isolated runtime is acquired and offline imports pass. The fixed
eight-study pilot is frozen; its persistent scheduling controller is confirmed
live waiting for a free CPU slot. No BiomedCLIP model forward, MRI parity,
throughput or predictive result is established yet.

## Actual runtime evidence

Six publisher-pinned wheels total **14,720,706 bytes**: OpenCLIP **2.23.0**,
Transformers **4.35.2**, tokenizers **0.15.2**, ftfy **6.3.1**, sentencepiece
**0.2.1**, protobuf **4.25.8**. Acquisition matches complete PyPI SHA-256/size.
The first import exposes an absent transitive `wcwidth` dependency; that failure
is preserved. One distinct **34,166-byte wcwidth 0.2.13** addition resolves it,
without retrying the original acquisition. Offline imports then pass and the
actual installed `model.py`/`timm_model.py` bytes match the pinned official
OpenCLIP source captured in [the research](../research/medical-image-encoder-options-2026-10-02.md).

The new environment references the already admitted CPU Torch **2.7.1**,
TorchVision **0.22.1**, timm **1.0.22**, Pillow **11.2.1** and NumPy **2.2.6**.
Existing environments are preserved; no text tower or downloaded tokenizer
model is constructed. Runtime storage reserves **2 GB**, with retained Ticket
64 allowances, projected **1,838,637,055,555 bytes**, below the 2 TB budget.

## Frozen pilot and scheduling

Private root: `artifacts/private/ticket66-biomedclip-mri-pilot-v1/`.

| Admission | SHA-256 |
| --- | --- |
| Runtime plan | `173e18e79834da8dd9219995fe94f561f2f4ed0bf5987f10e714b144ef03961c` |
| MRI pilot | `36a7524de3929e18d229102ed292c7a10e6bd49bcdeaa510eddce9ac1e63ff90` |
| Scheduling controller | `f70715bc6bf524324ef11c1e817ce784f50afd78e8ef2833c6f405641fde817f` |

Exactly eight report-only studies, ascending SHA-256(UID), **344 valid slices**;
no target values participate. Original archive entries are present. Preserve
whole-field 224-pixel grayscale MRI, masks, descriptors and selected indices;
repeat RGB and use pinned CLIP normalization. FP32 microbatch one, learned
512-D projected output, explicit L2 normalization. All **14,643** prediction-
bearing source/runtime/input files are fingerprinted. Strict vision loads and
official reference path must match within 1e-6. A separate process repeats all
outputs from original DICOMs and verifies finite nonconstant vectors/padding.

One **960-second /16-CPU /64-GiB** offline nonexclusive allocation includes
the pilot and independent replay under one **900-second** pipeline watchdog.
No retry, GPU/TPU use, spend, fitting, AUC, public submission or promotion.
Model-only and native-read-inclusive extrapolations remain distinct measured
costs; full extraction needs a separate decision after actual evidence.

`rsna-ticket66-cpu-handoff-v1.service` is live, PID **225672**, phase
`wait_for_cpu_slot`. It temporarily holds only the fit scheduler; both
**8035 /8036** continue. After an exact successful stage frees a slot, it admits
the sole pilot and immediately recreates the exact fit coordinator. At most
two owner CPU jobs include pending requests. A consumed handoff cannot restart.
Any preparation/launch exception restores scheduling. One-hour queue allowance
and pilot allocation reserve **4,560 calendar seconds**, before science freeze.
Eleven selection/handoff tests pass; all live source snapshots and launch hashes
remain exact. The final Ticket 64 MRI verifier remains active.

The preceding full stage **8032** completed `0:0`, **6,685 CPU-node seconds**,
exactly **1,000 updates**, no intermediate prediction. Job **8036** consumes
checkpoint SHA `39ee8050f3210ec861c765f3e9ac25d72129f1ae06b3cf294451bf789683d298`,
and its actual training log begins from step 1,000 then advances to 1,001.
Neither this continuation nor runtime readiness establishes improved AUC,
complementarity to the scored 0.936 anchor or first place.
