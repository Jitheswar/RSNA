# Ticket 66: BiomedCLIP MRI parity and cost admission

Sole offline CPU pilot **8037** completed `0:0`, **217 CPU-node seconds**.
Strict official vision loading and independent original-DICOM replay pass
for all eight studies /344 valid slices with maximum output error **0**.
This establishes representation parity and measured cost; predictive gain
and leaderboard improvement remain untested.

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

The consumed `rsna-ticket66-cpu-handoff-v1.service` completed successfully.
After **8035** reached its successful 1,000-update boundary, it admitted
**8037** and immediately restored the exact fitting coordinator at **08:56:30
UTC**. Neither live fit was restarted; **8039** continues fold 1 at stage 1
alongside fold-0 stage-1 **8036**. The handoff reconciled the pilot terminal
at **09:00:11 UTC**. Eleven selection/handoff guards pass; preserve the
completed request, frozen runtime, sources and receipts.

## Actual measured outputs and cost

Strict state loading covers **151 FP32 vision tensors**. Official factory
reference and repeated inference have maximum error **0**; fresh-process
original-DICOM reconstruction and model inference independently reproduce
all **344** outputs exactly. Finite unit 512-D vectors vary with image input.
Pilot elapsed **123.490 s**, independent replay **90.978 s**, combined actual
allocation **217 s**; peak pilot RSS **3,109,085,184 bytes**. Vision loading
**3.133 s**, median valid-slice inference **0.038754 s**, maximum **0.069421 s**.
Native read/packing **52.492 s**, maximum single study **17.789 s**.

A conservative two-times-maximum model-only forecast for **187,272** slices
is **26,007.616 s /7.224 h**. Repeating native decoding would forecast
**176,428.721 s /49.008 h**. Ticket 62 already independently verified every
whole-field pixel and all cache reads, permitting separately admitted extraction
from those exact inputs without repeating native decoding. Neither forecast
is an actual full-cohort extraction measurement.

| Actual evidence | SHA-256 |
| --- | --- |
| Pilot result | `c9dbfd784df6c0eec731767930f6d165743c3e3605aacb9be0beb7f4fcec86ee` |
| Independent native replay | `573bb80c773eadb62829784f7f5a88e28b71b9e98ec675e74bda0297dd242f66` |
| Successful retained terminal | `be031815b4c3408cb469b7a2e25436c1fe7c454c62b18bde2d42ded96cb43a81` |

The preceding full stage **8032** completed `0:0`, **6,685 CPU-node seconds**,
exactly **1,000 updates**, no intermediate prediction. Job **8036** consumes
checkpoint SHA `39ee8050f3210ec861c765f3e9ac25d72129f1ae06b3cf294451bf789683d298`,
and its actual training log begins from step 1,000 then advances to 1,001.
Neither this continuation nor runtime readiness establishes improved AUC,
complementarity to the scored 0.936 anchor or first place.
