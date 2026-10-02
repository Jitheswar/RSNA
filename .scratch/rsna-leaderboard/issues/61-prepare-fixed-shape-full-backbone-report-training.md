# 61: Prepare fixed-shape full-backbone report training

Type: task
Status: claimed
Labels: ready-for-agent
Parent: [RSNA campaign](../spec.md)

## Purpose

Prepare a genuinely different learning regime from the inconclusive frozen-head
and layer4-only experiments: adapt the entire admitted compact MRI encoder on
fold-isolated report targets. Ticket 60 must establish actual TPU capacity
before any dependent accelerator launch. Code preparation can proceed while
that exact allocation request is queued.

## First implementation

Register a separate versioned full-backbone ResNet-18 model. Use fixed-shape
packed slices and a fixed encoder chunk size rather than data-dependent valid
indices. Preserve the admitted grayscale/ImageNet normalization, anatomical
pooling and checkpoint tensor names. All encoder parameters receive gradients;
normalization running statistics stay fixed so padded examples cannot influence
valid slices. Padded pixels must be excluded even when they contain NaN/Inf.
Use a fixed-shape masked BCE with zero gradient from absent targets and correct
normalization over supervised entries; admit nonempty labels on the CPU before
device transfer. No host scalar reads or variable-length target selection in
the device loss.

## Requirements

- [x] Implement versioned reconstruction and strict checkpoint compatibility.
- [x] Verify frozen/native logit parity before updates, early/late encoder
  gradients and changes, padding isolation, fixed normalization buffers, and
  exact CPU checkpoint replay after an update.
- [x] Verify masked BCE values/gradients against the established selected-entry
  implementation, including absent NaN targets and empty masks.
- [ ] Once actual TPU exists, freeze a separate real-MRI whole-backbone pilot
  and complete cost/checkpoint-delivery plan; no launch from code readiness alone.
- [ ] Complete the prespecified three-fold comparison and actual offline
  delivery verification before promotion; no gain is claimed from code tests.
- [x] Independently measure the CPU alternative under a separate 900-second,
  16-CPU/64-GiB, nonexclusive, one-attempt plan. Reuse the four label-blind
  fold-0 training reports and exact MRI packing previously admitted by Ticket
  56. One warmup/three timed whole-encoder updates, fixed normalization,
  retained diagnostic checkpoint and fresh-process exact CPU replay. No full
  training campaign or candidate promotion follows from cost alone.

## Comments

October 2: Initial code preparation passed 39 focused lifecycle/training tests
and typing for 83 source files. A separate bounded CPU real-MRI throughput
pilot is now prepared under standing owner authorization; it does not depend
on TPU allocation. No new MRI extraction, weight acquisition, accelerator
request, full fitting campaign or candidate selection.
Existing native frozen/layer4 identities and their results remain intact.

October 2: CPU **8016** completed `0:0`, **127 allocated node seconds**.
Four already admitted reports, 232 valid/512 padded slices, all encoder stages
update, native initial logits and fresh-process diagnostic checkpoint replay
are exact. Median whole-backbone update **5.955313 seconds**, peak RSS **17.88
GB**. 2× training-only scenario: 1,500 steps/fold **4.963 h**, 7,500 **24.814
h**; complete three-fold/final-fit/I/O/checkpoint/delivery admission is still
required. [Evidence](../../../docs/evidence/ticket-61-full-backbone-cpu-cost.md).
No generalization gain or complete training route is inferred from the pilot.
