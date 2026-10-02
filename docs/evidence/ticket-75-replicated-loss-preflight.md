# Ticket 75: CPU preflight for the TPU pilot

The eight-rank masked-loss seam is implemented independently of all frozen live
CPU training and analysis sources. It preserves the original global four-study
mean under a subsequent eight-rank mean gradient reduction:

```text
local replica loss = 8 × supervised BCE numerator / global supervised-entry count
mean reduced gradient = gradient(original combined four-study masked BCE mean)
```

CPU admission rejects empty, nonfinite, out-of-range or incorrectly shaped
supervision. Absent NaN targets/logits are replaced before BCE. Ranks0–3 retain
the two reference/two report draws, while ranks4–7 contribute zero supervision.
The per-replica loss uses tensor operations without reading a device scalar or
selecting variable-length indices. It requires FP32 and the fixed eight-rank
contract. It does not implement or claim actual distributed execution.

The new suite passes19 tests. Six existing whole-backbone guards also passed,
for25 relevant checks. The tests establish:

- Unequal-mask loss and logits-gradient equivalence to the original masked mean;
  incorrect averaging of local means differs on the same fixture.
- Four empty padding ranks add zero loss and gradient, including absent NaNs.
- Original whole-backbone fixture gradients and an ordinary Adam update agree
  within rtol1e-4/atol1e-6, with every normalization buffer unchanged.
- Sixteen synthetic updates with changing global supervision counts preserve the original encoder1e-5/head1e-4 Adam
  rates, default betas/epsilon and model/moment/step state within those bounds.
- Fresh checkpoint reload at step8 reproduces the continuous replicated CPU
  path exactly at step16, including every model tensor and Adam moment/step.
- Unsupported ranks, world size, FP32/device/shape contracts and invalid CPU
  supervision are rejected.

The smaller whole-backbone fixture uses32-pixel synthetic inputs. This does not
prove real224-pixel original-MRI parity, Torch2.8 host equivalence, TPU precision,
collective behavior, compile stability, throughput, memory or predictive gain.
The constant scheduler, full RNG/export format and actual source-input state
checks remain requirements of the separately frozen real-MRI pilot.
Project type checks pass across89 source files.

The frozen original cohort confirms fold0 has38 reference and2731 report-only
studies. Its existing sampler and original seeds select32+32 distinct studies
in the first16 batches. This is a concrete attachment constraint: raw FP32 pixels
plus three full reference checkpoints exceed the2GB input envelope. The real
pilot must retain all CPU states locally and attach only initial/step3 reference
states; unchanged raw MRI plus two conservative state bounds is1,914,309,506
bytes before code/metadata. Actual complete bytes must still pass the hard bound.
No compression assumption, scientific budget reduction or checkpoint deletion
is used to make the proposal fit.

No pilot resource admission, CPU job, attachment publication, accelerator
reservation or scored attempt has occurred. Existing8113/8119 training and the
T74 native-completion waiter remain live. The next step is a complete separately
frozen real-MRI preparation and combined route admission, preserving both live
routes and all delivery/review reserves. First place is not established.

Latest authoritative poll confirms8113/8119 RUNNING, optimizer steps3709/4419
respectively, and the existing T74 waiter process930775 remains live.
