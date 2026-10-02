# Ticket 75: Real-study selection and input contracts

The real fold0 pilot selection is pinned in private selection receipt
`b08d52294efac12435fb49717ffac8b6f51bfb53c22fb1dc7658fb834e2ea8ce`.
It preserves the complete original38-reference/2731-report pool order and the
first16 deterministic batches with original seeds. Exactly64 distinct training
studies are selected; every original held-out study is excluded. The complete
original T73 fitting input/source admission passed, and the selected receipt's
seven frozen metadata-source hashes were independently rechecked.

The receipt binds each selected pixel-cache file and its original source-identity
hash, corrected fold checksum, original full-pool fingerprints and audited
supervision admission. No MRI pixels, selected target arrays or initial model
state have been exported. This is metadata admission, not native MRI replay or
a completed CPU reference request. No resource or accelerator has been allocated.

The new independent input-contract module rejects selected-pool resampling,
changed source-pool sizes, duplicate identities and reference/report/held-out
overlap. Its immutable selection preserves the two reference and two report draws
in every batch. Reversing a pool changes its digest and selection rather than
silently treating pool order as irrelevant.

Before any actual export, every real study must have the original16-series,
8-slice, single-channel224-pixel FP32 geometry; original boolean slice/series
masks; ten FP32 descriptors; and twelve FP32 targets with boolean uncertainty
masks. The validator rejects resolution/dtype conversion, malformed descriptors,
nonfinite pixels, empty real images or supervision, and supervised NaN targets.
Absent NaN targets remain absent; it performs no imputation or resizing.

The attachment bound counts all raw FP32 geometry plus initial/step3 states and
code/metadata, and rejects a complete total above2GB. Later CPU reference states
remain on the cluster for the independent checks; no study/update or replay gate
is removed to fit the bound.

The **17 new metadata/input tests pass**. Project type checks pass across
**90 source files**. All live T73/T74 source files and plans remain unchanged.
This does not prove real source-pixel equivalence, Torch2.8/TPU portability,
throughput, collective behavior or predictive gain. The CPU reference worker and
combined resource ownership/admission remain the next required work; first place
and an improved submission are unverified. No daily scoring attempt was made.
