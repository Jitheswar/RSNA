# Choose the first bounded campaign to establish useful image learning

Parent: ../map.md
Type: grilling
Labels: wayfinder:grilling
Mode: HITL
Status: resolved
Assignee: root
Blocked by: 02, 03, 04, 05

## Question

Which ordered set of diagnostics and controlled experiments should first establish useful image learning, and when should each continue, stop, or change direction? Select supervision, initialization, anatomical coverage, training duration, and controls using the research, validation policy, and available resources. Include a queue-unavailable path, an explicit GPU proposal budget, resource estimates, and evidence-driven expansion criteria. The decision must discriminate a broken/undertrained pipeline from a genuinely unhelpful representation or label source before costly ensembles.

Before selecting an external resource for execution, recover the applicable current official rules and document the proposed checkpoint's exact revision, source, rights, and eligibility. The research shortlist is not an admission decision. If those facts cannot be established, retain the uncertainty and choose an admissible alternative or create a dedicated prerequisite investigation.

## Comments

### First campaign proposal presented to the owner, 2026-09-25

1. Diagnose data/label alignment, valid inputs, gradients and tiny training-only memorization within the existing reference image scope. Then evaluate frozen ResNet-18 features with anatomical pooling; consider limited fine-tuning only after the diagnostic and measured resource checks. DINOv2 is a second representation comparator if the first route is unhelpful, not a simultaneous architecture sweep. External-weight use remains conditional on source/eligibility admission; the existing model supports initial CPU diagnostics while that review proceeds.
2. Audit the already-generated learned report outputs for provenance and training eligibility, and evaluate a prospective target-level admission policy before spending time on another extractor. Keep deterministic and reference-only comparators. If those existing outputs fail admission or provide insufficient trustworthy coverage, evaluate one local multilingual alternative; do not silently admit uncertain labels or authorize more remote report processing.
3. Propose a first-campaign planning ceiling of 20 aggregate GPU-hours, including smoke runs, feature extraction, training and retries, with one GPU request at a time. Each initial full job remains at most six hours, one GPU, 16 CPUs and 64 GB host RAM; a measured smoke precedes full requests. No GPU availability is promised and exact jobs still require the existing concrete resource approval. CPU diagnostics and bounded data preparation continue while queued; new pretrained feature extraction on CPU is conditional on measured throughput, not assumed complete within six hours.

The owner subsequently accepted all three choices in visible chat; the resolution below records the route and operational decision rules. Full-corpus image preparation requires a storage ledger and bounded checksum-covered admission; the 4,407-study header inventory cannot substitute for that work.

## Answer

Resolved through the owner's **“yes to all”** on 2026-09-25. Select the staged ResNet-18 route, audit existing learned report outputs before another extraction run, and cap the first campaign at 20 total allocated GPU-hours with one GPU request at a time. This is a planning decision, not approval of an unprepared job or weight download.

### Ordered campaign and stop rules

| Stage | Work to prepare for execution | Continue or stop |
| --- | --- | --- |
| Input and resource admission | Verify corrected v2 folds and exact selected input/cache lineage; establish the project storage ledger; recover current resource rules and review `ResNet18_Weights.IMAGENET1K_V1` source, full digest and rights before use. | Missing integrity or eligibility blocks the dependent path. Existing-model CPU diagnostics may continue. A failed resource review does not silently admit DINOv2 or different weights. |
| Learning diagnosis | Use only a tiny training-side subset to trace study/label order, pixel content, masks, loss, gradients, parameter updates and memorization. Compare the current short training budget with bounded longer training curves. | Broken alignment, inert gradients or failure to fit the training subset leads to diagnosis/repair, not a larger model or ensemble. Training-only success establishes optimization competence only. Freeze the bounded run's numerical diagnostic thresholds in its execution request. |
| First representation | Use frozen ResNet-18 slice features, preserve plane/series identity with missingness masks, and fit small fold-specific heads on reference labels. Keep the full knee field, a fixed geometry-aware slice policy and a simple pooling comparator. New 224-pixel preprocessing must be derived from admitted images, not mistaken for the existing 128-pixel cache. | Finish the declared three-fold comparison and report pooled/within-fold evidence against the existing model and prevalence controls. Uninformative features trigger a bounded alternative representation comparison or a return to input diagnosis, not unrestricted hyperparameter searching. |
| Report supervision | Audit existing learned-output provenance and evaluate a prospective fold-isolated target-level admission policy. Keep unknown/uncertain/failed calls masked and reference labels authoritative. Compare reference-only and admitted weak-supervision training with matched reference exposure. | Reject ambiguous lineage. Admit no target merely on a model's asserted confidence or overall accuracy. If reliable added coverage is inadequate, propose one local multilingual extractor with its own exact resource review and measured cost. No fresh remote report transfer is included. |
| Image-scope expansion | Prepare additional eligible report-only images in deterministic, restartable batches within the storage ledger and retained source/fold identity. Track per-target admitted supervision as well as study counts. | Expand only for the selected label/model comparison, within measured resource estimates. Never call 158 prepared studies the whole corpus. Stop batch growth when the soft storage limit or remaining time would be exceeded. |
| Controlled model expansion | Consider limited encoder fine-tuning after optimization competence, label integrity and resource fit are established. Use DINOv2 as the next representation comparator if the first route is unhelpful; adjacent-slice 2.5D remains a subsequent controlled change. | A single-fold positive result is insufficient for promotion. Complete comparable OOF evidence under the agreed validation policy. No full 3D/large-model/self-supervised-pretraining sweep is part of this first campaign. |

### Budget and queue policy

- Count allocation time for every GPU attempt, including failed jobs, feature extraction, report extraction, smoke tests and retries, against the 20 GPU-hour first-campaign cap. Record queue wait separately; it does not grant more allocated hours. One request may be active or pending at a time.
- Prepare a smoke request of at most 30 minutes before a new full training shape. The proposed full-job ceiling is one GPU, 16 CPUs, 64 GB and six hours, with actual memory/isolation checked inside the allocation. These are ceilings, not a claim of fit or an exact authorized request.
- Size any three-fold GPU comparison so all folds plus required preflight fit inside the remaining cap. If the estimates do not fit, reduce the planned scope before freezing the comparison or keep the frozen-feature/CPU-head route. Do not interpret an unfinished set of folds as equivalent OOF evidence.
- Use CPU diagnostics, the storage ledger, eligibility review, label audits and bounded data preparation while queued. CPU feature extraction requires a measured throughput estimate and restartable chunks within the existing six-hour ceiling. If no practical feature route fits, record that the pretrained comparison did not run and preserve the existing candidate.
- Pause GPU expansion at the cap or the later chosen modeling cutoff. A larger campaign requires a new concrete budget decision; no automatic additional 20 hours or new spending is implied.

Implementation must bind exact seeds, membership, sampling, optimizer/checkpoint policy and numerical run limits before execution. Those details instantiate this route and must not be chosen retrospectively from held-out outcomes. Resource facts that remain inaccessible are explicit execution prerequisites with the existing-model fallback, not claims that the external resource is cleared.
