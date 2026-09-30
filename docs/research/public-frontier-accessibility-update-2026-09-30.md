# Public frontier accessibility update — 2026-09-30

## Finding

**No newly verified, complete runnable public route above the Yamadan D4 author's 0.943 was established.** The strongest checked public notebooks still require inaccessible private assets. There is one concrete new inference hypothesis: an explicitly public Apache-2.0 DINOv2-Base meniscus specialist, now included in Roman's version 24. Its observed linked submission has no published score. This is a newly accessible asset, not a demonstrated leaderboard gain.

This audit uses current authenticated Kaggle metadata/source APIs and primary public notebook pages. The [earlier frontier audit](public-frontier-options-2026-09-30.md) establishes Yamadan's 0.943 as author submission `56696090`, script `353578144`, version 1; version 2 changes attribution markdown only. Neither author's score is our pending submission `56705536`.

No weights were downloaded; no model ran; no launch, push, scoring, publication, payment, rules acceptance or access workaround occurred. The shared 72,000 GPU-second ledger and Ticket 39/40 restrictions remain unchanged. The completed local ConvNeXt CPU branch remains inconclusive and unpromoted; no further representation/training sweep is proposed.

## Current score provenance and access

The fresh competition `scoreDescending` query returned 40 public notebooks. Its metadata has no numeric score field; sorting position alone is not score evidence. Primary `GetKernelViewModel` responses and rendered notebook pages supply the following associations:

| Route | Exact observed score provenance | Current access finding |
| --- | --- | --- |
| [PJ D4 blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend?scriptVersionId=353440703) | Author **0.946**, submission `56623920`, script `353440703`, v3 | Public Apache-2.0 source; three private dataset placeholders; required eff6/ens14 assets still HTTP 403 |
| [Aastik S75 W50](https://www.kaggle.com/code/aastikrajan15/knee-s75-w50?scriptVersionId=353382736) | Author **0.945**, submission `56617152`, script `353382736`, v1 | Public Apache-2.0 source; private `knee-ckpt` still HTTP 403 |
| [Evgen Versia5](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5?scriptVersionId=353111736) | Author **0.943**, submission `56589234`, script `353111736`, v18 | Public Apache-2.0 source/page; does not establish a stronger score |
| [Roman scored v23](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5?scriptVersionId=353896859) | Best author **0.943**, script `353896859`, v23, explicitly named in current best-score metadata | Previously retrieved v23 source remains available locally; score is not v24's score |
| [Roman current v24](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5?scriptVersionId=354167350) | Linked submission `56710105`, script `354167350`, v24; **no score field in the observed submission record** | All sixteen dataset attachments identify public assets; new specialist package has complete declared inference inventory |

PJ D4 blend v3, PJ D4 lite v1 and Aastik W50 v1 source bytes are unchanged from the preceding audit. Lite remains near the top of the sort, but no new numeric score was attributed to it in this update. Its inaccessible dependencies already prevent a complete independent execution.

The authenticated [top-five snapshot](../../artifacts/private/leaderboard-snapshots/2026-09-30-frontier-top5.json) records rhoskeri 0.961, Wassim Dobbi 0.960, Scott Willis 0.958, TomKroumov 0.958 and 🦵YOKKOISHO🦵 0.958. The primary [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) rendered the same top five during this check. **No exact leader submission/script version and complete publicly licensed inference bundle were established.** Team score/name is not permission or notebook provenance. Competition-kernel queries for `rhoskeri`, `wassimdobbi`, `scottgwillis` and `tomkroumov` returned empty arrays; only Scott's username was independently matched to his primary profile. Other handle spellings and the fifth team's membership were not resolved. These bounded queries do not prove the leaders have never published code.

## Why the higher-scoring public sources remain incomplete

Fresh authenticated `GetDatasetMetadata` calls returned HTTP **403 Forbidden** for:

- [pjmathematician/rsna-knee-eff6-assets](https://www.kaggle.com/datasets/pjmathematician/rsna-knee-eff6-assets)
- [pjmathematician/rsna-knee-ens14-assets](https://www.kaggle.com/datasets/pjmathematician/rsna-knee-ens14-assets)
- [aastikrajan15/knee-ckpt](https://www.kaggle.com/datasets/aastikrajan15/knee-ckpt)

PJ's exact source requires `arms/*/weights/*.pt`, expecting eight eff6 checkpoints and 43 ens14 checkpoints, plus `ens14/src/infer_kernel.py`. Blend uses `S_dist`, `S_coat`, `S_cnxl`; lite uses `S_dist`. Its default extra-arm weight is 0.45. Three attached private datasets remain unnamed in the API, so the third cannot be assigned a license or assumed redundant. The source's fallbacks cannot reproduce the author's full-arm score when private inputs are absent.

Aastik's source embeds a separate runner requiring private `model.py`/`KneeNet`, checkpoint-contained model arguments, and ten selected ConvNeXt Tiny 320×24 checkpoint basenames spanning five folds. A `Knee3DNet` branch is also supported. Accessible inference wrappers and model names do not supply the private architecture definition, checkpoints, training labels or their redistribution rights. The public common ensemble can run separately; it does not inherit the 0.945 attribution.

## Exact new public specialist

Authenticated metadata for [renta0426/rsna-knee-public0033-meniscus-bag-v1](https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1) reports **`isPrivate=false`, Apache-2.0**. Roman's exact v24 attachment identifies dataset-version source ID **`19356288`**; a future preflight must pin that attachment and its file hashes rather than silently follow the dataset's latest version. Its current card explicitly licenses the author's checkpoint/runtime artifacts and permits private/public competition notebooks and redistribution under those terms. It does not relicense competition images, reports, labels or third-party assets.

The card and preserved run contract describe DINOv2-Base, 4,349 report-supervised studies, exclusion of the 58 reference-labelled studies, 30 epochs, seed 2027, and adaptation of the last six transformer blocks plus final LayerNorm. The manifest explicitly says `fold_safe_oof=false`. This is full-fit provenance, not OOF validation or proof of a blend gain.

| Public inference file | Listed bytes | Audit status |
| --- | ---: | --- |
| `checkpoint_epoch30.safetensors` | 347,965,736 | Listed; **not downloaded**; manifest SHA-256 `e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7` |
| `checkpoint_epoch30.safetensors.json` | 564 | Listed and hash declared |
| `dinov2_config.json` | 548 | Listed and hash declared |
| `fullfit0033_meniscus_bag.yaml` | 5,854 | Listed and hash declared |
| `public0033_meniscus10_residual.yaml` | 8,131 | Listed and hash declared |
| `public0033_runtime.py` | 42,887 | Retrieved; manifest hash matches `9541f82a993d7dc2942ca2a5112e110471285f8d80701846079a3dc708761683` |
| `run_contract.json` | 6,607 | Retrieved; manifest hash matches `107ef5133238dc514eb990bf75e05a0c5d8262873c95158957565bdc7278e11f` |
| `task_complete.json` | 979 | Listed and hash declared |
| `bundle_manifest.json` | 9,163 | Retrieved; declares all inputs and hashes |

The runtime contains its own model/head definition and constructs `Dinov2Model` from the local config, then strictly loads the local safetensors checkpoint. It needs NumPy, PyTorch, Transformers and safetensors, rather than a private repository or an online pretrained fetch. It requires **exactly two Tesla T4 devices**, FP32, 32-study batches, 233 state tensors, and ten overlapping windows. It consumes the existing DINO uint8 cache `(N,6,12,336,336)` and float32 missing-slot mask `(N,6)`; raw outputs are Medial/Lateral Meniscus probabilities. All listed runtime files are mandatory and hash checked. The absence of Kaggle's control-plane `dataset-metadata.json` is expressly permitted.

The manifest's historical `dataset.private=true`, upload-authorization flags and `complete_local_not_uploaded` string conflict with the package's later publication state. They describe the original preparation record; current authenticated public metadata and the current explicit license are the evidence for present accessibility. They are not a reason to bypass runtime integrity checks. This audit has not mounted the checkpoint or tested model execution.

### Roman v24 is not a single-variable copy of scored v23

Exact source SHA-256 is `35715915648ea652740f3833b130e3e9451db897905d596ff41cd883f03c571a`, 275,454 bytes. The preserved code-cell diff shows v24 removes v23's default **0.70** CoAt/Raptor routing overlay and restores the **0.60** default before adding the specialist. Consequently its assertion that eleven targets remain unchanged is relative to its reconstructed v24 anchor, **not the scored v23 output**.

For Medial Meniscus, v24 MAIN replaces half of the existing 20% Transformer vote: `10% Transformer + 80% CoAt/Raptor + 10% specialist`, followed by rank conversion. It writes 5%, 10% and 15% variants. Exceptions in specialist/runtime/coverage checks silently restore the anchor via `public0033_v5_fallback`. The observed successful 360.8-second visible-data run does not prove specialist use, hidden-test latency or scored gain. Any future experiment must require specialist completion and zero fallback rather than accept a successful notebook status.

## Complete attachment and rights inventory

`accessibility-findings.json` preserves every exact SDK attachment slug and browser source reference/version ID. The common D4 graph has fourteen public datasets, two notebook-output sources, DINOv2-Small model version 1 and the competition. Fresh metadata refreshed all fifteen non-specialist dataset attachments from Roman v24:

- CC0: three Dread Raptor packages; Mattia fold weights; three Mattia CoAt packages; Pilkwang labels and weights.
- CC-BY-NC-SA-4.0: Marwan RadImageNet weights and Antoine E9/E11 heads.
- `other`: Prvsiyan V52 heads and Roman's additionally attached Tonylica DINO/RadImageNet/Raptor assets.
- `unknown`: Mattia's OpenCV wheel package.
- Apache-2.0: the new public0033 package, checked separately.

All sixteen live dataset metadata records report public visibility. Roman v24 also attaches `sofiaanjenje/rsna-knee-e11-train`, `sofiaanjenje/rsna-knee-e13-train`, and Meta DINOv2 Base/Small version 1. The preceding audit contains the notebook-output inventory and underlying model/license evidence. Apache licensing of notebook code does not override CC-BY-NC-SA, unspecified package terms or third-party conditions. The extra attached Tonylica package is not evidence that every attached file is consumed; its exact declared attachment is preserved rather than silently omitted.

## Smallest grounded next hypothesis

Keep the frozen Yamadan D4 graph as the anchor and investigate **one 10% Medial Meniscus specialist overlay**, replacing half its Transformer vote, with all other target columns preserved. Do not adopt Roman's 0.70→0.60 routing change or its three blend choices as a search campaign. The asset is public and its inference dependencies are declared; its predictive complementarity and runtime remain unestablished.

The next useful work is a concrete **CPU-only asset/source preflight proposal**: pin the specialist dataset version, verify its nine mounted files against manifest hashes, statically check the two inserted cache/inference cells against the anchor, check dependency availability, and make specialist failure fatal. No weights need be downloaded into this workspace for the research audit. A later model/inference/scoring experiment would require terminal reconciliation of pending `56705536`, an exact resource reservation, unchanged zero-spend rules, and the remaining science/delivery gates. No such launch is authorized by this research result.

## Evidence and transfer limitation

Raw evidence/hashes are under `artifacts/private/external-notebook-audit/frontier-2026-09-30/accessibility-update/`: four authenticated kernel responses, fresh private-access errors, fifteen refreshed dataset metadata records, specialist metadata/file inventory/manifest/runtime/run contract, four primary notebook view-model responses and rendered page text, the sorted top-40 query, owner queries, exact Roman source and code-cell diff. `accessibility-findings.json` records source identities and exact score/submission associations; `artifact-manifest.json` fingerprints the retained evidence.

Retained selected notebook/source/dataset response payloads total **3,398,804 bytes**, plus small query/error/text records. **The 5 MB total-transfer limit cannot be certified:** the browser also received a full public leaderboard response which was rejected by its evidence-capture cap after receipt; its decoded size was not recorded. Broad leaderboard fetching stopped. This limitation does not affect the saved notebook score associations or authenticated asset access tests, but retained-file size must not be presented as complete HTTP transfer accounting.
