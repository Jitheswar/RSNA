"""Pin real training identities/cache bindings without loading MRI or allocating."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sys.path.insert(0, str(PROJECT / 'src'))
from rsna_knee.folds import load_fold_manifest
from rsna_knee.tpu_pilot_inputs import select_original_pilot_batches

ROOT = PROJECT / 'artifacts/private/ticket75-real-mri-selection-v1'
FIT = PROJECT / 'artifacts/private/ticket64-full-backbone-comparison-v4'
FIT_SHA = '9844fd6a7e62f3f1aaf5d0678784f197b8373d702e14c19e669ccde5244a36c8'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze():
    if ROOT.exists() or digest(FIT / 'plan.json') != FIT_SHA:
        raise ValueError('Unused selection and exact original configured science required')
    fit = read(FIT / 'plan.json')
    cohort_path = Path(fit['cohort_parent_plan'])
    cohort = read(cohort_path)
    fold = next(row for row in cohort['folds'] if row['fold'] == 0)
    manifest_binding = next(row for row in cohort['frozen_files'] if Path(row['path']).name == 'fold-manifest.json')
    if digest(manifest_binding['path']) != manifest_binding['sha256']:
        raise ValueError('Original immutable fold file changed')
    manifest = load_fold_manifest(manifest_binding['path'])
    if manifest.superseded or manifest.manifest_checksum != cohort['fold_manifest_checksum']:
        raise ValueError('Original corrected fold identity required')
    selection = select_original_pilot_batches(fold['gold_training_study_uids'], fold['weak_training_study_uids'],
        sorted(manifest.held_out_study_uids(0)))
    pixel_path = Path(fit['pixel_root']) / 'pixel-manifest.json'
    pixels = read(pixel_path)
    rows = {row['study_instance_uid']: row for row in pixels['pixels']}
    if not set(selection.study_order) <= set(rows):
        raise ValueError('Every real selected training MRI must belong to the admitted cache')
    bindings = [dict(study_instance_uid=uid, pixel_path=str(Path(fit['pixel_root']) / 'pixels' / rows[uid]['pixel_file']),
        pixel_file_sha256=rows[uid]['pixel_file_sha256'], source_identity_sha256=rows[uid]['source_identity_sha256'])
        for uid in selection.study_order]
    source_files = [Path(__file__), PROJECT / 'src/rsna_knee/tpu_pilot_inputs.py', PROJECT / 'src/rsna_knee/training.py',
        FIT / 'plan.json', cohort_path, Path(manifest_binding['path']), pixel_path]
    receipt = dict(selection.to_private_dict(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        fit_plan_sha256=FIT_SHA, complete_original_input_fingerprints=fit['input_fingerprints']['0'],
        original_supervision_admission=fold['supervision_admission'], selected_original_pixel_bindings=bindings,
        fold_manifest_checksum=manifest.manifest_checksum, all_original_held_out_studies_excluded=True,
        frozen_metadata_sources=[dict(path=str(p), sha256=digest(p)) for p in source_files],
        actual_MRI_pixels_loaded=False, actual_target_arrays_exported=False, initial_state_exported=False,
        CPU_reference_request_prepared=False, resource_launch_admitted=False, actual_TPU_execution=False)
    ROOT.mkdir(mode=0o700)
    with (ROOT / 'selection.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='SELECTION_METADATA_ONLY', receipt_sha256=digest(ROOT / 'selection.json'),
        selected_training_studies=len(selection.study_order), original_held_out_overlap=0,
        actual_MRI_pixels_loaded=False, resource_launch_admitted=False)))


if __name__ == '__main__':
    freeze()
