"""Fast actual-import repro for copied helpers' workspace-root resolution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())


def check(snapshot):
    sys.path.insert(0,str(snapshot))
    sys.path.insert(0,str(snapshot / 'src'))
    from ticket63_full_backbone_training_cost import t36, PARENT
    from rsna_knee.folds import load_fold_manifest
    plan = json.loads((PROJECT / 'artifacts/private/ticket64-full-backbone-comparison-v4/plan.json').read_text())
    cohort = json.loads(Path(plan['cohort_parent_plan']).read_text())
    expected = next(r for r in cohort['frozen_files'] if Path(r['path']).name == 'fold-manifest.json')
    # This is the exact failing call, without unrelated MRI loading or fitting.
    manifest = load_fold_manifest(t36.FOLD_MANIFEST_PATH)
    if Path(t36.FOLD_MANIFEST_PATH).resolve() != Path(expected['path']).resolve():
        raise ValueError('Actual copied helper resolved the wrong original fold path')
    with Path(expected['path']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != expected['sha256']:
            raise ValueError('Original manifest source identity changed')
    if manifest.manifest_checksum != cohort['fold_manifest_checksum'] or manifest.superseded:
        raise ValueError('Original corrected fold identity changed')
    for path in (t36.WEIGHTS_PATH, t36.TICKET30_ARTIFACT, t36.FEATURE_ROOT / 'feature-manifest.json',
                 PARENT / 'report-supervision/fold-0/checkpoints/final.pt'):
        if not Path(path).is_file():
            raise ValueError('Copied helper source/model input is missing')
    print(json.dumps(dict(status='ACTUAL_HELPER_PATHS_PASS', original_manifest_sha256=expected['sha256'],
        original_fold_identity_verified=True, model_and_supervision_paths_present=True,
        MRI_loaded=False, optimizer_updates=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('snapshot',type=Path)
    check(parser.parse_args().snapshot.resolve())
