"""Exercise the actual worker cache call with one original hash-bound entry.

No native decode, model construction, training, allocation or submission.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
SELECTION_SHA = 'b08d52294efac12435fb49717ffac8b6f51bfb53c22fb1dc7658fb834e2ea8ce'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def replay(worker):
    worker = Path(worker).resolve()
    selection_path = PROJECT/'artifacts/private/ticket75-real-mri-selection-v1/selection.json'
    assert digest(selection_path) == SELECTION_SHA
    selection = json.loads(selection_path.read_text())
    binding = selection['selected_original_pixel_bindings'][0]
    assert type(binding['pixel_path']) is str
    pixels_root = Path(binding['pixel_path']).parent.parent
    pixel_plan = json.loads((pixels_root/'plan.json').read_text())
    # Import the exact frozen loader for a snapshot, or its identical original
    # project module for an editable corrected worker.
    module_root = worker.parent/'src'
    if not module_root.exists(): module_root = PROJECT/'src'
    sys.path.insert(0,str(module_root))
    from rsna_knee.pixel_cache import load_pixel_entry
    calls = [n for n in ast.walk(ast.parse(worker.read_text()))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'load_pixel_entry']
    assert len(calls) == 1
    context = dict(Path=Path,binding=binding,uid=binding['study_instance_uid'],
        source=binding['source_identity_sha256'],pixels_root=pixels_root,pixel_plan=pixel_plan,
        digest=digest,load_pixel_entry=load_pixel_entry)
    # Execute the actual worker expression, preserving argument construction.
    entry = eval(compile(ast.Expression(calls[0]),str(worker),'eval'),context)
    assert entry.metadata['study_instance_uid'] == binding['study_instance_uid']
    assert digest(binding['pixel_path']) == binding['pixel_file_sha256']
    return dict(status='ORIGINAL_CACHE_CALL_PASS',worker_sha256=digest(worker),
        loader_sha256=digest(module_root/'rsna_knee/pixel_cache.py'),
        selection_sha256=SELECTION_SHA,pixel_sha256=binding['pixel_file_sha256'],
        actual_MRI_decode=False,optimizer_updates=0,competition_submission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',required=True,type=Path)
    parser.add_argument('--expect',choices=('pass','legacy-path-failure'),default='pass')
    args = parser.parse_args()
    try:
        result = replay(args.worker)
    except AttributeError as exc:
        if args.expect != 'legacy-path-failure' or str(exc) != "'str' object has no attribute 'is_file'": raise
        print(json.dumps(dict(status='EXACT_FAILED_CACHE_CALL_REPRODUCED',worker_sha256=digest(args.worker),
            error=str(exc),optimizer_updates=0,competition_submission=False),sort_keys=True))
    else:
        assert args.expect == 'pass', 'Expected failure did not occur'
        print(json.dumps(result,sort_keys=True))
