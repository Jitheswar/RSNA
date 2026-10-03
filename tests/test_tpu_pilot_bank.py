"""Sparse synthetic224 arrays verify loader behavior, not original MRI identity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna_knee.targets import TARGET_COLUMNS
from rsna_knee.tpu_pilot_bank import FrozenPilotInputBank,FIELDS


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def save_manifest(root,manifest):
    path = root/'input-bank-manifest.json';path.write_text(json.dumps(manifest)+'\n');return digest(path)


@pytest.fixture(scope='module')
def bank_files(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-pilot-bank')
    selection = dict(schema_version='rsna-ticket75-fixed-pilot-selection-v1',fold=0,successful_updates_per_path=16,
        target_columns=list(TARGET_COLUMNS),real_source_roles=['reference','reference','weak','weak'],padding_ranks=[4,5,6,7],
        reference_pool_sha256='a'*64,report_pool_sha256='b'*64,study_order=[f'synthetic-{i}' for i in range(64)],
        batch_study_indices=[list(range(4*i,4*i+4)) for i in range(16)])
    (root/'selection.json').write_text(json.dumps(selection)+'\n')
    rows = []
    specs = dict(images=((16,8,1,224,224),np.float32),slice_mask=((16,8),np.bool_),series_mask=((16,),np.bool_),
        series_descriptors=((16,10),np.float32),targets=((12,),np.float32),supervision_mask=((12,),np.bool_))
    for index in range(64):
        directory = root/f'studies/{index:02d}';directory.mkdir(parents=True)
        for name,(shape,dtype) in specs.items():
            path = directory/f'{name}.npy'
            # Truncated zero image pages are sparse; no lossy codec or altered
            # production geometry is used for these synthetic source fixtures.
            value = np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=dtype)
            if name == 'slice_mask': value[0,0] = True
            if name == 'series_mask': value[0] = True
            if name == 'series_descriptors': value[0,0] = index+1
            if name == 'supervision_mask': value[:index%11+1] = True
            if name == 'targets':
                value[:] = float('nan');value[:index%11+1] = index%2
            value.flush();del value
            rows.append(dict(study_index=index,field=name,path=str(path.relative_to(root)),sha256=digest(path),bytes=path.stat().st_size))
    manifest = dict(schema_version='rsna-ticket75-lossless-input-bank-v1',selection=dict(path='selection.json',sha256=digest(root/'selection.json')),arrays=rows)
    return root,manifest


def test_exact_original_batch_order_masks_copy_isolation_and_final_byte_check(bank_files):
    root,manifest = bank_files
    sha = save_manifest(root,manifest);bank = FrozenPilotInputBank(root,expected_manifest_sha256=sha)
    batch = bank.global_batch(9)
    torch.testing.assert_close(batch['series_descriptors'][:,0,0],torch.tensor([37.,38.,39.,40.]))
    assert batch['images'].shape == (4,16,8,1,224,224)
    assert batch['supervision_mask'].sum(1).tolist() == [4,5,6,7]
    assert torch.isnan(batch['targets'][~batch['supervision_mask']]).all()
    batch['images'].fill_(1)
    assert not bank.global_batch(9)['images'].any()
    assert not bank.study_arrays(36)['images'].flags.writeable
    assert bank.replay_inputs()['images'].shape == (8,16,8,1,224,224)
    assert len(bank.ordered_masks()) == 16
    result = bank.verify_final_bytes()
    assert result['original_array_files'] == 384 and result['lossless_bytes_reverified']
    assert result['original_MRI_identity_verified'] is result['actual_TPU_execution_proven'] is result['competition_submission'] is False


@pytest.mark.parametrize('defect',['manifest_hash','missing_field','duplicate_field','escaped_path','array_hash','wrong_draw_count','resampled_subset','selection_hash'])
def test_altered_or_incomplete_frozen_bank_rejected_before_training(bank_files,defect):
    root,original = bank_files;manifest = deepcopy(original)
    selection_path = root/'selection.json';selection_bytes = selection_path.read_bytes()
    try:
        if defect == 'missing_field': manifest['arrays'].pop()
        if defect == 'duplicate_field': manifest['arrays'][1] = manifest['arrays'][0]
        if defect == 'escaped_path': manifest['arrays'][0]['path'] = '../outside.npy'
        if defect == 'array_hash': manifest['arrays'][0]['sha256'] = '0'*64
        if defect in ('wrong_draw_count','resampled_subset'):
            selection = json.loads(selection_bytes)
            if defect == 'wrong_draw_count': selection['batch_study_indices'].pop()
            else: selection['batch_study_indices'][0][0] = 1
            selection_path.write_text(json.dumps(selection)+'\n');manifest['selection']['sha256'] = digest(selection_path)
        if defect == 'selection_hash': manifest['selection']['sha256'] = '0'*64
        sha = save_manifest(root,manifest)
        if defect == 'manifest_hash': sha = '0'*64
        with pytest.raises(ValueError): FrozenPilotInputBank(root,expected_manifest_sha256=sha)
    finally:
        selection_path.write_bytes(selection_bytes);save_manifest(root,original)


def test_changed_file_detected_after_admission_before_reuse(bank_files):
    root,manifest = bank_files
    bank = FrozenPilotInputBank(root,expected_manifest_sha256=save_manifest(root,manifest))
    path = root/'studies/00/series_descriptors.npy';original = path.read_bytes()
    try:
        value = np.load(path,allow_pickle=False);value[0,0] = 999;np.save(path,value,allow_pickle=False)
        with pytest.raises(ValueError,match='after byte verification'): bank.global_batch(0)
        with pytest.raises(ValueError,match='during execution'): bank.verify_final_bytes()
    finally:
        path.write_bytes(original)


def test_hash_bound_wrong_uncertainty_mask_dtype_rejected(bank_files):
    root,original_manifest = bank_files;manifest = deepcopy(original_manifest)
    path = root/'studies/00/supervision_mask.npy';original_bytes = path.read_bytes()
    try:
        np.save(path,np.load(path,allow_pickle=False).astype(np.uint8),allow_pickle=False)
        row = next(row for row in manifest['arrays'] if row['study_index'] == 0 and row['field'] == 'supervision_mask')
        row.update(sha256=digest(path),bytes=path.stat().st_size)
        with pytest.raises(ValueError,match='dtype'): FrozenPilotInputBank(root,expected_manifest_sha256=save_manifest(root,manifest))
    finally:
        path.write_bytes(original_bytes);save_manifest(root,original_manifest)
