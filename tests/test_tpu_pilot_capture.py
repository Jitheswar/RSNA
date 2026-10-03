"""CPU tree/collective contracts; no actual eight-rank TPU execution."""
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch

from rsna_knee.tpu_pilot_capture import convert_checkpoint_tree, checkpoint_tree_sha256, collect_rank_captures, collect_master_publication
from rsna_knee.tpu_pilot_state import capture_host_rng

CONTEXT = dict(identity=dict(pilot_request_sha256='a'*64),path='staged',successful_steps=8,worker_generation=0)


def tree():
    value = OrderedDict(weight=torch.arange(12,dtype=torch.float32).reshape(3,4),step=torch.tensor(3.))
    value._metadata = OrderedDict([('',dict(version=1))])
    return dict(model=value,groups=[dict(params=[0],lr=1e-5)],rng=capture_host_rng())


def test_flat_conversion_keeps_full_tree_order_metadata_and_no_aliases():
    original = tree();seen = []
    def converter(tensors):
        seen.extend(tensors);return [tensor.clone() for tensor in tensors]
    copied = convert_checkpoint_tree(original,converter)
    assert len(seen) == 4
    assert type(copied['model']) is OrderedDict and copied['model']._metadata == original['model']._metadata
    assert checkpoint_tree_sha256(copied) == checkpoint_tree_sha256(original)
    copied['model']['weight'].fill_(99)
    assert not (original['model']['weight'] == 99).any()
    assert checkpoint_tree_sha256(copied) != checkpoint_tree_sha256(original)


@pytest.mark.parametrize('defect',['missing','shape','dtype','device','nonfinite','executable'])
def test_lossy_or_unsupported_conversion_rejected(defect):
    original = tree()
    if defect == 'executable': original['object'] = object()
    def converter(tensors):
        if defect == 'executable': pytest.fail('Unsupported source must fail before conversion')
        result = [tensor.clone() for tensor in tensors]
        if defect == 'missing': result.pop()
        if defect == 'shape': result[0] = torch.zeros(1)
        if defect == 'dtype': result[0] = result[0].double()
        if defect == 'device': result[0] = torch.zeros_like(result[0],device='meta')
        if defect == 'nonfinite': result[0][0,0] = float('nan')
        return result
    with pytest.raises(ValueError): convert_checkpoint_tree(original,converter)


@pytest.mark.parametrize('defect',['metadata','order','type','shape','dtype','scalar','tensor'])
def test_content_fingerprint_covers_every_checkpoint_detail(defect):
    original = tree();changed = deepcopy(original)
    if defect == 'metadata': changed['model']._metadata['']['version'] = 2
    if defect == 'order': changed['model'].move_to_end('weight')
    if defect == 'type': changed['groups'] = tuple(changed['groups'])
    if defect == 'shape': changed['model']['weight'] = changed['model']['weight'].reshape(4,3)
    if defect == 'dtype': changed['model']['weight'] = changed['model']['weight'].double()
    if defect == 'scalar': changed['groups'][0]['lr'] = 1e-4
    if defect == 'tensor': changed['model']['weight'][0,0] = 1
    assert checkpoint_tree_sha256(original) != checkpoint_tree_sha256(changed)


@pytest.mark.parametrize('defect',[None,'missing','duplicate','bool_rank','context','divergent_model','rng','bad_sha'])
def test_all_eight_states_must_agree_with_all_rank_rng(defect):
    sha = 'e'*64
    rows = [dict(rank=rank,context=CONTEXT,state_sha256=sha,rng=dict(host=capture_host_rng(),xla_seed=100+rank)) for rank in range(8)]
    if defect == 'missing': rows.pop()
    if defect == 'duplicate': rows[7]['rank'] = 6
    if defect == 'bool_rank': rows[0]['rank'] = False
    if defect == 'context': rows[7]['context'] = dict(CONTEXT,successful_steps=16)
    if defect == 'divergent_model': rows[7]['state_sha256'] = 'f'*64
    if defect == 'rng': rows[7]['rng'].pop('host')
    if defect == 'bad_sha': sha = 'invalid'
    if defect:
        with pytest.raises(ValueError): collect_rank_captures(rows,expected_context=CONTEXT,expected_state_sha256=sha)
    else:
        result = collect_rank_captures(rows,expected_context=CONTEXT,expected_state_sha256=sha)
        assert result['replicas_checked'] == 8
        assert [result['rank_rng_states'][i]['xla_seed'] for i in range(8)] == list(range(100,108))


@pytest.mark.parametrize('defect',[None,'missing','duplicate','context','second_writer','master_absent'])
def test_only_global_master_can_publish_and_all_ranks_acknowledge(defect):
    rows = [dict(rank=rank,context=CONTEXT,sha256='e'*64 if rank == 0 else None) for rank in range(8)]
    if defect == 'missing': rows.pop()
    if defect == 'duplicate': rows[7]['rank'] = 6
    if defect == 'context': rows[7]['context'] = dict(CONTEXT,path='continuous')
    if defect == 'second_writer': rows[7]['sha256'] = 'f'*64
    if defect == 'master_absent': rows[0]['sha256'] = None
    if defect:
        with pytest.raises(ValueError): collect_master_publication(rows,expected_context=CONTEXT)
    else: assert collect_master_publication(rows,expected_context=CONTEXT) == 'e'*64
