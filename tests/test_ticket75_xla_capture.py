"""Full original CPU model capture; only collective/device boundaries are mocked."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsna_knee.model import ModelConfig,build_study_model
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID,RESNET18_FULL_FINETUNE_MODEL_ID,ANATOMICAL_POOLING_ID
from rsna_knee.tpu_pilot_checkpoint import load_pilot_checkpoint
from rsna_knee.tpu_pilot_state import capture_host_rng
from rsna_knee.tpu_pilot_capture import checkpoint_tree_sha256

spec = importlib.util.spec_from_file_location('ticket75_xla_capture_adapter',Path(__file__).parents[1]/'scripts/ticket75_xla_update.py')
update = importlib.util.module_from_spec(spec);spec.loader.exec_module(update)
IDENTITY = dict(pilot_request_sha256='a'*64,input_sha256='b'*64,initial_state_sha256='c'*64,scientific_plan_sha256='d'*64)
MASKS = [torch.ones(4,12,dtype=torch.bool) for _ in range(16)]


@pytest.fixture(scope='module')
def trained():
    before = torch.get_num_threads();torch.set_num_threads(1)
    m = build_study_model(ModelConfig(image_size=(32,32),model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,pooling_id=ANATOMICAL_POOLING_ID))
    optimizer = update.original_adam(m);scheduler = update.original_scheduler(optimizer)
    for _ in range(3):
        for p in m.parameters(): p.grad = torch.full_like(p,.001)
        optimizer.step();scheduler.step()
    yield m,optimizer,scheduler
    torch.set_num_threads(before)


def backend(m,rank=0,defect=None):
    b = update.XlaPilotUpdate.__new__(update.XlaPilotUpdate)
    b.rank = rank;b.device = torch.device('cpu');b.failed_or_inflight = False
    b.worker_identity = ('continuous',0);b.next_step = 4;b.checkpoint_restore_consumed = False
    b.checkpoint_captures_consumed = set()
    b.normalization = {n:v.clone() for n,v in m.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
    events = [];seed = [100+rank]
    b.xla = SimpleNamespace(sync=lambda **kwargs:events.append('sync'))
    b.metrics = SimpleNamespace(metric_names=lambda:[],counter_names=lambda:[],executed_fallback_ops=lambda:[])
    def convert(tensors,**kwargs):
        events.append('convert_all_rank_state')
        assert kwargs == dict(convert=True)
        if defect == 'conversion': raise RuntimeError('Actual conversion failure')
        return [t.detach().clone() for t in tensors]
    def mesh(tag,local,reducer):
        events.append(tag.rsplit('-',1)[-1])
        if tag.endswith('-capture'):
            rows = [dict(rank=i,context=local['context'],state_sha256=local['state_sha256'],
                rng=dict(host=local['rng']['host'],xla_seed=100+i)) for i in range(8)]
            if defect == 'divergence': rows[7]['state_sha256'] = 'f'*64
            if defect == 'missing_rank': rows.pop()
        else:
            rows = [dict(rank=i,context=local['context'],sha256=(local['sha256'] if rank == 0 else 'e'*64) if i == 0 else None) for i in range(8)]
            if defect == 'rng_change': seed[0] += 1
        return reducer(rows)
    b.xm = SimpleNamespace(_maybe_convert_to_cpu=convert,mesh_reduce=mesh,get_rng_state=lambda **kwargs:seed[0])
    return b,events


def capture(b,trained,destination,**kwargs):
    m,o,s = trained
    return b.capture_checkpoint(m,o,s,expected_identity=IDENTITY,ordered_masks=MASKS,
        expected_config=dict(model=m.config.to_dict()),path='continuous',successful_step=3,worker_generation=0,
        destination=destination,byte_limit=kwargs.get('byte_limit',200_000_000))


def test_full_architecture_master_capture_reload_metadata_and_rng_unchanged(trained,tmp_path):
    m,o,s = trained;b,events = backend(m);before = checkpoint_tree_sha256(capture_host_rng())
    path = tmp_path/'continuous-3.pt';result = capture(b,trained,path)
    saved = load_pilot_checkpoint(path,expected_sha256=result['published_checkpoint_sha256'],expected_identity=IDENTITY,ordered_masks=MASKS)
    assert saved['model_state_dict']._metadata == m.state_dict()._metadata
    for name,value in m.state_dict().items(): torch.testing.assert_close(value,saved['model_state_dict'][name],rtol=0,atol=0)
    assert [saved['rank_rng_states'][i]['xla_seed'] for i in range(8)] == list(range(100,108))
    assert before == checkpoint_tree_sha256(capture_host_rng())
    assert result['global_master_writer'] and result['all_eight_rank_states_agreed'] and result['all_eight_rank_rng_states_captured']
    assert result['local_host_device_rng_preserved'] and result['global_post_capture_rng_preservation_verified'] is False
    assert result['actual_complete_pilot_verified'] is result['actual_fresh_PJRT_replay_verified'] is result['competition_submission'] is False
    assert events[0:2] == ['sync','convert_all_rank_state'] and events[-1] == 'sync'
    assert not b.failed_or_inflight
    with pytest.raises(ValueError,match='Consumed checkpoint boundary'): capture(b,trained,path)


def test_nonmaster_converts_and_collects_but_never_writes(trained,tmp_path):
    b,events = backend(trained[0],rank=7);path = tmp_path/'must-not-exist.pt'
    result = capture(b,trained,path)
    assert result['published_checkpoint_sha256'] == 'e'*64 and result['global_master_writer'] is False
    assert 'convert_all_rank_state' in events and not list(tmp_path.iterdir())


@pytest.mark.parametrize('defect',['conversion','divergence','missing_rank','rng_change','overbound'])
def test_failure_consumes_capture_and_blocks_training_without_retry(trained,tmp_path,defect):
    b,events = backend(trained[0],defect=defect);path = tmp_path/'retained.pt'
    with pytest.raises((ValueError,RuntimeError)):
        capture(b,trained,path,byte_limit=1 if defect == 'overbound' else 200_000_000)
    assert b.failed_or_inflight
    if defect in ('conversion','divergence','missing_rank'): assert not list(tmp_path.iterdir())
    if defect == 'overbound': assert not path.exists() and path.with_name(path.name+'.tmp').exists()
    if defect == 'rng_change': assert path.exists()
    before = list(events)
    with pytest.raises(ValueError,match='Failed or in-flight'): capture(b,trained,path)
    assert events == before
    with pytest.raises(ValueError,match='failed or in-flight'):
        b.update(None,None,None,None,path='continuous',successful_step=4,worker_generation=0)


def test_wrong_prefix_rejected_before_backend_operations(trained,tmp_path):
    b,events = backend(trained[0]);b.next_step = 3
    with pytest.raises(ValueError,match='actual completed original worker prefix'): capture(b,trained,tmp_path/'unused.pt')
    assert events == [] and b.checkpoint_captures_consumed == set() and not list(tmp_path.iterdir())
