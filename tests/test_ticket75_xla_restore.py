"""Actual CPU model/state restoration with mocked device RNG/sync boundaries."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsna_knee.model import ModelConfig,build_study_model
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID,RESNET18_FULL_FINETUNE_MODEL_ID,ANATOMICAL_POOLING_ID
from rsna_knee.tpu_pilot_checkpoint import build_pilot_checkpoint
from rsna_knee.tpu_pilot_state import capture_host_rng

spec = importlib.util.spec_from_file_location('ticket75_xla_restore_adapter',Path(__file__).parents[1]/'scripts/ticket75_xla_update.py')
update = importlib.util.module_from_spec(spec);spec.loader.exec_module(update)
IDENTITY = dict(pilot_request_sha256='a'*64,input_sha256='b'*64,initial_state_sha256='c'*64,scientific_plan_sha256='d'*64)
MASKS = [torch.ones(4,12,dtype=torch.bool) for _ in range(16)]


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model():
    return build_study_model(ModelConfig(image_size=(32,32),model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,pooling_id=ANATOMICAL_POOLING_ID))


def advance(m,optimizer,scheduler,count):
    for _ in range(count):
        for p in m.parameters(): p.grad = torch.full_like(p,.001)
        optimizer.step();scheduler.step()


@pytest.fixture(scope='module')
def saved():
    before = torch.get_num_threads();torch.set_num_threads(1)
    try:
        m = model();optimizer = update.original_adam(m);scheduler = update.original_scheduler(optimizer)
        advance(m,optimizer,scheduler,8)
        host = capture_host_rng()
        return build_pilot_checkpoint(identity=IDENTITY,path='staged',step=8,worker_generation=0,
            model_state=m.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
            parameter_names=[[f'encoder.{n}' for n,_ in m.encoder.named_parameters()],[f'head.{n}' for n,_ in m.head.named_parameters()]],
            config=dict(model=m.config.to_dict(),synthetic_test=True),ordered_masks=MASKS,
            rank_rng_states={rank:dict(host=host,xla_seed=75+rank) for rank in range(8)})
    finally:
        torch.set_num_threads(before)


def backend():
    b = object.__new__(update.XlaPilotUpdate)
    b.device = torch.device('cpu');b.rank = 7
    b.checkpoint_restore_consumed = b.failed_or_inflight = False
    b.restored_checkpoint = None
    b.worker_identity = b.next_step = b.initialized_baseline = b.normalization = None
    events = [];seed = {}
    b.xla = SimpleNamespace(sync=lambda **kwargs:events.append('sync'))
    def set_rng(value,**kwargs): seed['value'] = value;events.append('restore_device_rng')
    b.xm = SimpleNamespace(set_rng_state=set_rng,get_rng_state=lambda **kwargs:seed['value'])
    return b,events


def test_original_full_model_adam_scheduler_and_rank7_rng_restored_without_source_alias(saved):
    b,events = backend()
    m,optimizer,scheduler,receipt = b.restore_staged_checkpoint(model(),saved,
        expected_identity=IDENTITY,ordered_masks=MASKS,expected_config=saved['config'])
    assert receipt['next_batch_index'] == 8 and receipt['fresh_worker_generation'] == 1 and receipt['rank'] == 7
    assert receipt['actual_complete_pilot_verified'] is receipt['competition_submission'] is False
    assert events == ['sync','restore_device_rng']
    assert b.restored_checkpoint == dict(identity=IDENTITY,rank=7,successful_steps=8)
    torch.testing.assert_close(capture_host_rng()['torch'],saved['rank_rng_states'][7]['host']['torch'],rtol=0,atol=0)
    assert capture_host_rng()['python'] == saved['rank_rng_states'][7]['host']['python']
    for name,value in m.state_dict().items(): torch.testing.assert_close(value,saved['model_state_dict'][name],rtol=0,atol=0)
    update.validate_optimizer(m,optimizer,scheduler,8)
    assert all(state['step'].device.type == 'cpu' for state in optimizer.state.values())
    assert all(state['exp_avg'].data_ptr() != saved['optimizer_state_dict']['state'][i]['exp_avg'].data_ptr()
        for i,state in enumerate(optimizer.state.values()))
    advance(m,optimizer,scheduler,8)
    update.validate_optimizer(m,optimizer,scheduler,16)
    assert all(float(state['step']) == 8 for state in saved['optimizer_state_dict']['state'].values())
    with pytest.raises(ValueError,match='Fresh unused worker'):
        b.restore_staged_checkpoint(model(),saved,expected_identity=IDENTITY,ordered_masks=MASKS,expected_config=saved['config'])


@pytest.mark.parametrize('defect',['used_worker','wrong_path','wrong_identity','wrong_config','wrong_shape','wrong_architecture','device_rng_failure'])
def test_restore_rejects_bad_context_or_state_and_preserves_consumed_device_failure(saved,defect):
    b,events = backend();m = model();value = saved;identity = IDENTITY;config = saved['config']
    if defect == 'used_worker': b.worker_identity = ('staged',0)
    if defect == 'wrong_path': value = dict(saved,path='continuous')
    if defect == 'wrong_identity': identity = dict(IDENTITY,input_sha256='f'*64)
    if defect == 'wrong_config': config = dict(saved['config'],changed=True)
    if defect == 'wrong_shape':
        value = dict(saved,model_state_dict=deepcopy(saved['model_state_dict']))
        value['model_state_dict']['imagenet_mean'] = torch.zeros(1)
    if defect == 'wrong_architecture': m = torch.nn.Linear(4,12)
    if defect == 'device_rng_failure': b.xm.get_rng_state = lambda **kwargs:0
    before = {k:v.clone() for k,v in m.state_dict().items()}
    with pytest.raises(ValueError): b.restore_staged_checkpoint(m,value,expected_identity=identity,ordered_masks=MASKS,expected_config=config)
    assert b.checkpoint_restore_consumed is (defect == 'device_rng_failure')
    if defect != 'device_rng_failure':
        assert not events
        for name,tensor in m.state_dict().items(): torch.testing.assert_close(tensor,before[name],rtol=0,atol=0)
    else:
        with pytest.raises(ValueError,match='Fresh unused worker'):
            b.restore_staged_checkpoint(m,value,expected_identity=identity,ordered_masks=MASKS,expected_config=config)


def test_fresh_generation_cannot_train_before_successful_exact_restore():
    b,_ = backend()
    with pytest.raises(ValueError,match='restoration required'):
        b.update(None,None,None,None,path='staged',successful_step=9,worker_generation=1)
