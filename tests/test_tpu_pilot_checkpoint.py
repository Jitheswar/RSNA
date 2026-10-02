"""CPU format/replay checks; no actual XLA state or device is acquired."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from rsna_knee.tpu_pilot_checkpoint import (build_pilot_checkpoint,load_pilot_checkpoint,
    publish_pilot_checkpoint,validate_pilot_checkpoint,verify_exact_pilot_replay)
from rsna_knee.tpu_pilot_state import capture_host_rng,restore_host_rng

IDENTITY = dict(pilot_request_sha256='a'*64,input_sha256='b'*64,initial_state_sha256='c'*64,scientific_plan_sha256='d'*64)
MASKS = [torch.ones(4,12,dtype=torch.bool) for _ in range(16)]
MASKS[2][2,3:] = False


def fresh():
    torch.manual_seed(75)
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(4,7)
    model.head = torch.nn.Linear(7,12)
    optimizer = torch.optim.Adam([dict(params=model.encoder.parameters(),lr=1e-5),dict(params=model.head.parameters(),lr=1e-4)])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
    return model,optimizer,scheduler


def advance(model,optimizer,scheduler,count):
    for _ in range(count):
        x = torch.randn(4,4)*(random.random()+float(np.random.normal()))
        y = torch.rand(4,12)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(model.head(model.encoder(x)),y).backward()
        optimizer.step();scheduler.step()


def packet(model,optimizer,scheduler,step,path='continuous'):
    host = capture_host_rng()
    return build_pilot_checkpoint(identity=IDENTITY,path=path,step=step,worker_generation=int(path == 'staged' and step == 16),
        model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
        parameter_names=[[f'encoder.{n}' for n,_ in model.encoder.named_parameters()],[f'head.{n}' for n,_ in model.head.named_parameters()]],
        config=dict(disposable_synthetic_CPU_test=True),ordered_masks=MASKS,
        rank_rng_states={rank:dict(host=host,xla_seed=75+rank) for rank in range(8)})


def test_fresh_restricted_checkpoint_restoration_replays_complete_state_exactly(tmp_path):
    random.seed(75);np.random.seed(75)
    m,optimizer,scheduler = fresh()
    advance(m,optimizer,scheduler,8)
    saved = packet(m,optimizer,scheduler,8,path='staged')
    path = tmp_path/'step8.pt'
    sha = publish_pilot_checkpoint(saved,path,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=1_000_000)
    loaded = load_pilot_checkpoint(path,expected_sha256=sha,expected_identity=IDENTITY,ordered_masks=MASKS)
    advance(m,optimizer,scheduler,8)
    continuous = packet(m,optimizer,scheduler,16)
    restarted,new_optimizer,new_scheduler = fresh()
    restarted.load_state_dict(loaded['model_state_dict'],strict=True)
    new_optimizer.load_state_dict(loaded['optimizer_state_dict'])
    new_scheduler.load_state_dict(loaded['scheduler_state_dict'])
    restore_host_rng(loaded['rank_rng_states'][0]['host'])
    advance(restarted,new_optimizer,new_scheduler,8)
    staged = packet(restarted,new_optimizer,new_scheduler,16,path='staged')
    result = verify_exact_pilot_replay(continuous,staged,expected_identity=IDENTITY,ordered_masks=MASKS)
    assert result['tensor_exact_replay'] and result['all_eight_rank_rng_states_exact']
    assert result['actual_TPU_execution_proven'] is result['original_fit_updates_credited'] is result['competition_submission'] is False
    with pytest.raises(ValueError,match='without retry'):
        publish_pilot_checkpoint(saved,path,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=1_000_000)
    with pytest.raises(ValueError,match='bytes changed'):
        load_pilot_checkpoint(path,expected_sha256='f'*64,expected_identity=IDENTITY,ordered_masks=MASKS)


@pytest.mark.parametrize('defect',['missing_rank','rank_bool','missing_device_rng','bad_host_rng','wrong_input','wrong_generation','credited_fit','missing_moment',
    'step','moment_shape','moment_negative','moment_nonfinite','missing_parameter','parameter_order','optimizer_order','model_dtype','reordered_scope','scheduler','padding_exposure','extra_object','non_cpu'])
def test_incomplete_or_relabelled_checkpoint_rejected(defect):
    m,optimizer,scheduler = fresh();advance(m,optimizer,scheduler,3)
    value = packet(m,optimizer,scheduler,3)
    if defect == 'missing_rank': value['rank_rng_states'].pop(7)
    if defect == 'rank_bool': value['rank_rng_states'][False] = value['rank_rng_states'].pop(0)
    if defect == 'missing_device_rng': value['rank_rng_states'][7].pop('xla_seed')
    if defect == 'bad_host_rng': value['rank_rng_states'][7]['host']['torch'] = torch.zeros(1,dtype=torch.uint8)
    if defect == 'wrong_input': value['identity']['input_sha256'] = 'f'*64
    if defect == 'wrong_generation': value['worker_generation'] = 1
    if defect == 'credited_fit': value['original_fit_updates_credited'] = True
    if defect == 'missing_moment': value['optimizer_state_dict']['state'][0].pop('exp_avg')
    if defect == 'step': value['optimizer_state_dict']['state'][0]['step'].fill_(2)
    if defect == 'moment_shape': value['optimizer_state_dict']['state'][0]['exp_avg'] = torch.zeros(1)
    if defect == 'moment_negative': value['optimizer_state_dict']['state'][0]['exp_avg_sq'].flatten()[0] = -1
    if defect == 'moment_nonfinite': value['optimizer_state_dict']['state'][0]['exp_avg_sq'].flatten()[0] = float('nan')
    if defect == 'missing_parameter': value['parameter_names'][0].pop()
    if defect == 'parameter_order': value['parameter_names'][0].reverse()
    if defect == 'optimizer_order': value['optimizer_state_dict']['param_groups'][0]['params'].reverse()
    if defect == 'model_dtype': value['model_state_dict']['encoder.weight'] = value['model_state_dict']['encoder.weight'].double()
    if defect == 'reordered_scope': value['parameter_names'].reverse()
    if defect == 'scheduler': value['scheduler_state_dict']['last_epoch'] = 2
    if defect == 'padding_exposure': value['progress']['sample_exposures'] = 24
    if defect == 'extra_object': value['config']['object'] = object()
    if defect == 'non_cpu': value['model_state_dict']['encoder.weight'] = torch.zeros(7,4,device='meta')
    with pytest.raises(ValueError): validate_pilot_checkpoint(value,expected_identity=IDENTITY,ordered_masks=MASKS)


def test_overbound_publication_preserves_partial_without_overwriting_or_retry(tmp_path):
    m,optimizer,scheduler = fresh()
    value = packet(m,optimizer,scheduler,0)
    path = tmp_path/'initial.pt'
    with pytest.raises(ValueError,match='byte bound'):
        publish_pilot_checkpoint(value,path,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=1)
    assert not path.exists() and path.with_name('initial.pt.tmp').exists()
    with pytest.raises(ValueError,match='without retry'):
        publish_pilot_checkpoint(value,path,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=1_000_000)


@pytest.mark.parametrize('changed',['model','adam','rng7','device_rng7','sampler','config','metadata'])
def test_same_backend_replay_requires_exact_tensors_and_all_rank_rng(changed):
    m,optimizer,scheduler = fresh();advance(m,optimizer,scheduler,16)
    continuous = packet(m,optimizer,scheduler,16)
    staged = deepcopy(continuous);staged['path'] = 'staged';staged['worker_generation'] = 1
    if changed == 'model': staged['model_state_dict']['encoder.weight'].flatten()[0] += 1e-7
    if changed == 'adam': staged['optimizer_state_dict']['state'][0]['exp_avg'].flatten()[0] += 1e-7
    if changed == 'rng7': staged['rank_rng_states'][7]['host']['numpy']['position'] = 0
    if changed == 'device_rng7': staged['rank_rng_states'][7]['xla_seed'] += 1
    if changed == 'sampler': staged['sampler_state']['next_batch_index'] = 15
    if changed == 'config': staged['config']['different'] = True
    if changed == 'metadata': staged['model_state_dict']._metadata['encoder']['version'] += 1
    with pytest.raises(ValueError): verify_exact_pilot_replay(continuous,staged,expected_identity=IDENTITY,ordered_masks=MASKS)


def test_original_full_backbone_scope_and_buffers_survive_restricted_publication(tmp_path):
    from rsna_knee.model import ModelConfig,build_study_model
    from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID,RESNET18_FULL_FINETUNE_MODEL_ID,ANATOMICAL_POOLING_ID
    threads = torch.get_num_threads();torch.set_num_threads(1)
    try:
        m = build_study_model(ModelConfig(image_size=(32,32),model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
            encoder_id=RESNET18_FIXTURE_ENCODER_ID,pooling_id=ANATOMICAL_POOLING_ID))
        optimizer = torch.optim.Adam([dict(params=m.encoder.parameters(),lr=1e-5),dict(params=m.head.parameters(),lr=1e-4)])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        # Actual original architecture/ordinary Adam state, with synthetic fixed
        # gradients. This validates serialization rather than claiming MRI fit.
        for _ in range(3):
            for p in m.parameters(): p.grad = torch.full_like(p,.001)
            optimizer.step();scheduler.step()
        value = packet(m,optimizer,scheduler,3)
        path = tmp_path/'full-model.pt'
        sha = publish_pilot_checkpoint(value,path,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=200_000_000)
        loaded = load_pilot_checkpoint(path,expected_sha256=sha,expected_identity=IDENTITY,ordered_masks=MASKS)
        assert sum(map(len,loaded['parameter_names'])) == len(tuple(m.parameters()))
        assert loaded['model_state_dict']._metadata == m.state_dict()._metadata
        for name,tensor in m.state_dict().items(): torch.testing.assert_close(loaded['model_state_dict'][name],tensor,rtol=0,atol=0)
    finally:
        torch.set_num_threads(threads)
