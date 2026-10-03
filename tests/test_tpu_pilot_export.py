"""Synthetic original-architecture CPU export checks, without native MRI proof."""
from copy import deepcopy

import pytest
import torch

from rsna_knee.model import ModelConfig,build_study_model
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID,RESNET18_FULL_FINETUNE_MODEL_ID,ANATOMICAL_POOLING_ID
from rsna_knee.tpu_pilot_checkpoint import build_pilot_checkpoint,publish_pilot_checkpoint
from rsna_knee.tpu_pilot_export import derive_model_export,load_model_export,publish_model_export,replay_export_logits,restore_cpu_export,validate_model_export
from rsna_knee.tpu_pilot_state import capture_host_rng

IDENTITY = dict(pilot_request_sha256='a'*64,input_sha256='b'*64,initial_state_sha256='c'*64,scientific_plan_sha256='d'*64)
MASKS = [torch.ones(4,12,dtype=torch.bool) for _ in range(16)]
PROVENANCE = dict(source_bindings=[dict(name='synthetic-source',sha256='e'*64)],license_bindings=[dict(name='synthetic-license-record',sha256='f'*64)])


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fresh():
    return build_study_model(ModelConfig(image_size=(32,32),model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,pooling_id=ANATOMICAL_POOLING_ID))


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    threads = torch.get_num_threads();torch.set_num_threads(1)
    try:
        m = fresh()
        optimizer = torch.optim.Adam([dict(params=m.encoder.parameters(),lr=1e-5),dict(params=m.head.parameters(),lr=1e-4)])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
        for _ in range(16):
            for p in m.parameters(): p.grad = torch.full_like(p,.001)
            optimizer.step();scheduler.step()
        host = capture_host_rng()
        config = dict(model=m.config.to_dict(),synthetic_test=True)
        value = build_pilot_checkpoint(identity=IDENTITY,path='staged',step=16,worker_generation=1,
            model_state=m.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
            parameter_names=[[f'encoder.{n}' for n,_ in m.encoder.named_parameters()],[f'head.{n}' for n,_ in m.head.named_parameters()]],
            config=config,ordered_masks=MASKS,rank_rng_states={rank:dict(host=host,xla_seed=75+rank) for rank in range(8)})
        root = tmp_path_factory.mktemp('pilot-export')
        parent = root/'parent.pt'
        parent_sha = publish_pilot_checkpoint(value,parent,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=200_000_000)
        export = derive_model_export(parent,checkpoint_sha256=parent_sha,expected_identity=IDENTITY,
            ordered_masks=MASKS,expected_config=config,provenance=PROVENANCE)
        return export,dict(expected_identity=IDENTITY,expected_checkpoint_sha256=parent_sha,expected_config=config,expected_provenance=PROVENANCE),parent
    finally:
        torch.set_num_threads(threads)


def inputs():
    torch.manual_seed(75)
    result = dict(images=torch.rand(8,16,8,1,32,32),slice_mask=torch.zeros(8,16,8,dtype=torch.bool),
        series_mask=torch.zeros(8,16,dtype=torch.bool),series_descriptors=torch.zeros(8,16,10))
    result['slice_mask'][:,0,0] = True;result['series_mask'][:,0] = True
    result['series_descriptors'][:,0,0] = 1
    return result


def test_model_only_restricted_round_trip_reproduces_batch_two_logits(exported,tmp_path):
    packet,expected,parent = exported
    path = tmp_path/'model-only.pt'
    sha = publish_model_export(packet,path,byte_limit=100_000_000,**expected)
    loaded = load_model_export(path,expected_sha256=sha,**expected)
    assert path.stat().st_size < parent.stat().st_size
    assert not {'optimizer_state_dict','scheduler_state_dict','rank_rng_states','progress'} & loaded.keys()
    first,second = fresh(),fresh()
    restore_cpu_export(first,packet,**expected);restore_cpu_export(second,loaded,**expected)
    batch = inputs()
    expected_logits = replay_export_logits(first,batch)
    actual_logits = replay_export_logits(second,batch)
    torch.testing.assert_close(actual_logits,expected_logits,rtol=0,atol=1e-6)
    assert loaded['actual_TPU_execution_proven'] is loaded['original_MRI_replay_verified'] is loaded['competition_submission'] is False
    with pytest.raises(ValueError,match='without retry'): publish_model_export(packet,path,byte_limit=100_000_000,**expected)
    with pytest.raises(ValueError,match='bytes changed'): load_model_export(path,expected_sha256='0'*64,**expected)


@pytest.mark.parametrize('defect',['parent','source','config','license','missing_license','duplicate_source','training_state','dtype','parameter_order','nonfinite','promotion','native_claim'])
def test_altered_or_promoted_export_rejected(exported,defect):
    packet,expected,_ = exported;value = deepcopy(packet)
    if defect == 'parent': value['parent_checkpoint_sha256'] = '0'*64
    if defect == 'source': value['identity']['input_sha256'] = '0'*64
    if defect == 'config': value['config']['synthetic_test'] = False
    if defect == 'license': value['provenance']['license_bindings'][0]['sha256'] = '0'*64
    if defect == 'missing_license': value['provenance']['license_bindings'] = []
    if defect == 'duplicate_source': value['provenance']['source_bindings'] *= 2
    if defect == 'training_state': value['optimizer_state_dict'] = {}
    if defect == 'dtype': value['model_state_dict']['encoder.conv1.weight'] = value['model_state_dict']['encoder.conv1.weight'].double()
    if defect == 'parameter_order': value['parameter_names'][0].reverse()
    if defect == 'nonfinite': value['model_state_dict']['encoder.conv1.weight'].flatten()[0] = float('nan')
    if defect == 'promotion': value['candidate_promotion'] = True
    if defect == 'native_claim': value['original_MRI_replay_verified'] = True
    with pytest.raises(ValueError): validate_model_export(value,**expected)


def test_actual_parent_hash_is_required_before_derivation(exported):
    _,expected,parent = exported
    with pytest.raises(ValueError,match='bytes changed'):
        derive_model_export(parent,checkpoint_sha256='0'*64,expected_identity=IDENTITY,ordered_masks=MASKS,
            expected_config=expected['expected_config'],provenance=PROVENANCE)


def test_architecture_mismatch_fails_before_cpu_model_mutation(exported):
    packet,expected,_ = exported;value = deepcopy(packet)
    value['model_state_dict']['imagenet_mean'] = torch.zeros(1)
    m = fresh();before = {k:v.clone() for k,v in m.state_dict().items()}
    with pytest.raises(ValueError,match='before restore'): restore_cpu_export(m,value,**expected)
    for name,tensor in m.state_dict().items(): torch.testing.assert_close(tensor,before[name],rtol=0,atol=0)


@pytest.mark.parametrize('defect',['empty_study','wrong_shape','bad_descriptor','bad_pixels','wrong_dtype'])
def test_replay_rejects_missing_mri_or_input_corruption_before_inference(defect):
    m = fresh();batch = inputs()
    if defect == 'empty_study': batch['series_mask'][7] = False
    if defect == 'wrong_shape': batch['images'] = batch['images'][:7]
    if defect == 'bad_descriptor': batch['series_descriptors'][7,0,0] = float('nan')
    if defect == 'bad_pixels': batch['images'][7,0,0,0,0,0] = float('inf')
    if defect == 'wrong_dtype': batch['slice_mask'] = batch['slice_mask'].float()
    with pytest.raises(ValueError): replay_export_logits(m,batch)


def test_nonfinal_checkpoint_cannot_be_exported_as_a_final_model(exported,tmp_path):
    _,expected,_ = exported
    m = fresh()
    optimizer = torch.optim.Adam([dict(params=m.encoder.parameters(),lr=1e-5),dict(params=m.head.parameters(),lr=1e-4)])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda _:1.)
    host = capture_host_rng()
    packet = build_pilot_checkpoint(identity=IDENTITY,path='continuous',step=0,worker_generation=0,
        model_state=m.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
        parameter_names=[[f'encoder.{n}' for n,_ in m.encoder.named_parameters()],[f'head.{n}' for n,_ in m.head.named_parameters()]],
        config=expected['expected_config'],ordered_masks=MASKS,rank_rng_states={rank:dict(host=host,xla_seed=75+rank) for rank in range(8)})
    path = tmp_path/'initial.pt'
    sha = publish_pilot_checkpoint(packet,path,expected_identity=IDENTITY,ordered_masks=MASKS,byte_limit=100_000_000)
    with pytest.raises(ValueError,match='final step16'):
        derive_model_export(path,checkpoint_sha256=sha,expected_identity=IDENTITY,ordered_masks=MASKS,
            expected_config=expected['expected_config'],provenance=PROVENANCE)
