"""Original-architecture synthetic CPU mechanics; no CPU2.8/runtime proof."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec = importlib.util.spec_from_file_location('ticket75_host_cpu_reference',SCRIPTS/'ticket75_host_cpu_reference.py')
host = importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
from rsna_knee.model import ModelConfig,build_study_model
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID,RESNET18_FULL_FINETUNE_MODEL_ID,ANATOMICAL_POOLING_ID
from rsna_knee.tpu_pilot_parity import compare_fixed_first_three_updates
from rsna_knee.training import _model_state_sha256


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model():
    return build_study_model(ModelConfig(image_size=(32,32),model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID,pooling_id=ANATOMICAL_POOLING_ID))


def initial(m):
    return dict(model_state_dict=deepcopy(m.state_dict()),config=dict(model=m.config.to_dict()),disposable=True,
        initial_model_state_sha256=_model_state_sha256(m),rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()))


def inputs(count):
    result = dict(images=torch.rand(count,16,8,1,32,32),slice_mask=torch.zeros(count,16,8,dtype=torch.bool),
        series_mask=torch.zeros(count,16,dtype=torch.bool),series_descriptors=torch.zeros(count,16,10))
    result['slice_mask'][:,0,0] = True;result['series_mask'][:,0] = True;result['series_descriptors'][:,0,0] = 1
    return result


def batches():
    result = []
    for _ in range(3):
        batch = inputs(4);batch['targets'] = torch.zeros(4,12)
        batch['supervision_mask'] = torch.ones(4,12,dtype=torch.bool)
        batch['supervision_mask'][2,3:] = False;batch['supervision_mask'][3,8:] = False
        batch['targets'][~batch['supervision_mask']] = float('nan')
        result.append(batch)
    return result


def test_original_masked_index_loss_three_updates_produce_complete_comparable_reference():
    torch.manual_seed(75);random.seed(75);np.random.seed(75)
    m = model();source = initial(m);values = batches();replay = inputs(8)
    source_before = {k:v.clone() for k,v in source['model_state_dict'].items()}
    result = host._run_three_cpu_updates(m,source,values,replay)
    state = result['first_three_state']
    names = [[f'encoder.{n}' for n,_ in m.encoder.named_parameters()],[f'head.{n}' for n,_ in m.head.named_parameters()]]
    report = compare_fixed_first_three_updates(state,state,expected_model_state=source['model_state_dict'],parameter_names=names)
    assert report['passed'] and state['losses'].shape == (3,)
    assert result['initial_logits'].shape == (8,12) and result['scheduler_state_dict']['last_epoch'] == 3
    assert len(result['update_intervals_seconds']) == 3 and all(v > 0 for v in result['update_intervals_seconds'])
    assert result['pinned_CPU28_host_runtime_checked'] is result['original_MRI_identity_verified'] is result['actual_TPU_execution_proven'] is result['competition_submission'] is False
    assert result['observed_runtime']['torch'] == host.importlib.metadata.version('torch')
    for name,tensor in source['model_state_dict'].items(): torch.testing.assert_close(tensor,source_before[name],rtol=0,atol=0)
    assert any(not torch.equal(m.state_dict()[name],tensor) for name,tensor in source['model_state_dict'].items())
    assert all(torch.equal(m.state_dict()[name],tensor) for name,tensor in source['model_state_dict'].items()
        if name.endswith(('running_mean','running_var','num_batches_tracked')))


@pytest.mark.parametrize('defect',['few_batches','bad_geometry','empty_draw','bad_target','wrong_dtype','bad_descriptor','wrong_config'])
def test_invalid_reference_inputs_fail_before_any_model_load_or_optimizer_update(defect):
    m = model();source = initial(m);values = batches()
    if defect == 'few_batches': values.pop()
    if defect == 'bad_geometry': values[0]['images'] = values[0]['images'][:3]
    if defect == 'empty_draw': values[0]['supervision_mask'][3] = False
    if defect == 'bad_target': values[0]['targets'][0,0] = float('inf')
    if defect == 'wrong_dtype': values[0]['slice_mask'] = values[0]['slice_mask'].float()
    if defect == 'bad_descriptor': values[0]['series_descriptors'][0,0,0] = float('nan')
    if defect == 'wrong_config': source['config']['model']['head_hidden_size'] += 1
    before = {k:v.clone() for k,v in m.state_dict().items()}
    with pytest.raises(ValueError): host._run_three_cpu_updates(m,source,values,inputs(8))
    for name,tensor in m.state_dict().items(): torch.testing.assert_close(tensor,before[name],rtol=0,atol=0)


def test_unpinned_local_runtime_rejected_before_any_reference_fit():
    if host.importlib.metadata.version('torch') == '2.8.0+cpu': pytest.skip('This check requires the actual original CPU2.7 test environment')
    with pytest.raises(ValueError,match='before any reference fitting'):
        host.run_pinned_cpu28_reference(None,None,None,None,expected_source_hashes={})
