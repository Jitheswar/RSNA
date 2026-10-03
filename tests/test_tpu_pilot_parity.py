"""Synthetic CPU comparisons exercise gates, never establish backend proof."""
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch

from rsna_knee.tpu_pilot_parity import PilotParityFailure,compare_cpu_replay_logits,compare_cross_backend_state


def state():
    return dict(model=OrderedDict(weight=torch.tensor([0.,1.,-1.,100.]),counter=torch.tensor(3,dtype=torch.int64)),
        adam=dict(step=3,exp_avg=torch.ones(4),exp_avg_sq=torch.ones(4)*.01),
        gradients=dict(weight=torch.tensor([1.,0.,2.,-1.])),logits=torch.zeros(8,12),loss=torch.tensor(.5),
        config=dict(lr=1e-5,betas=(.9,.999),next_batch_index=3))


def test_fixed_reference_relative_tolerance_and_zero_absolute_floor():
    reference = state();actual = deepcopy(reference)
    actual['model']['weight'] += torch.tensor([.9e-6, .9e-4,-.9e-4,.009])
    report = compare_cross_backend_state(reference,actual)
    assert report['passed'] and report['rtol'] == 1e-4 and report['atol'] == 1e-6
    assert report['tensor_count'] == 7 and report['finite_support'] == report['elements']
    assert report['actual_TPU_execution_proven'] is report['actual_CPU28_runtime_verified'] is report['competition_submission'] is False
    actual['model']['weight'][0] = 1.1e-6
    with pytest.raises(PilotParityFailure) as failure: compare_cross_backend_state(reference,actual)
    row = next(row for row in failure.value.report['tensors'] if row['name'] == 'state.model.weight')
    assert row['mismatched_elements'] == 1 and row['zero_reference_elements'] == 1
    assert row['maximum_absolute_error'] >= .008


def test_failure_report_keeps_all_tensor_errors_and_finite_support():
    reference = state();actual = deepcopy(reference)
    actual['model']['weight'][1] = 2
    actual['gradients']['weight'][3] = float('nan')
    actual['logits'][7,11] = float('inf')
    with pytest.raises(PilotParityFailure) as failure: compare_cross_backend_state(reference,actual)
    report = failure.value.report
    assert report['failed_tensors'] == 3 and report['tensor_count'] == 7
    assert report['finite_support'] == report['elements'] - 2
    assert report['passed'] is False and report['candidate_promotion'] is False


@pytest.mark.parametrize('defect',['missing_gradient','moment_shape','dtype','non_cpu','metadata','config','progress','integer_counter','sequence','empty','unknown_object'])
def test_exact_structure_scope_metadata_and_nonfloating_state_cannot_drift(defect):
    reference = state();actual = deepcopy(reference)
    if defect == 'missing_gradient': actual['gradients'].pop('weight')
    if defect == 'moment_shape': actual['adam']['exp_avg'] = torch.zeros(3)
    if defect == 'dtype': actual['model']['weight'] = actual['model']['weight'].double()
    if defect == 'non_cpu': actual['model']['weight'] = actual['model']['weight'].to('meta')
    if defect == 'metadata': actual['model']._metadata = dict(version=1)
    if defect == 'config': actual['config']['lr'] = 1.000001e-5
    if defect == 'progress': actual['adam']['step'] = 4
    if defect == 'integer_counter': actual['model']['counter'] += 1
    if defect == 'sequence': actual['config']['betas'] = list(actual['config']['betas'])
    if defect == 'empty': reference = {};actual = {}
    if defect == 'unknown_object': reference['bad'] = actual['bad'] = object()
    with pytest.raises(ValueError): compare_cross_backend_state(reference,actual)


def test_cpu_replay_uses_absolute_only_and_keeps_rank7_target_failure():
    reference = torch.full((8,12),100.)
    actual = reference.clone();actual[7,11] += .001
    # This difference meets cross-backend relative tolerance but must fail CPU replay.
    assert compare_cross_backend_state(dict(logits=reference),dict(logits=actual))['passed']
    with pytest.raises(PilotParityFailure) as failure: compare_cpu_replay_logits(reference,actual)
    assert failure.value.report['tensors'][0]['mismatched_elements'] == 1
    assert compare_cpu_replay_logits(torch.zeros(8,12),torch.full((8,12),.9e-6))['passed']
    with pytest.raises(PilotParityFailure): compare_cpu_replay_logits(torch.zeros(8,12),torch.full((8,12),1.1e-6))


def test_near_maximum_fp32_errors_do_not_overflow_diagnostic_arithmetic():
    reference = torch.full((1,),torch.finfo(torch.float32).max)
    actual = -reference
    with pytest.raises(PilotParityFailure) as failure:
        compare_cross_backend_state(dict(value=reference),dict(value=actual))
    row = failure.value.report['tensors'][0]
    assert row['finite_support'] == 1 and row['maximum_relative_error_on_nonzero_reference'] == 2
    assert row['maximum_absolute_error'] < float('inf')


def three_update_state():
    m = torch.nn.Module();m.encoder = torch.nn.Linear(2,3);m.head = torch.nn.Linear(3,12)
    initial = deepcopy(m.state_dict())
    names = [[f'encoder.{name}' for name,_ in m.encoder.named_parameters()],[f'head.{name}' for name,_ in m.head.named_parameters()]]
    optimizer = torch.optim.Adam([dict(params=m.encoder.parameters(),lr=1e-5),dict(params=m.head.parameters(),lr=1e-4)])
    gradients = [];losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m.head(m.encoder(torch.randn(4,2))),torch.ones(4,12))
        loss.backward();gradients.append({name:p.grad.clone() for name,p in m.named_parameters()});losses.append(loss.detach())
        optimizer.step()
    return dict(model_state_dict=m.state_dict(),optimizer_state_dict=optimizer.state_dict(),reduced_gradients=gradients,
        losses=torch.stack(losses),logits=torch.zeros(8,12),successful_steps=3),initial,names


def test_complete_first_three_updates_require_every_original_gradient_and_moment():
    from rsna_knee.tpu_pilot_parity import compare_fixed_first_three_updates
    reference,initial,names = three_update_state()
    report = compare_fixed_first_three_updates(reference,deepcopy(reference),expected_model_state=initial,parameter_names=names)
    assert report['passed'] and report['actual_CPU28_runtime_verified'] is False
    assert all(any(row['name'] == f'state.reduced_gradients.{step}.encoder.weight' for row in report['tensors']) for step in range(3))


@pytest.mark.parametrize('defect',['missing_early_gradient','missing_gradient_both','missing_adam_both','wrong_model_shape','wrong_loss_count','loss_nan',
    'missing_logit','wrong_step','wrong_adam_scalar','negative_moment','wrong_lr_both','extra_tensor','missing_buffer_both'])
def test_fixed_first_three_gate_cannot_pass_mutually_incomplete_or_changed_records(defect):
    from rsna_knee.tpu_pilot_parity import compare_fixed_first_three_updates
    reference,initial,names = three_update_state();actual = deepcopy(reference)
    if defect == 'missing_early_gradient': actual['reduced_gradients'].pop(0)
    if defect == 'missing_gradient_both':
        reference['reduced_gradients'][0].pop('encoder.weight');actual['reduced_gradients'][0].pop('encoder.weight')
    if defect == 'missing_adam_both':
        reference['optimizer_state_dict']['state'].pop(0);actual['optimizer_state_dict']['state'].pop(0)
    if defect == 'wrong_model_shape': actual['model_state_dict']['encoder.weight'] = torch.zeros(3,1)
    if defect == 'wrong_loss_count': actual['losses'] = actual['losses'][:2]
    if defect == 'loss_nan': actual['losses'][0] = float('nan')
    if defect == 'missing_logit': actual['logits'] = actual['logits'][:7]
    if defect == 'wrong_step': actual['successful_steps'] = 2
    if defect == 'wrong_adam_scalar': actual['optimizer_state_dict']['state'][0]['step'] += 1e-5
    if defect == 'negative_moment': actual['optimizer_state_dict']['state'][0]['exp_avg_sq'].flatten()[0] = -1
    if defect == 'wrong_lr_both':
        reference['optimizer_state_dict']['param_groups'][0]['lr'] = actual['optimizer_state_dict']['param_groups'][0]['lr'] = 2e-5
    if defect == 'extra_tensor': actual['unregistered'] = torch.ones(1)
    if defect == 'missing_buffer_both':
        initial['imagenet_mean'] = torch.zeros(3,1,1)
    with pytest.raises(ValueError): compare_fixed_first_three_updates(reference,actual,expected_model_state=initial,parameter_names=names)


def test_failure_report_includes_fixed_tolerances_for_preserved_evidence():
    with pytest.raises(PilotParityFailure) as failure: compare_cpu_replay_logits(torch.zeros(8,12),torch.ones(8,12))
    assert failure.value.report['rtol'] == 0 and failure.value.report['atol'] == 1e-6


def test_fixed_first_three_gate_accepts_complete_original_full_backbone_inventory():
    from rsna_knee.model import ModelConfig,build_study_model
    from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID,RESNET18_FULL_FINETUNE_MODEL_ID,ANATOMICAL_POOLING_ID
    from rsna_knee.tpu_pilot_parity import compare_fixed_first_three_updates
    threads = torch.get_num_threads();torch.set_num_threads(1)
    try:
        m = build_study_model(ModelConfig(image_size=(32,32),model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
            encoder_id=RESNET18_FIXTURE_ENCODER_ID,pooling_id=ANATOMICAL_POOLING_ID))
        initial = deepcopy(m.state_dict())
        names = [[f'encoder.{name}' for name,_ in m.encoder.named_parameters()],[f'head.{name}' for name,_ in m.head.named_parameters()]]
        optimizer = torch.optim.Adam([dict(params=m.encoder.parameters(),lr=1e-5),dict(params=m.head.parameters(),lr=1e-4)])
        gradients = []
        for _ in range(3):
            for p in m.parameters(): p.grad = torch.full_like(p,.001)
            gradients.append({name:p.grad.clone() for name,p in m.named_parameters()})
            optimizer.step()
        # Original architecture/Adam, synthetic gradients/losses/logits only.
        value = dict(model_state_dict=m.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            reduced_gradients=gradients,losses=torch.ones(3),logits=torch.zeros(8,12),successful_steps=3)
        result = compare_fixed_first_three_updates(value,value,expected_model_state=initial,parameter_names=names)
        assert result['passed'] and result['elements'] > 10_000_000
        assert result['actual_TPU_execution_proven'] is result['original_MRI_identity_verified'] is False
    finally:
        torch.set_num_threads(threads)
