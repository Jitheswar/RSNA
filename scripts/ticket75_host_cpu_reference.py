"""Original CPU loss/Adam reference for a future admitted offline pilot worker.

No CLI, device acquisition, source upload or resource launch. The surrounding
worker must verify original attachment/source/admission before invoking this.
"""
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import platform
import time

import torch

from ticket75_xla_update import original_adam,original_scheduler,validate_optimizer,finite_training_flag,require
from rsna_knee.full_backbone import ResNet18FullBackboneStudyModel
from rsna_knee.replicated_masked_loss import admit_global_supervised_entries
from rsna_knee.tpu_pilot_export import replay_export_logits
from rsna_knee.tpu_pilot_state import capture_host_rng,restore_host_rng,encode_host_rng
from rsna_knee.training import masked_bce_loss,_model_state_sha256

MODEL_FIELDS = ('images','slice_mask','series_mask','series_descriptors')


def _validate_batch(model,batch):
    h,w = model.config.image_size
    shapes = dict(images=(4,16,8,1,h,w),slice_mask=(4,16,8),series_mask=(4,16),series_descriptors=(4,16,10),
        targets=(4,12),supervision_mask=(4,12))
    require(set(batch) == set(shapes), 'All four original source draws and input fields required')
    for name,shape in shapes.items():
        value = batch[name]
        require(type(value) is torch.Tensor and value.device.type == 'cpu' and tuple(value.shape) == shape
            and value.dtype == (torch.bool if name in ('slice_mask','series_mask','supervision_mask') else torch.float32),
            'Exact original CPU batch geometry/dtype required')
    require(bool(torch.isfinite(batch['images']).all()) and bool(torch.isfinite(batch['series_descriptors']).all())
        and bool((batch['slice_mask'] & batch['series_mask'].unsqueeze(-1)).flatten(1).any(1).all())
        and bool(batch['supervision_mask'].any(1).all()), 'All real draws require finite valid MRI and original nonempty supervision')
    admit_global_supervised_entries(batch['targets'],batch['supervision_mask'])


def _run_three_cpu_updates(model,initial_payload,batches,replay_inputs):
    """CPU mechanics seam; synthetic tests do not assert actual CPU2.8 execution."""
    started = time.monotonic()
    require(type(model) is ResNet18FullBackboneStudyModel and len(batches) == 3
        and all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.requires_grad for p in model.parameters()),
        'Original complete CPU full-backbone model and first three global batches required')
    require(model.config.to_dict() == initial_payload['config']['model'] and initial_payload['disposable'] is True,
        'Exact disposable initial reference configuration required')
    for batch in batches:
        _validate_batch(model,batch)
    current,state = model.state_dict(),initial_payload['model_state_dict']
    require(current.keys() == state.keys() and getattr(current,'_metadata',None) == getattr(state,'_metadata',None)
        and all(state[name].device.type == 'cpu' and state[name].shape == current[name].shape
            and state[name].dtype == current[name].dtype for name in current), 'Complete original initial state required')
    model.load_state_dict(state,strict=True)
    require(_model_state_sha256(model) == initial_payload['initial_model_state_sha256'], 'Original initialized-model tensor hash changed')
    optimizer = original_adam(model);scheduler = original_scheduler(optimizer)
    normalization = {name:value.clone() for name,value in model.named_buffers()
        if name.endswith(('running_mean','running_var','num_batches_tracked'))}
    initial_logits = replay_export_logits(model,replay_inputs)
    # Fresh model/optimizer construction and initial replay must not shift the
    # original first-update host RNG position.
    restore_host_rng(encode_host_rng(initial_payload['rng_state']))
    gradients = [];losses = [];intervals = []
    for step,batch in enumerate(batches,start=1):
        step_started = time.monotonic()
        validate_optimizer(model,optimizer,scheduler,step-1)
        model.train();optimizer.zero_grad(set_to_none=True)
        logits = model(*(batch[name] for name in MODEL_FIELDS))
        loss = masked_bce_loss(logits,batch['targets'],batch['supervision_mask'])
        require(loss is not None and bool(torch.isfinite(loss)), 'Original combined-entry masked-index CPU BCE must be finite')
        loss.backward()
        require(bool(finite_training_flag(model,optimizer,loss,normalization)), 'CPU loss/gradient/state must be finite before update')
        gradients.append({name:p.grad.detach().clone() for name,p in model.named_parameters()})
        optimizer.step();scheduler.step()
        require(bool(finite_training_flag(model,optimizer,loss,normalization)), 'CPU model/Adam/normalization must pass after update')
        validate_optimizer(model,optimizer,scheduler,step)
        losses.append(loss.detach().clone());intervals.append(time.monotonic()-step_started)
    final_logits = replay_export_logits(model,replay_inputs)
    return dict(initial_logits=initial_logits,
        first_three_state=dict(model_state_dict=deepcopy(model.state_dict()),optimizer_state_dict=deepcopy(optimizer.state_dict()),
            reduced_gradients=gradients,losses=torch.stack(losses),logits=final_logits,successful_steps=3),
        scheduler_state_dict=deepcopy(scheduler.state_dict()),final_host_rng=capture_host_rng(),
        elapsed_seconds=time.monotonic()-started,update_intervals_seconds=intervals,
        observed_runtime=dict(python=platform.python_version(),torch=importlib.metadata.version('torch'),
            numpy=importlib.metadata.version('numpy')),
        pinned_CPU28_host_runtime_checked=False,original_MRI_identity_verified=False,
        actual_TPU_execution_proven=False,competition_submission=False,candidate_promotion=False)


def run_pinned_cpu28_reference(model,initial_payload,batches,replay_inputs,*,expected_source_hashes):
    """Invoke only after actual resource/attachment admission, before PJRT starts."""
    require(importlib.metadata.version('torch') == '2.8.0+cpu' and importlib.metadata.version('numpy') == '2.5.0'
        and platform.python_version() == '3.12.13' and torch.version.cuda is None and not torch.cuda.is_available()
        and torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled('cpu'),
        'Exact original offline CPU2.8 runtime/FP32 required before any reference fitting')
    require(set(expected_source_hashes) == {'torch.optim.optimizer','torch.optim.adam'}, 'Pinned optimizer and ordinary Adam installed source hashes required')
    for name,expected in expected_source_hashes.items():
        module = importlib.import_module(name)
        with open(module.__file__,'rb') as stream:
            require(hashlib.file_digest(stream,'sha256').hexdigest() == expected, 'Installed CPU reference optimizer source changed')
    require(model.config.image_size == (224,224), 'Original real-MRI224 reference configuration required')
    prior_threads = torch.get_num_threads();torch.set_num_threads(16)
    try:
        result = _run_three_cpu_updates(model,initial_payload,batches,replay_inputs)
        require(result['elapsed_seconds'] <= 180, 'Fixed180-second host reference phase exhausted; no retry or reduction')
        result.update(pinned_CPU28_host_runtime_checked=True,CPU_threads=16,
            installed_optimizer_source_hashes=dict(expected_source_hashes))
        return result
    finally:
        torch.set_num_threads(prior_threads)
