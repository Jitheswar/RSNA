"""One original-science replicated update inside a future admitted PJRT worker.

This is not a launcher: no device is acquired at import, there is no CLI, and
actual source/input/resource admission and fresh-worker supervision are separate.
"""
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import os
import platform
import time

import torch

from rsna_knee.full_backbone import ResNet18FullBackboneStudyModel
from rsna_knee.replicated_masked_loss import replica_masked_bce
from rsna_knee.tpu_pilot_telemetry import PilotUpdateObservation


def require(value, message):
    if not value:
        raise ValueError(message)


def original_adam(model):
    require(type(model) is ResNet18FullBackboneStudyModel, 'Original full-backbone model required')
    require(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()), 'All original FP32 encoder/head parameters must train')
    return torch.optim.Adam([dict(params=model.encoder.parameters(), lr=1e-5), dict(params=model.head.parameters(), lr=1e-4)], lr=1e-4)


def original_scheduler(optimizer):
    require(type(optimizer) is torch.optim.Adam, 'Ordinary original Adam required')
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def validate_optimizer(model, optimizer, scheduler, before_step):
    require(type(model) is ResNet18FullBackboneStudyModel and type(optimizer) is torch.optim.Adam
        and type(scheduler) is torch.optim.lr_scheduler.LambdaLR and scheduler.optimizer is optimizer,
        'Original full model, ordinary Adam and constant scheduler required')
    expected = (tuple(model.encoder.parameters()), tuple(model.head.parameters()))
    require(len(optimizer.param_groups) == 2 and scheduler.last_epoch == before_step, 'Exactly two original groups and scheduler position required')
    for group, parameters, lr in zip(optimizer.param_groups, expected, (1e-5,1e-4), strict=True):
        require(tuple(map(id, group['params'])) == tuple(map(id, parameters)) and group['lr'] == lr
            and group['betas'] == (.9,.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
            and group['amsgrad'] is False and group['maximize'] is False and group['capturable'] is False
            and group['differentiable'] is False and group['foreach'] is None and group['fused'] is None,
            'Exact original parameter order and ordinary Adam settings required')
    for parameter in model.parameters():
        require(parameter.requires_grad and parameter.dtype == torch.float32, 'Original full FP32 parameter scope required')
        state = optimizer.state.get(parameter)
        if before_step == 0:
            require(not state, 'Original initial Adam cannot contain learned state')
        else:
            require(state is not None and set(state) == {'step','exp_avg','exp_avg_sq'}
                and state['step'].device.type == 'cpu' and state['step'].numel() == 1
                and float(state['step']) == before_step
                and state['exp_avg'].shape == state['exp_avg_sq'].shape == parameter.shape
                and state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32,
                'Complete original Adam moment shapes and exact CPU step scalar required')


def metrics_snapshot(metrics):
    result = dict(metrics={}, counters={})
    for name in metrics.metric_names():
        value = metrics.metric_data(name)
        require(value is not None and len(value) == 3 and type(value[0]) is int and value[0] >= 0,
                'Actual complete registered metric required')
        result['metrics'][name] = dict(total_samples=value[0], accumulator=value[1], samples=value[2])
    for name in metrics.counter_names():
        value = metrics.counter_value(name)
        require(type(value) is int and value >= 0, 'Actual complete registered counter required')
        result['counters'][name] = value
    result['executed_fallback_ops'] = list(metrics.executed_fallback_ops())
    return result


def metric_delta(before, after, names):
    available = [name for name in names if name in after['metrics']]
    if not available:
        return None
    count = sum(after['metrics'][name]['total_samples'] - before['metrics'].get(name, {}).get('total_samples',0) for name in available)
    require(count >= 0, 'Backend metric reset or backwards sample count is not a valid phase')
    return count


def fallback_deltas(before, after):
    names = sorted({n for r in (before,after) for n in r['counters'] if n.startswith('aten::')})
    deltas = tuple((n,after['counters'].get(n,0)-before['counters'].get(n,0)) for n in names)
    require(all(count >= 0 for _,count in deltas), 'Backend fallback counter reset is not valid phase evidence')
    # A new fallback op with no named counter is also an unsupported observation;
    # do not let a counter-only gate silently omit that operation.
    new_ops = set(after['executed_fallback_ops']) - set(before['executed_fallback_ops'])
    require(all(name in after['counters'] and after['counters'][name] > before['counters'].get(name,0) for name in new_ops),
            'Fallback op lacked an observed counter; preserve failure')
    return deltas


class XlaPilotUpdate:
    """Construct only inside an already admitted fresh eight-device worker."""
    def __init__(self, *, admitted_worker, expected_source_hashes):
        require(admitted_worker is True, 'Actual worker admission must precede backend initialization')
        require(expected_source_hashes and set(expected_source_hashes) == {
            'torch_xla.core.xla_model', 'torch_xla.debug.metrics', 'torch_xla.backends', 'torch_xla.torch_xla'},
            'All exact pinned backend API source hashes required')
        for name in ('XLA_USE_BF16','XLA_DOWNCAST_BF16','XLA_USE_FP16','XLA_DOWNCAST_FP16'):
            require(os.environ.get(name) in (None,'','0'), 'Reduced-precision environment fails the original FP32 pilot')
        # Deferred import: unit tests and CPU preparation do not acquire devices.
        import torch_xla
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        import torch_xla.backends as backends
        require(importlib.metadata.version('torch') == '2.8.0+cpu'
            and importlib.metadata.version('torch_xla') == '2.8.0'
            and importlib.metadata.version('numpy') == '2.5.0' and importlib.metadata.version('libtpu') == '0.0.17'
            and platform.python_version() == '3.12.13' and torch.version.cuda is None
            and not torch.cuda.is_available() and not xr.is_spmd(), 'Exact offline CPU-wheel replicated runtime required')
        for name, expected in expected_source_hashes.items():
            module = importlib.import_module(name)
            with open(module.__file__, 'rb') as stream:
                require(hashlib.file_digest(stream,'sha256').hexdigest() == expected, 'Installed backend API source changed')
        backends.set_mat_mul_precision('highest')
        require(backends.get_mat_mul_precision() == 'highest', 'Actual highest backend precision required')
        self.device = torch_xla.device()
        require(xr.device_type() == 'TPU' and xr.world_size() == xr.global_device_count() == 8,
                'Actual eight-rank TPU topology required')
        self.rank = xr.global_ordinal()
        self.xla, self.xm, self.metrics = torch_xla, xm, metrics
        self.normalization = None

    def graph_hash(self, tensors):
        value = self.xla._XLAC._get_graph_hash(tensors)
        require(isinstance(value, bytes) and len(value) in (16,32), 'Actual pinned phase graph hash required')
        return value.hex()

    def update(self, model, optimizer, scheduler, replica, *, path, successful_step, worker_generation):
        require(path in ('continuous','staged') and type(successful_step) is int and successful_step in range(1,17)
            and worker_generation == int(path == 'staged' and successful_step >= 9), 'Exact declared pilot path/step/generation required')
        require(replica.rank == self.rank and next(model.parameters()).device == self.device,
                'Original input rank and model device must match actual worker')
        validate_optimizer(model,optimizer,scheduler,successful_step-1)
        model.train()
        require(all(not m.training for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d)),
                'Original encoder normalization must remain fixed')
        if self.normalization is None:
            self.normalization = {n:v.detach().clone() for n,v in model.named_buffers()
                if n.endswith(('running_mean','running_var','num_batches_tracked'))}
        self.xla.sync(wait=True)
        before = metrics_snapshot(self.metrics)
        started = time.monotonic()
        inputs = tuple(v.to(self.device) for v in replica.model_inputs)
        targets = replica.targets.to(self.device)
        mask = replica.supervision_mask.to(self.device)
        denominator = replica.global_supervised_entries.to(self.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = replica_masked_bce(logits,targets,mask,denominator)
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        require(all(g is not None and g.dtype == torch.float32 for g in gradients), 'Every original parameter requires an FP32 gradient on every rank')
        global_loss = self.xm.all_reduce(self.xm.REDUCE_SUM,loss.detach(),scale=1/8)
        forward_graph = self.graph_hash([global_loss,*gradients])
        # Exactly one mean collective, then ordinary Adam. Calling optimizer_step
        # again would reduce twice; this is its pinned ordinary-update body.
        self.xm.reduce_gradients(optimizer)
        optimizer.step()
        scheduler.step()
        state_tensors = [v for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor) and v.device.type == 'xla']
        optimizer_graph = self.graph_hash([*model.parameters(),*state_tensors])
        self.xla.sync(wait=True)
        trained = metrics_snapshot(self.metrics)
        training_seconds = time.monotonic()-started
        flags = [torch.isfinite(global_loss),*(torch.isfinite(v).all() for v in gradients),
            *(torch.isfinite(v).all() for v in model.parameters()),*(torch.isfinite(v).all() for v in state_tensors)]
        flags.extend((state['exp_avg_sq'] >= 0).all() for state in optimizer.state.values())
        flags.extend((dict(model.named_buffers())[name] == value).all() for name,value in self.normalization.items())
        finite = torch.stack(flags).all().to(dtype=torch.float32)
        all_finite = self.xm.all_reduce(self.xm.REDUCE_SUM,finite)
        self.xla.sync(wait=True)
        require(float(all_finite.cpu()) == 8, 'Nonfinite model/gradient/Adam or changed normalization fails the pilot')
        value = float(global_loss.cpu())
        exported_gradients = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()} if successful_step <= 3 else None
        guarded = metrics_snapshot(self.metrics)
        observation = PilotUpdateObservation(path,successful_step,self.rank,worker_generation,time.monotonic()-started,
            metric_delta(before,trained,('CompileTime',)), metric_delta(before,trained,('ExecuteTime','ExecuteReplicatedTime')),
            forward_graph,optimizer_graph,True,fallback_deltas(before,trained))
        require(not any(count > 0 for _,count in observation.fallback_counter_deltas), 'Unsupported training CPU fallback fails the pilot')
        validate_optimizer(model,optimizer,scheduler,successful_step)
        return dict(observation=asdict(observation), global_loss=value, training_seconds=training_seconds,
            finite_guard_and_prescribed_gradient_export_seconds=observation.synchronized_interval_seconds-training_seconds,
            metrics_before=before,metrics_after_training=trained,metrics_after_guard=guarded,
            reduced_gradients=exported_gradients, actual_complete_pilot_verified=False,
            competition_submission=False,candidate_promotion=False)
