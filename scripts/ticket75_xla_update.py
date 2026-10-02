"""One original-science replicated update inside a future admitted PJRT worker.

This is not a launcher: no device is acquired at import, there is no CLI, and
actual source/input/resource admission and fresh-worker supervision are separate.
"""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import os
import platform
import time

import torch

from rsna_knee.full_backbone import ResNet18FullBackboneStudyModel
from rsna_knee.replicated_masked_loss import replica_masked_bce
from rsna_knee.tpu_pilot_checkpoint import validate_pilot_checkpoint
from rsna_knee.tpu_pilot_state import restore_host_rng
from rsna_knee.tpu_pilot_telemetry import PilotUpdateObservation, validate_rank_update


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
            and group.get('decoupled_weight_decay', False) is False
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
                and state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
                and state['exp_avg'].device == state['exp_avg_sq'].device == parameter.device,
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


def finite_training_flag(model, optimizer, global_loss, normalization):
    """Device tensor only: inspect reduced gradients and state before mutation."""
    parameters = tuple(model.parameters())
    require(all(p.grad is not None and p.grad.dtype == torch.float32 for p in parameters),
        'Every original parameter requires an FP32 gradient on every rank')
    flags = [torch.isfinite(global_loss),*(torch.isfinite(p.grad).all() for p in parameters),
        *(torch.isfinite(p).all() for p in parameters)]
    for state in optimizer.state.values():
        flags.extend((torch.isfinite(state['exp_avg']).all(),torch.isfinite(state['exp_avg_sq']).all(),
            (state['exp_avg_sq'] >= 0).all()))
    buffers = dict(model.named_buffers())
    flags.extend((buffers[name] == value).all() for name,value in normalization.items())
    return torch.stack(flags).all().to(dtype=torch.float32)


def training_phase_deltas(phases):
    """Combine training intervals while retaining separate intentional guards."""
    result = {}
    for label,names in (('compile',('CompileTime',)),('execute',('ExecuteTime','ExecuteReplicatedTime'))):
        counts = [metric_delta(before,after,names) for before,after in phases]
        result[label] = None if any(count is None for count in counts) else sum(counts)
    counters = {}
    for before,after in phases:
        for name,count in fallback_deltas(before,after):
            counters[name] = counters.get(name,0) + count
    result['fallbacks'] = tuple(sorted(counters.items()))
    return result


class XlaPilotUpdate:
    """Construct only inside an already admitted fresh eight-device worker."""
    def __init__(self, *, admitted_worker, expected_source_hashes):
        require(admitted_worker is True, 'Actual worker admission must precede backend initialization')
        require(expected_source_hashes and set(expected_source_hashes) == {
            'torch_xla.core.xla_model', 'torch_xla.debug.metrics', 'torch_xla.backends', 'torch_xla.torch_xla', 'torch.optim.optimizer'},
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
        self.initialized_baseline = None
        self.worker_identity = None
        self.next_step = None
        self.failed_or_inflight = False
        self.last_update_evidence = None
        self.checkpoint_restore_consumed = False
        self.restored_checkpoint = None

    def graph_hash(self, tensors):
        value = self.xla._XLAC._get_graph_hash(tensors)
        require(isinstance(value, bytes) and len(value) in (16,32), 'Actual pinned phase graph hash required')
        return value.hex()

    def restore_staged_checkpoint(self, model, packet, *, expected_identity, ordered_masks, expected_config):
        """Use only in the newly admitted generation1 worker, before any update."""
        require(not self.checkpoint_restore_consumed and self.worker_identity is None and self.next_step is None
            and self.initialized_baseline is None and not self.failed_or_inflight,
            'Fresh unused worker required; never restore twice or reuse a trained runtime')
        validate_pilot_checkpoint(packet,expected_identity=expected_identity,ordered_masks=ordered_masks)
        require(packet['path'] == 'staged' and packet['successful_steps'] == 8 and packet['worker_generation'] == 0,
            'Only the original staged step8 checkpoint can enter fresh generation1')
        require(type(model) is ResNet18FullBackboneStudyModel and packet['config'] == expected_config
            and model.config.to_dict() == expected_config['model']
            and all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.requires_grad for p in model.parameters()),
            'Exact registered config and original fresh CPU full-backbone model required')
        current = model.state_dict()
        restored = packet['model_state_dict']
        names = [[f'encoder.{name}' for name,_ in model.encoder.named_parameters()],
            [f'head.{name}' for name,_ in model.head.named_parameters()]]
        require(names == packet['parameter_names'] and current.keys() == restored.keys()
            and getattr(current,'_metadata',None) == getattr(restored,'_metadata',None)
            and all(current[name].shape == restored[name].shape and current[name].dtype == restored[name].dtype for name in current),
            'Original complete model architecture, buffers, metadata and parameter order required before mutation')
        # Everything is validated before consuming the only restore. A failure
        # during device transfer/state loading remains consumed, with no retry.
        self.checkpoint_restore_consumed = True
        model.load_state_dict(restored,strict=True)
        model.to(self.device)
        # Construct groups after device transfer so the optimizer cannot retain
        # stale CPU Parameter objects. The pinned loader casts moments to their
        # corresponding parameters while ordinary Adam's step stays on CPU.
        optimizer = original_adam(model)
        scheduler = original_scheduler(optimizer)
        optimizer.load_state_dict(deepcopy(packet['optimizer_state_dict']))
        scheduler.load_state_dict(deepcopy(packet['scheduler_state_dict']))
        validate_optimizer(model,optimizer,scheduler,8)
        model.train()
        self.normalization = {name:value.detach().clone() for name,value in model.named_buffers()
            if name.endswith(('running_mean','running_var','num_batches_tracked'))}
        self.xla.sync(wait=True)
        # Restore RNG only after model construction, transfer, state loading and
        # normalization initialization have completed. Queue no further work.
        rng = packet['rank_rng_states'][self.rank]
        self.xm.set_rng_state(rng['xla_seed'],device=self.device)
        require(self.xm.get_rng_state(device=self.device) == rng['xla_seed'], 'Actual device RNG restore failed')
        restore_host_rng(rng['host'])
        self.restored_checkpoint = dict(identity=dict(expected_identity),rank=self.rank,successful_steps=8)
        return model,optimizer,scheduler,dict(restored_successful_steps=8,next_batch_index=8,rank=self.rank,
            fresh_worker_generation=1,actual_complete_pilot_verified=False,competition_submission=False,candidate_promotion=False)

    def update(self, model, optimizer, scheduler, replica, *, path, successful_step, worker_generation):
        require(path in ('continuous','staged') and type(successful_step) is int and successful_step in range(1,17)
            and type(worker_generation) is int and worker_generation == int(path == 'staged' and successful_step >= 9), 'Exact declared pilot path/step/generation required')
        require(not self.failed_or_inflight, 'Consumed failed or in-flight update cannot be retried')
        start = 9 if worker_generation else 1
        identity = (path,worker_generation)
        require((self.worker_identity is None and successful_step == start)
            or (self.worker_identity == identity and successful_step == self.next_step),
            'Fresh worker must execute its original consecutive prefix without retry or path reuse')
        if worker_generation == 1:
            require(self.restored_checkpoint is not None and self.checkpoint_restore_consumed
                and self.restored_checkpoint['rank'] == self.rank and self.restored_checkpoint['successful_steps'] == 8,
                'Successful exact fresh staged checkpoint restoration required before generation1 training')
        elif self.checkpoint_restore_consumed:
            require(False, 'Restored worker belongs only to the fresh staged generation')
        require(replica.rank == self.rank and next(model.parameters()).device == self.device,
                'Original input rank and model device must match actual worker')
        validate_optimizer(model,optimizer,scheduler,successful_step-1)
        model.train()
        require(all(not m.training for m in model.encoder.modules() if isinstance(m,torch.nn.BatchNorm2d)),
                'Original encoder normalization must remain fixed')
        if self.normalization is None:
            self.normalization = {n:v.detach().clone() for n,v in model.named_buffers()
                if n.endswith(('running_mean','running_var','num_batches_tracked'))}
        self.failed_or_inflight = True
        self.last_update_evidence = dict(path=path,successful_step=successful_step,worker_generation=worker_generation,
            rank=self.rank,status='IN_FLIGHT',actual_complete_pilot_verified=False,competition_submission=False,candidate_promotion=False)
        self.xla.sync(wait=True)
        before = metrics_snapshot(self.metrics)
        self.last_update_evidence['metrics_before'] = before
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
        # Reduce exactly once and complete backward before the all-rank guard.
        # A nonfinite reduced gradient must never mutate ordinary Adam state.
        self.xm.reduce_gradients(optimizer)
        self.xla.sync(wait=True)
        backward_done = metrics_snapshot(self.metrics)
        backward_seconds = time.monotonic()-started
        self.last_update_evidence['metrics_after_backward'] = backward_done
        guard_started = time.monotonic()
        finite = finite_training_flag(model,optimizer,global_loss,self.normalization)
        all_finite = self.xm.all_reduce(self.xm.REDUCE_SUM,finite)
        self.xla.sync(wait=True)
        finite_count = float(all_finite.cpu())
        before_optimizer = metrics_snapshot(self.metrics)
        self.last_update_evidence['metrics_before_optimizer'] = before_optimizer
        self.last_update_evidence['pre_update_finite_rank_count'] = finite_count
        pre_update_guard_seconds = time.monotonic()-guard_started
        require(finite_count == 8, 'All eight ranks must have finite model/loss/reduced gradients/Adam and unchanged normalization before any update')
        optimizer_started = time.monotonic()
        optimizer.step()
        scheduler.step()
        state_tensors = [v for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor) and v.device.type == 'xla']
        optimizer_graph = self.graph_hash([*model.parameters(),*state_tensors])
        self.xla.sync(wait=True)
        trained = metrics_snapshot(self.metrics)
        self.last_update_evidence['metrics_after_training'] = trained
        optimizer_seconds = time.monotonic()-optimizer_started
        training_seconds = backward_seconds+optimizer_seconds
        finite = finite_training_flag(model,optimizer,global_loss,self.normalization)
        all_finite = self.xm.all_reduce(self.xm.REDUCE_SUM,finite)
        self.xla.sync(wait=True)
        require(float(all_finite.cpu()) == 8, 'Nonfinite model/gradient/Adam or changed normalization fails the pilot')
        value = float(global_loss.cpu())
        exported_gradients = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()} if successful_step <= 3 else None
        guarded = metrics_snapshot(self.metrics)
        self.last_update_evidence['metrics_after_guard'] = guarded
        deltas = training_phase_deltas(((before,backward_done),(before_optimizer,trained)))
        observation = PilotUpdateObservation(path,successful_step,self.rank,worker_generation,time.monotonic()-started,
            deltas['compile'],deltas['execute'],forward_graph,optimizer_graph,True,deltas['fallbacks'])
        self.last_update_evidence['observation'] = asdict(observation)
        validate_rank_update(observation, initialized_baseline=self.initialized_baseline)
        if successful_step == start + 1:
            self.initialized_baseline = observation
        self.worker_identity = identity
        self.next_step = successful_step + 1
        validate_optimizer(model,optimizer,scheduler,successful_step)
        self.last_update_evidence['status'] = 'UPDATE_GATES_PASSED'
        self.failed_or_inflight = False
        return dict(observation=asdict(observation), global_loss=value, training_seconds=training_seconds,
            forward_backward_seconds=backward_seconds,pre_update_guard_seconds=pre_update_guard_seconds,optimizer_seconds=optimizer_seconds,
            finite_guard_and_prescribed_gradient_export_seconds=observation.synchronized_interval_seconds-training_seconds,
            metrics_before=before,metrics_after_backward=backward_done,metrics_before_optimizer=before_optimizer,
            metrics_after_training=trained,metrics_after_guard=guarded,pre_update_finite_rank_count=finite_count,
            reduced_gradients=exported_gradients, actual_complete_pilot_verified=False,
            competition_submission=False,candidate_promotion=False)
