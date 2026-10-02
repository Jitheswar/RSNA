"""CPU contract checks only; no XLA import. Collective failure boundaries are mocked."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsna_knee.model import ModelConfig, build_study_model
from rsna_knee.resnet18 import RESNET18_FIXTURE_ENCODER_ID, RESNET18_FULL_FINETUNE_MODEL_ID, ANATOMICAL_POOLING_ID

spec = importlib.util.spec_from_file_location('ticket75_xla_update', Path(__file__).parents[1] / 'scripts/ticket75_xla_update.py')
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model():
    torch.manual_seed(75)
    return build_study_model(ModelConfig(image_size=(32,32), model_id=RESNET18_FULL_FINETUNE_MODEL_ID,
        encoder_id=RESNET18_FIXTURE_ENCODER_ID, pooling_id=ANATOMICAL_POOLING_ID))


def test_original_ordinary_adam_scope_and_full_encoder_gradient_update():
    m = model();optimizer = update.original_adam(m);scheduler = update.original_scheduler(optimizer)
    update.validate_optimizer(m,optimizer,scheduler,0)
    before = {k:v.clone() for k,v in m.named_buffers()}
    m.train()
    images = torch.rand(4,2,2,1,32,32)
    slices = torch.ones(4,2,2,dtype=torch.bool);series = torch.ones(4,2,dtype=torch.bool)
    descriptors = torch.zeros(4,2,10);descriptors[:,:,0] = 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(images,slices,series,descriptors),torch.ones(4,12))
    loss.backward();optimizer.step();scheduler.step()
    update.validate_optimizer(m,optimizer,scheduler,1)
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in m.parameters())
    assert all(torch.equal(before[k],v) for k,v in m.named_buffers())


@pytest.mark.parametrize('defect', ['lr', 'weight_decay', 'decoupled_weight_decay', 'betas', 'foreach', 'group_order', 'scheduler_position', 'frozen_encoder', 'not_full_model'])
def test_changed_original_scope_rejected(defect):
    m = model();optimizer = update.original_adam(m);scheduler = update.original_scheduler(optimizer)
    if defect == 'lr': optimizer.param_groups[0]['lr'] = 2e-5
    if defect == 'weight_decay': optimizer.param_groups[0]['weight_decay'] = .1
    if defect == 'decoupled_weight_decay': optimizer.param_groups[0]['decoupled_weight_decay'] = True
    if defect == 'betas': optimizer.param_groups[0]['betas'] = (.8,.999)
    if defect == 'foreach': optimizer.param_groups[0]['foreach'] = True
    if defect == 'group_order': optimizer.param_groups[0]['params'].reverse()
    if defect == 'scheduler_position': scheduler.last_epoch = 1
    if defect == 'frozen_encoder': next(m.encoder.parameters()).requires_grad_(False)
    if defect == 'not_full_model': m = torch.nn.Linear(4,12)
    with pytest.raises(ValueError): update.validate_optimizer(m,optimizer,scheduler,0)


@pytest.mark.parametrize('defect', ['unadmitted', 'missing_sources', 'low_precision'])
def test_backend_initialization_fails_before_any_XLA_import(monkeypatch, defect):
    hashes = {k:'a'*64 for k in ('torch_xla.core.xla_model','torch_xla.debug.metrics','torch_xla.backends','torch_xla.torch_xla','torch.optim.optimizer')}
    admitted = True
    if defect == 'unadmitted': admitted = False
    if defect == 'missing_sources': hashes = {}
    if defect == 'low_precision': monkeypatch.setenv('XLA_USE_BF16','1')
    import builtins
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('torch_xla'): pytest.fail('Invalid worker must not import/acquire backend')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError): update.XlaPilotUpdate(admitted_worker=admitted, expected_source_hashes=hashes)


def snapshot(compile=3, execute=4, replicated=None, counters=None, fallback=()):
    result = dict(metrics={},counters=counters or {},executed_fallback_ops=list(fallback))
    for name,count in (('CompileTime',compile),('ExecuteTime',execute),('ExecuteReplicatedTime',replicated)):
        if count is not None: result['metrics'][name] = dict(total_samples=count,accumulator=0.,samples=[])
    return result


def test_both_actual_execution_metric_names_are_observed_and_absence_not_zero():
    before = snapshot(1,2,3);after = snapshot(1,3,5)
    assert update.metric_delta(before,after,('CompileTime',)) == 0
    assert update.metric_delta(before,after,('ExecuteTime','ExecuteReplicatedTime')) == 3
    assert update.metric_delta(snapshot(None,None),snapshot(None,None),('CompileTime',)) is None
    assert update.metric_delta(snapshot(None,None),snapshot(None,None,1),('ExecuteTime','ExecuteReplicatedTime')) == 1
    with pytest.raises(ValueError,match='backwards'): update.metric_delta(before,snapshot(0,4),('CompileTime',))


def test_all_training_fallback_deltas_are_retained_and_unobserved_op_rejected():
    before = snapshot(counters={'aten::_local_scalar_dense':2},fallback=['aten::_local_scalar_dense'])
    after = snapshot(counters={'aten::_local_scalar_dense':2,'aten::unsupported':1},fallback=['aten::_local_scalar_dense','aten::unsupported'])
    assert update.fallback_deltas(before,after) == (('aten::_local_scalar_dense',0),('aten::unsupported',1))
    bad = deepcopy(after);bad['counters'].pop('aten::unsupported')
    with pytest.raises(ValueError,match='lacked'): update.fallback_deltas(before,bad)
    bad = deepcopy(before);bad['counters']['aten::_local_scalar_dense'] = 1
    with pytest.raises(ValueError,match='reset'): update.fallback_deltas(before,bad)


def test_metrics_capture_preserves_raw_samples_counters_and_fallback_inventory():
    api = SimpleNamespace(metric_names=lambda:['CompileTime'], metric_data=lambda n:(3,2.5,[(1.,.5),(2.,2.)]),
        counter_names=lambda:['aten::unsupported'],counter_value=lambda n:1, executed_fallback_ops=lambda:['aten::unsupported'])
    result = update.metrics_snapshot(api)
    assert result['metrics']['CompileTime']['samples'] == [(1.,.5),(2.,2.)]
    assert result['counters'] == {'aten::unsupported':1} and result['executed_fallback_ops'] == ['aten::unsupported']
    api.counter_value = lambda n:None
    with pytest.raises(ValueError,match='complete registered counter'): update.metrics_snapshot(api)


def test_unknown_graph_hash_cannot_be_replaced_by_synthetic_success():
    backend = object.__new__(update.XlaPilotUpdate)
    backend.xla = SimpleNamespace(_XLAC=SimpleNamespace(_get_graph_hash=lambda _:b'\x01'*16))
    assert backend.graph_hash([]) == '01'*16
    backend.xla._XLAC._get_graph_hash = lambda _:None
    with pytest.raises(ValueError,match='actual|Actual'): backend.graph_hash([])


@pytest.mark.parametrize('defect', ['consumed_failure','skipped_start','skipped_next','repeated_step','path_reuse','generation_reuse','bool_generation'])
def test_worker_prefix_or_consumed_failure_rejected_before_any_backend_work(defect):
    backend = object.__new__(update.XlaPilotUpdate)
    backend.failed_or_inflight = defect == 'consumed_failure'
    backend.checkpoint_restore_consumed = False;backend.restored_checkpoint = None
    backend.worker_identity = None
    backend.next_step = None
    path,step,generation = 'continuous',1,0
    if defect == 'skipped_start': step = 3
    if defect in ('skipped_next','repeated_step','path_reuse','generation_reuse'):
        backend.worker_identity = ('continuous',0)
        backend.next_step = 4
    if defect == 'skipped_next': step = 5
    if defect == 'repeated_step': step = 3
    if defect == 'path_reuse': path = 'staged'
    if defect == 'generation_reuse': path,step,generation = 'staged',9,1
    if defect == 'bool_generation': generation = False
    # No backend, model, optimizer or replica exists. The actual update must fail
    # admission first; this neither mocks nor runs the XLA training computation.
    with pytest.raises(ValueError): backend.update(None,None,None,None,path=path,successful_step=step,worker_generation=generation)


@pytest.mark.parametrize('defect', ['none','loss_nan','reduced_gradient_inf','parameter_nan','moment_nan','second_moment_nan','second_moment_negative','changed_normalization','missing_gradient'])
def test_pre_update_device_flag_detects_failures_without_mutating_model_or_adam(defect):
    m = model();optimizer = update.original_adam(m)
    for p in m.parameters(): p.grad = torch.ones_like(p)
    optimizer.step()
    normalization = {n:v.clone() for n,v in m.named_buffers()
        if n.endswith(('running_mean','running_var','num_batches_tracked'))}
    loss = torch.tensor(1.)
    parameter = next(m.parameters())
    if defect == 'loss_nan': loss.fill_(float('nan'))
    if defect == 'reduced_gradient_inf': parameter.grad.flatten()[0] = float('inf')
    if defect == 'parameter_nan':
        with torch.no_grad(): parameter.flatten()[0] = float('nan')
    if defect == 'moment_nan': optimizer.state[parameter]['exp_avg'].flatten()[0] = float('nan')
    if defect == 'second_moment_nan': optimizer.state[parameter]['exp_avg_sq'].flatten()[0] = float('nan')
    if defect == 'second_moment_negative': optimizer.state[parameter]['exp_avg_sq'].flatten()[0] = -1
    if defect == 'changed_normalization':
        name = next(n for n in normalization if n.endswith('running_mean'))
        dict(m.named_buffers())[name].flatten()[0] += 1
    if defect == 'missing_gradient': parameter.grad = None
    before = deepcopy(m.state_dict());adam_before = deepcopy(optimizer.state_dict())
    if defect == 'missing_gradient':
        with pytest.raises(ValueError,match='gradient'): update.finite_training_flag(m,optimizer,loss,normalization)
    else:
        flag = update.finite_training_flag(m,optimizer,loss,normalization)
        assert flag.device.type == 'cpu' and flag.dtype == torch.float32 and flag.shape == ()
        assert flag.item() == int(defect == 'none')
    for name,value in m.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0,equal_nan=True)
    for key,state in optimizer.state_dict()['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,adam_before['state'][key][name],rtol=0,atol=0,equal_nan=True)


def test_initial_empty_adam_state_has_a_valid_pre_update_flag():
    m = model();optimizer = update.original_adam(m)
    for p in m.parameters(): p.grad = torch.ones_like(p)
    normalization = {n:v.clone() for n,v in m.named_buffers()
        if n.endswith(('running_mean','running_var','num_batches_tracked'))}
    assert update.finite_training_flag(m,optimizer,torch.tensor(1.),normalization).item() == 1
    assert not optimizer.state


def test_training_phase_deltas_exclude_intentional_guard_and_retain_raw_training_fallbacks():
    before = snapshot(2,4,counters={'aten::_local_scalar_dense':0})
    backward_done = snapshot(3,5,counters={'aten::_local_scalar_dense':0})
    # The explicit finite guard has its own compile/execute and scalar readback.
    before_optimizer = snapshot(5,9,counters={'aten::_local_scalar_dense':1})
    trained = snapshot(5,10,counters={'aten::_local_scalar_dense':1,'aten::unsupported':2},fallback=['aten::unsupported'])
    result = update.training_phase_deltas(((before,backward_done),(before_optimizer,trained)))
    assert result == dict(compile=1,execute=2,fallbacks=(('aten::_local_scalar_dense',0),('aten::unsupported',2)))
    assert update.metric_delta(before,trained,('CompileTime',)) == 3
    assert update.training_phase_deltas(((snapshot(None,None),snapshot(None,None)),(before_optimizer,trained)))['execute'] is None
    with pytest.raises(ValueError,match='reset|backwards'):
        update.training_phase_deltas(((backward_done,before),))


@pytest.mark.parametrize('failure', ['other_rank','nonfinite_reduced_gradient'])
def test_failed_collective_pre_guard_leaves_original_parameters_adam_and_scheduler_unchanged(failure):
    m = model();optimizer = update.original_adam(m);scheduler = update.original_scheduler(optimizer)
    before = deepcopy(m.state_dict())
    if failure == 'nonfinite_reduced_gradient':
        next(m.parameters()).register_hook(lambda gradient: torch.full_like(gradient,float('nan')))
    backend = object.__new__(update.XlaPilotUpdate)
    backend.device = torch.device('cpu');backend.rank = 0
    backend.checkpoint_restore_consumed = False;backend.restored_checkpoint = None
    backend.normalization = backend.initialized_baseline = backend.worker_identity = backend.next_step = None
    backend.failed_or_inflight = False;backend.last_update_evidence = None
    reductions = []
    # Actual CPU forward/backward, with only backend synchronization/collective
    # boundaries replaced. This cannot establish actual XLA/TPU behavior.
    backend.xla = SimpleNamespace(sync=lambda **kwargs:None,
        _XLAC=SimpleNamespace(_get_graph_hash=lambda _:b'a'*16))
    def all_reduce(operation,value,**kwargs):
        if 'scale' in kwargs: return value.detach()
        assert value.item() == int(failure == 'other_rank')
        return torch.tensor(7. if failure == 'other_rank' else 0.)
    backend.xm = SimpleNamespace(REDUCE_SUM='sum',all_reduce=all_reduce,
        reduce_gradients=lambda optimizer:reductions.append('mean'))
    backend.metrics = SimpleNamespace(metric_names=lambda:['CompileTime','ExecuteTime'],
        metric_data=lambda name:(1,1.,[]),counter_names=lambda:[],executed_fallback_ops=lambda:[])
    replica = SimpleNamespace(rank=0,model_inputs=(torch.rand(1,2,2,1,32,32),torch.ones(1,2,2,dtype=torch.bool),
        torch.ones(1,2,dtype=torch.bool),torch.zeros(1,2,10)),targets=torch.ones(1,12),
        supervision_mask=torch.ones(1,12,dtype=torch.bool),global_supervised_entries=torch.tensor(48.))
    with pytest.raises(ValueError,match='before any update'):
        backend.update(m,optimizer,scheduler,replica,path='continuous',successful_step=1,worker_generation=0)
    assert reductions == ['mean'] and not optimizer.state and scheduler.last_epoch == 0
    for name,value in m.state_dict().items(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert backend.failed_or_inflight is True
    assert backend.last_update_evidence['pre_update_finite_rank_count'] == (7 if failure == 'other_rank' else 0)
    assert 'metrics_before_optimizer' in backend.last_update_evidence
    with pytest.raises(ValueError,match='cannot be retried'):
        backend.update(m,optimizer,scheduler,replica,path='continuous',successful_step=1,worker_generation=0)
