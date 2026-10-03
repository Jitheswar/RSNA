"""Fixed numerical gates for the independently admitted disposable pilot.

Inputs are already CPU-converted. Comparisons alone establish neither actual
backend/runtime identity nor original MRI/source provenance. No configurable
or retrospectively widened tolerances are exposed.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

CROSS_BACKEND_RTOL = 1e-4
CROSS_BACKEND_ATOL = 1e-6
CPU_REPLAY_ATOL = 1e-6


class PilotParityFailure(ValueError):
    """Keep numerical failure evidence for the future worker's failure recorder."""
    def __init__(self, message: str, report: Mapping[str,Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _cpu_tensor(value: Any, label: str) -> None:
    _require(type(value) is torch.Tensor and value.device.type == 'cpu', f'Ordinary CPU-converted tensor required: {label}')
    _require(value.dtype in (torch.float32,torch.int64,torch.bool,torch.uint8), f'Original declared tensor dtype required: {label}')


def _tensor_report(reference: torch.Tensor, actual: torch.Tensor, label: str, *, relative: bool) -> dict[str,Any]:
    _cpu_tensor(reference,label);_cpu_tensor(actual,label)
    _require(reference.shape == actual.shape and reference.dtype == actual.dtype, f'Exact tensor shape/dtype required: {label}')
    if reference.dtype != torch.float32:
        mismatches = int((reference != actual).sum())
        return dict(name=label,dtype=str(reference.dtype),shape=list(reference.shape),elements=reference.numel(),
            finite_support=reference.numel(),mismatched_elements=mismatches,passed=mismatches == 0,comparison='exact')
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    finite_count = int(finite.sum())
    # Compute diagnostics in float64 to avoid overflow of otherwise finite FP32
    # endpoints. Acceptance uses exactly the fixed reference-relative threshold.
    left,right = reference.to(torch.float64),actual.to(torch.float64)
    errors = (right-left).abs()
    allowed = CROSS_BACKEND_ATOL + CROSS_BACKEND_RTOL*left.abs() if relative else torch.full_like(left,CPU_REPLAY_ATOL)
    accepted = finite & (errors <= allowed)
    finite_errors = errors[finite]
    maximum_error = float(finite_errors.max()) if finite_count else None
    nonzero = finite & (left != 0)
    relative_errors = errors[nonzero]/left[nonzero].abs()
    maximum_relative = float(relative_errors.max()) if relative_errors.numel() else None
    return dict(name=label,dtype=str(reference.dtype),shape=list(reference.shape),elements=reference.numel(),finite_support=finite_count,
        mismatched_elements=int((~accepted).sum()),passed=bool(accepted.all()),maximum_absolute_error=maximum_error,
        maximum_relative_error_on_nonzero_reference=maximum_relative,zero_reference_elements=int((finite & (left == 0)).sum()),
        comparison='fixed_cross_backend' if relative else 'fixed_CPU_replay')


def _walk(reference: Any, actual: Any, label: str, rows: list[dict[str,Any]]) -> None:
    _require(type(reference) is type(actual), f'Exact comparison structure/type required: {label}')
    if isinstance(reference,torch.Tensor):
        rows.append(_tensor_report(reference,actual,label,relative=True))
    elif isinstance(reference,Mapping):
        _require(type(reference) in (dict,OrderedDict) and reference.keys() == actual.keys(), f'Complete exact comparison key inventory required: {label}')
        for key in reference:
            _require(type(key) in (str,int), f'Original plain state key required: {label}')
            _walk(reference[key],actual[key],f'{label}.{key}',rows)
        if hasattr(reference,'_metadata') or hasattr(actual,'_metadata'):
            _walk(getattr(reference,'_metadata',None),getattr(actual,'_metadata',None),f'{label}._metadata',rows)
    elif isinstance(reference,(tuple,list)):
        _require(len(reference) == len(actual), f'Exact state sequence length required: {label}')
        for i,(left,right) in enumerate(zip(reference,actual,strict=True)):
            _walk(left,right,f'{label}.{i}',rows)
    else:
        _require(reference is None or type(reference) in (int,float,str,bool), f'Unsupported comparison value: {label}')
        _require(reference == actual, f'Exact original config/progress/scalar required: {label}')


def _finish(rows: list[dict[str,Any]], *, gate: str) -> dict[str,Any]:
    _require(bool(rows) and all(row['elements'] > 0 for row in rows), 'Complete nonempty finite tensor support required')
    _require(gate in ('fixed_CPU28_vs_TPU','fixed_CPU_replay'), 'Declared fixed numerical gate required')
    report = dict(gate=gate,rtol=0. if gate == 'fixed_CPU_replay' else CROSS_BACKEND_RTOL,
        atol=CPU_REPLAY_ATOL if gate == 'fixed_CPU_replay' else CROSS_BACKEND_ATOL,tensors=rows,tensor_count=len(rows),elements=sum(row['elements'] for row in rows),
        finite_support=sum(row['finite_support'] for row in rows),failed_tensors=sum(not row['passed'] for row in rows),
        passed=all(row['passed'] for row in rows),actual_CPU28_runtime_verified=False,actual_TPU_execution_proven=False,
        original_MRI_identity_verified=False,competition_submission=False,candidate_promotion=False)
    if not report['passed']:
        raise PilotParityFailure('Fixed pilot numerical gate failed; retain report without widening or retry',report)
    return report


def compare_cross_backend_state(reference_CPU28: Mapping[str,Any], actual_TPU: Mapping[str,Any]) -> dict[str,Any]:
    """Original CPU2.8 reference versus TPU, with exact configuration metadata.

    The surrounding worker must independently require the complete expected
    model/Adam/gradient/logit keys, checkpoint boundary and actual backend inputs.
    """
    rows: list[dict[str,Any]] = []
    _walk(reference_CPU28,actual_TPU,'state',rows)
    report = _finish(rows,gate='fixed_CPU28_vs_TPU')
    report.update(rtol=CROSS_BACKEND_RTOL,atol=CROSS_BACKEND_ATOL)
    return report


def compare_cpu_replay_logits(reference: torch.Tensor, actual: torch.Tensor) -> dict[str,Any]:
    """Eight prescribed twelve-target logits, absolute CPU tolerance only."""
    _require(reference.shape == actual.shape == (8,12) and reference.dtype == actual.dtype == torch.float32,
        'All eight original twelve-target FP32 logits required')
    report = _finish([_tensor_report(reference,actual,'eight-study-logits',relative=False)],gate='fixed_CPU_replay')
    report.update(rtol=0.,atol=CPU_REPLAY_ATOL)
    return report


def compare_fixed_first_three_updates(reference_CPU28: Mapping[str,Any], actual_TPU: Mapping[str,Any], *,
    expected_model_state: Mapping[str,torch.Tensor], parameter_names: Sequence[Sequence[str]]) -> dict[str,Any]:
    """Require all original tensors, three gradient/loss records and step3 state."""
    _require(len(parameter_names) == 2 and all(bool(group) for group in parameter_names), 'Original complete encoder/head parameter groups required')
    names = [name for group in parameter_names for name in group]
    buffers = {'imagenet_mean','imagenet_std'} | {name for name in expected_model_state
        if name.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
    _require(names == [name for name in expected_model_state if name not in buffers]
        and all(name.startswith(prefix) for group,prefix in zip(parameter_names,('encoder.','head.'),strict=True) for name in group),
        'Full original initialization parameter scope and ordering required')
    fields = {'model_state_dict','optimizer_state_dict','reduced_gradients','losses','logits','successful_steps'}
    for value in (reference_CPU28,actual_TPU):
        _require(set(value) == fields and type(value['successful_steps']) is int and value['successful_steps'] == 3,
            'Complete fixed step3 reference and all first-three gradient/loss evidence required')
        model = value['model_state_dict']
        _require(model.keys() == expected_model_state.keys() and getattr(model,'_metadata',None) == getattr(expected_model_state,'_metadata',None),
            'Complete original model/buffer/metadata inventory required')
        for name,initial in expected_model_state.items():
            _cpu_tensor(model[name],name)
            _require(model[name].shape == initial.shape and model[name].dtype == initial.dtype, 'Original model tensor shape/dtype required')
        gradients = value['reduced_gradients']
        _require(type(gradients) is list and len(gradients) == 3, 'All three successful reduced-gradient records required')
        for gradient in gradients:
            _require(type(gradient) is dict and list(gradient) == names, 'Every original parameter gradient/order required at every step')
            for name in names:
                _cpu_tensor(gradient[name],name)
                _require(gradient[name].shape == expected_model_state[name].shape and gradient[name].dtype == torch.float32,
                    'Complete FP32 original gradient shape required')
        for key,shape in (('losses',(3,)),('logits',(8,12))):
            _cpu_tensor(value[key],key)
            _require(value[key].shape == shape and value[key].dtype == torch.float32, 'All three FP32 losses and eight-study logits required')
        optimizer = value['optimizer_state_dict']
        _require(set(optimizer) == {'state','param_groups'} and len(optimizer['param_groups']) == 2
            and set(optimizer['state']) == set(range(len(names))), 'Complete original ordinary Adam inventory required')
        start = 0
        for group,group_names,lr in zip(optimizer['param_groups'],parameter_names,(1e-5,1e-4),strict=True):
            _require(group['params'] == list(range(start,start+len(group_names))) and group['lr'] == lr
                and group['betas'] == (.9,.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
                and all(group[key] is False for key in ('amsgrad','maximize','capturable','differentiable'))
                and group['foreach'] is None and group['fused'] is None and group.get('decoupled_weight_decay',False) is False,
                'Exact original ordinary Adam groups/settings/order required')
            start += len(group_names)
        for index,name in enumerate(names):
            state = optimizer['state'][index]
            _require(set(state) == {'step','exp_avg','exp_avg_sq'}, 'Every original Adam step/moment required')
            _cpu_tensor(state['step'],name)
            _require(state['step'].shape == () and state['step'].dtype == torch.float32 and float(state['step']) == 3,
                'Exact original Adam step3 scalar required')
            for key in ('exp_avg','exp_avg_sq'):
                _cpu_tensor(state[key],name)
                _require(state[key].shape == expected_model_state[name].shape and state[key].dtype == torch.float32,
                    'Complete original FP32 Adam moment shapes required')
            _require(bool((state['exp_avg_sq'] >= 0).all()), 'Nonnegative Adam second moments required')
    return compare_cross_backend_state(reference_CPU28,actual_TPU)
