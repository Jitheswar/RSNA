"""Synthetic observations exercise gates; they do not establish backend proof."""
from dataclasses import replace

import pytest

from rsna_knee.tpu_pilot_telemetry import PilotUpdateObservation, summarize_fixed_pilot_updates, validate_fixed_pilot_prefix, validate_rank_update


def observations():
    return [PilotUpdateObservation(path, step, rank, int(path == 'staged' and step >= 9),
        10.0 if step == 1 or path == 'staged' and step == 9 else 1.0,
        1 if step in (1,2) or path == 'staged' and step in (9,10) else 0, 1,
        'a'*32, 'b'*32, True, (('aten::nonzero',0),))
        for path in ('continuous','staged') for step in range(1,17) for rank in range(8)]


def test_exact_three_fresh_generations_and_all_timed_real_updates_required():
    result = summarize_fixed_pilot_updates(observations())
    assert result['physical_updates'] == 32 and result['distinct_diagnostic_learning_updates'] == 16
    assert result['all_rank_update_observations'] == 256 and result['fresh_worker_generations'] == 3
    assert result['remaining_update_count'] == 29 and result['remaining_synchronized_update_total_seconds'] == 29
    assert result['cold_update_maximum_across_ranks_seconds'] == [10,10,10]
    assert result['actual_TPU_execution_proven'] is False
    assert result['full_phase_budget_verified'] is False
    assert result['checkpoint_state_parity_verified'] is False
    assert result['full_training_admitted'] is result['candidate_promotion'] is result['competition_submission'] is False


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'generation', 'rank', 'unknown_compile', 'unknown_execute', 'no_execution',
    'missing_forward_graph', 'missing_optimizer_graph', 'unobserved_fallback', 'fallback', 'scalar_fallback', 'negative_delta', 'churn_compile',
    'churn_forward', 'churn_optimizer', 'nonfinite_time', 'zero_time', 'cold_overrun', 'steady_overrun'])
def test_missing_actual_evidence_or_fixed_gate_failure_cannot_pass(defect):
    values = observations()
    # Select a steady step from rank7; failures cannot be hidden by rank0.
    index = next(i for i,o in enumerate(values) if (o.path,o.successful_step,o.rank) == ('staged',12,7))
    changes = dict(generation=dict(worker_generation=0), rank=dict(rank=True), unknown_compile=dict(compile_samples_delta=None),
        unknown_execute=dict(execute_samples_delta=None), no_execution=dict(execute_samples_delta=0),
        missing_forward_graph=dict(forward_backward_graph_sha=None), missing_optimizer_graph=dict(optimizer_graph_sha=None),
        unobserved_fallback=dict(fallback_counters_observed=False), fallback=dict(fallback_counter_deltas=(('aten::unsupported',1),)),
        scalar_fallback=dict(fallback_counter_deltas=(('aten::_local_scalar_dense',1),)), negative_delta=dict(compile_samples_delta=-1),
        churn_compile=dict(compile_samples_delta=1), churn_forward=dict(forward_backward_graph_sha='c'*32),
        churn_optimizer=dict(optimizer_graph_sha='c'*32), nonfinite_time=dict(synchronized_interval_seconds=float('nan')),
        zero_time=dict(synchronized_interval_seconds=0), steady_overrun=dict(synchronized_interval_seconds=500))
    if defect == 'missing': values.pop(index)
    elif defect == 'duplicate': values[index] = values[index-1]
    elif defect == 'cold_overrun':
        index = next(i for i,o in enumerate(values) if (o.path,o.successful_step,o.rank) == ('continuous',1,7))
        values[index] = replace(values[index], synchronized_interval_seconds=601)
    else: values[index] = replace(values[index], **changes[defect])
    with pytest.raises(ValueError): summarize_fixed_pilot_updates(values)


def test_max_across_ranks_and_whole_steady_budget_are_enforced():
    values = observations()
    values = [replace(o,synchronized_interval_seconds=20.0) if o.synchronized_interval_seconds == 1 else o for o in values]
    # Each single step looks modest but29×20 exceeds the prespecified480 s.
    with pytest.raises(ValueError, match='timing envelope'): summarize_fixed_pilot_updates(values)


def test_every_complete_prefix_passes_without_claiming_completed_backend():
    values = observations()
    for count in range(8,257,8):
        result = validate_fixed_pilot_prefix(values[:count])
        assert result['completed_physical_updates'] == count // 8
        assert result['complete_fixed_pilot_observations'] is (count == 256)
        assert result['actual_TPU_execution_proven'] is False
    assert result['remaining_synchronized_update_total_seconds'] == summarize_fixed_pilot_updates(values)['remaining_synchronized_update_total_seconds']


@pytest.mark.parametrize('path,step', [('continuous',3),('staged',3),('staged',11)])
def test_first_steady_churn_fails_immediately_before_remaining_updates(path,step):
    values = observations()
    end = next(i for i,o in enumerate(values) if (o.path,o.successful_step,o.rank) == (path,step,7)) + 1
    values = values[:end]
    values[-1] = replace(values[-1],compile_samples_delta=1)
    with pytest.raises(ValueError,match='churn'): validate_fixed_pilot_prefix(values)
    baseline_step = 10 if step == 11 else 2
    baseline = next(o for o in values if (o.path,o.successful_step,o.rank) == (path,baseline_step,7))
    with pytest.raises(ValueError,match='churn'): validate_rank_update(values[-1],initialized_baseline=baseline)


@pytest.mark.parametrize('defect', ['missing_rank','skipped_step','duplicate','empty','cold_overrun','aggregate_overrun','no_execute','missing_baseline','foreign_baseline'])
def test_prefix_cannot_continue_past_missing_rank_evidence_or_budget_failure(defect):
    values = observations()[:24]
    if defect == 'missing_rank': values.pop()
    if defect == 'skipped_step': values = values[:8] + values[16:]
    if defect == 'duplicate': values.append(values[-1])
    if defect == 'empty': values = []
    if defect == 'cold_overrun': values[7] = replace(values[7],synchronized_interval_seconds=601)
    if defect == 'aggregate_overrun':
        values[15] = replace(values[15],synchronized_interval_seconds=250)
        values[23] = replace(values[23],synchronized_interval_seconds=231)
    if defect == 'no_execute': values[23] = replace(values[23],execute_samples_delta=0)
    if defect in ('missing_baseline','foreign_baseline'):
        baseline = None if defect == 'missing_baseline' else replace(values[15],worker_generation=1)
        with pytest.raises(ValueError,match='baseline'): validate_rank_update(values[23],initialized_baseline=baseline)
    else:
        with pytest.raises(ValueError): validate_fixed_pilot_prefix(values)
