"""Synthetic observations exercise gates; they do not establish backend proof."""
from dataclasses import replace

import pytest

from rsna_knee.tpu_pilot_telemetry import PilotUpdateObservation, summarize_fixed_pilot_updates


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
