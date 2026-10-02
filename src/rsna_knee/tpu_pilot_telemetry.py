"""Fixed engineering gates for the disposable32-update/eight-rank pilot.

The caller must separately prove actual backend, exact source/runtime and input
identity. Synthetic observations passing these gates cannot prove TPU execution.
Training phase observations exclude separately recorded export/finite-guard work.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class PilotUpdateObservation:
    path: str
    successful_step: int
    rank: int
    worker_generation: int
    synchronized_interval_seconds: float
    compile_samples_delta: int | None
    execute_samples_delta: int | None
    forward_backward_graph_sha: str | None
    optimizer_graph_sha: str | None
    fallback_counters_observed: bool
    fallback_counter_deltas: tuple[tuple[str, int], ...]


def _graph_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) in (32, 64) and all(c in '0123456789abcdef' for c in value)


def summarize_fixed_pilot_updates(observations: Sequence[PilotUpdateObservation]) -> dict[str, object]:
    """No discarded warmup, hidden retry, missing replica or retrospective gate."""
    expected = {(path, step, rank) for path in ('continuous', 'staged') for step in range(1,17) for rank in range(8)}
    actual = [(o.path, o.successful_step, o.rank) for o in observations]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Exactly32 physical fixed updates with all eight actual rank observations required')
    by_key = {key: value for key,value in zip(actual,observations,strict=True)}
    generations = (('continuous',0,1,16), ('staged',0,1,8), ('staged',1,9,16))
    cold: list[float] = []
    steady: list[float] = []
    for path, generation, start, end in generations:
        for rank in range(8):
            baseline = by_key[path, start + 1, rank]
            for step in range(start,end + 1):
                o = by_key[path,step,rank]
                if type(o.successful_step) is not int or type(o.rank) is not int or type(o.worker_generation) is not int or o.worker_generation != generation:
                    raise ValueError('Exact declared fresh-worker generation and original step/rank required')
                if type(o.synchronized_interval_seconds) not in (float,int) or not math.isfinite(o.synchronized_interval_seconds) or o.synchronized_interval_seconds <= 0:
                    raise ValueError('Actual positive synchronized update interval required')
                if type(o.compile_samples_delta) is not int or o.compile_samples_delta < 0 or type(o.execute_samples_delta) is not int or o.execute_samples_delta <= 0:
                    raise ValueError('Actual nonnegative compile and positive backend execute sample deltas required')
                if not _graph_hash(o.forward_backward_graph_sha) or not _graph_hash(o.optimizer_graph_sha):
                    raise ValueError('Both actual phase graph hashes required; absence is not stability')
                if o.fallback_counters_observed is not True:
                    raise ValueError('Actual fallback counter observation required')
                names = [name for name,_ in o.fallback_counter_deltas]
                if len(names) != len(set(names)) or any(not name.startswith('aten::') or type(count) is not int or count < 0 for name,count in o.fallback_counter_deltas):
                    raise ValueError('Unambiguous complete training CPU fallback counter deltas required')
                if any(count > 0 for _,count in o.fallback_counter_deltas):
                    raise ValueError('Unsupported training CPU fallback fails the fixed pilot')
                # First update includes cold compile/execute; second establishes
                # initialized ordinary Adam. Export/reference/finite phases are
                # separately timed and cannot conceal a training compilation.
                if step >= start + 2 and (o.compile_samples_delta != 0
                    or o.forward_backward_graph_sha != baseline.forward_backward_graph_sha
                    or o.optimizer_graph_sha != baseline.optimizer_graph_sha):
                    raise ValueError('Initialized steady training graph churn fails the fixed pilot')
        cold.append(max(by_key[path,start,rank].synchronized_interval_seconds for rank in range(8)))
        steady.extend(max(by_key[path,step,rank].synchronized_interval_seconds for rank in range(8))
            for step in range(start + 1,end + 1))
    if any(seconds > 600 for seconds in cold) or sum(steady) > 480:
        raise ValueError('Fixed cold/remaining29-update timing envelope exceeded')
    return dict(physical_updates=32, distinct_diagnostic_learning_updates=16, world_size=8,
        fresh_worker_generations=3, all_rank_update_observations=256,
        cold_update_maximum_across_ranks_seconds=cold, remaining_update_count=len(steady),
        remaining_synchronized_update_total_seconds=sum(steady),
        median_remaining_global_update_seconds=median(steady), maximum_remaining_global_update_seconds=max(steady),
        initialized_training_graphs_stable=True, observed_training_CPU_fallback=False,
        actual_backend_identity_verified=False, full_phase_budget_verified=False,
        original_MRI_identity_verified=False, checkpoint_state_parity_verified=False,
        actual_TPU_execution_proven=False, full_training_admitted=False, candidate_promotion=False,
        competition_submission=False)
