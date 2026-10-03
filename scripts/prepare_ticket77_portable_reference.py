"""Derive restricted-load CPU reference packets after actual preparation success.

Original files are unchanged. This creates no uploaded/final TPU attachment and
cannot launch a CPU job, accelerator session or competition submission.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sys.path.insert(0, str(PROJECT / 'src'))
from ticket75_cpu_reference import digest, read, require, write
from rsna_knee.tpu_pilot_state import portable_cpu_reference, load_cpu_reference

CPU = PROJECT / 'artifacts/private/ticket75-cpu-reference-v3'
CPU_SHA = 'd4551d20a9d380d811fe94f1ca64d16556fc11b1f0853a0032f32ec7d592a46c'
ROOT = PROJECT / 'artifacts/private/ticket75-portable-reference-v3'


def verified_sources():
    require(digest(CPU / 'plan.json') == CPU_SHA, 'Exact original CPU request required')
    require((CPU / 'terminal-verification.json').exists(), 'Actual completed CPU preparation required before conversion')
    from ticket73_configured_full_continuation import successful_terminal
    seconds, terminal = successful_terminal(CPU, CPU_SHA, '00:30:00')
    accepted, result, verified = read(CPU / 'accepted-launch.json'), read(CPU / 'reference-result.json'), read(CPU / 'terminal-verification.json')
    require(verified['status'] == result['status'] == 'COMPLETE' and verified['plan_sha256'] == result['plan_sha256'] == CPU_SHA
            and verified['job_id'] == result['job_id'] == str(accepted['job_id'])
            and verified['result_sha256'] == digest(CPU / 'reference-result.json')
            and verified['terminal_sha256'] == digest(terminal) and verified['actual_cpu_node_seconds'] == seconds
            and result['native_MRI_replayed'] == 64 and result['staged_reference_boundaries'] == [3, 8, 16]
            and all(result[k] is False for k in ('TPU_verified', 'CPU28_verified', 'competition_submission', 'candidate_promotion')),
            'Exact successful actual CPU-only reference chain required')
    plan = read(CPU / 'plan.json')
    require(all(digest(r['path']) == r['sha256'] for r in plan['frozen_files']), 'Original frozen preparation source/input changed')
    require(len(result['checkpoints']) == 3 and {r['step'] for r in result['checkpoints']} == {3, 8, 16}, 'All original reference boundaries required')
    bound = {str(Path(r['path']).resolve()): r['sha256'] for r in result['attachment_files']}
    paths = {0: CPU / 'attachment/initial.pt', **{r['step']: Path(r['path']) for r in result['checkpoints']}}
    for row in result['checkpoints']:
        require(Path(row['path']).resolve().is_relative_to(CPU.resolve()) and digest(row['path']) == row['sha256'], 'Exact local reference checkpoint required')
    require(digest(paths[0]) == bound[str(paths[0].resolve())], 'Exact local initial state required')
    require(all(Path(r['path']).resolve().is_relative_to(CPU.resolve()) and digest(r['path']) == r['sha256'] for r in result['attachment_files']),
            'Original lossless MRI/target/mask attachment bytes changed')
    return paths, result


def prepare():
    require(not ROOT.exists(), 'Unused immutable portable-reference preparation required')
    paths, result = verified_sources()
    import torch
    # These exact hash-bound files originate from our trusted immutable CPU job.
    # Only this local conversion reads its legacy NumPy RNG pickle. The derived
    # remote packets are verified with restricted weights_only=True loading.
    original_bytes = sum(p.stat().st_size for p in CPU.rglob('*') if p.is_file())
    conservative_extra = 5 * max(p.stat().st_size for p in paths.values()) + 10_000_000
    require(original_bytes + conservative_extra <= 8_000_000_000, 'Whole retained source/packets/serialization temporary exceeds pilot envelope')
    ROOT.mkdir(mode=0o700)
    job = read(CPU / 'accepted-launch.json')['job_id']
    sources = [Path(__file__), PROJECT / 'src/rsna_knee/tpu_pilot_state.py', CPU / 'plan.json',
        CPU / 'reference-result.json', CPU / 'terminal-verification.json', CPU / 'accepted-launch.json',
        CPU / 'source-replay.json', CPU / f'controller-{job}/terminal-controller.txt',
        CPU / f'controller-{job}/terminal-observation.json']
    packets = []
    for step, path in sorted(paths.items()):
        parent_sha = digest(path)
        original = torch.load(path, map_location='cpu', weights_only=False)
        require(digest(path) == parent_sha, 'Original checkpoint changed during trusted conversion')
        packet = portable_cpu_reference(original, parent_sha256=parent_sha, step=step)
        temporary = ROOT / f'.cpu-step-{step}.pt.tmp'
        destination = ROOT / f'cpu-step-{step}.pt'
        with temporary.open('xb') as stream:
            torch.save(packet, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        # The strict loader proves remote compatibility without NumPy globals.
        packet_sha = digest(destination)
        load_cpu_reference(destination, packet_sha256=packet_sha, parent_sha256=parent_sha, step=step)
        packets.append(dict(step=step, original_path=str(path), original_sha256=parent_sha,
            portable_path=str(destination), portable_sha256=packet_sha, bytes=destination.stat().st_size,
            restricted_load_passed=True, attached_in_future_bundle=step in (0, 3)))
        sources.append(path)
        del original, packet
    require(sum(p.stat().st_size for base in (CPU, ROOT) for p in base.rglob('*') if p.is_file()) <= 8_000_000_000,
            'Actual source plus portable reference assets exceeded pilot envelope')
    write(ROOT / 'portable-reference-receipt.json', dict(status='COMPLETE', created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_CPU_plan_sha256=CPU_SHA, original_MRI_verified_by_actual_CPU_job=True,
        operational_recovery=read(CPU / "plan.json")["operational_recovery"],
        exact_original_checkpoints_preserved=True, packets=packets,
        source_bindings=[dict(path=str(p), sha256=digest(p)) for p in sources],
        all_remote_reference_loads_weights_only=True, legacy_pickle_read_only_on_trusted_source_host=True,
        final_attachment_prepared=False, uploaded=False, accelerator_launch_admitted=False,
        actual_TPU_execution=False, competition_submission=False, candidate_promotion=False))
    print(dict(status='PORTABLE_REFERENCE_ONLY', receipt_sha256=digest(ROOT / 'portable-reference-receipt.json')))


if __name__ == '__main__':
    prepare()
