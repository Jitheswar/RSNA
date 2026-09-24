from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_benchmark_cli_rejects_login_node_execution(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "benchmark-output"
    environment = os.environ.copy()
    for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE"):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(repository_root / "src")

    result = subprocess.run(
        (
            sys.executable,
            str(repository_root / "scripts/ticket25_efficiency_benchmark.py"),
            "--output-root",
            str(output_root),
        ),
        cwd=repository_root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "inside an approved Slurm allocation" in result.stderr
    assert not output_root.exists()
