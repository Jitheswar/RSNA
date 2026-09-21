#!/usr/bin/env bash
# Deprecated compatibility name. Ticket 16 no longer launches a local model.
# Use run_ticket16_astra_handoff.sbatch for approved private batch preparation.
set -euo pipefail
echo "ticket-16 local model execution was removed; use slurm/run_ticket16_astra_handoff.sbatch" >&2
exit 2
