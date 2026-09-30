# Inventory usable compute and existing reusable artifacts

Parent: ../map.md
Type: task
Labels: wayfinder:task
Mode: AFK
Status: resolved
Assignee: baseline_facts
Blocked by: none

## Question

What existing caches, complete label artifacts, pretrained weights, storage headroom, GPU types, queue constraints, and observed allocation history can an initial experiment proposal actually rely on? Resolve through read-only inventory and scheduler queries, without launching work or exposing credentials or patient identifiers. Record missing access and distinguish physical capacity from granted resources. This inventory unblocks an experiment-budget decision; it does not authorize training or guarantee future queue availability.

## Answer

Resolved 2026-09-25 from the researcher's saved aggregate inventory, reviewed by the root agent after the researcher's turn was interrupted. Root also refreshed the read-only scheduler and filesystem observations. CPU capacity is available subject to existing ceilings; both GPU partitions have pending jobs and offer no guaranteed allocation date. The previous verified allocation was an A30X with approximately 25.34 GB assigned VRAM, not evidence of an available A40.

Existing image-training membership covers 158 studies (58 reference and 100 report-only); the reconciled 4,407-study header inventory and broad report extraction do not establish full-corpus pixel/cache readiness. No reusable pretrained weights were identified in the bounded conventional cache locations. Shared filesystem free space is approximately 23 TiB, but remaining room inside the project's 2 TB soft budget is unmeasured. Slurm historical accounting was inaccessible. The initial plan must include a bounded storage ledger, exact input/cache validation and measured throughput before expansion; these operational prerequisites are not promises of capacity.

Asset: [Compute and reusable-artifact inventory](../../../docs/research/leaderboard-compute-inventory.md). The report records inspected sources and limitations; no jobs, downloads or resource changes were performed. The available facts suffice to choose a bounded initial campaign with a CPU route and conditional GPU expansion.
