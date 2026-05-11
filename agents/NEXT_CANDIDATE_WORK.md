# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Continue stabilizing reported output-shape materialization across CUDA
   device-pointer and torch paths.
2. Add profile/report cache regression coverage for sourceable symbolic
   expression shapes once a real CUDA profiling fixture is cheap enough for CI.
3. Improve runtime/container contracts for allocator, graph, pool, and profiling
   behavior before op-specific assumptions spread.
