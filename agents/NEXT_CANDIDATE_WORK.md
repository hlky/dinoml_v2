# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Add CUDA-backed integration coverage for reported output-shape capacity
   checks when a cheap NVCC/torch fixture is available in CI; the Python
   device-pointer contract now has non-CUDA regression coverage.
2. Add profile/report cache regression coverage for sourceable symbolic
   expression shapes once a real CUDA profiling fixture is cheap enough for CI.
3. Improve runtime/container contracts for allocator, graph, pool, and profiling
   behavior before op-specific assumptions spread.
