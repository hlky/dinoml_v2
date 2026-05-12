# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Add CUDA-backed integration coverage for reported output-shape capacity
   checks when a cheap NVCC/torch fixture is available in CI; the Python
   device-pointer contract now has non-CUDA regression coverage, including
   getter-bound malformed negative reported-shape rejection.
2. Add profile/report cache regression coverage for sourceable symbolic
   expression shapes once a real CUDA profiling fixture is cheap enough for CI.
3. Continue improving runtime/container contracts for allocator, graph, pool,
   profiling, and constant-state failure behavior before op-specific
   assumptions spread; module-close now invalidates live Python sessions before
   freeing the native module handle.
