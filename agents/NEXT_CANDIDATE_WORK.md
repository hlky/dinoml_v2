# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Design expression-only runtime source recovery, or keep it explicitly
   rejected until the runtime can infer named `Dim` values from inverse
   expressions without ambiguity.
2. Continue stabilizing reported output-shape materialization across CUDA
   device-pointer and torch paths.
3. Add profile/report cache regression coverage for sourceable symbolic
   expression shapes once a real CUDA profiling fixture is cheap enough for CI.
