# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Design expression-only runtime source recovery, or keep it explicitly
   rejected until the runtime can infer named `Dim` values from inverse
   expressions without ambiguity.
2. Add sourceable symbolic-expression coverage to execution-plan conflict tests,
   cache-key audits, and profile-assisted compile smoke tests.
3. Continue stabilizing reported output-shape materialization across CUDA
   device-pointer and torch paths.
