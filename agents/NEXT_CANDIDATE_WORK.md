# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Add profiling workload support for sourceable `int_expr` shape specs, including
   concrete workload expansion and execution-plan shape keys.
2. Design expression-only runtime source recovery, or keep it explicitly
   rejected until the runtime can infer named `Dim` values from inverse
   expressions without ambiguity.
3. Continue stabilizing reported output-shape materialization across CUDA
   device-pointer and torch paths.
