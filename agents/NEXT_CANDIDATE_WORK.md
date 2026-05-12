# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Filtered CUTLASS SM80 reduced-precision GEMM/BMM candidate manifests by
  layout-aware tensor-op thread-map divisibility so generated support sources
  no longer instantiate unbuildable RRR N=96/160/224 TensorOp policies.

## Ranked Backlog

1. Continue GGUF/offload foundation with a small policy-execution slice, such
   as load-time CUDA dequant staging or explicit CPU/GPU residency state, if it
   can preserve the dense runtime ABI and land with transactional tests.
2. Stabilize CUTLASS profile/cache execution-plan robustness around persistent
   cache concurrency or stale support provenance without broadening op surface.
3. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
