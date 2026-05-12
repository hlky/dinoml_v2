# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Hardened runtime lifecycle bookkeeping so closed modules no longer expose
  stale constant residency snapshots through `constant_load_state()` or
  `is_constant_loaded()`, keeping constant-state introspection aligned with the
  rest of the closed-module error contract.

## Ranked Backlog

1. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
2. Continue GGUF/offload foundation with the next bounded runtime-supported
   slice, preferably explicit CPU/GPU residency-state transitions or load-time
   CUDA dequant staging that preserves the current dense runtime ABI.
3. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
