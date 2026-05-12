# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added a real-libgguf CPU runtime integration test that compiles a deferred
  artifact from `gguf_constant(...)`, inspects
  `encoded_constant_load_plan()`, manually hydrates the encoded constant with
  `load_encoded_constants()`, verifies loaded-state transitions, and proves the
  loaded dense weight changes CPU execution output.

## Ranked Backlog

1. Stabilize CUTLASS profile/cache execution-plan robustness around persistent
   cache concurrency or stale support provenance without broadening op surface.
2. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
3. Continue GGUF/offload foundation with the next bounded runtime-supported
   slice, preferably explicit CPU/GPU residency-state transitions or load-time
   CUDA dequant staging that preserves the current dense runtime ABI.
