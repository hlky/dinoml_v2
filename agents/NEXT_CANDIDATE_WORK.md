# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added a newcomer-visible CPU runtime lifecycle smoke test that compiles an
  artifact with deferred constants, verifies a run fails before constants are
  loaded, calls `load_constants_from_file()`, runs successfully, and closes the
  session/module explicitly.

## Ranked Backlog

1. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
2. Add a small documented smoke entry point or helper only if it makes the
   lifecycle validation path easier to discover without becoming README churn.
3. Continue GGUF/offload foundation with the next bounded runtime-supported
   slice, preferably explicit CPU/GPU residency-state transitions or load-time
   CUDA dequant staging that preserves the current dense runtime ABI.
4. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
