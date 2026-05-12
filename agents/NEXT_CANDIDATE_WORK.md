# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added CUDA regression coverage for mixed dense plus
  `manual_runtime_load` GGUF constant residency so module open,
  `unload_constants()`, and `load_constants_from_file()` still require an
  explicit `load_encoded_constants(...)` step before manual encoded weights are
  runnable again.

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
