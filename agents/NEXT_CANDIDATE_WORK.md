# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Hardened GGUF encoded-constant path handling by normalizing compile-time
  source metadata to absolute paths and resolving runtime-relative manifest
  paths against the artifact directory before manual encoded loads.

## Ranked Backlog

1. Continue GGUF/offload foundation with a small validated execution slice,
   preferably a real-libgguf CPU runtime integration test for
   `encoded_constant_load_plan()` / `load_encoded_constants()`, or failing that
   load-time CUDA dequant staging or explicit CPU/GPU residency state that
   preserves the dense runtime ABI.
2. Stabilize CUTLASS profile/cache execution-plan robustness around persistent
   cache concurrency or stale support provenance without broadening op surface.
3. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
