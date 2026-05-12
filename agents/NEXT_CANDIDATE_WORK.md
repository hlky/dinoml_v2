# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Hardened CUTLASS execution-plan application by rejecting duplicate static and
  guarded selection entries instead of silently letting the last keyed entry
  win, with targeted manifest/compile regression tests covering ambiguous
  duplicate payloads.

## Ranked Backlog

1. Stabilize CUTLASS profile/cache execution-plan robustness around persistent
   cache concurrency or stale support provenance without broadening op surface;
   duplicate execution-plan payload rejection is landed, but support-fingerprint
   freshness and malformed provenance recovery still need a bounded slice.
2. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
3. Continue GGUF/offload foundation with the next bounded runtime-supported
   slice, preferably explicit CPU/GPU residency-state transitions or load-time
   CUDA dequant staging that preserves the current dense runtime ABI.
