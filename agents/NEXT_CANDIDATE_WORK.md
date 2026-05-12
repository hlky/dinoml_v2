# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added a newcomer-visible CPU runtime lifecycle smoke test that compiles an
  artifact with deferred constants, verifies a run fails before constants are
  loaded, calls `load_constants_from_file()`, runs successfully, and closes the
  session/module explicitly. Added a short Development note pointing at the
  exact focused pytest command for that smoke path.
- Re-ran the highest-value CUDA runtime lifecycle subset after the recent
  runtime lifecycle, GGUF encoded-constant, and CUDA mixed-residency changes.
  `test_cuda_artifact_runs_without_torch`,
  `test_runtime_constant_update_changes_output`,
  `test_cuda_runtime_mixed_dense_and_manual_encoded_constant_reload_requires_explicit_encoded_load`,
  `test_cuda_runtime_supports_dynamic_shapes`,
  `test_cuda_runtime_materializes_direct_input_output`,
  `test_cuda_runtime_set_constant_accepts_dynamic_shape`, and
  `test_cuda_runtime_supports_dynamic_generic_broadcast` all passed. Human
  guidance says the full CUDA suite was recently run and is not the current
  priority; do not spend the next loop rerunning it unless new broad CUDA
  changes or reproduced failures justify that cost.

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
