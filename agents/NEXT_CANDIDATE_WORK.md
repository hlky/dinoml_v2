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
  `test_cuda_runtime_supports_dynamic_generic_broadcast` all passed. A full
  `python -m pytest -q tests/test_cuda_runtime.py` rerun remains pending
  because the file rebuilds multiple CUTLASS support libraries in per-test temp
  caches and did not finish within this bounded validation loop.

## Ranked Backlog

1. Re-run the full CUDA runtime file end to end,
   `python -m pytest -q tests/test_cuda_runtime.py`, using a longer validation
   window or a cache-aware approach so the remaining non-subset CUDA runtime
   coverage is confirmed after the recent lifecycle and encoded-constant work.
2. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
3. Continue GGUF/offload foundation with the next bounded runtime-supported
   slice, preferably explicit CPU/GPU residency-state transitions or load-time
   CUDA dequant staging that preserves the current dense runtime ABI.
4. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
