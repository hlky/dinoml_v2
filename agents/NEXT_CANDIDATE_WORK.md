# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added the bounded CUDA runtime branch for GGUF encoded constants. When
  `RuntimeModule.load_encoded_constants()` loads a CUDA artifact and
  `libgguf.libgguf_cuda` has a registered `torch.ops._C_gguf.dequantize`, the
  runtime now reads the packed GGUF rows, dequantizes supported rows such as
  real `Q4_0` storage into a CUDA torch tensor, synchronizes that tensor, and
  installs it with the existing dense `set_constant_device_pointer` path.
  CPU artifacts, missing Torch/libgguf CUDA extensions, and dense GGUF `F32` or
  `F16` storage still use the existing host materialization plus
  `set_constant_numpy` fallback. No new residency mode, scheduler, prefetch, or
  public op surface was added.
- Added focused CUDA integration coverage that compiles a real libgguf `Q4_0`
  GGUF-backed constant artifact with `manual_runtime_load`, proves the runtime
  load does not call CPU `libgguf.dequantize_rows`, verifies
  `constant_load_state()`, unload/reload behavior, and output correctness.
  The existing real-libgguf CPU encoded test and frontend encoded manifest tests
  still pass.
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
2. Continue GGUF/offload foundation with explicit CPU/GPU residency-state
   transitions, but keep them policy-visible and separate from the dense CUDA
   load-time dequant path.
3. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
