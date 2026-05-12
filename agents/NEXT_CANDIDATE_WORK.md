# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Probed the proposed libgguf CUDA dequant staging dependency before wiring it
  into `RuntimeModule.load_encoded_constants()`. `libgguf` imports from
  `/workspace/libgguf`, CUDA and Torch are available, and
  `libgguf.libgguf_cuda` imports, but the optional Torch extension op is not
  registered (`torch.ops._C_gguf.dequantize` is absent). The upstream
  `test_cuda_dequantize_matches_libgguf` cases all skip for that reason in this
  environment, so v2 should not add a CUDA encoded-constant branch until the
  extension is built and a real dequant call succeeds.
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

1. Make the libgguf CUDA dequant dependency runnable, then add load-time CUDA
   dequant staging for GGUF-backed constants while preserving the dense runtime
   ABI. Bounded first step: build/install `/workspace/libgguf` with
   `LIBGGUF_BUILD_CUDA_KERNELS=ON`, prove
   `torch.ops._C_gguf.dequantize` works on a small `Q4_0` tensor, then wire
   `RuntimeModule.load_encoded_constants()` to dequantize into a CUDA tensor
   and call the existing `set_constant_device_pointer` path for CUDA artifacts.
2. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
3. Continue GGUF/offload foundation with explicit CPU/GPU residency-state
   transitions only after the CUDA dequant dependency boundary above is cleared.
4. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
