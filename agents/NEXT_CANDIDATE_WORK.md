# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added focused CUDA lifecycle coverage for the bounded GGUF runtime-dequant
  `gemm_rrr` path: unload now explicitly invalidates the encoded RHS residency
  for the live session, reloading encoded constants restores execution, closing
  a runtime-dequant module closes the live session before freeing the module,
  re-opening the artifact starts with unloaded encoded residency again, and
  repeated session/module close calls stay idempotent without stale loaded
  state.
- Added focused CUDA runtime-dequant test coverage for malformed GGUF encoded
  metadata on `load_encoded_constants(...)`: mismatched qtype,
  `encoded_nbytes`, and `n_per_row` now have explicit regression tests proving
  the runtime rejects the load before installing encoded bytes or mutating the
  per-constant loaded-state snapshot.
- Tightened the bounded GGUF runtime-dequant slice so
  `materialization="dequantize_on_gpu_before_launch"` is admitted only for the
  lowered CUDA `gemm_rrr` RHS path. Unsupported uses now fail clearly at
  compile/runtime admission instead of being reported as runtime-loadable
  encoded constants, and runtime load plans include precise blocked reasons.
- Cached native libgguf CUDA dequant launcher lookup by extension path so
  repeated encoded-constant loads do not reopen a new `ctypes.CDLL` handle.
- Updated README and architecture docs to describe the current narrow CUDA
  runtime-dequant contract and the still-unsupported surface.
- Landed the first bounded runnable GGUF dequantize-before-GEMM path for
  CUTLASS `gemm_rrr` with a GGUF RHS constant declared as
  `materialization="dequantize_on_gpu_before_launch"` and
  `residency="manual_runtime_load"`.
- Generated CUDA now stores that constant as encoded bytes, exposes an explicit
  runtime-set `libgguf_cuda_dequantize_rows_on_stream` boundary, allocates a
  separate session-owned dense RHS scratch buffer, dequantizes on the same
  session stream immediately before the existing dense CUTLASS GEMM launch, and
  fails precisely when the native launcher is unavailable.
- Runtime encoded-constant loading now has a CUDA branch for this policy that
  installs encoded GGUF bytes into generated module storage while preserving the
  older `dequantize_full_before_launch` dense load-time path.
- Added generated-code/lowering coverage for scratch allocation, encoded
  constant storage, native dequant call ordering, and missing-launcher failure,
  plus a focused CUDA integration test using real libgguf `Q4_0` RHS storage
  compared against a dense dequantized GEMM reference.

## Ranked Backlog

1. Stabilize the new bounded `gemm_rrr` GGUF runtime-dequant path with the
   remaining runtime failure-mode coverage, especially precise missing native
   launcher behavior at runtime for encoded runtime-dequant loads versus run
   time, while preserving the public loaded-state snapshot.
2. Improve broader runtime/container lifecycle coverage for session/module
   close, allocator cleanup, and constant residency transitions beyond the new
   bounded runtime-dequant regression before adding larger offload scheduling.
3. Consider the next narrow GGUF RHS GEMM extension only after the `gemm_rrr`
   path is stable: likely `gemm_rcr` or a base GEMM epilogue, still using
   explicit encoded storage, same-stream native dequant, and session-owned
   scratch.
4. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
