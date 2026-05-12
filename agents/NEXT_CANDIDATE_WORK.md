# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

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

1. Stabilize the new bounded `gemm_rrr` GGUF runtime-dequant path with lifecycle
   and failure-mode coverage, especially encoded constant unload/reload,
   missing native launcher behavior at runtime, and malformed encoded byte-size
   handling.
2. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
3. Consider the next narrow GGUF RHS GEMM extension only after the `gemm_rrr`
   path is stable: likely `gemm_rcr` or a base GEMM epilogue, still using
   explicit encoded storage, same-stream native dequant, and session-owned
   scratch.
4. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
