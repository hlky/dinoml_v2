# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added the bounded admission/planning slice for the future GGUF
  dequantize-before-GEMM path. CUTLASS GEMM manifests now mark a GGUF encoded
  constant used as the GEMM RHS with `materialization="dequantize_on_gpu_before_launch"`
  using a `gguf_runtime_dequant` record. The record carries qtype, encoded byte
  size, logical dense shape, session scratch byte size, and the intended handoff
  to the existing dense CUTLASS launcher.
- Generated CUDA GEMM lowering now rejects that planned policy with a precise
  native libgguf CUDA dequant launcher ABI message. This keeps the desired
  runtime policy artifact-visible and prevents it from being confused with the
  existing `RuntimeModule.load_encoded_constants()` load-time dense dequant
  branch, which still uses libgguf's Python/Torch CUDA op when available.
- Added focused planning coverage proving the manifest records the
  `gguf_runtime_dequant` plan for the new policy, does not mark the existing
  `dequantize_full_before_launch` path, and fails generated CUDA GEMM lowering
  at the intended ABI boundary.

## Ranked Backlog

1. With PM approval, add or depend on a native libgguf CUDA dequant launcher
   ABI that generated C++ can call, likely by adding `hlky/libgguf` as an
   explicit submodule/dependency boundary, then wire one narrow
   `gemm_rrr`/`gemm_rcr` GGUF RHS constant through a session-owned dense
   scratch buffer before the existing CUTLASS GEMM launcher.
2. Improve runtime/container lifecycle coverage for session/module close,
   allocator cleanup, and constant residency transitions before adding larger
   offload scheduling.
3. Continue GGUF/offload foundation with explicit CPU/GPU residency-state
   transitions, but keep them policy-visible and separate from the dense CUDA
   load-time dequant path.
4. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
