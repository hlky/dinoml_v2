# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added focused CUDA allocator/session lifecycle regression for the missing
  `_cuda_runtime_dll` cleanup path: when the CUDA helper handle is absent during
  session cleanup, staged buffers are cleared and the session teardown still
  proceeds.
- Added focused CUDA allocator/session lifecycle regressions around the
  remaining cleanup/retry edge cases: a failed staging-buffer grow now has a
  regression proving the newly allocated buffer is rolled back when the old
  buffer free fails, and `Session.close()` now has a regression proving that a
  cleanup failure followed by a native destroy failure still leaves the session
  retryable until both paths succeed on a later close.
- Added CUDA reopen-parity lifecycle coverage for mixed dense and
  manual-runtime-load encoded constants, matching the CPU regression slice:
  reopening an eager artifact resets the encoded constant back to unloaded, and
  closing a deferred artifact with a live session restores both constants to
  their initial deferred residency state instead of leaking prior runtime loads
  across module instances.
- Added broader CPU runtime/container lifecycle coverage for mixed dense and
  manual-runtime-load encoded constants: reloading still requires an explicit
  encoded load after `unload_constants()`/`load_constants_from_file()`, closing
  and reopening an eager artifact resets the manual residency bit back to
  unloaded, and closing a deferred artifact with a live session restores the
  initial deferred residency state on reopen instead of leaking prior runtime
  loads across module instances.
- Added focused CUDA runtime-dequant coverage for the remaining native-launcher
  failure mode on the bounded GGUF `gemm_rrr` path: `load_encoded_constants(["weight"])`
  now has a regression test proving that a missing
  `libgguf_cuda_dequantize_rows_on_stream` symbol fails before encoded bytes are
  installed and leaves `constant_load_state()` untouched, and `session.run_*`
  now has a regression test proving that clearing the module's launcher pointer
  after encoded load fails with the generated missing-launcher error instead of
  falling back to dense dequant.
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

1. Consider the next narrow GGUF RHS GEMM extension only after the `gemm_rrr`
   path is stable: likely `gemm_rcr` or a base GEMM epilogue, still using
   explicit encoded storage, same-stream native dequant, and session-owned
   scratch.
2. Revisit CUTLASS only for another bounded compile-visible robustness slice,
   such as persistent cache concurrency, if it directly affects provider
   selection or compile/profile correctness.
