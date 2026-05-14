# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Connected the existing bounded `conv2d_bias`/`cutlass_conv` scaffold to the
  first profile-visible provider step without adding a runtime launcher:
  `build_profile_workloads(...)` now emits a `cutlass_conv` workload scaffold
  from the manifest's explicit NCHW/OIHW semantic metadata, NHWC/OHWI provider
  metadata, layout-pack/unpack plan, weight-transform metadata, Conv2d attrs,
  shapes, candidates, and profiler symbol. The scaffold refuses manifests that
  omit `cutlass_conv_plan` transform metadata, preserving the artifact-visible
  layout contract. Added focused tests for the emitted workload JSON and the
  missing-transform guard. CUDA compile still rejects before module build with
  `manifest_scaffold_only`; no CUTLASS Conv runtime, support build, or profiler
  execution is claimed.
- Started the bounded ConvNd provider lane without claiming a CUDA runtime yet:
  added a public/reference-only `conv2d_bias` surface with NCHW activation,
  OIHW weight, bias `[Cout]`, groups=`1`, static rank-4/static channel+kernel
  limits, and CPU reference parity against PyTorch. CUDA compile now reaches a
  `cutlass_conv` manifest/codegen scaffold that records the intended NHWC/OHWI
  provider layout and explicit layout/weight-transform metadata as
  `manifest_scaffold_only`, then rejects before module build until a real
  launcher exists; CPU compile still rejects at backend admission.
- Tightened CUTLASS support-cache/source-manifest reuse with another bounded
  compile-visible robustness slice: cache hits now also reject
  `src/source_manifest.json` payloads whose embedded `used_candidate_plan`
  content no longer hashes to the stored `used_candidate_plan_key`, even when
  the top-level manifest key remains self-consistent. Added a focused
  backend-registry regression in the existing source-manifest test area that
  mutates the embedded selected-candidate payload, recomputes the outer
  `source_manifest_key`, and proves the support library rebuilds instead of
  reusing stale provider provenance.
- Tightened the CUTLASS profile-cache persistence contract with a small
  compile-visible robustness slice: cache reads now reject entries whose
  embedded key payload no longer hashes to the stored `profile_key` or whose
  embedded target drifts from the cache target, and cache writes now drop those
  stale on-disk payloads instead of merging them back. Added focused profiling
  regressions for stale embedded payload hashes, cross-target payload drift, and
  stale-on-disk entry rejection during a normal write.
- Closed the top-ranked trust-building GGUF/CUDA workflow gap with a focused
  float32 `gemm_rrr_bias` runtime regression: real libgguf `Q4_0` RHS storage,
  dense bias loaded from `constants.bin`, `manual_runtime_load` encoded weight,
  load -> run -> unload -> reload plus reopen, and dense reference comparisons
  across each successful execution. Updated the gap audit to describe the
  proven dense-bias lifecycle slice instead of only one-shot correctness.
- Closed the reviewer follow-up on native manual GGUF autoload parity: added a
  bounded direct CUDA native-boundary regression for mixed dense plus
  `manual_runtime_load` constants that mirrors the CPU ABI test shape. The test
  now calls generated native CUDA `dino_module_load()`,
  `dino_module_load_constants()`, `dino_module_set_constant()`, and
  `dino_session_run()` directly enough to prove `constants.bin` does not eagerly
  materialize the manual GGUF constant, and that after native unload/reload the
  module still requires an explicit native load/set before run. Updated the gap
  audit to describe the proven CPU/CUDA native coverage precisely.
- Closed the remaining native GGUF load-path parity gap for mixed dense plus
  `manual_runtime_load` constants: generated CPU/CUDA native
  `dino_module_load_constants()` now skips eager `constants.bin`
  materialization for any GGUF constant that declares
  `residency="manual_runtime_load"`, matching the Python open/reload contract
  instead of only the lowered CUDA runtime-dequant slice. Added direct CPU
  native-boundary coverage proving `dino_module_load()` and
  `dino_module_load_constants()` leave the manual GGUF constant unloaded across
  eager open and reload until an explicit setter call, plus mixed CPU/CUDA
  generated-source regressions that pin the skip path.
- Tightened the bounded GGUF runtime-dequant -> CUTLASS GEMM contract so only
  `residency="manual_runtime_load"` produces
  `lowered_runtime_dequant_scratch`: manifest planning now marks non-manual
  residency as `planned_not_lowered` with a clear residency-specific blocked
  reason, compile admission now fails with an explicit manual-residency
  requirement, and generated CUDA native load paths no longer eagerly materialize
  lowered encoded runtime-dequant constants from `constants.bin`. Added focused
  planning/generated-code coverage for the native eager-load skip plus a
  non-manual-residency planning/admission regression. Follow-up to keep in view:
  audit whether the non-Python native `dino_module_load_constants()` path should
  also honor `manual_runtime_load` for older dense GGUF policies, since this
  loop intentionally fixed the encoded runtime-dequant slice only.
- Closed the shared-scratch coverage gap for bounded GGUF runtime-dequant
  CUTLASS GEMMs: added a focused planning/codegen regression proving that
  multiple lowered runtime-dequant GEMM nodes in one CUDA artifact share a
  single session-owned scratch allocation sized to the maximum dense RHS while
  each launch still checks its own required scratch bytes before native
  libgguf dequant.
- Closed the reviewer follow-up gap in the bounded GGUF runtime-dequant CUDA
  coverage: added a focused `gemm_rcr_bias` float16 integration regression
  that uses real libgguf `Q4_0` RHS storage, dense bias, same-stream native
  dequant, and a dense reference comparison. This keeps the support surface
  unchanged while proving the bias + float16 runtime slice directly on CUDA.
- Extended the bounded CUDA GGUF runtime-dequant-before-GEMM path from base
  GEMM to the bias-only epilogue slice: manifests now lower RHS GGUF constants
  with `materialization="dequantize_on_gpu_before_launch"` and
  `residency="manual_runtime_load"` for `gemm_rrr_bias`/`gemm_rcr_bias`
  `float32`/`float16` outputs, compile/runtime admission accepts those bias
  uses alongside the base `gemm_rrr`/`gemm_rcr` path, and generated CUDA reuses
  the same-stream native libgguf dequant into session-owned dense RHS scratch
  before the existing dense CUTLASS bias launcher. Added planning/admission/
  lowering coverage for both layouts and a real CUDA integration test using
  libgguf `Q4_0` RHS storage plus dense bias compared against a dense reference.
- Extended the bounded CUDA GGUF runtime-dequant-before-GEMM path from base
  `gemm_rrr` to base `gemm_rcr`: manifests now lower RHS GGUF constants with
  `materialization="dequantize_on_gpu_before_launch"` and
  `residency="manual_runtime_load"` for `float32`/`float16` outputs, admission
  accepts only base `gemm_rrr`/`gemm_rcr` RHS uses, and generated CUDA reuses
  same-stream native libgguf dequant into session-owned dense RHS scratch. The
  dense CUTLASS RCR launcher consumes that scratch through the existing
  column-major RHS ABI. Added planning/codegen/admission coverage plus a CUDA
  integration test using real libgguf `Q4_0` `gemm_rcr` RHS storage compared
  against a dense dequantized reference, plus a float16 CUDA runtime regression
  that covers encoded load, runtime dequant, and CUTLASS RCR handoff with a
  reduced-precision tolerance.
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
- Added focused CUDA runtime-dequant coverage for remaining native-launcher failure modes on
  bounded GGUF `gemm_rrr`/`gemm_rcr` paths: `load_encoded_constants(["weight"])`
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

1. Continue the first bounded ConvNd provider slice described in
   `agents/plans/conv_cutlass_plan.md` by connecting the existing
   `conv2d_bias` public/reference surface, `cutlass_conv`
   `manifest_scaffold_only` compile metadata, and profile workload scaffold to
   the next honest provider step. Prefer a small source-manifest scaffold or
   generated pack/unpack lowering test before attempting a full CUTLASS runtime.
   Keep the work narrow:
   no conv3d, no transposed/depthwise/grouped expansion, no hidden channel
   padding, no runtime-set packed weights, and no public NHWC toggle.
2. Revisit CUTLASS/provider maturity only for another bounded compile-visible
   robustness slice if a new concrete stale-payload edge appears in an existing
   cache/test area; otherwise keep provider-cache work paused and avoid
   speculative broadening.
3. Add one more bounded native regression only if another GGUF loader edge
   appears, preferably around encoded-runtime-dequant native reload behavior
   rather than broadening the runtime surface.
