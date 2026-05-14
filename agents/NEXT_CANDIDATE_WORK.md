# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Landed the first bounded rotary table-generation slice as a helper-only
  public `dml.ops.get_1d_rotary_pos_embed(...)`, deliberately without adding a
  new op/provider/kernel family or any fused `apply_rotary_emb` ABI. The new
  helper stays out of `OP_REGISTRY`, composes existing `arange`, fp32 trig,
  `repeat_interleave`, `concatenate`, and `cast` primitives, and returns a
  public `(cos, sin)` tensor pair rather than a coupled Q/K apply result.
  The admitted contract is explicit: positive even static `dim`; `pos` as
  either a positive integer sequence length or a rank-1 dense
  `float32`/`float16`/`bfloat16` tensor with static positive length; positive
  finite `theta`/`linear_factor`/`ntk_factor`; explicit duplicated-real output
  variants through `repeat_interleave_real=True` (repeat-interleave pairs) or
  `False` (concat/split-half style); and explicit output `dtype` across the
  same float surface with internal fp32 math. Focused tests now pin helper-only
  admission, reference parity against the v1/diffusers-style formula for both
  duplication conventions and reduced-precision outputs, static tensor-position
  parity, and clear frontend rejection for odd dims, `use_real=False`, dynamic
  lengths, bad dtypes, and non-finite scaling parameters. No standalone CUDA
  parity is claimed yet for this helper-only slice.
- Captured the recent RoPE/apply-rotary exploration as durable project memory
  before any implementation starts. Added `agents/plans/rotary_apply_plan.md`
  to pin the old `/workspace/apply_rotary_emb` prototype's real ABI and limits:
  CUDA-only Torch extension, coupled Q/K pair contract, contiguous rank-4
  `[B,H,S,D]` inputs plus rank-2 `[S,D]` cos/sin tables, effective float32-only
  behavior through raw `data_ptr<float>()`, real-pair layout switching via
  `use_real_unbind_dim`, two-tensor return reality, and missing validation for
  dtype/shape compatibility and honest complex support. The new plan also
  records the broader variant taxonomy across v1/diffusers/transformers
  (split-half, interleaved, partial-prefix, complex, multi-axis, scaled-table,
  rotate-V, and cache-order-sensitive families), separates table generation
  from application kernels, and recommends the first bounded v2 slice as
  `get_1d_rotary_pos_embed` table generation with duplicated-real variants
  rather than a fused public `apply_rotary_emb` ABI.
- Landed the bounded weight-optional normalization helper slice:
  public `dml.ops.rms_norm(x, weight=None, eps=1e-6)` now stays helper-only and
  reuses the existing `t5_layer_norm` generated CPU/CUDA backend instead of
  adding a new op, provider, or kernel family. The weighted path delegates
  directly to `t5_layer_norm`, while the weightless path materializes a
  same-dtype static ones vector `[hidden]` and delegates to that same node, so
  lowered IR still contains only `t5_layer_norm`. Added focused regressions
  proving the helper stays out of `OP_REGISTRY`, that the weighted IR is the
  same single `t5_layer_norm` node, that the unweighted IR adds only a ones
  constant plus `t5_layer_norm`, that CPU parity holds for weighted/unweighted
  `float32`/`float16`/`bfloat16`, that CUDA runtime parity works for one
  weighted and one unweighted `float32` case, and that dynamic hidden size,
  bad rank/dtype, weight shape mismatch, and mixed builder/dtype contracts fail
  clearly without widening the normalization/provider surface.
- Advanced the bounded ConvNd/CUTLASS maturity lane by turning the existing
  `cutlass_conv` manifest/codegen scaffold into a manifest-only support-cache
  scaffold: CUDA compile still rejects before module build, but it now writes
  `lib/cutlass_conv_manifest.json` and `src/source_manifest.json` under the
  advertised support `cache_dir`, carrying the used candidate plan, candidate
  config keys, and explicit NCHW/OIHW -> NHWC/OHWI transform provenance. Added
  focused regressions that prove the Conv used-candidate plan now preserves the
  scaffold candidate payloads and that a failing CUDA compile still materializes
  the support scaffold before the expected `manifest_scaffold_only` rejection.
- Closed the reduced-precision CUDA `get_timestep_embedding` helper gap
  without widening the pass or op surface: the helper now unsqueezes timesteps
  before casting to fp32, which keeps the internal math in fp32 when practical
  but moves the input cast onto the elementwise side of the view boundary so
  it fuses with the multiply/sin/cos chain instead of surviving as a raw CUDA
  `cast` op. Added a focused lowered-graph regression that proves the first
  node becomes `fused_elementwise` with `cast`/`mul`/`sin`/`cos`, plus reduced-
  precision CUDA runtime parity tests for `float16` and `bfloat16`. Dynamic
  `N` remains out because `concatenate` is still static-shape only.
- Fixed the narrow generated-CUDA `concatenate` wrapper bug that blocked
  intermediate producer outputs from feeding `concatenate`: the wrapper-local
  CUDA launch now passes wrapper parameters `x0`, `x1`, ... into the generated
  kernel instead of undefined IR pointer names like `ptr_t3`. Added a focused
  fused-elementwise (`sin`/`cos`) -> `concatenate` CUDA regression that checks
  the generated launch site and compiles/runs the CUDA artifact, then extended
  `get_timestep_embedding` coverage with honest float32 CUDA compile parity for
  even-width plus odd-width/`flip_sin_to_cos` cases and a representative
  even-width float32 CUDA runtime regression. The helper contract is now
  updated accordingly: dynamic `N` remains out because `concatenate` is still
  static-shape only, while reduced-precision CUDA parity is still not claimed
  because the separate raw-cast-across-view admission/fusion gap remains.
- Landed the next bounded small/custom helper slice around
  `get_timestep_embedding`: public `dml.ops.get_timestep_embedding(...)` is now
  a helper-only composition over existing v2 primitives instead of a new
  registered op, provider, or custom kernel family. The helper keeps the
  Diffusers/v1 sinusoid contract for rank-1 dense floating timesteps with
  finite attrs, precomputes the static frequency vector as a traced constant,
  does the internal trig/frequency math in fp32 when practical, preserves the
  input float storage dtype on the public output, swaps sin/cos halves when
  requested, and appends the odd-width zero column. Focused tests now pin even
  and odd embedding widths, `flip_sin_to_cos`, `downscale_freq_shift`, `scale`,
  `max_period`, dtype preservation, the `embedding_dim == 1` zero-column edge,
  and rejection of dynamic timestep length, bad rank/dtype, and invalid
  parameter combinations. The honest current bounds are documented in the
  checklist: dynamic `N` is still out because `concatenate` remains static-only,
  and CUDA parity is now proven for float32 plus reduced-precision runtime
  slices without expanding the helper into a dedicated op/provider surface.
- Landed the smallest honest v1/HuggingFace custom-op helper slice around
  `gelu_new`: public `dml.ops.gelu_new(x)` is now a bounded frontend helper
  that rewrites directly to the existing tanh-approximation `gelu` op instead
  of expanding provider or kernel surface. Focused tests now pin that the
  traced IR is identical to `gelu`, the lowered path stays on
  `fused_elementwise`, CPU reference execution matches the HuggingFace/v1 tanh
  GELU-new formula, CUDA codegen uses the existing `dinoml::math::gelu` path,
  and helper-only admission stays honest by keeping `gelu_new` out of
  `OP_REGISTRY` while delegating unsupported dtype rejection to `gelu`.
- Closed the reviewer-noted reduced-precision CUDA runtime gap for the bounded
  `t5_layer_norm` slice: `float16` and `bfloat16` now have a numeric CUDA
  runtime parity regression in `tests/test_t5_layer_norm_ops.py`, using the
  existing trace/reference helpers and the generated CUDA artifact path. The
  source-side fp32-accumulation assertions stayed in place, so the bounded
  T5/RMSNorm contract remains unchanged while the reduced-precision runtime
  path is now exercised directly.
- Landed the first bounded normalization slice away from the Conv metadata
  lane: public `t5_layer_norm` now covers the T5/RMSNorm-style form
  `x * rsqrt(mean(x^2) + eps) * weight` over rank >= 1 dense tensors with a
  positive static last dimension and required affine weight `[hidden]`, across
  `float32`, `float16`, and `bfloat16` storage. The slice keeps fp32
  accumulation semantics, preserves dynamic leading-dimension shape metadata,
  adds CPU reference execution plus generated CPU/CUDA kernels, and has focused
  frontend/IR rejection coverage for dynamic hidden size and bad weight
  contracts. The docs/checklist now call this out explicitly as a bounded
  RMS/T5-only slice; full LayerNorm, grouped, sigmoid-mul, adaptive, and
  provider-backed normalization variants remain unimplemented.
- Closed a reviewer-found P1 in the bounded `cutlass_conv` profiling scaffold:
  scaffold-only `ConvProfileWorkload` objects now fail explicitly before the
  GEMM/BMM-only profiling cache-key, profile-result, cache read/write, or
  execution-plan code can touch them. `profile_artifact(...)` also rejects
  unsupported Conv scaffold workloads at the profiling boundary for future
  safety, and focused profiling regressions pin the new error contract so
  scaffold-only Conv results cannot silently disappear from execution-plan
  generation.
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

1. Keep the small/custom-op lane on honest helper or bounded-op slices:
   with `gelu_new`, `get_timestep_embedding`, and the bounded helper-only
   `rms_norm` plus `get_1d_rotary_pos_embed` slices closed, prefer the next
   smallest candidate such as `cropped_pos_embed` or the next bounded rotary
   follow-up before revisiting
   broader LayerNorm, GroupNorm, fused sigmoid/swish variants, or dynamic
   normalized-dimension work. RoPE exploration/planning is now recorded in
   `agents/plans/rotary_apply_plan.md`; the next honest rotary slice is now a
   downstream consumer of the landed 1D tables, such as a bounded one-tensor
   real-pair application helper or a 2D/3D table-preparation helper, not a
   speculative fused public `apply_rotary_emb` CUDA ABI. If the next rotary
   helper needs dynamic concatenation or compiled CUDA parity, first decide
   whether that belongs in the helper slice or in the existing
   collection/codegen gaps called out by `get_timestep_embedding`.
2. Continue the first bounded ConvNd provider slice described in
   `agents/plans/conv_cutlass_plan.md` by connecting the existing
   `conv2d_bias` public/reference surface, `cutlass_conv`
   `manifest_scaffold_only` compile metadata, and profile workload scaffold to
   the next honest provider step. The support-cache/source-manifest scaffold is
   now in place, so prefer the next small artifact-visible increment such as a
   generated pack/unpack lowering metadata/test slice or another narrow
   codegen-plan/profiler-provenance follow-up before attempting a full CUTLASS
   runtime.
   Keep the work narrow:
   no conv3d, no transposed/depthwise/grouped expansion, no hidden channel
   padding, no runtime-set packed weights, and no public NHWC toggle.
3. Revisit CUTLASS/provider maturity only for another bounded compile-visible
   robustness slice if a new concrete stale-payload edge appears in an existing
   cache/test area; otherwise keep provider-cache work paused and avoid
   speculative broadening.
4. Add one more bounded native regression only if another GGUF loader edge
   appears, preferably around encoded-runtime-dequant native reload behavior
   rather than broadening the runtime surface.
