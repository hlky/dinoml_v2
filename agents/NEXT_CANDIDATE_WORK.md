# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Added generic full-artifact session benchmarking at the generated runtime
  boundary. CPU/CUDA/ROCm generated modules now export
  `dino_session_benchmark`, which runs warmups plus measured iterations around
  `dino_session_run` and returns per-iteration milliseconds to Python. The
  runtime exposes `Session.benchmark_numpy(...)` and
  `Session.benchmark_device_pointers(...)`, and the CLI now has
  `dinoml benchmark <artifact> --against <inputs.py>` for model-level timing
  without provider-selection side effects. While validating on Windows, the CPU
  backend was taught to use the Visual Studio 2022 `x64` CMake generator by
  default, emit/load platform-native CPU DLL artifact names, and link generated
  modules through MSVC import libraries when they exist. Validation covered the
  benchmark unit/CLI/template tests plus a real `.venv/rocm` CPU elementwise
  artifact compile/run smoke with the MSVC generator.
- Admitted ROCm generated HIP modules for the simple generated-template op
  families. CUDA-only `*_cuda.cu.j2` simple templates were moved to shared
  `*_gpu.j2` templates using explicit target facts for stream type, error
  checks, async memset, storage type, and warp shuffle masks; ROCm op registry
  entries now route those simple ops through the shared lowering path. Added an
  opt-in ROCm artifact contract under `tests/rocm/` that compiles, loads, runs,
  and reference-checks every non-provider standard case from `.venv/rocm`.
  The test uses the normal cache instead of patching `DINOML_CACHE_DIR`.
  Provider-backed GEMM/BMM/Conv and CK remain outside the admitted ROCm runtime
  surface.
- Added the first v1-inspired cross-backend lowering metadata slice without
  admitting ROCm op support: generated-code target facts now live in
  `dinoml.lowering.target_specs`, `full`/`arange` use shared template loading
  plus a parameterized GPU template for CUDA stream/error/storage names, and
  the common `dinoml/math.h` / `tensor_accessor.h` headers are HIP-aware for
  future shared HIP kernels. ROCm target facts are visible (`.hip`,
  `hipStream_t`, `DINO_ROCM_CHECK`, HIP-backed `half`/`dinoml::bfloat16`
  storage names), and this scaffold has since been superseded by the admitted
  simple generated HIP artifact path recorded above. Added an opt-in real `hipcc`
  header-compile smoke for `device.h`, `runtime_rocm.h`, `math.h`, and
  `tensor_accessor.h` under `DINOML_RUN_ROCM_HEADER_COMPILE_SMOKE=1`, so the
  shared-header ROCm compile check is now in pytest instead of only being a
  manual validation command. The shared `dinoml/device.h` now carries the v1-
  style CUDA/HIP aliases (`dinoml::bfloat16`, `dinoml::DeviceStream`, `LDG`)
  that generated GPU code should use instead of spelling backend-specific
  types in every op. Installing `pytest` into `.venv/rocm` exposed one real
  environment bug: direct `.venv/rocm/Scripts/python.exe` smoke runs were still
  resolving ROCm through PATH/system defaults instead of the active interpreter.
  The Python ROCm backend now tries `sys.executable -m rocm_sdk` before PATH
  Python fallbacks, and the opt-in real support-library smoke passes from the
  ROCm venv.
- Tightened the ROCm scaffold environment contract after review: ROCm support
  builds now assume the developer has activated `.venv/rocm`, resolve the SDK
  through the active `rocm_sdk` package before `hipconfig` so PyTorch-style
  installs use `_rocm_sdk_devel` rather than `_rocm_sdk_core`, remove the
  separate `DINOML_ROCM_PYTHON_EXECUTABLE` path, require Ninja for ROCm support
  builds, and filter the imported Visual Studio environment to the MSVC/Windows
  SDK keys needed for clang/link/rc instead of copying every `vcvars64.bat`
  value.
- Added the first honest ROCm backend scaffold for the Windows `.venv/rocm`
  lane. `Target("rocm")` now resolves to `gfx1201`, the CLI lists `rocm`, CMake
  resolves pip/venv ROCm SDK layouts through `rocm-sdk` on `PATH`, and the HIP
  support-library smoke builds `dinoml_runtime`, `dinoml_rocm_runtime`, and
  `dinoml_rocm_kernels` with the local toolchain. This was the pre-admission
  scaffold before the later generated HIP artifact path landed.
  Validation covered registration/codegen-plan fences plus the opt-in real ROCm
  support build under `.venv/rocm`.
- Added ROCm Composable Kernel as a plain third-party source submodule under
  `third_party/composable_kernel`, pinned to AMD's `rocm-7.2.3` release tag
  from the `ROCm/composable_kernel` mirror. The docs now record why the mirror
  is used instead of a sparse checkout of the ROCm monorepo: AMD's canonical
  source is `ROCm/rocm-libraries/projects/composablekernel`, but the mirror
  keeps CK at repository root and avoids hidden local sparse-checkout state in
  fresh clones or CI. This is dependency provenance only and does not broaden
  any ROCm provider/runtime support surface.
- Finished the CUTLASS source-layout refactor around the intended provider
  files: `kernels/cuda/src/` now carries `cutlass_bmm.cu`, `cutlass_conv.cu`,
  `cutlass_gemm.cu`, and shared `cutlass_common.cuh`, while the checked-in
  `cutlass_gemm_units/` instantiation directory has been removed. CMake now
  generates chunked GEMM op/dtype instantiation sources from `cutlass_gemm.cu`
  under the build tree using `tools/generate_cutlass_gemm_unit.py`, matching
  the existing BMM/Conv generated-unit flow while preserving GEMM compile
  parallelism and the existing op/dtype static archive target names. GEMM
  support provenance now hashes the shared common
  header, GEMM template, generator, and GEMM candidate/provider descriptors so
  profile/support manifests notice source or generation changes.
- Rebuilt the test suite around intent-specific contract coverage after moving
  the eager NumPy graph interpreter out of `dinoml.backends.cpu` into
  `dinoml.reference.reference_numpy`. `tests/ir/` now checks frontend/IR shape
  and reference-executor contracts, `tests/cpu/` compiles and runs CPU
  artifacts against the reference, and `tests/cuda/` compiles and runs CUDA
  artifacts when the CUDA toolchain is present. The fresh matrix covers the
  current broadly supported op families across IR/CPU/CUDA, with provider-gap
  exceptions for GEMM/BMM/Conv kept to IR admission coverage. The rewrite also
  fixed two codegen bugs caught by compiled tests: fused elementwise float
  literals now render valid C++ (`0.0f` instead of `0f`), and CUDA stack wrapper
  launches pass wrapper parameters rather than module-scope pointer names.
- Cleaned up the GGUF runtime-dequant support boundary so CUDA artifacts no
  longer expose or depend on
  `dino_module_set_libgguf_cuda_dequantize_rows_on_stream()`. DinoML CMake now
  owns explicit libgguf CUDA object/static-archive targets from vendored
  libgguf sources, generated modules directly call the linked
  `libgguf_cuda_dequantize_rows_on_stream(...)` symbol, and Python runtime code
  no longer resolves native libgguf CUDA symbols through `ctypes`; Python
  libgguf/Torch dequant remains available only for load-time dense
  materialization before a run. Focused tests/docs were updated to pin the
  direct-link contract and remove the old fallback expectations.
- Extended the cached-checkpoint CLIP benchmark JSON to expose CUDA-side
  DinoML hot-path timings without changing model/runtime behavior. The CUDA
  benchmark now always reports GPU-resident `Session.run_device_pointers`
  latency using preallocated torch CUDA tensors plus explicit input/output
  shapes, retains the existing `run_numpy` and Transformers timings, and emits
  `cuda_run_numpy_overhead_ms` so `run_numpy` host/device staging overhead can
  be compared directly against the device-pointer path. `Session.run_torch`
  remains best-effort metadata only for this CLIP harness because current
  CUDA torch entrypoint dtype checks reject CLIP `torch.int64` token inputs;
  the report now records that unavailability instead of failing the benchmark.
  Focused tests cover the CUDA schema and helper behavior with fakes, and the
  cached CPU benchmark smoke still passes. This is benchmark instrumentation
  only: no runtime dtype-policy change, no new CUDA frontend contract, and no
  CLIP model parity surface expansion.
- Added a repo-visible cached-checkpoint CLIP benchmark harness without
  changing CLIP model/runtime behavior. The new
  `tools/benchmark_clip_checkpoint.py` reuses the existing deterministic
  `examples/clip_checkpoint_workflow.py` checkpoint loading, synthetic inputs,
  target limits, and parity helpers, then reports JSON for checkpoint id,
  target, input/output shapes, parity/max absolute differences, compile time,
  runtime load/session creation time, DinoML `run_numpy` latency, and local
  Transformers forward latency. CUDA runs synchronize around timed sections and
  keep the same cached local checkpoint assumptions as the workflow. Focused
  tests cover CLI parsing, timing/schema helpers with stubbed runtime work, and
  a guarded cached-base CPU benchmark smoke. This is benchmark instrumentation
  only: no tokenizer/processor runtime plumbing, no broader CLIP checkpoint
  support claim, and no provider/runtime behavior changes.
- Closed the remaining real CUDA `profile_artifact(..., refresh=True)` proof
  gap for public no-bias `dml.ops.conv2d(...)` without adding a separate
  no-bias Conv provider ABI or broadening the static rank-4 groups=1 NCHW/OIHW
  contract. The new CUDA-gated smoke compiles the public no-bias Conv artifact,
  runs the real artifact profiler, and asserts `debug/profile_report.json` plus
  `debug/execution_plan.json` preserve `source_op=conv2d`,
  `bias_mode=explicit_zero_constant`, selected CUTLASS Conv candidate/kernel/
  profiler metadata, and the existing `dinoml_cutlass_conv2d_bias_v1` launch
  ABI. The smoke stays confidence-aware like the fused Conv profile smokes,
  accepting either a consumable static selection or an explicit low-confidence
  non-selection while still requiring the bridge/provider metadata to remain
  intact.
- Closed the public no-bias `dml.ops.conv2d(...)` profile/candidate-selection
  evidence gap without adding a new provider ABI or broadening the Conv
  contract. The explicit-zero bridge metadata (`source_op=conv2d` and
  `bias_mode=explicit_zero_constant`) now remains artifact-visible in the
  `cutlass_conv` manifest/plan, profile workload/report/cache payloads, static
  execution-plan selections, `execution_plan_selection`, and generated
  wrapper-stage metadata. A focused non-CUDA regression proves the bridge
  metadata and selected candidate survive profile-result construction,
  execution-plan selection, compile-time plan consumption, and generated
  lowering, and strict plan application now rejects missing, wrong, or spurious
  bridge metadata before it can be copied into `execution_plan_selection`, while
  the existing CUDA compile bridge smoke still passes. This is
  not a separate no-bias Conv provider family, hidden padding, grouped/
  depthwise/transposed/3D Conv, public NHWC toggles, runtime-set packed
  weights, or sigmoid Conv.
- Audited `fast_gelu` and `quick_gelu` against the local
  `/workspace/transformers` activation definitions and CLIP configs, then fixed
  a real CPU semantic split bug without changing public op names or CLIP
  routing. `fast_gelu` now uses the Transformers/v1 tanh/Hendrycks formula in
  the Python CPU reference, shared generated math helper, and generated naive
  CPU `gemm_rcr_bias_fast_gelu` epilogue, while `quick_gelu` remains the CLIP
  QuickGELU formula `x * sigmoid(1.702 * x)` and CLIP adapters continue to
  route `hidden_act="quick_gelu"` to `gemm_rcr_bias_quick_gelu`. Focused
  regressions now prove standalone CPU `fast_gelu`, CPU GEMM fast/quick GELU,
  and generated CPU GEMM epilogues stay distinct; validation reran the
  activation-focused CPU slice plus frontend/provider and CLIP routing tests.
  This does not broaden public op surface, add non-CLIP QuickGELU elementwise
  API, or change CUTLASS `gemm_rcr_bias_quick_gelu` behavior.
- Added a repo-visible cached-checkpoint CLIP workflow so the recent
  full-checkpoint proofs are discoverable outside the test suite. The new
  `examples/clip_checkpoint_workflow.py` defaults to cached
  `openai/clip-vit-base-patch32`, imports the local `/workspace/transformers`
  checkout, traces through the public CLIP adapter helpers, compiles a CPU or
  CUDA artifact, runtime-loads it, runs deterministic synthetic inputs, and
  prints a JSON parity summary with artifact paths, shapes, limits, allclose
  flags, and max-absolute diffs. The README now points to this workflow and
  updates the Conv runtime-support note to match the admitted bounded
  `conv2d_bias` family. Validation reran
  `tests/test_clip_checkpoint_workflow_example.py` from the PM side
  (`3 passed in 103.98s`) plus `git diff --check`. This was directly prompted
  by external-developer feedback that CLIP/Conv proofs were hard to discover;
  it does not add tokenizer/processor runtime plumbing, broaden checkpoint
  cache behavior beyond local files, or exercise the new workflow's CUDA path
  in CI-style tests.
- Advanced the known-checkpoint CLIP-L CUDA boundary from “unknown” to a real
  opt-in compiled artifact smoke. Cached `openai/clip-vit-large-patch14` now
  compiles, loads, and runs through the existing
  `DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CUDA_SMOKE=1` path via
  `DINOML_CLIP_CHECKPOINT_ID`, with the test report carrying the resolved
  checkpoint id and per-checkpoint limits. The default base checkpoint keeps
  the existing strict `1e-5` bar for logits and embeds; CLIP-L keeps embeds at
  `1e-5` but uses an explicit logits-only `2e-5` cap after the first run showed
  clean embeds (`~3e-7`) and scalar logits at `1.1444091796875e-05`, just over
  the base scalar threshold. Validation reran the compiled CUDA smoke for both
  cached `openai/clip-vit-large-patch14` and default
  `openai/clip-vit-base-patch32`. This is now an operational CUDA artifact
  parity-scale proof for CLIP-L, not tokenizer/processor support, broader
  batch/sequence/image coverage, or a claim that the final logit reduction has
  the exact same tolerance behavior as the smaller base checkpoint.
- Closed the next full-checkpoint CLIP-L compiled-artifact blocker. The
  non-legacy OpenAI CLIP text pooling branch (`eos_token_id != 2`) traces
  `input_ids == eos_token_id` as a pure fused `eq` node with `int64` inputs and
  bool output; validation already admitted integer `eq`, but the compiler's
  backend-wide dtype sweep still rejected the internal `fused_elementwise`
  node. The compiler/runtime contract now exempts only pure fused integer
  `eq` inputs (`int32`/`int64`) from that rejection, while preserving the
  existing guard that unrelated integer tensors remain unsupported. A focused
  CPU artifact regression compiles and runs `int64 eq -> bool` above
  float-exact range so the path cannot pass by float rounding. Validation
  reran targeted relational and argmax contract tests plus both cached
  `openai/clip-vit-base-patch32` and `openai/clip-vit-large-patch14` compiled
  CPU checkpoint smokes; the large checkpoint now compiles and executes as a
  CPU artifact. This does not broaden integer elementwise arithmetic generally
  or claim CUDA CLIP-L runtime parity.
- Closed the remaining profile-artifact proof gap for the newest admitted
  residual-plus-activation Conv surface `conv2d_bias_add_relu` without changing
  provider/runtime behavior or adding public op surface. The CUDA-gated smoke
  now mirrors the existing `conv2d_bias_relu` and `conv2d_bias_add` artifact
  profiling proofs: it compiles a tiny CUDA artifact, runs
  `profile_artifact(..., refresh=True)`, and asserts both
  `debug/profile_report.json` and `debug/execution_plan.json` preserve
  `op=conv2d_bias_add_relu`, `epilogue=bias_add_relu`,
  `epilogue_config={"inputs":["bias","d0"],"activation":"relu"}`, launch ABI
  `dinoml_cutlass_conv2d_bias_add_relu_v1`, selected launcher/profiler symbol
  identity, and the expected static-selection versus low-confidence behavior.
  Validation reran that exact CUDA-gated smoke plus the closest
  static-execution-plan and compile-lowering checks. The op-porting checklist
  now reflects the stronger proof set; remaining Conv work should move to a new
  concrete parity gap such as guarded/dynamic dispatch, broader dtype/runtime
  coverage, or the next safe epilogue, not more artifact-profile repetition.
- Closed the real CLIP-L checkpoint adapter-trace blocker caused by frontend
  constant aliasing on ephemeral explicit-zero Conv bridges. `GraphBuilder` now
  keys traced constants by the `Parameter` object itself instead of raw
  `id(parameter)`, so long traces cannot reuse a garbage-collected parameter id
  and accidentally hand a later `conv2d_bias` node a stale zero-bias constant
  from an earlier bridge. A focused regression now monkeypatches `id(...)` to
  collide on two distinct local `conv2d_zero_bias` parameters and proves the
  frontend still emits separate `[4]` and `[1024]` constants plus distinct bias
  inputs for the two no-bias `conv2d` nodes. Validation reran the targeted
  frontend regression and the cached CLIP checkpoint adapter-state smoke lane;
  this closes the known adapter-state admission blocker without broadening any
  CLIP-L runtime, tokenizer/processor, or provider maturity claims.
- Closed the next CLIP known-checkpoint usability gap at the processor/tokenizer
  boundary without changing runtime surface, provider claims, or model code.
  The new opt-in local-cache smoke
  `DINOML_RUN_CLIP_CHECKPOINT_PROCESSOR_SMOKE=1` loads cached
  `openai/clip-vit-base-patch32` model plus `CLIPProcessor` from the local
  `/workspace/transformers` checkout with `local_files_only=True`, builds one
  short prompt and one synthetic RGB PIL image, feeds the resulting processor
  `input_ids`, `attention_mask`, and `pixel_values` into the existing adapted
  `LegacyCLIPModel`, and compares DinoML CPU-reference logits/embeds against
  local Transformers on those same processor outputs. The smoke skips clearly
  when the cached processor/model or PIL-backed processor deps are unavailable.
  This proves DinoML can consume real Hugging Face processor outputs at the
  admitted boundary, but it is intentionally not runtime tokenizer/processor
  plumbing, compiled artifact coverage, CLIP-L, interpolation, or loss support.
- Closed the remaining user-visible profile-artifact gap for the already
  admitted residual Conv surface `conv2d_bias_add` without changing
  provider/runtime behavior. The new CUDA-gated smoke mirrors the existing
  `conv2d_bias_relu` artifact-profile proof: it compiles a tiny CUDA artifact,
  runs `profile_artifact(..., refresh=True)`, and asserts both
  `debug/profile_report.json` and `debug/execution_plan.json` preserve
  `op=conv2d_bias_add`, `epilogue=bias_add`,
  `epilogue_config={"inputs":["bias","d0"]}`, the residual launch ABI
  `dinoml_cutlass_conv2d_bias_add_v1`, top-level kernel/profiler symbol
  identity, and the expected static-selection versus low-confidence execution-
  plan behavior. Validation reran that exact CUDA-gated smoke plus the nearest
  residual static-plan/lowering regressions; no runtime/provider code changes
  were needed. Remaining Conv work should move to the next bounded
  profile-selection or runtime-parity gap instead of revisiting this residual
  profile-artifact path.
- Closed the Conv profiling/execution-plan proof gap for the already admitted
  residual public surface `conv2d_bias_add` without changing provider/runtime
  behavior. The focused profiling regressions now match the existing
  `conv2d_bias_relu` and `conv2d_bias_add_relu` static-plan discipline: one
  test proves static execution-plan application preserves explicit `bias_add`
  epilogue metadata plus the selected candidate id/symbols in both
  `required_kernels[*]` and `cutlass_conv_plan.selected_candidate`, and one
  compile-path test proves generated lowering consumes that static selection by
  writing the profiled candidate into `kernel_manifest.json`,
  `kernel_codegen_plan.json`, `compile_config.json`, `manifest.json`, and
  `debug/execution_plan.json` with the expected residual wrapper-stage launch
  ABI `dinoml_cutlass_conv2d_bias_add_v1`. Validation reran the targeted
  profiling/execution-plan regression slice only; no runtime/provider code
  changes were needed. The remaining Conv maturity work should move to the next
  small profile/candidate-selection or runtime-parity gap rather than revisiting
  this residual static-plan path.
- Admitted the next bounded Conv v1-style residual+activation epilogue as
  public `conv2d_bias_add_relu`. The slice keeps the existing static rank-4
  groups=1 NCHW/OIHW public contract, requires one same-shape residual tensor,
  records explicit `bias_add_relu` epilogue metadata
  (`op=conv2d_bias_add_relu`, `epilogue=bias_add_relu`,
  `epilogue_config={"inputs":["bias","d0"],"activation":"relu"}`,
  residual output shape, `residual_pack` wrapper stage, and launch ABI
  `dinoml_cutlass_conv2d_bias_add_relu_v1`) through candidate sets, manifests,
  profile workloads, execution plans, support/source manifests, and generated
  lowering, and reuses the existing artifact-visible residual NCHW->NHWC pack
  stage. Validation covered frontend/base regression checks, CPU reference and
  generated CPU parity, profiling/execution-plan metadata, real float16-path
  support-library `nvcc` compile proof, focused float32 SIMT CUDA runtime
  parity, and focused fp16 TensorOp CUDA runtime parity on the admitted
  FewChannels `C=3`, FixedChannels `C=4`/`C=8`, and optimized aligned `C=16`
  lanes against Torch `conv2d + bias + residual + relu`. This is useful Conv
  v1-core progress, but it is not a claim for bfloat16, broader float32
  TensorOp, grouped/depthwise/transposed/3D Conv, or richer residual chains.
- Added focused fp16 CUDA runtime parity coverage for the just-landed
  residual Conv slice `conv2d_bias_add` without changing provider/runtime
  surface. The new CUDA-gated tests reuse the shared DinoML CUDA support cache,
  compile real artifacts for the admitted fp16 TensorOp FewChannels `C=3`,
  FixedChannels `C=4`/`C=8`, and optimized aligned `C=16` lanes, assert
  explicit `bias_add` metadata
  (`op=conv2d_bias_add`, `epilogue=bias_add`, residual output shape,
  `residual_pack` wrapper stage, selected TensorOp candidate id/symbol, and
  launch ABI `dinoml_cutlass_conv2d_bias_add_v1`), and compare runtime output
  against Torch `conv2d + bias + residual` with fp16 tolerances. Compact
  non-CUDA regression reran the float16 CPU-reference parity row. This closes
  the immediate anti-drift gap for the full admitted fp16 residual TensorOp
  lane family, but it is not a claim for bfloat16, broader float32 TensorOp,
  grouped/depthwise/transposed/3D Conv, or richer residual epilogues.
- PM-reviewed and merged the first bounded residual Conv epilogue slice as
  public `conv2d_bias_add`. The slice keeps the existing static rank-4
  groups=1 NCHW/OIHW public contract, requires one same-shape residual tensor,
  records explicit `bias_add` epilogue metadata through candidate sets,
  manifests, profile workloads, execution plans, support/source manifests, and
  generated lowering, and launches through
  `dinoml_cutlass_conv2d_bias_add_v1` with artifact-visible residual NCHW->NHWC
  packing. Skeptical review found a base `conv2d_bias` shape-resolution
  regression and stale plan wording; PM fixed those before merge and also fixed
  the existing CPU reference call sites after the shared helper grew an explicit
  residual parameter. Validation covered frontend/base regression checks, CPU
  reference and generated CPU parity for the add slice, profiling metadata,
  real support-library compile coverage, and focused float32 SIMT CUDA runtime
  parity. This is useful Conv v1-core progress, but it is not grouped/depthwise/
  transposed/3D Conv, sigmoid, add+activation chains, multiple residual inputs,
  bfloat16, or broad fp16 residual runtime parity.
- PM-reviewed and merged refreshed cached `openai/clip-vit-base-patch32` CUDA
  checkpoint evidence after the explicit QuickGELU fix. The opt-in full compiled
  CUDA smoke now asserts parity instead of a loose drift envelope on the
  admitted batch-1 short-sequence synthetic-input smoke: refreshed max absolute
  errors were about `7.63e-6` on logits, `1.94e-7` on text embeds, and
  `3.02e-7` on image embeds. The tower/full breakdown now asserts standalone
  text/image feature artifacts and recomposed logits are also parity-scale, and
  the compact CUDA op audit is clean through the current provider-heavy
  frontier: `layer_norm`, patch `conv2d_bias`, `gemm_rcr_bias`,
  `gemm_rcr_bias_quick_gelu`, `bmm_rcr`, `softmax`, and `bmm_rrr`. This is not
  a tokenizer/processor, CLIP-L, broader sequence/image, or generalized
  checkpoint-family claim, but it does close the cached base-checkpoint CUDA
  numerical blocker for the admitted smoke inputs.
- PM-reviewed and merged stronger `conv2d_bias_relu` profiling and
  execution-plan evidence. The new tests prove fused `bias_relu` metadata
  survives static execution-plan application, compile-time execution-plan
  consumption, wrapper-stage/codegen symbol selection, and a real CUDA-gated
  `profile_artifact` smoke that writes `debug/profile_report.json` plus
  `debug/execution_plan.json` for a fused ReLU Conv artifact. The real profiling
  smoke is deliberately confidence-aware: it accepts either a consumable static
  selection or a low-confidence non-selection while still asserting the fused
  ABI/symbol metadata remains intact. This directly advances the requested
  GEMM-like profiling/candidate-selection maturity for Conv without adding new
  public op surface.
- PM-reviewed and merged a Conv sigmoid epilogue blocker note after a bounded
  `conv2d_bias_sigmoid` admission attempt failed the required CUDA
  support-library compile gate and was fully backed out. The public op surface
  remains absent. The recorded classification is effectively hard-blocked for
  now: CUTLASS exposes `LinearCombinationSigmoid`, but the current Conv
  `ImplicitGemmConvolution` source-C/bias launcher wiring did not compile under
  real `nvcc` once that epilogue replaced the admitted bias/ReLU epilogues.
  Future sigmoid Conv work must first solve that provider/runtime ABI mismatch
  and prove a real support-library build plus CUDA parity test; do not land
  frontend-only sigmoid surface.
- PM-reviewed and merged focused public no-bias `conv2d` CUDA runtime parity
  coverage without adding a separate no-bias provider ABI. The new tests keep
  `dml.ops.conv2d(...)` as the explicit-zero `conv2d_bias` bridge and prove the
  bridge on real compiled CUDA artifacts for fp16 TensorOp FewChannels `C=3`,
  FixedChannels `C=4`/`C=8`, and optimized aligned `C=16`, with manifest
  assertions preserving `source_op=conv2d`, `bias_mode=explicit_zero_constant`,
  TensorOp candidate selection, and runtime parity against Torch. Worker
  validation ran the then-current legacy Conv CUDA-heavy suite; PM validation
  reran the explicit-zero bridge compile check plus a FixedChannels C4 no-bias
  TensorOp runtime parity test. This keeps Conv first and advances v1-core
  parity evidence for bias/no-bias without broadening groups/layouts.
- PM-reviewed and merged a distinct CLIP-focused
  `gemm_rcr_bias_quick_gelu` fused GEMM slice without changing the existing
  `fast_gelu` surface. The new path carries explicit `quick_gelu` activation
  metadata, generated CPU lowering/reference semantics for
  `x * sigmoid(1.702 * x)`, a dedicated CUTLASS `BiasQuickGeluEpilogue`, CLIP
  MLP wiring, and updated CLIP wrapper/workflow/provider expectations. Focused
  validation covered frontend/provider metadata, CPU reference and generated
  CPU artifact lowering, CUDA source rendering, distinct fast/quick GELU
  surfaces, a generated CUDA CLIP MLP runtime parity check, and the cached
  checkpoint CUDA op audit; the audit now treats the text `fc1` QuickGELU row
  as clean instead of the first known fused-activation drift. This was aligned
  with human steering to keep `fast_gelu` intact and make QuickGELU explicit.
- Added a focused cached `openai/clip-vit-base-patch32` CUDA op audit and a
  CLIP-shape CUDA `layer_norm` regression. The audit uses actual cached
  checkpoint tower activations and stops at the first drifty provider row:
  `text_layer_norm1`, `vision_pre_layer_norm`, `vision_patch_conv2d_bias`,
  `text_q_proj_gemm_rcr_bias`, and `vision_q_proj_gemm_rcr_bias` are clean at
  tight tolerances, while the CLIP MLP `fc1` fused activation row currently
  drifts by about `2.07e-2`. That row is intentionally still named
  `gemm_rcr_bias_fast_gelu` because it is the existing path being misused by
  CLIP; do not change `fast_gelu` semantics. The next fix should add an
  explicit QuickGELU surface/provider path for CLIP
  (`x * sigmoid(1.702 * x)`) and update the model to use it.
- Added an opt-in cached OpenAI CLIP base checkpoint compiled-CUDA artifact
  tractability smoke. The new smoke is gated by
  `DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CUDA_SMOKE=1`, forces
  `HF_HOME=/workspace/.cache/huggingface`, uses
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)`, reuses
  the shared CUDA support cache fixture, traces the same bounded batch-1 short
  sequence adapter-built `LegacyCLIPModel`, compiles a CUDA `.dinoml`, loads it
  through `dinoml.runtime`, and runs the full cached
  `openai/clip-vit-base-patch32` two-tower artifact. This proves checkpoint
  loading, tracing, CUDA compile admission, support-library linking, runtime
  load, and execution are tractable for the cached base checkpoint, but it is
  explicitly not a parity claim: the current CUDA artifact still drifts from
  local Transformers on the bounded smoke inputs by roughly `0.795` on logits,
  `0.0295` on text embeds, and `0.0739` on image embeds. Treat the next CLIP
  CUDA lane as isolating that numerical drift, not reopening checkpoint loading
  or compile admission.
- Closed the last explicit runtime-parity gap in the current bounded fused
  Conv ReLU TensorOp family by proving `conv2d_bias_relu` on the admitted fp16
  CUTLASS FixedChannels `C=4` candidate. The CUDA-gated regression compiles a
  real artifact for semantic `NCHW/OIHW` shape
  `x=[2,4,7,8], weight=[8,4,3,2], bias=[8]`, asserts the fused
  `bias_relu` epilogue and `fixed_channels_c4` TensorOp candidate with no
  hidden channel padding, and checks runtime output parity against Torch. The
  bounded ReLU Conv path now has focused runtime proofs for float32 SIMT, fp16
  FewChannels `C=3`, fp16 FixedChannels `C=4`/`C=8`, and fp16 optimized aligned
  `C=16`; broader epilogues, bfloat16, broader float32 TensorOp, grouped/
  depthwise/transposed/3D Conv, and dynamic dispatch remain out of scope.
- Added an opt-in cached OpenAI CLIP base checkpoint compiled-CPU artifact
  parity smoke for the PM-refreshed `/workspace/.cache/huggingface`
  checkpoint. The new smoke is gated by
  `DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CPU_SMOKE=1`, forces
  `HF_HOME=/workspace/.cache/huggingface`, still loads
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)`, traces
  the adapter-built `LegacyCLIPModel` on the same bounded batch-1 short
  sequence inputs as the runtime smoke, compiles a CPU `.dinoml`, runs it
  through `dinoml.runtime`, and compares `logits_per_image`,
  `logits_per_text`, `text_embeds`, and `image_embeds` against local
  Transformers. This proves the full two-tower cached
  `openai/clip-vit-base-patch32` checkpoint is tractable as a bounded compiled
  CPU artifact on this worktree; it remains explicitly heavy, opt-in, cache-
  only, and not a tokenizer/processor, CUDA, interpolation, loss, or broader
  checkpoint-family claim.
- Added an opt-in cached OpenAI CLIP base checkpoint CPU-reference runtime
  parity smoke after refreshing `openai/clip-vit-base-patch32` into
  `/workspace/.cache/huggingface`. The new smoke is gated by
  `DINOML_RUN_CLIP_CHECKPOINT_RUNTIME_SMOKE=1`, uses
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)`, builds
  DinoML `LegacyCLIPModel` through the adapter, traces a batch-1 short-sequence
  input that respects the checkpoint config, runs DinoML `reference_numpy`, and
  compares `logits_per_image`, `logits_per_text`, `text_embeds`, and
  `image_embeds` against local Transformers. This proves real cached
  checkpoint parity for the CPU reference path only; it is not a tokenizer/
  processor, compiled CPU, CUDA, interpolation, loss, or broader CLIP-L runtime
  claim.
- Closed two more concrete runtime-parity gaps in the bounded fused Conv
  epilogue lane by proving `conv2d_bias_relu` on the admitted fp16 CUTLASS
  TensorOp FewChannels `C=3` and optimized aligned `C=16` candidates. The new
  CUDA-gated regressions compile real artifacts for semantic `NCHW/OIHW`
  shapes `x=[2,3,7,8], weight=[4,3,3,2], bias=[4]` and
  `x=[2,16,7,8], weight=[16,16,3,2], bias=[16]`, assert that manifest
  selection keeps the fused `bias_relu` epilogue on the expected
  `few_channels_c3` and `optimized_align8` TensorOp candidates with no hidden
  channel padding, and check runtime output parity against Torch. The bounded
  fused ReLU path now has real CUDA runtime proofs on float32 SIMT, fp16
  FewChannels `C=3`, fp16 FixedChannels `C=4`/`C=8`, and fp16 optimized aligned
  `C=16`.
- Added an opt-in cached-checkpoint admission smoke for the new Transformers
  CLIP adapter. The smoke is gated by
  `DINOML_RUN_CLIP_CHECKPOINT_ADAPTER_STATE_SMOKE=1`, uses
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)`, defaults
  to `openai/clip-vit-large-patch14` with `DINOML_CLIP_CHECKPOINT_ID` override
  support, and skips clearly when the checkpoint is not already cached. When a
  cached checkpoint is present, it proves config adaptation, required
  state-dict import into the `LegacyCLIPModel` weight namespace, full
  fixed-shape tracing, IR validation, and CUDA manifest/codegen-plan admission.
  This is intentionally not a large-checkpoint runtime-parity claim and adds no
  tokenizer/processor/download surface.
- Added the bounded Transformers checkpoint adapter for the existing
  `LegacyCLIPModel` path. `src/dinoml/models/clip.py` can now derive
  `LegacyCLIPTextConfig`, `LegacyCLIPVisionConfig`, and the DinoML CLIP weight
  namespace from a local Transformers `CLIPModel`/`CLIPConfig` plus
  `state_dict()`, then build the existing bounded wrapper directly. Focused
  tests prove tiny local-Transformers parity through the adapter for both
  `eos_token_id == 2` and non-2 EOS pooling branches, reject non-`quick_gelu`
  configs, and reject missing required checkpoint weights. This moves CLIP
  integration toward known-checkpoint readiness without claiming tokenizer/
  processor plumbing, positional interpolation, loss, FlashAttention, or full
  large-checkpoint runtime parity.
- Closed the concrete runtime-parity gap in the newly admitted fused Conv
  epilogue path by proving `conv2d_bias_relu` on a real fp16 CUTLASS TensorOp
  FixedChannels `C=8` candidate. The new CUDA-gated regression compiles an
  artifact for semantic `NCHW/OIHW` shape `x=[2,8,7,8]`,
  `weight=[8,8,3,2]`, `bias=[8]`, asserts that manifest selection keeps the
  fused `bias_relu` epilogue on the `fixed_channels_c8` candidate with no
  hidden channel padding, and checks runtime output parity against Torch. This
  closes the immediate anti-drift gap called out after the `conv2d_bias_relu`
  merge: ReLU now has at least one real fp16 TensorOp runtime proof in
  addition to the existing float32 SIMT smoke. Follow-on loops have since
  proved the `C=3` few-channels, FixedChannels `C=4`, and optimized `C>=16`
  lanes too.
- Proved the already-emitted fp16 CUTLASS Conv FixedChannels `C=8` runtime
  path end to end for the admitted static rank-4, `groups=1` public
  `conv2d_bias` contract. The new CUDA-gated regression compiles a real artifact
  for semantic `NCHW/OIHW` shape `x=[2,8,7,8]`, `weight=[8,8,3,2]`,
  `bias=[8]`, asserts that manifest selection pins the
  `fixed_channels_c8` candidate with no hidden channel padding, and checks
  runtime output parity against Torch. This closes the earlier gap where `C=8`
  was only artifact-visible in manifests/source manifests; the provider/runtime
  story for the admitted fixed-channel families now has explicit parity proofs
  for `C=4` and `C=8` alongside `C=3`, optimized `C>=16`, and float32 SIMT.
- Landed the first fused Conv epilogue slice as a bounded extension of the
  existing `cutlass_conv` path: public `conv2d_bias_relu`. This loop did not
  broaden Conv semantics beyond the admitted static rank-4, `groups=1`,
  NCHW/OIHW contract; it added one useful v1-style fused epilogue with
  end-to-end visibility. Frontend/admission now registers
  `dml.ops.conv2d_bias_relu(...)`, CPU reference execution supports it, and
  generated CPU artifacts reuse the existing naive `conv2d_bias` loop family
  with an explicit fused ReLU clamp. CUDA/provider visibility is now wired
  through `cutlass_conv`: candidate-set ids, manifest metadata, profile
  workloads, execution-plan compatibility checks, support-cache/source-manifest
  payloads, and generated lowering all record the fused `bias_relu` epilogue
  plus `epilogue_config` and the new launch ABI
  `dinoml_cutlass_conv2d_bias_relu_v1`. Runtime coverage is intentionally the
  same bounded candidate family already admitted for base `conv2d_bias`: fp16
  SIMT, fp16 TensorOp few-channels (`C=3`), fp16 TensorOp fixed-channels
  (`C=4`/`C=8`), fp16 TensorOp optimized (`C >= 16` with channel alignment),
  and float32 SIMT only. Focused tests cover traced IR/frontend shape
  preservation, CPU reference parity, generated CPU artifact parity,
  manifest/codegen/profile visibility, the opt-in float32 SIMT CUDA runtime
  smoke, and fp16 TensorOp CUDA runtime parity on the `C=3` FewChannels,
  `C=8` FixedChannels, and optimized aligned `C=16` lanes. This is the first
  fused Conv epilogue slice only; add/sigmoid/residual epilogues,
  broader float32 TensorOp or bf16 runtime, grouped/depthwise/transposed/3D
  Conv, and guarded/dynamic Conv dispatch remain out of scope.
- Tightened CUTLASS Conv execution-plan discipline to match the established
  GEMM/BMM flow instead of treating Conv as a special happy-path case. Conv now
  keeps the same visible profile-selection path end to end: candidate sets in
  the manifest, profile workloads/report/cache, static execution-plan
  selections, compile-time execution-plan summaries in artifact metadata, and
  generated lowering/codegen visibly consuming the selected Conv kernel symbol.
  The concrete gap closed in this loop was stale/incompatible plan handling:
  strict execution-plan application still rejects an incompatible Conv
  candidate, while relaxed application now skips that bad selection and keeps
  the manifest default instead of partially mutating or hard-failing. Focused
  regressions cover the relaxed-vs-strict Conv behavior, stale compile-time
  plan rejection, and compile artifacts preserving the selected Conv execution
  plan in `compile_config.json`, top-level `manifest.json`, and generated
  `debug/execution_plan.json`.
- Landed the bounded public no-bias `conv2d` bridge without pretending a new
  provider family exists. `dml.ops.conv2d(x, weight, ...)` now performs its own
  static NCHW/OIHW/groups=1 validation, computes the no-bias output shape in
  frontend utilities, and then emits an artifact-visible `conv2d_bias` core
  node with attrs `source_op=conv2d` and
  `bias_mode=explicit_zero_constant` plus an explicit traced zero-bias constant
  tensor. Focused tests cover the traced IR/constant visibility, CPU reference
  and CPU artifact parity against Torch `F.conv2d(..., bias=None)`, CUDA
  compile-time manifest/codegen visibility, an opt-in small CUDA runtime smoke,
  and CLIP patch projection now using the public `conv2d` surface instead of a
  model-local synthetic zero-bias parameter. This is an honest no-bias bridge
  over the existing `conv2d_bias` runtime/provider path, not a distinct no-bias
  CUTLASS family or a claim about fused epilogues, grouped/depthwise/
  transposed/3D Conv, or dynamic dispatch.
- Expanded CUTLASS Conv float32 runtime coverage beyond the original exact CLIP
  patch-projection slice. Static rank-4 public NCHW/OIHW `conv2d_bias` with
  `groups=1` now uses a bounded float32 SIMT runtime/profiler candidate, and a
  non-CLIP stride/padding/dilation CUDA parity case proves the generated
  pack/launch/unpack path against Torch. This is a real v1-core-parity step for
  the existing `conv2d_bias` surface, not a claim for no-bias/fused epilogues,
  grouped/depthwise/transposed/3D Conv, dynamic dispatch, or persistent packed
  weights.
- Fixed the first concrete CUDA CLIP contrastive-head drift boundary in model
  generated reductions. The CUDA `vector_norm` kernel had been reusing its
  per-element `acc + value * value` expression while reducing already-partial
  sums across the warp/shared-memory tree, which squared partial sums a second
  time and blew up CLIP feature normalization in the two-tower CUDA path. The
  reduction lowering now distinguishes element accumulation from partial-sum
  reduction, focused CUDA source/runtime regressions cover `vector_norm`, and
  the opt-in CLIP two-tower CUDA smoke now expects allclose parity for
  normalized embeds and logits instead of asserting intentional drift. Local
  validation in this worktree covered the focused CUDA reduction runtime path
  plus the broader CLIP suite entry changes, while the full-model CUDA CLIP
  smoke remains opt-in and environment-sensitive.
- Added an opt-in CUDA CLIP two-tower blocker smoke that uses the newly landed
  exact patch-projection runtime in the real model path and records the next
  honest failure boundary. The generated CUDA `LegacyCLIPModel`
  `get_text_features` and `get_image_features` artifacts now have focused
  regression coverage proving they both stay near local Transformers on CUDA,
  including the bounded float32 SIMT `cutlass_conv` patch projection inside the
  image tower. The same smoke then compiles and runs the full two-tower CUDA
  artifact and proves the next blocker has moved past Conv: normalized embeds
  and logits drift badly only after the contrastive head (`vector_norm`/`div`/
  final similarity assembly). This keeps the CUDA artifact story honest without
  widening Conv claims.
- Landed the exact CLIP float32 CUDA Conv runtime slice. The
  Transformers-shaped patch projection used by `LegacyCLIPVisionEmbeddings`
  (`[B,3,4,4]` input, `[6,3,2,2]` weights, stride 2, padding 0, groups 1)
  selected a bounded-runtime SIMT `cutlass_conv` candidate, built the support
  library launcher when CUDA tooling was available, and matched Torch/local
  Transformers through a CUDA artifact boundary. The later broader float32
  Conv loop supersedes its exact-shape-only limitation for static groups=1
  `conv2d_bias`.
- Upgraded the visible `examples/clip_model_workflow.py` proof from CPU
  reference-only to a compiled CPU artifact lifecycle smoke. The example now
  self-bootstraps the worktree source, traces the bounded `LegacyCLIPModel`,
  compiles a CPU `.dinoml`, loads it with `dinoml.runtime`, runs
  `session.run_numpy(...)`, and prints artifact-vs-reference and
  artifact-vs-local-Transformers parity with explicit artifact details. Focused
  tests smoke the plain script, keep the workflow hermetic, and include a
  regression so bridge-kernel reporting cannot mistake `gemm_rcr_bias` for
  `gemm_rcr`.
- Landed the bounded naive compiled CPU bridge for `conv2d_bias` as the final
  blocker for the current bounded CLIP CPU artifact path. Generated CPU
  artifacts now run a static-shape NCHW/OIHW `groups=1` loop for admitted
  `float32`/`float16` Conv, keep CUDA/CUTLASS Conv behavior scaffolded and
  honest, and make `LegacyCLIPVisionEmbeddings`,
  `LegacyCLIPVisionModelWithProjection`, and the bounded two-tower
  `LegacyCLIPModel` match local `/workspace/transformers` as CPU artifacts.
  A reviewer-requested follow-up made the focused tests hermetic without
  relying on ambient `PYTHONPATH` and corrected stale CLIP plan language.
- Landed the bounded naive compiled CPU bridge for `gemm_rcr_bias_fast_gelu`
  and a deeper CLIP text CPU artifact proof. The then-existing `fast_gelu`
  epilogue ran in generated CPU GEMM artifacts for dynamic folded-`M` shapes,
  and the two-layer CLIP text wrapper matched local `/workspace/transformers`
  as a CPU artifact without explicit `position_ids`.
  Full two-tower CPU compilation now moves honestly to the vision-side
  `conv2d_bias` blocker. The loop also fixed a real generated-identifier drift:
  CPU/CUDA top-level lowering and op-local lowerings now share one
  `shape_buffers.c_ident` helper, with a regression proving tensor names like
  `x_0`, `y_0`, and `out_0` compile and run consistently.
- Landed a bounded naive compiled CPU bridge for `bmm_rrr`, completing the two
  CLIP attention matmul layouts needed by the current text artifact path.
  Generated CPU artifacts now distinguish column-major-logical `B[B,N,K]` for
  `bmm_rcr` from row-major-logical `B[B,K,N]` for `bmm_rrr`, retain zero-stride
  batch broadcast, and keep CUDA/CUTLASS BMM behavior unchanged. Focused tests
  cover dynamic token shapes and batch broadcast across the admitted BMM CPU
  layouts/dtypes, compiled CPU CLIP attention with a padding mask against the
  reference path, and the CLIP text/two-tower CPU compile boundary. The next
  text-side CPU artifact blocker is now precisely `gemm_rcr_bias_fast_gelu`.
- Landed a bounded naive compiled CPU bridge for `bmm_rcr` as the next CLIP
  text artifact unblocker, still explicitly a temporary generated loop rather
  than a CPU library/provider path. Generated CPU artifacts keep rank-3
  row-major `A[B,M,K]`, column-major-logical `B[B,N,K]`, row-major `C[B,M,N]`,
  support zero-stride batch broadcast, and preserve CUDA/CUTLASS BMM behavior.
  Focused tests cover dynamic token shapes for `float32`, `float16`, and
  `bfloat16`, batch broadcast, existing naive GEMM coverage, CLIP text/two-tower
  CPU compile boundaries, and BMM lowering/profile metadata. The deeper CLIP
  text and two-tower CPU blockers now move forward to `bmm_rrr`.
- Landed a bounded naive compiled CPU bridge for `gemm_rcr` and
  `gemm_rcr_bias` as an explicit CLIP artifact unblocker, not an optimized CPU
  provider/library path. Generated CPU artifacts now flatten `A[..., K]` into
  runtime `M`, run a row-major accumulation loop with optional rank-1 or
  `[1, N]` bias, and keep CUDA/CUTLASS behavior unchanged. Focused tests prove
  dynamic folded-`M` CPU runtime parity, keep `gemm_rrr` rejected on CPU, run
  the zero-layer CLIP text wrapper as a CPU artifact against local
  `/workspace/transformers`, and move deeper CLIP text/two-tower CPU compile
  blockers forward to `bmm_rcr`. A zero-text/zero-vision two-tower compile test
  keeps the vision-side CPU blocker honest at `conv2d_bias`.
- Pinned the exact CLIP patch-projection CUDA runtime boundary. The
  Transformers-shaped float32 patch Conv used by `LegacyCLIPVisionEmbeddings`
  (`[B,3,4,4]` input, `[6,3,2,2]` weights, stride 2, padding 0, groups 1) now
  has focused coverage proving that the manifest selects a float32 SIMT
  `cutlass_conv` candidate with `manifest_scaffold_only` status and
  `cutlass_conv_runtime_launcher_not_implemented` as the explicit blocker. When
  CUDA tooling is available, the artifact compiles and then fails at the
  generated scaffold runtime boundary instead of disappearing into a vague
  provider gap.
- Pinned the current CLIPModel artifact blockers without widening provider
  surface. Focused tests now prove that full two-tower CPU compilation fails
  first at the existing `gemm_rcr_bias` compiled-CPU boundary, before the
  vision Conv path, and that CUDA manifest/codegen planning keeps the single
  CLIP Conv node artifact-visible as `cutlass_conv` with scaffold-only status
  plus explicit activation-pack, weight-pack, provider-launch, and output-unpack
  wrapper stages. This gives future runtime work a precise test-backed boundary
  to flip into an artifact/runtime smoke.
- Admitted zero-layer CLIP text parity after verifying local
  `/workspace/transformers` supports `num_hidden_layers=0` for
  `CLIPTextModelWithProjection` and `CLIPModel`. `LegacyCLIPTextConfig` now
  accepts non-negative text layer counts, zero-layer text wrapper tests cover
  both supported EOS pooling branches with explicit/default traced
  `position_ids`, and the two-tower suite includes a zero-text/zero-vision
  `LegacyCLIPModel` parity case. No tokenizer/processor plumbing, positional
  interpolation, FlashAttention, or provider claims were added.
- Proved multi-layer CLIP text parity without changing source. Focused tests
  now exercise a deterministic two-layer `LegacyCLIPTextModelWithProjection`
  against local `/workspace/transformers` for both supported EOS pooling
  branches and for both explicit and default traced `position_ids`. The tiny
  two-tower `LegacyCLIPModel` parity test now uses two text layers plus the
  already-admitted two-layer vision path, comparing helper features,
  normalized embeds, and logits against local Transformers while keeping
  tokenizer/processor plumbing, positional interpolation, FlashAttention, and
  new provider claims out of scope.
- Admitted stacked CLIP vision encoder blocks for the bounded vision wrapper.
  `LegacyCLIPVisionConfig` now accepts non-negative `num_hidden_layers`, and
  the wrapper reuses the existing dense noncausal attention + quick-gelu MLP
  block for each layer without adding new ops, FlashAttention, positional
  interpolation, tokenizer/processor plumbing, or provider claims. Focused
  tests pin zero-, one-, and two-layer `CLIPVisionModelWithProjection` outputs
  against local `/workspace/transformers`, upgrade the tiny two-tower
  `LegacyCLIPModel` parity test to two vision layers, and keep Conv represented
  honestly as a CUTLASS scaffold-only manifest entry.
- Added a compact runnable CLIPModel two-tower workflow proof. The new
  `examples/clip_model_workflow.py` traces the bounded `LegacyCLIPModel` on
  synthetic text/image tensors, runs the CPU reference path, prints projected
  text/image features, normalized embeds, logits, node/kernel ownership, and
  test-visible limits: no explicit `position_ids`, fixed square NCHW image
  shape, and CUTLASS Conv still represented as a scaffold-only manifest entry.
  Focused tests compare the example outputs against local `/workspace/transformers`
  `CLIPModel`, smoke the runnable script, and keep the existing two-tower parity
  tests green without adding tokenizer/processor plumbing, positional
  interpolation, FlashAttention, or new provider claims.
- Landed the first bounded CLIPModel-style two-tower contrastive workflow.
  `LegacyCLIPModel` now composes the admitted text tower and one-layer vision
  tower, exposes bounded `get_text_features` / `get_image_features`, normalizes
  projected features, applies `exp(logit_scale)`, and produces
  `logits_per_text` plus transposed `logits_per_image`. Focused tests pin
  projected features, normalized embeds, and both logits against local
  `/workspace/transformers` `CLIPModel` for a deterministic tiny config, while
  preserving provider/model manifest ownership. Remaining limits: static traced
  text length, default traced text positions, fixed square NCHW vision input,
  vision depth admitted only up to one layer, no tokenizer/processor plumbing,
  no positional interpolation, no loss path, and no compiled full-model CUDA
  runtime parity yet.

## Next Recommended Lane

- Human steering on 2026-05-19 makes ROCm backend integration the active lane.
  The simple generated-template ROCm surface is now admitted with real
  `.venv/rocm` compile/load/run contracts. The next bounded ROCm task should
  either harden that surface with targeted edge cases or start a provider-backed
  lane such as CK only with artifact-visible manifest/support/generated
  lowering/runtime/parity proof. Do not treat the checked-in CK source or the
  simple-template contract as GEMM/BMM/Conv provider support.
- Human steering on 2026-05-15 makes Conv the first lane. The next bounded Conv
  work should stay inside the existing `cutlass_conv` contract and build on the
  admitted `conv2d_bias_relu`, `conv2d_bias_add`, and `conv2d_bias_add_relu`
  slices rather than drifting into alias polish or broad plumbing.
  Highest-value follow-ons are: broader runtime coverage for the current
  epilogues, especially `conv2d_bias_add_relu` fp16 TensorOp lanes or bfloat16
  and float32 TensorOp gaps beyond the admitted SIMT float32 path; a different
  fused epilogue only if it can complete the same frontend/provider/runtime/
  docs slice, but not `conv2d_bias_sigmoid` until the recorded CUTLASS Conv
  epilogue ABI blocker is solved with a real `nvcc` build; or deeper
  execution-plan evidence around Conv candidate selection on CUDA-capable
  hardware. The fused ReLU Conv path now has real profile/report/execution-plan
  metadata coverage, the residual add slice now has focused float32 SIMT
  runtime parity plus the full admitted fp16 TensorOp lane family, and the
  new residual+ReLU slice now has focused float32 SIMT runtime parity plus the
  matching admitted fp16 TensorOp lane family, but guarded/dynamic dispatch
  remains unsupported.
  Keep the exact coverage honest: today only `conv2d_bias`, explicit-zero
  `conv2d`, fused `conv2d_bias_relu`, fused `conv2d_bias_add`, and fused
  `conv2d_bias_add_relu` are admitted on the static rank-4 groups=1 path.
  The public no-bias `conv2d` bridge now has real fp16 TensorOp runtime parity
  across the same core candidate families as `conv2d_bias`; do not split it
  into a separate provider ABI without a fresh full admission slice.
- Keep converting the bounded CLIPModel surface toward usable artifacts and
  local Transformers parity with one concrete, test-backed gap at a time after
  the Conv lane is stable. The adapter now reaches cached-checkpoint
  config/state import, trace/manifest admission, a real cached
  `openai/clip-vit-base-patch32` CPU-reference runtime parity smoke, and a
  matching opt-in compiled CPU artifact smoke for the full two-tower base
  checkpoint. The cached base checkpoint now also compiles, loads, and runs as
  an opt-in CUDA artifact with parity-scale error on the admitted bounded smoke
  inputs after the explicit QuickGELU fix. The next high-value CLIP CUDA task is
  no longer this base-checkpoint drift investigation; broaden only by one honest
  admission surface or audited helper row at a time, such as cached padding/mask
  false-path helpers, pooling/layout helper rows, tokenizer/processor-adjacent
  workflow proof, or a separate known-checkpoint/config slice.
  Keep local `/workspace/transformers` parity as the acceptance bar and keep all
  non-parity limits explicit.
- If moving into runtime/provider work, tie it directly to a CLIP artifact test
  and keep the existing Conv limitations honest. Do not broaden tokenizer,
  processor, positional interpolation, FlashAttention, or Conv provider claims
  without a full admission slice.
- Conv next steps should prioritize GEMM-like provider maturity for the existing
  `conv2d_bias`/explicit-zero `conv2d`/`conv2d_bias_relu` path: candidate sets,
  profile workloads, profile reports, execution-plan selections, and generated
  lowering visibly consuming the selected Conv candidate. The stale/
  incompatible static-plan rejection path is now covered, and the first fused
  epilogue is admitted; next work in this lane should stay bounded to deeper
  profile-assisted Conv selection maturity or one more narrowly admitted
  epilogue/runtime gap rather than alias polish. Keep grouped/depthwise/
  transposed/3D and dynamic/guarded dispatch deferred until a separate
  admission slice.
- Human steering on 2026-05-15 allows a naive compiled CPU GEMM implementation
  as a temporary bridge. Do not treat the lack of a final CPU library/BLAS path
  as a blocker for CLIP artifact smoke work, but keep any naive CPU bridge small,
  measured by tests, and explicit about performance limits. The current CLIP CPU
  artifact blocker chain is complete for the bounded two-tower surface; do not
  keep polishing the same naive CPU bridge lane without a concrete failing test
  or unsupported-fence gap.
- Older landed-slice notes below are historical context. Their remaining-limit
  clauses predate later CLIP CPU artifact closure unless explicitly restated in
  the current recommendation above.
- Landed the first real CLIP vision encoder layer. The bounded vision wrapper
  now admits `num_hidden_layers` in `{0, 1}` and matches local Transformers for
  the one-layer surface: fixed-size embeddings, `pre_layrnorm`, dense noncausal
  self-attention, residuals, quick-gelu MLP, CLS pool, `post_layernorm`, and
  bias-free visual projection. Focused tests pin `last_hidden_state`,
  `pooler_output`, and `image_features` for both zero-layer and one-layer
  configs and verify provider/model ownership for Conv, GEMM/BMM, softmax,
  LayerNorm, and sequence assembly. Remaining limits at that point were fixed
  square NCHW only, no positional interpolation, no arbitrary image sizes, and
  no vision padding/causal mask path; the historical CPU artifact boundary is
  superseded by the current completed loop above.
- Landed a bounded zero-layer CLIP vision wrapper/projection slice. DinoML now
  matches local Transformers `CLIPVisionModelWithProjection` for the admitted
  zero-encoder-layer surface: fixed-size vision embeddings, `pre_layrnorm`, CLS
  pool, `post_layernorm`, and bias-free visual projection. Focused tests pin
  `last_hidden_state`, `pooler_output`, and projected `image_features` against
  local Transformers and keep CUDA provider/model ownership visible without
  broadening Conv runtime claims. Remaining limits: `num_hidden_layers == 0`
  only, fixed square NCHW inputs, no positional interpolation, no real vision
  encoder block, no full `CLIPModel`/processor plumbing, and CPU artifact
  compilation still stops at the existing `conv2d_bias` backend boundary.
- Landed the first bounded CLIP vision-side parity slice: fixed-size
  `LegacyCLIPVisionEmbeddings` now matches local Transformers
  `CLIPVisionEmbeddings` for semantic NCHW pixel input, bias-free patch
  projection modeled as `conv2d_bias(..., zero_bias)`, spatial flatten +
  sequence transpose, CLS prepend, and learned absolute position add. Focused
  tests compare full embeddings and the zero-bias patch-projection substep
  against local Transformers, historically recorded the then-existing
  `conv2d_bias` CPU backend boundary, and verify CUDA
  manifest/generated-source ownership without broadening Conv provider maturity.
  Remaining limits: fixed
  square image size only, no positional interpolation, no vision encoder or
  projection head, no full `CLIPVisionModel`, and no widened Conv claims; the
  later CPU Conv bridge supersedes the no-compiled-artifact limit for the
  bounded admitted path.
- Landed the bounded CLIP text-wrapper default-position slice. Callers may now
  omit `position_ids`; the wrapper falls back to a traced static int64
  `[0, 1, ..., S-1]` position sequence for the current static sequence length,
  matching Transformers CLIP default behavior while keeping the explicit
  `position_ids` path working. The visible CLIP text workflow example now omits
  `position_ids`, and focused tests prove both EOS pooling branches with and
  without explicit positions. Remaining limits: text-only wrapper, static
  traced sequence length, no tokenizer/processor plumbing, no vision tower, and
  default positions are traced constants rather than runtime-generated dynamic
  indices.
- Added a visible CLIP text workflow proof without adding new CLIP behavior,
  ops, providers, tokenizer/processor plumbing, FlashAttention, or expensive
  CUDA runtime requirements. `examples/clip_text_workflow.py` traces the
  current `LegacyCLIPTextModelWithProjection`, runs the CPU reference path, and
  prints JSON summarizing the node counts, EOS pooling branch, generated CUDA
  kernel coverage, and provider/model split. Focused tests prove both
  `eos_token_id == 2` and non-2 EOS branches, assert CUTLASS ownership for
  GEMM/BMM pieces, assert model-generated ownership for embedding/LayerNorm/
  softmax/pooling-side kernels, and smoke the runnable example script. Remaining
  limits: text-only proof, compiled CPU wrapper support was not part of that
  proof, explicit
  `position_ids`, static traced sequence length, no vision tower, and no full
  contrastive artifact workflow yet.
- Landed the bounded CLIP non-2 EOS pooling branch without adding a new pooling
  op, vision tower, tokenizer/processor plumbing, or FlashAttention/provider
  surface. `LegacyCLIPTextModelWithProjection` now matches both Transformers
  CLIP text pooling paths: legacy OpenAI configs with `eos_token_id == 2` keep
  the highest-token-id `argmax(input_ids)` compatibility path, while newer
  non-2 EOS configs use first-match `(input_ids == eos_token_id).argmax(...)`
  followed by the existing `batch_gather(...)->squeeze(...)` composition.
  Public `eq` admission was widened only for `int32`/`int64` inputs so token-id
  equality does not float-cast large integers; other relational ops remain
  float/reduced-precision-input only. Focused tests pin local Transformers
  parity for both EOS branches, generated CPU/CUDA source keeps integer storage
  for `eq`, and manifest checks keep provider/model ownership honest. Remaining
  limits: text-only wrapper, explicit `position_ids`, static traced sequence
  length, tokenizer-prepared EOS presence assumption, no vision tower, and no
  full contrastive wrapper artifact workflow yet.
- Replaced the GGUF CUDA runtime-dequant native boundary with a required
  direct-link path: DinoML CMake now owns explicit libgguf CUDA object/archive
  targets from vendored libgguf sources, CUDA module builds link the resulting
  `libgguf_cuda_native` static archive into GGUF runtime-dequant artifacts, and
  generated lowering calls `libgguf_cuda_dequantize_rows_on_stream(...)`
  directly. The runtime function-pointer setter fallback and Python `ctypes`
  native-symbol resolver are removed; Python libgguf/Torch CUDA dequant remains
  only for load-time dense materialization before a run. Focused planning/unit
  coverage now pins the direct archive path without widening GGUF policy,
  epilogue coverage, or public provider admission.
- Use worktrees for independent branches if running parallel agents. Keep
  feature write sets disjoint, keep shared queue/tracking doc reconciliation on
  the main line when possible, and require PM review plus validation before
  merge and push.
- Keep `cutlass_conv` bounded while tightening the now-static profiled path:
  decide whether dynamic Conv buckets/guarded dispatch need admission, and keep
  rejecting grouped/depthwise/transposed/3D, hidden padding, persistent packed
  weights, and public NHWC semantics until a separate design pass admits them.
- Landed a bounded CLIP text encoder-layer composition slice without adding
  `CLIPTextModel`, a new op, or a flash provider path: focused regressions now
  prove one tiny float32 text encoder layer as
  `layer_norm -> dense causal self-attention -> residual -> layer_norm ->
  gemm_rcr_bias_fast_gelu -> gemm_rcr_bias -> residual`, with CPU NumPy parity
  for both static additive causal masking and an optional bool padding mask.
  A light CUDA manifest check keeps provider ownership honest by showing
  `layer_norm`/`softmax` stay model-generated while GEMM/BMM pieces stay
  CUTLASS-backed. This is still a composition proof, not `CLIPTextModel`, not
  a dynamic causal-mask builder, and not a fused attention admission.
- Landed a bounded CLIP contrastive-head composition slice without adding a
  public head op: focused regressions now prove L2-normalized text/image
  features via `vector_norm(..., keepdim=True)` plus division, then
  `gemm_rcr(text_features, image_features.T)`, `exp(logit_scale)`, multiply,
  and transpose/permute orientation for `logits_per_image`. CPU NumPy parity
  covers unequal text/image batch sizes, and manifest checks keep `gemm_rcr`
  CUTLASS-backed while normalization and scalar math stay model-generated.
  This is still a composition proof, not `CLIPModel`, not encoder/projection
  coverage, and not a new public `contrastive_head` op.
- Landed the bounded CLIP text MLP / quick_gelu composition slice without
  adding a new public helper or model: focused regressions now prove
  `gemm_rcr_bias_fast_gelu` as the first projection and `gemm_rcr_bias` as the
  second projection for a tiny CLIP-style text MLP, with CPU NumPy parity and
  manifest/lowering checks confirming the first projection uses the CUTLASS
  `bias_fast_gelu` family while the second stays on `gemm_rcr_bias`. This is
  still a composition proof, not `CLIPTextModel`, not an encoder block, and not
  a new `quick_gelu` public op.
- Landed a bounded CLIP text dense-attention composition slice without adding a
  public attention op or FlashAttention/provider surface: focused regressions
  now prove a tiny static CLIP-style text self-attention path built from
  existing `gemm_rcr_bias`, static shape views, `permute0213`, rank-3
  `bmm_rcr`/`bmm_rrr`, scale multiply, static additive causal-mask constant,
  optional bool padding mask via `reshape`/`expand`/`where`, last-dim
  `softmax`, and output projection. CPU reference parity covers unpadded and
  padded cases, and CUDA manifest coverage keeps provider ownership honest by
  showing CUTLASS GEMM/BMM kernels remain provider-backed while softmax remains
  model-generated. This is still a static composition proof, not
  `CLIPTextModel`, not a dynamic mask builder, and not a fused attention
  admission.
- Landed a bounded CLIP text-embedding composition slice without adding any
  new public op: focused regressions now prove
  `token_embedding(input_ids) + position_embedding(position_ids)` for both
  rank-1 broadcast `position_ids [S]` and explicit batched `position_ids
  [B, S]`, with CPU NumPy parity plus generated CPU and CUDA kernel/runtime
  coverage on the existing embedding and fused-add contracts. The slice keeps
  the honest limits explicit: it does not add `CLIPTextModel`, it does not
  widen `arange`, and it relies on the current embedding/add composition rather
  than a CLIP-specific embedding op.
- Closed the bounded CLIP text-tower pooling slice without adding a new public
  pooling op: focused regressions now prove the legacy OpenAI CLIP
  highest-token-id path as
  `input_ids.argmax(dim=-1, keepdim=True) -> batch_gather(hidden_states, indices)
  -> squeeze(axis=1)` through CPU reference, generated CPU artifact runtime,
  and CUDA source/runtime smoke. The slice keeps the honest limits explicit:
  it only proves the legacy highest-token-id pooling composition and still does
  not solve non-2 EOS equality matching or the broader text-tower
  attention/masking path.
- Landed the next narrow CLIP text-tower blocker without broadening general
  integer tensor support: public/generated `dml.ops.argmax` now admits
  `int32`/`int64` input tensors alongside the existing
  `float32`/`float16`/`bfloat16`/`bool` surface, while preserving the same
  static-shape, last-dim-only, `keepdim`, and `int64` output contract.
  Generated CPU/CUDA lowering now compares `int32`/`int64` values as integers
  instead of casting through float, while float inputs keep the existing
  fp32-plus-NaN behavior. Focused regressions pin frontend/IR admission, CPU
  reference behavior, generated CPU/CUDA source, CPU artifact runtime, the
  compiler/runtime dtype exception boundary, and a legacy OpenAI CLIP-style
  `input_ids.argmax(dim=-1)` EOT pooling case with first-index tie semantics.
  Keep docs honest: this unblocks only the legacy highest-token-id CLIP EOT
  pooling step and does not solve non-2 EOS equality matching or the full text
  pooling gather flow on its own.
- Landed the next bounded CLIP/BERT text-enabling primitive as a real generated
  lookup op instead of another helper composition: public
  `dml.ops.embedding(table, indices)` is now a registered op with dedicated
  validation, CPU reference execution, and generated CPU/CUDA lowering for a
  positive static table `[vocab, hidden]`, `float32`/`float16`/`bfloat16`
  table storage, `int64`/`int32` indices, output dtype matching the table, and
  output shape `indices.shape + [hidden]`. The landed slice preserves dynamic
  leading index dims while keeping the table static, emits explicit CPU/CUDA
  runtime output-size checks plus out-of-bounds index rejection, and adds
  focused regressions for frontend/IR shape-spec propagation, int32/int64
  index support, validation failures, generated-source/kernel-manifest
  ownership, dynamic-batch CPU artifact runtime, CUDA compile/runtime parity,
  and CPU runtime OOB rejection.
- Landed the first real affine LayerNorm primitive needed for the CLIP/ViT/BERT
  first-model sprint without widening beyond the bounded static-hidden slice:
  public `dml.ops.layer_norm(x, weight, bias, eps=...)` is now a registered op
  with dedicated validation plus generated CPU/CUDA lowering rather than a
  helper composition, preserving dynamic leading dims while requiring a
  positive static last dimension and matching rank-1 affine tensors
  `[hidden]`. Focused regressions now cover traced/lowered IR ownership, shape
  and dtype validation, kernel-manifest/generated-source provenance, CPU
  artifact runtime across dynamic leading dims, CUDA compile/runtime parity,
  and reduced-precision (`float16`/`bfloat16`) fp32-accumulation behavior.
- Closed the remaining bounded CUDA runtime-validation gap around
  `get_1d_rotary_pos_embed` tensor-position dynamics without widening the op
  surface: focused regressions now compile one dynamic `float32` CUDA artifact
  with rank-1 tensor positions and prove the generated cos/sin component kernels
  run correctly across multiple runtime sequence lengths through the NumPy
  staging path, while preserving the existing table-generation-only contract,
  mixed-variant provenance coverage, and no-input integer-position runtime
  coverage.
- Hardened the bounded `get_timestep_embedding` runtime contract without
  widening the op surface: added a focused CUDA dynamic-shape artifact
  regression that compiles one `float32` artifact with dynamic timestep length
  `N` and proves the generated kernel runs correctly across multiple runtime
  lengths while preserving the existing single-op lowering and in-kernel
  sinusoidal math contract. This closes the remaining gap between the documented
  dynamic-`N` claim and runtime validation, which previously existed only on
  the CPU artifact path.
- Hardened the already-registered named permute specialization surface without
  widening its contract: focused regressions now prove CPU artifact runtime
  execution for `permute021`, `permute0213`, `permute102`, and `permute210`
  across the admitted `float32`, `float16`, `bfloat16`, and `bool` storage
  surface, add reduced-precision/bool CUDA generated-source checks for named
  kernels, and exercise CUDA runtime parity for the named float32
  specializations when CUDA is available. The slice stays honest about using
  the existing generated dense permute-copy strategy rather than claiming v1
  tiled/coalesced kernel parity.
- Hardened the bounded helper-only `rms_norm` contract without widening the
  op surface: helper-level regressions now prove that both weighted and
  unweighted `dml.ops.rms_norm(...)` inherit the admitted `t5_layer_norm`
  runtime behavior instead of only claiming it in docs. Added focused coverage
  for dynamic-leading-dimension CPU artifact execution and reduced-precision
  (`float16`/`bfloat16`) CUDA runtime parity with fp32 accumulation, while
  preserving the helper-only lowering contract that still emits only
  `t5_layer_norm` nodes/kernels plus the synthetic ones constant for the
  weightless path.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` wrapper-source lane
  without weakening the current compile rejection: rejected CUDA artifacts now
  emit `debug/generated_src/scaffold_source_manifest.json` plus a guarded
  scaffold-only `.cu` wrapper snippet for each Conv wrapper-stage group, and
  `kernel_codegen_plan.json` links those emitted sources back to the recorded
  activation-pack, weight-pack, provider-launch, and output-unpack stage
  sequence. Focused tests now pin the stage-to-source linkage, emitted file
  path, guarded `#if 0` snippet shape, and artifact-side manifest wiring while
  CUDA compile still rejects before module build with the existing
  `manifest/codegen scaffold only` boundary.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` wrapper-metadata lane
  without weakening the current compile rejection: `kernel_codegen_plan.json`
  now records explicit per-node wrapper stages for activation NCHW -> NHWC
  pack, OIHW -> OHWI weight pack, planned provider launch, and NHWC -> NCHW
  output unpack, all derived from the validated `cutlass_conv_plan`
  temporary/layout contract and linked to the selected helper or launcher
  symbols plus static shape/attr call arguments. Added a small source-render
  helper that turns those stage entries into future CUDA wrapper call snippets,
  with focused tests proving the stage order, temporary-buffer usage, helper
  symbol wiring, and rendered call shapes while CUDA compile still stops at the
  existing `manifest/codegen scaffold only` boundary before module build.
- Closed the remaining native-boundary regression gap around the bounded GGUF
  runtime-dequant CUDA slice without widening policy: direct native
  `dino_module_load()` has CUDA-gated coverage for a mixed dense-bias plus
  encoded GGUF RHS `gemm_rrr_bias` artifact, proving that native module load
  autoloads only the dense bias, native encoded-weight installation makes the
  directly linked lowered runtime-dequant path runnable, and native
  `dino_module_unload_constants()` / `dino_module_load_constants()` require
  only the encoded weight to be reinstalled.
- Closed the skeptical-reviewer follow-ups on the bounded GGUF
  runtime-dequant scratch-resource slice without widening policy: support
  library cache keys no longer churn from generated-module/runtime
  `session_resources`, while the full manifest `cache_key` still tracks that
  runtime allocation metadata; CUDA lowering now has an explicit legacy-manifest
  regression proving it falls back to scanning lowered
  `gguf_runtime_dequant` plans when top-level `session_resources` is absent;
  and the shared-scratch claim now has CUDA-gated runtime coverage with two
  GGUF RHS GEMM nodes sharing one max-sized session scratch allocation and
  matching dense dequantized references.
- Made the bounded GGUF runtime-dequant -> CUTLASS GEMM scratch policy more
  artifact-visible without widening the runtime surface: `kernel_manifest.json`
  now records a `session_resources` entry for the shared per-session
  `gguf_runtime_dequant_scratch` CUDA allocation, sized to the maximum dense RHS
  requirement across all lowered `gguf_runtime_dequant` GEMM plans and linked
  back to the source node/constant scratch plans. CUDA lowering now consumes
  that manifest resource when allocating the session-owned scratch buffer while
  retaining the existing lowered-plan fallback for older manifests. Focused
  planning/codegen coverage pins the max-sized shared allocation and its source
  plan provenance; encoded-load plan regressions remain green. No new GGUF
  materialization policy, offload scheduler, op surface, or non-bias GEMM
  epilogue support was added.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` support-library lane by
  compiling the next honest prerequisite for a real launcher without widening
  the runtime claim: the CUTLASS Conv scaffold now emits exported CUDA layout
  transform helpers for the manifest-recorded NCHW -> NHWC activation pack,
  OIHW -> OHWI weight pack, and NHWC -> NCHW output unpack contract, with
  helper symbols threaded back into `cutlass_conv_plan`
  `layout_translation`/`weight_transform` metadata for both `float16` and
  `float32`. The support manifest, source manifest, and codegen support-library
  metadata now all expose those helper exports/symbols explicitly, and focused
  tests prove the scaffold compiles, exports the helper ABI coherently, still
  preserves the launcher/profiler stub contract, and optionally matches Torch
  layout permutations on real CUDA for both supported dtypes. CUDA model
  compile still rejects before final manifest/module build, so no generated
  wrapper lowering, CUTLASS implicit-GEMM conv launch, profiler execution, or
  `conv2d_bias` model runtime is claimed yet.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` runtime-maturity lane
  without enabling a model runtime claim: the support-cache scaffold now renders
  concrete launcher/profiler stub exports for the planned
  `dinoml_cutlass_conv2d_bias_v1` ABI and, when `nvcc` is available, compiles
  them into `lib/libdinoml_cutlass_conv.so` with `compiled_stub_only` status,
  library hash, build command, source-manifest symbols, and explicit export
  metadata. CUDA model compile still rejects before final manifest/module build
  while the kernel manifest remains `manifest_scaffold_only`, so no generated
  pack/unpack lowering, CUTLASS implicit-GEMM launcher, profiler execution, or
  CUDA runtime parity is claimed. Focused conv tests now prove the compiled
  support stub exists, exports the expected symbols, returns the documented
  unsupported status, preserves NHWC/OHWI transform provenance, and still
  rejects module compile honestly.
- Started the `src/dinoml/ops/__init__.py` decomposition without widening the
  op surface: public `dml.ops.where(...)` now lives in
  `src/dinoml/ops/where.py`, while the small broadcast shape-spec inference
  helper it depends on moved into private `src/dinoml/ops/_frontend_utils.py`
  so the registry/export wiring in `dml.ops` can keep overriding the generic
  registered frontend with the bespoke `where` contract. Focused `where`
  frontend, CPU reference, generated CPU/CUDA source, and runtime smoke tests
  continue to cover the public API, and no broader op extraction or checklist
  churn was introduced in this structural slice.
- Repaired the named permute specialization surface so it is now honest instead
  of alias-shaped. Public `dml.ops.permute021`, `permute0213`, `permute102`,
  and `permute210` are now real registered bounded ops with fixed-rank,
  fixed-dims contracts; traced and lowered IR preserve those node names instead
  of silently rewriting them to generic `permute`; and generated CPU/CUDA
  lowering, `kernel_manifest.json`, and generated-source provenance now carry
  op-specific symbols/function names for the specialized nodes. This slice
  intentionally reuses the existing generated dense permute-copy strategy with
  compile-time dims/strides, so it is truthful about being a bounded generated
  specialization rather than v1 tiled kernel parity. Focused regressions now
  cover specialized frontend/IR emission, registry default-attr/schema
  coherence for the fixed `dims` contract, fixed-dims validation against attr
  drift, CPU reference parity, artifact-level CPU manifest/source-manifest
  provenance, and optional CUDA compile coverage for a representative named
  specialization.
- Closed the reviewer follow-ups on the just-landed generated
  `get_1d_rotary_pos_embed` slice without widening the op surface. Model-owned
  generated-kernel provenance is now artifact-visible enough to distinguish
  mixed rotary variants in one graph: `build_kernel_manifest()` no longer
  collapses distinct component variants solely by the shared
  `generated_get_1d_rotary_pos_embed` symbol, and model kernels now carry
  generated function/source provenance through `kernel_manifest.json` and
  `kernel_codegen_plan.json`. The rotary component registry contract is also
  now truthful about input arity by admitting exactly zero-or-one inputs rather
  than pretending to accept arbitrary variadic counts. Focused regressions now
  cover mixed int-pos plus tensor-pos variants in one artifact, artifact-level
  no-input integer-pos CPU runtime execution, optional no-input CUDA runtime
  execution through `run_numpy`, and the zero-or-one `accepts_input_count`
  contract on the internal component ops.
- Finished the half-landed `get_timestep_embedding` slice as a real registered
  generated op instead of a helper composition. Public
  `dml.ops.get_timestep_embedding(...)` now lives in `OP_REGISTRY`, traced IR
  and lowered IR carry a single `get_timestep_embedding` node, dynamic rank-1
  `N` is preserved through output shape-spec propagation, and generated CPU and
  CUDA kernels compute the full sinusoidal table in one op/kernel with fp32
  internal math, output-dtype preservation for `float32`/`float16`/`bfloat16`,
  odd-width zero padding, and `flip_sin_to_cos`. Focused tests now cover the
  registered frontend/IR contract, generated-source/kernel-manifest ownership,
  CPU formula parity, dynamic-`N` CPU artifact execution, and CUDA compile plus
  runtime parity.
- Hardened the bounded ConvNd/CUTLASS scaffold contract around its
  artifact-visible layout/weight transform metadata. The shared
  `cutlass_conv_plan` now validates its own NCHW/OIHW -> NHWC/OHWI semantics,
  dtype/shape-derived temporary sizes, padded-channel bookkeeping, and
  temporary-buffer inventory before profiling, codegen-plan generation, or the
  support-cache/source-manifest scaffold consume it. Candidate metadata must
  also agree with the recorded semantic/provider layouts, so manifest drift now
  fails explicitly instead of propagating incoherent provenance into workload
  JSON or support manifests. The Conv support scaffold now also revalidates and
  normalizes each caller-supplied used-plan entry before persisting
  `cutlass_conv_manifest.json` or `source_manifest.json`: it re-derives the
  selected scaffold candidate from the entry candidate list, validates
  candidate-set provenance, carries `node_id` when present, and rejects direct
  caller mutations to selected-candidate layout/dtype metadata before any
  support-manifest payload is written or trusted. Added focused regressions
  that prove profiling rejects malformed transform byte counts, codegen/support
  provenance rejects candidate-layout drift against the recorded transform plan,
  and direct mutated used-plan payloads fail before manifest writes.
- Finished the half-landed `get_1d_rotary_pos_embed` surface as a bounded
  generated-op slice instead of helper math composition. Public
  `dml.ops.get_1d_rotary_pos_embed(...)` still returns a `(cos, sin)` tuple,
  but current v2 IR/runtime remain single-output per node, so the public API
  now lowers explicitly to two generated component ops,
  `get_1d_rotary_pos_embed_cos` and `get_1d_rotary_pos_embed_sin`, rather than
  claiming full v1 single-launch/two-output-kernel parity. The admitted
  contract is now explicit and tested: positive even static `dim`; `pos` as a
  positive integer sequence length or rank-1 dense
  `float32`/`float16`/`bfloat16` tensor with positive static or dynamic length;
  positive finite `theta`/`linear_factor`/`ntk_factor`; `use_real=True`
  duplicated-real outputs with both duplication conventions; `use_real=False`
  base cos/sin outputs of shape `[S, dim/2]`; and float16/float32/bfloat16
  output storage with fp32 internal math from float32 positions. Integer `pos`
  now lowers directly as two no-input generated component nodes with a static
  `sequence_length` attr instead of adding an `arange` launch. Focused tests
  now pin the two-component-node IR/lowered-IR contract, generated
  source/manifest ownership, CPU formula parity, dynamic-`S` CPU artifact
  execution, CUDA compile coverage for both real/base modes, and CUDA runtime
  parity for one `use_real=False` float32 case plus reduced-precision real
  outputs.
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
  `t5_layer_norm` slice: `float16` and `bfloat16` gained numeric CUDA runtime
  parity coverage using the existing trace/reference helpers and the generated
  CUDA artifact path. The source-side fp32-accumulation assertions stayed in
  place, so the bounded T5/RMSNorm contract remains unchanged while the
  reduced-precision runtime path is now exercised directly.
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

1. Continue the ROCm backend lane after the admitted simple generated-template
   surface by hardening edge coverage or starting exactly one provider-backed
   path with full proof. Any CK/GEMM/BMM/Conv ROCm work must include
   artifact-visible support libraries, generated lowering, copied runtime
   libraries, and a real `.venv/rocm` compile/load/run numeric proof before
   broadening dtype/op/provider claims.
2. Keep the small/custom-op lane on honest helper or bounded-op slices:
   with `gelu_new`, the now-registered generated `get_timestep_embedding`, the
   completed bounded `get_1d_rotary_pos_embed` component-op slice, and the
   newly runtime-hardened helper-only `rms_norm` slice in place, and the now-
   registered generated `layer_norm` primitive unblocking CLIP/ViT/BERT hidden-
   state normalization, plus the newly registered generated `embedding`
   primitive for learned token/position tables, plus the now-bounded
   `argmax(int32/int64)` admission needed for legacy OpenAI CLIP EOT index
   selection, prefer the next first-model enabling surface that is still
   half-finished: the remaining text-pooling gather/masking contract or a
   standard transformer masking/attention slice before
   revisiting grouped/fused normalization variants or dynamic normalized-
   dimension work. RoPE exploration/planning is now recorded in
   `agents/plans/rotary_apply_plan.md`; the next honest rotary slice is a
   downstream consumer of the landed 1D tables, such as a bounded one-tensor
   real-pair application helper or a 2D/3D table-preparation helper, not a
   speculative fused public `apply_rotary_emb` CUDA ABI. Do not restart
   `cropped_pos_embed` without new human direction. The landed `argmax`
   integer-input exception is intentionally specific to this direct CLIP
   blocker; do not use it as a reason to widen unrelated integer tensor
   support or claim broader CLIP pooling parity before non-2 EOS matching and
   the pooled hidden-state gather path are actually covered.
3. Continue the bounded ConvNd provider slice described in
   `agents/plans/conv_cutlass_plan.md` toward v1-core parity without widening
   public semantics. Human steering keeps Conv first: keep `conv2d`/explicit
   zero-bias, `conv2d_bias`, fused `conv2d_bias_relu`, and fused
   `conv2d_bias_add` honest, then add one bounded follow-on epilogue or
   no-bias/runtime-profile slice only if it
   includes frontend/admission, manifest/profile/execution-plan visibility,
   generated lowering, CUDA runtime parity, and checklist updates. The current
   useful direction is GEMM-like Conv maturity: candidate profiling/selection,
   TensorOp coverage, epilogues, and real runtime proofs for the admitted
   static rank-4, groups=1 path, not broad Conv semantics.
   Keep the work narrow:
   no conv3d, no transposed/depthwise/grouped expansion, no hidden channel
   padding, no runtime-set packed weights, and no public NHWC toggle.
4. Revisit CUTLASS/provider maturity only for another bounded compile-visible
   robustness slice if a new concrete stale-payload edge appears in an existing
   cache/test area; otherwise keep provider-cache work paused and avoid
   speculative broadening.
5. Add one more bounded GGUF regression only if another concrete loader or
   native/runtime contract edge appears, preferably around runtime load-plan
   edge cases or mixed dense/manual encoded constants on the lowered
   runtime-dequant path rather than broadening the runtime surface.
