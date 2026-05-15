# DinoML v1 Gap Audit

This is the short list of v1 foundations that should be settled before broad op
porting. It intentionally excludes the op inventory, which lives in
`agents/plans/op_porting_checklist.md`.

## Highest Priority

- Dynamic shape IR and runtime: v1 has `IntVar`, `IntImm`, `JaggedIntVar`,
  symbolic arithmetic, bucketed profile shapes, and max-shape runtime checks.
  V2 now records `Dim` metadata, validates runtime shapes, infers dynamic
  outputs from inputs in Python runtime helpers, and materializes CPU/CUDA
  shape buffers for generated kernels. Generated runtime sessions now also
  expose minimal post-run C ABI output-shape queries via
  `dino_session_get_output_shape`. Runtime Python frontends now additionally
  materialize returned NumPy/torch tensors to each output’s reported post-run
  shape, which allows variable-size outputs without changing the frontend ABI.
  The CUDA device-pointer frontend now also validates after each run that the
  reported output shape fits inside the caller-bound output buffer shape, so
  direct pointer callers get the same capacity error contract as materializing
  NumPy/torch paths. Python runtime output-shape queries, materialization, and
  caller-bound capacity checks reject malformed negative reported dimensions
  before callers can use them for slicing or reshaping output buffers, but now
  explicitly accept zero-length reported dimensions such as `[0]` as a
  post-run shape. The Python getter also rejects ABI results whose second
  output-shape query reports a larger rank than the shape buffer allocated from
  the first query. The
  caller-bound CUDA device-pointer capacity check now has CUDA-backed
  integration coverage through a cheap generated identity artifact. Generated
  CPU/CUDA modules can now opt selected outputs into internal
  `metadata.output_shape_reports` kind `shape_buffer`, causing post-run shape
  reports to come from the generated output shape buffer rather than the
  caller-provided output descriptor. CUDA shape-buffer-backed reports preserve
  the external-stream contract by copying device shape buffers to host and
  synchronizing only on the internal stream path; externally streamed runs leave
  those reports invalid instead of blocking the caller-provided stream. The
  Python CUDA direct-pointer path now tracks externally supplied streams and
  skips only those unavailable shape-buffer report capacity checks while keeping
  caller-shape report checks intact. Missing
  pieces for value-dependent outputs are op-local generated shape-buffer count
  updates and admission of a concrete static-rank op contract.
  V2 also has a bounded frontend-only
  symbolic integer expression scaffold for add/sub/mul/floor-div over static
  integers and dynamic `Dim` metadata, and now admits those expressions into
  Python `Shape`/`TensorSpec` specs with interval-derived max-shapes plus Python
  runtime validation/output-shape inference from named input dims. Sourceable
  symbolic expressions now lower into generated CPU/CUDA shape-buffer math and
  runtime expression checks; expression leaves must have direct runtime
  `Dim` sources so lowering does not silently substitute max bounds. Profiling
  workload expansion now evaluates sourceable symbolic expressions from
  input/constant `Dim` bucket or max-shape assignments and rejects output-only
  or expression-only dimensions with a clear source error. Missing pieces:
  recovering runtime values from expression-only dimensions, full
  symbolic-expression execution-plan policy, and jagged dimensions. Bucketed
  execution plans are now implemented for CUTLASS paths: runtime shape buckets
  expand into `dim_buckets` profile workloads, build execution plans, and feed
  guarded/static selections into the kernel manifest.
- Shared dtype ABI: v1 has dtype aliases, byte sizes, torch mappings, and C ABI
  enum values for fp16, fp32, int32, int64, bool, bf16, and fp8. V2 now has the
  same enum slots plus CPU/CUDA fused-elementwise fp16/bf16 storage support for
  `run_numpy`, torch/device-pointer execution where applicable, and runtime
  constants. CUTLASS GEMM is wired for base, bias, ReLU, v1-style bias
  activation epilogues including `ELUp1`, and first rank-2 residual epilogue
  `float32`, `float16`, and `bfloat16` families, while broader broadcast epilogues, softmax, and
  reductions remain narrower.
- Runtime/container contract: v1 has module/container/session concepts for
  streams, sync, CUDA graph mode, constants, output shape reporting, runtime
  pools, and profiling. V2 now has minimal per-session CUDA stream binding via
  `dino_session_set_stream` plus post-run output-shape reporting via
  `dino_session_get_output_shape` and Python-side capacity checks for
  materialized and caller-bound output buffers. The Python getter rejects
  negative reported dimensions at the ABI boundary, and using closed modules or
  closed sessions for public lifecycle-sensitive operations now fails with clear
  Python lifecycle errors before C ABI calls. Closing a runtime module now also
  closes live Python sessions before freeing the native module, preventing stale
  session handles from retaining dangling module pointers. Dense constant
  operations still validate constant names and encoded-load policy before
  enforcing open-module state, and encoded loads now reject manifest entries
  that are not present in runtime metadata before opening encoded storage.
  Generated modules preflight constant-file extents before reload so truncated
  files cannot partially overwrite resident dense constants. Runtime
  encoded-constant loads now materialize all selected supported storage before
  calling constant setters, so a later GGUF read or validation failure does not
  partially apply earlier selected constants. Encoded loads now also restore the
  public per-constant loaded-state snapshot if a materialized constant setter
  fails, so a failed selected load does not partially advance `_constant_loaded`.
  Encoded-constant manifests are validated for object entries, names, and
  duplicate names before load planning or materialization can open external
  storage. Runtime module open, generated native `dino_module_load()` /
  `dino_module_load_constants()`, and Python `load_constants_from_file()` now
  autoload only dense constants plus GGUF constants that still declare eager
  dense residency; `manual_runtime_load` GGUF constants stay explicitly
  unloaded across open/unload/reload until `load_encoded_constants(...)` or an
  explicit native setter call materializes them. Mixed dense plus manual GGUF
  coverage now includes both Python runtime reload tests and direct CPU/CUDA
  native-boundary regressions proving eager native open/reload still require an
  explicit load/set before run.
  The Python CUDA staging allocator now preserves the currently cached session
  buffer when a grow allocation fails, so allocator failures do not leave the
  session tracking a freed pointer. CUDA staging-buffer cleanup also removes
  each successfully freed cached pointer before continuing, so a later free
  failure does not leave retry paths tracking already-freed buffers. Python
  session close now still attempts native session destruction after staging
  buffer cleanup fails, while leaving failed cleanup or failed destruction state
  retryable on the session. Runtime input/output maps now reject unexpected
  tensor names before NumPy host staging, torch dispatch validation, or direct
  CUDA pointer packing, so stale caller bindings cannot be silently ignored.
  `run_torch` also rejects mixed CUDA-device inputs before allocating outputs
  or packing raw pointers, keeping the Python device contract explicit instead
  of relying on generated CUDA failure modes. Zero-input CUDA artifacts now
  fail through an explicit `run_torch` contract error before output allocation
  because the torch frontend infers output placement from caller-provided CUDA
  inputs. Constant loaded-state introspection now also rejects closed modules
  instead of returning stale residency snapshots after `module.close()`.
  If module close sees a live-session close failure, it still attempts the
  remaining live sessions and keeps the native module handle open while
  reporting the first cleanup error. CUDA constant updates also preserve the
  primary setter/copy failure when temporary device-buffer cleanup fails, and
  Python runtime error reporting now includes CUDA helper-library last-error
  messages for allocator/copy/free failures without letting stale module
  last-error messages mask fresh CUDA helper failures. Generated CUDA session
  creation now destroys partially initialized sessions when session-owned
  workspace, temporary, or shape-buffer allocation/copy fails before returning a
  handle to Python. Python runtime module construction now also frees a native
  module handle when metadata initialization fails after native load succeeds,
  and rejects successful native module/session creation calls that return null
  handles before metadata reads or session tracking can proceed.
  Generated CPU/CUDA sessions now clear the previous output-shape report
  immediately after rejecting a null session and before each native run's
  input/output pointer, count, validation, and constant-readiness checks, so
  failed attempted runs do not expose stale post-run shapes from an earlier
  successful execution.
  The remaining graph, pool, profiling, and broader allocator contracts should
  grow before op-specific runtime assumptions spread.
- Target/backend registry: v1 registers targets and backend ops through target
  contexts and CUDA/ROCm target definitions. V2 now has a typed CPU/CUDA
  `BackendSpec` registry for target defaults, dtype validation, support
  libraries, build dispatch, and first CUTLASS GEMM policy flags for optional
  TF32 and fp16 accumulation. Missing pieces: richer backend capability metadata
  for profiler generation, external-library availability, layout support, and
  future ROCm/Metal/Vulkan parity. CUDA GEMM now resolves
  `float32`/`float16`/`bfloat16` launcher variants through op-owned kernel
  bindings, and the first explicit profiler runner consumes those variants for
  explicit CUTLASS tensor-op candidate sets, including bias, ReLU, v1-style
  bias activation, and first rank-2 residual epilogue variants.
  `use_fp16_acc=True` now changes the manifest/profile/build candidate set for
  fp16 GEMM; `no_tf32=True` now filters float32 GEMM, including residual
  broadcast epilogues, to v1 SM80 SIMT f32 fallback candidates.
- Profiling/cache: v1 builds candidate profilers, runs them, and stores
  hardware/compiler/op keyed cache entries. V2 has manifests, codegen-plan
  hooks, and a JSON cache/report for CUTLASS GEMM candidate profiles.
  Profile keys now include a best-effort CUDA hardware/toolchain fingerprint,
  support-library source/binary hashes, CUTLASS support-build provenance, and
  target-policy-specific candidate/config keys. Profiling now also writes
  `debug/execution_plan.json`, selecting the fastest measured candidate per
  profiled node/shape and exposing a static overlay when all profiled shapes for
  an op/dtype/candidate-set agree. Compile can consume that static overlay via
  `execution_plan=...` or `--execution-plan` before CUDA lowering/codegen, and
  now verifies keyed execution plans against their payload before applying
  provider selections. Artifacts that consume an execution plan expose the plan
  summary in both `compile_config.json` and top-level `manifest.json`.
  The first opt-in `profile=True` / `compile --profile` path now automates the
  build-profile-rebuild loop around the existing artifact profiler. That
  profile-assisted compile path now runs graph passes and constants
  materialization once, then builds candidate and selected artifacts from the
  same lowered IR.
  GEMM profiling expands explicit `Dim.buckets` into concrete workload cases and
  carries bucket case metadata into profile reports and execution plans. The
  alignment context now prunes CUTLASS profiler workloads from shape-derived
  caps, partial A/B dense layout alignment, known storage offsets, and current
  C/epilogue alignment metadata. Generated CUDA can fall back through
  lower-alignment CUTLASS candidates when runtime logical pointers do not meet
  the selected vectorized candidate. The first split-K surface preserves
  `split_k` and `workspace_nbytes` through profile results, cache keys,
  execution plans, and static overlays. Base, bias/activation, and additive
  residual CUTLASS GEMMs now profile v1-style split-K variants and lower
  `split_k > 1` static overlays through companion launcher/profiler symbols plus
  a session-owned workspace. Profiling can now collect repeated timing samples
  per workload, store median/mean/min/max/stddev timing statistics, and select
  on median elapsed time only when repeat-count, absolute/relative margin, and
  confidence-interval thresholds clear the runner-up; close/noisy winners are
  recorded as non-consumable low-confidence selections, and explicit
  low-confidence static/guarded execution-plan payloads are rejected or skipped
  during plan application before they can mutate a CUTLASS manifest. CUTLASS
  execution-plan application also rejects or skips stale launcher/profiler
  symbol payloads, malformed guarded shape metadata, missing or stale guarded
  `node_id` values, and duplicate static or guarded selections for the same
  manifest key before attaching guarded dispatch to a manifest.
  Profile cache loading now discards malformed entry
  maps and refuses cache hits whose stored timing
  statistics have malformed count fields, do not contain enough samples for
  the requested repeat-count confidence policy, or carry a `profile_key` that is
  missing or inconsistent with the cache map key, or whose embedded key payload
  no longer matches the current hardware/support/profile key payload. Cache
  reads now also reject entries whose embedded payload no longer hashes to the
  stored `profile_key` or whose embedded target no longer matches the cache
  target, and cache writes drop those stale on-disk entries instead of merging
  them forward. Profile
  cache writes now merge valid same-target on-disk entries before writing so a
  stale writer preserves entries added by another profiling process while still
  replacing its own profile keys. CUTLASS support-cache reuse now also rejects
  malformed or stale `src/source_manifest.json` payloads instead of treating
  mere file presence as a cache hit, including embedded `used_candidate_plan`
  payloads that no longer hash to the stored `used_candidate_plan_key`. The
  CUDA artifact profiler now scopes
  allocator/copy/free helper errors to the CUDA runtime helper library before
  falling back to common runtime errors, and profiler cleanup retains failed
  device pointers for retry instead of dropping them before a successful free.
  Non-additive residual and
  broader broadcast split-K remain intentionally disabled until their fused
  epilogues have correct partition behavior.
  Remaining gaps are non-additive residual/broadcast split-K coverage and
  persistent SQLite/shared cache workflows.

## Important Before Large Model Ports

- Memory planning: v1 has lifetime-based reuse, alias/view handling, dynamic
  bucket plans, output lifetime extension, and workspace policies. V2 currently
  uses per-session max-shape temporaries and shape buffers, and now has a
  validated `metadata.views` to `metadata.memory_plan.views` contract for
  zero-offset shape-only aliases. Runtime/lowering consume those aliases for
  direct views of inputs, constants, temporaries, and owning tensors, and
  materialize public alias outputs into ABI output buffers. Public `identity`,
  `reshape`, `flatten`, `squeeze`, and `unsqueeze` use that path. Remaining
  gaps: view-of-view normalization, liveness extension beyond the current static
  temporary plan and strided/layout views.
- Layout and accessors: v1 models tensor accessors, alignment, channel-last
  conventions, and GEMM layout descriptors. V2 has a small TensorAccessor,
  CUDA vectorized dense elementwise paths, and ABI v7 fields for strides, byte
  capacity, device type, flags, and alignment. Current generated modules apply
  ABI byte offsets to logical tensor pointers, still require row-major
  contiguous tensors, and most lowering assumes dense layout; layout views and
  NHWC/channel-last policies remain open. ConvNd planning is now explicitly
  captured in `agents/plans/conv_cutlass_plan.md`: keep public ConvNd semantics
  source-faithful NCHW/NCDHW, treat NHWC/NDHWC as guarded provider-internal
  islands only, and require generated pack/unpack temporaries plus manifest
  metadata instead of relying on ABI strides for layout translation. The first
  `conv2d_bias` slice now lets CUDA compile emit intended NHWC/OHWI provider
  transforms as kernel-manifest/codegen metadata and runs bounded fp16 plus
  exact float32 SIMT groups=1 static rank-4 provider paths. The static profile workload builder
  emits `cutlass_conv` workloads from that explicit transform plan, rejects
  missing transform metadata, and carries Conv-specific layout transform,
  weight transform, Conv config, candidate/config, and support provenance into
  profile reports and cache keys. CUDA compile now also materializes a
  support-cache/source-manifest
  boundary for `cutlass_conv` so provider transform provenance is visible; with
  `nvcc` available that boundary compiles `libdinoml_cutlass_conv.so` with
  transform helpers, correctness-first SIMT CUTLASS
  `device::ImplicitGemmConvolution` Fprop+bias launchers for fp16 and exact
  float32, a
  v1-inspired TensorOp `IteratorAlgorithm::kFewChannels` fp16 launcher selected
  only for semantic input `C=3`, v1-inspired TensorOp
  `IteratorAlgorithm::kFixedChannels` fp16 launchers selected only for semantic
  input `C=4` or `C=8`, a regular TensorOp
  `IteratorAlgorithm::kOptimized` fp16 launcher selected only for naturally
  aligned non-small-channel shapes (`C >= 16` with input/output channels
  divisible by 8), and real profiler exports for all emitted runtime
  candidates. The shared
  Conv scaffold transform plan is now also validated for internal coherence
  before profiling/codegen/support-cache consumers can reuse it, so layout
  drift, incorrect temporary byte counts, and inconsistent padded-channel
  metadata fail explicitly instead of silently propagating through artifact
  provenance. `kernel_codegen_plan.json` now additionally records explicit
  wrapper-stage metadata for activation pack, weight pack, planned provider
  launch, and output unpack, and rejected/compiled CUDA artifacts also emit
  guarded debug wrapper scaffold `.cu` sources plus a small scaffold-source
  manifest under `debug/generated_src/`, linked from `kernel_codegen_plan.json`
  for artifact-side inspection. Generated CUDA modules now consume that same
  plan enough to allocate the per-session Conv pack/unpack temporaries, call
  the support-library transform helpers, call the selected provider launcher
  symbol, and unpack outputs back to NCHW. Focused CUDA runtime parity covers
  the bounded fp16 C=3 few-channel path, C=4 fixed-channel path, and optimized
  C=16/O=16 path against Torch, plus the exact CLIP float32 patch-projection
  SIMT path against local Transformers/Torch, while manifest/source tests keep
  C=8 artifact-visible and keep unaligned non-small-channel shapes on the SIMT
  fallback with no hidden channel padding. Other float32 Conv shapes remain
  scaffold-only. Conv profile workload construction
  now filters candidates through the same shape/layout/dtype predicate used by
  manifest selection, so incompatible C=3/C=4/C=16 candidates are no longer
  emitted. `profile_artifact` now profiles those Conv candidates on
  provider-layout buffers, writes report/cache/plan artifacts, and static Conv
  execution-plan application updates both manifest symbols and the
  `cutlass_conv_plan["selected_candidate"]` payload consumed by generated
  lowering. Dynamic Conv profiling, guarded Conv dispatch, and general
  channel-last runtime layout remain unimplemented.
- Constants lifecycle: v1 distinguishes bound/unbound/owned constants, original
  names, constant folding inputs, and runtime setters. V2 now has symbolic
  parameters and runtime-settable constants. Runtime constant setters now
  consistently reject unknown constant names before backend availability or
  tensor-specific checks, but V2 still has no constant-folding lifecycle.
- Build/cache system: v1 has build-cache hashing, backend-specific builders,
  profiler builds, constants object embedding, standalone mode, and environment
  knobs. V2 has CMake support-library caching, CUTLASS support-build
  provenance, and manifest-derived used-candidate plans for the first GEMM
  library, but future generated profiler/support builds still need deterministic
  cache keys covering ABI, dtype, layout, target, and toolchain.
- Codegen templates: v1 templates cover dynamic dims, bucket guards, constants,
  profiling, multistream paths, and debug metadata. V2 templates now cover
  runtime dynamic shape buffers, sourceable symbolic integer expression shape
  math, constants, minimal externally supplied CUDA streams, generated fused
  elementwise, per-op debug source files, and source manifests. Remaining gaps:
  bucket guards beyond current CUTLASS plan dispatch, expression-only profiling
  source recovery, richer debug metadata, and source dedup by normalized codegen
  signature.

## Future Large-Model Runtime Work

- CPU weight offloading: constants should be able to begin resident on CPU and
  move to GPU on first run or planned prefetch. Design this as an extensible
  runtime policy rather than a one-off copy path so it can grow into sequential
  offload, grouped/block/layer offload, additional CUDA streams, and explicit
  prefetch/eviction scheduling. Runtime modules now expose deferred initial
  constant loading plus an explicit unload/reload lifecycle for dense constants,
  which gives a future offload scheduler the basic residency and eviction
  primitives while keeping module load eager by default for existing callers.
  Artifacts can now declare an eager or deferred constant-load policy, and the
  Python runtime honors that policy by default while still allowing callers to
  override it. The Python runtime also exposes per-constant loaded-state
  introspection so future offload policy code can reason about residency before
  adding selective prefetch/eviction.
- GGUF weight ingestion: evaluate `hlky/libgguf` for GGUF read/convert support
  and CUDA quantize/dequantize kernels. The integration should allow weights to
  load from GGUF, copy to GPU, and either dequantize the whole weight before
  launch or feed quantized storage to kernels that can dequantize directly.
  The first scaffold treats GGUF as encoded constant storage metadata with dense
  logical dtype rather than adding GGUF quantization types as normal `DinoDtype`
  values; this keeps current dense GEMM/CUTLASS lowering intact while leaving
  room for fused quantized-RHS candidate families. The Python runtime can now
  expose an encoded-constant load plan, selectively rehydrate runtime-supported
  GGUF storage metadata, and refresh constants through the dense setter path,
  giving us the initial dequant-then-kernel behavior without adding streaming
  or offload execution yet.
  Artifacts with GGUF-backed constants now also write `encoded_constants.json`,
  which normalizes source metadata, logical dense shape/size, and the declared
  materialization/residency policy. Runtime-supported policy now covers dense
  dequantization before launch with eager dense device residency and manual
  runtime encoded-constant loading through
  `RuntimeModule.load_encoded_constants(names=...)`. Encoded loads validate the
  selected names, declared policies, and runtime metadata membership before
  enforcing open-module lifecycle, then reject closed modules before opening or
  materializing encoded storage. GGUF source metadata now normalizes relative
  compile-time paths to absolute source paths, and runtime encoded loads
  resolve relative manifest paths against the artifact directory before opening
  GGUF storage.
  CUDA artifacts now have a bounded load-time GGUF dequant branch: if
  `libgguf.libgguf_cuda` is importable, Torch CUDA is available, and
  `torch.ops._C_gguf.dequantize` is registered, selected supported packed rows
  are dequantized into a CUDA tensor and copied into the generated module's
  dense constant storage through `set_constant_device_pointer`. Missing CUDA
  dequant support, CPU artifacts, and dense GGUF `F32`/`F16` storage preserve
  the existing host materialization path. The current tested CUDA runtime slice
  uses real libgguf `Q4_0` storage with manual runtime loading, unload/reload
  state checks, and output correctness. The CUDA load is synchronous with
  respect to the produced torch tensor and still uses the dense runtime ABI.
  A generated pre-GEMM runtime path now exists for `gemm_rrr`, `gemm_rcr`,
  `gemm_rrr_bias`, and `gemm_rcr_bias` with a GGUF RHS constant declared as
  `dequantize_on_gpu_before_launch` plus
  `manual_runtime_load`: the artifact stores encoded bytes, Python runtime
  loading installs those encoded bytes into CUDA constant storage, generated
  manifests expose a shared per-session `gguf_runtime_dequant_scratch`
  resource sized to the maximum lowered dense RHS requirement, and modules now
  prefer a direct link against a native `libgguf_cuda_native` library built
  from the repo-pinned `third_party/libgguf` submodule. The old explicit native
  `libgguf_cuda_dequantize_rows_on_stream` function-pointer setter remains as a
  bounded fallback/testing path, sessions own a separate dense dequant scratch
  buffer, and lowering dequantizes on the session stream immediately before the
  existing dense CUTLASS GEMM launch. The tested slice uses real libgguf `Q4_0`
  storage, includes dense-bias load/unload/reload coverage, and compares
  against a dense dequantized reference. Non-bias GEMM epilogues, `bfloat16`, direct fused
  dequant-in-kernel, prefetch/eviction, and new CPU/GPU residency policies
  remain future work.
- Beyond-v1 CUTLASS epilogues: after v1 epilogue parity is solid, evaluate
  additional CUTLASS epilogue functors and visitor forms that can fuse common
  post-GEMM elementwise patterns beyond what DinoML v1 exposed.
- Code layout discipline: prefer reusable concrete `.h`/`.cu` support sources
  and explicit backend registries over broad text templating. V1 patterns are a
  good reference when they separate op metadata, candidate generation, profiler
  support, and handwritten kernel code.
