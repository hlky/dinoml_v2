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
  before callers can use them for slicing or reshaping output buffers, and the
  Python getter now rejects ABI results whose second output-shape query reports a
  larger rank than the shape buffer allocated from the first query.
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
  enforcing open-module state, and generated modules preflight constant-file
  extents before reload so truncated files cannot partially overwrite resident
  dense constants. The Python CUDA staging allocator now preserves the currently
  cached session buffer when a grow allocation fails, so allocator failures do
  not leave the session tracking a freed pointer. CUDA staging-buffer cleanup
  also removes each successfully freed cached pointer before continuing, so a
  later free failure does not leave retry paths tracking already-freed buffers.
  If module close sees a live-session close failure, it still attempts the
  remaining live sessions and keeps the native module handle open while
  reporting the first cleanup error. CUDA constant updates also preserve the
  primary setter/copy failure when temporary device-buffer cleanup fails, and
  Python runtime error reporting now includes CUDA helper-library last-error
  messages for allocator/copy/free failures without letting stale module
  last-error messages mask fresh CUDA helper failures.
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
  the first opt-in `profile=True` / `compile --profile` path now automates the
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
  recorded as non-consumable low-confidence selections. Profile cache loading
  now discards malformed entry maps and refuses cache hits whose stored timing
  statistics have malformed count fields or do not contain enough samples for
  the requested repeat-count confidence policy. The CUDA artifact profiler now
  scopes allocator/copy/free helper errors to the CUDA runtime helper library
  before falling back to common runtime errors, and profiler cleanup retains
  failed device pointers for retry instead of dropping them before a successful
  free. Non-additive residual and
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
  NHWC/channel-last policies remain open.
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
  selected names and declared policies before enforcing open-module lifecycle,
  then reject closed modules before opening or materializing encoded storage.
  GPU dequant, direct fused dequant-in-kernel, and CPU/offload prefetch/eviction
  residency modes remain future policies so the artifact contract can grow
  without changing the dense ABI again. Next GGUF work is true load-time CUDA
  dequantization, CPU/GPU offload, prefetch, and eviction policy execution.
- Beyond-v1 CUTLASS epilogues: after v1 epilogue parity is solid, evaluate
  additional CUTLASS epilogue functors and visitor forms that can fuse common
  post-GEMM elementwise patterns beyond what DinoML v1 exposed.
- Code layout discipline: prefer reusable concrete `.h`/`.cu` support sources
  and explicit backend registries over broad text templating. V1 patterns are a
  good reference when they separate op metadata, candidate generation, profiler
  support, and handwritten kernel code.
