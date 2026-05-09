# DinoML v1 Gap Audit

This is the short list of v1 foundations that should be settled before broad op
porting. It intentionally excludes the op inventory, which lives in
`docs/op_porting_checklist.md`.

## Highest Priority

- Dynamic shape IR and runtime: v1 has `IntVar`, `IntImm`, `JaggedIntVar`,
  symbolic arithmetic, bucketed profile shapes, and max-shape runtime checks.
  V2 now records `Dim` metadata, validates runtime shapes, infers dynamic
  outputs from inputs in Python runtime helpers, and materializes CPU/CUDA
  shape buffers for generated kernels. Generated runtime sessions now also
  expose minimal post-run C ABI output-shape queries via
  `dino_session_get_output_shape`. Missing pieces: symbolic arithmetic, jagged
  dimensions, and bucketed execution plans.
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
  `dino_session_get_output_shape`, while the remaining allocator, graph, pool,
  and profiling contracts should grow before op-specific runtime
  assumptions spread.
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
  recorded as non-consumable low-confidence selections. Non-additive residual and
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
  CUDA vectorized dense elementwise paths, and ABI v5 fields for strides, byte
  capacity, device type, flags, and alignment. Current generated modules apply
  ABI byte offsets to logical tensor pointers, still require row-major
  contiguous tensors, and most lowering assumes dense layout; layout views and
  NHWC/channel-last policies remain open.
- Constants lifecycle: v1 distinguishes bound/unbound/owned constants, original
  names, constant folding inputs, and runtime setters. V2 now has symbolic
  parameters and runtime-settable constants, but no constant-folding lifecycle.
- Build/cache system: v1 has build-cache hashing, backend-specific builders,
  profiler builds, constants object embedding, standalone mode, and environment
  knobs. V2 has CMake support-library caching, CUTLASS support-build
  provenance, and manifest-derived used-candidate plans for the first GEMM
  library, but future generated profiler/support builds still need deterministic
  cache keys covering ABI, dtype, layout, target, and toolchain.
- Codegen templates: v1 templates cover dynamic dims, bucket guards, constants,
  profiling, multistream paths, and debug metadata. V2 templates now cover
  runtime dynamic shape buffers, constants, minimal externally supplied CUDA
  streams, generated fused elementwise, per-op debug source files, and source
  manifests. Remaining gaps: bucket guards, profiler integration, richer debug
  metadata, and source dedup by normalized codegen signature.

## Future Large-Model Runtime Work

- CPU weight offloading: constants should be able to begin resident on CPU and
  move to GPU on first run or planned prefetch. Design this as an extensible
  runtime policy rather than a one-off copy path so it can grow into sequential
  offload, grouped/block/layer offload, additional CUDA streams, and explicit
  prefetch/eviction scheduling.
- GGUF weight ingestion: evaluate `hlky/libgguf` for GGUF read/convert support
  and CUDA quantize/dequantize kernels. The integration should allow weights to
  load from GGUF, copy to GPU, and either dequantize the whole weight before
  launch or feed quantized storage to kernels that can dequantize directly.
  The first scaffold treats GGUF as encoded constant storage metadata with dense
  logical dtype rather than adding GGUF quantization types as normal `DinoDtype`
  values; this keeps current dense GEMM/CUTLASS lowering intact while leaving
  room for fused quantized-RHS candidate families. The Python runtime can now
  rehydrate GGUF storage metadata and refresh runtime constants through the
  dense setter path, giving us the initial dequant-then-kernel behavior.
  Artifacts with GGUF-backed constants now also write `encoded_constants.json`,
  which normalizes source metadata, logical dense shape/size, and the declared
  materialization/residency policy. The only runtime-supported policy remains
  dense dequantization before launch with eager dense device residency; GPU
  dequant, direct fused dequant-in-kernel, and CPU/offload residency modes are
  declared as future policies so the artifact contract can grow without changing
  the dense ABI again. Next GGUF work is load-time CUDA dequant/offload policy
  execution.
- Beyond-v1 CUTLASS epilogues: after v1 epilogue parity is solid, evaluate
  additional CUTLASS epilogue functors and visitor forms that can fuse common
  post-GEMM elementwise patterns beyond what DinoML v1 exposed.
- Code layout discipline: prefer reusable concrete `.h`/`.cu` support sources
  and explicit backend registries over broad text templating. V1 patterns are a
  good reference when they separate op metadata, candidate generation, profiler
  support, and handwritten kernel code.
