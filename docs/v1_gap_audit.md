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
  same enum slots plus CUDA fused-elementwise fp16/bf16 storage support. CPU and
  non-elementwise lowering remain narrower.
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
  libraries, and build dispatch. Missing pieces: richer backend capability
  metadata for profiler generation, external-library availability, layout
  support, and future ROCm/Metal/Vulkan parity.
- Profiling/cache: v1 builds candidate profilers, runs them, and stores
  hardware/compiler/op keyed cache entries. V2 has manifests and codegen-plan
  hooks, but no profiler runner or persistent result cache.

## Important Before Large Model Ports

- Memory planning: v1 has lifetime-based reuse, alias/view handling, dynamic
  bucket plans, output lifetime extension, and workspace policies. V2 currently
  uses per-session max-shape temporaries and shape buffers, and now has a
  validated `metadata.views` to `metadata.memory_plan.views` contract for
  zero-offset shape-only aliases. Runtime/lowering still reject non-empty view
  metadata, so public view ops remain blocked until alias output binding and
  temporary alias consumption are implemented.
- Layout and accessors: v1 models tensor accessors, alignment, channel-last
  conventions, and GEMM layout descriptors. V2 has a small TensorAccessor and
  CUDA vectorized dense elementwise paths, but still assumes contiguous dense
  tensors for runtime ABI and most lowering.
- Constants lifecycle: v1 distinguishes bound/unbound/owned constants, original
  names, constant folding inputs, and runtime setters. V2 now has symbolic
  parameters and runtime-settable constants, but no constant-folding lifecycle.
- Build/cache system: v1 has build-cache hashing, backend-specific builders,
  profiler builds, constants object embedding, standalone mode, and environment
  knobs. V2 has CMake support-library caching but still needs deterministic
  cache keys that cover ABI, dtype, layout, target, and toolchain.
- Codegen templates: v1 templates cover dynamic dims, bucket guards, constants,
  profiling, multistream paths, and debug metadata. V2 templates now cover
  runtime dynamic shape buffers, constants, minimal externally supplied CUDA
  streams, generated fused elementwise, per-op debug source files, and source
  manifests. Remaining gaps: bucket guards, profiler integration, richer debug
  metadata, and source dedup by normalized codegen signature.
