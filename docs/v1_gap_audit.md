# DinoML v1 Gap Audit

This is the short list of v1 foundations that should be settled before broad op
porting. It intentionally excludes the op inventory, which lives in
`docs/op_porting_checklist.md`.

## Highest Priority

- Dynamic shape IR and runtime: v1 has `IntVar`, `IntImm`, `JaggedIntVar`,
  symbolic arithmetic, bucketed profile shapes, and max-shape runtime checks.
  V2 now records `Dim` metadata, validates runtime shapes, infers dynamic
  outputs from inputs in Python runtime helpers, and materializes CPU/CUDA
  shape buffers for generated kernels. Missing pieces: symbolic arithmetic,
  jagged dimensions, bucketed execution plans, and C ABI output-shape reporting.
- Shared dtype ABI: v1 has dtype aliases, byte sizes, torch mappings, and C ABI
  enum values for fp16, fp32, int32, int64, bool, bf16, and fp8. V2 now has the
  same enum slots plus CUDA fused-elementwise fp16/bf16 storage support. CPU and
  non-elementwise lowering remain narrower.
- Runtime/container contract: v1 has module/container/session concepts for
  streams, sync, CUDA graph mode, constants, output shape reporting, runtime
  pools, and profiling. V2 has a smaller C ABI and should grow these contracts
  before op-specific runtime assumptions spread.
- Target/backend registry: v1 registers targets and backend ops through target
  contexts and CUDA/ROCm target definitions. V2 now keeps `Target` under
  `dinoml.backends`, but still needs a richer backend registry and capability
  model.
- Profiling/cache: v1 builds candidate profilers, runs them, and stores
  hardware/compiler/op keyed cache entries. V2 has manifests and codegen-plan
  hooks, but no profiler runner or persistent result cache.

## Important Before Large Model Ports

- Memory planning: v1 has lifetime-based reuse, alias/view handling, dynamic
  bucket plans, output lifetime extension, and workspace policies. V2 currently
  uses per-session max-shape temporaries and shape buffers, without alias/view
  reuse or bucket-specific workspace plans.
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
  runtime dynamic shape buffers, constants, and generated fused elementwise, but
  still need stream ownership, bucket guards, profiler integration, richer debug
  metadata, per-op generated source files, source dedup by normalized codegen
  signature, and stable artifact source manifests.
