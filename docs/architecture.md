# DinoML v2 Architecture Notes

This snapshot is shaped for porting real kernels and profiling without turning
generated model code into the kernel library.

## Boundaries

- `dinoml.ops`
  User-facing op constructors and per-family op specs.
- `dinoml.ops.registry`
  Backend-independent registry primitives: public frontend binding, typed op
  schema, shape inference, dtype support, backend kernel symbols, and profiler
  symbols.
- `dinoml.ops.definitions`
  The small assembly point that registers per-family op definitions.
- `dinoml.passes`
  IR validation, shape/type inference, DCE, fusion metadata, memory planning,
  and backend lowering markers.
- `dinoml.fusions`
  Graph rewrites such as elementwise subgraph fusion.
- `dinoml.lowering`
  Target-specific launch sequencing and generated-wrapper render context.
- `dinoml.lowering.ops`
  Per-op lowering registrations. Adding a compiled op should add a module here
  with generated-kernel rendering, launch-code rendering, and any op-local
  templates.
- `dinoml.backends`
  Toolchain, CMake, support-library cache, and artifact library copying.
- `runtime/`
  Stable common ABI plus CUDA runtime helper library.
- `kernels/`
  Reusable CPU/CUDA kernel libraries plus shared headers such as
  `dinoml/math.h`. Model artifacts link against reusable kernels and embed only
  model-specific generated kernels, such as fused-elementwise combinations.

## Frontend Contract

`Parameter` is symbolic by default, matching the v1 direction: it records shape,
dtype, name, and optional value. Concrete weights are bound through
`trace(..., constants=...)`, `ModelSpec.bind_constants(...)`, or runtime constant
loading. This keeps model definition separate from checkpoint storage.

`TensorSpec` accepts static integers and `Dim` values. Dynamic dimensions are
serialized into `shape_spec` metadata with min/max/divisibility/bucket fields,
while `shape` remains the max-shape allocation view used for artifact metadata
and workspace sizing. Generated CPU/CUDA modules validate runtime shapes and
maintain per-session shape buffers. CUDA shape buffers live on device, so
generated kernels can consume `const int64_t*` shape metadata without reading
host memory.

The Python runtime and generated modules share the same shape contract:
callers pass a `DinoTensor.shape` pointer containing `ndim` host `int64_t`
values, each run validates those values against `shape_spec`, and output shapes
are inferred by matching named dynamic dimensions from the inputs. The reusable
helpers in `dinoml.shapes` are the Python source of truth for runtime validation
and output-shape inference.

The shared dtype table now mirrors the v1 ABI direction: fp16, fp32, int32,
int64, bool, bf16, and fp8 enum slots are defined in Python and C. CPU runtime
lowering currently accepts float32. CUDA fused-elementwise lowering supports
float32, float16, and bfloat16 storage with optional fp32 accumulation.

## Kernel and Profiler Readiness

Each compiled artifact gets:

```text
kernel_manifest.json
kernel_codegen_plan.json
```

The manifest records the unique backend kernel/profiler symbols required by the
lowered IR. The codegen plan maps that manifest to a support-library cache
directory. This is the hook where CUTLASS, CK, Triton, or handwritten profiler
source generation can generate all used kernels/profilers once, compile them into
a support library, and let many model artifacts reuse the result.

Current reusable kernels are intentionally simple:

- CUDA: reusable `libdinoml_cuda_kernels.so` for future stable primitives
- CPU: reusable `libdinoml_cpu_kernels.so`; OpenMP is optional and can be
  disabled with `-DDINOML_ENABLE_OPENMP=OFF`
- common: `dinoml/math.h` exposes inline `dinoml::math::<name>` scalar helpers
  that generated fused kernels call on CPU and CUDA. The helpers are templated
  so dtype-specific elementwise lowering has a place to land next.

The generated `module.so` owns metadata loading, constant binding, pointer
binding, workspace/session allocation, shape checks, runtime shape buffers,
launch order, and model-specific generated fused-elementwise kernels. Runtime
metadata is stored as `metadata.json` in the artifact rather than embedded as a
large raw string in generated source. Fixed reusable kernels and future
CUTLASS/CK/CUB profiler libraries stay in shared support libraries and are
cached by manifest key.

Common runtime helper code used by generated modules lives in C++ headers under
`runtime/include/dinoml/`, so the Jinja2 templates only carry the model-specific
ABI structs, constants, pointer binding, and launch sequence.

## Generated Source Cleanliness Roadmap

The current artifact layout is intentionally reviewable but still coarse:

```text
artifact/
  manifest.json
  metadata.json
  graph.dinoir.json
  compile_config.json
  kernel_manifest.json
  kernel_codegen_plan.json
  module.so
  lib/
  debug/
    pass_dumps/
    generated_src/
      module.cpp | module.cu
      CMakeLists.txt
      build/
```

Recent cleanup moved runtime metadata into `metadata.json`, changed generated
fused-elementwise function names to signature hashes such as
`fused_elementwise_<hash>`, and added baseline exact source-key deduplication for
generated kernels inside one artifact. The next cleanup should make generated
sources more normalized and easier to inspect without changing the runtime ABI.

Source dedup should happen in levels:

- Level 0: current behavior, one rendered module translation unit containing
  wrapper code plus model-generated kernels, with exact source-key deduplication
  where an op lowering provides a stable key.
- Level 1: suppress equivalent generated kernels inside one artifact by hashing a
  normalized codegen signature and emitting one function body with multiple
  launches pointing at it.
- Level 2: split model-generated kernels into per-op source files under
  `debug/generated_src/ops/<op>/<hash>.<cpp|cu>` and keep the module source as
  ABI wrapper plus includes/launch sequencing.
- Level 3: promote reusable generated kernels to the support-library cache when
  their normalized signature no longer depends on model-local tensor names,
  strides, constants, or dynamic-shape symbols.

Per-op generated files should be keyed by a normalized signature that rewrites
operands and temporaries to positional names before hashing. The existing
per-node signature is still useful for stable launch names because generated
parameter and local variable names currently include tensor identifiers. Once
per-op files exist, the debug layout should include a small source manifest that
maps graph node id, launch symbol, normalized kernel hash, source path, template
name, target, dtype/layout assumptions, and dynamic-shape dependencies.

Launch names should remain stable and human-scannable:

- Public ABI exports stay fixed (`dino_module_load`, `dino_session_run`, and
  friends).
- Wrapper-local launch helpers should use graph-order or node-stable names, not
  raw fused node ids.
- Kernel implementation names should be content-addressed by normalized
  signature. If a debug name includes an op family, use it as a prefix only,
  e.g. `fused_elementwise_<hash>`.

The artifact debug/source layout should preserve everything needed to reproduce
or review a generated module without making generated code part of the committed
kernel library. Build trees may stay under `debug/generated_src/build/`, but the
reviewable files should be outside the build tree and grouped by role: wrapper
source, per-op generated sources, source manifest, rendered CMake, and pass
dumps.

## Op Registration Contract

Every v2 op family should keep its semantic definitions in a per-family module
under `dinoml.ops` and register through `dinoml.ops.registry`:

- `OpSchema` records tensor input names and typed attributes.
- `FrontendBinding` exposes a Python constructor and default attrs.
- `KernelBinding` records the CPU/CUDA symbol, shared kernel library, and
  optional profiler symbol.

The user-facing functions in `dinoml.ops` are generated from that registry. This
keeps frontend ops, pass validation, manifest generation, support-library
selection, and future profiler codegen pointed at the same source of truth.

## Fused Elementwise Slice

V1 represents scalar `elementwise` ops first, then rewrites connected
elementwise subgraphs into an internal `fused_elementwise` op. The backend emits
one kernel that evaluates the topologically sorted scalar subexpressions and can
eventually handle broadcasting, vectorized reads, jagged indexing, and multiple
outputs.

V2 now follows that direction for registered dense elementwise ops. The pass
rewrites connected unary/binary elementwise subgraphs into a `fused_elementwise`
node with topologically sorted `sub_ops` metadata. CPU/CUDA lowering renders a
model-specific kernel that calls `dinoml::math::<name>` for each scalar
operation, so combinations such as `mul -> add -> sigmoid -> sub -> relu` do not
require fixed-pattern support-library kernels. Generated function names are
stable signature hashes such as `fused_elementwise_<hash>`, so graph node IDs do
not leak into source names.

Current limits are deliberate: contiguous dense buffers, no jagged tensors yet,
and only simple dense/suffix/generic broadcasting. CUDA has vectorized paths for
aligned dense fused-elementwise graphs and reduced-precision storage. CPU is
kept scalar and portable for now.

## V1 Audit Priorities

The next foundations to settle before broad op porting are:

- richer target/backend registry beyond the current `dinoml.backends.Target`
- runtime/container contract for streams, allocators, graph mode, metadata,
  output-shape reporting, and externally supplied shape buffers
- profiler source generation, runner, and cache schema
- liveness-based memory planning with alias/view support
- layout/accessor metadata for strides, alignment, and channel-last conventions
