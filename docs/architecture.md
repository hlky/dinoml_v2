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
  Toolchain, CMake, support-library cache, and artifact library copying. Backend
  extension starts in `src/dinoml/backends/registry.py`, where each target
  declares its default architecture, runtime dtype surface, build entrypoint,
  CMake capabilities, and artifact support libraries.
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

`TensorSpec` accepts static integers and `Dim` values. Frontend tensors,
parameters, and specs all canonicalize through `dinoml.Shape`, which owns the
rank, max-shape allocation view, dynamic constraints, and stable JSON
`shape_spec`. Dynamic dimensions are serialized into `shape_spec` metadata with
min/max/divisibility/bucket fields, while `shape` remains the max-shape
allocation view used for artifact metadata and workspace sizing. Generated
CPU/CUDA modules validate runtime shapes and maintain per-session shape buffers.
CUDA shape buffers live on device, so generated kernels can consume
`const int64_t*` shape metadata without reading host memory.

The Python runtime and generated modules share the same shape contract:
callers pass a `DinoTensor.shape` pointer containing `ndim` host `int64_t`
values, each run validates those values against `shape_spec`, and output shapes
are inferred by matching named dynamic dimensions from the inputs. The reusable
helpers in `dinoml.shapes` are the Python source of truth for runtime validation
and output-shape inference.
Artifacts can also mark selected public outputs with internal
`metadata.output_shape_reports` entries of kind `shape_buffer`. Those generated
CPU/CUDA modules report the post-run shape from the output tensor's generated
shape buffer instead of echoing the caller-provided output descriptor, giving
future value-dependent kernels a visible place to publish their final runtime
shape without adding public op surface. CUDA shape-buffer output reports are
host-visible only for internally synchronized runs; when callers install an
external stream, generated modules preserve the queued external-stream contract
by not copying shape buffers back to the host or synchronizing that stream, so
`dino_session_get_output_shape` remains unavailable for those shape-buffer
reports after that run.
The Python direct device-pointer frontend mirrors that contract: it still checks
reported output capacity for caller-described output shapes, but it does not
force a post-run shape query for shape-buffer-backed outputs while an external
CUDA stream is active, because generated modules intentionally avoid the
synchronization needed to make those reports host-visible in that mode.

`DinoTensor` ABI v7 also carries optional contiguous-layout metadata: host
element strides, byte capacity, byte offset, device type, flags, and pointer
alignment. Current generated modules still require row-major contiguous tensors
when stride metadata is supplied, but they apply byte offsets to logical tensor
pointers and use pointer-alignment metadata for alignment-sensitive launch
policies. These fields keep the ABI ready for future strided views and
layout-aware kernels.
The graph IR mirrors that direction with optional per-tensor `layout` metadata;
v1 currently accepts only schema-v1 dense row-major layouts with zero storage
offset and canonical element strides.

The shared dtype table now mirrors the v1 ABI direction: fp16, fp32, int32,
int64, bool, bf16, and fp8 enum slots are defined in Python and C. Runtime ABI
v7 carries dtype enums, byte sizes, dense layout metadata, eager or deferred
constant loading hooks, constant unload/reload hooks, and NumPy/Torch dtype
bridges for `float32`, `float16`, and `bfloat16`.

Current dtype support matrix:

| Surface | float32 | float16 | bfloat16 |
| --- | --- | --- | --- |
| CPU fused-elementwise runtime | yes | yes | yes |
| CPU reference executor | yes | storage/reference only | storage/reference only |
| CUDA fused-elementwise runtime | yes | yes | yes |
| CUDA `run_numpy` host staging | yes | yes | yes, stored as uint16 and returned as float32 |
| CUDA torch/device-pointer runtime | yes | yes | yes |
| CUDA runtime constants | yes | yes | yes |
| CUDA softmax/reductions | yes | no | no |
| CUTLASS GEMM runtime | yes | yes | yes |

CUDA fused-elementwise uses reduced-precision storage with fp32 accumulation by
default for fp16/bf16; the op may opt into native storage accumulation through
its lowering attributes or the development override used by benchmarks. CPU
fused-elementwise uses fp16/bf16 storage wrappers and fp32 compute by design;
native reduced-precision CPU arithmetic is intentionally not exposed until the
wrapper types gain explicit operators and vectorized conversion helpers.

## Kernel and Profiler Readiness

Each compiled artifact gets:

```text
kernel_manifest.json
kernel_codegen_plan.json
```

The manifest records the unique backend kernel/profiler symbols required by the
lowered IR. The codegen plan maps that manifest to a support-library cache
directory and carries provider-owned used-candidate keys for external libraries.
This is the hook where CUTLASS, CK, Triton, or handwritten profiler source
generation can generate all used kernels/profilers once, compile them into a
support library, and let many model artifacts reuse the result.

`dinoml profile <artifact>` is the first explicit profiler runner. It reads the
artifact graph, `kernel_manifest.json`, and `kernel_codegen_plan.json`, profiles
currently supported CUTLASS GEMM profiler symbols, writes
`debug/profile_report.json`, writes the first profile-selected
`debug/execution_plan.json`, and stores a small `profile_cache.v7.json` beside
the support-library cache. Profiling accepts repeat samples per workload and
records median/mean/min/max/stddev timing statistics while using the median
elapsed time for candidate selection. The execution plan chooses the lowest
median-time candidate per profiled node/shape and exposes a static candidate
overlay when all profiled shapes for an op/dtype/candidate-set agree and the
winner clears repeat-count, absolute/relative margin, and confidence-interval
thresholds over the runner-up. Low-confidence winners are retained in the report
for audit but are not emitted as consumable execution-plan selections; applying
an execution plan also rejects or skips explicit low-confidence selection
payloads before they can change a CUTLASS manifest. Guarded dispatch selections
also have to carry a manifest-matching `node_id`, so stale profile plans cannot
attach branches that generated lowering will ignore. GEMM profiling expands
explicit `Dim.buckets` into concrete workload cases when no runtime override is
supplied, and carries case IDs plus dynamic dim values through the report and
execution plan. Manifest/profile candidate filters now carry a CUTLASS
alignment context that combines shape/divisibility guarantees, partial A/B dense
layout alignment, known tensor or layout storage offsets, output/epilogue
alignment metadata, and the effective candidate filter. Generated CUDA tries
the selected vectorized candidate when runtime A/B logical pointers satisfy its
alignment and falls back through lower-alignment CUTLASS candidates before
failing. Profile results and execution plans now carry `split_k` and
`workspace_nbytes` as profiled launch metadata. Base, bias/activation, and
additive residual CUTLASS GEMMs expand
v1-style split-K profile variants, query the CUTLASS workspace requirement, and
lower `split_k > 1` static overlays to companion split-K launcher symbols with a
session-owned workspace. Non-additive residual and broader broadcast epilogue
families remain restricted to `split_k=1`; they need epilogue-specific
partition behavior before serial split-K can avoid reapplying residual inputs or
final activations.
`dml.compile` and `dinoml compile` can consume the static overlay
through `execution_plan=...` / `--execution-plan`, applying it before
manifest/codegen/backend build so CUDA lowering calls the profiled candidate.
If a supplied execution plan carries an `execution_plan_key`, compile verifies
that key against the plan payload before mutating the kernel manifest, and
artifacts that consume a plan record the applied plan summary in both
`compile_config.json` and top-level `manifest.json`.
As a first closed-loop compile path, `dml.compile(..., profile=True)` and
`dinoml compile --profile` now build a candidate artifact, run the CUTLASS
profiler, load the generated execution plan, and rebuild the artifact with that
plan applied. That profile-assisted path reuses one lowered IR and one
constants materialization for the candidate and final artifacts, keeping graph
preparation deterministic while still refreshing generated CUDA for the selected
plan.
Profile reports and cache keys include a best-effort CUDA hardware/toolchain
fingerprint plus support-library source/binary hashes, toolchain/dependency
provenance, so timings do not silently float across different GPUs or
regenerated support libraries. GEMM manifests now emit
explicit CUTLASS tensor-op candidates under each dtype/layout-specific candidate
set. The candidate set records provider, layout, epilogue, accumulator, target
policy, launch ABI, generator id, candidate config keys, and its own
`candidate_set_key`; future work should replace the static seed candidates with
generated CUTLASS manifest sets. `Target(use_fp16_acc=True)` filters fp16
candidate sets to fp16 accumulation and flows through CUDA lowering, profile
workload construction, support-source pruning, and cache keys. `Target(no_tf32=True)`
filters float32 GEMM to the v1 SM80 SIMT f32 fallback candidates, so TF32 stays
the default selected path while exact f32 accumulation remains available through
target policy. Residual broadcast GEMMs use a local CUTLASS selector for
TensorOp versus SIMT broadcast epilogues, so the same no-TF32 policy applies to
those fused epilogues. The CUTLASS support cache also writes a
`dinoml.support_source_manifest` at
`src/source_manifest.json`, mapping the rendered support source to the candidate
set keys, candidate config keys, launcher/profiler symbols, source metrics, and
support build units actually required by the artifact. The support manifest also
records source-size/candidate-symbol counts and total NVCC wall time for the
build. The rendered support source is pruned from the checked-in CUTLASS source
to the symbols required by the artifact's used candidate plan.

Current reusable kernels are intentionally simple:

- CUDA: reusable `libdinoml_cuda_kernels.so` for future stable primitives
- CPU: reusable `libdinoml_cpu_kernels.so`; OpenMP is optional and can be
  disabled with `-DDINOML_ENABLE_OPENMP=OFF`
- common: `dinoml/math.h` exposes inline `dinoml::math::<name>` scalar helpers
  that generated fused kernels call on CPU and CUDA. The helpers are templated
  so dtype-specific elementwise lowering has a place to land next.

Performance-sensitive ports should carry a benchmark harness before they grow a
larger policy surface. The current examples are
`tools/benchmark_fused_elementwise.py` and `tools/benchmark_softmax.py`; both
compare DinoML CPU/CUDA hot C ABI execution against NumPy and Torch references,
write JSON timing results under `tmp/`, and copy generated sources to
`tmp/.../generated_review/` for codegen inspection.
`tools/benchmark_reductions.py` follows the same pattern for row reductions.
For CUDA softmax, v2 now has small v1-inspired static last-dim policies: a
warp-per-row register path for odd/tail `K`, a float2/float4 packed
local-register path for selected divisible `K`, and a shared-memory fallback for
large reductions. CUDA reductions now use a warp-per-row path for `K <= 1024`
and a shared-memory fallback for larger rows. Broader v1-style K1/K2/K4/K8
small/middle/block policy parity and profiler selection should land before
treating softmax or reductions as done.

The generated `module.so` owns metadata loading, constant binding, pointer
binding, workspace/session allocation, shape checks, runtime shape buffers,
launch order, and model-specific generated fused-elementwise kernels. Runtime
metadata is stored as `metadata.json` in the artifact rather than embedded as a
large raw string in generated source. Fixed reusable kernels and future
CUTLASS/CK/CUB profiler libraries stay in shared support libraries and are
cached by manifest key.
For CUDA artifacts, session creation now keeps ownership local until all
session-owned workspace, temporary, and shape buffers are allocated and
initialized; if an allocation or initialization copy fails, the partially
initialized native session is destroyed before the error returns to Python.

The first CUTLASS path is concrete but still intentionally compact:
`dinoml.backends.cutlass` generates a cached `libdinoml_cutlass_gemm.so` with
real CUTLASS `gemm_rcr`, `gemm_rrr`, bias, ReLU, and v1-style bias activation
GEMM epilogue launchers, including `ELUp1`, plus first residual epilogue launchers and v1 RCR
compound residual activation epilogue launchers and profiler entrypoints for
`float32`, `float16`, and `bfloat16`.
Public `dml.ops.gemm_*` lower into dtype-resolved calls to that support library,
so model wrappers bind pointers/shapes and link `libdinoml_cutlass_gemm.so`
without embedding a handwritten matmul. These ops preserve dynamic `M/N`
metadata and launch with runtime `M/N/K`; folded leading dimensions on
`A[..., K]` are flattened into CUTLASS `m` while retaining logical `C[..., N]`
shape metadata, with first RCR/RRR folded residual coverage for
`gemm_{rcr,rrr}_bias_{add,add_relu,mul,add_add,mul_add,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}`.
The bias epilogue accepts a rank-1 `N` bias or rank-2 `[1, N]` bias, and
activation/residual epilogues instantiate CUTLASS thread epilogue functors
directly. The checked-in macro-backed support source is rendered down to the
used launcher/profiler symbols for each support build. Richer broadcast/visitor
epilogues, broader non-trailing BMM broadcast forms, grouped GEMM, and public
`matmul` layout selection remain follow-up work. Base `bmm_*` layouts now use a
separate `cutlass_bmm` support library and profile path with batch-count,
batch-stride, and leading-dimension ABI fields, v1 layout semantics, C-column
output handling, batch-broadcast strides, static profile-selected candidate
consumption, and guarded profile dispatch for conflicting batch/M/N/K shapes.
BMM `_add` variants use a separate CUTLASS add ABI for full-output and
v1-style trailing-bias `d0` tensors, passing `d0` as source C so the add remains
a CUTLASS epilogue.

GEMM metadata now has a contributor-facing split:

- `dinoml.kernels.families.gemm` owns backend-neutral layout, shape, and
  epilogue descriptors.
- `dinoml.kernels.providers.cutlass.gemm` owns CUTLASS symbol naming and
  candidate/candidate-set metadata.
- `dinoml.kernels.gemm` remains a compatibility facade for older imports.

Common runtime helper code used by generated modules lives in C++ headers under
`runtime/include/dinoml/`, so the Jinja2 templates only carry the model-specific
ABI structs, constants, pointer binding, and launch sequence.

## Alias and View Output Contract

View-only ops such as `identity`, `reshape`, `flatten`, `squeeze`, and
`unsqueeze` must not lower as empty kernels that leave public output buffers
stale. They produce an alias: one tensor name refers to the same storage as
another tensor name with a shape-only reinterpretation.
Directly returning an input or constant as a public output is normalized to the
same identity-alias metadata during tracing, so generated wrappers keep separate
ABI input and output bindings.

The authoring IR records these relationships under `metadata.views`:

```json
{
  "version": 1,
  "views": [
    {
      "tensor": "y",
      "source": "x",
      "kind": "shape_view",
      "transform": "reshape",
      "offset_elements": 0,
      "shape": [3, 2],
      "shape_spec": [3, 2]
    }
  ]
}
```

`tensor` is the alias tensor and `source` is the storage owner. For v2's initial
contract, shape views must preserve dtype, element count, and zero offset. Layout
views with non-contiguous strides, slices, storage offsets, or permutation
semantics are intentionally outside this contract until the runtime ABI carries
strides and storage ownership explicitly.

The `memory_plan` pass validates `metadata.views` and copies the normalized form
to `metadata.memory_plan.views`, which is also serialized into `metadata.json`.
CPU/CUDA lowering consumes that runtime-visible alias metadata by binding alias
tensor pointers to their owning source storage and by giving the alias tensor its
own runtime shape buffer. Alias tensors do not receive temporary storage.

When a public graph output is a shape-view alias, the ABI still supplies an
output buffer. Generated modules materialize the alias into that output buffer
after producer kernels run: CPU uses a contiguous `std::memcpy`, and CUDA
enqueues a device-to-device `cudaMemcpyAsync` on the session stream. CUDA only
synchronizes for the default internal stream path; external streams observe the
copy as queued work. Public `identity`, `reshape`, `flatten`, `squeeze`, and
`unsqueeze` now use this path and emit no compute nodes. View-of-view aliases are
rejected for now, so each alias must point directly at an input, constant,
temporary, or other owning tensor.

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
      source_manifest.json
      ops/
        <op>/
          <source-hash>.cpp | <source-hash>.cu
      CMakeLists.txt
      build/
```

Recent cleanup moved runtime metadata into `metadata.json`, changed generated
fused-elementwise function names to signature hashes such as
`fused_elementwise_<hash>`, and added baseline exact source-key deduplication for
generated kernels inside one artifact. Artifacts now also emit reviewable
per-op generated source files under `debug/generated_src/ops/` plus
`debug/generated_src/source_manifest.json`. The wrapper module remains the
compiled translation unit and still contains the generated kernel bodies until
multi-translation-unit module builds are wired deliberately.

Source dedup should happen in levels:

- Level 0: current behavior, one rendered module translation unit containing
  wrapper code plus model-generated kernels, with exact source-key deduplication
  where an op lowering provides a stable key.
- Level 1: current debug layout, write a source manifest and per-op generated
  files keyed by exact source-key hash while keeping the wrapper module as the
  compiled translation unit.
- Level 2: suppress equivalent generated kernels inside one artifact by hashing a
  normalized codegen signature and emitting one function body with multiple
  launches pointing at it.
- Level 3: promote reusable generated kernels to the support-library cache when
  their normalized signature no longer depends on model-local tensor names,
  strides, constants, or dynamic-shape symbols.

Today, per-op generated files are keyed by exact source-key hash, matching the
deduplication used inside the rendered module. The existing per-node signature is
still useful for stable launch names because generated parameter and local
variable names currently include tensor identifiers. A future normalized dedup
pass should rewrite operands and temporaries to positional names before hashing.
The source manifest maps graph node id, op, target, generated function name,
source key/hash, emitted source path, and whether that node emitted a new source
or reused an existing one.

Launch names should remain stable and human-scannable:

- Public ABI exports stay small and explicit (`dino_module_load`,
  `dino_session_run`, `dino_session_get_output_shape`, and friends).
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

## Backend and Target Registration

Targets validate through the typed backend registry in
`src/dinoml/backends/registry.py`. A new backend should start by adding a
`BackendSpec` there with:

- target name and default architecture
- supported runtime dtypes for the current compiled ABI
- build function import path, such as `dinoml.backends.cpu.build_cpu_module`
- CMake capability flags and support-library build targets
- artifact support-library paths emitted into `manifest.json`

`dinoml.backends.Target` reads defaults from this registry, so
`dml.Target("cuda")` maps to `sm_86` and `dml.Target("cpu")` maps to `native`.
`dinoml.compiler.compile` also uses the same spec for runtime dtype validation,
manifest library entries, and backend build dispatch. Lowering and per-op
support still live under `dinoml.lowering` and `dinoml.lowering.ops`; the backend
registry is only the connection point between a target name, build support, and
lowered artifact generation.

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

- richer backend capability metadata for profilers, external libraries, layout
  support, and future ROCm/Metal/Vulkan parity. V2 now has a typed
  `BackendSpec` registry for CPU/CUDA target defaults, dtype validation,
  support libraries, build dispatch, and the first CUTLASS GEMM target policy
  knobs (`no_tf32`, `use_fp16_acc`).
- runtime/container contract for allocators, graph mode, metadata,
  output-shape reporting, and externally supplied shape buffers. V2 now exposes
  `dino_session_set_stream(DinoSession*, void*)`; CUDA generated modules store a
  per-session stream, pass it to generated launches, and preserve synchronous
  default `run_numpy` behavior when no external stream is set. Generated CPU and
  CUDA sessions also expose minimal post-run output-shape reporting through
  `dino_session_get_output_shape(DinoSession*, size_t, int64_t*, size_t*)`.
  CUDA shape-buffer-backed output reports copy device shape buffers to host and
  synchronize only on the internal stream path; externally supplied streams keep
  queued/asynchronous semantics and leave those reports invalid for the run.
  Public Python session entry points reject closed sessions before C ABI calls.
  Runtime input/output maps reject unexpected tensor names before staging or
  direct pointer packing so stale caller bindings cannot be silently ignored,
  and closing a Python runtime module closes its live sessions before releasing
  the native module handle. Python runtime module construction also releases a
  native module handle if metadata initialization fails after native load
  succeeds, and rejects null native module/session handles returned from
  otherwise successful load/create calls. Generated sessions invalidate their
  previous post-run output-shape
  report immediately after rejecting a null session and before input/output
  pointer, count, tensor-validation, or constant-readiness failures, so an
  attempted native run cannot leave stale shape metadata visible through
  `dino_session_get_output_shape`. The Python
  CUDA staging allocator now grows cached session buffers
  transactionally: failed grow allocations leave the previous buffer tracked
  for later reuse or cleanup. Python session close also keeps native-session
  destruction and staging-buffer cleanup retryable separately, so a staging free
  failure does not prevent best-effort native session teardown.
- profiler source generation and multi-candidate cache selection
- liveness-based memory planning with alias/view support
- layout/accessor metadata for strides, alignment, and channel-last conventions
