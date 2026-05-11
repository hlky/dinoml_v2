# DinoML v1 Op Porting Checklist

This checklist maps ops from `/workspace/dinoml/src/dinoml/compiler/ops` to the
DinoML v2 port. It is organized by implementation family, not by every v1 Python
class, so porting can happen through reusable kernels and registrations instead
of one-off clones.

Status markers:

- [x] Available in v2 MVP.
- [ ] Not ported yet.

For each public op or family, add schema, frontend binding, CPU binding, CUDA
binding, profiler binding when relevant, shape/type inference, and tests. Prefer
one semantic v2 op plus backend variants over separate user-facing registrations
for every v1 fused/layout specialization.

## Local PyTorch Reference

Brief local scan: `torch 2.9.1+cu128` is installed. Broad support exists in
`torch`, `torch.nn`, and `torch.nn.functional` for common tensor math, matmul,
BMM, reductions, indexing, conv/pool/pad/upsample, normalization, activations,
and scaled-dot-product attention.

Broad gaps versus core PyTorch are the v1-specific categories: layout-fused
GEMM/BMM epilogues, jagged/ragged conversion semantics, positional/model helper
ops, FIR resampling helpers, NMS/ROI detection helpers, NHWC/NDHWC packing
helpers, and exact flash/memory-efficient attention variants.

## V2 MVP Surface

- [x] Dense elementwise math surface - public frontend ops lower into
  model-generated `fused_elementwise` kernels.
- [x] `gelu` - v2 native frontend op, tanh approximation.
- [x] Runtime shape buffers for dynamic shape validation and generic
  fused-elementwise broadcasting.

## Common Primitives

These should be reusable building blocks. They generally map to `torch` or
`torch.nn.functional` semantics for reference tests.

### Elementwise, Activations, and Scalar Math

- [x] `elementwise`: initial dense coverage for arithmetic,
  min/max, trig/log/exp/sqrt, activations, `nan_to_num`,
  `clamp_nan_to_num`, `pow`, `floor_div`, `floor`, and relational ops
  `eq`/`ge`/`gt`/`le`/`lt`/`ne` with bool outputs. Remaining v1 parity work:
  jagged broadcasting, broader CPU/vector accessors, scalar dtype promotion,
  and exhaustive edge-case tests.
- [x] `fused_elementwise`: connected registered unary/binary elementwise
  subgraphs lower to model-generated CPU/CUDA kernels that call
  `dinoml::math::<name>` helpers. CPU and CUDA support float32, float16, and
  bfloat16 storage; CUDA has optional fp32 accumulation and vectorized dense
  paths, while CPU reduced precision always computes in fp32 for now. Runtime
  shape buffers support generic broadcasting, and standalone relational fused
  outputs use bool storage while keeping float inputs typed as float pointers.
  Public `cast(x, dtype)` is also covered for dense one-input casts between
  `float32`, `float16`, `bfloat16`, and `bool`, preserving shape/spec and
  lowering through fused elementwise with mixed input/output pointer types.
  `int32`/`int64` casts remain out of scope until generated storage/lowering
  support exists for those dtypes.
  Multi-output same-shape metadata is represented; broader tests and v1-style
  jagged codegen remain.
- [x] `int_elementwise`: frontend-only symbolic integer expression scaffold for
  `ADD`, `SUB`, `MUL`, and `DIV` via `dml.ops.int_add`, `int_sub`, `int_mul`,
  and `int_div`. Pure static expressions constant-fold, dynamic expressions
  serialize as JSON-compatible dicts, and `DIV` uses Python floor-division
  semantics. Bounded shape-spec support now admits `kind: int_expr` dimensions
  into `Shape`/`TensorSpec`, computes max-shapes from recursive intervals,
  validates runtime expression values in Python helpers, and infers output
  expression dimensions from named input `Dim` values. Sourceable expressions
  lower into generated CPU/CUDA shape-buffer math with runtime expression
  checks; lowering now rejects expressions whose named leaves lack direct
  runtime sources instead of falling back to max bounds. Profiling workload
  expansion now evaluates sourceable expression dimensions from bucket/max
  assignments and rejects unsourced expressions; expression-only source recovery
  remains future work.
- [x] Public math helpers: `tanh`, `cos`, `sin`, `sign`, `abs`, `log`, `log1p`,
  `exp`, `sqrt`, `max`, `min`, `sigmoid`, `leaky_relu`, `hardtanh`, `relu`,
  `silu`, `nan_to_num`, `pow`, `fast_gelu`, `softplus`, `elu`, `softsign`,
  `floor_div`, `celu`, `floor`, `eq`, `ge`, `gt`, `le`, `lt`, `ne`, plus
  `sub`, `mul`, `div`, and `clamp_nan_to_num`.

Library hints: CPU can use scalar loops first, then `std::simd` or xsimd for
vector paths. CUDA/HIP elementwise kernels are usually simpler than library
calls. GEMM epilogue activations should be expressed through CUTLASS or CK
epilogues where possible.

### Views, Layout, Shape, and Selection

- [x] View-only: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `identity`.
  These public frontend ops now emit `metadata.views` shape aliases with no
  compute nodes; lowering/runtime consume the validated
  `metadata.memory_plan.views` form and materialize public alias outputs into ABI
  output buffers. Current limits: view-of-view aliases are rejected, reshape only
  accepts static input shapes, flatten only accepts static dimensions in the
  flattened range, and scalar view tensors are not exposed yet. Layout-changing
  `permute`/`transpose` are available as bounded materialized dense copies
  rather than metadata-only views. Specialized frontend layout helpers
  `permute021`, `permute0213`, `permute102`, and `permute210` are public wrappers
  over `permute`, sharing the same static-shape, dtype, validation, and generated
  kernel limits without adding separate kernels.
- [x] Symbolic shape/container helpers: `size`, `getitem`, `tuple_construct`,
  `list_construct` are available as bounded public Python helpers for
  model-building. They do not emit IR nodes or metadata: `size` reads tensor
  `shape_spec` entries, including dynamic `Dim` JSON metadata, while
  `getitem`, `tuple_construct`, and `list_construct` preserve normal Python
  container behavior. Current limits: `size` accepts only integer non-bool
  dimensions with normalized negative axes, and `getitem` rejects bool indexes
  before delegating to Python indexing.
- [x] Layout: `pixel_shuffle`, `pixel_unshuffle` are available as bounded
  frontend helpers for rank-4 static-shape tensors with positive integer
  factors and required channel/spatial divisibility. They compose reshape view
  metadata with the existing generated `permute` materialized copy, so they
  share the generated float/reduced-precision/bool storage surface and do not
  add separate kernels. General `permute` and frontend `transpose` are available
  for one static-shape tensor, full normalized permutations without duplicates,
  and that same generated storage surface. Dynamic shapes and non-rank-4 pixel
  shuffle variants remain out of scope.
- [x] Creation/shape values: `meshgrid` is available as a bounded frontend helper
  for a non-empty list/tuple of rank-1 static tensors with matching
  `float32`/`float16`/`bfloat16`/`bool` dtype and `indexing="ij"` only; it
  composes reshape view metadata with generated `expand` copy nodes, so dynamic
  lengths, `xy` indexing, mixed dtypes, and non-rank-1 inputs remain out of
  scope. `full` is now available for non-empty positive static dense shapes with
  `float32`, `float16`, `bfloat16`, and `bool` storage, using CPU reference
  execution plus generated CPU/CUDA fill kernels.
  `arange` is available for non-empty static ranges with positive and negative
  steps across `float32`, `float16`, and `bfloat16` storage, using CPU reference
  execution plus generated CPU/CUDA kernels. `randn` is available for non-empty
  positive static dense shapes across `float32`, `float16`, and `bfloat16`
  storage, using an explicit integer `seed` attr and stateless generated
  CPU/CUDA kernels. Dynamic shapes, zero-sized creation outputs, and integer
  arange/randn dtypes remain out of scope for this bounded port. `cast` is
  available for dense tensor casts across the current generated
  float/reduced-precision/bool storage surface.
- [ ] Selection/scatter: remaining bounded gap is `masked_select`.
  `dynamic_slice` is available as a bounded dense materialized copy for one
  static-shape tensor with static integer `start_indices`/`slice_sizes` attrs
  across the generated float/reduced-precision/bool storage surface.
  `index_select` is available as a bounded dense materialized copy for one
  static-shape tensor with a normalized static `dim` and non-empty Python
  sequence of in-bounds non-bool integer `indices`, replacing the selected
  output dimension with `len(indices)` and preserving input dtype across the
  same generated storage surface. `gather` is available for one static-shape
  dense input tensor plus one static-shape `int64`/`int32` index tensor with
  matching rank, normalized `dim`, no broadcasting, output shape equal to the
  index shape, and output dtype equal to the input dtype. CPU reference and
  generated CPU kernels read runtime index storage and fail on out-of-bounds
  gather indices; generated CUDA kernels include a device-side bounds assert.
  `batch_gather(x, indices)` is available for one static-shape dense rank >= 2
  input shaped `[B, N, ...]` plus one static-shape rank-2 `int64`/`int32`
  indices tensor shaped `[B, K]`, producing `[B, K, ...]` with input dtype
  preserved. CPU reference and generated CPU kernels read runtime indices and
  fail on out-of-bounds axis-1 selections; generated CUDA kernels include a
  device-side bounds assert.
  `argmax` is available for one static-shape ranked dense tensor over a
  positive static last dimension after negative `dim` normalization, with
  `keepdim` and scalar fallback shape `[1]`. It supports
  `float32`/`float16`/`bfloat16`/`bool` inputs, compares reduced-precision
  inputs in fp32, returns first max indices on ties, and materializes `int64`
  output tensors through an op-specific compiler/runtime contract exception.
  Public `topk(x, k, dim=-1, largest=True, sorted=True)` is available as two
  internal single-output ops (`topk_values`, `topk_indices`) for one
  static-shape ranked dense tensor over a positive static last dimension only,
  with positive non-bool static integer `k <= last_dim`, `float32`/`float16`/
  `bfloat16`/`bool` inputs, value dtype preserved, `int64` indices, stable
  first-index tie ordering, and sorted descending largest results. Sparse grad,
  dynamic shapes, bool/float gather or batch_gather indices,
  non-last-dimension argmax/topk, smallest/unsorted topk, `masked_select`, and
  true multi-output IR nodes remain out of scope.
  `slice_scatter` is available as the bounded write-side companion with static
  integer `start_indices`, static-shape `x`/`update`, matching rank/dtype, and
  the same generated storage surface. `slice_reshape_scatter` is available as a
  bounded frontend helper that reshapes a static-shape `update` to a positive
  static `slice_shape`, then reuses `slice_scatter`; dynamic shapes and
  view-of-view reshape inputs remain limited by existing shape-view lowering.
  `where` is available for dense bool-condition plus matching
  float/reduced-precision/bool `x`/`y` through fused elementwise CPU/CUDA
  generation.
- [x] Collections/broadcasting: no remaining named v1 collection gaps in this
  bounded subset. `expand` is available as a materialized dense broadcast copy
  for static shapes across the generated float/reduced-precision and bool
  storage surface. `concatenate` is available as a bounded materialized
  dense copy for non-empty static-shape tensor sequences with matching
  rank/non-concat dims, normalized negative `dim`, and the same generated
  float/reduced-precision/bool storage surface. `stack` is available as a
  bounded materialized dense copy for non-empty static-shape tensor sequences
  with exactly matching shapes, normalized insertion `dim`, and that same
  generated storage surface. `flip` is available as a bounded materialized dense
  copy for one static-shape tensor, non-empty normalized `dims` without
  duplicates, and the same generated storage surface. `repeat_interleave` is
  available as a bounded materialized dense copy for one static-shape tensor
  with positive integer scalar `repeats`, required normalized `dim`, and the
  same generated storage surface; per-element repeat tensors remain out of
  scope. `split` and `chunk` are available as bounded frontend helpers for one
  static-shape tensor, normalized negative `dim`, positive integer/section
  sizing, PyTorch-like remainder handling, and multiple public tensor outputs;
  they lower to existing `dynamic_slice` nodes and do not introduce separate
  kernels.
- [x] Relational ops: `eq`, `ge`, `gt`, `le`, `lt`, `ne`.
- [x] Tensor helpers that should not become separate kernel families unless
  profiling proves it: `concatenate_tanh`, `concatenate_fast`,
  `expand_static_shape`. These are resolved as bounded public frontend helpers
  only: `concatenate_fast` delegates to existing `concatenate`,
  `concatenate_tanh` composes existing `concatenate` with elementwise `tanh`
  so local passes emit the concatenate node plus fused elementwise tanh, and
  `expand_static_shape` delegates to existing `expand`. They inherit the
  existing static-shape, dtype, storage, and validation limits and introduce no
  new op/kernel families.

Library hints: most are metadata-only or simple copies. `topk`/sort-like paths
can use CUB on CUDA; CPU paths can start with standard library algorithms and
add xsimd or `std::simd` only where measurable.

### Reductions and Softmax

- [x] Basic reductions: `reduce_max`, `reduce_mean`, `reduce_min`,
  `reduce_sum` for dense contiguous `float32`, `float16`, and `bfloat16`
  tensors over a positive static last dimension, with negative dim
  normalization and `keepdim`. Reduced-precision storage uses fp32 accumulation
  and stores output back to the input dtype. CPU and CUDA use generated row
  reductions and validate against NumPy/Torch-style semantics. CUDA includes a
  warp-per-row path for static reductions up to `K=1024` and a shared-memory
  fallback for larger reductions. Remaining parity work: non-last dimensions,
  multi-axis rejection that mirrors v1 more closely, optional output dtype,
  configurable fp16/bf16 accumulation policy, v1 CUTLASS/`reduce_3d` strategy,
  and profiler selection.
- [x] `var`, `vector_norm`: initial public ports for dense contiguous float32
  tensors over a positive static last dimension, with negative dim
  normalization and `keepdim`. `var` defaults to population variance and
  exposes an `unbiased` flag; `vector_norm` currently supports L2 norm only.
- [x] `softmax`: initial public `dml.ops.softmax(x, dim=-1)` port for dense
  contiguous `float32`, `float16`, and `bfloat16` tensors on CPU and CUDA.
  Current implementation supports only the last dimension with a positive static
  reduction extent, uses stable max-subtract/exp/sum normalization, and targets
  attention-row shapes such as `[batch_heads * queries, keys]`. Reduced-precision
  storage uses fp32 computation/accumulation and stores output back to the input
  dtype. CUDA now has a warp-per-row register-cached specialization for odd/tail
  `K <= 2048`, a float2/float4 packed local-register path for divisible float32
  reductions up to the initial v1-style thresholds, and a shared-memory fallback
  for larger reductions. Reduced-precision CUDA kernels conservatively avoid the
  packed float vector reinterpret path. Non-last dimensions, generic dynamic
  reduction extents, strided/layout-aware tensors, full v1 K1/K2/K4/K8
  small/middle/block policy parity, and profiler-selected variants remain
  unported.

Library hints: CUB is a good CUDA baseline for generic reductions and scans;
oneDNN has CPU softmax/reduction coverage; CK/MIOpen may cover selected GPU
normalization or softmax patterns, otherwise use custom block reductions.

### GEMM, BMM, and Fused Linear Families

- [x] Base GEMM layouts: `gemm_rcr`, `gemm_rrr` are explicit CUDA ops for
  `float32`, `float16`, and `bfloat16`, backed by cached CUTLASS launchers with
  explicit tensor-op manifest candidate sets and CPU reference
  execution but no CPU compiled GEMM. Float32 candidate parity includes 221
  default candidates: optional v1 SM80 regular TF32 TensorOp candidates,
  optional fast TensorOp families for `multiply_add_fast_f16`,
  `multiply_add_fast_bf16`, and 3xTF32 `multiply_add_fast_f32`, plus the exact
  f32 SIMT fallback set. The manifest carries target policy for optional TF32
  and fp16 accumulation; fp16 accumulation and TF32 opt-out now select
  policy-specific CUTLASS candidate sets, and rendered policy aliases apply the
  selected candidate alignment and math operator. Residual broadcast epilogues
  now select a TensorOp or SIMT CUTLASS broadcast epilogue path to keep the
  exact-f32 no-TF32 fallback available. Public `matmul` should wait
  until layout selection, multi-candidate profiler selection, and epilogue
  contracts are ready.
- [x] First bias epilogues: `gemm_rcr_bias`, `gemm_rrr_bias` support rank-1
  `[N]` and rank-2 `[1, N]` bias contracts through CUTLASS launcher/profiler
  symbols for `float32`, `float16`, and `bfloat16`, with CPU reference coverage.
- [x] First activation epilogue: `gemm_rcr_bias_relu` and
  `gemm_rrr_bias_relu` use CUTLASS `LinearCombinationRelu` with the same bias
  shape/dtype/runtime/profiler contracts as the bias-only GEMM ops.
- [x] v1-style bias activation epilogue names:
  `gemm_{rcr,rrr}_bias_{gelu,fast_gelu,sigmoid,tanh,swish,hardswish,elup1}`
  are registered as explicit GEMM family ops with CUTLASS candidate metadata,
  CUDA support-library symbols, candidate profiling coverage, and CPU reference
  execution through CUTLASS thread epilogue functors.
- [x] First residual epilogues:
  `gemm_{rcr,rrr}_bias_{add,add_add,mul,mul_add,add_relu,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}`
  support rank-2 residual tensors through fused CUTLASS epilogues, CUDA
  lowering/profiler pointer ABIs, and CPU reference execution. These do not use
  a post-GEMM activation or elementwise launch.
- [x] First folded-M residual coverage:
  `gemm_{rcr,rrr}_bias_{add,add_relu,mul,add_add,mul_add,add_add_relu,mul_tanh,sigmoid_mul,sigmoid_mul_tanh}`
  accept `A[..., K]`, preserve output/residual shape `[..., N]`, and flatten
  leading `A` dimensions into the CUTLASS `m` argument for CUDA lowering and
  profiling.
- [x] Base BMM layouts:
  `bmm_{ccc,ccr,crc,crr,rcc,rcr,rrc,rrr}` now have frontend contracts, CPU
  reference execution, and a separate `cutlass_bmm` CUDA support-library path
  using CUTLASS batched GEMM candidates. The launch ABI preserves v1 layout
  semantics, C-column output `[B, N, M]`, batch broadcasting through zero batch
  strides, target-policy candidate filtering, runtime alignment fallbacks, and
  profile/report/cache workloads keyed by batch-aware BMM problem shapes. Static
  BMM profile selections are consumed during compile, and conflicting BMM
  profile selections now emit guarded batch/M/N/K dispatch with default fallback.
- [x] First BMM add epilogue:
  `bmm_{ccc,ccr,crc,crr,rcc,rcr,rrc,rrr}_add` now registers CUTLASS candidate
  sets and a `dinoml_cutlass_bmm_add_v1` launcher/profiler ABI for full-output
  `d0` tensors and v1-style trailing-bias `d0` tensors. The support source uses
  CUTLASS source-C/beta epilogue semantics instead of a post-BMM elementwise
  kernel, uses zero source-C stride/leading-dimension for trailing bias, and
  profiling includes `d0` in epilogue alignment metadata. Remaining BMM work:
  split-K/grouped extensions and broader non-trailing broadcast forms.
- [x] First profile-selected execution-plan artifact:
  `dinoml profile` now writes `debug/execution_plan.json`, selecting the fastest
  measured candidate per profiled node/shape and emitting a static overlay only
  when all profiled shapes for an op/dtype/candidate-set agree on the same
  winner.
- [x] Static execution-plan consumption:
  `dml.compile(..., execution_plan=...)` and
  `dinoml compile --execution-plan` apply matching static selections before
  writing kernel manifests or generated CUDA, so lowering uses the profiled
  candidate instead of the manifest seed candidate when no shape conflict exists.
- [x] First profile-assisted compile loop:
  `dml.compile(..., profile=True)` and `dinoml compile --profile` build a
  candidate CUDA artifact, run the existing CUTLASS artifact profiler, then
  rebuild with the generated execution plan applied. The bootstrap timing report
  is retained as `debug/bootstrap_profile_report.json`.
- [x] Pass-once profile-assisted compile:
  profile-assisted compile now runs graph passes and writes constants once, then
  materializes the candidate and final artifacts from the same lowered IR while
  refreshing only generated CUDA sources for the selected execution plan.
- [x] First dynamic-shape profiling buckets:
  GEMM profiling expands explicit `Dim.buckets` into concrete workload cases
  when no runtime override is supplied, preserves shared named dim values across
  inputs, rejects conflicting bucket metadata for the same dim name, and carries
  bucket case metadata into execution-plan selections/conflicts.
- [x] First guarded dynamic-shape dispatch:
  execution-plan shape conflicts can attach per-node guarded CUTLASS selections
  to the kernel manifest, generated CUDA branches on profiled `M/N/K` cases,
  supports split-K dispatch workspace sizing, and falls back to the safe manifest
  default when no guard matches.
- [x] First static alignment-aware profiling filter:
  when dense layout element alignment is present on both GEMM A and B, profiling
  prunes CUTLASS candidates whose A/B policy alignment exceeds the smaller
  operand alignment. Bias/residual/C alignment is intentionally ignored until
  separate epilogue/source alignment requirements exist.
- [x] v1-style shape-derived A/B alignment filtering:
  manifest defaults use the all-runtime shape contract (`K` for RCR,
  `gcd(K, N)` for RRR, with dynamic dims capped by `divisible_by`), and
  profiling workloads use each concrete bucket/override/max shape before timing
  candidates. Execution-plan overlays now validate selected candidate alignment
  against the manifest cap before replacing the safe default.
- [x] First CUTLASS runtime alignment guard:
  generated CUDA checks selected-candidate A/B pointer byte alignment before
  launching vectorized CUTLASS GEMMs. The shared module support path already
  validates offset-adjusted byte capacity and non-contiguous row-major strides
  when stride metadata is supplied.
- [x] First ABI byte-offset support:
  generated CPU/CUDA modules apply `DinoTensor.byte_offset` to logical tensor
  pointers for inputs, outputs, runtime constants, and public alias
  materialization; CUTLASS pointer-alignment checks now see the logical pointer.
- [x] First tensor-accessor alignment selection:
  manifest/profile filtering now records a CUTLASS alignment context that folds
  shape-derived caps, partial A/B dense layout alignment, known tensor/layout
  storage offsets, and current C/epilogue alignment metadata into candidate
  filters. Generated CUDA falls back through lower-alignment CUTLASS candidates
  when runtime logical A/B pointers do not satisfy the selected candidate.
- [ ] Full non-dense tensor-accessor GEMM:
  extend the CUTLASS launch ABI beyond dense leading dimensions before treating
  arbitrary non-row-major strides as correct GEMM inputs.
- [x] First split-K profile metadata surface:
  CUTLASS GEMM candidates and candidate sets advertise `split_k_values: [1]`,
  profile reports/cache keys/execution plans preserve `split_k` and
  `workspace_nbytes`, static overlays require agreement on candidate and
  `split_k`.
- [x] Split-K launcher/profiler ABI for base and bias/activation CUTLASS GEMMs:
  companion split-K symbols preserve the old v1 ABI for `split_k=1`, profiler
  workloads expand v1-style split-K values, workspace queries feed profile
  results/execution plans, and generated modules allocate one session workspace
  when a static overlay selects `split_k > 1`.
- [x] First profiler repeat statistics:
  `dinoml profile --repeats` and `dml.compile(..., profile=True,
  profile_repeats=...)` collect multiple timing samples per CUTLASS workload,
  record median/mean/min/max/stddev plus relative stddev in profile reports and
  cache entries, and use the median elapsed time for execution-plan selection.
- [x] First confidence-gated profiler selection:
  execution plans now require repeat-count, absolute/relative margin, and
  confidence-interval thresholds before emitting consumable static or guarded
  candidate selections; close/noisy winners stay in `low_confidence_selections`
  for audit and fall back to manifest defaults at compile/run time.
- [x] Additive residual split-K coverage:
  `bias_add`, `bias_add_add`, `bias_add_relu`, and `bias_add_add_relu` CUTLASS
  residual epilogues now advertise v1-style split-K search metadata, use
  partition-aware serial split-K epilogues, and lower/profile through companion
  split-K symbols.
- [ ] Extend split-K coverage to non-additive residual/broadcast CUTLASS
  epilogues after their `GemmUniversalWithBroadcast` workspace behavior is
  proven and their fused epilogues implement correct partition behavior.
- [x] Base BMM frontend/CPU contracts:
  `bmm_{ccc,ccr,crc,crr,rcc,rcr,rrc,rrr}` and matching `_add` variants now
  exist as explicit frontend ops with v1-compatible A/B/C layout shape
  semantics, batch broadcasting, dynamic shape metadata, v1-style trailing-bias
  addend validation, and CPU reference execution.
- [x] CUTLASS BMM launch/profiling family:
  base BMM and `_add` BMM use real batched CUTLASS ABIs with batch strides, C
  row/column output layout, candidate metadata, profiling workloads, and
  execution-plan selections.
- [ ] Remaining bias/broadcast epilogues: broader broadcast forms beyond
  full-output and v1-style trailing-bias residual tensors.
- [x] Remaining activation epilogue: `elup1` is available as
  `gemm_{rcr,rrr}_bias_elup1` through the existing CUTLASS bias-activation
  ABI. The v1 `gemm_rcr_permute_elup1` layout-fused form remains under
  permuted/layout-fused output families.
- [ ] Beyond-v1 CUTLASS epilogues: once v1 parity is stable, evaluate additional
  CUTLASS epilogue functors and visitor forms that can remove useful post-GEMM
  elementwise launches.
- [ ] Permuted/layout-fused output families: `gemm_*_permute*`,
  `bmm_*_permute`, `perm021fc_*`, `perm102_bmm_*`.
- [ ] Grouped GEMM: `group_gemm_rcr*`.
- [ ] Softmax/attention matmul chains: `bmm_softmax_bmm*`,
  `bmm_rcr_softmax`, `gemm_rcr*_softmax`, `dual_bmm_rrr_div`.
- [ ] v1 dual-GEMM and dual-output GEMM epilogue families:
  `dual_gemm_rcr_*`.
- [ ] Specialized small/degenerate kernels: `gemm_rrr_small_nk`,
  `bmm_rcr_n1`, `bmm_rrr_k1_tanh`, `batched_dense_vec_jagged_2d_mul`.
- [ ] Back-to-back BMM: `classic_b2b_bmm`, `fmha_style_b2b_bmm`,
  `grouped_classic_b2b_bmm`, `grouped_fmha_style_b2b_bmm`.
- [x] Direct-import helpers: `bmm`, `bmm_xxx`, `bmm_xxx_add`.
- [x] First encoded constant source scaffold:
  constant values can now materialize through a source object before
  `constants.bin` is written, and `gguf_constant(...)` records GGUF provenance
  plus dense logical dtype/materialization policy while preserving the existing
  dense runtime ABI.
- [x] First runtime encoded constant materializer:
  `RuntimeModule.load_encoded_constants()` rehydrates GGUF storage metadata,
  dequantizes through the source layer, and updates constants through the
  existing dense `set_constant_numpy` path. This is the initial
  dequant-then-kernel hook before CUDA/offload-specific materialization, with
  optional real-libgguf fixture coverage for quantized row materialization.
- [x] Encoded constant runtime load planning:
  `RuntimeModule.encoded_constant_load_plan()` reports encoded constant names,
  logical dtype/shape/size, storage provenance, policy support status, and
  whether the current runtime can load each constant now.
  `RuntimeModule.load_encoded_constants(names=...)` can selectively rehydrate
  supported encoded constants and rejects declared future policies before trying
  to materialize storage.
- [x] Manual runtime GGUF encoded constant loading:
  GGUF constants declared with `residency="manual_runtime_load"` are now
  runtime-supported for the existing dense
  `dequantize_full_before_launch` materialization path. This still rehydrates
  through the dense setter path rather than adding CPU/GPU prefetch, eviction,
  CUDA dequantization, or direct in-kernel quantized RHS execution.
- [ ] Future weight-loading/offload path: CPU-resident constants that can move
  to GPU at run time, later expanding to sequential, grouped/block/layer, and
  multi-stream offload policies. GGUF support should evaluate `hlky/libgguf`
  CUDA quantize/dequantize kernels for load-time full dequantization and
  kernel-local direct dequantization strategies. First GGUF support should model
  GGUF as quantized constant storage with a dense logical dtype, copy packed
  bytes, fully dequantize into a dense weight buffer before GEMM, and leave
  fused quantized-RHS CUTLASS candidate families for a later step. The current
  foundation includes artifact-level eager/deferred constant-load policy plus
  runtime reload/unload primitives, encoded-constant load planning, and
  selective dense-path rehydration, including manual runtime loading of GGUF
  encoded constants; remaining work is policy execution for selective CPU/GPU
  residency, prefetch, eviction, and CUDA/GGUF dequantization.

Library hints: CUTLASS is the primary CUDA candidate for GEMM/BMM, grouped GEMM,
and epilogue visitors. CK is the corresponding AMD path. oneDNN matmul/brgemm is
the CPU fallback target. Plain `torch.matmul`, `torch.bmm`, and `torch.addmm`
are good semantic references, but not replacements for v1 fused layout/epilogue
behavior.

### Convolution, Pooling, Padding, and Upsampling

- [ ] Convolution: `conv2d`, `conv3d`, `conv3d_bias`, `depthwise_conv3d`,
  `depthwise_conv3d_bias`, `transposed_conv2d`.
- [ ] Pooling: `avg_pool1d_compress_time`.
- [x] `avg_pool1d`: bounded public `dml.ops.avg_pool1d(x, kernel_size,
  stride=None, padding=0)` for rank-3 NCL static-shape `float32`, `float16`,
  and `bfloat16` tensors. CPU reference and generated CPU/CUDA kernels use
  fp32 accumulation and store back to the input dtype. Semantics are fixed to
  PyTorch floor output shape with zero padding included in the `kernel_size`
  divisor; `ceil_mode`, `count_include_pad=False`, `divisor_override`, dynamic
  shapes, bool/integer tensors, and `avg_pool1d_compress_time` remain out of
  scope.
- [x] `avg_pool2d`: bounded public `dml.ops.avg_pool2d(x, kernel_size,
  stride=None, padding=0)` for rank-4 NCHW static-shape `float32`,
  `float16`, and `bfloat16` tensors. CPU reference and generated CPU/CUDA
  kernels use fp32 accumulation and store back to the input dtype. Semantics are
  fixed to PyTorch floor output shape with zero padding included in the
  `kernel_h * kernel_w` divisor; `ceil_mode`, `count_include_pad=False`,
  `divisor_override`, dynamic shapes, and bool/integer tensors remain out of
  scope.
- [x] `max_pool2d`: bounded public `dml.ops.max_pool2d(x, kernel_size,
  stride=None, padding=0)` for rank-4 NCHW static-shape `float32`,
  `float16`, and `bfloat16` tensors. CPU reference and generated CPU/CUDA
  kernels compare in fp32 and store back to the input dtype. Semantics are
  fixed to PyTorch floor output shape with implicit negative-infinity padding;
  dynamic shapes, bool/integer tensors, dilation, `ceil_mode`, and
  `return_indices` remain out of scope.
- [x] `pad`, `pad_last_dim`: initial bounded static constant-padding port for
  ranked static tensors. `pad` uses PyTorch/F.pad trailing-pair order and accepts
  non-empty even-length non-negative non-bool integer pad widths, with output
  shape statically expanded and dtype preserved across float32/float16/bfloat16/
  bool storage. CPU reference plus generated CPU/CUDA copy/fill kernels are in
  place. Out of scope for this port: dynamic shapes, non-constant modes,
  negative pads/cropping, integer dtypes, and layout packing helpers.
- [ ] Padding/layout packing: `nhwc3to4`, `nhwc3to8`, `ndhwc3to8`,
  `prepare_for_transposed_conv2d`.
- [ ] Upsampling: `upsampling{1d,2d,3d}`, `_add` variants, and
  `upsampling3d_compress_time`.

Library hints: cuDNN/MIOpen should cover most conv and pooling paths; oneDNN is
the CPU target. Packing helpers are likely custom copy kernels. Use
`torch.nn.functional` conv/pool/pad/interpolate as reference behavior.

### Normalization

- [ ] GroupNorm: `group_norm`, `group_norm_swish`.
- [ ] LayerNorm family: `layernorm`, `t5_layer_norm`, `group_layernorm`,
  `batch_layernorm_sigmoid_mul`, `layernorm_sigmoid_mul`,
  `group_layernorm_sigmoid_mul`.

Library hints: cuDNN/MIOpen/oneDNN can cover common norm shapes, but fused
sigmoid/mul/swish forms may need custom reductions or CUTLASS/CK epilogues when
paired with GEMM.

### Attention

- [ ] `flash_attention`.
- [ ] `flash_attn`.
- [ ] `mem_eff_attention`.

Library hints: `torch.nn.functional.scaled_dot_product_attention` is a semantic
reference. Validate causal masks, dropout, head layout, kv-cache, and variable
length behavior before deciding whether to wrap FlashAttention-style kernels,
CUTLASS/CK kernels, or compose GEMM/softmax/GEMM families.

### Jagged and Ragged Tensors

- [ ] `make_jagged`.
- [ ] `jagged_lengths_to_offsets`.
- [ ] `jagged_lengths_to_presences`.
- [ ] `jagged_to_padded_dense`.
- [ ] `padded_dense_to_jagged`.

Library hints: CUB scans are useful for lengths-to-offsets on CUDA. Most dense
to/from jagged transforms need custom kernels and careful shape metadata tests.
Use PyTorch only as partial reference; core `torch` does not directly match v1
JaggedIntVar semantics.

## Custom and Model-Fused Helper Ops

These are not common primitives. Port them after their underlying tensor,
elementwise, layout, and reduction pieces exist, unless a model requires one
early. Prefer implementing them as small graph rewrites over dedicated kernels
when compile-time constants make that practical.

### Embedding and Positional Helpers

- [ ] Embedding/model helpers: `bert_embeddings`, `relative_attention_bias`,
  `sinusoidal_positional_embedding`, `gaussian_fourier_projection`,
  `gelu_new`, `cropped_pos_embed`, `get_timestep_embedding`.
- [ ] Rotary/sincos helpers: `get_1d_rotary_pos_embed`,
  `get_2d_rotary_pos_embed`, `get_2d_rotary_pos_embed_lumina`,
  `get_2d_sincos_pos_embed`, `get_2d_sincos_pos_embed_cogview3plus`,
  `get_3d_rotary_pos_embed`, `get_3d_rotary_pos_embed_allegro`,
  `get_3d_sincos_pos_embed`, `get_3d_sincos_pos_embed_cogvideox`,
  `get_fourier_embeds_from_boundingbox`.

Library hints: no major external kernel library is expected to own these. Use
common primitive composition for CPU/CUDA first; add fused kernels only if these
become profiler-visible.

### Filtering and Resampling Helpers

- [ ] FIR/filter helpers: `fir_downsample2d`, `fir_filter_pad2`,
  `fir_upsample2d`.
- [ ] Kernel weight builders: `kdownsample2d_weight`, `kupsample2d_weight`.

Library hints: compose from pad/conv/upsample where possible. cuDNN/MIOpen may
help if represented as convolution; otherwise these are small custom kernels.

### Vision Detection Helpers

- [ ] NMS: `nms`, `batched_nms`, `efficient_nms`.
- [ ] ROI: `roi_align`, `multi_level_roi_align`.

Library hints: these are not in core `torch`, `torch.nn`, or
`torch.nn.functional`. TorchVision can be a reference if available. CUDA NMS may
use CUB for sorting/selection support, but box suppression and ROI sampling are
custom kernels.

## Suggested Porting Order

1. Harden elementwise parity: vectorized generated kernels, jagged/accessor
   support, dtype coverage, scalar promotion, and exhaustive numerical tests.
2. Add view/layout/selection/reduction primitives needed by model builders:
   reshape, permute, concatenate, split, slice, gather, topk, softmax.
3. Build the GEMM/BMM backbone once: base layouts, bias, activation epilogues,
   permuted outputs, grouped variants, and profiler/cache integration.
4. Add the weight-loading/offload foundation before large model artifacts
   depend on a fixed constants lifecycle: CPU-starting weights, optional GPU
   prefetch/copy policies, and an integration point for GGUF quantized storage.
5. Port normalization, convolution, pooling, padding, and upsampling with
   library-backed paths where available.
6. Add attention and jagged/ragged support, because they combine multiple
   primitive families and shape rules.
7. Finish model-fused helpers, FIR resampling, NMS, and ROI after the reusable
   primitives are stable or when a target model makes one urgent.
