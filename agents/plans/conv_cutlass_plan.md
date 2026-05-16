# ConvNd CUTLASS Provider Plan

This plan captures the recent ConvNd exploration so future implementation can
start from settled constraints instead of re-learning v1 behavior or drifting
into hidden layout state.

The first implementation target is intentionally narrow: a CUDA-only
`conv2d_bias` provider-backed slice with artifact-visible internal layout
translation.

## Current implemented status

The original base slice is now real, and one bounded fused epilogue has been
added on top of it:

- `conv2d_bias` is admitted on the static rank-4, `groups=1`, public
  NCHW/OIHW contract with CPU reference execution, generated naive CPU
  artifacts for `float32`/`float16`, and CUDA `cutlass_conv` manifest/profile/
  execution-plan/generated-lowering support.
- Public no-bias `conv2d` exists only as an explicit-zero bridge over that same
  core path. It is not a distinct provider family or runtime ABI, but focused
  runtime parity now proves that bridge on the admitted fp16 TensorOp
  FewChannels `C=3`, FixedChannels `C=4`/`C=8`, and optimized aligned `C=16`
  lanes while preserving `source_op=conv2d` and
  `bias_mode=explicit_zero_constant`.
- `conv2d_bias_relu` is now the first fused Conv epilogue slice. It reuses the
  same public contract, profiling flow, and layout-transform metadata as
  `conv2d_bias`, but records fused `bias_relu` epilogue state through candidate
  sets, manifests, profile workloads, execution plans, support-cache/source
  manifests, and generated lowering.
- `conv2d_bias_add` is now admitted as the first residual Conv slice. It keeps
  the same static rank-4, `groups=1`, NCHW/OIHW public contract, adds a
  same-shape residual tensor, and records explicit residual epilogue state
  through candidate sets, manifests, profile workloads, execution plans,
  support-cache/source manifests, generated lowering wrapper stages, and the
  launch ABI `dinoml_cutlass_conv2d_bias_add_v1`. Because CUTLASS consumes the
  residual in provider NHWC layout, the wrapper now materializes an explicit
  `residual_pack` temporary alongside activation pack, weight pack, provider
  launch, and output unpack stages.
- Current CUDA runtime coverage for both `conv2d_bias` and `conv2d_bias_relu`
  is still deliberately narrow: fp16 SIMT, fp16 TensorOp few-channels
  (`C=3`), fp16 TensorOp fixed-channels (`C=4`/`C=8`), fp16 TensorOp optimized
  (`C >= 16` with aligned channels), and float32 SIMT only. Broader float32
  TensorOp, bfloat16, sigmoid, richer residual/add+activation epilogues, and
  grouped/depthwise/transposed/3D Conv are not landed. Focused runtime parity
  currently proves
  base `conv2d_bias` across the admitted fp16 TensorOp/SIMT lanes plus float32
  SIMT, proves the bridged public no-bias `conv2d` path on the fp16
  FewChannels `C=3`, FixedChannels `C=4`/`C=8`, and optimized aligned `C=16`
  TensorOp lanes, and proves fused `conv2d_bias_relu` on float32 SIMT plus the
  same fp16 TensorOp lane family. The new `conv2d_bias_add` slice deliberately
  stays inside that same bounded candidate family and now has focused
  compile/runtime proof on float32 SIMT general-shape parity, fp16 TensorOp
  FewChannels `C=3`, fp16 TensorOp FixedChannels `C=4`, and real support-
  library compile coverage for the fp16 residual path. Residual TensorOp
  runtime proof for `C=8`, optimized aligned `C=16`, bfloat16, and broader
  float32 TensorOp shapes remains open.
- A bounded `conv2d_bias_sigmoid` follow-up was explored and intentionally not
  landed. CUTLASS does ship
  `cutlass/epilogue/thread/linear_combination_sigmoid.h`, but the current
  `cutlass::conv::device::ImplicitGemmConvolution` launcher wiring used for the
  admitted Conv bias/bias_relu paths did not accept the same source-C/bias
  argument construction once `LinearCombinationSigmoid` replaced the epilogue.
  Real `nvcc` support-library compilation failed before runtime parity could
  start, so no public sigmoid Conv op should exist until a provider/runtime
  design compiles and passes a real CUDA parity test.

## Why this needs its own plan

ConvNd is the first likely provider family where v1 behavior, library
constraints, and v2 artifact rules pull in different directions:

- v1 convolution execution was GPU-first and channels-last internally.
- Diffusers and most PyTorch-facing model code are semantically
  NCHW/NCDHW-first.
- v2 runtime ABI still assumes contiguous row-major tensors and should not be
  treated as a generic strided layout ABI.
- v2 provider policy requires profiling, execution plans, and generated code to
  make provider decisions visible in artifacts.

That combination makes ConvNd easy to get wrong with an apparently small port.

## Settled lessons from v1

### Layout and caller behavior

- The v1 compiler conv stack used NHWC/NDHWC activations and OHWI/ODHWI weights
  inside compiled conv ops.
- PyTorch-facing tests and callers commonly permuted NCHW/NCDHW tensors into
  that provider-facing layout by hand.
- v1 PyTorch weight import permuted `OIHW -> OHWI` and `OIDHW -> ODHWI`.
- Some small-channel 2D cases padded channels to 4 or 8 for provider/kernel
  requirements.

### Backend coverage

- v1 CUDA `conv2d`, `transposed_conv2d`, and `conv3d` were CUTLASS
  implicit-GEMM style codegen/profile/cache flows.
- v1 ROCm 2D convolution used CK.
- v1 `depthwise_conv3d` used a custom CUDA path.
- No native CPU convolution backend was found in the explored v1 stack.

### What not to copy forward blindly

- Hidden caller-visible layout mutation is not acceptable in v2.
- Hidden auto-padding of channels should not come back as implicit provider
  behavior.
- The v1 internal NHWC choice is a useful provider implementation hint, not a
  public semantic contract for v2.

## v2 architectural stance

### Public semantic layout stays source-faithful

For Diffusers and other PyTorch-derived graphs, initial semantic translation
should stay faithful to source NCHW/NCDHW tensor meaning. Public ConvNd ops,
shape inference, tests, and reference execution should describe PyTorch-style:

- activations: NCHW for 2D, NCDHW for 3D
- weights: OIHW for 2D, OIDHW for 3D
- attrs: stride, padding, dilation, groups in source axis semantics

Treat NHWC/NDHWC as a guarded provider-internal optimization island, not as a
public API toggle and not as a hidden global layout pass.

### Runtime ABI remains row-major contiguous

Current v2 runtime ABI assumes contiguous row-major storage. Do not rely on ABI
stride fields to smuggle NHWC tensors through a nominally NCHW public contract.

If a provider uses NHWC/NDHWC internally, generated wrappers must materialize
explicit pack/unpack temporaries around the provider call and those transforms
must be recorded in artifact metadata.

### Provider transforms must be artifact-visible

Any internal layout or weight transform needed by the provider should be
recorded in manifests and generated lowering metadata, for example:

- semantic layout
- provider layout
- activation layout translation
- weight transform key
- optional channel-padding transform key and padded extents
- temporary/workspace sizes

Do not hide these as backend-specific auto-rewrites with no artifact trace.

## Recommended provider family shape

Add a new provider family such as `cutlass_conv` rather than forcing ConvNd
through the GEMM provider surface. The flow should mirror the mature CUTLASS
GEMM/BMM contract:

1. frontend ConvNd op/family with bounded attrs
2. provider candidate set generation
3. support-library/source-manifest cache
4. profiler workload generation and profile report
5. execution-plan selection
6. generated lowering that visibly consumes the selected provider candidate

This should feel parallel to `cutlass_gemm`, not like a special-case side path.

## Required artifact metadata

The first `cutlass_conv` manifest/profile/execution-plan schema should carry at
least:

- provider family name, version, and target/backend
- op family (`conv2d`, `conv2d_bias`, later `conv3d`, `transposed_conv2d`, ...)
- semantic activation layout (`nchw`, `ncdhw`)
- semantic weight layout (`oihw`, `oidhw`)
- provider activation layout (`nhwc`, `ndhwc`, or `nchw` if a candidate uses
  source-faithful layout)
- provider weight layout (`ohwi`, `odhwi`, or source-faithful layout)
- `layout_translation` metadata describing explicit input/output pack or unpack
  requirements
- `weight_transform` metadata describing source layout permutation and any
  provider-required channel padding
- helper symbols or equivalent support-library export metadata for each
  recorded activation pack, weight pack, and output unpack transform
- dtype and accumulator dtype
- stride, padding, dilation, groups
- candidate set key and per-candidate metadata
- selected candidate id
- workspace size and temporary-pack sizes
- support-library provenance and source-manifest key

For channel padding, prefer explicit keys such as:

- `channel_pad_multiple`
- `padded_input_channels`
- `padded_output_channels`
- `padding_fill_value`

That keeps any future small-channel packing rule inspectable and testable.

## Support library and cache expectations

The provider should eventually own a support library such as
`libdinoml_cutlass_conv.so` with launcher/profiler exports per candidate family.
Like CUTLASS GEMM/BMM, cache reuse should depend on:

- target arch/toolchain
- provider version
- candidate/config keys
- support source/binary hashes
- source-manifest payload consistency
- transform schema keys that affect generated wrappers

The source manifest should map rendered source/build units to the used ConvNd
candidate plan so stale embedded payloads are rejected instead of silently
reused.

The current scaffold status is narrower but should still stay explicit:
support-source generation and `nvcc` builds may compile reusable CUDA layout
transform helpers for the bounded NCHW -> NHWC, OIHW -> OHWI, and NHWC -> NCHW
contract before the real CUTLASS launcher exists. Those helpers should remain
artifact-visible through `cutlass_conv_plan` symbol metadata plus support
manifest/source-manifest export records rather than becoming hidden runtime
helpers.

## Execution-plan expectations

ConvNd should not bypass execution plans once profiling exists. Profile reports
and execution plans should be able to answer:

- which candidate set was considered
- which candidate won for which shape/config bucket
- whether layout packing or workspace requirements changed between candidates
- whether a guarded selection is needed for profiled shape buckets

The generated wrapper should consume that selected candidate visibly, the same
way CUTLASS GEMM generated code does today.

## First bounded implementation slice

Start with exactly one honest slice:

- op: `conv2d_bias`
- backend: CUDA only
- provider family: `cutlass_conv`
- public semantics: NCHW activation, OIHW weight, PyTorch-compatible bias
- internal layout: artifact-visible NHWC/OHWI translation if CUTLASS requires
  it
- groups: `1` only
- ranks: static rank-4 only
- attrs: static channels/kernel/stride/pad/dilation
- shapes: static channel and kernel attrs; dynamic H/W profiling is deferred
- dtype: `float16` first
- optional dtype only if clean: `float32`
- validation: CPU/PyTorch reference execution plus CUDA provider comparison

Why this slice:

- It exercises the layout and weight-transform design without opening grouped,
  depthwise, transposed, or 3D complexity.
- It matches the current CUDA-first provider maturity in the repo.
- It keeps the public semantic contract faithful to the source model while
  still allowing an internal NHWC CUTLASS island.

## Admission and validation criteria for that slice

Do not treat `conv2d_bias` as landed until it has all of:

- frontend contract with explicit bounded attrs
- shape/type inference for NCHW tensors
- CPU/PyTorch reference path
- provider manifest metadata for layout and weight transforms
- support-library cache/provenance path
- profiling workload generation and report schema, even if candidate count is
  initially small
- execution-plan consumption in generated lowering
- generated wrapper tests proving explicit pack/unpack behavior when provider
  layout differs from semantic layout
- weight import tests proving source `OIHW` to provider layout transforms are
  explicit artifact-visible metadata
- regression coverage that hidden channel padding cannot occur without manifest
  metadata recording it

## Initial test matrix

Keep the first validation grid small and boring:

- PyTorch reference parity for static NCHW `conv2d_bias`
- CUDA provider parity for representative kernel/stride/pad cases
- source weight import parity for `OIHW -> provider-layout`
- artifact inspection tests for:
  - semantic vs provider layout fields
  - `layout_translation`
  - `weight_transform`
  - candidate set / selected candidate / workspace metadata
- negative admission tests for:
  - `groups != 1`
  - unsupported dtype
  - dynamic H/W profiling attempts
  - missing provider transform metadata
  - any attempt to expose public NHWC semantics for this first slice

## Explicit deferrals

Do not broaden the first ConvNd landing to include:

- `conv3d`
- `depthwise_conv3d`
- `transposed_conv2d`
- grouped or depthwise 2D convolution
- fused epilogue expansion beyond the admitted `conv2d_bias_relu` slice
- dynamic H/W profiling or guarded runtime shape dispatch
- global layout pass work
- runtime-set transformed weights or hidden persistent packed-weight state
- public NHWC/NDHWC toggles
- ABI-stride-based NHWC execution
- hidden small-channel auto-padding
- ROCm/CK parity
- CPU compiled conv backend

## First fused epilogue extension

With the base slice admitted, the next bounded extension is exactly one fused
epilogue that can satisfy the same artifact-visible contract. The first landed
one is:

- op: `conv2d_bias_relu`
- public semantics: unchanged from `conv2d_bias` (NCHW activation, OIHW weight,
  rank-1 bias, static rank-4, groups=`1`)
- manifest/profile/runtime identity: same provider family and candidate shapes
  as `conv2d_bias`, but with explicit `epilogue=bias_relu`,
  `epilogue_config={"inputs":["bias"],"activation":"relu"}`, and launch ABI
  `dinoml_cutlass_conv2d_bias_relu_v1`
- CPU path: reference execution plus generated naive CPU artifact with fused
  ReLU clamp
- CUDA runtime: only where the corresponding base candidate already exists
  today, namely fp16 SIMT/TensorOp few-channels/fixed-channels/optimized and
  float32 SIMT

Do not treat this as general Conv epilogue parity. It is one admitted fused
activation slice chosen because it fits the existing bias-family launch ABI and
selection flow cleanly.

These all deserve separate admission and design updates after the first bounded
slice is proven.

## First residual epilogue extension

The next bounded extension after `conv2d_bias_relu` is now also admitted:

- op: `conv2d_bias_add`
- public semantics: unchanged from `conv2d_bias` plus one residual tensor that
  must be static rank-4, same dtype, and exactly match the Conv output shape
- manifest/profile/runtime identity: same provider family and candidate shapes
  as `conv2d_bias`, but with explicit `epilogue=bias_add`,
  `epilogue_config={"inputs":["bias","d0"]}`, `residual_shape=[N,C,H,W]`, and
  launch ABI `dinoml_cutlass_conv2d_bias_add_v1`
- CPU path: reference execution plus generated naive CPU artifact with explicit
  residual add before the final store
- CUDA/provider path: explicit residual NCHW -> NHWC temporary pack recorded in
  `layout_translation`, `temporary_buffers`, wrapper stages, support-cache
  manifests, source manifests, and generated lowering
- validation proof: targeted non-CUDA parity, profiling metadata coverage, real
  support-library compile coverage, and focused float32 SIMT runtime parity

Residual risks remain intentionally narrow: only the already admitted fp16
SIMT/few-channels/fixed-channels/optimized and float32 SIMT candidate family is
covered, grouped/depthwise/transposed/3D Conv remain out of scope, and no
broader residual epilogues (add+relu, add+activation chains, multiple residual
inputs, or sigmoid) should be inferred from this slice.

## Design traps to avoid

- Do not model ConvNd as a public NHWC op just because CUTLASS likes NHWC.
- Do not rely on the current ABI stride fields as a substitute for explicit
  layout translation.
- Do not persist transformed weights only in opaque runtime state with no
  manifest record.
- Do not add a global NCHW -> NHWC rewrite before local ConvNd layout islands
  are proven.
- Do not infer that v1 channel padding should always happen; if needed, record
  it explicitly as a provider transform.
- Do not over-generalize from `conv2d_bias` into automatic `conv3d`,
  transposed, or depthwise coverage.

## Follow-up slices after the first landing

Only after `conv2d_bias` is real and boring should follow-up work consider:

1. A true no-bias provider/runtime family for `conv2d`, beyond the current
   explicit-zero bridge that reuses the `conv2d_bias` core path.
2. Wider dtype coverage, starting with clean `float32`.
3. Dynamic spatial bucket profiling and guarded execution-plan dispatch.
4. Broader Conv2d epilogues or fused activation forms. A future sigmoid retry
   must first solve the CUTLASS Conv epilogue ABI mismatch seen with
   `LinearCombinationSigmoid`; do not add public/frontend-only sigmoid surface
   before a real support-library `nvcc` build and runtime parity test pass.
5. `conv3d` with a separate admission pass and explicit NDHWC design review.
6. ROCm `ck_conv` parity using the same artifact-visible transform contract.
7. Transposed/depthwise families with their own provider metadata and tests.

## Current status

- A bounded `conv2d_bias` surface exists in v2: public semantics are still
  NCHW/OIHW/bias `[O]`, CPU reference execution validates against PyTorch, and
  CUDA lowering records explicit NHWC/OHWI provider transforms in
  `cutlass_conv` manifest/codegen metadata.
- A bounded no-bias `conv2d` public surface now also exists as an explicit-zero
  bridge. `dml.ops.conv2d(x, weight, ...)` performs its own static rank-4
  NCHW/OIHW/groups=1 validation, then emits a `conv2d_bias` core node with
  `source_op=conv2d`, `bias_mode=explicit_zero_constant`, and a traced zero-bias
  constant tensor. This keeps artifacts honest for future provider work and
  removes CLIP's old model-local synthetic zero-bias parameter, but it is not a
  separate no-bias CUTLASS family yet.
- The `float16` and exact `float32` SIMT, static rank-4, groups=1 CUDA paths now
  have correctness-first CUTLASS runtime launchers. Generated code allocates per-session
  NHWC/OHWI/NHWC temporaries, calls the support-library NCHW -> NHWC activation
  pack helper, OIHW -> OHWI weight pack helper, the manifest-selected provider
  launcher, and NHWC -> NCHW output unpack helper. The candidate set now records
  a SIMT fallback for both admitted dtypes, plus float16-only v1-inspired TensorOp
  `IteratorAlgorithm::kFewChannels` candidate selected only for semantic input
  `C=3`, and v1-inspired TensorOp `IteratorAlgorithm::kFixedChannels`
  candidates selected only for semantic input `C=4` or `C=8`, all with no
  channel padding. Focused CUDA runtime parity validates the selected C=3
  few-channel path, selected C=4 fixed-channel path, optimized C=16/O=16 path,
  the CLIP float32 patch-projection SIMT path, and a representative non-CLIP
  float32 SIMT shape against Torch or local Transformers, while manifest/source
  tests keep C=8 artifact-visible and keep non-3/4/8 fp16 shapes on the SIMT
  fallback. Static groups=1 float32 Conv now uses the bounded SIMT runtime and
  profiler boundary instead of remaining exact-shape scaffold-only.
- The profile workload builder now emits real static `cutlass_conv` workloads
  for compatible runtime candidates and preserves the same layout translation,
  weight-transform, Conv config, candidate/config, and source provenance in
  profile reports and cache keys. `dinoml profile` can time the exported Conv
  profiler symbols on provider-layout NHWC/OHWI buffers, write
  `debug/profile_report.json`, update the support-cache profile cache, and
  write `debug/execution_plan.json`.
- Static Conv execution-plan selections are consumable during compile. Applying
  a static selection updates the manifest `selected_candidate_id`,
  kernel/profiler symbols, `execution_plan_selection`, and
  `cutlass_conv_plan["selected_candidate"]` payload consumed by generated
  lowering. Stale or incompatible Conv selections are rejected in strict mode,
  and guarded/dynamic Conv dispatch remains explicitly unsupported. Focused
  fused-epilogue tests now prove that `conv2d_bias_relu` preserves
  `bias_relu` metadata through profile workloads, static execution-plan
  application, and compile-time plan consumption. A CUDA-gated
  `profile_artifact` smoke also proves a real `conv2d_bias_relu` artifact
  writes `debug/profile_report.json` and `debug/execution_plan.json` with the
  fused launch ABI/symbol metadata intact, while honestly allowing the
  confidence gate to leave the emitted plan low-confidence and non-consumable
  when real timings are too close.
- CUDA compile now materializes a `cutlass_conv` support-cache boundary under
  the advertised support `cache_dir`, including `lib/cutlass_conv_manifest.json`
  and `src/source_manifest.json` with the used candidate plan,
  candidate/config keys, and explicit layout/weight-transform provenance. When
  `nvcc` is available, the support cache also builds
  `lib/libdinoml_cutlass_conv.so` with concrete transform helper exports, the
  fp16 and float32 SIMT fallback launcher exports, the fp16 TensorOp FewChannels C=3
  launcher export, the C=4/C=8 FixedChannels exports, the regular Optimized
  align8 export, and matching real profiler exports for all emitted runtime
  candidates using the `dinoml_cutlass_conv2d_bias_v1` ABI. If `nvcc` is
  unavailable, the manifest records source-only status.
- The shared `cutlass_conv_plan` scaffold metadata is now validated before
  profile workload generation, codegen-plan support-library enumeration, and
  support-cache/source-manifest emission consume it. The current bounded
  contract explicitly rejects layout/candidate drift, incorrect temporary byte
  counts, inconsistent padded-channel bookkeeping, and malformed temporary-pack
  inventories instead of letting incoherent NCHW/OIHW -> NHWC/OHWI provenance
  propagate deeper into artifacts.
- `kernel_codegen_plan.json` now also records explicit scaffold wrapper stages
  for activation pack, weight pack, planned provider launch, and output unpack.
  Those entries are derived from the validated `cutlass_conv_plan`, keep the
  temporary-buffer/layout contract artifact-visible, and are source-renderable
  into future CUDA wrapper call snippets for tests without claiming runtime
  lowering is wired yet. Rejected CUDA artifacts now also emit guarded debug
  wrapper-scaffold `.cu` snippets plus
  `debug/generated_src/scaffold_source_manifest.json`, and
  `kernel_codegen_plan.json` links those source files back to the same per-node
  stage groups for artifact inspection only.
- Support-scaffold emission now also revalidates caller-supplied Conv
  `used_candidate_plan` payloads entry by entry before it writes those manifest
  files: the scaffold re-derives the selected candidate from the embedded
  candidate list, validates candidate-set provenance against the current
  NHWC/OHWI scaffold contract, and checks `node_id` when present. Direct
  caller-side mutations to selected-candidate layout or dtype metadata now fail
  before support manifests are emitted, so the scaffold does not become a
  side-door for stale candidate provenance.
- Model CUDA compile now builds a generated module when `nvcc` can compile the
  support library, and the artifact manifest carries
  `lib/libdinoml_cutlass_conv.so`. The selected provider launcher is
  the float32 SIMT implicit-GEMM fallback for static groups=1 shapes, or, for
  float16, either the SIMT implicit-GEMM fallback, the C=3 TensorOp FewChannels
  Fprop+bias call, the C=4/C=8 TensorOp FixedChannels Fprop+bias call, or the
  regular TensorOp Optimized Fprop+bias call for naturally aligned
  non-small-channel shapes (`C >= 16`, input/output channels divisible by 8),
  selected from artifact-visible candidate predicates. The same predicate
  evaluator now filters Conv profile workload construction so incompatible
  small-channel and optimized candidates are not emitted for a node's
  shape/layout/dtype contract. CUDA runtime parity covers C=3 FewChannels,
  C=4 FixedChannels, C=8 FixedChannels, optimized C=16/O=16, the CLIP float32
  patch-projection SIMT path, and a non-CLIP float32 stride/padding/dilation
  SIMT path against Torch/local Transformers; CUDA-gated profile smoke validates the real Conv
  profiler exports through
  `profile_artifact`.
  No dynamic Conv profiling, guarded Conv dispatch,
  grouped/depthwise/transposed/3D coverage, runtime-set packed weights, or
  public NHWC semantics are claimed.

## Next Provider-Maturity Lane

The next `cutlass_conv` slice should keep tightening maturity without
broadening public ConvNd semantics:

- decide whether dynamic spatial/profile buckets and guarded Conv dispatch need
  admission, or keep rejecting them explicitly;
- keep profile/cache/execution-plan schemas stable as Conv gains more shapes or
  candidates;
- do not add grouped/depthwise/transposed/3D Conv, runtime-persistent packed
  weights, hidden channel padding, or public NHWC semantics without a separate
  admission pass.
