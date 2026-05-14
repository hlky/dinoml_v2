# ConvNd CUTLASS Provider Plan

This plan captures the recent ConvNd exploration so future implementation can
start from settled constraints instead of re-learning v1 behavior or drifting
into hidden layout state.

The first implementation target is intentionally narrow: a CUDA-only
`conv2d_bias` provider-backed slice with artifact-visible internal layout
translation. This is a design-first plan, not an implementation status update.

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
- fused epilogue expansion beyond bias
- dynamic H/W profiling or guarded runtime shape dispatch
- global layout pass work
- runtime-set transformed weights or hidden persistent packed-weight state
- public NHWC/NDHWC toggles
- ABI-stride-based NHWC execution
- hidden small-channel auto-padding
- ROCm/CK parity
- CPU compiled conv backend

These all deserve separate admission and design updates after the first bounded
slice is proven.

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

1. `conv2d` without bias if it naturally shares the provider family.
2. Wider dtype coverage, starting with clean `float32`.
3. Dynamic spatial bucket profiling and guarded execution-plan dispatch.
4. Broader Conv2d epilogues or fused activation forms.
5. `conv3d` with a separate admission pass and explicit NDHWC design review.
6. ROCm `ck_conv` parity using the same artifact-visible transform contract.
7. Transposed/depthwise families with their own provider metadata and tests.

## Current status

- A reference/scaffold-only `conv2d_bias` surface exists in v2: public semantics
  are NCHW/OIHW, CPU reference execution validates against PyTorch, CUDA compile
  emits `cutlass_conv` manifest/codegen metadata with NHWC/OHWI transform plans,
  and then rejects before module build.
- The profile workload builder now has a scaffold-only `cutlass_conv` workload
  that preserves the same layout translation and weight-transform metadata, and
  refuses manifests that omit that transform plan.
- CUDA compile now also materializes a manifest-only `cutlass_conv` support
  cache scaffold under the advertised support `cache_dir`, including
  `lib/cutlass_conv_manifest.json` and `src/source_manifest.json` with the used
  candidate plan, candidate/config keys, and explicit layout/weight-transform
  provenance. This is still scaffold-only metadata: no ConvNd profiler
  execution, compiled support library, or runtime launcher exists yet.
