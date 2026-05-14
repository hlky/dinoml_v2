# Rotary / `apply_rotary_emb` Plan

This note turns the recent RoPE exploration into project memory before any v2
implementation starts. It is intentionally narrower than a feature proposal:
the goal is to record what the old prototype actually does, what variant space
exists across v1/diffusers/transformers, and what the first honest v2 slice
should be.

## Why This Exists

- The old prototype lives in `/workspace/apply_rotary_emb`, not in this repo.
- v1 already has separate rotary table-generation helpers such as
  `get_1d_rotary_pos_embed`, `get_2d_rotary_pos_embed`,
  `get_3d_rotary_pos_embed`, and `get_3d_rotary_pos_embed_allegro`.
- Current diffusers/transformers planning docs repeatedly depend on RoPE, but
  not on one single universal application ABI:
  - Stable Audio needs partial 1D RoPE on Q/K after table generation.
  - Mistral/Ministral families need text RoPE, YaRN-style scaling, cache-order
    honesty, and in some cases query-only post-RoPE scaling.
  - Wan/SkyReels need 3D real-table RoPE preparation.
  - Z-Image needs multi-axis complex RoPE and position-id preparation.

The main lesson is that "RoPE support" is not one thing. Table generation,
layout preparation, and Q/K/V application variants must be tracked separately.

## Old Prototype Contract

The old prototype is a PyTorch CUDA extension registered as:

`apply_rotary_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, bool use_real, int use_real_unbind_dim) -> Tensor[]`

Observed contract from `/workspace/apply_rotary_emb`:

- CUDA-only Torch op. Registration is `torch::kCUDA` only.
- Takes a coupled Q/K pair ABI:
  `query`, `key`, `cos`, `sin`, `use_real`, `use_real_unbind_dim`.
- Expects:
  - `query` rank 4, contiguous, CUDA, shape `[B, H, S, D]`
  - `key` rank 4, contiguous, CUDA, shape `[B, H, S, D]`
  - `cos` and `sin` rank 2, contiguous, CUDA, shape `[S, D]`
- Returns two tensors `{out_query, out_key}`. The Python wrapper docstring says
  it returns a stacked tensor `[2, B, H, S, D]`, but the C++ binding actually
  returns `Tensor[]`, and the tests unpack two tensors.
- The implementation uses `query.data_ptr<float>()`, `key.data_ptr<float>()`,
  `cos.data_ptr<float>()`, and `sin.data_ptr<float>()` in both paths, so the
  runtime is effectively float32-only even though there is no dtype check.
- It only validates:
  - CUDA residency
  - contiguity
  - query/key rank 4
  - cos/sin rank 2
  - even `query.size(3)`
- It does not validate:
  - query/key dtype match
  - cos/sin dtype match
  - key last-dimension evenness
  - query/key same shape
  - cos/sin shape compatibility with `S` and `D`
  - even `D` before the reinterpret-cast path on every tensor

### Layout Behavior

There are only two real behaviors:

1. Adjacent real pairs / interleaved mode
   - Triggered by `!use_real || use_real_unbind_dim == -1`
   - Reinterprets input/output/cos/sin as `float2*`
   - Treats channels as adjacent real pairs `(x0, x1), (x2, x3), ...`
   - This is the only path exercised by the tests

2. Split-half real pairs / non-interleaved mode
   - Triggered by `use_real && use_real_unbind_dim == -2`
   - Treats first `D/2` channels as the real half and second `D/2` as the
     paired half

`use_real=False` does not implement honest complex support. It falls into the
same adjacent-pair path as `use_real_unbind_dim == -1`, with no
`view_as_complex`, no complex dtype ABI, and no complex-valued output surface.

### Why It Cannot Be Ported Directly

The prototype is useful as a narrow exploration, but it is not a v2-ready ABI:

- It is a fused Torch extension, not a v2 frontend/helper/op/provider design.
- It couples Q and K into one public op even though several families need only
  table generation first, or apply to one tensor at a time, or apply to
  prefixes only.
- It hides layout semantics behind `use_real` and `use_real_unbind_dim` rather
  than naming the actual pairing/layout contract.
- It is not honest about dtype support.
- It is not honest about complex support.
- It assumes one fixed `[B, H, S, D]` layout and same-shape Q/K.
- It has no partial-rotary prefix+tail behavior.
- It has no multi-axis, 2D, 3D, or M-RoPE support.
- It has no rotate-V variant.
- It has no artifact-visible provider or manifest story.
- It has no v2-style admission story around static vs dynamic shape, bounded
  frontend helpers, CPU reference, or generated runtime lowering.

## Variant Taxonomy To Preserve

The exploration across v1, diffusers plans, transformers plans, and the old
prototype found several variant families that should stay separate in design.

### 1. Table Generation / Preparation Variants

- 1D rotary tables from scalar positions or `arange(S)`
- Real-table duplication styles:
  - `repeat_interleave(2)` style
  - `concat([x, x])` style
- Theta / inverse-frequency parameterization
- Dynamic/scaled long-context families:
  - linear / NTK-like scaling
  - YaRN
  - LongRoPE
  - Llama 3 / Llama 4 families
- Position-id-driven preparation instead of plain `arange(S)`
- Multi-axis table generation:
  - 2D mesh
  - 3D temporal-height-width
  - M-RoPE axis partitioning

### 2. Application Layout Variants

- Split-half full-head rotary
- Even-odd / adjacent-pair interleaved rotary
- Diffusers real-pair single-tensor style
- Transposed real-pair variants where attention tensors are not already laid
  out as `[B, H, S, D]`
- Complex `view_as_complex` style application
- `rope_interleave` / prelayout variants where Q/K projection or weight layout
  has already been prepared for a specific rotary convention

### 3. Coverage Variants Beyond "rotate full Q and K"

- Partial rotary prefix + unrotated tail
- Q-only or K-only apply helpers
- Coupled Q/K pair helper
- Multi-axis 2D / 3D / M-RoPE application
- Rotate-V variants
- Cache-order-sensitive decoder application, where keys are cached after RoPE
  but values are not

These variants appear in different model families and should not be flattened
into one prematurely broad public op.

## Recommended v2 Separation

Keep table generation/preparation and application kernels separate.

### A. Table Generation / Preparation

This is the best first surface because it is artifact-visible and easier to
admit honestly:

- static attrs and visible outputs
- no hidden provider/runtime coupling
- easier CPU reference
- easier shape contract
- directly useful to multiple families before attention-provider work exists

Likely examples:

- `get_1d_rotary_pos_embed`
- later `get_2d_rotary_pos_embed`, `get_3d_rotary_pos_embed`, and
  model-specific axis/index helpers

### B. Application

Treat application as a later family of bounded helpers or ops, not as an
assumed universal public ABI. Possible slices:

- one-tensor real-pair helper
- coupled Q/K helper
- partial-prefix helper
- complex apply helper
- multi-axis helper

The exact surface should follow the model lane that needs it, not the old
prototype's Torch extension ABI.

## v2 Admission / Readiness Guidance

- Do not add a universal public `apply_rotary_emb` yet.
- Prefer bounded helpers or narrow ops whose layout contract is explicit in the
  name or attrs, not encoded in a magic integer like `use_real_unbind_dim`.
- Preserve static/dynamic-shape honesty. If a helper only supports static
  sequence length or static rotary dimension, say so.
- Preserve dtype honesty. Do not imply generic float or complex support without
  tests.
- Preserve artifact-visible design:
  provider choices, if any, must flow through manifests/profile reports/
  execution plans where relevant.
- No provider/runtime claims for rotary application without generated lowering
  and focused validation.
- Use v1 as a behavioral reference, not as a license to clone its public
  surface wholesale.

## Recommended First Bounded Target

Prefer a small table-generation helper slice around `get_1d_rotary_pos_embed`,
not a fused Q/K CUDA application ABI.

Recommended first slice:

- bounded helper or op for 1D rotary cos/sin table generation
- real-table output only
- support both real duplication conventions:
  - `repeat_interleave_real=True`
  - `repeat_interleave_real=False`
- explicit statement that this is table generation only, not Q/K application

Why this first:

- It matches real downstream demand from Stable Audio and text-decoder families.
- It preserves the separation already visible in v1.
- It avoids prematurely freezing a coupled Q/K ABI.
- It keeps shape/dtype/layout contracts tractable for the first admission.
- It is useful even before attention or cache providers are ready.

The old prototype is still valuable as a later reference for one application
variant, specifically adjacent-pair and split-half real Q/K application, but it
should not define the first v2 surface.

## Design Questions To Leave In-Plan

These are not policy blocks yet, but they should be answered before broader
integration:

1. Should the first 1D helper be frontend-only composition or a bounded op?
2. Should real-table generation accept only static integer `sequence_length`, or
   also a bounded position tensor input?
3. For application helpers, should partial rotary be expressed as
   `(rotary_prefix, pass_through_tail)` composition instead of a monolithic op?
4. Should one-tensor apply helpers come before a coupled Q/K helper?
5. Which layout naming is clearest for real-pair variants:
   `interleaved`, `split_half`, `repeat_interleave_real`, or something else?
6. When decoder/cache families arrive, should "keys cached after RoPE, values
   unrotated" be represented as helper-level composition rather than a fused
   attention/provider promise?
7. For complex RoPE families, is the right public surface true complex tables,
   explicit real/imag pair tensors, or model-local helpers?

## Minimal Test / Admission Matrix

### First Slice: `get_1d_rotary_pos_embed` Table Generation

- Frontend/admission:
  - reject odd rotary dimension
  - pin real-table output shape
  - pin both duplication conventions
  - document whether dynamic `S` or tensor `pos` is admitted
- Reference:
  - compare against v1/diffusers-style NumPy or torch reference for a few
    small `S`/`D` cases
- Dtype:
  - be explicit about first admitted dtype, ideally `float32` first
  - only add reduced-precision output claims with focused parity tests
- CPU/CUDA:
  - if helper-only composition, prove lowered graph shape and runtime parity on
    the admitted path
  - if bounded op, require generated CPU/CUDA parity before claiming both
- Docs:
  - update checklist and model plans that depend on 1D RoPE tables

### Later Application Slices

- Layout parity:
  - split-half full-head
  - interleaved adjacent-pair
  - partial-prefix + tail
- ABI honesty:
  - one-tensor vs coupled Q/K
  - explicit return-shape tests
- Dtype honesty:
  - no hidden float32-only behavior
- Shape/layout honesty:
  - pin admitted rank/layout exactly
- Model-facing parity:
  - at least one focused downstream parity test in the model lane that uses the
    helper, not only synthetic tensor tests

## Suggested Next Implementation Order

1. `get_1d_rotary_pos_embed` bounded first slice with duplicated-real variants
2. whichever application helper is demanded by the first live model lane
   (likely partial real-pair Q/K apply, not universal fused apply)
3. model-specific 2D/3D/M-RoPE preparation helpers
4. only then revisit whether any fused/provider-backed rotary application is
   worth admitting
