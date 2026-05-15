# Current Focus

## Primary Focus

- Human-directed feature target: start the first full-model sprint with CLIP as
  the primary model target. The sprint should move from the recently landed
  CLIP text/contrastive composition slices toward a real model implementation,
  using the model audits in `agents/plans/transformers/clip/` and pinned
  Transformers source/reference snippets rather than the ambient installed
  package alone.
- CLIP integration is a practical parity target against the pinned/local
  Transformers implementation for the supported inference surfaces. Bounded
  slices are acceptable only when their admitted behavior matches Transformers
  or their remaining non-parity limits are explicit and test-backed.
- libgguf direct linking is landed. Keep any follow-up narrowly tied to concrete
  direct-link failures or validation gaps; do not reopen GGUF policy, quantized
  GEMM families, epilogue coverage, or public provider surface as part of that
  lane.
- Preserve the model-integration flow: identify a concrete op/provider/runtime
  gap, fill it completely or as far as safely possible with validation, then
  return to the model path. CUTLASS Conv remains a legitimate blocking lane for
  CLIP vision work; keep authoring model graphs in semantic NCHW and let the
  provider transform to NHWC/OHWI through artifact-visible plans.
- FlashAttention is a priority optimization for the attention path, but dense
  semantic attention remains the truth source. Use v1 and the external
  FlashAttention reference as guidance only after the dense CLIP contract is
  pinned.

## Near-Term Priorities

- CLIP first-model sprint: text model, vision patch path, projections, and
  contrastive wrapper integration, with Transformers parity as the acceptance
  bar for each admitted surface.
- CUTLASS Conv maturity needed by CLIP vision: static profiling is landed;
  next gaps include C=8 parity if useful, dynamic/guarded admission decisions,
  and avoiding unsupported grouped/depthwise/transposed/3D claims.
- Attention path: preserve dense attention parity first, then explore
  FlashAttention-style provider integration in the v1 manner.
- libgguf direct-link follow-up only if a concrete runtime/build/cache failure
  appears after the landed submodule-backed integration.
- Op porting: pull small/custom v1 ops only when they unblock the CLIP/model
  sprint or expose a clear admission/validation gap.
- Stabilization: review recently landed ops/providers for CUDA behavior,
  dynamic-shape support, and optimized kernels before widening public surface.

## Preferred Next Work

- On 2026-05-15, the bounded CLIP text-feature wrapper exists, covers both
  source EOS pooling branches, has a visible text workflow proof, and no longer
  requires explicit `position_ids`. The vision side now has fixed-size
  embeddings plus a zero-layer pool/projection wrapper. Preferred next slice:
  add the first real vision encoder layer with Transformers parity, or address
  Conv/provider runtime maturity only if it is a concrete blocker to CLIP
  vision artifacts.
