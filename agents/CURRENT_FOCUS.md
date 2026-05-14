# Current Focus

## Primary Focus

- Human-directed feature target: start the first full-model sprint with CLIP as
  the primary model target. The sprint should move from the recently landed
  CLIP text/contrastive composition slices toward a real model implementation,
  using the model audits in `agents/plans/transformers/clip/` and pinned
  Transformers source/reference snippets rather than the ambient installed
  package alone.
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
  contrastive wrapper integration.
- CUTLASS Conv maturity needed by CLIP vision: static profiling is landed;
  next gaps include C=8 parity if useful, dynamic/guarded admission decisions,
  and avoiding unsupported grouped/depthwise/transposed/3D claims.
- Attention path: preserve dense attention parity first, then explore
  FlashAttention-style provider integration in the v1 manner.
- Op porting: pull small/custom v1 ops only when they unblock the CLIP/model
  sprint or expose a clear admission/validation gap.
- Stabilization: review recently landed ops/providers for CUDA behavior,
  dynamic-shape support, and optimized kernels before widening public surface.

## Preferred Next Work

- On 2026-05-15, begin with a bounded CLIP model integration loop. Preferred
  first slice: pin the exact CLIP reference source/audit, then add the smallest
  real `CLIPTextModel` or text-feature wrapper path that can reuse the already
  landed embedding, LayerNorm, dense attention, MLP, pooling, projection, and
  contrastive composition coverage. If that immediately exposes a provider gap,
  stop and finish that gap before continuing the model.
