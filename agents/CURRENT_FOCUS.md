# Current Focus

## Primary Focus

- Human-directed feature target: start the first full-model sprint with CLIP as
  the primary model target. The sprint should move from the recently landed
  CLIP text/contrastive composition slices toward a real model implementation,
  using the model audits in `agents/plans/transformers/clip/` and pinned
  Transformers source/reference snippets rather than the ambient installed
  package alone.
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
  contrastive wrapper integration.
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

- On 2026-05-15, the bounded CLIP text-feature wrapper exists and covers both
  source EOS pooling branches. Preferred next slice: make that model path more
  visible with a small example or artifact-inspection test that proves the
  text-wrapper workflow and provider/model split without relying on the
  expensive CUDA smoke. After that, continue with either reducing the explicit
  `position_ids` requirement or beginning the vision patch path, keeping each
  slice narrow and parity-backed.
