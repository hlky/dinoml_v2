# Current Focus

## Primary Focus

- Human-directed feature target: continue the first full-model sprint with CLIP
  as the primary model target. The sprint has moved from isolated
  text/vision/contrastive composition slices into a bounded `LegacyCLIPModel`
  surface; keep advancing it against the model audits in
  `agents/plans/transformers/clip/` and pinned Transformers source/reference
  snippets rather than the ambient installed package alone.
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
- Naive generated CPU GEMM is acceptable as a temporary bridge for compiled CPU
  artifacts. Do not block CLIP artifact progress solely waiting for a future CPU
  BLAS/library integration; keep any naive path clearly scoped, validated, and
  honest about performance.
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
  requires explicit `position_ids`. Text and vision tower depth, including
  zero-layer text and vision paths, are covered by deterministic parity tests,
  and the first bounded two-tower `LegacyCLIPModel` workflow matches local
  Transformers for projected features, normalized embeds, and logits with a
  compact runnable workflow proof. The current artifact boundary is now pinned:
  naive generated CPU bridges exist for `gemm_rcr`, `gemm_rcr_bias`,
  `bmm_rcr`, and `bmm_rrr`, so zero-layer text CPU artifacts can run and both
  CLIP attention matmuls can compile/run on CPU artifacts. Deeper
  text/two-tower CPU compilation now fails first at
  `gemm_rcr_bias_fast_gelu`. Exact CLIP float32 patch Conv can compile to a CUDA
  artifact but still fails through the scaffolded CUTLASS Conv runtime launcher
  boundary, and zero-text/zero-vision two-tower CPU compilation still stops
  honestly at `conv2d_bias`. Preferred next slice: narrow the CPU
  `gemm_rcr_bias_fast_gelu` blocker, advance the exact Conv launcher boundary,
  or pick a new
  narrow Transformers parity gap that is not already covered by the layer-count
  proofs.
