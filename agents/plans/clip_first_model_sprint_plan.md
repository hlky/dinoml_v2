# CLIP First-Model Sprint Plan

This note captures the first full-model sprint for the transformers family work.
The sprint is intentionally text-tower-first on CLIP, with ViT as the conservative
vision-side companion path and T5 explicitly deferred.

## Parity target

The CLIP sprint target is practical parity with the pinned/local Transformers
implementation for supported inference surfaces, not a CLIP-like approximation.
Each bounded DinoML slice should either match the relevant Transformers behavior
for the surface it admits or state the remaining non-parity limits plainly in the
same tests/docs that land the slice. The local `/workspace/transformers`
implementation and the audit in `agents/plans/transformers/clip/report.md` are
the behavior source unless a later explicit design decision says otherwise.

## Why CLIP text first

CLIP text tower work gives the highest-signal bounded slice:

- It proves token embedding, absolute position embedding, dense causal attention,
  EOS pooling, projection, and `get_text_features` without needing generation or
  cross-attention.
- It reuses the recently landed `layer_norm` primitive and stays close to the
  minimal CLIP text contract described in the audit.
- It gives a clear reference shape for later wrapper models such as
  `CLIPModel`, `ChineseCLIP`, and `VisionTextDualEncoder`.

ViT remains the conservative alternative and the CLIP vision-side follow-up:

- It is a pure vision encoder with patch embedding, CLS pooling, and absolute
  positions.
- The patch path is easier to stage than the full dual-encoder surface.
- Once the CLIP text tower is stable, the same discipline should carry over to
  the vision branch and then to the contrastive head.

T5 is deferred for this sprint:

- Its relative attention bias is not a cosmetic detail; it is part of the
  semantic foundation.
- The decoder brings cross-attention and cache semantics that are materially
  broader than the CLIP text tower.
- That makes T5 a later sprint, not the first-model slice.

## Staged path

1. `layer_norm` is already landed.
2. Add token embedding and absolute position embedding.
3. Close the EOS pooling gap if it is missing in v2. Preserve the source CLIP
   behavior: the common `eos_token_id == 2` path uses the integer/EOT fast path,
   and newer non-2 EOS configs use first-match EOS pooling.
4. Land dense semantic attention for the text tower before any flash provider
   optimization. Dense eager-visible semantics are the truth source.
5. Add `CLIPTextModel` and `get_text_features` once embeddings, attention, norm,
   pooling, and projection are in place.
6. Add the vision patch path as the conservative companion: patch conv, flatten,
   CLS token, position add, encoder, CLS pool, projection.
7. Add the contrastive head: L2 normalize both branches, apply
   `exp(logit_scale)`, compute text-image similarity, and transpose for
   `logits_per_image`.

## Helper composition vs generated ops

Keep the line between helper wiring and admitted model math explicit:

- Honest helper composition: `quick_gelu` only if it is a true alias of the
  existing `fast_gelu` approximation; otherwise keep it as a shared helper, not
  a new public op. Pooling should stay helper-level only when it is just the
  admitted EOS-selection logic.
- Registered/generated ops: `layer_norm`, embeddings, and any other tower
  primitive that cannot be expressed honestly through already-admitted
  backend-visible ops.
- Prefer composition for projection and eager dense attention while existing
  GEMM/BMM/softmax/reduction primitives can preserve the source semantics. Add a
  new attention op only when the admission case is explicit and the validation
  story covers masks, fp32 score behavior, and layout.
- Wrapper-owned math can remain thin for the contrastive head if it stays
  explicit and testable. Do not hide the semantic tower inside wrapper glue.

## FlashAttention policy

FlashAttention is an optimization, not the semantic base.

- First land dense semantic parity for the CLIP text tower.
- Only after dense parity should a flash provider be considered.
- If a flash backend cannot match the dense contract, it does not define the
  model behavior.

## Validation gates

Each stage should have a narrow validation story:

- Embeddings and LayerNorm: frontend contract, shape/type inference, CPU
  reference, generated source provenance, and runtime parity.
- EOS pooling: explicit tests for integer-EOS and non-2 EOS paths.
- Dense attention: eager-visible causal + padding-mask parity first, then
  backend-specific runtime coverage.
- `CLIPTextModel` / `get_text_features`: end-to-end text feature parity against a
  tiny CLIP fixture and one real CLIP checkpoint when feasible.
- ViT patch path: patch count, CLS path, position add, pool output, and the
  same manifest/runtime checks used for the text tower.
- Contrastive head: normalized embeddings, scale application, logit shape, and
  transpose orientation.

## Worktree guidance

Keep this sprint isolated from queue-tracking churn:

- Use independent branches/worktrees for parallel model-family work.
- Avoid editing shared queue/tracking docs from this branch; reconcile those
  after merge in the main line.
- Keep the plan bounded to the first-model slice so later branches can add the
  vision side, wrapper models, and provider optimizations without conflict.

## 2026-05-15 kickoff

The next workday should start with a model-facing slice again, not another
open-ended provider sweep:

- Pin the exact CLIP reference source used for implementation. The audit in
  `agents/plans/transformers/clip/report.md` is the project-memory starting
  point; record any installed or vendored Transformers source version used for
  behavior details.
- Prefer the smallest real `CLIPTextModel` or `get_text_features` wrapper slice
  that can reuse the already landed text composition coverage: token/position
  embeddings, LayerNorm, dense causal attention, MLP/quick-gelu composition,
  legacy EOS pooling, projection, and contrastive-head math.
- If the wrapper slice exposes a concrete missing op/provider/runtime contract,
  pause the model work and finish that gap completely or as far as safely
  possible before returning to CLIP.
- For vision-side work, keep modeling code semantic NCHW. The CUTLASS Conv
  provider should own NHWC/OHWI transforms through manifest-visible
  `cutlass_conv_plan` metadata, static profiling, and generated wrapper stages.
- Do not start FlashAttention as the semantic baseline. Dense attention parity
  remains the acceptance gate; FlashAttention-style provider work should follow
  only after the dense CLIP path is pinned.

## 2026-05-15 landed text slice

A bounded text-only wrapper path is now in-tree at `src/dinoml/models/clip.py`.

- The landed surface is intentionally narrow: a legacy-OpenAI
  `get_text_features`-style wrapper composed from existing DinoML ops for token
  embedding, position embedding, any non-negative number of text encoder
  layers, final
  LayerNorm, CLIP EOS pooling (`eos_token_id == 2` argmax compatibility or
  first equality match for non-2 EOS), and bias-free text projection.
- The wrapper keeps the current honest limits explicit: static traced sequence
  length bounded by `max_position_embeddings`, text-only scope, and optional
  default `position_ids` that fall back to the traced `[0, 1, ..., S-1]`
  sequence when callers omit them. The explicit `position_ids` path still
  works. The non-2 EOS branch now relies on bounded integer `eq` admission plus
  bool `argmax`, and still assumes tokenizer-prepared sequences contain an EOS
  token the same way the Transformers source does.
- Focused wrapper-level tests compare the DinoML path against the pinned local
  Transformers CLIP source, now including the preserved zero-layer text path
  for both EOS pooling branches and both explicit/default `position_ids`, and
  keep manifest ownership honest by proving that no new public op or provider
  surface was introduced for this slice.

## 2026-05-15 landed vision-embeddings slice

A bounded CLIP vision-embeddings path is now in-tree at `src/dinoml/models/clip.py`.

- The landed surface is intentionally narrow: fixed square NCHW pixel input,
  source-faithful patch projection via `conv2d_bias(..., zero_bias)` to model
  the bias-free Transformers `nn.Conv2d`, patch flatten + `[B, hidden, gh, gw]`
  -> `[B, patches, hidden]` transpose, CLS prepend, and learned absolute
  position add from the local/pinned Transformers `CLIPVisionEmbeddings`
  source.
- The wrapper keeps the current honest limits explicit: no position
  interpolation, no arbitrary image sizes, no vision encoder/projection head,
  no grouped/depthwise/transposed/3D Conv claims, and no new provider surface.
  CPU reference execution proves parity for the admitted slice, and compiled
  CPU artifacts now also run through a bounded generated naive `conv2d_bias`
  bridge for the admitted static groups=1 float32 path. CUDA manifest checks
  keep ownership visible: the Conv node stays on the existing CUTLASS Conv
  scaffold plan while sequence assembly and position add remain model-generated
  kernels.
- For the exact CLIP patch-projection shape already used by
  `LegacyCLIPVisionEmbeddings` (`float32` NCHW input `[B,3,4,4]`,
  OIHW weights `[6,3,2,2]`, stride 2, padding 0, groups 1), CUDA planning still
  selects the float32 SIMT CUTLASS Conv scaffold with
  `blocked_reason: cutlass_conv_runtime_launcher_not_implemented`. The artifact
  can remain provider-visible, but runtime execution still fails through the
  generated scaffold boundary until a real float32 launcher is added or the
  admitted CLIP path is intentionally retargeted to a separately validated fp16
  runtime slice.
- Focused wrapper-level tests compare both the full embeddings output and the
  zero-bias patch-projection substep against the pinned local Transformers CLIP
  implementation.

## 2026-05-15 landed vision stem/pool/projection slice

A bounded CLIP vision-wrapper path is now in-tree at `src/dinoml/models/clip.py`.

- The landed surface is intentionally narrow: source-faithful fixed-size vision
  embeddings, `pre_layrnorm`, a no-op encoder admitted only as
  `num_hidden_layers == 0`, CLS pooling via the first sequence token,
  `post_layernorm`, and bias-free visual projection. This matches the local
  Transformers `CLIPVisionModelWithProjection` behavior for a zero-layer
  `CLIPVisionConfig` without claiming a full encoder layer yet.
- The wrapper keeps the current honest limits explicit: fixed square NCHW pixel
  input only, no positional interpolation, no arbitrary image sizes, no real
  vision encoder block, no full `CLIPModel`, and no widened Conv/provider
  claims. CPU reference execution proves parity for `last_hidden_state`,
  `pooler_output`, and projected `image_features`, and compiled CPU artifacts
  now also run through the bounded naive `conv2d_bias` bridge for the admitted
  static groups=1 float32 path.
- Focused wrapper-level tests compare the DinoML outputs against the pinned
  local Transformers implementation and keep CUDA manifest/generated-source
  ownership visible: the patch projection stays on the existing CUTLASS Conv
  scaffold, the final projection stays CUTLASS GEMM-backed, and the sequence
  assembly plus LayerNorm/pool path stay model-generated.

## 2026-05-15 landed bounded multi-layer vision encoder slice

The bounded CLIP vision wrapper now also admits stacked real encoder blocks.

- The admitted surface remains intentionally narrow: fixed-size semantic NCHW
  embeddings, `pre_layrnorm`, any non-negative `num_hidden_layers`, CLS
  pooling, `post_layernorm`, and bias-free visual projection. The encoder path
  matches the local Transformers `CLIPVisionModelWithProjection` contract as
  `layer_norm -> dense noncausal self-attention -> residual -> layer_norm ->
  quick_gelu MLP -> residual`, reusing the already landed GEMM/BMM/softmax and
  `gemm_rcr_bias_fast_gelu` composition rather than introducing a new public
  attention or activation op.
- The honest limits stay explicit: `num_hidden_layers` is admitted only for
  non-negative integers, there is still no positional interpolation, no
  arbitrary image sizes, no padding/causal mask handling in the vision block,
  no full `CLIPModel`, and no widened Conv/provider claims. CPU reference
  execution now proves the preserved zero-layer path plus deterministic one-
  and two-layer paths against local Transformers, and compiled CPU artifacts
  now run through the same bounded naive `conv2d_bias` bridge for the admitted
  static groups=1 float32 path.
- Focused tests keep ownership visible on the encoder path: patch projection
  remains on the CUTLASS Conv scaffold, attention/MLP GEMM+BMM pieces stay
  CUTLASS-backed, and sequence assembly plus LayerNorm/softmax/pool path stay
  model-generated.

## 2026-05-15 landed bounded two-tower contrastive slice

The smallest real CLIPModel-style assembly is now in-tree at
`src/dinoml/models/clip.py`.

- The admitted surface stays intentionally narrow: reuse the bounded
  `LegacyCLIPTextModelWithProjection` and bounded
  `LegacyCLIPVisionModelWithProjection` slices to produce projected text and
  image features, L2-normalize both with the existing `vector_norm` + `div`
  composition, apply `exp(logit_scale)`, compute `logits_per_text` with
  `gemm_rcr`, and transpose to `logits_per_image`. The wrapper also exposes
  `get_text_features` and `get_image_features` in the same bounded spirit as
  the local Transformers `CLIPModel`.
- The honest limits stay explicit: static traced text sequence length, default
  traced text positions only, fixed square NCHW pixel input only, no positional
  interpolation, no tokenizer/processor plumbing, no loss path, and no new op
  or provider surface. The top-level
  model remains a proof for the admitted tiny CLIPConfig surface, not a broad
  CLIP checkpoint claim. With the bounded naive compiled CPU `bmm_rcr`,
  `bmm_rrr`, `gemm_rcr_bias_fast_gelu`, and `conv2d_bias` bridges, the bounded
  full-model CPU artifact now matches local `/workspace/transformers`. CUDA
  artifact planning/codegen still exposes the single Conv node honestly as a
  `cutlass_conv` scaffold entry with visible activation/weight pack and output
  unpack wrapper stages.
- Focused tests compare projected features, normalized embeds, and
  `logits_per_text` / `logits_per_image` against the local
  `/workspace/transformers` `CLIPModel` with deterministic weights and keep
  provider/model ownership visible in the CUDA manifest.
- A compact runnable workflow proof now lives in
  `examples/clip_model_workflow.py`: it traces the bounded two-tower wrapper on
  synthetic token/image tensors, compiles a bounded CPU `.dinoml` artifact,
  loads it with `dinoml.runtime`, runs `session.run_numpy(...)`, proves parity
  against both the eager CPU reference and local `/workspace/transformers`, and
  records artifact-visible limits in the summary itself. The proof stays
  intentionally narrow by making the current admitted boundaries test-visible:
  no `position_ids` input because the text branch uses traced default
  positions, fixed square NCHW image shape, one Conv scaffold entry in the CUDA
  manifest, and no tokenizer/processor or positional-interpolation plumbing.
