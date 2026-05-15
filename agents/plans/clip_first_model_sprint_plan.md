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

- Honest helper composition: `quick_gelu` is not a silent alias of the
  existing `fast_gelu` approximation for CLIP. Keep `fast_gelu` as its own
  admitted surface and use a distinct fused `quick_gelu` GEMM op when the model
  contract requires `x * sigmoid(1.702 * x)`. Pooling should stay helper-level
  only when it is just the admitted EOS-selection logic.
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
  bridge for the admitted static groups=1 float32 path. CUDA manifest/runtime
  checks keep ownership visible: the Conv node stays on the CUTLASS Conv
  provider path with explicit pack/launch/unpack metadata while sequence
  assembly and position add remain model-generated kernels.
- For the exact CLIP patch-projection shape already used by
  `LegacyCLIPVisionEmbeddings` (`float32` NCHW input `[B,3,4,4]`,
  OIHW weights `[6,3,2,2]`, stride 2, padding 0, groups 1), CUDA planning now
  selects the bounded-runtime float32 SIMT CUTLASS Conv candidate. CUDA-gated
  coverage compiles the patch-projection artifact and compares the runtime
  output against local Transformers instead of stopping at the old scaffold
  launcher boundary.
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
  quick_gelu MLP -> residual`, reusing the landed GEMM/BMM/softmax families and
  the distinct fused `gemm_rcr_bias_quick_gelu` slice rather than introducing a
  new public attention op.
- The honest limits stay explicit: `num_hidden_layers` is admitted only for
  non-negative integers, there is still no positional interpolation, no
  arbitrary image sizes, no padding/causal mask handling in the vision block,
  no full `CLIPModel`, and no widened Conv/provider claims. CPU reference
  execution now proves the preserved zero-layer path plus deterministic one-
  and two-layer paths against local Transformers, and compiled CPU artifacts
  now run through the same bounded naive `conv2d_bias` bridge for the admitted
  static groups=1 float32 path.
- Focused tests keep ownership visible on the encoder path: patch projection
  remains on the bounded CUTLASS Conv provider path, attention/MLP GEMM+BMM
  pieces stay CUTLASS-backed, and sequence assembly plus LayerNorm/softmax/pool
  path stay model-generated.

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
  `bmm_rrr`, `gemm_rcr_bias_quick_gelu`, and `conv2d_bias` bridges, the bounded
  full-model CPU artifact now matches local `/workspace/transformers`. CUDA
  artifact planning/codegen now exposes the single Conv node honestly as a
  bounded-runtime `cutlass_conv` entry with visible activation/weight pack,
  provider launch, and output unpack wrapper stages.

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
  positions, fixed square NCHW image shape, one bounded-runtime Conv provider
  entry in the CUDA manifest, and no tokenizer/processor or
  positional-interpolation plumbing.

## 2026-05-15 CUDA layer_norm check

- A focused CUDA `layer_norm` regression now covers CLIP-base hidden shapes
  against Torch reference behavior at `(1, 4, 512)` and `(1, 50, 768)` with
  `eps=1e-5` and real affine-style float32 parameters.
- A separate cache-only local probe against
  `openai/clip-vit-base-patch32` using actual checkpoint LayerNorm weights and
  captured inputs from the text and vision towers stayed within about
  `2e-6` max absolute error on the checked sites, including
  `text_model.final_layer_norm`, `text_model.encoder.layers[0].layer_norm1`,
  `vision_model.pre_layrnorm`, `vision_model.encoder.layers[0].layer_norm1`,
  and `vision_model.post_layernorm`.
- For the current cached CLIP-base CUDA drift, LayerNorm is therefore unlikely
  to be the first culprit. Keep the next tower-focused investigation on the
  internal GEMM/BMM/attention/QuickGELU/Conv math rather than the final
  normalization or logit assembly, which already look clean.

## 2026-05-15 cached CLIP CUDA op-audit boundary

- A new opt-in cached-checkpoint CUDA op audit now walks a compact matrix of
  real `openai/clip-vit-base-patch32` tower op rows and stops at the first
  drifty family instead of compiling every remaining variant.
- The current boundary is clean through these checked rows: text
  `layer_norm1`, vision `pre_layrnorm`, vision patch `conv2d_bias`, text
  `q_proj` `gemm_rcr_bias`, and vision `q_proj` `gemm_rcr_bias`. Those rows all
  stayed within about `5e-6` max absolute error, with the patch projection
  matching exactly on the bounded audit input.
- The first drifty row was text-layer
  `gemm_rcr_bias_fast_gelu` at the cached checkpoint `fc1` site
  (`[1, 4, 512] -> [1, 4, 2048]`), which currently shows about `2.07e-2`
  max absolute error on CUDA. A side probe showed that magnitude matches the
  gap between CLIP QuickGELU (`x * sigmoid(1.702x)`) and Torch GELU on the same
  captured `fc1` activations, which strongly suggests the CUTLASS
  `fast_gelu` epilogue is not semantically matching the admitted DinoML
  `fast_gelu` contract for CLIP QuickGELU. The fix is a distinct
  `gemm_rcr_bias_quick_gelu` path, not changing `fast_gelu`.
- After that boundary is fixed, keep the next CUDA tower audit focused on
  proving the `gemm_rcr_bias_quick_gelu` row clean before widening to later
  BMM/softmax/helper rows.

## 2026-05-15 landed cached checkpoint adapter admission smoke

The Transformers-to-Legacy CLIP adapter now has one bounded opt-in proof for a
real cached checkpoint in `tests/test_clip_model_two_tower.py`.

- The smoke is skipped by default and only runs when
  `DINOML_RUN_CLIP_CHECKPOINT_ADAPTER_STATE_SMOKE=1` is set. It loads
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)` from the
  local cache, defaulting to `openai/clip-vit-large-patch14` with
  `DINOML_CLIP_CHECKPOINT_ID` override support, and skips clearly when the
  checkpoint files are absent instead of downloading anything.
- The proof is intentionally narrow and honest: it validates config adaptation,
  required state-dict import into the existing Legacy weight namespace, traces a
  fixed-shape `LegacyCLIPModel` using a tiny batch and short text sequence that
  still respect the checkpoint config, and then proves IR validation plus CUDA
  kernel-manifest/codegen-plan admission for the existing Conv/GEMM/BMM
  provider path.
- This smoke does not claim cached large-checkpoint runtime parity, tokenizer or
  processor plumbing, downloads, or any new op/provider surface. If future work
  wants runtime parity on a real checkpoint, that should land as a separate,
  explicitly heavier proof.

## 2026-05-15 landed Transformers checkpoint adapter slice

The bounded `LegacyCLIPModel` path can now be derived directly from a local
Transformers `CLIPModel` / `CLIPConfig` plus its `state_dict()`.

- The adapter stays intentionally narrow: `src/dinoml/models/clip.py` now
  exposes helpers that derive `LegacyCLIPTextConfig`,
  `LegacyCLIPVisionConfig`, and the existing DinoML CLIP weight namespace from
  a Transformers CLIP config/model without adding tokenizer/processor plumbing,
  position interpolation, loss, FlashAttention dispatch, or new op/provider
  surface.
- Admission remains honest about the current inference contract. The adapter
  rejects non-`quick_gelu` CLIP configs because the admitted MLP path now
  explicitly depends on the CLIP `quick_gelu` fused GEMM slice used by the
  local Transformers CLIP source.
- Focused validation now includes a deterministic tiny parity test that
  constructs a local Transformers `CLIPModel`, loads the existing tiny
  weights, builds the DinoML `LegacyCLIPModel` through the new adapter, and
  proves CPU reference parity for logits and normalized embeds. Adapter-specific
  coverage includes both the legacy `eos_token_id == 2` pooling branch and a
  non-2 EOS branch, plus a missing-state-dict-key rejection check for the
  exported weight converter.
- A skipped-by-default cache-only smoke is also in-tree for known checkpoints
  such as `openai/clip-vit-large-patch14`. Enable it with
  `DINOML_RUN_CLIP_CHECKPOINT_ADAPTER_STATE_SMOKE=1`, and optionally override
  the checkpoint id with `DINOML_CLIP_CHECKPOINT_ID=...`. The smoke uses
  `local_files_only=True` and validates cached config/state import plus trace
  and manifest/codegen admission only; it does not download or run full
  large-checkpoint runtime parity by default.

## 2026-05-15 landed cached base-checkpoint CPU runtime smoke

The adapter now also has one heavier but still bounded opt-in runtime proof for
the PM-refreshed cached `openai/clip-vit-base-patch32` checkpoint.

- The smoke is skipped by default and only runs when
  `DINOML_RUN_CLIP_CHECKPOINT_RUNTIME_SMOKE=1` is set. It loads
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)` from the
  local cache, defaults to `openai/clip-vit-base-patch32`, still honors
  `DINOML_CLIP_CHECKPOINT_ID=...`, and skips clearly instead of downloading when
  the checkpoint files are absent.
- The proof stays intentionally narrow: batch size 1, short traced text length
  `min(4, max_position_embeddings)`, synthetic already-shaped `pixel_values`,
  and CPU reference execution only. It traces the adapter-built
  `LegacyCLIPModel`, runs DinoML `execute_cpu`, runs the same cached
  Transformers checkpoint locally, and compares `logits_per_text`,
  `logits_per_image`, `text_embeds`, and `image_embeds`.
- This is a tractable cached runtime parity proof for the CPU reference path,
  not a claim about compiled CPU/CUDA runtime, tokenizer or processor
  plumbing, downloads, interpolation, loss, or broader checkpoint coverage.

## 2026-05-15 landed cached base-checkpoint compiled CPU smoke

The same cached `openai/clip-vit-base-patch32` checkpoint now also has a real
opt-in compiled CPU artifact proof.

- The smoke is skipped by default and only runs when
  `DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CPU_SMOKE=1` is set. It forces
  `HF_HOME=/workspace/.cache/huggingface`, still uses
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)`, keeps
  the same `DINOML_CLIP_CHECKPOINT_ID=...` override, and skips clearly instead
  of downloading when the checkpoint files are absent.
- The admitted proof stays intentionally heavy but bounded: batch size 1, short
  traced text length `min(4, max_position_embeddings)`, already-shaped
  synthetic `pixel_values`, adapter-built `LegacyCLIPModel`, CPU `.dinoml`
  compilation, `dinoml.runtime` session execution, and parity checks for
  `logits_per_text`, `logits_per_image`, `text_embeds`, and `image_embeds`
  against the same cached local Transformers checkpoint.
- In this bounded form the full two-tower base checkpoint is tractable on CPU:
  generated source stays finite, the artifact builds and loads, and runtime
  parity holds. Keep it opt-in and explicit about cost; this is not a default
  test, not a CUDA claim, and not broader checkpoint/model-family coverage.

## 2026-05-15 CUDA full-model blocker smoke

The next CUDA blocker after the exact patch-projection runtime is now explicit.

- Focused CUDA smoke coverage now proves the generated CLIP
  `get_text_features` and `get_image_features` artifacts each stay near local
  Transformers on CUDA, so the bounded float32 SIMT `cutlass_conv`
  patch-projection path is no longer the first failing edge in the real
  two-tower model workflow.
- The same smoke then compiles and runs the full two-tower CUDA artifact and
  shows the first remaining drift only after the contrastive head begins
  normalizing features and assembling logits: `text_embeds`,
  `image_embeds`, `logits_per_text`, and `logits_per_image` all diverge
  sharply from local Transformers.
- Treat that as the next bounded CUDA lane. Do not widen Conv claims or reopen
  tokenizer/processor, positional interpolation, FlashAttention, or other model
  surfaces while the contrastive-head CUDA runtime still fails this smoke.

## 2026-05-15 landed cached base-checkpoint compiled CUDA tractability smoke

The same cached `openai/clip-vit-base-patch32` checkpoint now also has a
bounded opt-in CUDA compiled-artifact tractability proof, but it is
intentionally not a parity claim.

- The smoke is skipped by default and only runs when
  `DINOML_RUN_CLIP_CHECKPOINT_COMPILED_CUDA_SMOKE=1` is set. It forces
  `HF_HOME=/workspace/.cache/huggingface`, keeps
  `transformers.CLIPModel.from_pretrained(..., local_files_only=True)`, reuses
  the shared CUDA support cache fixture for tractable runtime, and skips
  clearly when CUDA or the cached checkpoint is absent instead of downloading.
- The admitted proof stays intentionally narrow and honest: batch size 1, short
  traced text length `min(4, max_position_embeddings)`, synthetic already-shaped
  `pixel_values`, adapter-built `LegacyCLIPModel`, CUDA `.dinoml`
  compilation, `dinoml.runtime` session execution, generated-source and kernel
  manifest visibility for the expected Conv/GEMM/BMM providers, and finite
  output checks for `logits_per_text`, `logits_per_image`, `text_embeds`, and
  `image_embeds`.
- Current evidence on the refreshed local cache shows that the full base
  checkpoint is tractable to compile, load, and run on CUDA, but still drifts
  materially from both local Transformers and DinoML CPU on the bounded smoke
  inputs: `max_abs_diff ~= 0.7948` for both logits, `0.0295` for
  `text_embeds`, and `0.0739` for `image_embeds`. The opt-in smoke therefore
  holds only a loose non-parity envelope (`< 0.9` logits, `< 0.05` text embeds,
  `< 0.1` image embeds) so future loops can detect regressions while keeping
  the remaining CUDA numerical blocker explicit.

## 2026-05-15 cached base-checkpoint CUDA drift isolation boundary

The cached base checkpoint CUDA drift has been narrowed from "full model drifts"
to "standalone tower feature artifacts already carry the drift."

- A new opt-in helper test, gated by
  `DINOML_RUN_CLIP_CHECKPOINT_CUDA_DRIFT_ISOLATION=1`, traces separate cached
  checkpoint text-feature, image-feature, and full two-tower CUDA artifacts. It
  compares standalone tower outputs to local Transformers, recomposes normalized
  embeds/logits on the host from those standalone tower outputs, and checks that
  the full artifact's own logits are consistent with recomposition from its
  emitted embeds.
- PM artifact reuse from the completed probes showed the standalone tower
  outputs are already off: projected text features drift by about `0.7207` and
  projected image features by about `1.2052` before final CLIP normalization.
  Normalizing/recomposing those standalone tower outputs reproduces the full
  artifact drift (`~0.0294` text embeds, `~0.0738` image embeds, `~0.7948`
  logits).
- The full artifact is effectively identical to the standalone-tower
  recomposition at the final boundary: full-vs-tower normalized embeds are
  within about `6e-8`, full logits vs tower-composed logits within about
  `4e-6`, and full logits vs recomposition from full emitted embeds within
  about `2e-6`. That makes final normalization, scale, transpose, and logits
  assembly unlikely to be the first cached-checkpoint CUDA culprit. Next drift
  work should split the text and vision tower internals, especially GEMM/BMM,
  layer norm, attention masking, QuickGELU, and Conv/provider choices, instead
  of re-proving full-artifact compile/load/run.
