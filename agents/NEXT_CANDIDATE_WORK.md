# Next Candidate Work

This file should be updated after each major loop.

## Last Completed Loop

- Pinned the exact CLIP patch-projection CUDA runtime boundary. The
  Transformers-shaped float32 patch Conv used by `LegacyCLIPVisionEmbeddings`
  (`[B,3,4,4]` input, `[6,3,2,2]` weights, stride 2, padding 0, groups 1) now
  has focused coverage proving that the manifest selects a float32 SIMT
  `cutlass_conv` candidate with `manifest_scaffold_only` status and
  `cutlass_conv_runtime_launcher_not_implemented` as the explicit blocker. When
  CUDA tooling is available, the artifact compiles and then fails at the
  generated scaffold runtime boundary instead of disappearing into a vague
  provider gap.
- Pinned the current CLIPModel artifact blockers without widening provider
  surface. Focused tests now prove that full two-tower CPU compilation fails
  first at the existing `gemm_rcr_bias` compiled-CPU boundary, before the
  vision Conv path, and that CUDA manifest/codegen planning keeps the single
  CLIP Conv node artifact-visible as `cutlass_conv` with scaffold-only status
  plus explicit activation-pack, weight-pack, provider-launch, and output-unpack
  wrapper stages. This gives future runtime work a precise test-backed boundary
  to flip into an artifact/runtime smoke.
- Admitted zero-layer CLIP text parity after verifying local
  `/workspace/transformers` supports `num_hidden_layers=0` for
  `CLIPTextModelWithProjection` and `CLIPModel`. `LegacyCLIPTextConfig` now
  accepts non-negative text layer counts, zero-layer text wrapper tests cover
  both supported EOS pooling branches with explicit/default traced
  `position_ids`, and the two-tower suite includes a zero-text/zero-vision
  `LegacyCLIPModel` parity case. No tokenizer/processor plumbing, positional
  interpolation, FlashAttention, or provider claims were added.
- Proved multi-layer CLIP text parity without changing source. Focused tests
  now exercise a deterministic two-layer `LegacyCLIPTextModelWithProjection`
  against local `/workspace/transformers` for both supported EOS pooling
  branches and for both explicit and default traced `position_ids`. The tiny
  two-tower `LegacyCLIPModel` parity test now uses two text layers plus the
  already-admitted two-layer vision path, comparing helper features,
  normalized embeds, and logits against local Transformers while keeping
  tokenizer/processor plumbing, positional interpolation, FlashAttention, and
  new provider claims out of scope.
- Admitted stacked CLIP vision encoder blocks for the bounded vision wrapper.
  `LegacyCLIPVisionConfig` now accepts non-negative `num_hidden_layers`, and
  the wrapper reuses the existing dense noncausal attention + quick-gelu MLP
  block for each layer without adding new ops, FlashAttention, positional
  interpolation, tokenizer/processor plumbing, or provider claims. Focused
  tests pin zero-, one-, and two-layer `CLIPVisionModelWithProjection` outputs
  against local `/workspace/transformers`, upgrade the tiny two-tower
  `LegacyCLIPModel` parity test to two vision layers, and keep Conv represented
  honestly as a CUTLASS scaffold-only manifest entry.
- Added a compact runnable CLIPModel two-tower workflow proof. The new
  `examples/clip_model_workflow.py` traces the bounded `LegacyCLIPModel` on
  synthetic text/image tensors, runs the CPU reference path, prints projected
  text/image features, normalized embeds, logits, node/kernel ownership, and
  test-visible limits: no explicit `position_ids`, fixed square NCHW image
  shape, and CUTLASS Conv still represented as a scaffold-only manifest entry.
  Focused tests compare the example outputs against local `/workspace/transformers`
  `CLIPModel`, smoke the runnable script, and keep the existing two-tower parity
  tests green without adding tokenizer/processor plumbing, positional
  interpolation, FlashAttention, or new provider claims.
- Landed the first bounded CLIPModel-style two-tower contrastive workflow.
  `LegacyCLIPModel` now composes the admitted text tower and one-layer vision
  tower, exposes bounded `get_text_features` / `get_image_features`, normalizes
  projected features, applies `exp(logit_scale)`, and produces
  `logits_per_text` plus transposed `logits_per_image`. Focused tests pin
  projected features, normalized embeds, and both logits against local
  `/workspace/transformers` `CLIPModel` for a deterministic tiny config, while
  preserving provider/model manifest ownership. Remaining limits: static traced
  text length, default traced text positions, fixed square NCHW vision input,
  vision depth admitted only up to one layer, no tokenizer/processor plumbing,
  no positional interpolation, no loss path, and no compiled full-model CUDA
  runtime parity yet.

## Next Recommended Lane

- Keep converting the bounded CLIPModel surface toward usable artifacts and
  local Transformers parity with one concrete, test-backed gap at a time. Good
  next slices: close or narrow the compiled CPU GEMM-family blockers for the
  text side with a clearly scoped naive generated CPU bridge, or advance the
  exact CUDA Conv scaffold toward a CLIP-tied runtime smoke without broadening
  Conv claims. If staying purely in model parity, pick a new narrow
  Transformers gap that is not already covered by the layer-count proofs. Keep
  local `/workspace/transformers` parity as the acceptance bar and keep all
  non-parity limits explicit.
- If moving into runtime/provider work, tie it directly to a CLIP artifact test
  and keep the existing Conv limitations honest. Do not broaden tokenizer,
  processor, positional interpolation, FlashAttention, or Conv provider claims
  without a full admission slice.
- Human steering on 2026-05-15 allows a naive compiled CPU GEMM implementation
  as a temporary bridge. Do not treat the lack of a final CPU library/BLAS path
  as a blocker for CLIP artifact smoke work, but keep any naive CPU GEMM support
  small, measured by tests, and explicit about performance limits.
- Landed the first real CLIP vision encoder layer. The bounded vision wrapper
  now admits `num_hidden_layers` in `{0, 1}` and matches local Transformers for
  the one-layer surface: fixed-size embeddings, `pre_layrnorm`, dense noncausal
  self-attention, residuals, quick-gelu MLP, CLS pool, `post_layernorm`, and
  bias-free visual projection. Focused tests pin `last_hidden_state`,
  `pooler_output`, and `image_features` for both zero-layer and one-layer
  configs and verify provider/model ownership for Conv, GEMM/BMM, softmax,
  LayerNorm, and sequence assembly. Remaining limits: fixed square NCHW only,
  no positional interpolation, no arbitrary image sizes, no vision
  padding/causal mask path, no full `CLIPModel`, and CPU artifacts still stop
  at the existing `conv2d_bias` backend boundary.
- Landed a bounded zero-layer CLIP vision wrapper/projection slice. DinoML now
  matches local Transformers `CLIPVisionModelWithProjection` for the admitted
  zero-encoder-layer surface: fixed-size vision embeddings, `pre_layrnorm`, CLS
  pool, `post_layernorm`, and bias-free visual projection. Focused tests pin
  `last_hidden_state`, `pooler_output`, and projected `image_features` against
  local Transformers and keep CUDA provider/model ownership visible without
  broadening Conv runtime claims. Remaining limits: `num_hidden_layers == 0`
  only, fixed square NCHW inputs, no positional interpolation, no real vision
  encoder block, no full `CLIPModel`/processor plumbing, and CPU artifact
  compilation still stops at the existing `conv2d_bias` backend boundary.
- Landed the first bounded CLIP vision-side parity slice: fixed-size
  `LegacyCLIPVisionEmbeddings` now matches local Transformers
  `CLIPVisionEmbeddings` for semantic NCHW pixel input, bias-free patch
  projection modeled as `conv2d_bias(..., zero_bias)`, spatial flatten +
  sequence transpose, CLS prepend, and learned absolute position add. Focused
  tests compare full embeddings and the zero-bias patch-projection substep
  against local Transformers, keep CPU compile failure honest at the existing
  `conv2d_bias` backend boundary, and verify CUDA manifest/generated-source
  ownership without broadening Conv provider maturity. Remaining limits: fixed
  square image size only, no positional interpolation, no vision encoder or
  projection head, no full `CLIPVisionModel`, no widened Conv claims, and no
  compiled CPU artifact support for the Conv-backed wrapper path.
- Landed the bounded CLIP text-wrapper default-position slice. Callers may now
  omit `position_ids`; the wrapper falls back to a traced static int64
  `[0, 1, ..., S-1]` position sequence for the current static sequence length,
  matching Transformers CLIP default behavior while keeping the explicit
  `position_ids` path working. The visible CLIP text workflow example now omits
  `position_ids`, and focused tests prove both EOS pooling branches with and
  without explicit positions. Remaining limits: text-only wrapper, static
  traced sequence length, no tokenizer/processor plumbing, no vision tower, and
  default positions are traced constants rather than runtime-generated dynamic
  indices.
- Added a visible CLIP text workflow proof without adding new CLIP behavior,
  ops, providers, tokenizer/processor plumbing, FlashAttention, or expensive
  CUDA runtime requirements. `examples/clip_text_workflow.py` traces the
  current `LegacyCLIPTextModelWithProjection`, runs the CPU reference path, and
  prints JSON summarizing the node counts, EOS pooling branch, generated CUDA
  kernel coverage, and provider/model split. Focused tests prove both
  `eos_token_id == 2` and non-2 EOS branches, assert CUTLASS ownership for
  GEMM/BMM pieces, assert model-generated ownership for embedding/LayerNorm/
  softmax/pooling-side kernels, and smoke the runnable example script. Remaining
  limits: text-only proof, no compiled CPU wrapper support, explicit
  `position_ids`, static traced sequence length, no vision tower, and no full
  contrastive artifact workflow yet.
- Landed the bounded CLIP non-2 EOS pooling branch without adding a new pooling
  op, vision tower, tokenizer/processor plumbing, or FlashAttention/provider
  surface. `LegacyCLIPTextModelWithProjection` now matches both Transformers
  CLIP text pooling paths: legacy OpenAI configs with `eos_token_id == 2` keep
  the highest-token-id `argmax(input_ids)` compatibility path, while newer
  non-2 EOS configs use first-match `(input_ids == eos_token_id).argmax(...)`
  followed by the existing `batch_gather(...)->squeeze(...)` composition.
  Public `eq` admission was widened only for `int32`/`int64` inputs so token-id
  equality does not float-cast large integers; other relational ops remain
  float/reduced-precision-input only. Focused tests pin local Transformers
  parity for both EOS branches, generated CPU/CUDA source keeps integer storage
  for `eq`, and manifest checks keep provider/model ownership honest. Remaining
  limits: text-only wrapper, explicit `position_ids`, static traced sequence
  length, tokenizer-prepared EOS presence assumption, no vision tower, and no
  full contrastive wrapper artifact workflow yet.
- Replaced the GGUF CUDA runtime-dequant native boundary with a reproducible
  direct-link default: the repo now vendors `third_party/libgguf` as a pinned
  submodule from `https://github.com/hlky/libgguf`, CUDA module builds compile
  and link the native `libgguf_cuda_native` artifact from that submodule into
  GGUF runtime-dequant artifacts, generated lowering calls
  `libgguf_cuda_dequantize_rows_on_stream(...)` directly when linked, and
  artifact metadata/codegen plans record the linked native library plus source
  provenance and cache manifests. The runtime fallback no longer treats static
  archives as `ctypes.CDLL` candidates, cached native builds are invalidated on
  source/library provenance changes, and the old
  `dino_module_set_libgguf_cuda_dequantize_rows_on_stream()` path remains only
  as an explicit fallback/testing boundary when direct linking is unavailable or
  deliberately disabled. Focused planning/unit coverage plus CUDA-gated runtime
  regressions cover the direct-link default, forced fallback compatibility, and
  stale-cache/static-archive guardrails without widening GGUF policy, epilogue
  coverage, or public provider admission.
- Use worktrees for independent branches if running parallel agents. Keep
  feature write sets disjoint, keep shared queue/tracking doc reconciliation on
  the main line when possible, and require PM review plus validation before
  merge and push.
- Keep `cutlass_conv` bounded while tightening the now-static profiled path:
  add C=8 runtime parity if useful, decide whether dynamic Conv buckets/guarded
  dispatch need admission, and keep rejecting grouped/depthwise/transposed/3D,
  hidden padding, persistent packed weights, and public NHWC semantics until a
  separate design pass admits them.
- Landed a bounded CLIP text encoder-layer composition slice without adding
  `CLIPTextModel`, a new op, or a flash provider path: focused regressions now
  prove one tiny float32 text encoder layer as
  `layer_norm -> dense causal self-attention -> residual -> layer_norm ->
  gemm_rcr_bias_fast_gelu -> gemm_rcr_bias -> residual`, with CPU NumPy parity
  for both static additive causal masking and an optional bool padding mask.
  A light CUDA manifest check keeps provider ownership honest by showing
  `layer_norm`/`softmax` stay model-generated while GEMM/BMM pieces stay
  CUTLASS-backed. This is still a composition proof, not `CLIPTextModel`, not
  a dynamic causal-mask builder, and not a fused attention admission.
- Landed a bounded CLIP contrastive-head composition slice without adding a
  public head op: focused regressions now prove L2-normalized text/image
  features via `vector_norm(..., keepdim=True)` plus division, then
  `gemm_rcr(text_features, image_features.T)`, `exp(logit_scale)`, multiply,
  and transpose/permute orientation for `logits_per_image`. CPU NumPy parity
  covers unequal text/image batch sizes, and manifest checks keep `gemm_rcr`
  CUTLASS-backed while normalization and scalar math stay model-generated.
  This is still a composition proof, not `CLIPModel`, not encoder/projection
  coverage, and not a new public `contrastive_head` op.
- Landed the bounded CLIP text MLP / quick_gelu composition slice without
  adding a new public helper or model: focused regressions now prove
  `gemm_rcr_bias_fast_gelu` as the first projection and `gemm_rcr_bias` as the
  second projection for a tiny CLIP-style text MLP, with CPU NumPy parity and
  manifest/lowering checks confirming the first projection uses the CUTLASS
  `bias_fast_gelu` family while the second stays on `gemm_rcr_bias`. This is
  still a composition proof, not `CLIPTextModel`, not an encoder block, and not
  a new `quick_gelu` public op.
- Landed a bounded CLIP text dense-attention composition slice without adding a
  public attention op or FlashAttention/provider surface: focused regressions
  now prove a tiny static CLIP-style text self-attention path built from
  existing `gemm_rcr_bias`, static shape views, `permute0213`, rank-3
  `bmm_rcr`/`bmm_rrr`, scale multiply, static additive causal-mask constant,
  optional bool padding mask via `reshape`/`expand`/`where`, last-dim
  `softmax`, and output projection. CPU reference parity covers unpadded and
  padded cases, and CUDA manifest coverage keeps provider ownership honest by
  showing CUTLASS GEMM/BMM kernels remain provider-backed while softmax remains
  model-generated. This is still a static composition proof, not
  `CLIPTextModel`, not a dynamic mask builder, and not a fused attention
  admission.
- Landed a bounded CLIP text-embedding composition slice without adding any
  new public op: focused regressions now prove
  `token_embedding(input_ids) + position_embedding(position_ids)` for both
  rank-1 broadcast `position_ids [S]` and explicit batched `position_ids
  [B, S]`, with CPU NumPy parity plus generated CPU and CUDA kernel/runtime
  coverage on the existing embedding and fused-add contracts. The slice keeps
  the honest limits explicit: it does not add `CLIPTextModel`, it does not
  widen `arange`, and it relies on the current embedding/add composition rather
  than a CLIP-specific embedding op.
- Closed the bounded CLIP text-tower pooling slice without adding a new public
  pooling op: focused regressions now prove the legacy OpenAI CLIP
  highest-token-id path as
  `input_ids.argmax(dim=-1, keepdim=True) -> batch_gather(hidden_states, indices)
  -> squeeze(axis=1)` through CPU reference, generated CPU artifact runtime,
  and CUDA source/runtime smoke. The slice keeps the honest limits explicit:
  it only proves the legacy highest-token-id pooling composition and still does
  not solve non-2 EOS equality matching or the broader text-tower
  attention/masking path.
- Landed the next narrow CLIP text-tower blocker without broadening general
  integer tensor support: public/generated `dml.ops.argmax` now admits
  `int32`/`int64` input tensors alongside the existing
  `float32`/`float16`/`bfloat16`/`bool` surface, while preserving the same
  static-shape, last-dim-only, `keepdim`, and `int64` output contract.
  Generated CPU/CUDA lowering now compares `int32`/`int64` values as integers
  instead of casting through float, while float inputs keep the existing
  fp32-plus-NaN behavior. Focused regressions pin frontend/IR admission, CPU
  reference behavior, generated CPU/CUDA source, CPU artifact runtime, the
  compiler/runtime dtype exception boundary, and a legacy OpenAI CLIP-style
  `input_ids.argmax(dim=-1)` EOT pooling case with first-index tie semantics.
  Keep docs honest: this unblocks only the legacy highest-token-id CLIP EOT
  pooling step and does not solve non-2 EOS equality matching or the full text
  pooling gather flow on its own.
- Landed the next bounded CLIP/BERT text-enabling primitive as a real generated
  lookup op instead of another helper composition: public
  `dml.ops.embedding(table, indices)` is now a registered op with dedicated
  validation, CPU reference execution, and generated CPU/CUDA lowering for a
  positive static table `[vocab, hidden]`, `float32`/`float16`/`bfloat16`
  table storage, `int64`/`int32` indices, output dtype matching the table, and
  output shape `indices.shape + [hidden]`. The landed slice preserves dynamic
  leading index dims while keeping the table static, emits explicit CPU/CUDA
  runtime output-size checks plus out-of-bounds index rejection, and adds
  focused regressions for frontend/IR shape-spec propagation, int32/int64
  index support, validation failures, generated-source/kernel-manifest
  ownership, dynamic-batch CPU artifact runtime, CUDA compile/runtime parity,
  and CPU runtime OOB rejection.
- Landed the first real affine LayerNorm primitive needed for the CLIP/ViT/BERT
  first-model sprint without widening beyond the bounded static-hidden slice:
  public `dml.ops.layer_norm(x, weight, bias, eps=...)` is now a registered op
  with dedicated validation plus generated CPU/CUDA lowering rather than a
  helper composition, preserving dynamic leading dims while requiring a
  positive static last dimension and matching rank-1 affine tensors
  `[hidden]`. Focused regressions now cover traced/lowered IR ownership, shape
  and dtype validation, kernel-manifest/generated-source provenance, CPU
  artifact runtime across dynamic leading dims, CUDA compile/runtime parity,
  and reduced-precision (`float16`/`bfloat16`) fp32-accumulation behavior.
- Closed the remaining bounded CUDA runtime-validation gap around
  `get_1d_rotary_pos_embed` tensor-position dynamics without widening the op
  surface: focused regressions now compile one dynamic `float32` CUDA artifact
  with rank-1 tensor positions and prove the generated cos/sin component kernels
  run correctly across multiple runtime sequence lengths through the NumPy
  staging path, while preserving the existing table-generation-only contract,
  mixed-variant provenance coverage, and no-input integer-position runtime
  coverage.
- Hardened the bounded `get_timestep_embedding` runtime contract without
  widening the op surface: added a focused CUDA dynamic-shape artifact
  regression that compiles one `float32` artifact with dynamic timestep length
  `N` and proves the generated kernel runs correctly across multiple runtime
  lengths while preserving the existing single-op lowering and in-kernel
  sinusoidal math contract. This closes the remaining gap between the documented
  dynamic-`N` claim and runtime validation, which previously existed only on
  the CPU artifact path.
- Hardened the already-registered named permute specialization surface without
  widening its contract: focused regressions now prove CPU artifact runtime
  execution for `permute021`, `permute0213`, `permute102`, and `permute210`
  across the admitted `float32`, `float16`, `bfloat16`, and `bool` storage
  surface, add reduced-precision/bool CUDA generated-source checks for named
  kernels, and exercise CUDA runtime parity for the named float32
  specializations when CUDA is available. The slice stays honest about using
  the existing generated dense permute-copy strategy rather than claiming v1
  tiled/coalesced kernel parity.
- Hardened the bounded helper-only `rms_norm` contract without widening the
  op surface: helper-level regressions now prove that both weighted and
  unweighted `dml.ops.rms_norm(...)` inherit the admitted `t5_layer_norm`
  runtime behavior instead of only claiming it in docs. Added focused coverage
  for dynamic-leading-dimension CPU artifact execution and reduced-precision
  (`float16`/`bfloat16`) CUDA runtime parity with fp32 accumulation, while
  preserving the helper-only lowering contract that still emits only
  `t5_layer_norm` nodes/kernels plus the synthetic ones constant for the
  weightless path.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` wrapper-source lane
  without weakening the current compile rejection: rejected CUDA artifacts now
  emit `debug/generated_src/scaffold_source_manifest.json` plus a guarded
  scaffold-only `.cu` wrapper snippet for each Conv wrapper-stage group, and
  `kernel_codegen_plan.json` links those emitted sources back to the recorded
  activation-pack, weight-pack, provider-launch, and output-unpack stage
  sequence. Focused tests now pin the stage-to-source linkage, emitted file
  path, guarded `#if 0` snippet shape, and artifact-side manifest wiring while
  CUDA compile still rejects before module build with the existing
  `manifest/codegen scaffold only` boundary.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` wrapper-metadata lane
  without weakening the current compile rejection: `kernel_codegen_plan.json`
  now records explicit per-node wrapper stages for activation NCHW -> NHWC
  pack, OIHW -> OHWI weight pack, planned provider launch, and NHWC -> NCHW
  output unpack, all derived from the validated `cutlass_conv_plan`
  temporary/layout contract and linked to the selected helper or launcher
  symbols plus static shape/attr call arguments. Added a small source-render
  helper that turns those stage entries into future CUDA wrapper call snippets,
  with focused tests proving the stage order, temporary-buffer usage, helper
  symbol wiring, and rendered call shapes while CUDA compile still stops at the
  existing `manifest/codegen scaffold only` boundary before module build.
- Closed the remaining native-boundary regression gap around the bounded GGUF
  runtime-dequant CUDA slice without widening policy: direct native
  `dino_module_load()` now has CUDA-gated coverage for a mixed dense-bias plus
  encoded GGUF RHS `gemm_rrr_bias` artifact, proving that native module load
  autoloads only the dense bias, native encoded-weight installation plus the
  explicit `dino_module_set_libgguf_cuda_dequantize_rows_on_stream()` hook make
  the lowered runtime-dequant path runnable, native
  `dino_module_unload_constants()` / `dino_module_load_constants()` preserve the
  installed dequantizer pointer while requiring only the encoded weight to be
  reinstalled, and a freshly reopened native module handle again requires
  reinstalling the dequantizer hook before the same encoded bytes can run.
- Closed the skeptical-reviewer follow-ups on the bounded GGUF
  runtime-dequant scratch-resource slice without widening policy: support
  library cache keys no longer churn from generated-module/runtime
  `session_resources`, while the full manifest `cache_key` still tracks that
  runtime allocation metadata; CUDA lowering now has an explicit legacy-manifest
  regression proving it falls back to scanning lowered
  `gguf_runtime_dequant` plans when top-level `session_resources` is absent;
  and the shared-scratch claim now has CUDA-gated runtime coverage with two
  GGUF RHS GEMM nodes sharing one max-sized session scratch allocation and
  matching dense dequantized references.
- Made the bounded GGUF runtime-dequant -> CUTLASS GEMM scratch policy more
  artifact-visible without widening the runtime surface: `kernel_manifest.json`
  now records a `session_resources` entry for the shared per-session
  `gguf_runtime_dequant_scratch` CUDA allocation, sized to the maximum dense RHS
  requirement across all lowered `gguf_runtime_dequant` GEMM plans and linked
  back to the source node/constant scratch plans. CUDA lowering now consumes
  that manifest resource when allocating the session-owned scratch buffer while
  retaining the existing lowered-plan fallback for older manifests. Focused
  planning/codegen coverage pins the max-sized shared allocation and its source
  plan provenance; encoded-load plan regressions remain green. No new GGUF
  materialization policy, offload scheduler, op surface, or non-bias GEMM
  epilogue support was added.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` support-library lane by
  compiling the next honest prerequisite for a real launcher without widening
  the runtime claim: the CUTLASS Conv scaffold now emits exported CUDA layout
  transform helpers for the manifest-recorded NCHW -> NHWC activation pack,
  OIHW -> OHWI weight pack, and NHWC -> NCHW output unpack contract, with
  helper symbols threaded back into `cutlass_conv_plan`
  `layout_translation`/`weight_transform` metadata for both `float16` and
  `float32`. The support manifest, source manifest, and codegen support-library
  metadata now all expose those helper exports/symbols explicitly, and focused
  tests prove the scaffold compiles, exports the helper ABI coherently, still
  preserves the launcher/profiler stub contract, and optionally matches Torch
  layout permutations on real CUDA for both supported dtypes. CUDA model
  compile still rejects before final manifest/module build, so no generated
  wrapper lowering, CUTLASS implicit-GEMM conv launch, profiler execution, or
  `conv2d_bias` model runtime is claimed yet.
- Advanced the bounded `conv2d_bias`/`cutlass_conv` runtime-maturity lane
  without enabling a model runtime claim: the support-cache scaffold now renders
  concrete launcher/profiler stub exports for the planned
  `dinoml_cutlass_conv2d_bias_v1` ABI and, when `nvcc` is available, compiles
  them into `lib/libdinoml_cutlass_conv.so` with `compiled_stub_only` status,
  library hash, build command, source-manifest symbols, and explicit export
  metadata. CUDA model compile still rejects before final manifest/module build
  while the kernel manifest remains `manifest_scaffold_only`, so no generated
  pack/unpack lowering, CUTLASS implicit-GEMM launcher, profiler execution, or
  CUDA runtime parity is claimed. Focused conv tests now prove the compiled
  support stub exists, exports the expected symbols, returns the documented
  unsupported status, preserves NHWC/OHWI transform provenance, and still
  rejects module compile honestly.
- Started the `src/dinoml/ops/__init__.py` decomposition without widening the
  op surface: public `dml.ops.where(...)` now lives in
  `src/dinoml/ops/where.py`, while the small broadcast shape-spec inference
  helper it depends on moved into private `src/dinoml/ops/_frontend_utils.py`
  so the registry/export wiring in `dml.ops` can keep overriding the generic
  registered frontend with the bespoke `where` contract. Focused `where`
  frontend, CPU reference, generated CPU/CUDA source, and runtime smoke tests
  continue to cover the public API, and no broader op extraction or checklist
  churn was introduced in this structural slice.
- Repaired the named permute specialization surface so it is now honest instead
  of alias-shaped. Public `dml.ops.permute021`, `permute0213`, `permute102`,
  and `permute210` are now real registered bounded ops with fixed-rank,
  fixed-dims contracts; traced and lowered IR preserve those node names instead
  of silently rewriting them to generic `permute`; and generated CPU/CUDA
  lowering, `kernel_manifest.json`, and generated-source provenance now carry
  op-specific symbols/function names for the specialized nodes. This slice
  intentionally reuses the existing generated dense permute-copy strategy with
  compile-time dims/strides, so it is truthful about being a bounded generated
  specialization rather than v1 tiled kernel parity. Focused regressions now
  cover specialized frontend/IR emission, registry default-attr/schema
  coherence for the fixed `dims` contract, fixed-dims validation against attr
  drift, CPU reference parity, artifact-level CPU manifest/source-manifest
  provenance, and optional CUDA compile coverage for a representative named
  specialization.
- Closed the reviewer follow-ups on the just-landed generated
  `get_1d_rotary_pos_embed` slice without widening the op surface. Model-owned
  generated-kernel provenance is now artifact-visible enough to distinguish
  mixed rotary variants in one graph: `build_kernel_manifest()` no longer
  collapses distinct component variants solely by the shared
  `generated_get_1d_rotary_pos_embed` symbol, and model kernels now carry
  generated function/source provenance through `kernel_manifest.json` and
  `kernel_codegen_plan.json`. The rotary component registry contract is also
  now truthful about input arity by admitting exactly zero-or-one inputs rather
  than pretending to accept arbitrary variadic counts. Focused regressions now
  cover mixed int-pos plus tensor-pos variants in one artifact, artifact-level
  no-input integer-pos CPU runtime execution, optional no-input CUDA runtime
  execution through `run_numpy`, and the zero-or-one `accepts_input_count`
  contract on the internal component ops.
- Finished the half-landed `get_timestep_embedding` slice as a real registered
  generated op instead of a helper composition. Public
  `dml.ops.get_timestep_embedding(...)` now lives in `OP_REGISTRY`, traced IR
  and lowered IR carry a single `get_timestep_embedding` node, dynamic rank-1
  `N` is preserved through output shape-spec propagation, and generated CPU and
  CUDA kernels compute the full sinusoidal table in one op/kernel with fp32
  internal math, output-dtype preservation for `float32`/`float16`/`bfloat16`,
  odd-width zero padding, and `flip_sin_to_cos`. Focused tests now cover the
  registered frontend/IR contract, generated-source/kernel-manifest ownership,
  CPU formula parity, dynamic-`N` CPU artifact execution, and CUDA compile plus
  runtime parity.
- Hardened the bounded ConvNd/CUTLASS scaffold contract around its
  artifact-visible layout/weight transform metadata. The shared
  `cutlass_conv_plan` now validates its own NCHW/OIHW -> NHWC/OHWI semantics,
  dtype/shape-derived temporary sizes, padded-channel bookkeeping, and
  temporary-buffer inventory before profiling, codegen-plan generation, or the
  support-cache/source-manifest scaffold consume it. Candidate metadata must
  also agree with the recorded semantic/provider layouts, so manifest drift now
  fails explicitly instead of propagating incoherent provenance into workload
  JSON or support manifests. The Conv support scaffold now also revalidates and
  normalizes each caller-supplied used-plan entry before persisting
  `cutlass_conv_manifest.json` or `source_manifest.json`: it re-derives the
  selected scaffold candidate from the entry candidate list, validates
  candidate-set provenance, carries `node_id` when present, and rejects direct
  caller mutations to selected-candidate layout/dtype metadata before any
  support-manifest payload is written or trusted. Added focused regressions
  that prove profiling rejects malformed transform byte counts, codegen/support
  provenance rejects candidate-layout drift against the recorded transform plan,
  and direct mutated used-plan payloads fail before manifest writes.
- Finished the half-landed `get_1d_rotary_pos_embed` surface as a bounded
  generated-op slice instead of helper math composition. Public
  `dml.ops.get_1d_rotary_pos_embed(...)` still returns a `(cos, sin)` tuple,
  but current v2 IR/runtime remain single-output per node, so the public API
  now lowers explicitly to two generated component ops,
  `get_1d_rotary_pos_embed_cos` and `get_1d_rotary_pos_embed_sin`, rather than
  claiming full v1 single-launch/two-output-kernel parity. The admitted
  contract is now explicit and tested: positive even static `dim`; `pos` as a
  positive integer sequence length or rank-1 dense
  `float32`/`float16`/`bfloat16` tensor with positive static or dynamic length;
  positive finite `theta`/`linear_factor`/`ntk_factor`; `use_real=True`
  duplicated-real outputs with both duplication conventions; `use_real=False`
  base cos/sin outputs of shape `[S, dim/2]`; and float16/float32/bfloat16
  output storage with fp32 internal math from float32 positions. Integer `pos`
  now lowers directly as two no-input generated component nodes with a static
  `sequence_length` attr instead of adding an `arange` launch. Focused tests
  now pin the two-component-node IR/lowered-IR contract, generated
  source/manifest ownership, CPU formula parity, dynamic-`S` CPU artifact
  execution, CUDA compile coverage for both real/base modes, and CUDA runtime
  parity for one `use_real=False` float32 case plus reduced-precision real
  outputs.
- Captured the recent RoPE/apply-rotary exploration as durable project memory
  before any implementation starts. Added `agents/plans/rotary_apply_plan.md`
  to pin the old `/workspace/apply_rotary_emb` prototype's real ABI and limits:
  CUDA-only Torch extension, coupled Q/K pair contract, contiguous rank-4
  `[B,H,S,D]` inputs plus rank-2 `[S,D]` cos/sin tables, effective float32-only
  behavior through raw `data_ptr<float>()`, real-pair layout switching via
  `use_real_unbind_dim`, two-tensor return reality, and missing validation for
  dtype/shape compatibility and honest complex support. The new plan also
  records the broader variant taxonomy across v1/diffusers/transformers
  (split-half, interleaved, partial-prefix, complex, multi-axis, scaled-table,
  rotate-V, and cache-order-sensitive families), separates table generation
  from application kernels, and recommends the first bounded v2 slice as
  `get_1d_rotary_pos_embed` table generation with duplicated-real variants
  rather than a fused public `apply_rotary_emb` ABI.
- Landed the bounded weight-optional normalization helper slice:
  public `dml.ops.rms_norm(x, weight=None, eps=1e-6)` now stays helper-only and
  reuses the existing `t5_layer_norm` generated CPU/CUDA backend instead of
  adding a new op, provider, or kernel family. The weighted path delegates
  directly to `t5_layer_norm`, while the weightless path materializes a
  same-dtype static ones vector `[hidden]` and delegates to that same node, so
  lowered IR still contains only `t5_layer_norm`. Added focused regressions
  proving the helper stays out of `OP_REGISTRY`, that the weighted IR is the
  same single `t5_layer_norm` node, that the unweighted IR adds only a ones
  constant plus `t5_layer_norm`, that CPU parity holds for weighted/unweighted
  `float32`/`float16`/`bfloat16`, that CUDA runtime parity works for one
  weighted and one unweighted `float32` case, and that dynamic hidden size,
  bad rank/dtype, weight shape mismatch, and mixed builder/dtype contracts fail
  clearly without widening the normalization/provider surface.
- Advanced the bounded ConvNd/CUTLASS maturity lane by turning the existing
  `cutlass_conv` manifest/codegen scaffold into a manifest-only support-cache
  scaffold: CUDA compile still rejects before module build, but it now writes
  `lib/cutlass_conv_manifest.json` and `src/source_manifest.json` under the
  advertised support `cache_dir`, carrying the used candidate plan, candidate
  config keys, and explicit NCHW/OIHW -> NHWC/OHWI transform provenance. Added
  focused regressions that prove the Conv used-candidate plan now preserves the
  scaffold candidate payloads and that a failing CUDA compile still materializes
  the support scaffold before the expected `manifest_scaffold_only` rejection.
- Landed the smallest honest v1/HuggingFace custom-op helper slice around
  `gelu_new`: public `dml.ops.gelu_new(x)` is now a bounded frontend helper
  that rewrites directly to the existing tanh-approximation `gelu` op instead
  of expanding provider or kernel surface. Focused tests now pin that the
  traced IR is identical to `gelu`, the lowered path stays on
  `fused_elementwise`, CPU reference execution matches the HuggingFace/v1 tanh
  GELU-new formula, CUDA codegen uses the existing `dinoml::math::gelu` path,
  and helper-only admission stays honest by keeping `gelu_new` out of
  `OP_REGISTRY` while delegating unsupported dtype rejection to `gelu`.
- Closed the reviewer-noted reduced-precision CUDA runtime gap for the bounded
  `t5_layer_norm` slice: `float16` and `bfloat16` now have a numeric CUDA
  runtime parity regression in `tests/test_t5_layer_norm_ops.py`, using the
  existing trace/reference helpers and the generated CUDA artifact path. The
  source-side fp32-accumulation assertions stayed in place, so the bounded
  T5/RMSNorm contract remains unchanged while the reduced-precision runtime
  path is now exercised directly.
- Landed the first bounded normalization slice away from the Conv metadata
  lane: public `t5_layer_norm` now covers the T5/RMSNorm-style form
  `x * rsqrt(mean(x^2) + eps) * weight` over rank >= 1 dense tensors with a
  positive static last dimension and required affine weight `[hidden]`, across
  `float32`, `float16`, and `bfloat16` storage. The slice keeps fp32
  accumulation semantics, preserves dynamic leading-dimension shape metadata,
  adds CPU reference execution plus generated CPU/CUDA kernels, and has focused
  frontend/IR rejection coverage for dynamic hidden size and bad weight
  contracts. The docs/checklist now call this out explicitly as a bounded
  RMS/T5-only slice; full LayerNorm, grouped, sigmoid-mul, adaptive, and
  provider-backed normalization variants remain unimplemented.
- Closed a reviewer-found P1 in the bounded `cutlass_conv` profiling scaffold:
  scaffold-only `ConvProfileWorkload` objects now fail explicitly before the
  GEMM/BMM-only profiling cache-key, profile-result, cache read/write, or
  execution-plan code can touch them. `profile_artifact(...)` also rejects
  unsupported Conv scaffold workloads at the profiling boundary for future
  safety, and focused profiling regressions pin the new error contract so
  scaffold-only Conv results cannot silently disappear from execution-plan
  generation.
- Connected the existing bounded `conv2d_bias`/`cutlass_conv` scaffold to the
  first profile-visible provider step without adding a runtime launcher:
  `build_profile_workloads(...)` now emits a `cutlass_conv` workload scaffold
  from the manifest's explicit NCHW/OIHW semantic metadata, NHWC/OHWI provider
  metadata, layout-pack/unpack plan, weight-transform metadata, Conv2d attrs,
  shapes, candidates, and profiler symbol. The scaffold refuses manifests that
  omit `cutlass_conv_plan` transform metadata, preserving the artifact-visible
  layout contract. Added focused tests for the emitted workload JSON and the
  missing-transform guard. CUDA compile still rejects before module build with
  `manifest_scaffold_only`; no CUTLASS Conv runtime, support build, or profiler
  execution is claimed.
- Started the bounded ConvNd provider lane without claiming a CUDA runtime yet:
  added a public/reference-only `conv2d_bias` surface with NCHW activation,
  OIHW weight, bias `[Cout]`, groups=`1`, static rank-4/static channel+kernel
  limits, and CPU reference parity against PyTorch. CUDA compile now reaches a
  `cutlass_conv` manifest/codegen scaffold that records the intended NHWC/OHWI
  provider layout and explicit layout/weight-transform metadata as
  `manifest_scaffold_only`, then rejects before module build until a real
  launcher exists; CPU compile still rejects at backend admission.
- Tightened CUTLASS support-cache/source-manifest reuse with another bounded
  compile-visible robustness slice: cache hits now also reject
  `src/source_manifest.json` payloads whose embedded `used_candidate_plan`
  content no longer hashes to the stored `used_candidate_plan_key`, even when
  the top-level manifest key remains self-consistent. Added a focused
  backend-registry regression in the existing source-manifest test area that
  mutates the embedded selected-candidate payload, recomputes the outer
  `source_manifest_key`, and proves the support library rebuilds instead of
  reusing stale provider provenance.
- Tightened the CUTLASS profile-cache persistence contract with a small
  compile-visible robustness slice: cache reads now reject entries whose
  embedded key payload no longer hashes to the stored `profile_key` or whose
  embedded target drifts from the cache target, and cache writes now drop those
  stale on-disk payloads instead of merging them back. Added focused profiling
  regressions for stale embedded payload hashes, cross-target payload drift, and
  stale-on-disk entry rejection during a normal write.
- Closed the top-ranked trust-building GGUF/CUDA workflow gap with a focused
  float32 `gemm_rrr_bias` runtime regression: real libgguf `Q4_0` RHS storage,
  dense bias loaded from `constants.bin`, `manual_runtime_load` encoded weight,
  load -> run -> unload -> reload plus reopen, and dense reference comparisons
  across each successful execution. Updated the gap audit to describe the
  proven dense-bias lifecycle slice instead of only one-shot correctness.
- Closed the reviewer follow-up on native manual GGUF autoload parity: added a
  bounded direct CUDA native-boundary regression for mixed dense plus
  `manual_runtime_load` constants that mirrors the CPU ABI test shape. The test
  now calls generated native CUDA `dino_module_load()`,
  `dino_module_load_constants()`, `dino_module_set_constant()`, and
  `dino_session_run()` directly enough to prove `constants.bin` does not eagerly
  materialize the manual GGUF constant, and that after native unload/reload the
  module still requires an explicit native load/set before run. Updated the gap
  audit to describe the proven CPU/CUDA native coverage precisely.
- Closed the remaining native GGUF load-path parity gap for mixed dense plus
  `manual_runtime_load` constants: generated CPU/CUDA native
  `dino_module_load_constants()` now skips eager `constants.bin`
  materialization for any GGUF constant that declares
  `residency="manual_runtime_load"`, matching the Python open/reload contract
  instead of only the lowered CUDA runtime-dequant slice. Added direct CPU
  native-boundary coverage proving `dino_module_load()` and
  `dino_module_load_constants()` leave the manual GGUF constant unloaded across
  eager open and reload until an explicit setter call, plus mixed CPU/CUDA
  generated-source regressions that pin the skip path.
- Tightened the bounded GGUF runtime-dequant -> CUTLASS GEMM contract so only
  `residency="manual_runtime_load"` produces
  `lowered_runtime_dequant_scratch`: manifest planning now marks non-manual
  residency as `planned_not_lowered` with a clear residency-specific blocked
  reason, compile admission now fails with an explicit manual-residency
  requirement, and generated CUDA native load paths no longer eagerly materialize
  lowered encoded runtime-dequant constants from `constants.bin`. Added focused
  planning/generated-code coverage for the native eager-load skip plus a
  non-manual-residency planning/admission regression. Follow-up to keep in view:
  audit whether the non-Python native `dino_module_load_constants()` path should
  also honor `manual_runtime_load` for older dense GGUF policies, since this
  loop intentionally fixed the encoded runtime-dequant slice only.
- Closed the shared-scratch coverage gap for bounded GGUF runtime-dequant
  CUTLASS GEMMs: added a focused planning/codegen regression proving that
  multiple lowered runtime-dequant GEMM nodes in one CUDA artifact share a
  single session-owned scratch allocation sized to the maximum dense RHS while
  each launch still checks its own required scratch bytes before native
  libgguf dequant.
- Closed the reviewer follow-up gap in the bounded GGUF runtime-dequant CUDA
  coverage: added a focused `gemm_rcr_bias` float16 integration regression
  that uses real libgguf `Q4_0` RHS storage, dense bias, same-stream native
  dequant, and a dense reference comparison. This keeps the support surface
  unchanged while proving the bias + float16 runtime slice directly on CUDA.
- Extended the bounded CUDA GGUF runtime-dequant-before-GEMM path from base
  GEMM to the bias-only epilogue slice: manifests now lower RHS GGUF constants
  with `materialization="dequantize_on_gpu_before_launch"` and
  `residency="manual_runtime_load"` for `gemm_rrr_bias`/`gemm_rcr_bias`
  `float32`/`float16` outputs, compile/runtime admission accepts those bias
  uses alongside the base `gemm_rrr`/`gemm_rcr` path, and generated CUDA reuses
  the same-stream native libgguf dequant into session-owned dense RHS scratch
  before the existing dense CUTLASS bias launcher. Added planning/admission/
  lowering coverage for both layouts and a real CUDA integration test using
  libgguf `Q4_0` RHS storage plus dense bias compared against a dense reference.
- Extended the bounded CUDA GGUF runtime-dequant-before-GEMM path from base
  `gemm_rrr` to base `gemm_rcr`: manifests now lower RHS GGUF constants with
  `materialization="dequantize_on_gpu_before_launch"` and
  `residency="manual_runtime_load"` for `float32`/`float16` outputs, admission
  accepts only base `gemm_rrr`/`gemm_rcr` RHS uses, and generated CUDA reuses
  same-stream native libgguf dequant into session-owned dense RHS scratch. The
  dense CUTLASS RCR launcher consumes that scratch through the existing
  column-major RHS ABI. Added planning/codegen/admission coverage plus a CUDA
  integration test using real libgguf `Q4_0` `gemm_rcr` RHS storage compared
  against a dense dequantized reference, plus a float16 CUDA runtime regression
  that covers encoded load, runtime dequant, and CUTLASS RCR handoff with a
  reduced-precision tolerance.
- Added focused CUDA allocator/session lifecycle regression for the missing
  `_cuda_runtime_dll` cleanup path: when the CUDA helper handle is absent during
  session cleanup, staged buffers are cleared and the session teardown still
  proceeds.
- Added focused CUDA allocator/session lifecycle regressions around the
  remaining cleanup/retry edge cases: a failed staging-buffer grow now has a
  regression proving the newly allocated buffer is rolled back when the old
  buffer free fails, and `Session.close()` now has a regression proving that a
  cleanup failure followed by a native destroy failure still leaves the session
  retryable until both paths succeed on a later close.
- Added CUDA reopen-parity lifecycle coverage for mixed dense and
  manual-runtime-load encoded constants, matching the CPU regression slice:
  reopening an eager artifact resets the encoded constant back to unloaded, and
  closing a deferred artifact with a live session restores both constants to
  their initial deferred residency state instead of leaking prior runtime loads
  across module instances.
- Added broader CPU runtime/container lifecycle coverage for mixed dense and
  manual-runtime-load encoded constants: reloading still requires an explicit
  encoded load after `unload_constants()`/`load_constants_from_file()`, closing
  and reopening an eager artifact resets the manual residency bit back to
  unloaded, and closing a deferred artifact with a live session restores the
  initial deferred residency state on reopen instead of leaking prior runtime
  loads across module instances.
- Added focused CUDA runtime-dequant coverage for remaining native-launcher failure modes on
  bounded GGUF `gemm_rrr`/`gemm_rcr` paths: `load_encoded_constants(["weight"])`
  now has a regression test proving that a missing
  `libgguf_cuda_dequantize_rows_on_stream` symbol fails before encoded bytes are
  installed and leaves `constant_load_state()` untouched, and `session.run_*`
  now has a regression test proving that clearing the module's launcher pointer
  after encoded load fails with the generated missing-launcher error instead of
  falling back to dense dequant.
- Added focused CUDA lifecycle coverage for the bounded GGUF runtime-dequant
  `gemm_rrr` path: unload now explicitly invalidates the encoded RHS residency
  for the live session, reloading encoded constants restores execution, closing
  a runtime-dequant module closes the live session before freeing the module,
  re-opening the artifact starts with unloaded encoded residency again, and
  repeated session/module close calls stay idempotent without stale loaded
  state.
- Added focused CUDA runtime-dequant test coverage for malformed GGUF encoded
  metadata on `load_encoded_constants(...)`: mismatched qtype,
  `encoded_nbytes`, and `n_per_row` now have explicit regression tests proving
  the runtime rejects the load before installing encoded bytes or mutating the
  per-constant loaded-state snapshot.
- Tightened the bounded GGUF runtime-dequant slice so
  `materialization="dequantize_on_gpu_before_launch"` is admitted only for the
  lowered CUDA `gemm_rrr` RHS path. Unsupported uses now fail clearly at
  compile/runtime admission instead of being reported as runtime-loadable
  encoded constants, and runtime load plans include precise blocked reasons.
- Cached native libgguf CUDA dequant launcher lookup by extension path so
  repeated encoded-constant loads do not reopen a new `ctypes.CDLL` handle.
- Updated README and architecture docs to describe the current narrow CUDA
  runtime-dequant contract and the still-unsupported surface.
- Landed the first bounded runnable GGUF dequantize-before-GEMM path for
  CUTLASS `gemm_rrr` with a GGUF RHS constant declared as
  `materialization="dequantize_on_gpu_before_launch"` and
  `residency="manual_runtime_load"`.
- Generated CUDA now stores that constant as encoded bytes, exposes an explicit
  runtime-set `libgguf_cuda_dequantize_rows_on_stream` boundary, allocates a
  separate session-owned dense RHS scratch buffer, dequantizes on the same
  session stream immediately before the existing dense CUTLASS GEMM launch, and
  fails precisely when the native launcher is unavailable.
- Runtime encoded-constant loading now has a CUDA branch for this policy that
  installs encoded GGUF bytes into generated module storage while preserving the
  older `dequantize_full_before_launch` dense load-time path.
- Added generated-code/lowering coverage for scratch allocation, encoded
  constant storage, native dequant call ordering, and missing-launcher failure,
  plus a focused CUDA integration test using real libgguf `Q4_0` RHS storage
  compared against a dense dequantized GEMM reference.

## Ranked Backlog

1. Keep the small/custom-op lane on honest helper or bounded-op slices:
   with `gelu_new`, the now-registered generated `get_timestep_embedding`, the
   completed bounded `get_1d_rotary_pos_embed` component-op slice, and the
   newly runtime-hardened helper-only `rms_norm` slice in place, and the now-
   registered generated `layer_norm` primitive unblocking CLIP/ViT/BERT hidden-
   state normalization, plus the newly registered generated `embedding`
   primitive for learned token/position tables, plus the now-bounded
   `argmax(int32/int64)` admission needed for legacy OpenAI CLIP EOT index
   selection, prefer the next first-model enabling surface that is still
   half-finished: the remaining text-pooling gather/masking contract or a
   standard transformer masking/attention slice before
   revisiting grouped/fused normalization variants or dynamic normalized-
   dimension work. RoPE exploration/planning is now recorded in
   `agents/plans/rotary_apply_plan.md`; the next honest rotary slice is a
   downstream consumer of the landed 1D tables, such as a bounded one-tensor
   real-pair application helper or a 2D/3D table-preparation helper, not a
   speculative fused public `apply_rotary_emb` CUDA ABI. Do not restart
   `cropped_pos_embed` without new human direction. The landed `argmax`
   integer-input exception is intentionally specific to this direct CLIP
   blocker; do not use it as a reason to widen unrelated integer tensor
   support or claim broader CLIP pooling parity before non-2 EOS matching and
   the pooled hidden-state gather path are actually covered.
2. Continue the first bounded ConvNd provider slice described in
   `agents/plans/conv_cutlass_plan.md` by promoting the now-runnable
   `float16` SIMT fallback plus C=3 TensorOp `FewChannels` launcher toward
   provider maturity without widening public semantics. The next valuable
   slices are either a similarly explicit `FixedChannels` C=4/8 admission if it
   compiles and validates cleanly, or real Conv profiler execution and
   execution-plan consumption. Do not describe the current runtime launcher set
   as optimized/provider-mature Conv parity: it is a narrow static rank-4,
   groups=1 fp16 slice with pack/unpack provenance, C=3 few-channel Torch
   parity, and SIMT fallback coverage.
   Keep the work narrow:
   no conv3d, no transposed/depthwise/grouped expansion, no hidden channel
   padding, no runtime-set packed weights, and no public NHWC toggle.
3. Revisit CUTLASS/provider maturity only for another bounded compile-visible
   robustness slice if a new concrete stale-payload edge appears in an existing
   cache/test area; otherwise keep provider-cache work paused and avoid
   speculative broadening.
4. Add one more bounded GGUF regression only if another concrete loader or
   native/runtime contract edge appears, preferably around runtime load-plan
   edge cases or mixed dense/manual encoded constants on the lowered
   runtime-dequant path rather than broadening the runtime surface.
