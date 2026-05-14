# Transformers model-family assessment prompt

Transformers source is available in `X:/H/transformers`.

This is not a runtime code edit task. Do not implement Dinoml operators or run
Dinoml tests unless the user explicitly changes scope. Reading source files,
fetching Hugging Face configs, and writing Markdown reports is expected.

Model families are directories under `src/transformers/models`.

Output should be written in this workspace for dinoml_v2 under
`agents/plans/transformers/{{MODEL_FAMILY}}/report.md`. Cross-family prompt
review notes may be written under `agents/plans/transformers/report_review.md`.

## Prompt

You are helping develop `dinoml_v2`, a high-performance inference/runtime stack. Your task is to inspect a Hugging Face Transformers model implementation and write a standardized report that helps Dinoml agents plan operator coverage, graph rewrites, kernel fusion candidates, and staged integration work.

### Model target

- Model family / architecture: `{{MODEL_FAMILY}}`
- Hugging Face model id, if known: `{{HF_MODEL_ID}}`
- Transformers source file(s): `{{TRANSFORMERS_SOURCE_URLS}}`
- Commit or version to inspect: `{{TRANSFORMERS_COMMIT_OR_VERSION}}`
- Primary inference task: `{{TASK}}`
- Dinoml assumptions: `{{DINOML_ASSUMPTIONS}}`

Examples for `DINOML_ASSUMPTIONS`:

```text
NHWC preferred for vision tensors.
Inference-only first.
CUDA GPU target.
Prioritize batch throughput over single-request latency.
Prefer graph rewrite rules that canonicalize uncommon ops into GEMM/attention/norm primitives.
```

Treat NCHW/NCDHW -> NHWC/channel-last as a guarded layout/fusion optimization,
not as the default semantic translation. Initial graph translation should remain
faithful to Transformers/PyTorch axes unless a region is local and fully
controlled. Reports should identify candidate regions where layout translation
is safe, regions that need a no-layout-translation guard, and axis-sensitive
attrs that a layout pass would have to rewrite, such as `dim=1 -> dim=-1`,
concat/reduction/pooling axes, view/reshape assumptions, and downstream
consumer layout contracts.

### What to inspect

Inspect the official Transformers implementation and model config. Prefer exact source at a pinned commit. Use primary sources only when possible:

- `modeling_*.py`
- `configuration_*.py`
- `processing_*.py`
- `image_processing_*.py`
- `tokenization_*.py`, only if tokenizer/model coupling matters
- HF model `config.json`
- HF preprocessor / processor config
- any custom remote-code files if `trust_remote_code=True` is required

Do not rely only on README/model card claims. Confirm operations from code.

If a modeling file is generated from a modular source file, inspect both when
practical and state which file is authoritative for future source edits.

Inspect at least 3-5 representative checkpoint configs when available:

- a small/debug checkpoint
- a common production checkpoint
- any variant that changes operator structure, such as GQA/MQA, MoE, gated MLP,
  long context, sliding-window/local attention, vision/audio branches, custom
  position encoding, different tokenizer/vocab, or optional projection biases

Use official repos when accessible. If official repos are gated or unavailable,
use an open mirror and label that fact clearly. Separate facts that come from
`config.json` from facts that come from source defaults, safetensors/model
metadata, or Hugging Face repo metadata.

If representative checkpoints share a `model_type` but the config/source warns
they should use another class for correct behavior, mark them as out-of-scope
for the current report or create a separate follow-up target.

If checkpoint configs contain historical or remote-code feature flags, verify
whether the inspected in-library source actually implements them. Do not report
config-advertised behavior as required runtime behavior unless the current
source reads those fields or the report is explicitly scoped to remote code.
When native source and remote-code behavior diverge, label the report scope and
list unsupported config combinations that DinoML should reject or route to a
separate audit.

State the primary runtime target for the report, such as base encoder, masked
LM, causal LM, seq2seq LM, image classification, or multimodal generation. For
every other head implemented in the source, mark it as required, optional, or
deferred for the stated target.

Some directories under `src/transformers/models` are not autoregressive text
models, or are not Transformers architecturally despite living in the library.
For dual encoders, contrastive models, CNNs, retrieval/classification models,
and other non-generation families, state the real runtime contract instead of
forcing prefill/decode language. Document independently stageable encoders,
feature-projection heads, pooling/indexing rules, similarity heads, and output
matrix orientation when applicable.

For detection, segmentation, retrieval, or other structured-output tasks,
document postprocessing that is required for end-to-end inference parity, such
as box format conversion, score filtering, mask resizing/cropping, thresholding,
label mapping, similarity matrix orientation, or the explicit absence of NMS.
Separate this from training-only loss/matching paths.

For models that route through nested `backbone_config`, AutoBackbone, timm, or
another Transformers family, state whether the current report owns that
backbone's operator coverage or composes a separately audited family. Document
the exact backbone feature contract consumed by the model: selected feature
names/indices, tensor layout, channel widths, strides, masks, positional data,
and whether returned features are token sequences or image-like maps.

For wrapper or bridge families whose neural body is delegated to an external
library, do not infer a fixed operator surface from the wrapper alone. Document
the external owner/library/version, topology-selecting config fields or
`model_args`, preprocessing source, output ABI variants, weight-key mapping,
and an explicit admission policy: allowlist exact delegated bodies with
separate audits, route to a fallback, or reject unsupported combinations.
Treat wrapper-owned work as dispatch, metadata, preprocessing, and ABI handling
unless the inspected source includes the neural operators.

For multimodal models, inspect the processor/preprocessor config and document
the exact runtime tensors it produces. Include observed or config-derived shapes
for `pixel_values`, `input_features`, grid metadata, modality token type IDs,
placeholder tokens, packed sequence descriptors, and any `cu_seqlens`-style
metadata when applicable.

When source replaces placeholder token embeddings with image/audio/video
features using broad scatter APIs such as `masked_scatter`, document whether
the processor guarantees a stricter bounded pattern. Reports should spell out
placeholder IDs, count/order validation, contiguous prefix versus arbitrary
positions, row-major flatten order, and whether DinoML can lower the operation
to indexed row copy or prefix copy with rejection guards instead of admitting a
general boolean scatter.

For video models, document the frame/clip ABI explicitly: who owns video decode
and frame sampling, expected frame count/fps policy, processor output layout
such as `[B,T,C,H,W]` versus channel-last variants, any immediate source
permutes to `NCTHW`/`NCHW`, tubelet or per-frame patch embedding order,
temporal/spatial token order, temporal pooling, positional table dependence on
frame count, and layout-pass guard boundaries. Treat `NTHWC`/channel-last video
as a guarded fusion/layout opportunity unless the whole processor-to-token
region is controlled and all temporal/spatial axes are rewritten together.

For models that consume precomputed modality features rather than raw
`pixel_values` or `input_features`, document the external extractor boundary
explicitly. Include feature tensor rank and width, region/box/audio-frame
ordering, coordinate normalization or alignment metadata, masks, token/type
IDs, optional relationship or object-label side inputs, cacheability, and
whether DinoML should own, compose, or reject the upstream extractor for the
first integration. Do not infer NHWC/NCHW image-layout work inside a model that
only consumes rank-3 sequence or region features.

For tokenized-image or discrete-code multimodal models, document the image
tokenizer/codebook contract explicitly: input image size and layout, latent grid
shape, codebook size and embedding dimension, code-index to vocabulary-token
mapping, placeholder expansion count/order, logits masks or generation
constraints for image-code tokens, and whether image codes/embeddings can be
cached independently from the decoder KV cache.

For multilingual or tokenizer-controlled generation models, document language
control as an ABI separate from neural graph ops. Include tokenizer class and
language-code table, source/target prefix or suffix layout, decoder start token
rules, forced BOS/EOS ids, generation metadata required by translation helpers,
vocab-size differences across tokenizer variants, and whether language control
is packed into `decoder_input_ids` or enforced by the generation controller.

For staged composite models with optional output modalities, explicitly choose a
first useful DinoML runtime target and separate later stages. Examples include
text-output thinker parity before speech-code talker generation, and codec/DiT/
vocoder waveform synthesis after text/audio-token parity.

For prompt-conditioned models, identify the conditioning mechanism explicitly:
cross-attention, FiLM/adaptive scale-bias, prompt/query embeddings, prompt
encoders, feature similarity, masked attention, or indexed embedding stitch.
State whether condition embeddings can be cached independently, and whether the
conditioning changes tensor rank, batch/query packing, or output orientation.

For audio models, inspect the feature extractor/preprocessor config and
document the waveform contract and feature tensor contract: sampling rate,
mono/stereo expectations, chunk length, padding/truncation, STFT/FFT/hop/window
settings, mel/bin counts, normalization/clamp math, output shape, and whether
feature extraction is expected to run in the CPU/data pipeline or GPU/runtime.
If preprocessing splits or packs one user audio sample into multiple model
examples, document the split policy, overlap/boundary search, emitted mapping
metadata, decode/reassembly rules, and whether that metadata is consumed by the
model graph or only by postprocessing.

For time-series forecasting models, document the forecasting ABI instead of
forcing text-generation terms: `past_values`, `future_values`,
`past_time_features`, `future_time_features`, observed masks and missing-value
policy, static categorical/real features, context length, prediction length,
lag sequence and required history length, scaling/loc/scale features,
distribution-head parameterization, sampling/RNG contract,
`num_parallel_samples` batch expansion, and whether source generation uses
decoder caches or recomputes full prefixes. For patch-based time-series models,
record the exact patch axes, such as `[B,T,C] -> [B,C,N_patches,patch_length]`,
stride/overlap/tail-cropping rules, channel-independent versus shared weights,
channel-attention or channel-mixing axes, deterministic versus probabilistic
heads, and which task head is the first DinoML target.

For recurrent, state-space, linear-attention, or hybrid decoder models, do not
force KV-cache terminology. Document the exact state ABI: state tensor count,
shape, dtype, initialization, per-layer ownership, update order, cache reorder
or reset semantics, static-address mutation requirements, and which states grow
with sequence length versus remain fixed-size. For hybrid models, list the cache
manifest by layer type, such as attention KV layers plus Mamba/RWKV/conv/SSM
state layers.

For hidden-state memory or recurrence mechanisms that are not KV caches, such as
Transformer-XL/XLNet `mems`, document where memory is concatenated relative to
projection, whether memory stores layer inputs or outputs, detach/cutoff rules,
`mem_len=None` growing behavior, target-mapping/permutation-mask interactions,
and beam/cache reorder dimensions.

For sparse, local, block-sparse, or long-context attention models, document the
exact attention pattern and admission rules: local/window sizes, global tokens
or blocks, random/block plans, sequence padding or bucket divisibility, mask
value conventions, fallback-to-dense thresholds, unsupported causal/cross/dilated
variants, and whether output attention tensors require dense reconstruction
separate from the hidden-state fast path.

For randomized, hash-based, sorting-based, or bucketed attention, document the
determinism contract: RNG seed source, random tensors that are not weights,
stable/unstable sort requirements, bucket/hash shapes, reverse-index recovery,
decode cache contents, and whether production admission requires fixed seeds or
precomputed plans.

For source-specific bounded math or projection/post-score transforms, document
the exact placement, dtype, and enabled-by-config behavior. Examples include
Q/K/V clipping, Q/K post-projection norms, attention score softcaps, final logit
softcaps, embedding/attention/residual/logit multipliers, nonstandard attention
scaling, and any ordering relative to reshape, RoPE, cache update, mask addition,
softmax, residual add, or output projection.

Do not infer projection dimensions from `hidden_size` alone. Record explicit
source/config projection widths, `head_dim`, attention output width, rotary
dimension, and any cases where `hidden_size != num_attention_heads * head_dim`.
For fused projection rewrites, list the exact split order and packed weight
layout used by source weights.

For source-coupled quantized or packed weight formats, document storage tensor
names, metadata tensors, excluded dense modules, dequant/materialization path,
native-kernel requirements, and a safe dense fallback. Treat those as loading
and provider contracts, not just dtype annotations.

If a checkpoint config omits fields that the current Transformers config class
supplies by default, list the omitted fields and the effective defaults.
If a checkpoint config contains historical or implementation-specific fields
that the inspected modeling source does not read, list them as ignored for this
source basis instead of treating them as required runtime features.
If an official or representative checkpoint is gated or returns 401/403, include
the Hugging Face model URL as a clickable Markdown link, state what access would
resolve, and continue with clearly labeled mirrors/source defaults where useful.

### Report requirements

Write a Markdown report with the following sections.

## 1. Source basis

List exact files, URLs, commits, and model configs inspected.

Include:

```text
Transformers commit/version:
Model id:
Config source:
Source files inspected:
Any missing files or assumptions:
```

## 2. High-level architecture

Describe the model stages.

Examples:

```text
text-only decoder
encoder-decoder
vision encoder + text decoder
audio encoder + text decoder
MoE decoder
multimodal projector + LLM
```

Include a simple dataflow diagram in text:

```text
input preprocessing -> encoder/projector -> decoder/prefill -> decode -> logits/sampling
```

For multimodal or multi-stage models, also include a stage decomposition that
separates CPU/data-pipeline work, independently cacheable encoders/projectors,
prefix construction, prefill, and decode. Call out which stages can be validated
or optimized independently.

## 3. Important config dimensions

Extract dimensions from config and present them as a table.

Include relevant fields such as:

```text
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
intermediate_size
vocab_size
max_position_embeddings
rope/theta settings
vision/audio hidden sizes
patch sizes / stride / merge sizes
MoE expert counts
activation function
dtype
cache support
```

Also include a representative checkpoint sweep table. The sweep should make
operator-significant variation visible rather than only reporting the smallest
example.

## 3a. Family variation traps

List config-dependent behavior that invalidates naive assumptions, such as:

```text
hidden_size != num_heads * head_dim
num_key_value_heads < num_attention_heads
optional attention/MLP biases
different MLP activations or gated-vs-ungated FFNs
long-context RoPE variants
sliding-window/local attention
vocab/tokenizer changes
encoder/decoder layer count asymmetry
vision/audio/projector branch changes
MoE expert/routing differences
processor output format changes
placeholder/scatter token conventions
packed/varlen sequence metadata
legacy or model-specific packed projection weight layouts
shared-weight or tied-weight aliases that must remain one logical parameter
native-source behavior that ignores or rejects historical remote-code config flags
pooling/indexing compatibility branches such as EOS/EOT pooling
non-attention or non-transformer model bodies inside Transformers
NCHW/NCDHW modeling code versus NHWC/channel-last layout-pass candidates
axis-sensitive ops that need no-layout-translation guards
```

## 4. Operator coverage checklist

List required runtime operators grouped by category.

Use categories like:

```text
Tensor/layout ops
Neural network primitives
Attention primitives
Sparse/local/block attention pattern ops, if relevant
Hash/sort/bucket attention ops, if relevant
Projection clipping / post-projection norm / softcap ops, if relevant
Quantized/packed weight metadata ops, if relevant
Position/rotary/relative-bias ops
Generation/cache ops
Recurrent/state-space cache ops, if relevant
Hidden-state memory ops, if relevant
Preprocessing-coupled ops
Scatter/indexed update ops for multimodal embedding stitch
Discrete codebook / tokenizer ops, if relevant
Optional codec/diffusion/vocoder generation ops, if relevant
Packed/varlen sequence metadata ops, if relevant
Distributed/tensor-parallel ops, if relevant
```

Be explicit. Prefer `Linear(1536 -> 4608)` style shapes where known.

For vision operators, state source tensor layout and any candidate optimized
layout explicitly, especially around patch embeddings, convolutions, pooling,
normalization, and image processors. If an optimized layout would change axis
numbers, list those required axis rewrites instead of silently changing the
semantic graph.

For projection layers with nonstandard storage or packing, document the exact
weight layout and split order. Examples include GPT-2 `Conv1D` weights stored as
`[in_features, out_features]` and BLOOM QKV rows packed per head as
`[q, k, v]` groups rather than all-Q/all-K/all-V row blocks.

For models with parameter sharing or tied weights, state the aliasing contract.
Examples include ALBERT cross-layer sharing, tied token embeddings/LM heads, and
shared relative-position bias tensors. Reports should distinguish logical layer
applications from physical weight modules so lowering does not accidentally
clone mutable state or break weight identity.

## 5. Layer/block breakdown

For each major block, describe the forward path.

Example format:

```text
Decoder block, repeated N times:
  x = RMSNorm(x)
  q,k,v = Linear(...)
  q,k = RoPE(q,k)
  x = Attention(q,k,v, cache)
  x = residual + Linear(...)
  x = RMSNorm(x)
  x = MLP(...)
  x = residual + x
```

Include shapes and whether projections have bias.

## 6. Attention requirements

Describe every attention variant required.

If no attention is required, say so explicitly and identify which attention,
mask, cache, or generation sections are not applicable for the primary target.
If attention is present only in an encoder-style branch, such as a causal text
encoder with no KV cache, describe that distinction.

Include:

```text
causal or noncausal
self-attention or cross-attention
MHA/MQA/GQA
head count / KV head count / head dim
query/key width and value width when they differ
query length and key/value length when rectangular attention is used
masking style
packed/varlen support
sliding-window/local attention
ALiBi/relative bias/RoPE interactions
KV cache requirements
FlashAttention/SDPA compatibility
```

For query-driven cross-attention decoders, latent-attention models, Perceiver-like
architectures, and other non-autoregressive decoder heads, describe the query
construction ABI separately from generation decode. Include query source
(learned table, output positions, modality coordinates, prompt tokens, pooled
state), query shape, key/value source, q/k projection width, value projection
width, whether q/k/v widths are inferred from query or key/value tensors, and
whether masks apply to queries, keys, or only the input cross-attention. Do not
label these decoder heads as KV-cache generation unless the source implements
autoregressive cache reuse.

For deformable, sparse-sampling, or grid-sampling attention, document it as a
custom attention family rather than ordinary dense MHA. Include reference-point
shape and coordinate convention, spatial level order, `spatial_shapes`,
`level_start_index`, valid ratios, sampling-offset and attention-weight shapes,
normalization equations, interpolation mode, padding/`align_corners` behavior,
mask application order, and whether the source has a custom kernel or an eager
fallback such as `grid_sample`.

For generation models, state the exact per-layer cache tensor shapes before and
after any MQA/GQA/repeat expansion. Identify whether cached keys are stored
before or after position encoding. Call out eager fallback paths that are likely
too slow and identify the source optimized backend dispatch path, if any.
Document any source-specific attention math order, such as query scaling before
the backend call, upcasts/downcasts around softmax, masking order, or dropout
placement, because fused attention parity may depend on preserving it.

Distinguish cache types explicitly: autoregressive self-attention KV cache,
encoder-decoder cross-attention cache, independently cacheable encoder/projector
outputs, retrieval/contrastive branch embedding caches, precomputed prompt/audio
codes, and processor-derived metadata caches. Do not describe non-generation
branch caches as KV caches.

For stateful video, tracking, recurrent inference, or interactive segmentation
models, document the session/state ABI separately from KV cache. Include state
owners and lifetimes, per-frame feature caches, per-object histories, memory
tensor shapes/dtypes/layouts, object-pointer or prompt histories, update order,
eviction/window limits, reverse/forward propagation rules, and which state can
be recomputed versus persisted across calls.

## 7. Position encoding and custom math

Document RoPE, M-RoPE, ALiBi, relative bias, convolutional positional embeddings, or other nontrivial position math.

Include concise implementation snippets for custom functions that Dinoml may need to reproduce.

Snippet style:

```python
def apply_model_specific_rope(q, k, cos, sin):
    ...
```

Mention what can be precomputed and what depends on batch/dynamic inputs.

## 8. Preprocessing and input packing

Document model-coupled preprocessing that affects runtime graph shape.

Examples:

```text
image patch packing
video frame packing
audio feature extraction
special placeholder tokens
modality token type ids
grid metadata
packed patch rows
cu_seqlens / sequence length descriptors
masked scatter or indexed copy into token embeddings
position ids
attention masks
```

Separate CPU/data-pipeline work from GPU/runtime work.

For dual-encoder or contrastive models, document both branch input contracts,
branch feature shapes, pooling/indexing behavior, projection heads, feature
normalization, similarity/logit scaling, and whether branch outputs can be
cached independently before the final similarity matrix.

For text encoders, document special-token layout, segment/token type IDs,
default position IDs, padding side, and which of those enter the GPU graph.

For document/OCR/layout models, document the source of words and boxes
separately from the neural graph. Include whether OCR is invoked by the
processor or supplied by the caller, coordinate normalization scale, special
token boxes, word-to-subword box expansion, overflow/chunk mapping, image
duplication for overflow chunks, required bbox range/shape guards, and which
box-derived values become embedding indices or attention-bias inputs.

For markup/DOM-structured text models, document structural-tokenizer metadata
separately from OCR/layout boxes. Include HTML/XML parsing ownership, node/text
extraction rules, XPath or DOM path construction, tag dictionaries, unknown and
pad IDs, subscript/index clamping, max depth/width, how structure tensors expand
from nodes to subwords and special tokens, overflow mapping, and which
structure IDs feed embeddings, attention bias, or heads.

For multimodal generation models, document how modality embeddings are stitched
into text embeddings, including placeholder token IDs, shape checks, scatter or
indexed-copy behavior, and whether image/audio/video embeddings or prefix KV
caches can be precomputed.

For generation-heavy models, document generation-controller behavior that is not
part of the core module graph but is required for end-to-end parity: forced
decoder IDs, language/task prompt construction, suppress-token processors,
timestamp processors, no-speech thresholds, assistant/speculative paths, and
which pieces can be stubbed for first integration.

For detection and segmentation models, document postprocessing inputs and
outputs explicitly: target/original sizes, padded/reshaped sizes, box coordinate
conventions, class/background/no-object handling, mask upsample/crop/threshold
rules, per-image variable-length output records, and whether NMS is source
behavior or intentionally absent.

For query-based detectors, document query/proposal construction separately from
the decoder body. Include learned-query versus top-k proposal selection,
anchor/grid generation, reference-point shape and update rules, bbox refinement
math, distributional/DFL bin decoding when present, class-score selection before
or after decoding, tie behavior for `topk`, and whether decoder layers update
reference points iteratively or only in the final head.

## 9. Graph rewrite / lowering opportunities

Identify patterns that can be canonicalized into simpler primitives.

For each rewrite, include:

```text
name
source pattern
replacement pattern
exact preconditions
shape equations
weight transform
layout constraints
failure cases
parity test sketch
```

Example rewrite skeleton:

```markdown
### Rewrite: non-overlap Conv2d -> Linear

Preconditions:
- `kernel_size == stride`
- `padding == 0`
- `dilation == 1`
- `groups == 1`
- input dimensions divisible by kernel

Replacement:
```text
WindowFlatten -> MatMul(weight_flat.T) -> BiasAdd -> Reshape
```

Weight transform:
```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```
```

Be strict: do not propose unsafe rewrites without guards.

For convolution-to-linear/GEMM rewrites, include layout-aware preconditions:
`kernel_size == stride`, `padding == 0`, `dilation == 1`, `groups == 1`, input
spatial divisibility, activation flatten order, weight flatten/permutation, bias
handling, and dynamic-shape guards or fallback behavior. If upstream
preprocessing already emits flattened patches/windows, describe the exact
specialized pattern separately from the general ConvNd lowering.

For dynamic/local convolutional attention or generated-kernel paths, document
the kernel-generation graph, local-window extraction (`unfold`/gather/im2col),
window padding/alignment, per-token or per-head softmax axis, mask interaction,
temporary tensor shape, and axis/layout guards separately from static ConvNd
lowering. Do not collapse these paths into ordinary convolution rewrites unless
the generated weights are constant and the source padding/layout semantics are
fully preserved.

For layout rewrites, call out source `permute`/`transpose`/`contiguous` patterns
that can be eliminated by a guarded layout/fusion pass. Include required axis
rewrites, weight transforms, consumer layout constraints, and failure cases where
the source layout must be preserved for parity. If useful, identify regions that
should be protected by a conceptual `no_layout_translation()` guard.

## 10. Kernel fusion candidates

Rank likely fusion/kernel work by priority.

Use categories:

```text
Highest priority
Medium priority
Lower priority
```

For each candidate, explain why it matters.

Examples:

```text
RMSNorm
LayerNorm
QKV projection + RoPE
RoPE + attention prefill
GQA FlashAttention with KV cache
SwiGLU / GEGLU activation multiply
MoE routing + expert GEMM
Conv patch embedding lowered to GEMM
last-token-only logits
```

## 11. Runtime staging plan

Propose a staged Dinoml integration path.

Example:

```text
Stage 1: load config/weights and run one block parity
Stage 2: implement encoder-only parity
Stage 3: implement prefill parity
Stage 4: implement decode with KV cache
Stage 5: enable optimized attention
Stage 6: add graph rewrites/fusions
Stage 7: continuous batching / production scheduling
```

Include what can be stubbed initially.

## 12. Parity and validation plan

Give concrete parity tests.

Include:

```text
random tensor tests for custom ops
single-layer parity
after-N-layer parity
encoder/projector parity
prefill logits parity
decode token parity
end-to-end text/image/audio output parity
recommended tolerances for fp32/fp16/bf16
```

## 13. Performance probes

List benchmark probes that separate bottlenecks.

Examples:

```text
preprocessing throughput
encoder-only throughput
prefill-only throughput
decode-only tokens/sec
end-to-end requests/hour
batch-size sweep
sequence-length sweep
sparse pattern/window/block-size/global-token sweep
hash/sort/gather/bucket pipeline sweep
image/audio resolution sweep
KV cache memory usage
recurrent/state cache memory usage
attention backend comparison
scan/state-update backend comparison
packed/quantized weight load and dequant/provider comparison
```

If benchmark observations or prior measurements are included, label their
provenance and separate them from source-derived facts. Prefer probes that split
processor throughput, encoder/projector throughput, prefill, decode, logits,
and cache memory so bottlenecks do not get averaged together.

## 14. Skip/defer list

List features that can safely be deferred for first integration.

Examples:

```text
training
gradient checkpointing
beam search
video path
quantization
multi-GPU tensor parallel
rare RoPE variants
speculative decoding
```

Only defer features if they are not required for the stated task.

## 15. Final implementation checklist

End with a compact checklist Dinoml agents can copy into issues.

Use this format:

```markdown
- [ ] Parse config
- [ ] Load weights
- [ ] Implement operator X
- [ ] Implement custom function Y
- [ ] Add rewrite Z
- [ ] Add parity test A
- [ ] Benchmark B
```

### Style constraints

- Be concrete and shape-aware.
- Prefer exact source-derived claims over guesses.
- If making an inference, label it as an inference.
- Label whether dtype, parameter count, license, task, or model size comes from
  `config.json`, safetensors/index metadata, Hugging Face repo metadata, or an
  inference from source defaults.
- Distinguish required functionality from optimization opportunities.
- Include snippets for custom math, but keep them short.
- Do not paste large model source blocks.
- Do not overfit to PyTorch naming if a more general runtime concept is clearer.
- If source files are missing or remote-code-only, state that clearly.
