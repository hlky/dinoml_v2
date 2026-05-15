# XGLM Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary worked example: facebook/xglm-564M.
  Representative family configs: facebook/xglm-1.7B, facebook/xglm-2.9B,
  facebook/xglm-4.5B, facebook/xglm-7.5B.

Config source:
  https://huggingface.co/facebook/xglm-564M/raw/main/config.json
  https://huggingface.co/facebook/xglm-1.7B/raw/main/config.json
  https://huggingface.co/facebook/xglm-2.9B/raw/main/config.json
  https://huggingface.co/facebook/xglm-4.5B/raw/main/config.json
  https://huggingface.co/facebook/xglm-7.5B/raw/main/config.json
  Matching generation_config.json, tokenizer_config.json, and special_tokens_map.json
  were snapshotted where available.

Source files inspected:
  transformers/src/transformers/models/xglm/configuration_xglm.py
  transformers/src/transformers/models/xglm/modeling_xglm.py
  transformers/src/transformers/models/xglm/tokenization_xglm.py
  transformers/src/transformers/models/xglm/convert_xglm_original_ckpt_to_trfms.py
  transformers/tests/models/xglm/test_modeling_xglm.py
  transformers/tests/models/xglm/test_tokenization_xglm.py
  transformers/src/transformers/convert_slow_tokenizer.py for XGLMConverter.

Any missing files or assumptions:
  No remote-code source is required for standard XGLM. No model imports,
  executions, or DinoML tests were run. The report targets inference/generation
  parity for the native PyTorch XGLM source. Tokenizer model files were not
  downloaded in full; repository file metadata confirms sentencepiece.bpe.model
  and tokenizer.json are present for the primary/large checkpoints.
```

## 2. High-level architecture

XGLM is a multilingual text-only decoder-only causal language model. The source
is structurally close to a pre-LN Transformer decoder with scaled token
embeddings, sinusoidal absolute positions, full multi-head self-attention, an
ungated FFN, final LayerNorm, and a tied LM head.

```text
Unigram/metaspace multilingual tokenization
  -> token embedding * sqrt(d_model) + sinusoidal position embedding
  -> repeated pre-LN causal decoder blocks with optional KV cache
  -> final LayerNorm
  -> tied LM head
  -> logits/sampling
```

Runtime dataflow:

```text
input_ids + attention_mask + optional position_ids
  -> decoder prefill
  -> DynamicCache per layer
  -> one-token or chunked decode
  -> logits_to_keep slice
  -> logits [B, kept_tokens, vocab_size]
```

Stageable units are tokenizer/input packing, embedding plus position generation,
one decoder block, prefill attention, decode attention with cache update, final
LM projection, and generation-controller logic.

## 3. Important config dimensions

Worked example: `facebook/xglm-564M`.

| Field | Value | Source |
|---|---:|---|
| model_type | xglm | config.json |
| vocab_size / V | 256008 | config.json |
| d_model / hidden_size / H | 1024 | config.json |
| num_layers / L | 24 | config.json |
| attention_heads / A | 16 | config.json |
| head_dim / D | 64 | inferred from H / A; source requires divisibility |
| ffn_dim / I | 4096 | config.json |
| max_position_embeddings | 2048 | config.json |
| position encoding | sinusoidal absolute, offset=2 | source |
| activation_function | gelu | config.json |
| dropout / attention_dropout / activation_dropout | 0.1 / 0.1 / 0 | config.json |
| scale_embedding | true | config.json/source |
| use_cache | true | config.json/source |
| BOS/PAD/EOS | 0 / 1 / 2 | config/generation config |
| decoder_start_token_id | 2 | config/generation config |
| tie_word_embeddings | true effective default | source default and tied keys |

Representative checkpoint sweep:

| Checkpoint | H | L | A | D | I | V | max pos | activation | dtype metadata |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `facebook/xglm-564M` | 1024 | 24 | 16 | 64 | 4096 | 256008 | 2048 | gelu | omitted |
| `facebook/xglm-1.7B` | 2048 | 24 | 16 | 128 | 8192 | 256008 | 2048 | gelu | omitted |
| `facebook/xglm-2.9B` | 2048 | 48 | 16 | 128 | 8192 | 256008 | 2048 | gelu | omitted |
| `facebook/xglm-4.5B` | 2048 | 48 | 16 | 128 | 16384 | 256008 | 2048 | relu | float16 |
| `facebook/xglm-7.5B` | 4096 | 32 | 32 | 128 | 16384 | 256008 | 2048 | gelu | omitted |

Config default trap: the current `XGLMConfig` class defaults to the 564M-style
geometry (`H=1024`, `L=24`, `A=16`, `I=4096`). Real checkpoint JSON must drive
model sizing.

## 3a. Family variation traps

- Positional embeddings are not learned in this source. They are sinusoidal,
  stored as a non-persistent buffer, indexed by `position_ids + 2`, and extended
  on demand if a generated sequence exceeds the current buffer length.
- All inspected official configs use standard MHA, not GQA/MQA:
  `num_key_value_heads` is absent and KV heads equal query heads.
- `facebook/xglm-4.5B` uses ReLU FFN activation and a wider FFN (`I=16384`) even
  though smaller/larger siblings use GELU.
- `scale_embedding=true` multiplies token embeddings by `sqrt(d_model)` before
  adding positions.
- `lm_head.weight` is tied to `model.embed_tokens.weight`; do not materialize
  these as independent logical parameters unless weight tying is intentionally
  broken.
- The tokenizer is a multilingual Unigram/metaspace tokenizer with extra
  `<madeupword0>` through `<madeupword6>` special tokens. The model config alone
  is insufficient for end-to-end text parity.
- The source has an optional cross-attention path gated by
  `config.add_cross_attention`, but official inspected configs omit it/leave it
  false. Treat cross-attention as a gated non-primary variant.
- `logits_to_keep` is part of the generation ABI and slices hidden states before
  the expensive vocab projection.
- The current attention implementation is eager `bmm` attention, not SDPA or
  FlashAttention dispatch. A fused attention provider must preserve its mask,
  fp16 softmax upcast, and cache shapes.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids -> [B, S, H])` with padding index.
- Elementwise multiply by scalar `sqrt(H)` when `scale_embedding=true`.
- Position id construction: `arange(past_len, past_len + S).unsqueeze(0)` when
  position ids are not supplied.
- Sinusoidal position table lookup with `position_ids + 2`, then broadcast/add
  with token embeddings.
- Reshape/view/transpose for attention:
  `[B, S, H] -> [B, S, A, D] -> [B, A, S, D] -> [B*A, S, D]`.
- Final hidden-state slice by `logits_to_keep`: integer last-k slice or tensor
  index.

Neural network primitives:

- LayerNorm over `H` before self-attention, optional cross-attention, FFN, and
  final decoder output.
- Dense Q/K/V projections per layer: `Linear(H -> H)` with bias.
- Dense attention output projection per layer: `Linear(H -> H)` with bias.
- Ungated FFN per layer: `Linear(H -> I)` with bias, activation GELU or ReLU,
  `Linear(I -> H)` with bias.
- Dropout is training-only; inference can compile it away.
- LM head: `Linear(H -> V)` without bias, tied to token embedding weights.

Attention primitives:

- Causal self-attention mask from `create_causal_mask`, shaped
  `[B, 1, Q, K_total]`.
- Optional encoder bidirectional mask for the gated cross-attention path.
- Batched attention score GEMM:
  `[B*A, Q, D] x [B*A, D, K_total] -> [B*A, Q, K_total]`.
- Mask add plus clamp to dtype minimum.
- Softmax over key axis, fp32 softmax when scores are fp16, then cast back.
- Batched value GEMM:
  `[B*A, Q, K_total] x [B*A, K_total, D] -> [B*A, Q, D]`.

Position/rotary/relative-bias ops:

- Sinusoidal absolute position generation and indexed lookup. No RoPE, ALiBi,
  learned relative bias, or learned absolute position parameter is present.

Generation/cache ops:

- `DynamicCache` allocation when `use_cache=True` and no cache is supplied.
- Per-layer self-attention cache update with keys/values shaped
  `[B, A, K_total, D]`.
- Optional `EncoderDecoderCache` if cross-attention is enabled or the config is
  treated as encoder-decoder.
- Cache length query via `past_key_values.get_seq_length()`.

Preprocessing-coupled ops:

- Tokenizer normalizer: replace newlines/tabs with spaces, NFKC normalization,
  collapse repeated spaces.
- Metaspace pre-tokenization/decoding with prefix space by default.
- Unigram tokenization with `unk_id=3`; pretrained repos also include
  `sentencepiece.bpe.model` and `tokenizer.json`.
- Post-processing and special token handling must be sourced from tokenizer
  artifacts, not guessed from model config.

Quantized/packed weight metadata ops:

- None in native source/configs inspected. Any quantized checkpoint should be
  admitted as a separate loading/provider contract, not an XGLM source feature.

## 5. Layer/block breakdown

Embedding path:

```text
tokens = Embedding(V, H, padding_idx=1)(input_ids) * sqrt(H)
pos_ids = provided_position_ids or arange(past_len, past_len + S)[None, :]
pos = sinusoidal_table.index_select(pos_ids + 2)
x = dropout(tokens + pos)
```

Decoder block, repeated `L` times:

```text
residual = x
x = LayerNorm(H)(x)
q = Linear(H -> H, bias=True)(x) * (D ** -0.5)
k = Linear(H -> H, bias=True)(x or encoder_hidden_states)
v = Linear(H -> H, bias=True)(x or encoder_hidden_states)
k, v = cache.update(k, v) when cache is present
attn = causal_attention(q, k, v, mask)
x = residual + dropout(Linear(H -> H, bias=True)(attn))

if encoder_hidden_states is not None:
  residual = x
  x = LayerNorm(H)(x)
  x = residual + dropout(cross_attention(x, encoder_hidden_states, encoder_mask))

residual = x
x = LayerNorm(H)(x)
x = Linear(I -> H, bias=True)(dropout(act(Linear(H -> I, bias=True)(x))))
x = residual + dropout(x)
```

Final head:

```text
x = LayerNorm(H)(x)
hidden = x[:, logits_to_keep_slice, :]
logits = hidden @ embed_tokens.weight.T
```

Representative projection shapes:

| Checkpoint | Q/K/V/O | FFN up | FFN down | LM head |
|---|---|---|---|---|
| 564M | 1024->1024 | 1024->4096 | 4096->1024 | 1024->256008 |
| 1.7B / 2.9B | 2048->2048 | 2048->8192 | 8192->2048 | 2048->256008 |
| 4.5B | 2048->2048 | 2048->16384 | 16384->2048 | 2048->256008 |
| 7.5B | 4096->4096 | 4096->16384 | 16384->4096 | 4096->256008 |

## 6. Attention requirements

Primary attention is causal decoder self-attention.

```text
causal or noncausal: causal self-attention for primary path
self-attention or cross-attention: self-attention primary; cross-attention gated
MHA/MQA/GQA: MHA, KV heads = query heads
head count / KV head count / head dim:
  564M: A=16, KV=16, D=64
  1.7B/2.9B/4.5B: A=16, KV=16, D=128
  7.5B: A=32, KV=32, D=128
query/key/value width: H for all projections
query length and key/value length:
  prefill Q=S, K=S
  decode Q=new token/chunk length, K=past_len + Q
masking style: additive causal/padding mask [B,1,Q,K], clamp to dtype min
packed/varlen support: not source-native
sliding-window/local attention: none
ALiBi/relative bias/RoPE: none
KV cache requirements: per-layer self-attention keys and values [B,A,K,D]
FlashAttention/SDPA compatibility: possible optimization, not source dispatch
```

Cache behavior:

- Before flattening for `bmm`, projected keys/values are stored as
  `[B, A, src_len, D]`.
- `DynamicCache.update` appends current self-attention keys/values and returns
  updated tensors. Cached keys are stored after linear projection and before any
  flatten-to-`B*A` attention view. There is no RoPE or positional transform on
  K, so no pre/post-RoPE distinction is needed.
- `past_key_values_length` drives default position ids and causal-mask shape.
- For optional cross-attention under `EncoderDecoderCache`, cross K/V can be
  computed once, marked updated per layer, and reused for later decode steps.

Fused attention parity hazards:

- Query scaling happens immediately after Q projection and before the score
  GEMM.
- The attention mask is added to scores in `[B,A,Q,K]` view, then clamped to
  the dtype minimum.
- fp16 score softmax requests fp32 accumulation and casts probabilities back to
  fp16 before dropout/value GEMM.
- Attention weights are reshaped twice for output recording; inference without
  attentions can skip materializing returned attention maps.

## 7. Position encoding and custom math

XGLM uses sinusoidal absolute positions. The table is a non-persistent buffer,
created for `max_position_embeddings + 2`, and dynamically extended when
`2 + seq_len + past_key_values_length` exceeds the current table length.

Concise reproduction:

```python
def xglm_sinusoidal_table(num_embeddings, dim, padding_idx=None):
    half = dim // 2
    inv = exp(arange(half, dtype=float32) * -(log(10000.0) / (half - 1)))
    phase = arange(num_embeddings, dtype=float32)[:, None] * inv[None, :]
    table = concat([sin(phase), cos(phase)], axis=1).reshape(num_embeddings, -1)
    if dim % 2 == 1:
        table = concat([table, zeros([num_embeddings, 1])], axis=1)
    if padding_idx is not None:
        table[padding_idx, :] = 0
    return table

def xglm_position_embedding(position_ids, table):
    return table[(position_ids + 2).reshape(-1)].reshape(
        position_ids.shape[0], position_ids.shape[1], table.shape[-1]
    )
```

Precomputable:

- Base sinusoidal table for a chosen maximum context.

Dynamic inputs:

- `position_ids` if caller supplies them.
- `past_key_values_length` during generation.
- On-demand table extension if admitted context exceeds the current compiled
  or loaded position table.

First DinoML integration should choose either a fixed admitted maximum context
with a precomputed table or an explicit dynamic extension/fallback path. Silent
wrapping or learned-position loading would be wrong for this source.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Multilingual text normalization/tokenization with a pretrained XGLM tokenizer
  artifact.
- The tokenizer class uses NFKC normalization, whitespace cleanup, metaspace
  pre-tokenization/decoding, Unigram model, and seven made-up special tokens.
- Model input names are `input_ids` and `attention_mask`.
- Generation configs use BOS=0, PAD=1, EOS=2, decoder start token=2.

GPU/runtime graph inputs:

- `input_ids: int64/int32 [B,S]` or `inputs_embeds: float [B,S,H]`, exactly one.
- Optional `attention_mask: [B,S_total]`, converted by shared Transformers mask
  utilities into additive causal mask `[B,1,Q,K]`.
- Optional `position_ids: [B,S]`; when absent source creates a single row
  starting at `past_len`.
- Optional `past_key_values` cache.

Packing notes:

- Left padding is tested in generation; the attention mask must preserve padded
  prompt behavior with cache.
- There are no segment/token type ids, multimodal placeholders, packed sequence
  descriptors, or scatter-stitch operations.
- Tokenizer post-processing differs between direct tokenizer construction and
  pretrained tokenizer artifacts/converters in source-adjacent code. For
  end-to-end parity, DinoML should use the repo tokenizer files as the source
  of truth and keep neural graph parity separate from text preprocessing parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: tied LM head as embedding transpose

Source pattern:

```text
lm_head(hidden) with lm_head.weight tied to model.embed_tokens.weight
```

Replacement:

```text
MatMul(hidden, embed_tokens.weight.T)
```

Preconditions:

- `tie_word_embeddings=true` or `_tied_weights_keys` proves aliasing.
- `lm_head.bias is None`.
- Weight loader preserves one logical parameter alias for embedding and LM head.

Shape equations:

- hidden `[B,K,H]`, embedding weight `[V,H]`, logits `[B,K,V]`.

Failure cases:

- Untied/adapted checkpoints, LoRA overlays, or quantized heads that store a
  separate output projection.

Parity test sketch:

- Compare logits for full sequence and `logits_to_keep=1` against source.

### Rewrite: eager MHA to fused causal attention

Source pattern:

```text
Q/K/V linears -> reshape/transpose -> bmm scores -> mask -> softmax -> bmm V
```

Replacement:

```text
QKV projections + fused causal MHA prefill/decode
```

Preconditions:

- MHA only: `H % A == 0`, KV heads equal Q heads.
- No cross-attention for primary path.
- No requested output attentions, or provider can return/source-compatible
  attention probabilities.
- Preserve query scaling, additive mask semantics, fp16 fp32-softmax behavior,
  and cache layout.

Shape equations:

- Q `[B,A,Q,D]`, K/V `[B,A,K,D]`, output `[B,Q,H]`.

Failure cases:

- Optional cross-attention, attention output recording, unknown packed masks,
  or provider inability to match mask/clamp/upcast ordering.

Parity test sketch:

- Single-layer prefill and decode parity with padding mask in fp32 and fp16.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden_states[:, slice_indices, :] -> lm_head
```

Replacement:

```text
Only project requested positions through vocab GEMM
```

Preconditions:

- `logits_to_keep` is integer last-k or resolved tensor indices.
- Loss computation is absent or labels require matching full logits.

Shape equations:

- Full hidden `[B,S,H]`; kept hidden `[B,K,H]`; logits `[B,K,V]`.

Failure cases:

- Training loss over all positions, arbitrary gather indices not supported by
  the lowering, or downstream consumers requiring full logits.

Parity test sketch:

- Compare `logits_to_keep=0`, `1`, and small integer `K` on the same prompt.

### Rewrite: FFN activation fusion

Source pattern:

```text
Linear(H -> I) -> GELU/ReLU -> Linear(I -> H)
```

Replacement:

```text
GEMM + fused activation epilogue, then GEMM back down
```

Preconditions:

- Activation is exactly config-selected GELU or ReLU.
- Inference dropout removed.
- Bias semantics preserved.

Failure cases:

- 4.5B uses ReLU while most variants use GELU; activation must be config-driven.

Parity test sketch:

- Per-block FFN parity for GELU and ReLU configs.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `H`: appears three times per layer in the gated-cross-attention
  superset and twice per primary layer, plus final LayerNorm.
- Q/K/V projection region: official weights are separate linears, but a loader
  could pack them into a single `[3H,H]` runtime weight with split order Q, K, V
  if aliasing/provenance is explicit.
- Causal MHA with KV cache: primary decode bottleneck; must support `[B,A,K,D]`
  cache and the source mask/upcast order.
- Last-token-only logits: vocab is 256008, so avoiding full-sequence vocab GEMM
  matters for generation.

Medium priority:

- FFN activation epilogue fusion for GELU/ReLU.
- Residual-add plus dropout-elided inference fusion after attention/FFN.
- Embedding scale plus position add fusion for prefill.

Lower priority:

- Cross-attention support under `add_cross_attention`; not used by inspected
  official checkpoints.
- Training-only dropout/layerdrop/gradient-checkpointing behavior.
- Dynamic sinusoidal table extension beyond admitted maximum context; useful as
  a fallback but not a first optimized path.

## 11. Runtime staging plan

Stage 1: config/tokenizer/weight admission.

- Parse `XGLMConfig` fields and reject unsupported `add_cross_attention=true`
  for the first primary path.
- Preserve tied input/LM embedding aliasing.
- Admit fixed max context initially; document whether dynamic position extension
  is unsupported or handled by recompilation/fallback.

Stage 2: one-block fp32 parity.

- Implement embedding scale, sinusoidal lookup, pre-LN decoder block, eager MHA,
  FFN GELU/ReLU, residuals, and final LayerNorm.
- Validate with tiny/random source-shaped configs without importing during this
  audit.

Stage 3: full prefill parity.

- Run all layers for a small prompt with causal/padding mask and full logits.
- Add `logits_to_keep` slicing before LM projection.

Stage 4: decode with `DynamicCache` ABI.

- Allocate per-layer K/V cache `[B,A,max_cache,D]`.
- Validate one-token and multi-token continuation against no-cache prefill.
- Include left-padded batch prompts.

Stage 5: optimized attention and projection packing.

- Swap eager attention for a provider path after parity tests cover fp32/fp16
  mask/upcast/cache behavior.
- Add optional packed QKV lowering as a weight transform with clear split order.

Stage 6: production generation probes.

- Tokenizer-driven prompts, greedy/sample controller parity, throughput and KV
  memory sweeps.

Stub initially:

- Training losses, layerdrop/dropout, gradient checkpointing, returned
  attentions, cross-attention, and beam-search-specific cache reorder.

## 12. Parity and validation plan

Concrete tests:

- Sinusoidal table parity for even and odd `H`, padding row zeroing, offset=2,
  and extension beyond initial max.
- Embedding path parity with `scale_embedding=true` and a synthetic
  `scale_embedding=false` config.
- One-layer decoder parity for both GELU and ReLU FFN variants.
- Attention parity with no mask, causal mask, padding mask, fp32, and fp16
  softmax-upcast behavior.
- Cache parity:
  - prefill full sequence vs prefill plus one-token decode;
  - prefill plus three-token decode chunk;
  - left-padded batch with attention mask.
- Full model prefill logits parity for `facebook/xglm-564M` geometry on a small
  loaded checkpoint or reduced fixture.
- `logits_to_keep` parity for `0`, `1`, and `K>1`.
- Tokenizer/generation smoke parity with known IDs from upstream tests once
  model execution is allowed.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for block/logit parity.
- fp16: `rtol=1e-2`, `atol=1e-2`, with stricter slice checks where attention
  provider math matches the eager path.
- Cache/no-cache slice parity: upstream tests use `atol=1e-3`; use that as the
  initial acceptance target for source-compatible eager math.

## 13. Performance probes

- Tokenizer throughput by language/script and prompt length.
- Prefill-only latency and tokens/sec over sequence lengths 16, 128, 512, 2048.
- Decode-only tokens/sec with cache lengths 128, 512, 2048 and batch sizes 1,
  4, 8, 16.
- KV cache memory usage:
  `2 * L * B * A * max_seq * D * dtype_size`.
- Attention backend comparison: eager bmm, fused MHA/Flash-like provider,
  cache-aware decode kernel.
- FFN GEMM throughput by checkpoint geometry, especially 4.5B `H=2048,I=16384`
  and 7.5B `H=4096,I=16384`.
- Vocab projection cost with full logits vs `logits_to_keep=1`.
- Weight-load and tied-weight alias verification time/memory.
- End-to-end generation latency split into tokenization, prefill, decode, and
  sampling.

## 14. Skip/defer list

- Training, labels/loss, dropout, layerdrop, and gradient checkpointing.
- Returned attentions/cross-attentions unless a caller explicitly requires
  them.
- `add_cross_attention=true` variants; official inspected XGLM configs do not
  require this path.
- Beam-search cache reorder and advanced generation processors for first block
  and greedy decode parity.
- Quantized or packed checkpoint formats; no native source/config requirement
  was found.
- Dynamic context beyond the admitted maximum, except for a clear reject or
  fallback path.
- Distributed tensor parallelism/model parallelism.

## 15. Final implementation checklist

- [ ] Parse `XGLMConfig` aliases: `hidden_size=d_model`, `num_layers`, `attention_heads`.
- [ ] Load tokenizer metadata separately from neural config.
- [ ] Preserve tied `lm_head.weight` / `model.embed_tokens.weight` alias.
- [ ] Implement scaled token embedding.
- [ ] Implement XGLM sinusoidal position table with offset=2 and padding row.
- [ ] Implement causal mask and padding mask conversion to `[B,1,Q,K]`.
- [ ] Implement pre-LN decoder block with MHA, residuals, LayerNorm, FFN.
- [ ] Support GELU and ReLU FFN activations from config.
- [ ] Implement `DynamicCache`-compatible K/V storage `[B,A,K,D]`.
- [ ] Implement `logits_to_keep` before LM projection.
- [ ] Add one-block fp32 parity tests.
- [ ] Add cache vs no-cache decode parity tests.
- [ ] Add left-padding generation parity test.
- [ ] Add last-token-only logits parity/performance test.
- [ ] Add fused attention admission guards after eager parity is stable.
