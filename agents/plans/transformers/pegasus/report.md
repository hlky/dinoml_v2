# Pegasus DinoML Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from X:/H/transformers
Model id: pegasus family; representative checkpoints listed below
Config source: local configuration_pegasus.py plus HF raw config.json snapshots
Source files inspected:
  X:/H/transformers/src/transformers/models/pegasus/configuration_pegasus.py
  X:/H/transformers/src/transformers/models/pegasus/modeling_pegasus.py
  X:/H/transformers/src/transformers/models/pegasus/tokenization_pegasus.py
  X:/H/transformers/src/transformers/cache_utils.py
  X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
  No remote-code source is required for the inspected in-library family.
  No tests/imports were run, per request.
```

Representative HF config snapshots are saved under `agents/plans/transformers/pegasus/_sources/`:

- `google/pegasus-large`
- `google/pegasus-xsum`
- `google/pegasus-cnn_dailymail`
- `google/pegasus-arxiv`
- `google/pegasus-pubmed`
- `hf-internal-testing/tiny-random-PegasusModel`

Future source edits should use `modeling_pegasus.py` as authoritative. The file is copied heavily from BART/MBART/Marian helpers but is not generated from a modular Pegasus source in this checkout.

## 2. High-level architecture

Primary DinoML target: `PegasusForConditionalGeneration` seq2seq text generation for summarization/translation-like encoder-decoder generation.

Dataflow:

```text
SentencePiece tokens + masks -> shared token embedding + sinusoidal positions -> encoder stack
encoder hidden states + decoder tokens -> decoder self-attention/cross-attention stack
decoder hidden states -> tied LM projection + final_logits_bias -> logits -> generation controller
```

Stage decomposition:

- CPU/data pipeline: Pegasus tokenizer normalizes newlines/multiple spaces, uses metaspace pre-tokenization, adds EOS, emits `input_ids` and `attention_mask`.
- Encoder stage: noncausal MHA over source tokens; output `[B, S, d_model]` can be cached independently across decode steps.
- Decoder prefill: causal self-attention over target prefix plus cross-attention to encoder states.
- Decode step: append decoder self-attention K/V cache; reuse cross-attention K/V after first update.
- Logits stage: `lm_head(hidden) + final_logits_bias`; `lm_head.weight` is tied to shared embeddings for conditional generation.

Implemented heads in source:

- Required: `PegasusForConditionalGeneration`.
- Optional: `PegasusModel` without LM head for encoder-decoder hidden states.
- Deferred: `PegasusForCausalLM`, a decoder-only wrapper used with `EncoderDecoderModel`; it has `logits_to_keep` slicing and no encoder wrapper ownership.
- Deferred: loss/training paths, LayerDrop, gradient checkpointing, dropout.

## 3. Important config dimensions

Current source defaults:

| Field | Default | Runtime meaning |
|---|---:|---|
| `d_model` / `hidden_size` | 1024 | token, attention, residual width |
| `encoder_layers` / `num_hidden_layers` | 12 | source default only; common Google configs use 16 |
| `decoder_layers` | 12 | source default only; common Google configs use 16 |
| `encoder_attention_heads` | 16 | MHA, no GQA/MQA |
| `decoder_attention_heads` | 16 | MHA, no GQA/MQA |
| `head_dim` | `d_model / heads` | must divide exactly; 64 for 1024/16 |
| `encoder_ffn_dim` | 4096 | encoder FFN expansion |
| `decoder_ffn_dim` | 4096 | decoder FFN expansion |
| `vocab_size` | 50265 | common Google configs use 96103 |
| `max_position_embeddings` | 1024 | encoder and decoder frozen sinusoidal tables |
| `activation_function` | gelu | common Google configs use relu |
| `scale_embedding` | false | common Google configs multiply embeddings by `sqrt(d_model)` |
| `use_cache` | true | decoder generation cache enabled by default |
| `pad/eos/decoder_start/forced_eos` | 0/1/0/1 | generation ABI metadata |

Representative checkpoint sweep:

| Model | d_model | Layers enc/dec | Heads enc/dec | Head dim | FFN enc/dec | Vocab | Max pos | Activation | scale emb |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `google/pegasus-large` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true |
| `google/pegasus-xsum` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 512 | relu | true |
| `google/pegasus-cnn_dailymail` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true |
| `google/pegasus-arxiv` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true |
| `google/pegasus-pubmed` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true |
| `hf-internal-testing/tiny-random-PegasusModel` | 16 | 2/2 | 4/4 | 4 | 4/4 | 96103 | 200 | gelu | false |

## 3a. Family variation traps

- Source defaults are not production Google checkpoint dimensions. Do not instantiate `PegasusConfig()` and expect `google/pegasus-large` parity.
- `hidden_size == num_heads * head_dim` is enforced by source, so no GQA/MQA shape variants are required for this family.
- Production configs use `relu`, not source-default `gelu`; activation must come from config.
- `scale_embedding` changes the embedding add input by `sqrt(d_model)` and is true in common Google checkpoints.
- `max_position_embeddings` can be 512 or 1024 across task checkpoints; decoder position IDs advance by cache length during decode.
- Encoder and decoder token embeddings plus LM head are logical aliases of `shared.weight`; lowering must preserve tied weight identity.
- Position embeddings are frozen sinusoidal `nn.Embedding` weights, not learned parameters and not RoPE/ALiBi.
- Historical config fields such as `add_bias_logits`, `add_final_layer_norm`, `extra_pos_embeddings`, `normalize_before`, `normalize_embedding`, and `static_position_embeddings` appear in old configs but are not read by current `modeling_pegasus.py`; reject or ignore them for this source basis.
- Generation metadata such as `num_beams`, `length_penalty`, `max_length`, `min_length`, and `task_specific_params` belongs to the generation controller, not the neural graph.
- Layout guards: all model tensors are semantic `[B, T, C]` for hidden states and `[B, H, T, D]` for attention internals. There is no image/channel layout work; do not apply NHWC translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs and frozen position IDs.
- Add, residual add, scalar multiply for embedding scale.
- Reshape/view `[B, T, C] -> [B, T, H, D]`, transpose to `[B, H, T, D]`, final transpose/contiguous/reshape back to `[B, T, C]`.
- Concatenate/update cache along sequence axis for dynamic cache.
- Optional slice for `PegasusForCausalLM.logits_to_keep`.
- Mask creation/broadcast/add for 2D masks to backend attention masks.

Neural primitives:

- `LayerNorm(C)` with affine weight/bias.
- `Linear(C -> C)` for Q/K/V/out projections with bias.
- `Linear(C -> ffn_dim)` and `Linear(ffn_dim -> C)` with bias.
- Activation from `ACT2FN`: required `relu` and `gelu` for observed configs.
- LM projection `Linear(C -> vocab)` without bias plus `final_logits_bias [1, vocab]`.
- Optional fp16 clamp in encoder layer only: `clamp(x, +/- (finfo(fp16).max - 1000))`.

Attention primitives:

- Dense MHA self-attention and encoder-decoder cross-attention.
- Eager path: `Q @ K^T * head_dim^-0.5`, mask add, softmax over key axis, `P @ V`.
- Source also advertises SDPA, FlashAttention, and FlexAttention support via `ALL_ATTENTION_FUNCTIONS`; first DinoML target can canonicalize to one dense attention primitive with strict mask/cache parity.

Position ops:

- Frozen Pegasus sinusoidal table generation or constant loading.
- Position ID generation: encoder `0..S-1`, decoder `past_len..past_len+T-1`.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` for seq2seq generation.
- Per-layer self-attention cache K/V shape `[B, H, T_dec_seen, D]`.
- Per-layer cross-attention cache K/V shape `[B, H, S_enc, D]`, updated once then reused via `is_updated[layer_idx]`.
- Beam reorder and batch select/repeat are generation-controller requirements if DinoML owns beam/contrastive search.

Preprocessing-coupled ops:

- Tokenizer outputs `input_ids` and `attention_mask`; EOS is appended by tokenizer.
- `decoder_start_token_id=0`, `pad_token_id=0`, `eos_token_id=1`, `forced_eos_token_id=1`.

Current DinoML gated gaps from project checklist:

- `LayerNorm` family is not ported.
- Attention/FlashAttention family is not ported.
- Embedding and sinusoidal positional helpers are not ported.
- GEMM/BMM, softmax last-dim, elementwise, reshape/transpose/concatenate/slice are partially available but need graph integration and cache ABI.

## 5. Layer/block breakdown

Embeddings:

```text
tokens: input_ids [B, T] -> Embedding(vocab, C) [B, T, C]
if scale_embedding: tokens *= sqrt(C)
positions: position_ids [T] -> frozen sinusoidal Embedding(max_pos, C) [T, C]
x = dropout(tokens + positions)
```

Encoder layer, repeated `encoder_layers`:

```text
res = x
x = LayerNorm(C)(x)
q,k,v = Linear(C -> C, bias=True)(x), split heads to [B,H,S,D]
a = dense noncausal self-attention(q,k,v, encoder_mask)
x = res + dropout(Linear(C -> C, bias=True)(a))
res = x
x = LayerNorm(C)(x)
x = activation(Linear(C -> encoder_ffn_dim, bias=True)(x))
x = Linear(encoder_ffn_dim -> C, bias=True)(dropout(x))
x = res + dropout(x)
if dtype fp16: x = clamp(x)
```

Final encoder output:

```text
encoder_last_hidden = LayerNorm(C)(x)
```

Decoder layer, repeated `decoder_layers`:

```text
res = y
y = LayerNorm(C)(y)
q,k,v = Linear(C -> C, bias=True), update/reuse self KV cache
y = res + dropout(causal_self_attention(q,k,v, decoder_mask))

res = y
y = LayerNorm(C)(y)
q = Linear(C -> C)(y)
k,v = Linear(C -> C)(encoder_hidden_states), update/reuse cross KV cache
y = res + dropout(cross_attention(q,k,v, encoder_mask))

res = y
y = LayerNorm(C)(y)
y = activation(Linear(C -> decoder_ffn_dim, bias=True)(y))
y = Linear(decoder_ffn_dim -> C, bias=True)(dropout(y))
y = res + dropout(y)
```

Final decoder/logits:

```text
decoder_hidden = LayerNorm(C)(y)
logits = MatMul(decoder_hidden, shared_embedding_weight.T) + final_logits_bias
```

For common Google configs: `C=1024`, `H=16`, `D=64`, `ffn_dim=4096`, `vocab=96103`.

## 6. Attention requirements

Encoder self-attention:

- Noncausal, bidirectional MHA.
- Q/K/V shapes `[B, H, S, D]`.
- Mask from `create_bidirectional_mask`; 2D padding masks are converted to bool then to backend-specific additive/block masks.
- No KV cache required for normal encoder inference.

Decoder self-attention:

- Causal MHA.
- Prefill Q/K/V `[B, H, T, D]`; decode Q `[B,H,1,D]`, cached K/V `[B,H,T_seen,D]`.
- Cache stores post-projection, pre-attention K/V. No RoPE or position transform is applied to K/V.
- Decoder position IDs and mask offsets use `past_key_values.get_seq_length()`.

Decoder cross-attention:

- Rectangular noncausal attention from decoder queries `[B,H,T_dec,D]` to encoder keys/values `[B,H,S_enc,D]`.
- `EncoderDecoderCache.cross_attention_cache` stores projected encoder K/V per decoder layer.
- After first update, `past_key_values.is_updated[layer_idx]` causes source to reuse cached cross K/V without recomputing encoder projections.

Masking/backends:

- Eager path uses additive masks before softmax.
- SDPA/Flash/Flex paths may use specialized mask representations from `masking_utils`; DinoML should canonicalize masks early for first parity and add backend-specific fusions later.
- No sliding window, block sparse, packed varlen, ALiBi, RoPE, or relative bias is implemented in Pegasus source.

## 7. Position encoding and custom math

Pegasus uses static sinusoidal embeddings with sine channels in the first half and cosine channels in the second half, unlike the common interleaved sin/cos pattern.

```python
def pegasus_sinusoidal_table(n_pos, dim):
    position_enc = [[pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)]
    sentinel = dim // 2 if dim % 2 == 0 else dim // 2 + 1
    out[:, :sentinel] = sin(position_enc[:, 0::2])
    out[:, sentinel:] = cos(position_enc[:, 1::2])
    return out
```

Precompute:

- Full encoder and decoder position tables up to `max_position_embeddings`.
- Position IDs for fixed prefill lengths.

Dynamic inputs:

- Decoder `position_ids = arange(seq_length) + past_key_values_length`.
- Generation must guard `past_key_values_length + seq_length <= max_position_embeddings`.

## 8. Preprocessing and input packing

Tokenizer/runtime ABI:

- `PegasusTokenizer` is a Unigram/SentencePiece-style tokenizer backed by HF `tokenizers`.
- Normalization replaces `\n` with spaces and collapses repeated spaces unless a precompiled SentencePiece charsmap is loaded.
- Metaspace pre-tokenization/decoding uses replacement `▁`.
- Template processing appends EOS: single `$A </s>`, pair `$A $B </s>`.
- Model inputs are `input_ids [B,S]` and `attention_mask [B,S]`.

Decoder ABI:

- Generation begins with `decoder_start_token_id`, normally `0`, same as pad/BOS in observed configs.
- During teacher-forcing labels path, `shift_tokens_right` inserts decoder start at position 0 and replaces `-100` labels with pad.
- `forced_eos_token_id=1` is generation-controller behavior, not a model graph op.

No multimodal, packed patch, cu_seqlens, tokenizer image codebook, or scatter embedding stitch is present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections to packed QKV GEMM

Source pattern:

```text
q = Linear(C -> C)(x)
k = Linear(C -> C)(kv_source)
v = Linear(C -> C)(kv_source)
```

Replacement:

```text
PackedLinear(C -> 3C) -> split [q,k,v]
```

Preconditions:

- Self-attention only has identical source tensor for Q/K/V.
- All three projections have same input/output dtype, contiguous row-major weight semantics, and bias presence.
- Split order must be exactly `[q, k, v]` if weights are packed for DinoML; source module declaration order is `k_proj`, `v_proj`, `q_proj`, but forward computes Q first. Do not infer state-dict order from module declaration.

Failure cases:

- Cross-attention Q source is decoder hidden states while K/V source is encoder hidden states; only K/V can be packed together.
- Weight aliasing or quantized external storage must preserve exact per-projection names for loading.

Parity sketch:

- Compare packed and unpacked attention projection outputs before reshape for random `[B,T,C]`.

### Rewrite: cross-attention K/V precompute

Source pattern:

```text
k_enc = k_proj(encoder_hidden_states)
v_enc = v_proj(encoder_hidden_states)
cache cross K/V after first decode step
```

Replacement:

```text
EncoderOutput -> per-decoder-layer KV projection cache -> decode cross-attention
```

Preconditions:

- Encoder hidden states fixed for the request.
- Decoder layer weights fixed and no training dropout.
- Cache invalidates when encoder output, batch order, beam expansion, or decoder layer weights change.

Failure cases:

- Beam reorder/batch select must reorder cross cache along batch dimension.
- First-token prefill with multiple decoder tokens must populate cross cache once and share it for subsequent decode.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(decoder_hidden[:, :, :]) + final_logits_bias
```

Replacement:

```text
slice decoder_hidden[:, -1:, :] -> GEMM(C, vocab) -> bias
```

Preconditions:

- Generation only needs next-token logits.
- Training/loss and full-sequence score requests are not active.
- Conditional generation source does not expose `logits_to_keep`; DinoML generation wrapper must own this optimization.

### Rewrite: frozen sinusoidal embedding to generated constant

Source pattern:

```text
PegasusSinusoidalPositionalEmbedding(max_pos, C)
```

Replacement:

```text
compile-time constant table + gather(position_ids)
```

Preconditions:

- `max_position_embeddings` and `d_model` known at compile/load time.
- Use Pegasus non-interleaved formula.
- Position table dtype matches loaded model dtype or source weight dtype.

### Rewrite: GEMM epilogues for FFN and logits

Source pattern:

```text
Linear + bias + relu/gelu
Linear + bias + residual add
LM matmul + final_logits_bias
```

Replacement:

```text
CUTLASS GEMM bias activation epilogue where available
CUTLASS GEMM bias/add epilogue for residual-safe cases
```

Preconditions:

- Static or profiled matrix shapes satisfy current DinoML GEMM layout contracts.
- Dropout/training disabled.
- Residual shape is exact `[B*T,C]` or compatible flattened dense row-major view.

Layout constraints:

- Preserve semantic `[B,T,C]`; flatten only local GEMM inputs to `[B*T,C]`.
- No channel-last rewrite is applicable.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over last dimension `[B,T,C]`: currently a DinoML gap and appears before every attention/FFN subblock plus final encoder/decoder norms.
- Dense MHA with cache: self-attention prefill/decode and cross-attention are required for generation.
- GEMM bias activation for FFN `C -> 4C -> C`, especially `relu` for production Google configs.
- Last-token-only LM GEMM for decode, with tied embedding weight and `final_logits_bias`.

Medium priority:

- Packed self-attention QKV projection and cross-attention KV projection.
- Cross-attention K/V precompute per decoder layer.
- Mask add + softmax + value matmul fused attention path.
- Embedding lookup + scale + position add for prefill.

Lower priority:

- fp16 clamp in encoder layer as a small elementwise guard.
- Beam-search cache reorder/batch select kernels.
- Full-sequence logits for scoring/training-like parity.

## 11. Runtime staging plan

Stage 1: parse Pegasus config and load tied weights.

- Admit only in-library `model_type="pegasus"`.
- Preserve `shared.weight` aliases for encoder embeddings, decoder embeddings, and conditional LM head.

Stage 2: encoder-only parity.

- Implement token/position embedding, encoder MHA, LayerNorm, FFN, final LayerNorm.
- Validate `[B,S,C] -> [B,S,C]`.

Stage 3: decoder prefill without cache.

- Add causal mask, cross-attention to encoder states, logits.

Stage 4: seq2seq decode with cache.

- Add `EncoderDecoderCache` ABI: self K/V append and cross K/V reuse.
- Add cache reorder/batch select only when beam search is enabled.

Stage 5: optimized attention and graph rewrites.

- Packed QKV/KV projections, fused attention, cross K/V precompute, last-token logits.

Stage 6: generation controller parity.

- Decoder start, forced EOS, beam search metadata, length penalty, task-specific max lengths.

Initially stub:

- Training loss, dropout, LayerDrop, gradient checkpointing, hidden/attention output capture.
- `PegasusForCausalLM`.

## 12. Parity and validation plan

- Sinusoidal table parity for odd/even `d_model` against source formula.
- Embedding parity with and without `scale_embedding`.
- Single encoder layer parity in fp32 for random hidden states and masks.
- Single decoder layer parity for prefill self-attention, cross-attention, and FFN.
- Cache parity: one prefill token plus N decode steps equals full decoder forward over same target tokens.
- Cross-cache parity: verify encoder K/V projections are computed once and reused with identical logits.
- End-to-end summarization smoke against `google/pegasus-xsum` and `google/pegasus-cnn_dailymail` with greedy generation.
- Tolerances: fp32 `atol=1e-5, rtol=1e-4`; fp16/bf16 attention/logits `atol=1e-2, rtol=1e-2` unless fused attention requires looser documented backend tolerance.

No validation was run for this audit.

## 13. Performance probes

- Encoder throughput sweep over `B` and `S` for 512 and 1024 source lengths.
- Decoder prefill throughput over target prefix length.
- Decode tokens/sec with cross-cache enabled versus recomputing cross K/V.
- KV cache memory usage: self cache `2 * decoder_layers * B * H * T_dec * D * dtype_size`; cross cache `2 * decoder_layers * B * H * S_enc * D * dtype_size`.
- Attention backend comparison: eager decomposed, SDPA-style fused, FlashAttention-style fused if masks/cache are admissible.
- GEMM profile sweep for FFN and LM head, including vocab 96103.
- Last-token logits versus full-sequence logits in decode.
- Beam search cache reorder overhead.

## 14. Skip/defer list

- Training, loss, dropout, LayerDrop, gradient checkpointing.
- Attention weights/hidden-state recording.
- `PegasusForCausalLM` until seq2seq generation is stable.
- Beam search initially, except ABI design should not preclude cache reorder.
- Quantized or packed weight formats; none are source-coupled for Pegasus.
- Remote-code variants; none inspected or required.
- Layout translation/NHWC/channel-last work; no vision tensors.

## 15. Final implementation checklist

- [ ] Parse `PegasusConfig` with checkpoint override dimensions.
- [ ] Reject or ignore historical config fields not read by current source.
- [ ] Load shared token embeddings and preserve tied LM-head alias.
- [ ] Generate/load Pegasus non-interleaved sinusoidal position tables.
- [ ] Implement embedding scale/add path.
- [ ] Implement/port last-dim `LayerNorm`.
- [ ] Implement dense MHA self-attention and cross-attention.
- [ ] Implement encoder bidirectional mask and decoder causal mask contracts.
- [ ] Implement `EncoderDecoderCache` self K/V append and cross K/V reuse.
- [ ] Add FFN `relu` and `gelu` activation parity.
- [ ] Lower FFN and LM head through CUTLASS GEMM where admissible.
- [ ] Add packed QKV/KV rewrite with strict split-order tests.
- [ ] Add last-token-only logits rewrite for generation.
- [ ] Add encoder, decoder prefill, and decode-step parity tests.
- [ ] Add performance probes for encoder, prefill, decode, logits, and cache memory.
