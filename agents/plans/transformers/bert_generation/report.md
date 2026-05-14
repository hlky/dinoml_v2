# BertGeneration DinoML Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from X:/H/transformers
Model id: google/bert_for_seq_generation_L-24_bbc_encoder plus public tiny/community bert-generation configs
Config source: Hugging Face raw config.json URLs, fetched 2026-05-13
Source files inspected:
  X:/H/transformers/src/transformers/models/bert_generation/configuration_bert_generation.py
  X:/H/transformers/src/transformers/models/bert_generation/modeling_bert_generation.py
  X:/H/transformers/src/transformers/models/bert_generation/tokenization_bert_generation.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/src/transformers/cache_utils.py
Any missing files or assumptions:
  No processor/image/audio path exists. Tokenizer is SentencePiece-only and model-coupled only through token IDs.
  The only official public bert-generation checkpoint found by HF API search is the Google encoder checkpoint.
```

See `_sources/config_sweep.md` for config snapshots and `_sources/source_notes.md` for source anchors.

## 2. High-level architecture

`bert_generation` is a BERT-style text Transformer that can run as:

- Base encoder: bidirectional self-attention, no LM head, no KV cache.
- Causal LM decoder: same block with causal self-attention and a tied LM head.
- Seq2seq decoder: causal self-attention plus optional encoder-decoder cross-attention.

First useful DinoML target: encoder-only and causal decoder prefill/decode for `BertGenerationDecoder`. Seq2seq cross-attention is source-supported but should be a separate gate because it adds a second cache family and encoder-state ABI.

```text
SentencePiece tokenization -> input_ids/attention_mask
-> token + learned position embeddings
-> N Transformer blocks
-> encoder hidden states OR decoder hidden states
-> optional last-token/subset LM head -> logits/generation controller
```

Seq2seq decomposition:

```text
external encoder_hidden_states/cacheable encoder output
-> decoder token embeddings
-> causal decoder self-attention/cache
-> cross-attention over encoder K/V/cache
-> MLP blocks
-> LM head logits
```

## 3. Important config dimensions

| Field | Source default | Google L-24 encoder | Tiny random | Cross-attn example |
|---|---:|---:|---:|---:|
| `vocab_size` | 50358 | 50358 | 1024 | 50358 |
| `hidden_size` | 1024 | 1024 | 36 | 1024 |
| `num_hidden_layers` | 24 | 24 | 6 | 24 |
| `num_attention_heads` | 16 | 16 | 6 | 16 |
| inferred `head_dim` | 64 | 64 | 6 | 64 |
| `intermediate_size` | 4096 | 4096 | 62 | 4096 |
| `max_position_embeddings` | 512 | 512 | 512 | 512 |
| `hidden_act` | `gelu` | `gelu` | `gelu` | `gelu` |
| `layer_norm_eps` | `1e-12` | `1e-12` | `1e-12` | `1e-12` |
| `is_decoder` | false | omitted -> false | omitted -> false | true |
| `add_cross_attention` | false | omitted -> false | omitted -> false | true |
| `use_cache` | true | omitted -> true | true | true |
| tokenizer | SentencePiece | `spiece.model` expected | not verified | not verified |

Representative configs:

| Model id | Architecture field | Operator-significant notes |
|---|---|---|
| `google/bert_for_seq_generation_L-24_bbc_encoder` | older `BertForSeqGenerationEncoderModel` | Official encoder checkpoint; config omits current decoder/cache booleans, so source defaults apply. |
| `ybelkada/random-tiny-BertGenerationModel` | `BertGenerationEncoder` | Small debug shape; validates non-power-of-two head dim 6 and intermediate size 62. |
| `Zlovoblachko/testik_L1_sent_generator` | `BertGenerationDecoder` | Decoder architecture but omits `is_decoder`; must be rejected or normalized before causal generation. |
| `ammonbro/bert_sp_updown` | `BertGenerationDecoder` | Same decoder-without-`is_decoder` trap. |
| `YijunYang280/GuardT2I` | `BertGenerationDecoder` | `is_decoder=true`, `add_cross_attention=true`; first observed cross-attention config. |

## 3a. Family variation traps

- `architectures=["BertGenerationDecoder"]` is not enough for causal behavior. Source gates causal masks/cache on `config.is_decoder`.
- Cross-attention requires both `is_decoder=True` and `add_cross_attention=True`; otherwise passing `encoder_hidden_states` raises.
- Current source uses learned absolute positions only; historical `position_embedding_type` is ignored by this source basis.
- `directionality`, `gradient_checkpointing`, and community `return_scores` are config residue for inference graph purposes.
- `hidden_size % num_attention_heads == 0` is enforced; no GQA/MQA/head_dim override exists.
- All projections have bias because source uses `nn.Linear` defaults.
- LM head weights are tied to token embeddings by `_tied_weights_keys`; lowering must preserve the alias contract.
- `chunk_size_feed_forward` is inherited and may split the sequence dimension. First integration should require `chunk_size_feed_forward == 0`.
- No NCHW/NHWC layout issue exists; tensors are text sequences `[B, T, H]`. No layout translation should be applied around sequence axes.

## 4. Operator coverage checklist

Tensor/layout ops:
- Integer token embedding gather `[B,T] -> [B,T,H]`.
- Learned position embedding gather from sliced `position_ids`.
- Add token and position embeddings.
- Reshape/view `[B,T,H] -> [B,T,heads,head_dim]`, transpose to `[B,heads,T,head_dim]`, transpose back, contiguous/reshape.
- Slice hidden states for `logits_to_keep`: last `k` tokens or tensor indices.

Neural primitives:
- `LayerNorm(H, eps=1e-12)` after embeddings, attention residual, and MLP residual.
- Dense GEMM/Linear with bias:
  - Q/K/V: `H -> H`.
  - Attention output: `H -> H`.
  - MLP up: `H -> I`.
  - MLP down: `I -> H`.
  - LM head: `H -> vocab_size`.
- GELU activation.
- Residual add.
- Dropout is inference-disabled.

Attention primitives:
- Dense MHA, no GQA/MQA.
- Bidirectional self-attention for encoder mode.
- Causal self-attention for decoder mode.
- Optional dense cross-attention with decoder queries and encoder K/V.
- Attention backend dispatch advertises eager, SDPA, FlashAttention, and FlexAttention support; DinoML should first lower a single explicit mask/cache ABI and then enable optimized backends.

Generation/cache ops:
- Dynamic self-attention KV cache per layer, tensors `[B, heads, cache_T, head_dim]`.
- Encoder-decoder cache with separate self-attention and cross-attention caches.
- Cross-attention cache has per-layer `is_updated` flag; once encoder K/V are projected, subsequent decode steps reuse them.
- Cache reorder for beam search belongs to the generation controller and can be deferred initially.

Preprocessing-coupled ops:
- SentencePiece tokenization emits `input_ids` and `attention_mask`; `token_type_ids` are not model inputs.
- BOS/EOS/PAD IDs are tokenizer/generation ABI, not neural ops.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids [B,T] or inputs_embeds [B,T,H]
position_ids default = arange(max_position_embeddings)[past_len:past_len+T]
x = word_embedding(input_ids) + position_embedding(position_ids)
x = LayerNorm(x)
```

Encoder/decoder block, repeated `N` times:

```text
q = Linear(H -> H)(x).view(B,T,heads,head_dim).transpose(1,2)
k = Linear(H -> H)(x).view(B,T,heads,head_dim).transpose(1,2)
v = Linear(H -> H)(x).view(B,T,heads,head_dim).transpose(1,2)
if self_cache: k,v = cache.update(k,v, layer_idx)
a = Attention(q,k,v, mask, scale=head_dim**-0.5)
a = a.transpose(1,2).reshape(B,T,H)
x = LayerNorm(Linear(H -> H)(a) + residual)
if decoder_cross_attention and encoder_hidden_states:
  q = Linear(H -> H)(x)
  k,v = Linear(H -> H)(encoder_hidden_states), cached after first use
  x = LayerNorm(Linear(H -> H)(cross_attention(q,k,v)) + x)
m = GELU(Linear(H -> I)(x))
x = LayerNorm(Linear(I -> H)(m) + x)
```

Decoder LM head:

```text
selected = hidden_states[:, slice_or_indices, :]
logits = Linear(H -> vocab_size)(selected)
```

## 6. Attention requirements

Required variants:

| Mode | Attention | Mask | Cache |
|---|---|---|---|
| Encoder | self MHA | bidirectional padding mask or 4D mask | none |
| Causal decoder | self MHA | causal + optional padding mask | dynamic self KV |
| Seq2seq decoder | self MHA + cross MHA | causal self mask + bidirectional encoder mask | dynamic self KV + cached cross K/V |

Shape contract:
- Q/K/V before attention: `[B, heads, q_or_kv_T, head_dim]`.
- Attention scores: `[B, heads, q_T, kv_T]`.
- No sliding-window, sparse, ALiBi, RoPE, relative bias, packed varlen metadata, or MQA/GQA is implemented in this family.
- Mask helpers may return `None` for backend skip, or backend-specific mask tensors. First DinoML lowering should use explicit additive masks or a clearly typed SDPA/FlashAttention contract.
- Cached keys are stored after linear projection and before attention score matmul. No position encoding is applied inside attention.
- Cross-attention K/V are projected from encoder hidden states and stored in `cross_attention_cache`; they are not generated token by token.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding only.

```python
def bert_generation_positions(position_ids, past_key_values_length, seq_length):
    if position_ids is None:
        position_ids = arange(max_position_embeddings)[past_key_values_length:seq_length + past_key_values_length]
    return position_embedding(position_ids)
```

Precompute candidates:
- The full learned position embedding table is a constant.
- Default `position_ids` for common static/bucketed decode positions can be generated as integer ranges.

Dynamic guards:
- `past_key_values_length + seq_length <= max_position_embeddings`.
- Caller-provided `position_ids` must be in range and match `[B,T]` or broadcast-compatible source behavior.

## 8. Preprocessing and input packing

CPU/data pipeline:
- `BertGenerationTokenizer` uses SentencePiece with `spiece.model`.
- Tokenizer model inputs are `input_ids` and `attention_mask`.
- Default special tokens: BOS `<s>` id 2, EOS `</s>` id 1, UNK `<unk>`, PAD `<pad>` id 0, SEP `<::::>`.

GPU/runtime graph:
- `input_ids [B,T]`, optional `attention_mask [B,T_or_seen_plus_T]`, optional `position_ids [B,T]`, or `inputs_embeds [B,T,H]`.
- Exactly one of `input_ids` or `inputs_embeds` is required.
- Cross-attention mode additionally takes `encoder_hidden_states [B,S,H]` and optional `encoder_attention_mask [B,S]`.

No multimodal placeholder scatter, image/audio packing, token type IDs, bbox/layout inputs, or packed sequence metadata are required for this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection packing

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
packed = Linear(H, 3H)(x)
q,k,v = split(packed, [H,H,H], dim=-1)
```

Preconditions:
- Self-attention only.
- All Q/K/V consume the same `hidden_states`.
- All three projections have bias.
- Weight packing order is `[Q rows, K rows, V rows]` in output-feature dimension.

Failure cases:
- Cross-attention cannot pack decoder Q with encoder K/V into one GEMM; K/V may be packed separately over `encoder_hidden_states`.

Parity test sketch:
- Compare unpacked Q/K/V tensors and final attention output for random `B,T,H`.

### Rewrite: cross-attention KV preprojection cache

Source pattern:

```text
if not is_updated[layer]:
  k = Linear(H,H)(encoder_hidden_states)
  v = Linear(H,H)(encoder_hidden_states)
  cache.store(k,v)
```

Replacement:

```text
encoder_kv_cache[layer] = packed Linear(H,2H)(encoder_hidden_states)
decode steps reuse cached K/V
```

Preconditions:
- Encoder hidden states are immutable for the decode session.
- `encoder_attention_mask` is stable across decode steps.
- Cross-attention layer weights are fixed and loaded.

Failure cases:
- Changing encoder states/mask requires cache reset.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, -k:, :])
```

Replacement:

```text
gather/slice last k hidden rows -> GEMM(H, vocab)
```

Preconditions:
- Generation only needs the next-token logits, usually `logits_to_keep=1`.
- Loss is not computed.

Failure cases:
- Full-sequence logits requested, tensor index `logits_to_keep`, or training labels.

### Rewrite: residual Linear + LayerNorm fusion

Source pattern:

```text
y = Linear(x)
out = LayerNorm(y + residual)
```

Replacement:

```text
GEMM bias -> residual add -> LayerNorm fused epilogue/kernel
```

Preconditions:
- Inference dropout disabled.
- Dense contiguous `[B,T,H]` or flattened `[B*T,H]`.

Failure cases:
- Nonzero dropout/training or unsupported dynamic strides.

## 10. Kernel fusion candidates

Highest priority:
- LayerNorm with residual add for attention and MLP outputs; this occurs twice per layer and once in embeddings.
- QKV packed projection for self-attention; reduces three GEMMs to one logical provider call.
- SDPA/FlashAttention path for dense MHA with explicit bidirectional/causal masks and cache offsets.
- Last-token-only LM head for decode; avoids full sequence-vocab GEMM.

Medium priority:
- Cross-attention K/V packing and per-session preprojection cache.
- GELU MLP fusion: GEMM -> GELU -> GEMM scheduling, with possible activation kernel fusion.
- Embedding + position add + LayerNorm fusion for encoder/prefill.

Lower priority:
- Feed-forward chunking support; better to reject nonzero chunks first.
- Beam cache reorder; generation-controller feature after greedy/sampling decode parity.
- FlexAttention-specific paths; source advertises support, but dense MHA covers normal configs.

## 11. Runtime staging plan

1. Parse `BertGenerationConfig`; reject unsupported historical flags only when they alter required behavior.
2. Load encoder weights and run embeddings plus one encoder block with bidirectional mask.
3. Implement full encoder-only parity for the Google and tiny configs.
4. Add decoder mode gate requiring `is_decoder=True`; implement causal prefill without cache first.
5. Add dynamic self-attention KV cache for incremental decode.
6. Add tied LM head and `logits_to_keep=1` path.
7. Add seq2seq decoder cross-attention with explicit `encoder_hidden_states` ABI and cross K/V cache.
8. Enable optimized attention backends behind mask/cache/layout guards.

Initially stub or reject:
- Training/loss/labels.
- Nonzero dropout/training.
- `chunk_size_feed_forward != 0`.
- Beam cache reorder.
- Tensor-valued `logits_to_keep` until indexed gather is validated.

## 12. Parity and validation plan

- Config parser tests for official Google config omissions and decoder configs that omit `is_decoder`.
- Embedding parity: `input_ids` and explicit `position_ids`, including decode offset.
- Single self-attention block parity for encoder mask and causal mask.
- Full encoder parity for tiny random config in fp32.
- Decoder prefill logits parity for `is_decoder=True`.
- Decode step parity with self KV cache: prefill `T`, then one token; compare logits and cache shapes.
- Cross-attention parity: fixed random `encoder_hidden_states`, first decode step fills cross cache, second reuses it.
- LM head tied-weight alias test: one logical embedding/decoder weight.

Recommended tolerances:
- fp32: `rtol=1e-4`, `atol=1e-5`.
- fp16/bf16 optimized paths: start with `rtol=5e-2`, `atol=5e-2` for logits, tighten per kernel.

## 13. Performance probes

- Encoder throughput sweep: batch size and sequence length up to 512.
- Decoder prefill throughput: `[B,T]` sweep, causal mask overhead separated from attention kernel time.
- Decode tokens/sec: `B`, cache length, and `logits_to_keep=1` versus full logits.
- KV cache memory: `layers * 2 * B * heads * T * head_dim * dtype_size`; cross-attention adds another fixed K/V pair per layer.
- Attention backend comparison: eager-equivalent matmul/softmax, SDPA, FlashAttention where mask form permits.
- QKV packed versus separate GEMM projection.
- Cross-attention preprojection cost amortized over decode length.
- LM head GEMM cost with vocab 50358 and last-token-only slicing.

## 14. Skip/defer list

- Training, labels, and loss.
- Dropout behavior.
- Gradient checkpointing.
- Feed-forward chunking unless a checkpoint requires nonzero chunk size.
- Beam search cache reorder and advanced generation processors.
- FlexAttention-specific lowering.
- Tensor-valued `logits_to_keep`.
- Community configs with `architectures=Decoder` but no `is_decoder=True`, unless DinoML intentionally normalizes them.
- Cross-attention until encoder-only and decoder self-cache are stable.

## 15. Final implementation checklist

- [ ] Parse `BertGenerationConfig` and apply source defaults for omitted fields.
- [ ] Add admission guards for `hidden_size % num_attention_heads == 0`.
- [ ] Reject decoder generation when `is_decoder` is false, even if architecture says decoder.
- [ ] Load embeddings, learned positions, LayerNorm, Linear+bias, and tied LM head weights.
- [ ] Implement encoder bidirectional attention mask path.
- [ ] Implement causal decoder mask path with cache offset.
- [ ] Implement dynamic self-attention KV cache `[B, heads, T, head_dim]`.
- [ ] Implement optional encoder-decoder cross-attention cache with `is_updated` semantics.
- [ ] Add QKV packed projection rewrite for self-attention.
- [ ] Add K/V packed preprojection rewrite for cross-attention.
- [ ] Add last-token-only logits rewrite.
- [ ] Add LayerNorm/residual fusion candidates.
- [ ] Add parity tests for tiny encoder, Google encoder, decoder prefill, decode cache, and cross-attention.
- [ ] Benchmark encoder, prefill, decode, cache memory, attention backend, and LM head slices.
