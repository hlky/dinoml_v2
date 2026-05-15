# mT5 Transformers Family Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: representative official checkpoints `google/mt5-small`, `google/mt5-base`, `google/mt5-large`, `google/mt5-xl`, `google/mt5-xxl`

Config source: fetched public Hugging Face `config.json`, `tokenizer_config.json`, and `special_tokens_map.json` snapshots under `_sources/`.

Source files inspected:

- `transformers/src/transformers/models/mt5/configuration_mt5.py`
- `transformers/src/transformers/models/mt5/modeling_mt5.py`
- `transformers/src/transformers/models/t5/tokenization_t5.py`
- `transformers/src/transformers/models/auto/tokenization_auto.py`
- comparison basis only: copied-from comments and source-equivalent mT5 classes from T5; this report does not assume T5 equivalence unless the mT5 source explicitly copies or maps to it.

Source URLs:

- [configuration_mt5.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mt5/configuration_mt5.py)
- [modeling_mt5.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mt5/modeling_mt5.py)
- [tokenization_t5.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/t5/tokenization_t5.py)
- [tokenization_auto.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/tokenization_auto.py)

Any missing files or assumptions:

- No `src/transformers/models/mt5/tokenization_mt5.py` or fast tokenizer file exists in this checkout. `AutoTokenizer` maps `mt5` to `T5Tokenizer` when tokenizers is available.
- Public configs were accessible; no gated model gaps were encountered.
- Primary runtime target for DinoML: `MT5ForConditionalGeneration` text-to-text seq2seq inference, including encoder prefill, decoder prefill, autoregressive decode, logits, and encoder-decoder cache ABI.

## 2. High-level architecture

mT5 is a text-only encoder-decoder Transformer. Both encoder and decoder use T5-style pre-norm RMSNorm, relative position bias, bias-free attention projections, bias-free feed-forward projections, shared token embeddings, and a tied LM head for generation.

Dataflow:

```text
SentencePiece/T5 tokenizer -> input_ids + attention_mask
  -> shared token embedding
  -> encoder stack with bidirectional self-attention + shared relative bias
  -> decoder stack with causal self-attention + cross-attention over encoder states
  -> lm_head tied to shared embedding -> vocab logits -> generation controller
```

Stage decomposition:

- CPU/data pipeline: tokenizer backend, SentencePiece/Unigram vocabulary, EOS insertion, padding masks.
- Cacheable encoder stage: encoder hidden states `[batch, src_len, d_model]` and source attention mask can be reused across decode steps.
- Decoder prefill: causal decoder self-attention over prompt/shifted target tokens plus cross-attention to encoder states.
- Decode: one or more new decoder tokens update self-attention KV cache; cross-attention K/V are computed once and then reused via `EncoderDecoderCache.is_updated`.
- Logits: `lm_head(sequence_output)` over the full decoder sequence or decode step.

Implemented source heads:

- Required for primary target: `MT5ForConditionalGeneration`.
- Optional/deferred: bare `MT5Model`, `MT5EncoderModel`.
- Deferred task heads: sequence classification, token classification, question answering. They reuse the same body but add EOS pooling or small classifier/QA projections.

## 3. Important config dimensions

Source defaults from `MT5Config`:

| Field | Default / behavior |
|---|---:|
| `vocab_size` | 250112 |
| `d_model` | 512 |
| `d_kv` / head dim | 64 |
| `d_ff` | 1024 |
| `num_layers` | 8 |
| `num_decoder_layers` | defaults to `num_layers` when omitted |
| `num_heads` | 6 |
| `relative_attention_num_buckets` | 32 |
| `relative_attention_max_distance` | 128 default when omitted from checkpoint config |
| `feed_forward_proj` | `gated-gelu`; normalized to `dense_act_fn="gelu_new"` and `is_gated_act=True` |
| `dropout_rate` | 0.1, inference-disabled |
| `layer_norm_epsilon` | 1e-6 |
| `use_cache` | True |
| `pad_token_id` / `eos_token_id` / `decoder_start_token_id` | 0 / 1 / 0 |
| `tie_word_embeddings` | forced True by `MT5Config.__post_init__`, even when serialized configs say false |

Representative checkpoint sweep from fetched official configs:

| Checkpoint | Source arch field | `d_model` | `d_ff` | Heads | `d_kv` | Enc layers | Dec layers | Inner QKV dim | Vocab | Tokenizer class field |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `google/mt5-small` | `MT5ForConditionalGeneration` | 512 | 1024 | 6 | 64 | 8 | 8 | 384 | 250112 | `T5Tokenizer` |
| `google/mt5-base` | `MT5ForConditionalGeneration` | 768 | 2048 | 12 | 64 | 12 | 12 | 768 | 250112 | `T5Tokenizer` |
| `google/mt5-large` | `MT5ForConditionalGeneration` | 1024 | 2816 | 16 | 64 | 24 | 24 | 1024 | 250112 | `T5Tokenizer` |
| `google/mt5-xl` | `MT5ForConditionalGeneration` | 2048 | 5120 | 32 | 64 | 24 | 24 | 2048 | 250112 | `T5Tokenizer` |
| `google/mt5-xxl` | `T5ForConditionalGeneration` | 4096 | 10240 | 64 | 64 | 24 | 24 | 4096 | 250112 | `T5Tokenizer` |

The `google/mt5-xxl` config has `model_type="mt5"` but `architectures=["T5ForConditionalGeneration"]`. DinoML should dispatch by `model_type`/loaded class, not by the historical `architectures` string alone.

## 3a. Family variation traps

- `d_model != num_heads * d_kv` for `mt5-small`: hidden size is 512 but attention inner width is 384. Do not infer attention projection width from hidden size.
- Checkpoint configs serialize `tie_word_embeddings: false`, but the current config class discards that field and forces `tie_word_embeddings=True`. Treat `shared.weight`, encoder embeddings, decoder embeddings, and LM head as one logical tied parameter for generation.
- `relative_attention_max_distance` is omitted by the official configs inspected; the effective source default is 128.
- Relative position bias exists only in block 0 of each stack and is passed forward as a shared `position_bias` tensor across layers. Encoder block 0 and decoder block 0 have distinct bias tables; decoder cross-attention has no learned relative bias.
- `feed_forward_proj` can be `gated-*` or plain activation by config validation. Official mT5 checkpoints use `gated-gelu`, which means two input projections (`wi_0`, `wi_1`) and multiply.
- Tokenizer coupling is multilingual SentencePiece/Unigram through T5 tokenizer code. There is no mT5-local tokenizer implementation in this checkout; AutoTokenizer maps `mt5` to `T5Tokenizer`.
- `extra_ids` in fetched tokenizer configs are 0 for official mT5 checkpoints, unlike many T5 defaults that add 100 sentinels. Do not assume sentinel vocabulary beyond the model vocab/config.
- Left or right padding is allowed because the model uses relative position embeddings and masks rather than absolute position ids.
- No RoPE, ALiBi, sliding window, GQA/MQA, MoE, packed QKV weights, or vision/audio branches are implemented in this family.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids) -> [B, S, d_model]` with shared encoder/decoder/LM-head weight identity.
- `view`, `reshape`, `transpose(1, 2)`, `transpose(3, 2)`, `permute([2,0,1])`, `contiguous`.
- Mask construction: bidirectional encoder/cross masks and causal decoder masks with cache length.
- Slice mask to current key length: `mask[:, :, :, :key_len]`.
- Label shift-right: fill first token with `decoder_start_token_id`, shift labels, replace `-100` with pad id.
- EOS pooling and `unique_consecutive` only for optional sequence-classification head.

Neural primitives:

- Bias-free dense projections: `q/k/v: Linear(d_model -> num_heads*d_kv)`, `o: Linear(num_heads*d_kv -> d_model)`.
- Official FFN: `wi_0: Linear(d_model -> d_ff)`, `gelu_new`, `wi_1: Linear(d_model -> d_ff)`, elementwise multiply, `wo: Linear(d_ff -> d_model)`.
- RMSNorm/T5LayerNorm: fp32 variance over last dim, `rsqrt`, scale-only weight, no mean subtraction, no bias.
- Residual adds around self-attention, cross-attention, and FFN.
- Dropout is present in source but disabled for inference.
- Optional fp16 clamp after block sublayers exists in source; first inference integration can reject or de-prioritize this unless matching fp16 overflow behavior is required.

Attention primitives:

- Dense multi-head attention, no GQA/MQA: Q/K/V all shaped `[B, heads, T, d_kv]`.
- Encoder bidirectional self-attention.
- Decoder causal self-attention with autoregressive cache.
- Decoder cross-attention over encoder hidden states with cross-attention cache.
- Scores are `matmul(q, k.transpose(-1, -2))`; no explicit `1/sqrt(d_kv)` scaling in source.
- Add position bias plus mask before softmax.
- Softmax computes in fp32 and casts back to scores dtype.

Position/relative-bias ops:

- Integer relative bucket generation using arange, subtract, abs/min/where/log/floor-cast/min.
- Learned bias embedding `[num_buckets, num_heads]`, permuted to `[1, heads, q_len, k_len]`.
- `past_seen_tokens` offset for decoder self-attention bias during cached decode.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` when decoder cache is requested.
- Self-attention cache update appends or mutates per-layer K/V.
- Cross-attention cache stores projected encoder K/V once and reuses when `is_updated[layer_idx]` is true.
- Encoder outputs are independently cacheable across decode.
- Generation controller behavior comes from `GenerationMixin`; no mT5-local logits processor is required by source.

Preprocessing-coupled ops:

- T5 tokenizer Unigram/SentencePiece vocabulary file `spiece.model`.
- WhitespaceSplit + Metaspace pre-tokenization in the inspected tokenizer backend.
- Template post-processing appends `</s>` to single and pair sequences.
- Model input names are `input_ids`, `attention_mask`.

## 5. Layer/block breakdown

Encoder block, repeated `num_layers` times:

```text
x0: [B, src_len, d_model]
n = RMSNorm(x0)
q,k,v = Linear(n) -> [B, heads, src_len, d_kv]
bias = relative_attention_bias(src_len, src_len, bidirectional=True) only in block 0; later blocks reuse it
attn = softmax_fp32(q @ k^T + bias + encoder_mask)
x1 = x0 + Linear(attn @ v)
y = RMSNorm(x1)
ff = Linear(gelu_new(Linear(y)) * Linear(y))
out = x1 + ff
```

Decoder block, repeated `num_decoder_layers` times:

```text
x0: [B, tgt_len, d_model]
n = RMSNorm(x0)
q,k,v = self-attn Linear(n), update/reuse self KV cache
self_bias = relative_attention_bias(tgt_len, key_len, bidirectional=False, past_seen_tokens) only in block 0; later blocks reuse it
x1 = x0 + self_attention(q, k, v, causal_mask + self_bias)
c = RMSNorm(x1)
q = Linear(c); k,v = Linear(encoder_hidden_states), update/reuse cross-attn cache
x2 = x1 + cross_attention(q, k, v, encoder_mask)  # learned cross relative bias absent
y = RMSNorm(x2)
ff = Linear(gelu_new(Linear(y)) * Linear(y))
out = x2 + ff
```

Generation head:

```text
sequence_output: [B, tgt_len, d_model]
logits = sequence_output @ shared_embedding.T
```

No attention or FFN projection has bias in the primary body. Optional classifier/QA heads add biased dense layers.

## 6. Attention requirements

Attention variants required:

- Encoder self-attention: noncausal, bidirectional relative bias, mask shape broadcastable to `[B, heads, src_len, src_len]`.
- Decoder self-attention: causal, unidirectional relative bias, cache-aware key length, mask shape `[B, 1, tgt_len, key_len]`.
- Decoder cross-attention: rectangular query/key lengths, no learned relative bias, bidirectional source mask, cross K/V cache.

Shapes:

- Input hidden states: `[B, Q, d_model]`.
- Projection width: `inner_dim = num_heads * d_kv`, not always `d_model`.
- Q/K/V after reshape: `[B, num_heads, Q_or_K, d_kv]`.
- Scores: `[B, num_heads, Q, K]`.
- Cache per layer before any repeat expansion: keys and values are `[B, num_heads, cached_len, d_kv]`. There is no repeat expansion because this is MHA, not GQA/MQA.

Mask/math order:

```text
scores = q @ k.transpose(-1, -2)
position_bias = learned_or_zero_bias
position_bias = position_bias + mask[:, :, :, :key_len]
scores = scores + position_bias
attn = softmax(scores.float(), dim=-1).type_as(scores)
out = attn @ v
```

Cache ABI:

- `MT5Stack` creates `EncoderDecoderCache(DynamicCache(config), DynamicCache(config))` for decoder use when `past_key_values` is absent and `use_cache=True`.
- Self-attention uses `past_key_values.self_attention_cache`.
- Cross-attention uses `past_key_values.cross_attention_cache` and `is_updated[layer_idx]` to avoid re-projecting encoder states after the first decode step.
- Encoder stack forcibly clears `past_key_values=None`.

FlashAttention/SDPA compatibility:

- Dense attention can be lowered to a fused attention backend only if it supports additive per-head relative bias and masks, no explicit Q scaling, fp32 softmax accumulation, causal decode with cache length, and rectangular cross-attention.
- Cross-attention has zero learned position bias, so it is an easier fused-attention candidate after encoder K/V projection caching is available.

## 7. Position encoding and custom math

mT5 uses learned relative attention bias. No absolute position embeddings are passed as runtime inputs, and no RoPE/ALiBi is present.

Source-derived pseudocode:

```python
def mt5_relative_bucket(relative_position, bidirectional, num_buckets=32, max_distance=128):
    bucket = 0
    if bidirectional:
        num_buckets //= 2
        bucket += (relative_position > 0).long() * num_buckets
        relative_position = abs(relative_position)
    else:
        relative_position = -min(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        log(relative_position.float() / max_exact)
        / log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = min(large, num_buckets - 1)
    return bucket + where(is_small, relative_position, large)
```

For encoder self-attention, `relative_position = memory_position - context_position`. For decoder self-attention during cached decode, `context_position` is offset by `past_seen_tokens`. Bias can be precomputed for static `(query_length, key_length, past_seen_tokens)` cases, but decode step bias depends on the current cache length.

## 8. Preprocessing and input packing

Tokenizer/runtime input contract:

- Tokenizer class: `T5Tokenizer` via `AutoTokenizer` mapping for `mt5`.
- Backend in this commit is tokenizers `Unigram` with optional SentencePiece-derived vocab, `unk_id=2`, `byte_fallback=False`.
- Special ids in configs: `<pad>=0`, `</s>=1`, `<unk>=2`; fetched mT5 tokenizer configs set `extra_ids=0`.
- Tokenizer appends `</s>` to single and pair inputs through `TemplateProcessing`.
- Model consumes `input_ids: [B, src_len]`, optional `attention_mask: [B, src_len]`, optional `decoder_input_ids: [B, tgt_len]`, optional `decoder_attention_mask`.
- For training/teacher-forced generation with `labels`, source creates decoder inputs by shifting labels right and replacing `-100` with pad.

CPU/data-pipeline work:

- SentencePiece/Unigram tokenization, Unicode normalization from tokenizer model/config, padding/truncation, EOS insertion.

GPU/runtime work:

- Embedding lookup, mask expansion/causal mask construction, encoder/decoder forward, cache update, logits.

There is no multimodal packing, no token type IDs, no position IDs, no packed varlen descriptors, and no placeholder scatter path.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared relative bias precompute

Source pattern: block 0 computes bucket ids and bias; later layers reuse the `position_bias` tensor.

Replacement: precompute bucket id tensors for static or bucketed `(q_len, k_len, bidirectional, past_seen_tokens)` and lower bias lookup + permute once per stack per call.

Preconditions:

- `relative_attention_num_buckets`, `relative_attention_max_distance`, `is_decoder`, `q_len`, `k_len`, and decode cache offset are known for the case.
- Bias weight is the block-0 stack parameter.

Failure cases:

- Dynamic decode positions without a bucketed/past-length guard.
- Output attentions requiring exact intermediate bias exposure should still use the same tensor.

Parity test sketch: compare bucket ids and final attention logits for encoder full length, decoder prefill, and decoder one-token decode with nonzero cache length.

### Rewrite: gated FFN fusion

Source pattern:

```text
gelu_new(x @ wi_0.T) * (x @ wi_1.T) -> wo
```

Replacement: two-GEMM grouped/fused projection plus fused activation multiply, then output GEMM.

Preconditions:

- `feed_forward_proj` begins with `gated-`; official mT5 uses `gated-gelu`.
- Bias-free weights with shapes `[d_ff, d_model]`, `[d_ff, d_model]`, `[d_model, d_ff]`.
- Preserve source `gelu_new` approximation.

Failure cases:

- Non-gated config routes through single `wi` projection.
- Quantized or split weight layouts need separate admission.

### Rewrite: tied LM head as embedding transpose

Source pattern: `lm_head: Linear(d_model -> vocab_size, bias=False)` with forced weight tying.

Replacement: use shared embedding storage transposed for logits GEMM; keep a single logical constant.

Preconditions:

- `MT5Config.__post_init__` has enforced `tie_word_embeddings=True`.
- Weight loader preserves aliasing among `shared.weight`, encoder/decoder embeddings, and `lm_head.weight`.

Failure cases:

- External checkpoint with physically untied tensors but `model_type=mt5`; reject or canonicalize with explicit warning.

### Rewrite: cross-attention K/V cache materialization

Source pattern: first decode step projects encoder hidden states for each decoder layer cross-attention, then reuses cached K/V.

Replacement: after encoder stage, project all decoder-layer cross K/V once into cache buffers before token decode.

Preconditions:

- Encoder hidden state and encoder mask are unchanged for the request.
- Cross-attention weights are loaded and per-layer.

Failure cases:

- Caller passes new `encoder_outputs` or mask mid-generation.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm scale-only with fp32 accumulation: appears before every attention/FFN sublayer plus final stack norm.
- Dense attention with additive relative bias: central bottleneck; needs bias/mask support and exact no-scale math.
- Gated GELU FFN: two input projections, activation multiply, output projection; large FLOP share.
- Tied embedding logits GEMM: huge vocab 250112 dominates decode for small batch and should support last-token-only logits.

Medium priority:

- Relative bucket/bias generation: avoid repeated arange/log/where work, especially for decode.
- Cross-attention K/V preprojection and cache fill: separates encoder reuse from token decode.
- Q/K/V projection grouping for self-attention where weights remain separate but can be scheduled together.

Lower priority:

- Optional task heads: classifier/QA heads are not needed for first seq2seq generation parity.
- fp16 clamp behavior: preserve as an edge-case parity feature after main inference path is stable.

## 11. Runtime staging plan

Stage 1: parse `MT5Config`, load official small checkpoint config and weights, preserve shared/tied weight aliases, and instantiate a single encoder/decoder block graph.

Stage 2: implement encoder-only parity for `MT5EncoderModel`: embeddings, RMSNorm, bidirectional relative bias, MHA, gated FFN.

Stage 3: implement full seq2seq prefill without cache: encoder plus decoder causal self-attention and cross-attention.

Stage 4: implement `EncoderDecoderCache` ABI: decoder self K/V append/update, cross-attention K/V reuse with `is_updated` semantics, encoder output reuse.

Stage 5: implement generation logits with tied embedding transpose and last-token-only decode logits optimization.

Stage 6: add fused kernels/rewrites: RMSNorm, relative bias precompute, gated FFN fusion, fused attention with additive bias.

Stage 7: broaden checkpoint sizes and optional heads; keep non-gated FFN as a config-admitted fallback or explicit rejection until tested.

Initial stubs allowed:

- Dropout as no-op in inference.
- Training losses and gradient checkpointing.
- Sequence/token classification and QA heads.
- Beam search beyond the cache reorder contract if GenerationMixin integration is not first target.

## 12. Parity and validation plan

- Config parity: verify omitted `relative_attention_max_distance` resolves to 128 and serialized `tie_word_embeddings=false` resolves to true in current source.
- Tokenizer parity: tokenize/decode multilingual strings with official `spiece.model`; verify EOS insertion and special ids.
- Custom op tests: RMSNorm fp32 accumulation, `gelu_new`, relative bucket ids for bidirectional and causal modes, shift-right with `-100`.
- Single-layer parity: encoder block and decoder block with random weights, fixed masks, fp32 tolerance `1e-5`/`1e-4`.
- Attention parity: compare logits before/after bias addition and final attention output for encoder full sequence, decoder prefill, and decode with nonzero cache.
- Cache parity: first decode step fills self and cross caches; second decode step reuses cross cache and appends self cache. Validate K/V shapes `[B, heads, len, d_kv]`.
- End-to-end: `google/mt5-small` encoder-decoder logits for short multilingual prompts; then greedy decode token parity for a few steps.
- Reduced precision: fp16/bf16 tolerances should account for fp32 softmax/RMSNorm accumulation, e.g. logits atol/rtol around `5e-2` initially for fp16, tightened after kernel selection.

## 13. Performance probes

- Tokenization throughput for multilingual text and padding-side variants.
- Encoder throughput by `src_len` and batch for small/base/large.
- Decoder prefill throughput by `tgt_len` with fixed encoded source.
- Decode tokens/sec by batch, source length, and cache length.
- Logits GEMM cost by vocab size 250112; compare full-sequence logits vs last-token-only logits.
- Relative bias generation cost for prefill and one-token decode; compare dynamic generation vs cached bucket tables.
- Attention backend comparison: unfused matmul/softmax/matmul, fused dense attention with additive bias, cross-attention specialized path.
- Cache memory: per layer self K/V plus cross K/V across small/base/large/xl/xxl.
- FFN fusion probe: separate two input GEMMs plus multiply vs grouped/fused scheduling.

## 14. Skip/defer list

- Training losses, dropout behavior in training, and gradient checkpointing.
- Optional task heads: sequence classification, token classification, QA.
- Non-gated or non-`gelu_new` FFN configs until a checkpoint requires them.
- Beam-search cache reordering if first integration targets greedy/sampling decode; add before production generation.
- Output attentions/hidden states materialization unless used for parity debugging.
- Quantized/packed weights; official source here uses dense PyTorch modules.
- Multi-GPU/tensor parallel and sharded loading policies.

## 15. Final implementation checklist

- [ ] Parse `MT5Config`, including forced tied embeddings and defaulted `relative_attention_max_distance`.
- [ ] Route `model_type=mt5` to mT5 body even when historical `architectures` says `T5ForConditionalGeneration`.
- [ ] Load shared embeddings once and alias encoder, decoder, and LM head weight.
- [ ] Implement T5/mT5 RMSNorm with fp32 variance accumulation.
- [ ] Implement relative position bucket and learned bias lookup.
- [ ] Implement encoder bidirectional self-attention with shared block-0 bias.
- [ ] Implement decoder causal self-attention with cache-aware relative bias.
- [ ] Implement decoder cross-attention and cross K/V cache reuse.
- [ ] Implement gated `gelu_new` FFN and non-gated fallback/rejection policy.
- [ ] Implement shift-right and generation logits.
- [ ] Add tokenizer/SentencePiece integration boundary using T5 tokenizer semantics.
- [ ] Add single-block, full-prefill, and decode-cache parity tests.
- [ ] Benchmark encoder, decoder prefill, decode, logits, relative-bias generation, and cache memory.
