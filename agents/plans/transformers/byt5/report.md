# ByT5 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: representative official configs from [google/byt5-small](https://huggingface.co/google/byt5-small), [google/byt5-base](https://huggingface.co/google/byt5-base), [google/byt5-large](https://huggingface.co/google/byt5-large), [google/byt5-xl](https://huggingface.co/google/byt5-xl), and [google/byt5-xxl](https://huggingface.co/google/byt5-xxl).

Config source: downloaded `config.json`, `tokenizer_config.json`, and one representative `special_tokens_map.json` snapshot under `_sources/`. All inspected model configs are public and accessible; no gated gap was encountered.

Source files inspected:

- `_sources/tokenization_byt5.py`, copied from `src/transformers/models/byt5/tokenization_byt5.py`.
- `_sources/byt5__init__.py`, copied from `src/transformers/models/byt5/__init__.py`.
- `_sources/configuration_t5.py`, copied from `src/transformers/models/t5/configuration_t5.py`.
- `_sources/modeling_t5.py`, copied from `src/transformers/models/t5/modeling_t5.py`.
- `src/transformers/models/auto/tokenization_auto.py` for the `byt5 -> ByT5Tokenizer` tokenizer mapping.

Any missing files or assumptions: `src/transformers/models/byt5` has no `modeling_byt5.py` or `configuration_byt5.py` in this commit. Official ByT5 checkpoints declare `model_type: "t5"` and `architectures: ["T5ForConditionalGeneration"]`; the neural graph is therefore the in-library T5 graph. This report owns the ByT5 tokenizer/model-coupling audit plus the ByT5 config variants, and delegates shared block semantics to the inspected T5 source, not to the separate `t5` audit by assertion alone.

## 2. High-level architecture

ByT5 is a byte-level encoder-decoder seq2seq language model. The model body is T5: shared token embedding, encoder stack, decoder stack, relative-position self-attention, decoder cross-attention, T5 RMS-style layer norms, gated-GELU FFNs for official ByT5 checkpoints, and a tied LM head.

Dataflow:

```text
UTF-8 text -> byte tokenizer + special/sentinel ids -> shared embedding
  -> encoder stack
  -> decoder prefill/decode with self-attention KV cache + reusable cross-attention KV
  -> shared/tied LM head -> byte/sentinel vocabulary logits
```

Stage decomposition:

- CPU/data pipeline: UTF-8 byte tokenization, `</s>` insertion, padding/attention masks, optional sentinel-token construction for span corruption style prompts.
- Encoder: cacheable per source sequence; noncausal self-attention with relative position bias.
- Decoder prefill: causal self-attention plus encoder-decoder cross-attention over encoder states.
- Decode: one or more decoder tokens with self-attention cache growth and cross-attention K/V reuse.
- Logits/sampling: dense logits over 384 ids, then generation controller outside the model graph.

## 3. Important config dimensions

Effective defaults come from `T5Config` when omitted. The official configs omit `relative_attention_max_distance`, so the effective value is the T5 default `128`.

| Field | ByT5 source/config behavior |
|---|---|
| `model_type` | `t5` in all inspected configs |
| `tokenizer_class` | `ByT5Tokenizer` in configs/tokenizer configs |
| `vocab_size` | `384` in official configs |
| byte ids | specials occupy ids `0..2`; raw UTF-8 byte `b` maps to id `b + 3` |
| sentinel tokens | tokenizer default `extra_ids=125`; official tokenizer configs list `<extra_id_0>` through `<extra_id_124>` |
| `d_kv` / head dim | `64` |
| attention projection width | `num_heads * d_kv`, which is often less than `d_model` |
| attention type | MHA, no GQA/MQA |
| position encoding | learned relative attention bias, bucketed; no RoPE/ALiBi/absolute position embeddings |
| FFN | `gated-gelu`, normalized by `T5Config` to `dense_act_fn="gelu_new"` and `is_gated_act=True` |
| linear bias | T5 attention/FFN/LM projections are bias-free |
| cache | decoder self-attention `DynamicCache`; encoder-decoder cache wraps self-attention cache plus cross-attention cache |
| embedding aliases | `shared.weight`, encoder embedding, decoder embedding, and LM head weight are tied by source keys |
| decoder start | `decoder_start_token_id=0`, same as pad |

Representative checkpoint sweep:

| Checkpoint | `d_model` | `heads` | `inner_dim` | `d_ff` | encoder layers | decoder layers | vocab | FFN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `google/byt5-small` | 1472 | 6 | 384 | 3584 | 12 | 4 | 384 | gated-GELU |
| `google/byt5-base` | 1536 | 12 | 768 | 3968 | 18 | 6 | 384 | gated-GELU |
| `google/byt5-large` | 1536 | 16 | 1024 | 3840 | 36 | 12 | 384 | gated-GELU |
| `google/byt5-xl` | 2560 | 32 | 2048 | 6720 | 36 | 12 | 384 | gated-GELU |
| `google/byt5-xxl` | 4672 | 64 | 4096 | 12352 | 36 | 12 | 384 | gated-GELU |

## 3a. Family variation traps

- ByT5 is not a separate neural implementation in this commit. Admit it through T5 graph lowering with ByT5-specific tokenizer/config checks.
- `vocab_size=384` is small but not equal to `3 + 256 + 125 = 384` by accident. DinoML must preserve special ids, byte offset `3`, and sentinel ids from added special tokens. The tokenizer doc says `<extra_id_0>` is indexed from the end of the vocabulary; first integration should verify the exact loaded id map rather than infer it from `vocab_size`.
- The tokenizer `vocab_size` property returns only `256`; `get_vocab()` covers `vocab_size + offset` plus added tokens. Model embedding size must come from `config.vocab_size`, not tokenizer `vocab_size` alone.
- Official configs carry `tie_word_embeddings: false`, but current `T5Config` consumes this as `scale_decoder_outputs=False` and then forces `tie_word_embeddings=True`. Do not clone LM-head weights as independent parameters.
- `d_model != num_heads * d_kv` for all inspected ByT5 sizes. Q/K/V project from `d_model` to `inner_dim`; attention output projects `inner_dim -> d_model`.
- Encoder and decoder depths are asymmetric.
- Relative attention bias exists only on block 0 self-attention in each stack and is shared as a computed `position_bias` through later layers. Cross-attention has no learned relative bias.
- Decoder cross-attention K/V can be cached after the first generation step via `EncoderDecoderCache.is_updated`.
- No layout-translation opportunity exists for token sequence axes; rank-3 hidden states are `[batch, seq, hidden]`, attention tensors are `[batch, heads, query, key]`, and axis-sensitive softmax is `dim=-1`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token ids and embedding gather.
- Rank/shape views: `.view`, `.transpose(1, 2)`, `.transpose(3, 2)`, `.permute([2,0,1])`, `.unsqueeze(0)`, `.contiguous()`, `.reshape`.
- Attention mask expansion to additive masks compatible with `[batch, heads, query, key]`.
- Cache append/update and cache slice/reuse for decoder self-attention and cross-attention.

Neural network primitives:

- Shared `Embedding(384, d_model)`.
- Bias-free linear projections: Q/K/V `Linear(d_model -> inner_dim)`, O `Linear(inner_dim -> d_model)`.
- Bias-free gated FFN: `wi_0: d_model -> d_ff`, `wi_1: d_model -> d_ff`, `gelu_new(wi_0(x)) * wi_1(x)`, `wo: d_ff -> d_model`.
- T5 layer norm/RMSNorm variant: fp32 variance of squared hidden states over last dim, no mean subtraction, learned scale only.
- Residual adds and inference dropout elision.
- Optional fp16 clamp blocks appear in source; for inference, preserve only if lowered fp16 parity requires the same overflow guard.
- Tied LM head `Linear(d_model -> 384, bias=False)`.

Attention primitives:

- Dense MHA self-attention and cross-attention.
- Matmul QK^T, add relative/causal/padding bias, softmax in fp32 then cast back, matmul with V.
- Relative-position bucket calculation and bias embedding lookup.
- Encoder bidirectional self-attention, decoder causal self-attention, decoder bidirectional cross-attention over encoder states.

Position/relative-bias ops:

- `arange`, integer subtract, abs/min/where/log/floor-like cast to bucket ids.
- `Embedding(relative_attention_num_buckets=32, num_heads)` producing `[query, key, heads]`, permuted to `[1, heads, query, key]`.

Generation/cache ops:

- `_shift_right(labels)` for teacher-forced labels and decoder start token id `0`.
- Decoder self-attention cache tensor shape per layer: keys and values `[batch, heads, cached_decoder_length, d_kv]`.
- Cross-attention cache tensor shape per layer: keys and values `[batch, heads, encoder_length, d_kv]`, reusable after initial projection.
- Beam/cache reorder is inherited from common Transformers cache utilities rather than ByT5-specific source.

Preprocessing-coupled ops:

- UTF-8 byte encoding in tokenizer: Python string -> bytes -> ids `byte + 3`.
- Special tokens: pad id `0`, eos id `1`, unk id `2`, `</s>` appended to single or pair sequences.
- Token type ids are all zeros and do not enter the neural graph.

## 5. Layer/block breakdown

Encoder input:

```text
input_ids [B, S] -> shared_embedding [B, S, d_model] -> dropout(no-op at inference)
```

Encoder block, repeated `num_layers` times:

```text
x_norm = T5LayerNorm(x)
q = Linear(d_model -> H*d_kv, no bias)(x_norm).view(B, S, H, d_kv).transpose(1, 2)
k = Linear(d_model -> H*d_kv, no bias)(x_norm).view(...).transpose(1, 2)
v = Linear(d_model -> H*d_kv, no bias)(x_norm).view(...).transpose(1, 2)
position_bias = block0_relative_bias_or_reused_bias + encoder_padding_mask
attn = softmax((q @ k^T + position_bias).float(), dim=-1).to(scores_dtype)
x = x + Linear(H*d_kv -> d_model, no bias)(attn @ v)
y = T5LayerNorm(x)
ff = Linear(d_ff -> d_model, no bias)(gelu_new(wi_0(y)) * wi_1(y))
x = x + ff
```

Decoder block, repeated `num_decoder_layers` times:

```text
x = causal self-attention block as above, with decoder cache and unidirectional relative buckets
x_norm = T5LayerNorm(x)
q = decoder query projection from x_norm
k,v = cross-attention projections from encoder hidden states, or reused cross-attention cache
x = x + Linear(H*d_kv -> d_model, no bias)(softmax(q @ k^T + encoder_mask) @ v)
y = T5LayerNorm(x)
ff = gated-GELU FFN
x = x + ff
```

Output:

```text
decoder_hidden [B, T, d_model]
if scale_decoder_outputs: hidden *= d_model ** -0.5
lm_logits = hidden @ shared.weight.T
```

For official ByT5 configs, source-derived effective `scale_decoder_outputs` is `False` because config JSON supplies `tie_word_embeddings: false`.

## 6. Attention requirements

Required attention variants:

- Encoder self-attention: noncausal dense MHA, `query_length == key_length == source_length`, padding mask additive bias, bidirectional relative buckets.
- Decoder self-attention: causal dense MHA, query length can be prefill length or decode step length, key length includes cached tokens, unidirectional relative buckets with `past_seen_tokens` offset.
- Decoder cross-attention: dense MHA from decoder queries to encoder K/V, rectangular `[target_length, source_length]`, additive encoder padding mask, no learned relative bias.

Head contract:

- `num_key_value_heads == num_attention_heads`; no MQA/GQA.
- `head_dim = d_kv = 64`.
- Q/K/V width is `inner_dim = num_heads * d_kv`, not `d_model`.
- Value width equals key width.

Masking and math order:

```text
scores = q @ k.transpose(-1, -2)
position_bias = relative_bias_or_zeros
position_bias += mask when present
scores += position_bias
weights = softmax(scores.float(), dim=-1).type_as(scores)
output = weights @ v
```

FlashAttention/SDPA compatibility: possible only with a backend that accepts additive bias plus T5 relative-position bias, supports rectangular cross-attention, and preserves fp32 softmax accumulation/cast behavior. Relative bias generation remains a separate required op unless fused into an attention backend.

Cache ABI:

- Decoder self-attention cache appends projected K/V after projection and before attention matmul.
- Cached self-attention keys already include the model's relative-position convention indirectly through `past_seen_tokens`; there is no RoPE-applied K state.
- Cross-attention cache stores encoder-projected K/V and is marked updated per layer so later decode steps skip K/V projection.
- Encoder outputs themselves are independently cacheable for a fixed source sequence.

## 7. Tokenizer, preprocessing, and model coupling

ByT5 tokenizer has no vocab file. It encodes raw UTF-8 bytes:

```text
pad=0, eos=1, unk=2
byte b in [0,255] -> token id b + 3
```

`build_inputs_with_special_tokens` appends `</s>` to one sequence, or `A </s> B </s>` for pairs. `attention_mask` is the usual padding mask from `PreTrainedTokenizer`; token type ids are always zeros.

Sentinel tokens are added as special tokens, default `extra_ids=125`. Official tokenizer configs list `<extra_id_0>` through `<extra_id_124>`, and `special_tokens_map.json` repeats the same order. These ids must remain inside the 384-row embedding/LM-head vocabulary. Do not derive model `vocab_size` from the tokenizer's raw byte `vocab_size` property, because that property reports `256`.

The tokenizer decode path reconstructs bytes and decodes UTF-8 with `errors="ignore"`. End-to-end text parity should compare tokenizer decode behavior, not only id sequences.

## 8. Heads and task surface

Primary DinoML target: `T5ForConditionalGeneration` seq2seq generation.

Required for this target:

- `T5ForConditionalGeneration.forward`.
- Encoder stack, decoder stack, cache-aware generation, tied LM head.

Optional/deferred for first ByT5 integration:

- `T5EncoderModel` encoder-only use.
- `T5Model` bare encoder-decoder hidden states.
- Sequence classification, token classification, question answering heads. They are implemented in T5 source but are not required for ByT5 seq2seq generation parity.
- Training losses, labels, and gradient checkpointing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: bias-free linear stack to GEMM

Source pattern:

```text
nn.Linear(in, out, bias=False)(rank3_hidden)
```

Replacement: flatten `[B, S, in]` to `[B*S, in]`, GEMM with weight transposed as needed, reshape back to `[B, S, out]`.

Preconditions: contiguous hidden-state last dimension, no bias, dtype supported by GEMM provider, shape product fits runtime indexing.

Failure cases: non-contiguous or aliased hidden states that cannot be represented by DinoML's dense row-major contract.

Parity test sketch: compare each projection on random `[B,S,d_model]` tensors for all ByT5 widths.

### Rewrite: gated-GELU FFN fusion

Source pattern:

```text
gelu_new(wi_0(x)) * wi_1(x) -> wo
```

Replacement: two GEMMs plus fused activation/mul epilogue or a grouped dual-GEMM producer feeding a fused elementwise multiply before `wo`.

Preconditions: official ByT5 `feed_forward_proj="gated-gelu"`, no dropout in inference, no quantized custom modules.

Failure cases: alternate T5 configs with non-gated `relu` or other `feed_forward_proj` values should route through the generic T5 path.

### Rewrite: shared relative position bias cache

Source pattern: first block computes relative bias and later blocks reuse the same `position_bias`.

Replacement: generate relative bias once per stack per `(query_length, key_length, past_seen_tokens, bidirectional)` and pass it to all layers.

Preconditions: same head count and relative-bias table for that stack; block 0 owns the learned bias.

Failure cases: output attentions/debug paths that expect per-layer returned bias tensors still need ABI compatibility.

### Rewrite: last-token decode logits

Source pattern: decode with `query_length=1` but LM head over `[B, 1, d_model]`.

Replacement: keep rank-3 ABI but specialize GEMM for single-token decode or expose last-token-only logits buffer.

Preconditions: generation controller needs only the last token distribution.

Failure cases: callers requesting full decoder hidden states/logits for multiple decode positions.

## 10. Kernel fusion candidates

Highest priority:

- T5LayerNorm/RMSNorm-style kernel with fp32 accumulation and scale-only output.
- Dense additive-bias attention with relative position bias, fp32 softmax, and KV cache support.
- Gated-GELU FFN fusion: `wi_0`, `wi_1`, `gelu_new`, multiply, `wo`.
- Shared embedding/LM-head GEMM with tiny vocabulary `384`; optimize logits path for small N.

Medium priority:

- Relative-position bucket + embedding + bias layout generation for encoder and decoder.
- Cross-attention K/V projection materialization and cache reuse.
- Q/K/V projection grouping per attention module when three independent bias-free linears share input.

Lower priority:

- Classification/QA heads.
- Training-only dropout/loss/gradient-checkpointing behavior.
- Tokenizer acceleration; CPU preprocessing is simple but can dominate very small batches because byte sequences are longer than subword sequences.

## 11. Runtime staging plan

Stage 1: parse official ByT5 configs and tokenizer configs. Enforce `model_type="t5"`, `tokenizer_class="ByT5Tokenizer"`, `vocab_size=384`, and supported `feed_forward_proj`.

Stage 2: load shared/tied weights and run embedding, T5LayerNorm, FFN, and one attention block parity with random tensors.

Stage 3: encoder-only parity with relative bias generation and padding masks.

Stage 4: decoder prefill parity with causal self-attention and cross-attention over cached encoder outputs.

Stage 5: decode parity with `EncoderDecoderCache`: self-attention cache append and cross-attention K/V reuse.

Stage 6: optimize attention/FFN/LM-head kernels and relative-bias caching.

Stage 7: end-to-end generation parity through the byte tokenizer and generation controller.

Initially stub: training losses, classification/QA heads, output attentions, hidden-state dumps beyond debug parity, and beam-search controller details.

## 12. Parity and validation plan

- Tokenizer parity: Unicode strings with ASCII, multibyte UTF-8, invalid/special-token edge cases, pair inputs, existing trailing `</s>`, and sentinel tokens.
- Config parity: verify `T5Config` effective `tie_word_embeddings=True` and `scale_decoder_outputs=False` for official configs.
- Custom op tests: T5LayerNorm fp32 accumulation; relative bucket ids for bidirectional and causal modes; `gelu_new`.
- Single-layer parity: encoder block and decoder block against Transformers for fp32, then fp16/bf16 where supported.
- Encoder parity: full encoder outputs for small/base with fixed masks.
- Prefill logits parity: `T5ForConditionalGeneration` with no cache.
- Decode parity: one-token and multi-step decode with cache, including cross-attention cache reuse.
- End-to-end text parity: generate short outputs from `google/byt5-small` and compare token ids first, decoded text second.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 attention/logits need looser tolerances after softmax, starting around `rtol=5e-2, atol=5e-2` until kernels are numerically characterized.

## 13. Performance probes

- Tokenization throughput versus sequence byte length.
- Encoder-only throughput sweep over source lengths; byte-level tokenization makes sequences much longer than subword T5.
- Decoder prefill throughput over target length and source length.
- Decode tokens/sec with and without cross-attention cache reuse.
- Relative-bias generation cost by `(query_length, key_length)` and cache-hit behavior.
- Attention backend comparison: eager dense attention versus fused additive-bias attention.
- FFN GEMM throughput for ByT5 shapes, especially large `d_model` with `inner_dim < d_model`.
- LM-head small-vocabulary GEMM throughput and last-token-only logits path.
- KV cache memory usage by size variant and source/target lengths.

## 14. Skip/defer list

- Training, loss computation, gradient checkpointing, and dropout randomness.
- Sequence classification, token classification, and question answering heads.
- Beam search and generation-controller policy beyond cache/logits ABI.
- Remote-code or non-official configs that do not use in-library T5.
- Quantized or packed weight loading; inspected source has no ByT5-specific packed format.
- Multi-GPU tensor parallelism.
- Layout translation; sequence-major semantics are axis-sensitive and should remain faithful.

## 15. Final implementation checklist

- [ ] Parse ByT5 tokenizer/config metadata and enforce official in-library T5 route.
- [ ] Preserve `vocab_size=384`, byte id offset `3`, pad/eos/unk ids, and 125 sentinel tokens.
- [ ] Load tied `shared.weight` aliases for encoder embedding, decoder embedding, and LM head.
- [ ] Implement/validate T5LayerNorm with fp32 accumulation and scale only.
- [ ] Implement/validate relative-position bucket and bias generation.
- [ ] Lower bias-free Q/K/V/O projections where `inner_dim = num_heads * d_kv`.
- [ ] Lower gated-GELU FFN for official ByT5 configs.
- [ ] Implement encoder noncausal self-attention with padding mask.
- [ ] Implement decoder causal self-attention with KV cache append.
- [ ] Implement decoder cross-attention with reusable encoder K/V cache.
- [ ] Add tokenizer parity tests for UTF-8 bytes, EOS insertion, and sentinels.
- [ ] Add single-block, encoder, prefill, and decode parity tests.
- [ ] Benchmark byte-tokenized sequence-length sweeps and cache memory.
