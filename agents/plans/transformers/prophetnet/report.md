# ProphetNet Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: prophetnet
Config source: local configuration source plus HF config.json snapshots
Source files inspected:
- transformers/src/transformers/models/prophetnet/modeling_prophetnet.py
- transformers/src/transformers/models/prophetnet/configuration_prophetnet.py
- transformers/src/transformers/models/prophetnet/tokenization_prophetnet.py
- transformers/tests/models/prophetnet/test_modeling_prophetnet.py
Any missing files or assumptions:
- microsoft/prophetnet-large-uncased-wiki100-cased config returned 401 and is a gap.
- No remote-code files are required for the in-library implementation.
- The modeling file is the authoritative source; there is no modular source file in this family.
```

Pinned source URLs:

- `modeling_prophetnet.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/prophetnet/modeling_prophetnet.py
- `configuration_prophetnet.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/prophetnet/configuration_prophetnet.py
- `tokenization_prophetnet.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/prophetnet/tokenization_prophetnet.py

Local config snapshots written beside this report:

- `microsoft__prophetnet-large-uncased.config.json`
- `microsoft__prophetnet-large-uncased-cnndm.config.json`
- `microsoft__prophetnet-large-uncased-squad-qg.config.json`
- `hf-internal-testing__tiny-random-prophetnet.config.json`
- `patrickvonplaten__prophetnet-large-uncased-standalone.config.json`
- `microsoft__prophetnet-large-uncased-wiki100-cased.config.fetch.txt`

Primary runtime target for this report: `ProphetNetForConditionalGeneration` seq2seq generation. Standalone `ProphetNetForCausalLM`, `ProphetNetModel`, `ProphetNetEncoder`, and `ProphetNetDecoder` are implemented heads and are useful parity targets, but conditional generation owns the first end-to-end integration.

## 2. High-level architecture

ProphetNet is a text-only encoder-decoder with learned token embeddings, learned absolute position embeddings, post-norm Transformer encoder layers, and a custom decoder self-attention that runs one main stream plus `ngram` future-prediction streams. The seq2seq LM head projects the decoder predict streams through a tied vocabulary projection and uses stream 0 as ordinary next-token logits.

```text
WordPiece tokenizer / masks
  -> encoder embeddings + encoder MHA/FFN stack
  -> decoder main stream + ngram predict streams
  -> ngram self-attention + optional encoder cross-attention + FFN stack
  -> shared LM head
  -> generation controller / logits processors / beam search
```

Stage decomposition:

- CPU/data pipeline: ProphetNet WordPiece tokenizer, optional lowercasing/basic tokenization, padding, `attention_mask`, decoder-start handling.
- Cacheable encoder: `input_ids -> encoder_last_hidden_state [B,S,H]`; independent of decode tokens and reusable across beam hypotheses after reorder.
- Decoder prefill: full target prefix builds main stream `[B,T,H]` plus `ngram` predict streams `[B,ngram*T,H]`; creates relative bucket tables and ngram attention masks.
- Decoder decode: source asserts cache use is only supported when current decoder input length is 1; only main-stream self-attention K/V grow in the autoregressive cache.
- Logits: `last_hidden_state_ngram.view(B, ngram, T, H) -> Linear(H,V)`, then `logits = stream 0`; later ngram logits are optional outputs/training support.

## 3. Important config dimensions

Source defaults from `ProphetNetConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `vocab_size` | 30522 | Shared token embedding and LM head width. |
| `hidden_size` | 1024 | Model width. |
| `num_encoder_layers` | 12 | Encoder block count. |
| `num_decoder_layers` | 12 | Decoder block count and cache layer count. |
| `num_encoder_attention_heads` | 16 | Encoder MHA heads. |
| `num_decoder_attention_heads` | 16 | Decoder self/cross heads. |
| `head_dim` | `hidden_size / heads` = 64 | Asserted divisible in source. |
| `encoder_ffn_dim` | 4096 | Encoder FFN intermediate width. |
| `decoder_ffn_dim` | 4096 | Decoder FFN intermediate width. |
| `max_position_embeddings` | 512 | Learned position table and decoder relative bucket buffer bound. |
| `ngram` | 2 | Number of predict streams; decoder sequence axis is `(1+ngram)*T` internally. |
| `num_buckets` | 32 | Relative-position bucket count per head. |
| `relative_max_distance` | 128 | Log bucket saturation threshold. |
| `activation_function` | `gelu` | FFN activation. |
| `add_cross_attention` | `True` | Enables decoder encoder-attention block. |
| `use_cache` | `True` | Uses `DynamicCache` or `EncoderDecoderCache`. |
| `decoder_start_token_id` | 0 | Checkpoint configs usually override to 102. |
| `pad/bos/eos` | 0/1/2 | Checkpoint configs usually use pad 0 and bos/eos 102 for large models. |

Representative checkpoint sweep:

| Checkpoint | Architecture | H | Enc/Dec layers | Heads | FFN | Pos | ngram | Vocab | Notable config facts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `microsoft/prophetnet-large-uncased` | ConditionalGeneration | 1024 | 12/12 | 16/16 | 4096/4096 | 512 | 2 | 30522 | `decoder_start_token_id=102`, summarization generation defaults in config, historical `output_past=false`. |
| `microsoft/prophetnet-large-uncased-cnndm` | ConditionalGeneration | 1024 | 12/12 | 16/16 | 4096/4096 | 512 | 2 | 30522 | Summarization task params use beams 5 and length penalty 1.2. |
| `microsoft/prophetnet-large-uncased-squad-qg` | ConditionalGeneration | 1024 | 12/12 | 16/16 | 4096/4096 | 512 | 2 | 30522 | Question-generation checkpoint; same operator structure. |
| `hf-internal-testing/tiny-random-prophetnet` | ProphetNetModel | 16 | 4/4 | 4/4 | 32/32 | 30 | 2 | 30522 | Debug-size source-compatible checkpoint, `torch_dtype=float32`. |
| `patrickvonplaten/prophetnet-large-uncased-standalone` | ProphetNetDecoder | 1024 | 12/12 | 16/16 | 4096/4096 | 512 | 2 | 30522 | `is_decoder=true`, `is_encoder_decoder=false`, `add_cross_attention=false`. |

The inaccessible checkpoint was `microsoft/prophetnet-large-uncased-wiki100-cased`: https://huggingface.co/microsoft/prophetnet-large-uncased-wiki100-cased. Access to its `config.json` would confirm whether cased/tokenizer or vocab details change the ABI.

## 3a. Family variation traps

- `ngram` changes the decoder internal sequence length, ngram embedding table, predict attention rank, LM output split, and training loss expansion. Do not hard-code `ngram=2` in kernels, even though reachable production configs use 2.
- `max_position_embeddings` is the actual source field used by both learned embeddings and relative bucket buffering. Historical `encoder_max_position_embeddings` and `decoder_max_position_embeddings` appear in configs but are not read by the inspected source.
- `decoder_layerdrop`, `encoder_layerdrop`, `output_past`, `prefix`, and `gradient_checkpointing` appear in some checkpoint configs but are not runtime graph features in this source basis.
- Large checkpoints use token id 102 as `bos_token_id`, `eos_token_id`, and `decoder_start_token_id`; source defaults differ.
- Standalone decoder config disables cross-attention and changes the cache from `EncoderDecoderCache` to `DynamicCache`.
- `hidden_size` must equal `num_heads * head_dim`; source asserts divisibility and does not expose separate `head_dim`.
- Tied weights matter: seq2seq shares encoder/decoder token embeddings through `prophetnet.word_embeddings`, and LM head weight ties to that same embedding when `tie_word_embeddings=True`.
- Generation tests skip assisted/prompt-lookup decoding for ProphetNet with the note “special cache sizes”; DinoML should not assume ordinary decoder-only generation helper parity.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token, learned absolute position, and ngram stream embeddings.
- `cumsum(attention_mask, dim=1)`, clamp, scalar add for position ids.
- Rank reshapes/views/transposes/permutes/chunk/stack/concat/repeat/expand.
- `gather` for relative-position bucket lookup from generated per-token relative embedding tables.
- Mask construction with `full`, `triu`, diagonal fill equivalent, `cat`, and broadcast add.

Neural network primitives:

- Bias Linear: Q/K/V/out projections, FFN `H -> ffn_dim -> H`, relative-position projection `H -> num_buckets*num_heads`.
- Bias-free Linear: LM head `H -> vocab_size`, tied to token embedding in generation models.
- LayerNorm over hidden axis, epsilon from PyTorch default unless inherited module configuration changes.
- GELU activation by default through `ACT2FN`.
- Dropout is present in source but can be disabled for inference.

Attention primitives:

- Encoder self-attention: dense noncausal MHA, query scaled before matmul, additive padding mask `[B,H,1,S]`.
- Decoder cross-attention: dense MHA from decoder hidden states to encoder hidden states, cacheable encoder K/V.
- Decoder main-stream self-attention: causal dense MHA over cached main stream with learned relative-position score add.
- Decoder predict-stream self-attention: custom ngram attention over `[main K/V, predict K/V]` with key length `2*T` in prefill and `past+2` in cached decode.

Position/relative-bias ops:

- Learned absolute position embedding.
- T5-style unidirectional relative bucket mapping with exact small distances and logarithmic large-distance buckets.
- Dynamic per-layer relative score generation from `relative_pos_embeddings(hidden_states)`, not a static parameter table.

Generation/cache ops:

- `DynamicCache` for standalone decoder self-attention.
- `EncoderDecoderCache` with separate self-attention and cross-attention caches for seq2seq.
- Per-layer self cache stores only main-stream K/V `[B,H,T,D]`.
- Cross-attention cache stores encoder K/V `[B,H,S,D]` and `is_updated[layer]`.
- Beam/cache reorder must account for both self and cross caches through the Transformers cache abstraction.

Preprocessing-coupled ops:

- WordPiece tokenizer with no `token_type_ids`.
- Sequence special-token build appends `[SEP]`; large checkpoints use `[SEP]` id 102 as EOS/decoder start.

## 5. Layer/block breakdown

Encoder embeddings:

```text
input_ids [B,S] -> word_embeddings [B,S,H]
position_ids = cumsum(attention_mask)*attention_mask + pad_id, clamped
position_embeddings [B,S,H]
x = LayerNorm(word + pos)
x = dropout(x)
```

Encoder block, repeated `num_encoder_layers`:

```text
q,k,v = Linear(H,H)(x), split to [B,Hh,S,D]
q = q / sqrt(D)
scores = q @ k^T + optional padding_mask [B,Hh,1,S]
attn = softmax(scores)
x = LayerNorm(x + dropout(out_proj(attn @ v)))
ff = Linear(ffn,H)(dropout(gelu(Linear(H,ffn)(x))))
x = LayerNorm(x + dropout(ff))
```

Decoder embeddings for target length `T`:

```text
main = token_embed(decoder_ids) + position_embed(pos)
predict_i = ngram_embed[i] + position_embed(pos + 1), for i in 0..ngram-1
hidden = concat([main, predict_0, ..., predict_n], dim=1)  # [B,(1+ngram)*T,H]
hidden = LayerNorm(hidden)
```

Decoder block, repeated `num_decoder_layers`:

```text
q,k,v = Linear(H,H)(hidden), split streams by sequence axis
main K/V cache update appends only main stream
main_attn = causal_attention(main_q, main_kv, rel_bias_from_main_hidden)
predict_attn = ngram_attention(predict_q, cat(main_kv, predict_kv), rel_bias_from_predict_hidden)
self_out = concat(main_attn, predict_attn streams)
hidden = LayerNorm(hidden + dropout(out_proj(self_out)))
if cross_attention:
    hidden = LayerNorm(hidden + dropout(cross_attention(hidden, encoder_hidden_states)))
hidden = LayerNorm(hidden + FeedForward(hidden))
```

LM head:

```text
ngram_hidden = last_hidden_state_ngram.view(B, ngram, T, H)
predict_logits = lm_head(ngram_hidden)  # [B,ngram,T,V]
logits = predict_logits[:, 0]           # [B,T,V]
logits_ngram = predict_logits[:, 1:]    # optional [B,ngram-1,T,V]
```

## 6. Attention requirements

Encoder attention is ordinary dense bidirectional MHA with full Q/K/V projection width `hidden_size`, equal Q/K/V head dim, and additive padding mask. It is compatible with a standard dense attention backend if the source order is preserved: query scaling before score matmul, mask add before softmax, dropout after softmax.

Decoder cross-attention is ordinary dense MHA from decoder hidden length `(1+ngram)*T` to encoder length `S`. During cached seq2seq decode, cross K/V are computed once per layer and reused via `EncoderDecoderCache.cross_attention_cache`.

Decoder ngram self-attention is the special family surface:

- Main stream:
  - Query/key/value from the first stream only.
  - Prefill scores shape `[B, Hh, T, T]`.
  - Cached decode scores shape `[B, Hh, 1, past+1]`.
  - Causal mask is applied in prefill; cached decode uses the cache sequence to avoid future tokens and skips an explicit mask.
- Predict streams:
  - Predict query shape `[B, ngram, Hh, T, D]`.
  - Prefill key/value shape uses `cat(main_stream_KV, predict_stream_i_KV)` so source length is `2*T`.
  - Cached decode with one token uses source length `past+2`: cached main K/V plus current predict K/V.
  - Mask shape before source permute is `[B,Hh,ngram,T,2*T]`, then permuted to `[B,ngram,Hh,T,2*T]`.

There is no GQA/MQA, sliding-window, block-sparse, RoPE, ALiBi, or packed-varlen backend dispatch in this source. FlashAttention/SDPA are not used directly; eager `einsum` attention plus custom relative-bias gather is the source behavior.

Cache ABI:

```text
self K/V per decoder layer: [B, num_decoder_attention_heads, cached_T, head_dim]
cross K/V per decoder layer: [B, num_decoder_attention_heads, encoder_S, head_dim]
standalone decoder cache: DynamicCache
seq2seq cache: EncoderDecoderCache(self_attention_cache, cross_attention_cache)
```

Cached decode admission should require current decoder token length `T_current == 1`, matching the source assertion.

## 7. Position encoding and custom math

ProphetNet uses learned absolute positions plus dynamic relative score terms. Position ids are derived from the attention mask unless decoding from cache:

```python
if cache_length != 0:
    position_ids = full([1, 1], padding_idx + input_len + cache_length)
else:
    position_ids = cumsum(attention_mask, dim=1) * attention_mask + padding_idx
    position_ids = clamp(position_ids, 0, max_position_embeddings - 1)
```

Relative bucket math:

```python
def relative_bucket(relative_pos, num_buckets, max_distance):
    inv = max(-relative_pos, 0)
    max_exact = num_buckets // 2
    large = max_exact + log(inv / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)
    large = min(large, num_buckets - 1).int()
    return where(inv < max_exact, inv.int(), large)
```

Ngram predict attention mask:

```python
left = full([ngram, L, L], -inf)
right = clone(left)
for stream_idx in range(ngram):
    right[stream_idx].fill_diagonal_(0)
    left[stream_idx].triu_(-stream_idx + 1)
left[:, :, 0] = 0
predict_bias = cat([left, right], dim=2)  # [ngram,L,2L]
```

Relative score add is not a simple static bias table. Each layer projects hidden states:

```text
rel = Linear(H, num_buckets*num_heads)(stream_hidden)
rel = reshape/permute to align [B,Hh,T,num_buckets]
score_bias = gather(rel, bucket_ids)
scores += score_bias
```

Prefill can precompute bucket ids for a fixed maximum target length, but the projected relative scores depend on per-layer hidden states and must remain in the runtime graph.

## 8. Preprocessing and input packing

The tokenizer is a slow Python WordPiece tokenizer with basic tokenization, optional lowercasing, punctuation splitting, CJK char splitting, and accent stripping. It emits `input_ids` and `attention_mask`; `token_type_ids` are not part of `model_input_names`.

Special-token construction appends `[SEP]` to single sequences and `[SEP]` between/after pair sequences. Large ProphetNet checkpoints use id 102 for BOS/EOS/decoder start, so generation controller parity requires checkpoint config values rather than source defaults.

GPU/runtime graph inputs for first integration:

```text
input_ids: [B,S] int64/int32 acceptable after frontend policy
attention_mask: [B,S] 0/1
decoder_input_ids: [B,T] during prefill or [B,1] during cached decode
decoder_attention_mask: [B,T] for prefill
encoder_outputs: optional precomputed [B,S,H]
past_key_values: optional cache object/state
```

No image/audio/video processors, placeholder token scatter, packed sequence metadata, or modality stitching are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Tied LM Head To GEMM

Source pattern:

```text
predict_streams [B,ngram,T,H] -> lm_head(weight tied to token embedding) -> [B,ngram,T,V]
```

Replacement pattern:

```text
Flatten(B*ngram*T,H) -> GEMM_RCR(weight embedding[V,H]) -> Reshape(B,ngram,T,V)
```

Preconditions: LM head bias is false, weight alias to `prophetnet.word_embeddings.weight` is preserved, hidden tensor is contiguous or lowered with explicit stride handling.

Failure cases: untied user-modified weights, quantized packed weights with different storage layout.

Parity test sketch: compare logits and `logits_ngram` for random hidden states and checkpoint-tied weights.

### Rewrite: Relative Bucket Tables As Shape-Specialized Constants

Source pattern: bucket ids are recomputed from `position_ids` and max target positions during prefill.

Replacement pattern: for static/bucketed target length and left-padded policy known false, precompute integer bucket-id tensors for main `[T,T]` and predict `[T,2T]`, then retain runtime gather from projected relative embeddings.

Preconditions: absolute position ids follow contiguous `1..T` pattern from unpadded or right-padded target mask; `max_position_embeddings`, `num_buckets`, and `relative_max_distance` fixed.

Failure cases: custom `position_ids`, non-contiguous attention masks, cached decode, or padding patterns that alter `cumsum`.

Parity test sketch: compare bucket tensors from source helper over several `T`, `ngram`, and masked/unmasked cases.

### Rewrite: Ngram Self-Attention Split Into Main And Predict Kernels

Source pattern: one projection over concatenated streams, then chunk/stack/cat/einsum/gather.

Replacement pattern:

```text
QKV GEMM over [B,(1+n)T,H]
main fused attention over [B,Hh,T,T]
predict fused ngram attention over [B,n,Hh,T,2T]
concat streams -> out GEMM
```

Preconditions: dense contiguous stream layout `[main, predict_0, ...]`, no output-attention materialization, dropout disabled, `ngram` small static integer.

Failure cases: `output_attentions=True` requiring full probability tensors, training dropout, dynamic `ngram`, or custom cache length not supported by source.

Parity test sketch: one decoder layer with random states, masks, buckets, and cache/no-cache paths.

### Rewrite: Decode-Time Last-Token Logits

Source pattern computes LM head over `[B,ngram,1,H]` in cached decode.

Replacement pattern: generate only stream-0 logits `[B,V]` for token selection; skip streams `1..ngram-1` when caller does not request `logits_ngram`.

Preconditions: inference generation path only, no `output_scores` requirement for ngram logits, no training loss.

Failure cases: APIs that return `logits_ngram` or need ngram auxiliary outputs.

Parity test sketch: compare stream-0 logits for cached decode and assert skipped outputs are not requested.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual post-norm around attention and FFN, because every block uses this pattern.
- Dense GEMM for QKV/out/FFN/LM head with explicit tied-weight handling.
- Custom ProphetNet ngram attention prefill kernel or lowered composition, because predict attention has rank-5 score/prob tensors and `2*T` source length.
- Cached decode ngram attention kernel, because source shape is unusual (`past+2`) and generic attention backends will not directly match.

Medium priority:

- Relative-position projection + gather + score add. It is per-layer and hidden-state-dependent, so it can become a noticeable overhead around attention.
- Causal and predict-mask precompute for static or bucketed `T`.
- Encoder and cross-attention standard fused attention once additive mask and query scaling order are verified.
- GELU FFN fusion: `Linear -> GELU -> Linear` around existing GEMM provider work.

Lower priority:

- Tokenizer acceleration; keep it CPU/data-pipeline first.
- Materializing `output_attentions`; useful for debug but not first production path.
- Training loss and label smoothing.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `ProphetNetForConditionalGeneration`; preserve shared embeddings and LM head alias metadata.

Stage 2: implement encoder-only parity with embeddings, position ids, attention mask, MHA, FFN, and post-norm blocks.

Stage 3: implement decoder prefill without cache for `ngram=2`, including relative bucket helpers, predict attention masks, and `last_hidden_state_ngram`.

Stage 4: add seq2seq logits parity and tied LM head, initially returning only stream-0 logits for generation unless `logits_ngram` is requested.

Stage 5: add cache ABI for decoder self main-stream K/V and cross-attention K/V; enforce `decoder_input_ids` length 1 for cached decode.

Stage 6: enable generation-loop parity with greedy/beam search using the existing generation controller or a DinoML controller shim.

Stage 7: optimize ngram attention, relative-bias gather, and last-token logits; broaden to standalone `ProphetNetForCausalLM` and `ProphetNetDecoder`.

Initially stub or reject training loss, dropout, `output_attentions=True`, assisted/speculative generation, and inaccessible/gated checkpoint variants.

## 12. Parity and validation plan

- Unit-test `compute_relative_buckets`, `compute_all_stream_relative_buckets`, and `ngram_attention_bias` against Transformers for several target lengths, `ngram` values, and dtypes.
- Encoder single-layer parity with random small config: compare hidden states in fp32 at `atol=1e-5, rtol=1e-5`.
- Decoder one-layer prefill parity for `ngram=2` and `ngram=4`; include decoder attention mask with zeros.
- Cached decode parity: full-prefix run over `T+1` versus prefill cache plus one-token decode, using source tolerance around `1e-3` to `1e-2` for tested slices.
- Seq2seq checkpoint smoke: `microsoft/prophetnet-large-uncased` encoder hidden slice and logits slice from Transformers integration tests.
- CNNDM generation smoke: same prompt and tokenized-output comparison as Transformers slow test after generation controller integration.
- fp16 smoke: no NaNs for model forward, then numeric tolerances around `atol=1e-2, rtol=1e-2` for reduced precision.

## 13. Performance probes

- Encoder throughput over `B` and source length `S`.
- Decoder prefill throughput over target length `T`, especially rank-5 predict attention memory.
- Cached decode tokens/sec with and without cross-attention cache reuse.
- KV cache memory per layer: self `[B,Hh,T,D]*2`, cross `[B,Hh,S,D]*2`.
- Relative-bias gather time as a share of decoder layer time.
- LM head last-token versus full-stream logits cost.
- Beam-size sweep, because seq2seq cache reorder and encoder-output expansion can dominate small batches.
- `ngram` sweep on synthetic configs (`1,2,4`) to quantify custom attention scaling.

## 14. Skip/defer list

- Training loss, label smoothing, and `disable_ngram_loss`.
- Dropout behavior except as disabled inference ops.
- `output_attentions=True` materialization for main/predict/cross attention.
- Assisted decoding, prompt-lookup decoding, and speculative decoding; upstream tests skip ProphetNet for model-specific cache reasons.
- Gated/inaccessible wiki100 cased checkpoint until config and tokenizer files are accessible.
- General custom `position_ids`; first integration can require source-derived contiguous positions.
- Multi-GPU/tensor parallelism and quantized packed weight formats.

## 15. Final implementation checklist

- [ ] Parse `ProphetNetConfig`, including `ngram`, relative bucket fields, separate encoder/decoder layer/head/FFN counts, and checkpoint token ids.
- [ ] Load/tie token embeddings, encoder/decoder embedding aliases, and LM head weight alias.
- [ ] Implement ProphetNet learned position ids from masks and cached decode length.
- [ ] Implement standard encoder MHA/cross-attention with additive masks and query pre-scaling.
- [ ] Implement `ngram_attention_bias` and relative bucket helpers.
- [ ] Implement decoder ngram self-attention prefill for `[B,(1+ngram)T,H]`.
- [ ] Implement hidden-state-dependent relative-position projection/gather score adds.
- [ ] Implement seq2seq LM head split into `logits` and optional `logits_ngram`.
- [ ] Add decoder cache ABI: main-stream self K/V plus cross-attention K/V.
- [ ] Enforce cached decode length-1 admission.
- [ ] Add one-layer encoder, one-layer decoder, full prefill, and cached decode parity tests.
- [ ] Add checkpoint smoke tests for large uncased and CNNDM configs.
- [ ] Benchmark encoder, decoder prefill, cached decode, relative-bias gather, and LM head.
