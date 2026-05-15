# Marian Transformers Audit

## 1. Source basis

```text
Transformers commit/version: local checkout b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Marian family; representative checkpoints listed below
Config source: local MarianConfig plus fetched Hugging Face config/tokenizer/generation JSON snapshots
Source files inspected: modeling_marian.py, configuration_marian.py, tokenization_marian.py, masking_utils.py, cache_utils.py
Any missing files or assumptions: no target_vocab.json was accessible for sampled checkpoints; separate-vocab behavior is source-supported but unsampled
```

Primary source files:

- `transformers/src/transformers/models/marian/modeling_marian.py`
- `transformers/src/transformers/models/marian/configuration_marian.py`
- `transformers/src/transformers/models/marian/tokenization_marian.py`
- `transformers/src/transformers/masking_utils.py`
- `transformers/src/transformers/cache_utils.py`

Representative config snapshots are stored under `agents/plans/transformers/marian/_sources/` for:

- [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)
- [Helsinki-NLP/opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en)
- [Helsinki-NLP/opus-mt-ROMANCE-en](https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en)
- [Helsinki-NLP/opus-mt-en-ROMANCE](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE)
- [sshleifer/tiny-marian-en-de](https://huggingface.co/sshleifer/tiny-marian-en-de)

The inspected native source implements `MarianModel`, `MarianMTModel`, and `MarianForCausalLM`. This report targets `MarianMTModel` seq2seq translation inference. `MarianForCausalLM` is optional/deferred unless DinoML wants decoder-only compatibility.

## 2. High-level architecture

Marian is a text-only encoder-decoder translation model, BART-like but with frozen sinusoidal positional embeddings, no embedding layernorm, and generation beginning from the pad token rather than BOS.

```text
SentencePiece + optional target-language prefix -> source input_ids/attention_mask
  -> encoder token embedding + sinusoidal position embedding
  -> N bidirectional Transformer encoder blocks
  -> decoder pad-start token + generated target prefix
  -> N causal decoder blocks with self-attention cache and encoder cross-attention cache
  -> tied/untied LM head + final_logits_bias
  -> generation controller: bad token suppression, forced EOS, beam/sample
```

Stage decomposition:

- CPU/data pipeline: Moses punctuation normalization when available, SentencePiece source/target tokenization, optional `>>lang<<` prefix tokens, EOS append, attention-mask construction.
- Encoder runtime: input embedding lookup, static sinusoidal position lookup, bidirectional self-attention, FFN, LayerNorm.
- Decoder prefill: shifted target prefix or pad-start token, causal self-attention, cross-attention over cached encoder hidden states.
- Decode step: one or more new decoder tokens, self-attention KV cache update, cross-attention K/V reuse, logits for next-token selection.
- Generation controller: beam search, pad-token suppression via `bad_words_ids`, forced EOS. This is outside the core neural graph but required for end-to-end translation parity.

## 3. Important config dimensions

Source defaults from `MarianConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `d_model` | 1024 | Hidden width and attention projection width |
| `encoder_layers` / `decoder_layers` | 12 / 12 | Block counts |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | MHA head counts |
| `head_dim` | `d_model / heads` | Source enforces divisibility in attention init |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 4096 / 4096 | FFN inner widths |
| `vocab_size` | 58101 | Encoder/shared vocab |
| `decoder_vocab_size` | `vocab_size` if omitted | Decoder vocab when untied |
| `max_position_embeddings` | 1024 | Frozen sinusoidal table length |
| `activation_function` | `gelu` | Checkpoints commonly override to `swish` |
| `scale_embedding` | `False` | Checkpoints commonly set `True`, multiplying embeddings by `sqrt(d_model)` |
| `use_cache` | `True` | Enables encoder-decoder cache during generation |
| `share_encoder_decoder_embeddings` | `True` | Ties encoder/decoder embeddings and often LM head |
| `decoder_start_token_id` | 58100 | Usually equal to pad token |
| `pad_token_id` | 58100 | Decoder start and source padding |
| `eos_token_id` / `forced_eos_token_id` | 0 / 0 | EOS append and generation termination |

Representative checkpoint sweep:

| Checkpoint | d_model | Layers enc/dec | Heads enc/dec | FFN enc/dec | Activation | Vocab | Max pos | Decoder start | Notes |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `Helsinki-NLP/opus-mt-en-de` | 512 | 6/6 | 8/8 | 2048/2048 | swish | 58101 | 512 | 58100 | common production shape |
| `Helsinki-NLP/opus-mt-fr-en` | 512 | 6/6 | 8/8 | 2048/2048 | swish | 59514 | 512 | 59513 | explicit `decoder_vocab_size` |
| `Helsinki-NLP/opus-mt-ROMANCE-en` | 512 | 6/6 | 8/8 | 2048/2048 | swish | 65001 | 512 | 65000 | multilingual source side; no target prefix required |
| `Helsinki-NLP/opus-mt-en-ROMANCE` | 512 | 6/6 | 8/8 | 2048/2048 | swish | 65001 | 512 | 65000 | multilingual target side; 47 `>>lang<<` vocab tokens |
| `sshleifer/tiny-marian-en-de` | 2 | 2/2 | 1/1 | 2/2 | swish | 58101 | 512 | 58100 | debug-sized topology |

Fields omitted in some checkpoint configs use source defaults. In particular, omitted `share_encoder_decoder_embeddings` and `decoder_vocab_size` should be interpreted through `MarianConfig.__post_init__`, not as absent behavior.

## 3a. Family variation traps

- Checkpoint configs commonly use `activation_function="swish"` even though source default is GELU.
- `scale_embedding=True` is common in sampled checkpoints and changes embedding arithmetic by a `sqrt(d_model)` multiply.
- Generation starts from `decoder_start_token_id`, usually the pad token, not BOS.
- `final_logits_bias` is a separate `[1, target_vocab]` buffer added after the LM head; loading/lowering must preserve it even when it is all zeros.
- Embedding aliasing matters. With `share_encoder_decoder_embeddings=True`, encoder embeddings, decoder embeddings, and often `lm_head.weight` are one logical tied parameter.
- Source supports untied/separate decoder vocabularies via `decoder_vocab_size`, `separate_vocabs`, and `target_vocab.json`, but sampled checkpoints did not expose `target_vocab.json`.
- Multilingual target-side checkpoints encode language control as text prefix tokens such as `>>fr<<` before SentencePiece, not as a model-internal language embedding.
- Encoder and decoder layer counts can differ by config even though sampled production checkpoints are 6/6.
- Marian uses dense MHA only: no GQA/MQA, no RoPE, no ALiBi, no sliding-window/local attention in Marian source.
- `_attn_implementation` can route through eager, SDPA, FlashAttention, or flex attention helpers. DinoML should canonicalize to one dense attention semantic and gate optimized backends separately.
- Axis/layout guards: source tensors are batch-major `[B, T, C]`; attention reshapes to `[B, H, T, D]`; logits are `[B, T_dec, V]`. No NHWC/channel-last translation applies.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer `input_ids` / `decoder_input_ids` ABI, attention masks, label shifting for training/deferred.
- Embedding lookup for source and target token IDs: `[B, S] -> [B, S, d_model]`.
- Frozen sinusoidal position lookup: `[S] -> [S, d_model]`, broadcast over batch.
- Add, residual add, scalar multiply for `embed_scale`.
- View/reshape/transpose/contiguous: `[B,T,C] -> [B,T,H,D] -> [B,H,T,D]` and back.
- Slice last-token or selected-token logits for decoder-only optional path.

Neural network primitives:

- `Linear(d_model -> d_model)` for Q/K/V/O projections, with bias.
- `Linear(d_model -> ffn_dim)` and `Linear(ffn_dim -> d_model)` for encoder/decoder FFNs.
- `LayerNorm(d_model)` after each residual branch.
- Swish and GELU activation support, selected by config.
- Dropout is training-only; inference should compile as identity.
- Optional fp16 finite clamp exists only in encoder layer if fp16 produces non-finite hidden states; first inference integration can reject/omit this debug guard if parity scope is finite-input inference.

Attention primitives:

- Dense encoder bidirectional self-attention.
- Dense decoder causal self-attention with growing KV cache.
- Dense decoder cross-attention over encoder hidden states with per-layer cross-attention K/V cache.
- Mask addition before softmax, softmax over key dimension, attention value matmul.
- Attention backend should admit MHA with `num_kv_heads == num_attention_heads`.

Position/custom math:

- Static sinusoidal table generation at load/compile time.
- No RoPE/relative bias/ALiBi.

Generation/cache ops:

- `EncoderDecoderCache` containing separate self-attention and cross-attention caches.
- Self-attention cache tensors per decoder layer: keys and values `[B, H_dec, T_seen, head_dim]`.
- Cross-attention cache tensors per decoder layer: keys and values `[B, H_dec, S_src, head_dim]`, computed once then reused.
- Beam reorder requires `index_select` on batch dimension for both self and cross caches.
- Generation metadata: decoder start id, forced EOS, bad pad token suppression, beam count.

Preprocessing-coupled ops:

- SentencePiece tokenization and Moses punctuation normalization are CPU/data-pipeline work.
- Target language control is tokenizer-driven prefix tokens in source input IDs for target-multilingual checkpoints.
- EOS append and pad attention mask must match tokenizer behavior.

Quantized/packed weight metadata:

- No Marian-specific quantized or packed weight format in native Transformers source. DinoML GGUF or other storage would be an external loading/provider policy, not Marian ABI.

## 5. Layer/block breakdown

Encoder, repeated `encoder_layers`:

```text
x = Embed(input_ids) * embed_scale + SinusoidalPosition(seq)
self_q,self_k,self_v = Linear(x) split by module, each [B,S,d_model] -> [B,H,S,D]
self_attn = DenseAttention(self_q,self_k,self_v,bidirectional_padding_mask)
x = LayerNorm(x + Linear(self_attn))
ffn = Linear(d_model -> encoder_ffn_dim)(x)
ffn = activation(ffn)
ffn = Linear(encoder_ffn_dim -> d_model)(ffn)
x = LayerNorm(x + ffn)
```

Decoder, repeated `decoder_layers`:

```text
y = Embed(decoder_input_ids) * embed_scale + SinusoidalPosition(T_dec, past_len)
self_q,self_k,self_v = Linear(y) -> [B,H,T_new,D]
self_k,self_v = append_or_reuse_self_cache(layer, self_k, self_v)
self_attn = DenseAttention(self_q,self_k,self_v,causal_padding_mask)
y = LayerNorm(y + Linear(self_attn))
cross_q = Linear(y)
cross_k,cross_v = Linear(encoder_hidden) once per layer, then cache/reuse
cross_attn = DenseAttention(cross_q,cross_k,cross_v,encoder_padding_mask)
y = LayerNorm(y + Linear(cross_attn))
ffn = activation(Linear(d_model -> decoder_ffn_dim)(y))
ffn = Linear(decoder_ffn_dim -> d_model)(ffn)
y = LayerNorm(y + ffn)
logits = Linear(d_model -> target_vocab)(y) + final_logits_bias
```

All attention projections in source use bias. The LM head is bias-free; `final_logits_bias` is added separately.

## 6. Attention requirements

Required variants:

| Attention site | Causal | Source | Heads | KV heads | Head dim | Query len | KV len | Cache |
|---|---|---|---:|---:|---:|---|---|---|
| Encoder self-attn | no | encoder hidden | `encoder_attention_heads` | same | `d_model / heads` | `S_src` | `S_src` | none |
| Decoder self-attn | yes | decoder hidden | `decoder_attention_heads` | same | `d_model / heads` | `T_new` | `T_seen + T_new` | self K/V append |
| Decoder cross-attn | no | decoder query, encoder K/V | `decoder_attention_heads` | same | `d_model / heads` | `T_new` | `S_src` | cross K/V reuse |

Masking:

- Encoder self-attention uses a bidirectional padding mask derived from `attention_mask`.
- Decoder self-attention uses a causal mask combined with decoder attention mask. During generation the mask length is `past_key_values_length + seq_length`.
- Cross-attention uses a bidirectional mask over encoder keys from the source `attention_mask`.
- The eager math order is `q @ k.T * scale`, add mask, softmax over last dim, dropout in training, `attn @ v`.

Cache behavior:

- When `use_cache=True` and no cache is passed, Marian creates `EncoderDecoderCache(DynamicCache, DynamicCache)` for seq2seq operation.
- Self-attention updates every decode call by concatenating K/V on sequence dimension.
- Cross-attention projects encoder hidden states on the first use per layer, updates cross cache, sets `is_updated[layer_idx]=True`, and subsequent decode calls reuse cached K/V.
- Beam search cache reorder is batch-dimension index select for both self and cross caches.

FlashAttention/SDPA compatibility:

- Source advertises flash, SDPA, and flex support through shared Transformers attention interfaces.
- DinoML does not need to preserve backend dispatch as an ABI; it must preserve dense attention semantics, mask skipping conditions, and cache shapes.

## 7. Position encoding and custom math

Marian uses a frozen sinusoidal embedding table, not RoPE. The source initializes `MarianSinusoidalPositionalEmbedding.weight` and ignores learned updates.

Implementation summary:

```python
def marian_sinusoidal_weight(n_pos, dim):
    position_enc[pos, j] = pos / (10000 ** (2 * (j // 2) / dim))
    sentinel = dim // 2 if dim % 2 == 0 else dim // 2 + 1
    out[:, :sentinel] = sin(position_enc[:, 0::2])
    out[:, sentinel:] = cos(position_enc[:, 1::2])
    return out
```

Runtime lookup:

```python
position_ids = arange(past_len, past_len + seq_len)
pos = embed_positions(position_ids)
hidden = token_embedding * embed_scale + pos
```

This table can be precomputed from config and stored as a constant. Position IDs depend only on current decoder step and cache length.

## 8. Preprocessing and input packing

Tokenizer/model ABI:

- `MarianTokenizer` uses separate source and target SentencePiece processors plus JSON vocab lookup.
- Tokenization removes a leading `>>...<<` language-code prefix before SentencePiece and prepends that code token to the token list.
- Input construction appends EOS to a single sequence.
- `attention_mask` is caller/tokenizer supplied as `[B, S_src]`; model builds backend-specific masks internally.
- Decoder starts from `decoder_start_token_id`; for sampled checkpoints this equals `pad_token_id`.
- Official generation configs sampled set `bad_words_ids` to suppress pad and `forced_eos_token_id=0`.

CPU/data-pipeline work:

- Text normalization with `sacremoses` if available.
- SentencePiece encode/decode.
- Language-code prefix validation for multilingual target checkpoints.
- Padding, EOS append, and beam/sample generation policy.

GPU/runtime work:

- Integer token IDs and masks enter the graph.
- No packed varlen sequence descriptors, no multimodal placeholders, no scatter stitching, no image/audio tensors.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears to packed QKV GEMM

Source pattern:

```text
q = q_proj(x); k = k_proj(x); v = v_proj(x)
view/transpose each to [B,H,T,D]
```

Replacement:

```text
packed = Linear(x, concat(Wq,Wk,Wv), concat(bq,bk,bv))
split packed as Q,K,V along last dim
reshape/transpose to attention layout
```

Preconditions:

- Same input tensor for Q/K/V.
- Same output width `d_model` for all three projections.
- Bias presence is identical and all three biases are loaded.
- Weight layout transform respects PyTorch `nn.Linear` storage `[out_features, in_features]`.

Failure cases:

- Cross-attention has query from decoder hidden and K/V from encoder hidden; only K/V can be packed together.
- Any checkpoint with missing projection bias or nonstandard remote-code projection must reject this rewrite.

Parity test sketch:

- Compare packed and unpacked Q/K/V tensors before attention for random `[B,T,d_model]` in fp32 and fp16.

### Rewrite: cross-attention K/V precompute cache

Source pattern:

```text
for each decode step:
  if not is_updated[layer]: k,v = k_proj/v_proj(encoder_hidden); update cross cache
  else: reuse cross cache
```

Replacement:

```text
encoder output -> per decoder layer cross_kv constants for the session/prefix
decode step consumes cached cross_kv
```

Preconditions:

- Encoder hidden states are fixed for the generation request.
- Beam reorder is applied consistently to cross cache batch dimension.
- Source attention mask remains associated with cached encoder length.

Failure cases:

- Caller passes new `encoder_outputs` or changes source attention mask without rebuilding cache.

Parity test sketch:

- Run one-token decode twice and verify second step does not reproject encoder hidden states while logits match source.

### Rewrite: FFN swish epilogue fusion

Source pattern:

```text
hidden = swish(Linear1(x))
hidden = Linear2(hidden)
```

Replacement:

```text
GEMM + bias + swish -> GEMM + bias
```

Preconditions:

- `activation_function == "swish"` and activation is elementwise `x * sigmoid(x)`.
- Linear1 output is consumed only by activation feeding Linear2.
- Dropout disabled for inference.

Failure cases:

- Config uses GELU or another `ACT2FN` function.
- Training/dropout parity is required.

### Rewrite: last-step logits

Source pattern:

```text
lm_head(decoder_hidden[:, -1:, :]) + final_logits_bias
```

Replacement:

```text
GEMM only for last decoder token -> bias add
```

Preconditions:

- Decode step only needs next-token logits.
- Prefill/evaluation paths that request full logits must keep full `[B,T,V]`.

Failure cases:

- Sequence-level scoring or teacher-forced labels need all positions.

### Layout guard: preserve batch-major text axes

There is no image layout to translate. Guard all Marian lowering as batch-major `[B,T,C]` at public ABI boundaries. Internal attention may choose `[B,H,T,D]` or packed provider layouts, but `dim=-1` softmax and logits vocabulary axis must not be rewritten by generic layout passes.

## 10. Kernel fusion candidates

Highest priority:

- Embedding + position + scale + add: reduces launch count at encoder and decoder entry, simple shape contract.
- LayerNorm: required twice per encoder layer and three times per decoder layer; currently a hard operator gap for Marian parity.
- Dense attention prefill/decode: encoder bidirectional, decoder causal, and cross-attention dominate runtime and require cache-aware lowering.
- FFN `GEMM + bias + swish + GEMM`: sampled production checkpoints use swish and FFN width 2048, making this hot and regular.
- Last-token LM head for decode: avoids full `[B,T,V]` projection during incremental generation.

Medium priority:

- Packed self-attention QKV projection and cross-attention KV projection.
- Attention mask canonicalization to provider-friendly causal/bidirectional padding descriptors.
- Cross-attention K/V precompute per request.
- Beam cache reorder kernel or batched gather for cache tensors.

Lower priority:

- Full logits for teacher-forced scoring.
- `MarianForCausalLM` decoder-only wrapper.
- fp16 non-finite clamp parity.
- Training losses, label shifting, and dropout/layerdrop.

## 11. Runtime staging plan

Stage 1: config/tokenizer ABI loader.

- Parse `MarianConfig`, tokenizer metadata, generation config, and vocab files.
- Enforce supported first scope: `MarianMTModel`, shared vocab or explicitly handled decoder vocab, dense fp32/fp16 weights, no remote code.

Stage 2: encoder-only parity.

- Implement embedding lookup, sinusoidal positions, LayerNorm, dense bidirectional MHA, FFN activation.
- Validate encoder hidden states against Transformers for short padded batches.

Stage 3: decoder prefill parity without cache optimization.

- Run full decoder sequence with causal mask and cross-attention.
- Produce full logits plus `final_logits_bias`.

Stage 4: incremental decode cache.

- Add self-attention KV append and cross-attention K/V reuse.
- Implement cache reorder for beam search or initially restrict to greedy/batch decode.

Stage 5: optimized attention and GEMM rewrites.

- Enable packed QKV/KV projections, attention provider dispatch, FFN epilogues, last-token logits.

Stage 6: generation-controller parity.

- Support decoder-start pad token, forced EOS, pad suppression, target-language prefix, and beam search.

Stage 7: production scheduling.

- Add batching constraints for source length, target decode length, cache memory, and multilingual prefix handling.

## 12. Parity and validation plan

- Config parse tests for sampled checkpoints, including omitted/default fields.
- Tokenizer ABI tests using saved vocab/tokenizer configs: EOS append, pad decoder start, language-code prefix token preservation for `en-ROMANCE`.
- Sinusoidal table parity against source formula for even and odd hidden widths; include tiny `d_model=2`.
- Single encoder layer parity with fixed random weights and masks, fp32 tolerance around `1e-5`.
- Full encoder parity for `sshleifer/tiny-marian-en-de`.
- Decoder block parity without cache for causal self-attn plus cross-attn.
- Cache parity: prefill one token, decode next token, compare logits to full-sequence recomputation.
- Cross-cache reuse test: confirm encoder K/V projections are reused after first decode step.
- Beam reorder test: reorder both self and cross caches and compare selected beam logits.
- End-to-end greedy translation smoke for `opus-mt-en-de` after generation controller support.
- fp16 tests with relaxed tolerance, e.g. `rtol=1e-2`, `atol=1e-2`, after fp32 is stable.

## 13. Performance probes

- CPU tokenizer throughput by batch size and average source length.
- Encoder-only throughput sweep over `B` and `S_src`.
- Decoder prefill throughput over `B`, `S_src`, and prompt target length.
- Incremental decode tokens/sec over beam size, `S_src`, and generated length.
- Cache memory usage: self cache grows with target length; cross cache is fixed at `decoder_layers * 2 * B * heads * S_src * head_dim`.
- Attention backend comparison: eager composed BMM/softmax/BMM vs fused attention provider for encoder, decoder self, and cross-attention separately.
- FFN GEMM/activation fusion probe for swish.
- Last-token LM head vs full logits projection for decode.
- Beam cache reorder cost by layer count and source/target lengths.
- Config sweep for vocab-size impact on LM head and logits bandwidth: 58k, 59.5k, 65k.

## 14. Skip/defer list

- Training loss, labels, dropout, layerdrop, gradient checkpointing.
- `MarianForCausalLM` decoder-only wrapper.
- Separate-vocabulary checkpoints until a representative accessible `target_vocab.json` checkpoint is admitted.
- Beam search can be deferred behind greedy decode, but cache reorder must be added before claiming beam parity.
- FlashAttention/flex-specific source dispatch; preserve semantics first.
- Remote-code or non-native Marian variants.
- BPE-preprocessing legacy models that Transformers docs say are unsupported by Marian tokenizer.
- Quantized/packed weight loading beyond DinoML’s generic provider/storage policy.

## 15. Final implementation checklist

- [ ] Parse `MarianConfig`, generation config, and tokenizer metadata.
- [ ] Load/tie shared encoder/decoder/LM-head weights without cloning logical aliases.
- [ ] Add integer token embedding lookup for `[B,T] -> [B,T,C]`.
- [ ] Materialize Marian sinusoidal position constants and decoder `past_len` position lookup.
- [ ] Implement LayerNorm inference for `[B,T,C]`.
- [ ] Implement dense MHA encoder self-attention with padding mask.
- [ ] Implement dense causal decoder self-attention with KV cache append.
- [ ] Implement decoder cross-attention with encoder K/V cache reuse.
- [ ] Implement swish FFN path and GELU fallback gated by config.
- [ ] Implement `final_logits_bias` add and last-token LM-head optimization.
- [ ] Implement generation ABI: decoder start pad token, forced EOS, bad pad suppression, target-language prefix.
- [ ] Add cache reorder for beam search.
- [ ] Add graph rewrites for packed QKV, packed cross K/V, FFN swish epilogue, and last-token logits.
- [ ] Add parity tests: sinusoidal, one block, encoder, decoder prefill, decode cache, cross-cache reuse, tokenizer language-code path.
- [ ] Add performance probes for encoder, prefill, decode, LM head, cache memory, and tokenizer throughput.

## Gated DinoML gaps

- Embedding lookup over integer token IDs is a required first-class op for Marian; current op memory highlights embedding/model helpers as unported.
- LayerNorm is required throughout Marian and is still listed as unported in the v1 op checklist.
- Attention is the largest gate: DinoML needs dense MHA/SDPA-style masking, softmax, BMM/GEMM chains, and encoder-decoder cache ABI before `MarianMTModel` can run.
- Runtime cache state is more complex than decoder-only KV: cross-attention K/V is per-layer, fixed after encoder, and must survive beam reorder.
- Generation parity is not just logits: pad-token decoder start, forced EOS, pad suppression, and tokenizer language-code prefixes are ABI requirements.
- Layout translation should be disabled at the model boundary; only internal attention/GEMM provider layouts may change under local guards.
