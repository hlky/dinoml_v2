# BlenderBot Small Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/blenderbot_small-90M primary; facebook/blenderbot-90M legacy alias/config peer
Config source: local config class plus HF raw config snapshots under _sources/
Source files inspected:
  transformers/src/transformers/models/blenderbot_small/configuration_blenderbot_small.py
  transformers/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py
  transformers/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py
  transformers/src/transformers/models/blenderbot/modeling_blenderbot.py
  transformers/src/transformers/models/blenderbot/configuration_blenderbot.py
  transformers/src/transformers/masking_utils.py
  transformers/src/transformers/cache_utils.py
  transformers/tests/models/blenderbot_small/test_modeling_blenderbot_small.py
  transformers/tests/models/blenderbot_small/test_tokenization_blenderbot_small.py
Any missing files or assumptions:
  No remote-code files are required. No imports/tests were run. HF configs were fetched from public raw URLs.
```

Saved snapshots and notes are in `_sources/`. HF sources used include [facebook/blenderbot_small-90M](https://huggingface.co/facebook/blenderbot_small-90M), [facebook/blenderbot-90M](https://huggingface.co/facebook/blenderbot-90M), [Xenova/blenderbot_small-90M](https://huggingface.co/Xenova/blenderbot_small-90M), and small/community variants listed in `_sources/notes.md`.

Primary DinoML runtime target for this report: `BlenderbotSmallForConditionalGeneration` for conversational seq2seq generation. `BlenderbotSmallModel` encoder-decoder hidden-state export is optional. `BlenderbotSmallForCausalLM` standalone decoder is deferred unless DinoML wants decoder-only compatibility for fine-tuned community checkpoints.

## 2. High-level architecture

BlenderBot Small is a text-only encoder-decoder Transformer with learned absolute positional embeddings, tied token embeddings, MHA self-attention, decoder cross-attention, GELU FFNs, and an LM projection tied to the shared embedding matrix.

```text
BPE tokenizer/dialog history packing -> encoder tokens + padding mask
  -> token embedding + learned positions + embedding LayerNorm
  -> encoder self-attention blocks
  -> cached encoder_hidden_states
  -> decoder BOS/previous tokens + causal mask + learned positions + embedding LayerNorm
  -> decoder self-attention cache + encoder-decoder cross-attention cache
  -> tied LM head + final_logits_bias
  -> generation controller: beams/min/max/no-repeat/forced EOS
```

Independently stageable pieces:

- CPU/data pipeline: BPE tokenization, lowercasing/punctuation spacing, conversation delimiter tokens `__end__` and `__start__`, padding mask construction.
- Encoder precompute: `encoder_last_hidden_state` `[B, S_src, d_model]` and source attention mask can be cached across all decode steps for one prompt.
- Decoder prefill: fills self-attention KV for prompt/decoder prefix and cross-attention KV once per decoder layer.
- Decode: one or more new decoder tokens, growing self-attention KV, reused cross-attention KV, full-vocab or last-token logits.

## 3. Important config dimensions

Current source defaults:

| Field | Default | Runtime impact |
|---|---:|---|
| `model_type` | `blenderbot-small` | Routes to BlenderbotSmall classes. |
| `d_model` / hidden size | 512 | Embedding width, attention projection width, residual width. |
| `encoder_layers` / `decoder_layers` | 8 / 8 | Repeated encoder and decoder blocks. |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | MHA, no GQA/MQA. |
| `head_dim` | 32 inferred | `d_model // heads`; source rejects non-divisible configs. |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 2048 / 2048 | FFN `Linear(512 -> 2048 -> 512)`. |
| `vocab_size` | 50265 source default | Official checkpoints override to 54944. |
| `max_position_embeddings` | 512 | Learned absolute position table for encoder and decoder. |
| `activation_function` | `gelu` | Source uses `ACT2FN`; configs observed all GELU. |
| `scale_embedding` | false source default | Official 90M configs set true, so token embeddings are multiplied by `sqrt(512)`. |
| `use_cache` | true | Decoder creates `EncoderDecoderCache` for seq2seq. |
| `pad/bos/eos/decoder_start` | 0/1/2/1 | Generation ABI and mask construction. |
| `forced_eos_token_id` | 2 default | Generation-controller behavior, not core graph op. |

Representative checkpoint/config sweep:

| Repo/config | Architecture | d_model | Layers enc/dec | Heads | FFN | Vocab | Max pos | scale | Generation notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `facebook/blenderbot_small-90M` | ConditionalGeneration | 512 | 8/8 | 16 | 2048 | 54944 | 512 | true | beams 10, min 20, max 128, length penalty 0.65, no-repeat 3, forced EOS 2. |
| `facebook/blenderbot-90M` | ConditionalGeneration | 512 | 8/8 | 16 | 2048 | 54944 | 512 | true | Same generation shape; legacy ID used by local slow tests. |
| `Xenova/blenderbot_small-90M` | ConditionalGeneration | 512 | 8/8 | 16 | 2048 | 54944 | 512 | true | Transformers.js/ONNX mirror of the official config. |
| `lordtt13/blenderbot_small-news` | ConditionalGeneration | 512 | 8/8 | 16 | 2048 | 54944 | 512 | true | Fine-tuned checkpoint, same operator shape. |
| `kellyjiayixu/..._blenderbot_small` | CausalLM | 512 | 8/8 config, decoder-only runtime | 16 | 2048 | 54944 | 512 | true | `is_decoder=true`, `is_encoder_decoder=false`; out of scope for first seq2seq target. |
| `onnx-internal-testing/tiny-random-...ConditionalGeneration-ONNX` | ConditionalGeneration | 16 | 2/2 | 4 | 4 | 54944 | 20 | false | Useful tiny shape, not production. |

## 3a. Family variation traps

- Official 90M checkpoints override source defaults: `vocab_size=54944` and `scale_embedding=True`. DinoML must not hardcode the config-class defaults.
- Historical checkpoint fields `normalize_before`, `normalize_embedding`, `layernorm_variant`, `do_blenderbot_90_layernorm`, `add_final_layer_norm`, `static_position_embeddings`, `extra_pos_embeddings`, and `force_bos_token_to_be_generated` appear in configs but are not read by the inspected current source. Treat them as ignored for this source basis.
- Full BlenderBot is not just a larger Small. In inspected source, full `blenderbot` uses scaled embedding modules and pre-norm blocks with final encoder/decoder layer norms; `blenderbot_small` applies embedding LayerNorm and post-attention/post-FFN norms inside each block, with no final encoder/decoder layer norm.
- `encoder_attention_heads` and `decoder_attention_heads` may differ in the config schema, although observed checkpoints use both as 16. Cross-attention query heads follow decoder heads.
- `d_model` must be divisible by each attention head count; source computes `head_dim = embed_dim // num_heads` and raises if the product mismatches.
- No RoPE, ALiBi, relative position bias, sliding window, MQA/GQA, MoE, gated MLP, tensor parallelism, packed QKV weights, or source-coupled quantized weight format is implemented.
- `BlenderbotSmallForCausalLM` mutates config to decoder-only and uses only the decoder. It is a separate runtime contract from the seq2seq conversation target.
- Source tensor layout is sequence-major logical `[B, T, C]` for hidden states and `[B, H, T, D]` for attention. There is no NHWC/channel-last opportunity; layout rewrites should guard attention transpose/view contracts.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup `[B, S] -> [B, S, 512]`, padding index 0.
- Learned position embedding lookup from `arange(past_len, past_len + T)` for decoder and `arange(T)` for encoder.
- Add, residual add, scalar multiply by `sqrt(d_model)` when `scale_embedding=True`.
- `view/reshape` `[B, T, 512] -> [B, T, 16, 32]`, transpose to `[B, 16, T, 32]`, transpose back, contiguous, reshape.
- Slice last logits for `BlenderbotSmallForCausalLM(logits_to_keep)` only if decoder-only path is admitted.
- Mask creation/broadcast to backend-specific 4D or block masks.

Neural primitives:

- `Linear(512 -> 512)` Q/K/V/O with bias for every attention module.
- Encoder FFN: `Linear(512 -> 2048)` + GELU + `Linear(2048 -> 512)`, both biased.
- Decoder FFN: same dimensions.
- `LayerNorm(512)` for embedding norm, self-attn norm, cross-attn norm, final FFN norm.
- Inference dropout is identity.
- fp16 overflow clamp in encoder layers only when non-finite values occur; keep as a guard/fallback, not a hot-path op.

Attention primitives:

- Encoder bidirectional MHA self-attention.
- Decoder causal MHA self-attention with dynamic KV cache.
- Decoder bidirectional encoder-decoder cross-attention, rectangular query/key lengths.
- Attention math order in eager path: `matmul(q, k.T) * head_dim**-0.5`, add mask, softmax over key dim, dropout, `matmul(weights, v)`.
- Source advertises FlashAttention, SDPA, and FlexAttention compatibility through generic attention interfaces; DinoML can start with dense attention and later map to provider-backed kernels.

Position ops:

- Learned absolute positions only. No rotary math.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` for seq2seq.
- Self-attention cache grows per decoder layer with tensors shaped `[B, H_dec, T_dec_seen, 32]`.
- Cross-attention cache stores encoder-projected K/V per decoder layer shaped `[B, H_dec, S_src, 32]` and uses `is_updated[layer_idx]` to avoid recomputing after first decode step.
- Beam search requires cache reorder by `beam_idx` across both self and cross caches.
- LM head weight is tied to shared embedding; `final_logits_bias` buffer `[1, vocab]` is added to logits.

Preprocessing-coupled ops:

- Python BPE tokenizer lowercases, spaces punctuation/apostrophes, maps newline to `__newln__`, and uses special tokens `__start__`, `__end__`, `__unk__`, `__null__`.
- Conversation history is plain text with delimiter tokens, not a neural graph feature.

## 5. Layer/block breakdown

Encoder input:

```text
input_ids [B,S] -> shared/token embedding [B,S,512]
x = embedding * sqrt(512) when scale_embedding
pos = learned_position[0:S] [S,512], broadcast over batch
x = LayerNorm(x + pos)
x = dropout(x)  # inference identity
mask = bidirectional padding mask
```

Encoder block, repeated `encoder_layers`:

```text
res = x
q,k,v = Linear(512 -> 512, bias=True)(x), split to [B,16,S,32]
a = MHA(q,k,v, bidirectional source mask)
x = LayerNorm(res + Linear(512 -> 512, bias=True)(a))
res = x
x = GELU(Linear(512 -> 2048, bias=True)(x))
x = Linear(2048 -> 512, bias=True)(x)
x = LayerNorm(res + x)
```

Decoder input:

```text
decoder_input_ids [B,T] -> token embedding [B,T,512]
pos = learned_position[past_len : past_len + T] [T,512]
x = LayerNorm(token_embedding) + pos
self_mask = causal mask over past_len + T plus decoder padding
cross_mask = bidirectional source padding mask over encoder_hidden_states
```

Decoder block, repeated `decoder_layers`:

```text
res = x
q = Linear(512 -> 512)(x)
self_k,self_v = Linear(512 -> 512)(x), append/update self KV cache
x = LayerNorm(res + out_proj(SelfMHA(q,self_k,self_v,self_mask)))

res = x
q = Linear(512 -> 512)(x)
cross_k,cross_v = Linear(512 -> 512)(encoder_hidden_states), cache/reuse by layer
x = LayerNorm(res + out_proj(CrossMHA(q,cross_k,cross_v,cross_mask)))

res = x
x = GELU(Linear(512 -> 2048)(x))
x = Linear(2048 -> 512)(x)
x = LayerNorm(res + x)
```

LM head:

```text
logits = hidden_states @ shared_embedding.weight.T
logits = logits + final_logits_bias
```

Weight aliasing contract:

- `model.encoder.embed_tokens.weight`, `model.decoder.embed_tokens.weight`, and `model.shared.weight` are one logical tied embedding.
- `lm_head.weight` is tied to `model.shared.weight` for conditional generation.
- `final_logits_bias` is an independent zero-initialized/resizable buffer, not tied.

## 6. Attention requirements

Required variants:

| Variant | Causal | Q source | K/V source | Shape |
|---|---|---|---|---|
| Encoder self-attn | No | encoder hidden `[B,S,512]` | same | Q/K/V `[B,16,S,32]` |
| Decoder self-attn | Yes | decoder hidden `[B,T,512]` | decoder hidden plus past cache | Q `[B,16,T,32]`, K/V `[B,16,T_past+T,32]` |
| Decoder cross-attn | No | decoder hidden `[B,T,512]` | encoder hidden `[B,S,512]` | Q `[B,16,T,32]`, K/V `[B,16,S,32]` |

Masking:

- Encoder and cross-attention masks are bidirectional padding masks derived from 2D masks.
- Decoder self-attention mask combines causal ordering with optional decoder padding over `past_key_values_length + seq_length`.
- If `attention_mask` is omitted in decoder and not TorchDynamo compiling, source creates an all-ones mask of `[B, past_len + T]`.
- Dense attention can use additive masks; backend-specific source may return `None` or block masks for optimized paths.

Cache behavior:

- Current source creates `EncoderDecoderCache(DynamicCache(config), DynamicCache(config))` when `use_cache=True` and encoder hidden states are present.
- Self-attention K/V are stored after projection and after reshape/transpose, before attention softmax.
- Cross-attention K/V are also stored after projection and reshape/transpose. The `is_updated[layer_idx]` flag means the first decode step populates cross cache; later steps reuse it without reprojecting encoder states.
- Cache reorder for beam search must reorder both self and cross caches on batch/beam dimension.
- Packed/varlen attention is not model-specific; do not admit it as required for first parity.

FlashAttention/SDPA compatibility:

- Source dispatches via `ALL_ATTENTION_FUNCTIONS` using `config._attn_implementation`, passing `scaling=head_dim**-0.5` and dropout.
- First DinoML path can lower to explicit dense MHA. A later FlashAttention path must preserve mask semantics, rectangular cross-attention, and cache update order.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding:

```python
def blenderbot_small_position_ids(seq_len, past_len=0):
    return arange(past_len, past_len + seq_len)
```

Encoder uses `past_len=0`. Decoder uses `past_key_values.get_seq_length()` to offset positions during decode. Position tables are static constants of shape `[max_position_embeddings, d_model]`; position IDs depend on runtime sequence length and cache length. There is no RoPE, ALiBi, sinusoidal table, relative bias, or dynamic extrapolation.

Source-specific math:

```python
def encoder_input_embedding(token_embedding, pos_embedding, scale):
    return layer_norm(token_embedding * scale + pos_embedding)

def decoder_input_embedding(token_embedding, pos_embedding):
    return layer_norm(token_embedding) + pos_embedding
```

That encoder/decoder asymmetry is important for parity.

## 8. Preprocessing and input packing

CPU/data pipeline:

- `BlenderbotSmallTokenizer` reads `vocab.json` and `merges.txt`.
- Tokenizer lowercases text, inserts spaces around punctuation and apostrophes, collapses multiple spaces, maps newline to `__newln__`, and applies BPE with `@@` continuation markers.
- Special token strings from tokenizer config: `__start__` BOS, `__end__` EOS, `__unk__`, `__null__` PAD.
- Model inputs are `input_ids` and `attention_mask`; tokenizer config sets `model_max_length=512`.

GPU/runtime graph:

- `input_ids` `[B,S_src]`, `attention_mask` `[B,S_src]`.
- `decoder_input_ids` `[B,T]` or generation controller-supplied current tokens.
- `decoder_attention_mask` optional. For generation it may be omitted and created as all ones.
- `encoder_outputs` may be supplied directly to skip encoder recomputation; for DinoML this is a useful staged ABI.

Generation controller behavior outside the core graph:

- Official generation config: `num_beams=10`, `min_length=20`, `max_length=128`, `length_penalty=0.65`, `no_repeat_ngram_size=3`, `forced_eos_token_id=2`.
- Training-only `labels` path shifts tokens right and computes cross entropy. Defer for inference.

## 9. Graph rewrite / lowering opportunities

### Rewrite: tied LM head -> GEMM with aliased embedding

Source pattern:

```text
lm_logits = Linear(d_model -> vocab, bias=False)(decoder_hidden)
lm_logits += final_logits_bias
```

Replacement:

```text
GEMM_RCR(decoder_hidden_flat [B*T,512], shared_embedding [vocab,512]) + row bias [vocab]
```

Preconditions:

- `lm_head.weight` is tied to `model.shared.weight`.
- Weight storage is dense `[vocab, d_model]`.
- Output logits shape is `[B,T,vocab]`; optional last-token-only path may slice `T=1` before GEMM.

Failure cases:

- Resized embeddings with untied head must be represented as separate constants.
- Quantized/packed embeddings need a separate loading/provider contract.

Parity test sketch: compare logits for full `T` and decode `T=1` against Transformers with the same weights.

### Rewrite: attention projections -> packed QKV for self-attention

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x) as separate Linear(512 -> 512)
```

Replacement:

```text
single GEMM [B*T,512] x packed_weight [1536,512] -> split [q,k,v]
```

Preconditions:

- Applies only to self-attention where Q/K/V source tensor is identical.
- Packed row order must be `[q rows][k rows][v rows]` to match source split.
- Bias packs in same order.
- Cross-attention can pack K/V from encoder states but Q has a different source and decode cadence.

Failure cases:

- Cross-attention cache reuse should not recompute packed K/V after `is_updated`.
- Provider path must preserve `[B,H,T,D]` cache layout.

### Rewrite: cross-attention K/V precompute

Source pattern:

```text
for each decoder layer:
  k_proj(encoder_hidden_states), v_proj(encoder_hidden_states)
```

Replacement:

```text
encoder output -> per-layer K/V projection once -> cross_attention_cache
decode steps reuse cache
```

Preconditions:

- `encoder_hidden_states` and source mask are unchanged across generation.
- Decoder layer weights are fixed and per-layer distinct.
- Beam expansion/reorder semantics are explicit.

Failure cases:

- Changing encoder outputs between decode steps invalidates cross cache.
- Batch/beam expansion must apply to cached encoder K/V consistently.

### Rewrite: last-token-only logits

Source pattern:

```text
ConditionalGeneration always computes lm_head(outputs[0]) for all decoder positions.
```

Replacement:

```text
During incremental decode, slice hidden_states[:, -1:, :] before LM GEMM.
```

Preconditions:

- Only for generation steps where caller needs next-token logits.
- Not for teacher-forced full sequence logits or loss parity.

Failure cases:

- Beam scoring with processors still needs full vocab for selected positions, but not prior positions.

### Layout guard: sequence/attention axis preservation

Source pattern:

```text
[B,T,C] -> view(*input_shape, heads, head_dim) -> transpose(1,2) -> [B,H,T,D]
```

Guard:

- No global layout translation may rewrite sequence axis `dim=1`, head axis, or softmax `dim=-1` without rewriting every reshape/transpose/cache consumer in the attention region.
- This family has no image/channel-last region; treat hidden states as row-major sequence tensors.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual around attention/FFN: every block has multiple LayerNorms; parity must handle BlenderBot Small's post-norm placement and embedding norms.
- Dense MHA prefill/decode with KV cache: three attention variants are on the critical path; decode depends on efficient cache appends and cross-cache reuse.
- LM head GEMM with tied embedding and optional last-token-only logits: vocab 54944 makes logits expensive.
- FFN `GELU(Linear) -> Linear`: `512 -> 2048 -> 512` repeated 16 total blocks.

Medium priority:

- Self-attention packed QKV projection for encoder and decoder prefill.
- Cross-attention K/V precompute and cache materialization.
- Mask creation/lowering to provider-friendly forms for causal and bidirectional masks.
- Beam cache reorder and batch/beam view handling.

Lower priority:

- fp16 non-finite clamp fallback in encoder layers.
- Decoder-only `BlenderbotSmallForCausalLM` support.
- Full generation-controller parity for no-repeat n-gram and beam search; useful for end-to-end, not needed for block-level graph parity.

## 11. Runtime staging plan

Stage 1: parse config and load weights.

- Admit `model_type="blenderbot-small"`, dense weights, tied embeddings, learned positions, and ignored historical fields.
- Reject or defer decoder-only configs unless explicitly targeting `ForCausalLM`.

Stage 2: single block parity.

- Implement embedding norm asymmetry, encoder block, decoder block with dense attention and no cache.

Stage 3: encoder-only parity.

- Run `[B,S] -> encoder_last_hidden_state` with padding masks.
- Validate standalone encoder save/load equivalent shape/outputs conceptually.

Stage 4: seq2seq prefill parity.

- Run full encoder plus decoder prefix without cache optimization; compare logits.

Stage 5: decode cache ABI.

- Add `EncoderDecoderCache` equivalent with self KV append, cross KV once-per-layer update flag, and beam reorder.

Stage 6: optimized providers and rewrites.

- Add packed QKV/KV rewrites, last-token logits, cross-K/V precompute, and provider-backed attention/GEMM paths.

Stage 7: generation controller integration.

- Add beam search, no-repeat n-gram, min/max length, forced EOS, and tokenizer conversation packing as a host/controller layer.

## 12. Parity and validation plan

- Config tests: source defaults vs official 90M overrides, especially `vocab_size` and `scale_embedding`.
- Tokenizer smoke: `"sam"` maps as in local test for `facebook/blenderbot-90M`; decoded text lowercases and spaces punctuation.
- Operator tests: LayerNorm, GELU FFN, learned positional lookup with `past_len`, additive masks, attention softmax over last dim.
- Single-layer parity: encoder layer and decoder layer with random dense tensors and masks, fp32 tolerance around `1e-5` to `1e-4`.
- Encoder parity: compare `encoder_last_hidden_state` for short/long padded inputs.
- Prefill logits parity: full `BlenderbotSmallForConditionalGeneration` logits for fixed decoder prefix.
- Decode parity: no-past vs past incremental output should match selected slices within about `1e-3`, mirroring local test tolerance.
- Cache reorder parity: beam reorder should permute both self and cross caches on batch dimension.
- fp16 parity: relaxed tolerance around `1e-2` for logits/hidden states; include encoder clamp fallback stress only as a targeted edge test.
- End-to-end conversation parity: use official generation config and compare generated token IDs for short prompts after graph parity is stable.

## 13. Performance probes

- Tokenizer throughput and host conversation-packing latency.
- Encoder throughput over `B` and `S_src` sweep up to 512.
- Decoder prefill throughput over `T_prefill` with and without packed QKV.
- Decode tokens/sec for `B*beams`, `S_src`, and generated length sweeps.
- Cross-attention K/V precompute cost vs recompute baseline.
- KV cache memory: self cache `2 * decoder_layers * B * beams * heads * T_dec * head_dim * dtype_size`; cross cache `2 * decoder_layers * B * beams * heads * S_src * head_dim * dtype_size`.
- LM head cost for full sequence vs last-token-only logits.
- Attention backend comparison: dense eager equivalent, SDPA-like, FlashAttention-like for encoder, decoder self, and cross-attention.
- Beam-search controller overhead with `num_beams=10` and no-repeat 3.

## 14. Skip/defer list

- Training loss, labels path, gradient checkpointing, LayerDrop training behavior, dropout randomness.
- Decoder-only `BlenderbotSmallForCausalLM` until seq2seq path is accepted.
- TensorFlow/JAX/ONNX parity.
- Quantized or packed weight formats; no native source contract exists.
- Multi-GPU/tensor parallelism.
- FlexAttention block masks and packed sequence detection as first-class features; source compatibility exists, but checkpoints do not require them.
- Speculative decoding and continuous batching.
- Exact beam-search text parity in Stage 1; add after logits/cache parity.

## 15. Final implementation checklist

- [ ] Parse `BlenderbotSmallConfig` and official 90M config overrides.
- [ ] Load dense weights with tied shared embedding and LM head alias.
- [ ] Preserve ignored historical config fields as metadata/rejection notes, not runtime branches.
- [ ] Implement learned absolute position embedding with decoder `past_len` offset.
- [ ] Implement encoder embedding path: scaled token embedding + position + LayerNorm.
- [ ] Implement decoder embedding path: LayerNorm(token embedding) + position.
- [ ] Implement post-norm encoder block.
- [ ] Implement post-norm decoder block with self-attn, cross-attn, and FFN.
- [ ] Implement bidirectional and causal padding masks.
- [ ] Implement `EncoderDecoderCache` ABI with self KV append and cross KV reuse flags.
- [ ] Implement beam cache reorder for both cache families.
- [ ] Lower LM head to tied-embedding GEMM plus `final_logits_bias`.
- [ ] Add guarded packed QKV self-attention rewrite.
- [ ] Add guarded cross-attention K/V precompute rewrite.
- [ ] Add last-token-only logits rewrite for incremental generation.
- [ ] Add single-layer, encoder, prefill logits, decode-cache, and cache-reorder parity tests.
- [ ] Benchmark encoder, prefill, decode, LM head, and cache memory separately.

## Gated gaps for DinoML

- **Seq2seq cache ABI:** DinoML needs explicit artifact-visible self/cross KV cache state, including cross-cache `is_updated` behavior and beam reorder. This is the main gate for incremental conversation generation.
- **Attention provider coverage:** Dense bidirectional, causal, and rectangular cross-attention must all support `[B,H,T,D]` cache layout, additive/padding masks, and `head_dim=32`.
- **LayerNorm placement:** Full BlenderBot and BlenderBot Small differ. A shared rewrite must guard post-norm Small semantics and the encoder/decoder embedding norm asymmetry.
- **LM head/vocab pressure:** Official vocab is 54944, not the source default 50265. Last-token-only logits and tied-weight GEMM are important for decode performance.
- **Generation controller:** Official conversation parity depends on beam search, min/max length, length penalty, no-repeat n-gram, and forced EOS outside the neural graph.
- **Historical config fields:** Current source ignores several legacy flags. Admission should either ignore them with a source-basis note or reject only when a future source version actually reads them.
