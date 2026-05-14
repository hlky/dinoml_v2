# BlenderBot Transformers Audit

## 1. Source basis

Transformers commit/version: local checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary `facebook/blenderbot-400M-distill`; representative sweep also used `facebook/blenderbot-1B-distill`, `facebook/blenderbot-3B`, `hf-internal-testing/tiny-random-BlenderbotModel`, and `facebook/blenderbot-90M` only to confirm it belongs to the separate `blenderbot_small` family.

Config source:

- `https://huggingface.co/facebook/blenderbot-400M-distill/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-400M-distill/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot-400M-distill/raw/main/tokenizer_config.json`
- `https://huggingface.co/facebook/blenderbot-1B-distill/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-1B-distill/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot-3B/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-3B/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot-3B/raw/main/tokenizer_config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-BlenderbotModel/raw/main/config.json`

Source files inspected:

- `X:/H/transformers/src/transformers/models/blenderbot/configuration_blenderbot.py`
- `X:/H/transformers/src/transformers/models/blenderbot/modeling_blenderbot.py`
- `X:/H/transformers/src/transformers/models/blenderbot/tokenization_blenderbot.py`
- `X:/H/transformers/src/transformers/models/blenderbot/convert_blenderbot_original_pytorch_checkpoint_to_pytorch.py`
- `X:/H/transformers/src/transformers/masking_utils.py` by import contract only
- `X:/H/transformers/src/transformers/cache_utils.py` by import contract only

Any missing files or assumptions:

- No generated modular source was present for this family; `modeling_blenderbot.py` is the source basis.
- Primary runtime target is `BlenderbotForConditionalGeneration` for seq2seq conversation generation. `BlenderbotModel` is useful for staged encoder/decoder parity. `BlenderbotForCausalLM` is optional and lower priority because conversation checkpoints are encoder-decoder.
- `facebook/blenderbot-90M` is out of scope for this report because its config uses `model_type: blenderbot-small` and `BlenderbotSmallForConditionalGeneration`.

## 2. High-level architecture

BlenderBot is a text-only encoder-decoder Transformer for conversation generation. The encoder uses bidirectional self-attention. The decoder uses causal self-attention, encoder-decoder cross-attention, learned absolute positions, pre-attention/pre-MLP LayerNorm inside each block, and a final LayerNorm after all layers. Token embeddings are shared between encoder and decoder, and conditional generation ties the LM head to the shared embedding plus a separate `final_logits_bias`.

Dataflow:

```text
byte-level BPE tokenizer -> input_ids/attention_mask
  -> shared token embedding * optional sqrt(d_model) + learned encoder positions
  -> encoder bidirectional blocks -> encoder_hidden_states
  -> decoder start token / previous reply tokens + learned decoder positions
  -> decoder causal self-attn with self KV cache
  -> decoder cross-attn with reusable encoder K/V cache
  -> tied lm_head + final_logits_bias -> generation controller
```

Stage decomposition:

- CPU/data pipeline: byte-level BPE, chat text concatenation with `</s>`/`<s>` conventions, padding, attention masks, generation constraints.
- Encoder: independently cacheable per source conversation context, output `[B, S_src, d_model]`.
- Decoder prefill: target prefix through causal decoder plus cross-attention to encoder output.
- Decode: one or more new tokens with growing decoder self cache and reusable per-layer cross cache.
- Logits/controller: `lm_head(hidden) + final_logits_bias`; beam search and no-repeat n-gram rules are controller behavior.

## 3. Important config dimensions

Source-default `BlenderbotConfig` fields:

| Field | Default | Operator significance |
|---|---:|---|
| `vocab_size` | 8008 | Shared token embedding rows and LM head columns |
| `max_position_embeddings` | 128 | Learned position rows for encoder and decoder |
| `d_model` | 2560 | Hidden width |
| `encoder_layers` / `decoder_layers` | 2 / 24 | Repeated encoder/decoder block counts |
| `encoder_attention_heads` / `decoder_attention_heads` | 32 / 32 | MHA heads |
| `head_dim` | inferred 80 | `d_model / heads`; source requires exact divisibility |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 10240 / 10240 | FFN expansion widths |
| `activation_function` | `gelu` | FFN activation through `ACT2FN` |
| `dropout` / `attention_dropout` / `activation_dropout` | 0.1 / 0.0 / 0.0 | Inference dropout erased |
| `scale_embedding` | false | If true, token embedding multiplies by `sqrt(d_model)` |
| `use_cache` | true | Decode uses `EncoderDecoderCache` |
| `pad` / `bos` / `eos` / `decoder_start` | 0 / 1 / 2 / 1 | Token and generation ABI |
| `forced_eos_token_id` | 2 | Generation controller constraint |
| `tie_word_embeddings` | true | LM head/share weight aliasing contract |

Representative checkpoint sweep:

| Model id | `model_type` | Architecture | `d_model` | Enc/Dec layers | Heads | Head dim | FFN | Vocab | Max pos | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hf-internal-testing/tiny-random-BlenderbotModel` | `blenderbot` | `BlenderbotModel` | 16 | 2 / 2 | 4 / 4 | 4 | 4 / 4 | 1024 | 100 | debug checkpoint, no LM head architecture |
| `facebook/blenderbot-400M-distill` | `blenderbot` | `BlenderbotForConditionalGeneration` | 1280 | 2 / 12 | 32 / 32 | 40 | 5120 / 5120 | 8008 | 128 | common target; `scale_embedding=true` |
| `facebook/blenderbot-1B-distill` | `blenderbot` | `BlenderbotForConditionalGeneration` | 2560 | 2 / 12 | 32 / 32 | 80 | 10240 / 10240 | 8008 | 128 | larger hidden width, same layer counts as 400M |
| `facebook/blenderbot-3B` | `blenderbot` | `BlenderbotForConditionalGeneration` | 2560 | 2 / 24 | 32 / 32 | 80 | 10240 / 10240 | 8008 | 128 | source default shape target |
| `facebook/blenderbot-90M` | `blenderbot-small` | `BlenderbotSmallForConditionalGeneration` | 512 | 8 / 8 | 16 / 16 | 32 | 2048 / 2048 | 54944 | 512 | out of scope; separate source family |

Generation config observed for 400M/1B/3B: `num_beams=10`, `max_length=60`, `min_length=20`, `length_penalty=0.65`, `no_repeat_ngram_size=3`, `encoder_no_repeat_ngram_size=3`, `forced_eos_token_id=2`.

## 3a. Family variation traps

- `facebook/blenderbot-90M` must not be admitted under this report; it routes to `blenderbot_small`.
- Encoder and decoder depth are asymmetric for production checkpoints: 2 encoder layers and 12 or 24 decoder layers.
- `num_hidden_layers` in configs maps to `encoder_layers`; do not treat it as total layers or decoder layers.
- Current source ignores historical config fields such as `normalize_before`, `layernorm_variant`, `add_final_layer_norm`, `normalize_embedding`, `static_position_embeddings`, and `extra_pos_embeddings`.
- `scale_embedding` varies: production checkpoints set it true even though source default is false.
- Learned positions have no BART-style offset in this source; the embedding table is exactly `max_position_embeddings`, and lookup uses `arange(past_len, past_len + seq_len)`.
- Attention is plain full MHA, not GQA/MQA; `head_dim = d_model / heads`.
- Tokenizer vocab differs sharply from BlenderBot-small. In-scope large BlenderBot checkpoints use vocab 8008, while BlenderBot-small uses 54944.
- Layout translation should be guarded off for text tensors. Source semantics are `[batch, seq, hidden]` and attention `[batch, heads, seq, head_dim]`; only a fused attention lowering should own internal layout changes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token inputs `[B, S_src]`, decoder ids `[B, S_tgt]`, masks `[B, S]`.
- Embedding gather from shared table `[vocab, D]`; optional multiply by `sqrt(D)`.
- Learned position arange/gather from `[max_position_embeddings, D]`, including decoder `past_key_values_length` offset.
- Add, residual add, reshape/view, transpose `[B, S, D] <-> [B, H, S, Hd]`, contiguous materialization after attention transpose.
- Additive bidirectional and causal mask creation; 4D mask support should follow shared Transformers mask contracts.
- Cache append, cross-cache reuse, beam reorder/repeat/select.

Neural network primitives:

- LayerNorm over hidden dim with PyTorch default epsilon.
- Linear+bias for Q/K/V/O projections: `D -> D`.
- FFN `Linear(D -> F) -> GELU -> Linear(F -> D)`.
- LM head `Linear(D -> vocab, bias=False)` tied to shared embedding, plus `final_logits_bias [1, vocab]`.
- Inference erases dropout, activation dropout, attention dropout, and LayerDrop.
- Encoder fp16 clamp guard: after encoder FFN residual, source clamps fp16 hidden states to `[-finfo.max+1000, finfo.max-1000]`.

Attention primitives:

- Encoder noncausal self-attention over source sequence.
- Decoder causal self-attention with dynamic self KV cache.
- Decoder cross-attention over encoder hidden states with separately cached K/V.
- Eager attention math: `matmul(q, k.transpose(-2, -1)) * head_dim^-0.5`, additive mask, softmax over last dim, dropout, `matmul(probs, v)`.
- Source advertises FlashAttention/SDPA/flex attention support through generic dispatch, but graph parity can start from eager semantics.

Generation/cache ops:

- `shift_tokens_right(labels, pad_id, decoder_start_token_id)` for training/teacher-forced labels.
- `EncoderDecoderCache(DynamicCache, DynamicCache)` for seq2seq generation.
- Cross-cache `is_updated[layer_idx]` flag to avoid recomputing encoder K/V after first use.
- Generation-controller metadata: beam search, min/max length, no-repeat n-gram, encoder no-repeat n-gram, forced EOS.

Preprocessing-coupled ops:

- Byte-level BPE tokenizer with `add_prefix_space=true`.
- Tokenizer input names are `input_ids` and `attention_mask`.
- Conversation packing is text-level: prior turns are concatenated with special tokens, not a model-side segment tensor.

## 5. Layer/block breakdown

Embedding path:

```text
input_ids [B, S] -> shared embedding [B, S, D] * embed_scale
positions = learned_position[arange(past_len, past_len + S)] [S, D]
x = token_emb + positions
```

Encoder block, repeated `encoder_layers` times:

```text
residual = x
x_norm = LayerNorm(x)
q,k,v = Linear(D -> D, bias=True)(x_norm) -> [B, Henc, Ssrc, Hd]
attn = MHA(q,k,v, encoder padding mask)
x = residual + Linear(D -> D, bias=True)(attn)

residual = x
x_norm = LayerNorm(x)
ff = Linear(D -> Fenc, bias=True)(x_norm)
ff = GELU(ff)
ff = Linear(Fenc -> D, bias=True)(ff)
x = residual + ff
if fp16: x = clamp(x)
```

The encoder applies a final `LayerNorm(D)` after all blocks.

Decoder block, repeated `decoder_layers` times:

```text
residual = y
y_norm = LayerNorm(y)
q = Linear(D -> D)(y_norm)
k,v = Linear(D -> D)(y_norm), append to self cache
self = causal MHA(q,k,v, decoder mask)
y = residual + Linear(D -> D)(self)

residual = y
y_norm = LayerNorm(y)
q = Linear(D -> D)(y_norm)
k,v = Linear(D -> D)(encoder_hidden_states) or load cross cache
cross = MHA(q,k,v, encoder padding mask)
y = residual + Linear(D -> D)(cross)

residual = y
y_norm = LayerNorm(y)
ff = Linear(D -> Fdec)(y_norm) -> GELU -> Linear(Fdec -> D)
y = residual + ff
```

The decoder applies a final `LayerNorm(D)` after all blocks.

Conditional generation head:

```text
decoder_hidden [B, S_tgt, D]
logits = MatMul(decoder_hidden, shared_embedding.T) + final_logits_bias[1, vocab]
```

For `facebook/blenderbot-400M-distill`: `D=1280`, `H=32`, `Hd=40`, `F=5120`, encoder layers 2, decoder layers 12, vocab 8008.

## 6. Attention requirements

Required variants:

- Encoder self-attention: bidirectional full MHA, no KV cache, Q/K/V from same encoder hidden states.
- Decoder self-attention: causal full MHA, self cache grows in target sequence length.
- Decoder cross-attention: decoder queries attend to encoder K/V. Cross K/V are projected per decoder layer and reused once `EncoderDecoderCache.is_updated[layer]` is set.

Masking:

- Encoder uses `create_bidirectional_mask(config, inputs_embeds, attention_mask)`.
- Decoder self-attn creates an all-ones mask when none is supplied outside TorchDynamo, then calls `create_causal_mask` with self cache metadata.
- Decoder cross-attn uses `create_bidirectional_mask(..., encoder_hidden_states=encoder_hidden_states)` with source attention mask.
- Eager fallback uses additive mask values before softmax.

Cache ABI:

- Per decoder layer self cache stores key and value `[B, Hdec, S_self, Hd]`.
- Before decode step with one new token: `[B, Hdec, T, Hd]`; after update: `[B, Hdec, T + 1, Hd]`.
- Cross cache stores key/value `[B, Hdec, S_src, Hd]` per decoder layer. It is static for a fixed encoder output and should be beam-reordered/repeated consistently with self cache.
- Cached keys are stored after linear projection and reshape; there is no RoPE or ALiBi transform.

Backend compatibility:

- `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are true on the pretrained base class.
- Fused attention must preserve query scaling, additive mask order, softmax dimension, and output transpose/reshape. Dropout is zero in inference.
- No sliding window, local attention, block-sparse pattern, packed varlen metadata, GQA/MQA repetition, or rotary position math is required for this source.

## 7. Position encoding and custom math

BlenderBot uses learned absolute position embeddings for encoder and decoder. Unlike BART, this source does not add a position offset.

```python
def blenderbot_positions(weight, input_shape, past_len=0, position_ids=None):
    if position_ids is None:
        _, seq_len = input_shape[:2]
        position_ids = arange(past_len, past_len + seq_len)
    return embedding(weight, position_ids)
```

Decoder positions depend on `past_key_values.get_seq_length()`, so decode needs either a runtime `past_len` scalar or a bucketed decode plan.

Training/teacher-forced helper:

```python
def shift_tokens_right(ids, pad_id, start_id):
    shifted = zeros_like(ids)
    shifted[:, 1:] = ids[:, :-1]
    shifted[:, 0] = start_id
    shifted = where(shifted == -100, pad_id, shifted)
    return shifted
```

Other custom math is minimal: optional embedding scale is `sqrt(d_model)`, attention scale is `head_dim ** -0.5`, and encoder fp16 clamp uses `torch.finfo(float16).max - 1000`.

## 8. Preprocessing and input packing

Tokenizer and text ABI:

- `BlenderbotTokenizer` is a fast tokenizer wrapper backed by `tokenizers` BPE and ByteLevel pre-tokenizer/decoder.
- Default special strings: `<pad>`, `<s>`, `</s>`, `<unk>`, `<mask>`.
- In-scope checkpoints use ids `pad=0`, `bos=1`, `eos=2`, `unk=3`, `decoder_start=1`.
- Tokenizer config sets `add_prefix_space=true`; the 3B tokenizer config includes a chat template that concatenates message content and appends `eos_token`.
- Model input names are only `input_ids` and `attention_mask`; no token type ids.

GPU/runtime inputs:

- `input_ids [B, S_src]` and `attention_mask [B, S_src]`.
- Optional `decoder_input_ids [B, S_tgt]` for teacher-forced or prefill.
- During generation, the controller supplies the decoder start token id 1 and then generated tokens.

Generation-controller behavior:

- 400M/1B/3B generation configs use beam count 10, min length 20, max length 60, length penalty 0.65, no-repeat n-gram 3, encoder no-repeat n-gram 3, and forced EOS 2.
- These constraints are outside the neural graph but are required for end-to-end conversation parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: tied shared embedding and LM head

Source pattern: encoder and decoder use `model.shared`; conditional generation has `_tied_weights_keys` mapping `lm_head.weight` to `model.shared.weight`.

Replacement:

```text
EmbeddingGather(E[vocab, D], ids)
Logits = GEMM(hidden_flat[B*S, D], E.T[D, vocab]) + final_logits_bias
```

Preconditions:

- Checkpoint actually ties or byte-identically aliases `lm_head.weight` and shared embedding.
- Preserve `final_logits_bias` separately.

Failure cases:

- Resized/untied weights or vocab mismatch.

Parity test sketch:

- Compare embedding output and logits from tied GEMM against PyTorch for random ids.

### Rewrite: self-attention QKV packing

Source pattern: separate `q_proj`, `k_proj`, `v_proj` with the same input in encoder self-attn and decoder self-attn.

Replacement:

```text
PackedLinear(D -> 3D) -> split [Q, K, V]
```

Weight transform:

```python
W_qkv = concat([W_q, W_k, W_v], dim=0)
b_qkv = concat([b_q, b_k, b_v], dim=0)
```

Preconditions:

- Same source hidden tensor feeds Q/K/V.
- Bias and dtype/layout are compatible.

Failure cases:

- Cross-attention Q and K/V have different source tensors, so only K/V can be packed there.

Parity test sketch:

- Compare packed split tensors against independent projections before attention.

### Rewrite: cross-attention K/V precompute

Source pattern: each decoder layer projects encoder hidden states to cross K/V at first decode use, then marks the layer updated.

Replacement:

```text
After encoder: per decoder layer Linear(D -> 2D) over encoder_hidden_states -> cross_cache[layer]
Decode block consumes cross_cache[layer]
```

Preconditions:

- Encoder hidden states and decoder cross-attn weights are fixed for the request.
- Beam expansion/reorder applies to cross cache consistently.

Failure cases:

- Caller-supplied changing `encoder_outputs`, partial-source recomputation, or unsupported cache reorder.

Parity test sketch:

- Compare first-step source projection inside decoder versus precomputed K/V, then compare later decode outputs.

### Rewrite: last-token-only logits

Source pattern: `BlenderbotForConditionalGeneration` computes logits for all decoder hidden positions.

Replacement:

```text
hidden[:, -1:, :] -> LM head -> next-token logits
```

Preconditions:

- Decode/generation step only needs next-token logits.
- Loss and full teacher-forced logits are not requested.

Failure cases:

- Full-sequence scoring or teacher-forced parity tests.

Parity test sketch:

- Compare sliced full logits `[:, -1, :]` to direct last-token LM head.

### Rewrite: inference dropout/layerdrop erasure

Source pattern: embedding, attention, activation, and FFN dropout plus training-only LayerDrop.

Replacement: erase all dropout and LayerDrop branches in eval/inference artifacts.

Preconditions: inference-only compilation.

Failure cases: training or stochastic evaluation.

## 10. Kernel fusion candidates

Highest priority:

- Decoder self-attention with KV append and causal fused attention. Conversation generation is decode-heavy and cache correctness is gated.
- Decoder cross-attention with static cross-cache. Production checkpoints have many decoder layers, so reusing encoder K/V matters.
- LM head last-token GEMM against tied embedding. Vocab is small at 8008 but repeated full-sequence logits are still wasteful during decode.
- Pre-norm LayerNorm + projection chains. Every block normalizes before attention and before FFN.

Medium priority:

- Packed QKV for self-attention and packed KV for cross-attention.
- FFN `Linear -> GELU -> Linear` with GEMM epilogue GELU/bias where available.
- Embedding gather + scale + learned position add.
- Beam cache reorder kernels for self and cross caches.

Lower priority:

- Encoder fp16 clamp guard fusion.
- Attention mask materialization avoidance for SDPA/Flash-style paths.
- Full-sequence teacher-forced logits optimization.

## 11. Runtime staging plan

Stage 1: parse `BlenderbotConfig`, tokenizer metadata, shared embeddings, learned positions, all projection/FFN/LayerNorm weights, LM head tie, and `final_logits_bias`.

Stage 2: run encoder-only parity for one block and full encoder with bidirectional masks.

Stage 3: run teacher-forced seq2seq forward without cache: decoder causal self-attn, cross-attn, final logits.

Stage 4: implement `EncoderDecoderCache`: self-cache append, cross-cache reuse/update flags, and cache reorder.

Stage 5: implement greedy generation parity with decoder start token and forced EOS.

Stage 6: implement conversation generation controller features: beams, min/max length, no-repeat n-gram, encoder no-repeat n-gram, length penalty.

Stage 7: enable optimized attention, packed projections, cross-K/V precompute, last-token logits, and LayerNorm/FFN fusions.

Initially stub:

- Training loss, dropout, LayerDrop, gradient checkpointing, output attentions/hidden-state recording, decoder-only `BlenderbotForCausalLM`, and BlenderBot-small checkpoints.

## 12. Parity and validation plan

- Custom helper tests: learned position lookup with `past_len`, `shift_tokens_right`, embedding scale, encoder fp16 clamp, additive causal/bidirectional masks.
- Single-block parity: encoder layer and decoder layer in fp32 with random tensors and masks.
- Encoder parity: tokenized short conversation context through `facebook/blenderbot-400M-distill`.
- Teacher-forced seq2seq parity: compare logits for short source/target ids with cache disabled.
- Cache parity: prefill plus token-by-token decode must match full no-cache decode for identical target ids.
- Cross-cache parity: first decode step fills cross cache; later steps reuse it and match PyTorch outputs.
- Generation parity: greedy output first, then beam output with generation config for 400M and 3B.
- Suggested tolerances: fp32 hidden/logits `atol=1e-5`, `rtol=1e-4`; fp16/bf16 `atol=2e-2` with top-k/token parity checks.

## 13. Performance probes

- Tokenizer throughput and chat-template packing throughput.
- Encoder-only throughput over `S_src` sweep 16, 64, 128.
- Decoder prefill throughput over `S_tgt` sweep 1, 16, 60.
- Decode tokens/sec with self cache and cross cache, split by self-attn, cross-attn, FFN, and LM head.
- Beam-size sweep, especially beam 10 from representative generation configs.
- Cache memory: self cache `decoder_layers * 2 * B * beams * S_tgt * H * Hd * dtype_size`; cross cache `decoder_layers * 2 * B * beams * S_src * H * Hd * dtype_size`.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible fused path, and Dinoml native.
- Packed projection versus independent projection timing.
- Cross-K/V precompute versus lazy first-step projection timing.

## 14. Skip/defer list

- `facebook/blenderbot-90M` / `blenderbot-small`.
- Training loss, dropout, LayerDrop, gradient checkpointing.
- Decoder-only `BlenderbotForCausalLM`.
- Output attentions/hidden-state recording except for debug parity.
- Speculative/assisted decoding and full HF generation feature matrix beyond observed BlenderBot configs.
- Quantized/offloaded cache and packed weight formats; none are source-coupled in this family.
- Tokenizer execution inside DinoML runtime; accept pre-tokenized tensors initially.

## 15. Final implementation checklist

- [ ] Parse `BlenderbotConfig` with asymmetric encoder/decoder layers.
- [ ] Reject `model_type=blenderbot-small` under this audit path.
- [ ] Load shared token embedding, learned position embeddings, projections, FFN, LayerNorm, LM head tie, and `final_logits_bias`.
- [ ] Implement byte-level BPE metadata handoff or accept pre-tokenized `input_ids`/`attention_mask`.
- [ ] Implement embedding scale and learned position lookup with decoder `past_len`.
- [ ] Implement encoder bidirectional MHA block with pre-norm residuals and final encoder LayerNorm.
- [ ] Implement decoder causal self-attn, cross-attn, pre-norm FFN, and final decoder LayerNorm.
- [ ] Implement additive mask lowering for encoder, decoder causal, and cross-attention masks.
- [ ] Implement `EncoderDecoderCache` self-cache append, cross-cache reuse flags, and reorder.
- [ ] Implement tied LM head plus `final_logits_bias`.
- [ ] Add guarded QKV/KV packing rewrites.
- [ ] Add guarded cross-attn K/V precompute rewrite.
- [ ] Add last-token-only logits rewrite for decode.
- [ ] Add one-layer, full-forward, cache, and generation parity tests.
- [ ] Benchmark encoder, prefill, decode, beam cache memory, LM head, and attention backends.

