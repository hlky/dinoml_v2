# LED Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout transformers
Model id: LED family, primary target LEDForConditionalGeneration for long-document seq2seq summarization/generation
Config source: local configuration_led.py defaults plus representative Hugging Face config.json snapshots under _sources/
Source files inspected:
- transformers/src/transformers/models/led/modeling_led.py
- transformers/src/transformers/models/led/configuration_led.py
- transformers/src/transformers/models/led/__init__.py
Any missing files or assumptions:
- No tokenizer source is LED-specific; __init__.py aliases RobertaTokenizer as LEDTokenizer.
- No processor/preprocessor is model-coupled.
- allenai/led-large-16384-pubmed returned 401; accessible patrickvonplaten/led-large-16384-pubmed was used as a PubMed config representative.
- No DinoML tests/imports were run, per request.
```

Local snapshots and fetched configs are saved in `agents/plans/transformers/led/_sources/`.

## 2. High-level architecture

LED is a text-only encoder-decoder. The encoder is Longformer-style noncausal local sliding-window self-attention with optional global tokens. The decoder is BART-style causal self-attention plus dense encoder-decoder cross-attention with an `EncoderDecoderCache` for generation.

```text
Roberta/BART tokenization -> token ids + attention masks + global_attention_mask
  -> shared token embedding + learned encoder positions
  -> encoder local/global long-context blocks
  -> encoder hidden-state cache
  -> decoder token embedding + learned decoder positions
  -> causal decoder self-attn with KV cache + cached cross-attn KV
  -> tied LM head + final_logits_bias -> logits -> generation controller
```

Stageable pieces:

- CPU/data pipeline: tokenization, padding/truncation, `global_attention_mask` construction, generation parameters.
- Encoder: independently cacheable for a fixed source document; most important long-context ABI.
- Decoder prefill/decode: ordinary seq2seq generation with self-attention KV growth and reusable cross-attention keys/values.
- Heads: conditional generation is primary; QA is optional/deferred for first integration.

## 3. Important config dimensions

| Field | Source/default | DinoML relevance |
|---|---:|---|
| `model_type` | `led` | dispatch to LEDConfig and LED modeling |
| `vocab_size` | 50265 default | embedding and LM head width; some variants use 50264/50266 |
| `d_model` / `hidden_size` | 1024 default, 768 base | residual width |
| `encoder_layers` | 12 default, 6 base | local/global attention depth |
| `decoder_layers` | 12 default, 6 base/distilled variants | generation cache depth |
| `encoder_attention_heads` | 16 default, 12 base | MHA head count |
| `decoder_attention_heads` | 16 default, 12 base | decoder MHA/cross-attn head count |
| `head_dim` | inferred `d_model / heads` = 64 in swept configs | source rejects non-divisible hidden size |
| `encoder_ffn_dim` | 4096 default, 3072 base | encoder MLP GEMM width |
| `decoder_ffn_dim` | 4096 default, 3072 base | decoder MLP GEMM width |
| `max_encoder_position_embeddings` | 16384 default | learned absolute position table; PRIMERA uses 4096 |
| `max_decoder_position_embeddings` | 1024 default | decoder learned position table |
| `attention_window` | 512 default; official LED configs use 1024; PRIMERA uses 512 | must be even, positive, and per encoder layer |
| `activation_function` | `gelu` | source uses `ACT2FN` |
| `use_cache` | true | decoder generation cache enabled |
| `tie_word_embeddings` | true | embedding/LM-head alias must be preserved |

Representative checkpoint sweep:

| Model id | d_model | Enc/Dec layers | Heads | FFN | Max enc/dec | Window | Vocab | Operator-significant variation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `allenai/led-base-16384` | 768 | 6/6 | 12 | 3072 | 16384/1024 | 1024 | 50265 | smaller base shape |
| `allenai/led-large-16384` | 1024 | 12/12 | 16 | 4096 | 16384/1024 | 1024 | 50265 | common pretrain large shape |
| `allenai/led-large-16384-arxiv` | 1024 | 12/12 | 16 | 4096 | 16384/1024 | 1024 | 50265 | generation defaults `max_length=512`, beams 4 |
| `patrickvonplaten/led-large-16384-pubmed` | 1024 | 12/12 | 16 | 4096 | 16384/1024 | 1024 | 50265 | accessible PubMed variant |
| `HHousen/distil-led-large-cnn-16384` | 1024 | 12/6 | 16 | 4096 | 16384/1024 | 1024 | 50264 | decoder depth and vocab differ |
| `allenai/PRIMERA` | 1024 | 12/12 | 16 | 4096 | 4096/1024 | 512 | 50266 | shorter encoder context/window and vocab differ |

## 3a. Family variation traps

- `attention_window` can be int or list. Source mutates int to a list of length `num_hidden_layers`, but validates against `num_hidden_layers`, while layers are built from `encoder_layers`; reject mismatched lengths.
- Encoder sequence is automatically right-padded to a multiple of `max(attention_window)`, then unpadded on output. Internal sparse attention additionally asserts padded sequence length is a multiple of `2 * one_sided_window`, equivalent to `attention_window`.
- Global attention changes both tensor shapes and control flow. `max_num_global_attn_indices` is data-dependent by batch.
- Encoder and decoder attention are different families: encoder is noncausal sparse local/global; decoder is dense causal plus dense cross-attention.
- No GQA/MQA: all swept configs use full MHA with `num_key_value_heads == num_attention_heads`.
- No RoPE/ALiBi. Positions are learned absolute embeddings with separate encoder/decoder max lengths.
- Some configs include historical fields such as `classif_dropout`, `gradient_checkpointing`, `output_past`, `max_position_embeddings`, and generation defaults. Current LED source does not read most of these for inference graph construction.
- Tied weights matter: `led.shared.weight`, encoder embeddings, decoder embeddings, and `lm_head.weight` are one logical parameter when tying is active.
- Layout-sensitive source operations are sequence-major reshapes/transposes, `as_strided` overlapping chunks, indexed writes for global attention, and final unpadding. Do not apply generic layout translation across the encoder attention region.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for input ids and learned positions.
- Add, dropout-as-identity in inference, residual add.
- LayerNorm over hidden dimension.
- `view`, `reshape`, `transpose`, `contiguous`, `cat`, `narrow`, `pad`, `masked_fill`, `where`, `clamp`, `split`, `squeeze`, dynamic `nonzero`, and indexed scatter/gather.
- `as_strided` or a guarded replacement for overlapping local attention chunks.
- Runtime pad-to-window and unpad.

Neural network primitives:

- Dense Linear with bias for Q/K/V/out projections and MLPs.
- Biasless LM head `Linear(d_model -> vocab_size)` tied to shared embedding plus `final_logits_bias`.
- GELU MLP activation by default.
- Optional QA head `Linear(hidden_size -> 2)` over decoder sequence output.

Attention primitives:

- Encoder local sliding-window self-attention with window `w = attention_window / 2`.
- Encoder global-token path with separate `query_global`, `key_global`, `value_global`.
- Decoder dense causal MHA.
- Decoder dense encoder-decoder cross-attention.
- Softmax with fp32 accumulation in encoder local/global paths; decoder dense path uses ordinary `softmax`.
- Attention masks using dtype min/negative infinity semantics.

Position/custom math:

- Learned absolute positions: encoder starts at 0; decoder positions offset by `past_key_values_length`.
- Query scale by `1 / sqrt(head_dim)` before attention score matmul.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)`.
- Per-layer decoder self-attention K/V cache shape `[B, H, T_dec, D]`.
- Per-layer cross-attention K/V cache shape `[B, H, T_enc, D]`, updated once and reused via `is_updated[layer_idx]`.
- Beam/cache reorder is inherited through `GenerationMixin`/cache utilities, not implemented locally in LED.

Preprocessing-coupled ops:

- Roberta/BART-style token ids, `attention_mask`, `global_attention_mask`.
- `shift_tokens_right` for default decoder inputs and labels.

## 5. Layer/block breakdown

Encoder setup:

```text
inputs_embeds = shared_embedding(input_ids)        # [B, S, D]
attention_mask = ones if omitted
attention_mask = attention_mask * (global_attention_mask + 1) when global mask exists
pad S to a multiple of max(attention_window)
mask values: padding -> dtype min, local -> 0, global -> positive
x = inputs_embeds + learned_encoder_pos[0:S_pad]
x = LayerNorm(x)
```

Encoder block, repeated `encoder_layers` times:

```text
res = x
q,k,v = Linear(D -> D, bias=True)(x)
q = q / sqrt(head_dim)
local_scores = sliding_chunks_qk(q, k, window=w)      # [B, S_pad, H, 2w+1]
local_scores += sliding padding mask
if any global tokens:
  global_key_scores = einsum(q_all, k_global_selected)
  scores = concat(global_key_scores, local_scores)
probs = softmax(scores, dim=-1, dtype=float32)
probs[masked_tokens] = 0
local/global_value_output = sliding_chunks_probs_v + selected global V matmul
if any global tokens:
  qg = query_global(global_hidden); kg/vg = key_global/value_global(all_hidden)
  global_output = dense global-token attention over all source tokens
  scatter global_output back into token positions
x = Linear(D -> D)(attn_output)
x = LayerNorm(res + x)
res = x
x = Linear(D -> encoder_ffn_dim) -> GELU -> Linear(encoder_ffn_dim -> D)
x = LayerNorm(res + x)
```

Decoder setup:

```text
decoder_ids = shift_tokens_right(input_ids or labels) when omitted
past_len = past_key_values.get_seq_length() if cache exists
x = shared_embedding(decoder_ids) + learned_decoder_pos[past_len:past_len+T]
x = LayerNorm(x)
causal_mask = create_causal_mask(...) only when T > 1
cross_mask = create_bidirectional_mask(... encoder_attention_mask ...)
```

Decoder block, repeated `decoder_layers` times:

```text
res = x
q,k,v = Linear(D -> D)(x); q *= head_dim**-0.5
k,v = update/read self KV cache
x = dense causal attention(q,k,v, decoder mask)
x = LayerNorm(res + out_proj(x))
res = x
q = Linear(D -> D)(x)
k,v = Linear(D -> D)(encoder_hidden_states) or reuse cached cross K/V
x = dense cross attention(q,k,v, encoder mask)
x = LayerNorm(res + out_proj(x))
res = x
x = Linear(D -> decoder_ffn_dim) -> GELU -> Linear(decoder_ffn_dim -> D)
x = LayerNorm(res + x)
```

LM head:

```text
logits = Linear(D -> vocab, bias=False, weight tied to shared embedding)(decoder_x) + final_logits_bias
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention.
- Full MHA, not GQA/MQA.
- Local window per token covers `w` left, self, and `w` right tokens, with `w = attention_window / 2`.
- Source pads sequence length to a multiple of `attention_window`.
- Fast path uses overlapping chunks of size `2w` and overlap `w`, implemented by `as_strided`.
- Local score/output tensors are compact band tensors, not dense `[S, S]`.
- Global tokens are selected by `global_attention_mask`. Any token can be global. Non-global tokens attend to all global keys plus local window; global tokens attend densely over all non-masked source tokens through separate global Q/K/V projections.
- Output attention tensors, when requested, require dense-ish reconstruction conventions and separate `global_attentions`; this can be deferred for inference unless debugging parity needs it.

Decoder attention:

- Causal self-attention over decoder tokens with cache.
- Dense cross-attention from decoder queries to encoder hidden states.
- Full MHA with `head_dim = d_model / decoder_attention_heads`.
- Source implementation is manual `bmm`; no local LED dispatch to FlashAttention/SDPA.
- Cache stores K/V after projection and before any repeat expansion. No position encoding is applied to K/V beyond learned position added to hidden states before projections.

Cache ABI:

```text
self_attention_cache[layer].keys/values:  [B, H, T_dec_cache, head_dim]
cross_attention_cache[layer].keys/values: [B, H, S_enc, head_dim]
is_updated[layer]: marks cross-attn K/V reusable after first decode step
```

Admission guards:

- Encoder source length after padding must be within `max_encoder_position_embeddings`.
- Decoder generated length plus cache length must be within `max_decoder_position_embeddings`.
- `hidden_size % num_attention_heads == 0`.
- `attention_window` even and positive; list length must match encoder layer count for DinoML even if source validates against `num_hidden_layers`.
- Global attention count can be zero; kernels must handle both branches.

## 7. Position encoding and custom math

LED has no RoPE, ALiBi, or relative bias. Both encoder and decoder use learned absolute position embeddings.

```python
def led_positions(seq_len, past_len=0):
    return arange(past_len, past_len + seq_len)
```

Encoder local attention core:

```python
def local_band_scores(q, k, w):
    # q/k: [B, S, H, D], S % (2*w) == 0
    # chunk into overlapping [2*w] blocks with stride w
    chunks_q = overlap_chunks(q, block=2*w, step=w)
    chunks_k = overlap_chunks(k, block=2*w, step=w)
    scores = einsum("bcxd,bcyd->bcxy", chunks_q, chunks_k)
    return diagonalize_to_band(scores)  # [B, S, H, 2*w+1]
```

Global attention output is a separate dense attention for selected global query positions, followed by an indexed overwrite into the normal attention output.

## 8. Preprocessing and input packing

Tokenizer contract:

- LED uses Roberta/BART-like tokenization via the `LEDTokenizer` alias to `RobertaTokenizer`.
- Text graph inputs are `input_ids: [B, S]`, optional `attention_mask: [B, S]`, optional `global_attention_mask: [B, S]`, and decoder ids or labels.
- `global_attention_mask` is not generated by the neural graph. Summarization examples set token 0 global; QA commonly marks question tokens global.
- `attention_mask` values are 1 for valid tokens and 0 for padding. Merged encoder mask values are 0 no attention, 1 local, 2 global before conversion to additive mask values.
- Decoder default input ids are created by shifting right and placing `decoder_start_token_id` at position 0; `-100` labels become `pad_token_id`.

GPU/runtime work:

- Embedding, position add, mask conversion, pad-to-window, and attention are graph/runtime-owned.
- Tokenization, sequence truncation, global-mask policy, beam search, and sampling are controller/data-pipeline-owned.

## 9. Graph rewrite / lowering opportunities

### Rewrite: encoder QKV GEMM packing

Source pattern:

```text
q = Linear(x); k = Linear(x); v = Linear(x)
```

Replacement:

```text
packed_qkv = GEMM(x, concat([Wq, Wk, Wv])^T) + concat([bq,bk,bv])
split last dim into q,k,v
```

Preconditions: same input tensor, same dtype/device, all projections present with bias, no observer hooks. Encoder global Q/K/V are a second independent pack candidate. Decoder self-attn and cross-attn can use separate packs.

Failure cases: tied or externally mutated projection weights, quantized per-projection metadata that cannot be concatenated, output-attention debug requiring intermediate names.

Parity test: compare q/k/v tensors and full block output for random fp32/fp16 shapes.

### Rewrite: local sliding attention to provider op

Source pattern:

```text
as_strided overlap chunks -> einsum qk -> diagonal band -> mask -> softmax -> diagonalize probs -> einsum pv
```

Replacement:

```text
LocalBandAttention(q, k, v, window=w, additive_mask, return_band_probs=False)
```

Preconditions: sequence already padded to a multiple of `attention_window`, noncausal symmetric window, no dilation, no dense attention outputs requested, contiguous `[B,S,H,D]` logical layout.

Failure cases: `output_attentions=True`, ONNX export fallback path, unsupported dynamic global-token interop, odd/invalid window.

Parity test: run one encoder layer with no global tokens and compare hidden states against Transformers for S in `{window, 2*window, 3*window}` after unpadding.

### Rewrite: global-token branch to segmented dense attention

Source pattern:

```text
nonzero(global_mask) -> pack selected global tokens -> dense bmm over full sequence -> scatter overwrite
```

Replacement:

```text
SegmentedGlobalAttention(hidden, global_indices, padding_mask)
```

Preconditions: global indices known or dynamically packed per batch, max global count bounded for workspace, source positions remain in sequence layout.

Failure cases: unbounded global count, need for exact attention tensor outputs, batch elements with different global counts without padded segment support.

Parity test: compare encoder hidden states for zero, one, and multiple global tokens per batch element.

### Rewrite: last-token-only LM head during decode

Source pattern:

```text
lm_head(decoder_hidden[:, -1:, :]) + final_logits_bias
```

Replacement: GEMM only for the final token when generation controller needs next-token logits.

Preconditions: no caller requests full logits for all decoder positions, cache decode step.

Failure cases: teacher-forced prefill loss/logits, sequence scoring requiring all positions.

## 10. Kernel fusion candidates

Highest priority:

- Encoder local/global attention provider. This is the central LED blocker; dense fallback over 16k tokens is not viable.
- LayerNorm + residual patterns for encoder/decoder blocks.
- Packed QKV GEMM for encoder local projections, encoder global projections, decoder self-attn, and decoder cross-attn.
- Decoder dense attention with KV cache and cross-attn cache.

Medium priority:

- MLP `Linear -> GELU -> Linear` with activation fusion.
- Pad-to-window plus mask conversion as a lightweight runtime prepass.
- Last-token-only LM head for decode.
- Segmented global-token packing/scatter kernels.

Lower priority:

- Output attention tensor reconstruction.
- QA head and training losses.
- ONNX-export slow attention chunking path.

## 11. Runtime staging plan

Stage 1: config/weights loader and one decoder-only dense block parity using small synthetic lengths; preserve tied embeddings and `final_logits_bias`.

Stage 2: encoder no-global local attention parity for padded source lengths and window 512/1024.

Stage 3: encoder global attention parity with bounded max global tokens and dynamic batch variation.

Stage 4: full encoder-decoder prefill logits parity for `LEDForConditionalGeneration`.

Stage 5: decode with `EncoderDecoderCache`: self-KV growth, cross-KV one-time update/reuse, last-token logits.

Stage 6: optimized provider kernels for local band attention and segmented global attention; keep dense debug fallback only for tiny S.

Stage 7: generation-controller integration for beam search and long-document summarization defaults.

Can stub initially: dropout, LayerDrop, training losses, attention outputs, QA head, ONNX export path.

## 12. Parity and validation plan

- Config parse tests for base, large, PRIMERA, distilled decoder, and vocab-size variants.
- Unit tests for `shift_tokens_right`, pad-to-window/unpad, mask merge, and learned position offsets with `past_key_values_length`.
- Local attention op parity for fp32, then fp16/bf16 if supported, with no global tokens.
- Global attention parity for global counts `{0,1,4}` and uneven per-batch global counts.
- Single encoder layer parity at S `{window, window+1, 2*window-1}` to exercise padding.
- Single decoder layer parity for prefill and one-token decode.
- Full model prefill logits parity on a small loaded checkpoint or random initialized config.
- Decode token parity for 2-4 generated steps with cache enabled.
- Suggested tolerances: fp32 `rtol=1e-4/atol=1e-5`; fp16/bf16 start around `rtol=5e-2/atol=5e-2` for attention-heavy paths and tighten after provider math is fixed.

## 13. Performance probes

- Encoder throughput by source length: 1k, 4k, 8k, 16k.
- Window sweep: 512 vs 1024.
- Global-token count sweep: 0, 1, 8, 64, dense-like stress.
- Batch-size sweep for long documents.
- Encoder local attention provider vs dense fallback for tiny S only.
- Decoder prefill vs decode tokens/sec.
- KV cache memory: self cache grows with generated tokens; cross cache is `layers * 2 * B * H * S_enc * D`.
- Last-token LM head vs full-sequence LM head.
- Mask/pad prepass overhead relative to attention kernel time.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, and LayerDrop.
- `output_attentions=True` reconstruction for local/global attention.
- QA head and any sequence classification head not exported in `__all__`.
- ONNX export branch in `_chunk`.
- Beam search internals beyond honoring generation metadata at the controller boundary.
- Dense encoder fallback for production 16k contexts.
- Quantization/packed weights; no source-coupled quantized format appears in LED source.

## 15. Final implementation checklist

- [ ] Parse `LEDConfig`, including `attention_window` int/list normalization and strict length guards.
- [ ] Load shared embeddings, tied LM head, `final_logits_bias`, encoder/decoder layers, and optional QA head.
- [ ] Implement learned encoder/decoder positional embeddings with decoder past offset.
- [ ] Implement `shift_tokens_right`, mask merge, additive mask conversion, pad-to-window, and unpad.
- [ ] Implement dense Linear, GELU, LayerNorm, residual add, softmax, and matmul/BMM coverage.
- [ ] Implement encoder local band attention provider or guarded lowering.
- [ ] Implement encoder global-token pack, dense global attention, and scatter overwrite.
- [ ] Implement decoder causal self-attention and cross-attention with `EncoderDecoderCache`.
- [ ] Add QKV projection packing rewrites with bias and weight-layout tests.
- [ ] Add last-token-only LM head rewrite for decode.
- [ ] Add parity tests for no-global encoder, global encoder, prefill logits, and multi-step decode.
- [ ] Benchmark encoder source-length/window/global-count sweeps and decoder cache throughput.

## Gated gaps for DinoML

- LED should be gated until a sparse local/global encoder attention ABI exists. The source relies on compact band tensors, dynamic global-token packing, and indexed overwrite; pretending this is dense attention would erase the point of the model.
- Layout passes need a no-layout-translation guard over encoder attention chunking and global scatter unless the entire `[B,S,H,D]` attention region is owned by a dedicated provider.
- Generation parity needs both cache families: growing decoder self-attention K/V and one-time cached encoder cross-attention K/V.
- Dynamic admission must bound source length, padded length, window, and max global tokens so scratch allocation is artifact-visible.
