# Moonshine Transformers Audit

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: UsefulSensors/moonshine-tiny, UsefulSensors/moonshine-base, UsefulSensors/moonshine-tiny-ar, UsefulSensors/moonshine-base-ar
Config source: HF config.json snapshots plus configuration_moonshine.py defaults
Source files inspected: modeling_moonshine.py, modular_moonshine.py, configuration_moonshine.py, masking_utils.py, cache_utils.py, convert_usefulsensors_to_hf.py
Any missing files or assumptions: no family-local processor; processor behavior comes from repo preprocessor_config.json using Wav2Vec2FeatureExtractor
```

Primary source links:

- Transformers generated model source: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine/modeling_moonshine.py
- Transformers modular source, authoritative for future edits: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine/modular_moonshine.py
- Transformers config source: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine/configuration_moonshine.py
- Mask/cache helpers: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/masking_utils.py and https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py

Local snapshots are under `_sources/`.

## 2. High-Level Architecture

Moonshine is an audio encoder plus autoregressive text decoder for automatic speech recognition.

```text
raw mono waveform + optional waveform mask
-> Wav2Vec2FeatureExtractor padding only
-> Conv1d/tanh + GroupNorm + Conv1d/GELU + Conv1d/GELU audio compressor
-> noncausal RoPE Transformer encoder
-> decoder token embedding
-> causal decoder self-attention with KV cache
-> decoder cross-attention over encoder states with reusable cross-attention cache
-> gated decoder MLP
-> tied LM projection
-> generation controller / tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: audio decode, resample to 16 kHz, mono waveform construction, right padding, waveform attention mask.
- GPU/runtime encoder: waveform Conv1d stack, mask downsampling, encoder bidirectional attention, encoder final LayerNorm. Encoder output can be cached independently for repeated decode over the same audio.
- Decode prefill: token embedding, causal self-attention, cross-attention over encoder states, logits.
- Incremental decode: append one or more decoder tokens, update self-attention cache, reuse cross-attention K/V after first update.

`moonshine_streaming` is adjacent but separate; it should get its own audit.

## 3. Important Config Dimensions

Source defaults from `MoonshineConfig`:

| Field | Default | Runtime impact |
| --- | ---: | --- |
| `vocab_size` | 32768 | embedding and LM projection width |
| `hidden_size` | 288 | encoder/decoder channel width |
| `intermediate_size` | 1152 | encoder MLP width, decoder gated hidden width before split |
| `encoder_num_hidden_layers` | 6 | encoder block count |
| `decoder_num_hidden_layers` | 6 | decoder block count |
| `encoder_num_attention_heads` | 8 | encoder head count |
| `decoder_num_attention_heads` | 8 | decoder self/cross head count |
| `encoder_num_key_value_heads` | defaults to encoder heads | advertised GQA/MQA field |
| `decoder_num_key_value_heads` | defaults to decoder heads | advertised GQA/MQA field |
| `pad_head_dim_to_multiple_of` | `None` | pads q/k/v head dim before attention backend |
| `encoder_hidden_act` | `gelu` | encoder MLP activation |
| `decoder_hidden_act` | `silu` | decoder gated MLP activation |
| `max_position_embeddings` | 512 | RoPE cache and generation context default |
| `partial_rotary_factor` | 0.9 via `__post_init__` | default RoPE dimension fraction if omitted |
| `rope_theta` | standardized default if omitted | RoPE base |
| `attention_bias` | false | q/k/v projection bias only when true; output projection remains biasless |
| `use_cache` | true | encoder-decoder generation cache |
| `tie_word_embeddings` | true | `proj_out.weight` tied to decoder embeddings |

Representative checkpoint sweep:

| HF repo | Hidden | Intermediate | Enc/Dec layers | Heads/KV | Head dim | RoPE fraction | Max positions | Conv channels | Params / source |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `UsefulSensors/moonshine-tiny` | 288 | 1152 | 6/6 | 8/8 | 36, padded to 40 | 0.9 | 194 | 1 -> 288 -> 576 -> 288 | 27,092,736 F32 / HF metadata |
| `UsefulSensors/moonshine-base` | 416 | 1664 | 8/8 | 8/8 | 52, padded to 56 | 0.62 | 194 | 1 -> 416 -> 832 -> 416 | 61,513,920 F32 / HF metadata |
| `UsefulSensors/moonshine-tiny-ar` | 288 | 1152 | 6/6 | 8/8 | 36, padded to 40 | 0.9 | 194 | same tiny | same topology; HF API did not expose safetensors param summary |
| `UsefulSensors/moonshine-base-ar` | 416 | 1664 | 8/8 | 8/8 | 52, padded to 56 | 0.62 | 194 | same base | 61,513,920 F32 / HF metadata |

All inspected repos use `torch_dtype=float32`, `attention_bias=false`, `pad_head_dim_to_multiple_of=8`, `vocab_size=32768`, `decoder_start_token_id=1`, `eos_token_id=2`, and `pad_token_id=2`.

## 3a. Family Variation Traps

- Config advertises GQA/MQA through `*_num_key_value_heads`, but current generated source reshapes `q_proj` output using `num_key_value_heads`, not `num_attention_heads`. Inspected checkpoints all set heads equal to KV heads. DinoML should initially reject `num_key_value_heads != num_attention_heads` for this source basis or confirm upstream fix before admitting GQA/MQA.
- `hidden_size` is not divisible by `pad_head_dim_to_multiple_of` at the head level. Actual attention backend sees padded head dims: tiny `36 -> 40`, base `52 -> 56`. Output is sliced back before `o_proj`.
- RoPE is partial and interleaved. Tiny rotates `int(36 * 0.9) = 32` dims; base rotates `int(52 * 0.62) = 32` dims. This is a useful happy accident for current checkpoints, not a general rule.
- Encoder and decoder both use RoPE self-attention. Cross-attention does not apply RoPE to encoder K/V.
- Encoder changes time length. The returned encoder attention mask is downsampled by `attention_mask[..., ::384][..., :mask_len]` where `mask_len` follows the three Conv1d output-length equations.
- Non-streaming Moonshine consumes raw waveform, not mel spectrograms. Do not route through Whisper mel preprocessing.
- `pad_token_id` is `None` in config defaults but `2` in representative checkpoints. Training label shift requires pad ID; generation repos provide it.
- `rope_scaling` appears as `null` in configs; current source can use generic `ROPE_INIT_FUNCTIONS` for non-default `rope_type`, but representative checkpoints use default RoPE only.
- Tokenizer has `<unk>=0`, `<s>=1`, `</s>=2`; `</s>` also acts as pad in repo configs. Snapshot tokenizer includes timestamp-like special tokens `<<ST_0>>` through `<<ST_767>>` at ids `32000..32767`, but the inspected non-streaming model source does not implement timestamp-specific logits processing.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- `unsqueeze(input_values, dim=1)` to `[B, 1, T]`.
- Conv layout `[B, C, T]` through the audio compressor, then `permute(0, 2, 1)` to `[B, S_enc, H]`.
- `view`, `transpose`, `contiguous`, `reshape`, `chunk(2, dim=-1)`, `cat`, `repeat_interleave`/expand-reshape for KV head repetition, `pad` on last head-dim axis, slice to remove padded head dim.
- Mask slicing with stride 384 and truncation to computed encoder length.

Neural network primitives:

- Conv1d: `1 -> H`, kernel 127, stride 64, no bias, followed by `tanh`.
- GroupNorm: `num_groups=1`, `num_channels=H`, `eps=1e-5`, source layout `[B, H, T1]`.
- Conv1d: `H -> 2H`, kernel 7, stride 3, bias true, followed by GELU.
- Conv1d: `2H -> H`, kernel 3, stride 2, bias true, followed by GELU.
- LayerNorm without bias at encoder/decoder attention and MLP boundaries.
- Encoder MLP: `Linear(H -> 4H)` + GELU + `Linear(4H -> H)`.
- Decoder MLP: `Linear(H -> 8H)`, split to two `4H` halves, `silu(gate) * value`, `Linear(4H -> H)`.
- Embedding: `[vocab_size, H]`.
- LM projection: `Linear(H -> vocab_size, bias=false)` tied to decoder embedding weight.

Attention primitives:

- Dense noncausal encoder self-attention.
- Dense causal decoder self-attention.
- Dense decoder cross-attention over encoder states.
- Eager fallback uses matmul, additive mask, fp32 softmax, dropout in training only, matmul V.
- SDPA/FlashAttention dispatch is advertised by `_supports_sdpa` and `_supports_flash_attn`, but DinoML can first match eager math.

Position/rotary ops:

- Default RoPE with `rope_theta=10000`, partial rotary factor, interleaved rotate-half.
- Cos/sin generated in fp32 and cast to model dtype.
- Decoder position IDs offset by `past_key_values.get_seq_length()` during incremental decode.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` for decoder self-attention and cross-attention.
- Self-attention cache grows by decoder token length.
- Cross-attention cache stores projected encoder K/V once per decoder layer and reuses it when `is_updated[layer_idx]` is true.
- Beam reorder must reorder both self and cross caches.

Preprocessing-coupled ops:

- Wav2Vec2FeatureExtractor padding/right mask only: sample rate 16 kHz, feature size 1, no normalization, padding value 0.0, return attention mask true.
- No STFT, FFT, mel filterbank, log-mel clamp, or Whisper-style normalization in the inspected processor configs.

Gated/config gaps:

- Reject GQA/MQA configs until the `q_proj` reshape behavior is resolved.
- Reject or route separately `moonshine_streaming`.
- Reject non-default timestamp/logit processors unless implemented by generation controller, because non-streaming source does not consume timestamp IDs inside the graph.

## 5. Layer/Block Breakdown

Audio compressor:

```text
input_values: [B, T]
x = unsqueeze -> [B, 1, T]
x = tanh(Conv1d(1, H, kernel=127, stride=64, bias=false)) -> [B, H, T1]
x = GroupNorm(groups=1, channels=H)(x)
x = gelu(Conv1d(H, 2H, kernel=7, stride=3, bias=true)) -> [B, 2H, T2]
x = gelu(Conv1d(2H, H, kernel=3, stride=2, bias=true)) -> [B, H, T3]
x = permute(0, 2, 1) -> [B, T3, H]
```

Length equations from source:

```text
T1 = int((T - 127) / 64 + 1)
T2 = int((T1 - 7) / 3 + 1)
T3 = int((T2 - 3) / 2 + 1)
```

Encoder block, repeated `encoder_num_hidden_layers`:

```text
res = x
x = LayerNorm(H, bias=false)(x)
q = Linear(H -> heads * head_dim, bias=attention_bias)(x)
k,v = Linear(H -> kv_heads * head_dim, bias=attention_bias)(x)
q,k = partial interleaved RoPE(q,k)
attn = noncausal attention(q,k,v, encoder_mask)
x = res + Linear(heads * head_dim -> H, bias=false)(attn)
res = x
x = LayerNorm(H, bias=false)(x)
x = Linear(H -> intermediate) -> GELU -> Linear(intermediate -> H)
x = res + x
```

Then encoder final `LayerNorm(H, bias=false)`.

Decoder block, repeated `decoder_num_hidden_layers`:

```text
res = x
x = LayerNorm(H, bias=false)(x)
q,k,v = decoder self-attention projections
q,k = partial interleaved RoPE(q,k)
self_attn = causal attention(q,k,v,self_cache,decoder_mask)
x = res + o_proj(self_attn)

res = x
x = LayerNorm(H, bias=false)(x)
q = projection from decoder x
k,v = projection from encoder_hidden_states or cross-cache
cross_attn = noncausal attention(q,k,v,encoder_mask)
x = res + o_proj(cross_attn)

res = x
x = LayerNorm(H, bias=false)(x)
y, gate = chunk(Linear(H -> 2*intermediate)(x), 2, dim=-1)
x = res + Linear(intermediate -> H)(silu(gate) * y)
```

For ASR logits:

```text
decoder_last_hidden_state -> Linear(H -> 32768, bias=false, tied to embedding)
```

## 6. Attention Requirements

Encoder:

- Noncausal self-attention.
- MHA for inspected checkpoints: 8 query heads, 8 KV heads.
- Tiny: head dim 36, padded backend dim 40; base: 52 -> 56.
- Query/key/value tensors before backend: `[B, heads, S, D]`; after padding `[B, heads, S, D_pad]`.
- Uses bidirectional mask derived from downsampled waveform mask.
- No KV cache for primary encoder-only run.

Decoder self-attention:

- Causal self-attention.
- Same MHA dimensions as encoder for current checkpoints.
- Position IDs offset by self-cache sequence length.
- Cache stores K/V after RoPE for self-attention, because `update()` happens after `apply_rotary_pos_emb`.
- Per layer self-cache shape is effectively `[B, kv_heads, S_dec_seen, head_dim]` in source before optional backend padding. DinoML should store unpadded logical head dim unless attention provider explicitly owns padded cache ABI.

Decoder cross-attention:

- Noncausal cross-attention from decoder hidden states to encoder hidden states.
- Query length is decoder step/prefix length; key/value length is compressed encoder length.
- Cross-attention does not apply RoPE.
- Cross-cache stores projected encoder K/V. On first use per layer, source sets `is_updated[layer_idx]=True` and updates cross cache; later decode steps reuse `layers[layer_idx].keys/values`.
- Cross-cache shape is `[B, kv_heads, S_enc, head_dim]`.

Masking:

- Decoder self mask is produced by `create_causal_mask`.
- Encoder and cross masks are produced by `create_bidirectional_mask`.
- Attention mask values are additive in eager path, before fp32 softmax.
- Packed/varlen attention is not used by Moonshine source directly.

Backend compatibility:

- Source advertises FlashAttention and SDPA.
- Head-dim padding exists specifically for optimized attention implementations. Any fused attention provider must preserve scaling by original `head_dim ** -0.5`, not padded dimension.
- If GQA is admitted later, verify the query reshape and repeat-KV semantics against upstream source; current audited checkpoints avoid the mismatch.

## 7. Position Encoding and Custom Math

Moonshine uses default RoPE with partial rotary dimensions. The custom part is the interleaved rotation and partial concat:

```python
def moonshine_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_half_interleaved(q_rot) * sin
    k_rot = k_rot * cos + rotate_half_interleaved(k_rot) * sin
    return cat(q_rot, q_pass, dim=-1), cat(k_rot, k_pass, dim=-1)
```

Cos/sin are computed from:

```text
inv_freq = 1 / rope_theta ** (arange(0, rotary_dim, 2) / rotary_dim)
freqs = inv_freq @ position_ids
emb = cat(freqs, freqs, dim=-1)
```

Precompute opportunity: for fixed `max_position_embeddings=194`, precompute cos/sin tables for tiny/base. Dynamic RoPE types are source-supported through `ROPE_INIT_FUNCTIONS`, but not present in inspected configs; reject until audited.

## 8. Preprocessing and Input Packing

Representative processor config:

| Field | Value |
| --- | --- |
| `feature_extractor_type` | `Wav2Vec2FeatureExtractor` |
| `sampling_rate` | 16000 |
| `feature_size` | 1 |
| `do_normalize` | false |
| `padding_side` | right |
| `padding_value` | 0.0 |
| `return_attention_mask` | true |

Runtime tensors:

- `input_values`: float waveform `[B, T]`.
- `attention_mask`: optional integer/bool-like `[B, T]`, 1 for valid waveform samples.
- `decoder_input_ids`: token IDs `[B, S_dec]`; generation starts with ID 1.
- `decoder_attention_mask`: optional decoder token mask.
- `encoder_outputs`: optional precomputed encoder output object with `last_hidden_state` and compressed `attention_mask`.

Downsampled encoder mask:

```text
mask_len = conv_length(T)
encoder_mask = attention_mask[..., ::384][..., :mask_len]
```

Tokenizer/generation coupling:

- BPE tokenizer JSON with byte fallback.
- Special IDs observed in tokenizer/config: `<unk>=0`, `<s>=1`, `</s>=2`; pad/eos both use 2 in checkpoint configs.
- Generation config sets `max_length=194`, matching model `max_position_embeddings`.
- Timestamp-like `<<ST_0>>..<<ST_767>>` IDs are present in tokenizer snapshots, but no graph-side timestamp logic was found in `moonshine` source.

CPU/data pipeline boundary:

- DinoML should not implement audio decode/resampling in the graph for first integration.
- Feature extraction can remain CPU-side padding/mask creation. The first GPU graph should start at `[B, T]` waveform and optional mask.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Conv1d Audio Compressor To GEMM/Im2Col

Source pattern:

```text
Conv1d(k=127,s=64) -> tanh -> GroupNorm -> Conv1d(k=7,s=3) -> GELU -> Conv1d(k=3,s=2) -> GELU
```

Replacement:

```text
WindowExtract1d -> GEMM(weight_flat.T) -> activation/norm -> WindowExtract1d -> GEMM -> GELU -> WindowExtract1d -> GEMM -> GELU
```

Preconditions:

- NCT source layout `[B,C,T]`.
- `padding=0`, `dilation=1`, `groups=1`.
- Static kernel/stride exactly as source.
- Bias absent for conv1, present for conv2/conv3.
- Preserve floor output-length equations and reject too-short audio that would produce invalid extents.

Failure cases:

- Any grouped/dilated/padded variant.
- Channel-last layout rewrite unless the whole compressor and GroupNorm axes are rewritten together.

Parity sketch: compare compressor output after each conv/norm activation for tiny/base random waveforms and masks.

### Rewrite: Attention QKV Packing

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
single packed GEMM -> split [q, k, v]
```

Preconditions:

- Self-attention only.
- `num_attention_heads == num_key_value_heads` for initial admission.
- All three projections share input `x`.
- Bias policy identical across q/k/v. Current checkpoints have no attention bias.

Weight transform:

```text
W_packed = concat([W_q, W_k, W_v], output_feature_axis)
```

Failure cases:

- Cross-attention has q from decoder and k/v from encoder, so only k/v can be packed there.
- GQA/MQA must wait for source/shape validation.

### Rewrite: Decoder Gated MLP Fusion

Source pattern:

```text
fc1(x) -> chunk(value, gate) -> silu(gate) * value -> fc2
```

Replacement:

```text
packed GEMM -> fused SwiGLU-like elementwise -> GEMM
```

Preconditions:

- Split order is `[value, gate]`, not `[gate, value]`.
- Activation is `silu` from config.
- Last dimension split exactly into two equal `intermediate_size` halves.

Failure cases:

- Config changes `decoder_hidden_act` away from `silu`.

### Rewrite: Cross-Attention Encoder K/V Prefill Cache

Source pattern:

```text
for each decoder layer: k_proj(encoder_hidden_states), v_proj(encoder_hidden_states)
```

Replacement:

```text
precompute per-layer cross K/V once after encoder, persist in cross cache
```

Preconditions:

- Encoder output and encoder mask unchanged across decode.
- Beam reorder updates cross-cache batch axis.
- Cache lifetime tied to one audio request.

Failure cases:

- Dynamic encoder output replacement mid-generation.

### Layout Guard: Audio Compressor NCT Region

The source Conv1d and GroupNorm region is axis-sensitive. A channel-last layout pass must rewrite:

- Conv1d channel axis `1` and time axis `2`.
- GroupNorm `num_channels=H` over channel axis.
- Final `permute(0,2,1)` removal or relocation.
- Attention mask downsample remains time-axis only.

Initial lowering should preserve NCT until after `permute`.

## 10. Kernel Fusion Candidates

Highest priority:

- Conv1d audio compressor kernels or GEMM lowering. This dominates encoder front-end work and has awkward large first kernel `127`.
- LayerNorm biasless + attention projection region. Every encoder/decoder block uses biasless LayerNorm before attention.
- RoPE + padded attention prefill/decode. Padding from 36/52 to 40/56 is central to provider ABI.
- Decoder self-attention KV cache and cross-attention K/V reuse. Correct cache ABI is required for generation performance.
- Gated decoder MLP fused activation multiply.

Medium priority:

- Encoder bidirectional FlashAttention/SDPA-compatible provider for MHA.
- Cross-attention provider with static-ish encoder K/V lengths.
- Last-token-only logits for incremental decode.
- Fused tied LM projection with sampling/top-k if DinoML owns generation controller later.

Lower priority:

- Dynamic RoPE variants beyond default.
- Training loss and label shifting.
- Timestamp-token processors, unless an ASR product specifically needs timestamp output.
- Beam-search cache reorder optimizations beyond correctness.

## 11. Runtime Staging Plan

1. Parse config and processor snapshots for tiny/base only. Admit `attention_bias=false`, MHA only, default RoPE, `pad_head_dim_to_multiple_of=8`.
2. Load weights with tied `proj_out.weight` / decoder embedding identity preserved.
3. Implement audio compressor parity from waveform to `[B,S_enc,H]`, including mask downsampling.
4. Implement one encoder block parity with RoPE and bidirectional attention.
5. Implement full encoder parity and allow encoder output caching.
6. Implement decoder prefill with self-attention, cross-attention, gated MLP, and logits.
7. Implement incremental decode with `EncoderDecoderCache`: self-cache append, cross-cache reuse, position offset.
8. Enable optimized attention/GEMM fusions and last-token logits.
9. Add optional generation-controller parity: start token, eos/pad handling, max length, tokenizer decode.

Stub initially:

- Training loss and label shift.
- Beam search; greedy generation is enough for first ASR parity.
- Timestamp-specific output handling.
- GQA/MQA and non-default RoPE.

## 12. Parity and Validation Plan

- Processor parity: raw arrays of variable length -> padded `input_values` and `attention_mask` against HF `AutoProcessor`.
- Conv stack parity: compare after conv1/tanh, GroupNorm, conv2/GELU, conv3/GELU, and final `[B,S,H]` permutation.
- Mask parity: verify `conv_length(T)` and `attention_mask[..., ::384][..., :mask_len]` for several lengths, including padding.
- RoPE parity: tiny/base cos/sin and q/k after partial interleaved RoPE for position IDs with and without cache offsets.
- Single encoder layer parity in fp32.
- Full encoder parity for tiny and base.
- Single decoder layer prefill parity, including cross-attention.
- Cache parity: first decode step populates self and cross caches; second step reuses cross K/V and appends self K/V.
- End-to-end greedy ASR parity on a short sample with fixed `max_new_tokens`.

Suggested tolerances:

- fp32 graph: `atol=1e-4`, `rtol=1e-4` for blocks; logits may need `2e-4` after full model.
- fp16/bf16 optimized kernels: start with `atol=2e-2`, `rtol=2e-2` for logits, tighten after provider-specific validation.

## 13. Performance Probes

- Audio preprocessing throughput: decode/resample/pad outside DinoML versus graph time.
- Conv compressor throughput by audio length and batch size.
- Encoder-only throughput by compressed sequence length.
- Decoder prefill latency for prompt lengths 1, 8, 32.
- Incremental decode tokens/sec with and without cross-cache reuse.
- KV cache memory: self-cache grows with `S_dec`; cross-cache fixed at `S_enc` per layer.
- Attention backend comparison: eager dense, SDPA-style, FlashAttention-style with padded head dim.
- Last-token logits versus full-sequence logits during decode.
- Batch-size sweep for tiny/base.
- Audio duration sweep around short-command lengths and max configured generation length.

## 14. Skip/Defer List

- Training, labels, and loss.
- Gradient checkpointing and output recording.
- Beam search beyond cache reorder correctness.
- GQA/MQA configs until source behavior is confirmed.
- Non-default/dynamic RoPE configurations.
- `moonshine_streaming`.
- Timestamp-specific generation processors and timestamp segment postprocessing.
- Quantized/packed weight loading; representative repos are F32 safetensors.
- Multi-GPU/tensor parallelism.

## 15. Final Implementation Checklist

- [ ] Parse `MoonshineConfig` and reject unsupported config variants.
- [ ] Parse `Wav2Vec2FeatureExtractor` processor config for waveform/mask ABI.
- [ ] Load tiny/base weights and preserve tied LM head alias.
- [ ] Implement Conv1d audio compressor with exact length/mask downsampling.
- [ ] Implement GroupNorm groups=1 on NCT audio features.
- [ ] Implement biasless LayerNorm.
- [ ] Implement partial interleaved RoPE with position offset.
- [ ] Implement MHA attention with optional head-dim padding and original-dim scaling.
- [ ] Implement encoder bidirectional mask path.
- [ ] Implement decoder causal mask path.
- [ ] Implement decoder cross-attention and encoder mask path.
- [ ] Implement `EncoderDecoderCache` ABI with self-cache append and cross-cache reuse.
- [ ] Implement decoder gated MLP split order `[value, gate]`.
- [ ] Implement tied embedding/LM projection logits.
- [ ] Add conv, mask, RoPE, one-layer, encoder, decoder prefill, and decode parity tests.
- [ ] Benchmark conv, encoder, prefill, decode, and cache memory separately.
