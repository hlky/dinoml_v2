# Transformers Audit: `moonshine_streaming`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: UsefulSensors/moonshine-streaming-{tiny,small,medium}
Config source: official HF config.json files, fetched 2026-05-13
Primary runtime target: automatic speech recognition, encoder-decoder conditional generation
```

Source files inspected:

- [configuration_moonshine_streaming.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine_streaming/configuration_moonshine_streaming.py)
- [modeling_moonshine_streaming.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine_streaming/modeling_moonshine_streaming.py)
- [modular_moonshine_streaming.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine_streaming/modular_moonshine_streaming.py)
- [processing_moonshine_streaming.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine_streaming/processing_moonshine_streaming.py)
- Related: Moonshine base modeling/config and Wav2Vec2 feature extractor/padding helpers.

The generated modeling/processing files are the exact runtime surface. The
modular file is the authoritative future edit source.

Any missing files or assumptions:

- No remote code is required for the official HF checkpoints.
- Native Transformers `forward` does not expose a true incremental audio
  frontend state ABI. Official [ONNX streaming configs](https://huggingface.co/UsefulSensors/moonshine-streaming/tree/main)
  were inspected as an adjacent deployment contract, not as native source
  behavior.
- No DinoML imports, tests, model execution, or weight inspection were run.

## 2. High-level architecture

Moonshine Streaming is an ASR encoder-decoder:

```text
raw mono waveform + attention_mask
  -> Wav2Vec2 padding only
  -> learned audio frontend: frame CMVN -> asinh compression -> Linear -> causal Conv1d x2
  -> streaming-style encoder with per-layer sliding-window self-attention
  -> decoder cross-attention over encoder states + causal decoder self-attention
  -> vocab projection -> generation controller/tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: audio decode/resample to 16 kHz mono, pad waveform to a
  multiple of 80 samples, emit `input_values` and `attention_mask`.
- Audio frontend: source graph owns frame reshape and conv downsampling; ONNX
  deployment additionally exposes frontend recurrent buffers.
- Encoder: independently cacheable per audio chunk only if DinoML defines a
  streaming state ABI; native HF only returns compressed `last_hidden_state`
  and compressed `attention_mask`.
- Decoder prefill/decode: text-token autoregressive decoder with self KV cache
  and cross-attention KV cache derived from encoder outputs.
- Logits/sampling: `proj_out(hidden)` to vocab size 32768. Tokenizer and
  generation config own BOS/EOS/start-token behavior.

## 3. Important config dimensions

Representative checkpoint sweep:

| checkpoint | encoder hidden | decoder hidden | layers | heads | head_dim | encoder MLP | decoder MLP | rotary factor | windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `moonshine-streaming-tiny` | 320 | 320 | 6 | 8 | 40 | 1280 | 1280 | 0.8 | `(16,4),(16,4),(16,0),(16,0),(16,4),(16,4)` |
| `moonshine-streaming-small` | 620 | 512 | 10 | 8 | 64 | 2480 | 2048 | 0.5 | first 2 and last 2 are `(16,4)`, middle 6 are `(16,0)` |
| `moonshine-streaming-medium` | 768 | 640 | 14 | 10 | 64 | 3072 | 2560 | 0.5 | first 2 and last 2 are `(16,4)`, middle 10 are `(16,0)` |

Common fields:

| field | value / behavior |
| --- | --- |
| `model_type` | `moonshine_streaming` |
| `is_encoder_decoder` | `true` |
| `sample_rate` / `frame_ms` | 16000 Hz / 5 ms, so `frame_len=80` samples |
| `vocab_size` | 32768 |
| `num_key_value_heads` | equal to attention heads in official configs |
| `attention_bias` / dropout | `false` / `0.0` |
| decoder activation | gated SiLU MLP |
| encoder activation | GELU MLP |
| RoPE | default RoPE, theta 10000, partial rotary factor 0.8 tiny or 0.5 small/medium |
| cache support | `EncoderDecoderCache(DynamicCache, DynamicCache)` when `use_cache=True` |
| preprocessor | `Wav2Vec2FeatureExtractor`, no normalization, attention mask returned, pad to multiple of 80 |

## 3a. Family variation traps

- Small/medium have `encoder_config.hidden_size != hidden_size`; decoder adds a
  learned positional embedding in encoder width, then a bias-free projection to
  decoder width.
- Official configs include `encoder_hidden_size` and `ffn_mult`, but the
  inspected source does not read those fields directly.
- `MoonshineStreamingConfig` source defaults omit `num_key_value_heads`, but
  decoder source reads it. Official checkpoints include it; synthetic default
  configs should be rejected or patched before graph construction.
- Decoder attention reshapes queries using `num_key_value_heads`; official
  configs are MHA (`kv_heads == heads`). Treat GQA/MQA decoder configs as
  unsupported for this source basis unless upstream fixes/validates them.
- `pad_head_dim_to_multiple_of` pads Q/K/V only for backend attention calls and
  slices attention output afterward. Official configs set it to `null`.
- Native HF source is not chunk-incremental for audio. ONNX configs expose
  `sample_buffer`, `conv1_buffer`, `conv2_buffer`, `frame_count`, and
  `total_lookahead=16`; those are not native Python model inputs.
- Encoder attention windows are layer-specific and can look right by 4 encoded
  steps for edge layers. DinoML must keep mask semantics distinct from decoder
  causal attention.

## 4. Operator coverage checklist

Tensor/layout ops:

- Waveform reshape `[B, audio_len] -> [B, frames, 80]`; requires audio length
  divisible or padded to 80.
- Transpose `[B,T,C] <-> [B,C,T]` around Conv1d.
- `view`, `reshape`, `chunk(2, dim=-1)`, `cat`, `stack`, prefix slicing, pad
  last-dim for optional head padding.
- Attention-mask downsample and per-layer 4D additive masks.

Neural primitives:

- Frame CMVN: mean, subtract, square, mean, add eps, sqrt, divide over last dim.
- Learned asinh compression: `asinh(exp(log_k) * x)`.
- `Linear(80 -> encoder_hidden)`, bias false.
- Causal Conv1d kernel 5 stride 2 left pad 4:
  `encoder_hidden -> 2*encoder_hidden`, then `2*encoder_hidden -> encoder_hidden`.
- Encoder MLP: `Linear(Henc -> 4Henc) -> GELU -> Linear(4Henc -> Henc)`.
- Decoder gated MLP: `Linear(Hdec -> 2*intermediate)`, split into value/gate,
  `SiLU(gate) * value`, `Linear(intermediate -> Hdec)`.
- LayerNorm variants: encoder custom `LayerNorm(no affine) * (gamma + 1)`;
  decoder standard bias-free LayerNorm.

Attention primitives:

- Encoder: noncausal local MHA with layer-local window mask; Q/K/V/O are
  bias-free by official config.
- Decoder self-attention: causal MHA with RoPE on Q/K before cache update.
- Decoder cross-attention: noncausal MHA over encoder states, cross K/V cached
  after first generation step.
- Eager attention path uses fp32 softmax and casts probabilities back to query
  dtype. FlashAttention/SDPA are declared supported by Transformers.

Generation/cache ops:

- `EncoderDecoderCache` with separate self and cross `DynamicCache` objects.
- Cache reorder/reset are inherited from Transformers generation cache, not
  specialized in this source.
- BOS/start token id 1, EOS 2, pad 0. `shift_tokens_right` is training/loss
  support and can be deferred for inference-only.

Preprocessing-coupled ops:

- Wav2Vec2 padding emits `attention_mask`; `do_normalize=false`, so no feature
  extractor normalization is required.
- No mel/STFT. The model consumes raw waveform samples.

## 5. Layer/block breakdown

Audio frontend:

```text
x: [B, samples]
x = reshape(x, [B, frames, 80])
x = (x - mean(x,-1)) / sqrt(mean((x-mean)^2,-1) + 1e-6)
x = asinh(exp(log_k) * x)
x = silu(Linear80_to_Henc(x))
mask = frame mask from sample attention_mask
x = transpose to [B,Henc,T]
x, mask = causal Conv1d(k=5,s=2,left_pad=4), mask conv with ones > 0
x = silu(x)
x, mask = causal Conv1d(k=5,s=2,left_pad=4), mask conv with ones > 0
x = transpose to [B,Tenc,Henc]
```

Encoder layer, repeated `N`:

```text
res = x
x = StreamingLayerNorm(x)
q,k,v = Linear(x) as [B, heads, Tenc, head_dim]
x = local noncausal attention(q,k,v, per-layer sliding window)
x = res + Linear_out(x)
res = x
x = StreamingLayerNorm(x)
x = Linear(Henc,4Henc) -> GELU -> Linear(4Henc,Henc)
x = res + x
```

Decoder layer, repeated `N`:

```text
res = x
x = LayerNorm(x)
q,k,v = self projections; q,k = partial RoPE(q,k)
self K/V = update DynamicCache
x = causal self-attention(q,k,v,self_cache)
x = res + o_proj(x)
res = x
x = LayerNorm(x)
cross k,v = project encoder_hidden_states or reuse cross cache
x = bidirectional cross-attention(q from decoder, k/v from encoder)
x = res + o_proj(x)
res = x
x = LayerNorm(x)
x = gated SiLU MLP
x = res + x
```

Decoder input setup:

```text
encoder_hidden += learned_pos_emb[0:Tenc] in encoder width
encoder_hidden = Linear(Henc -> Hdec) if widths differ
token_ids -> embedding [B,Tdec,Hdec]
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention over compressed audio frames.
- MHA for official configs (`num_key_value_heads == num_attention_heads`).
- Layer-specific sparse/local mask:
  - left side: allow keys where `0 <= q-k < left_window_size`.
  - right side: allow keys where `0 < k-q < right_window_size`.
  - `(16,0)` is causal-looking local encoder attention; `(16,4)` admits four
    future compressed frames.
- No encoder KV cache in native source.

Decoder self-attention:

- Causal autoregressive self-attention.
- Q/K/V shapes before optional padding: `[B, heads, Tdec, head_dim]`.
- Self cache stores RoPE-applied keys and raw values after projection.
- `position_ids` start at cached sequence length.

Decoder cross-attention:

- Query from decoder tokens, K/V from projected encoder states.
- Cross K/V are cached per layer after first use; later decode steps reuse
  `past_key_values.cross_attention_cache.layers[layer_idx].keys/values`.
- Rectangular attention: `Tquery` usually 1 during decode, `Tkv=Tenc`.
- Cross-attention mask is bidirectional padding mask over encoder states.

FlashAttention/SDPA:

- Source advertises `_supports_flash_attn` and `_supports_sdpa`.
- Fused parity must preserve fp32 softmax semantics in eager fallback,
  optional head padding, RoPE before self-cache update, and cross-cache reuse.

## 7. Position encoding and custom math

Decoder self-attention RoPE uses a partial rotary prefix:

```python
dim = int(head_dim * partial_rotary_factor)
inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
cos = cat([freqs, freqs], dim=-1).cos()
sin = cat([freqs, freqs], dim=-1).sin()
cos = cos[..., :cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
sin = sin[..., :sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
q_rot, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
q = cat([q_rot * cos + rotate_half(q_rot) * sin, q_tail], dim=-1)
k = cat([k_rot * cos + rotate_half(k_rot) * sin, k_tail], dim=-1)
```

Encoder positional information is not RoPE. The decoder adds a learned
absolute positional embedding to encoder hidden states before cross-attention.

Audio frontend custom math:

```python
cmvn(x) = (x - mean(x, -1)) / sqrt(mean((x - mean)^2, -1) + 1e-6)
compress(x) = asinh(exp(log_k) * x)
```

## 8. Preprocessing and input packing

Processor:

- `MoonshineStreamingProcessor` wraps a Wav2Vec2 feature extractor plus a fast
  tokenizer.
- Defaults: `padding=True`, `pad_to_multiple_of=80`, `return_tensors="pt"`.
- Preprocessor config uses 16 kHz sampling, zero padding, no normalization, and
  returns an attention mask.
- The neural graph consumes raw `input_values`, not mel features.

Native model packing:

- Frames are formed by reshape into 80-sample chunks. Padding to a multiple of
  80 is therefore part of the ABI.
- `attention_mask.sum(-1) // 80` determines valid frame count; fractional tail
  samples are ignored after padding.
- Two stride-2 causal convs compress time to `ceil(ceil(frames/2)/2)`.
- The compressed mask is returned by the encoder because sequence length
  changes.

Streaming metadata:

- Official ONNX metadata exposes frontend state buffers and `total_lookahead=16`.
- DinoML should treat that as a separate streaming admission target:
  `frontend_state_in + audio_chunk -> frontend_state_out + encoded_frames`.
- Native HF parity can start with whole-waveform runs and later add a
  chunk/state ABI that matches ONNX metadata.

## 9. Graph rewrite / lowering opportunities

### Rewrite: frame Linear as batched GEMM

Source pattern: `reshape([B,S] -> [B,F,80]) -> Linear(80,Henc,bias=False)`.

Replacement: flatten `[B*F,80] -> GEMM_rrr(weight.T) -> reshape [B,F,Henc]`.

Preconditions:

- audio length padded/truncated to a multiple of 80.
- `frame_len == 80` from config.
- dense contiguous row-major frame layout.

Failure cases: non-16 kHz or changed `frame_ms`, non-contiguous input, or a
future feature extractor that emits precomputed features.

### Rewrite: causal Conv1d stride-2 as explicit provider op

Source pattern: left pad 4, Conv1d kernel 5 stride 2 in `[B,C,T]`.

Replacement: initially a Conv1d provider; later consider im2col/GEMM for fixed
small kernel.

Preconditions:

- `kernel_size=5`, `dilation=1`, `left_pad=4`, no right pad.
- Mask path must be lowered too: pad mask, conv with all-ones kernel, compare
  `> 0`, multiply output.

Failure cases: layout pass must preserve time/channel axes; NHWC-style
translation needs axis rewrites for `Conv1d` and mask convolution.

### Rewrite: local encoder attention mask to windowed attention

Source pattern: dense additive bidirectional mask built from
`sliding_window_mask_function`.

Replacement: windowed attention kernel parameterized by `(left,right)` per
layer.

Preconditions:

- No arbitrary user mask beyond padding mask.
- Window pairs exactly match config, and right window is 0 or 4 for official
  configs.
- Attention output tensors are not requested, or dense reconstruction is
  separately implemented.

Failure cases: `output_attentions=True`, custom window config, or backend that
cannot handle right-lookahead.

### Rewrite: decoder cross K/V precompute

Source pattern: cross-attention projects encoder states on first decoder step
and writes to cross cache.

Replacement: explicit `cross_kv` stage after encoder/projector and before
decode loop.

Preconditions:

- Encoder states fixed for the request.
- Cache layout exactly `[B, heads, Tenc, head_dim]`.
- Learned encoder positional embedding and optional `Henc -> Hdec` projection
  run before K/V projection.

Failure cases: cache invalidation when encoder outputs are overridden,
multi-utterance packing without separate encoder masks.

## 10. Kernel fusion candidates

Highest priority:

- Audio frontend frame CMVN + asinh + Linear + SiLU. This dominates the unique
  nonstandard math and is easy to validate independently.
- Causal Conv1d + mask update for streaming chunks. Needed before true
  low-latency inference.
- Local-window encoder attention for `(16,0)` and `(16,4)` masks; dense
  attention wastes work.
- Decoder self-attention with RoPE + KV cache; standard decode hot path.
- Cross K/V precompute and cache ABI; avoids repeated encoder projection.

Medium priority:

- Encoder custom LayerNorm scale form.
- Decoder gated SiLU MLP fusion.
- Last-token-only logits for decode.
- Optional head-dim padding support if a future checkpoint sets
  `pad_head_dim_to_multiple_of`.

Lower priority:

- Training loss and `shift_tokens_right`.
- Attention output materialization.
- Alternate RoPE types beyond official `default`.

## 11. Runtime staging plan

Stage 1: Config and whole-waveform frontend parity.

- Parse official configs and reject missing `num_key_value_heads`.
- Implement processor ABI expectations: 16 kHz, `pad_to_multiple_of=80`,
  attention mask.
- Validate audio frontend output and compressed mask on random tensors.

Stage 2: Encoder-only parity.

- Implement custom LayerNorm, local masks, encoder MHA, and encoder MLP.
- First use dense attention with window mask, then add windowed backend.

Stage 3: Decoder prefill parity.

- Add token embeddings, learned encoder position embedding, optional
  `Henc -> Hdec` projection, decoder blocks, and logits.
- Run without cache first.

Stage 4: Decode cache parity.

- Implement `EncoderDecoderCache` ABI with self cache and cross cache separated.
- Add cross-K/V precompute stage.

Stage 5: True streaming ABI.

- Introduce frontend state buffers matching ONNX metadata:
  `sample_buffer`, `sample_len`, `conv1_buffer`, `conv2_buffer`, `frame_count`.
- Add chunk/window metadata and guards for `total_lookahead=16`.

Stage 6: Optimizations.

- Windowed encoder attention, fused audio frontend, fused MLPs, last-token
  logits, and continuous batching around decoder cache.

## 12. Parity and validation plan

- Unit parity for `FrameCMVN`, `AsinhCompression`, custom LayerNorm, RoPE, and
  sliding-window mask predicate.
- Frontend parity for input lengths: 0/1 frame edge cases, exactly 80, odd
  frame counts, and processor-padded lengths.
- Conv/mask parity for both causal conv layers, checking output mask and
  zeroing behavior.
- Single encoder layer parity with dense mask.
- Full encoder parity for tiny, then small/medium shape-only compile.
- Decoder block parity for self-attention no-cache, self-cache append, and
  cross-cache reuse.
- Prefill logits parity on tiny with whole-waveform input.
- Decode token parity for a short utterance using generation config ids.
- Streaming parity, once implemented, against ONNX chunk metadata and final
  whole-waveform reconciliation.

Suggested tolerances: fp32 `1e-4` absolute for frontend/blocks, fp16/bf16
`1e-2` around attention/logits unless backend-specific accumulation differs.

## 13. Performance probes

- Processor/audio padding throughput versus model frontend throughput.
- Frontend-only samples/sec by chunk size and batch size.
- Encoder-only throughput by compressed frame length, window type `(16,0)` vs
  `(16,4)`, and dense versus windowed attention backend.
- Decoder prefill throughput by target prompt length and encoder length.
- Decode tokens/sec with and without cross-K/V precompute.
- Cache memory usage: self cache grows with decoded tokens; cross cache fixed
  at encoder compressed length.
- Streaming latency: audio chunk size, lookahead budget, frontend state copy
  cost, encoder window boundary handling.
- Tiny/small/medium sweep, especially `Henc != Hdec` projection overhead for
  small/medium.

## 14. Skip/defer list

- Training loss and label shifting.
- Gradient checkpointing.
- Beam search and advanced generation processors.
- GQA/MQA decoder variants; official configs are MHA and source reshape is not
  safe for decoder GQA.
- Alternate RoPE types and non-null head-dim padding until a checkpoint needs
  them.
- Native true streaming state for first whole-waveform parity, but keep ABI
  design visible because it is the product-critical path.
- ONNX `ten-vad` and external VAD/postprocessing.

## 15. Final implementation checklist

- [ ] Parse `MoonshineStreamingConfig` and nested `MoonshineStreamingEncoderConfig`.
- [ ] Reject configs missing decoder `num_key_value_heads` or with decoder
      `num_key_value_heads != num_attention_heads`.
- [ ] Load Wav2Vec2 feature-extractor metadata: 16 kHz, zero pad,
      `return_attention_mask`, `pad_to_multiple_of=80`.
- [ ] Implement frame reshape, CMVN, asinh compression, frame Linear, and SiLU.
- [ ] Implement causal Conv1d k5/s2/left-pad4 plus mask propagation.
- [ ] Implement encoder custom LayerNorm.
- [ ] Implement per-layer local attention masks and dense fallback.
- [ ] Add windowed encoder attention backend for `(16,0)` and `(16,4)`.
- [ ] Implement encoder MLP and final norm.
- [ ] Implement decoder token embeddings, learned encoder position embedding,
      and optional encoder-to-decoder projection.
- [ ] Implement partial RoPE exactly, including interleaved cos/sin behavior.
- [ ] Implement decoder self-attention KV cache ABI.
- [ ] Implement decoder cross-attention K/V precompute and cross-cache reuse.
- [ ] Implement gated SiLU decoder MLP.
- [ ] Implement logits projection with untied output weights.
- [ ] Add whole-waveform tiny parity, then small/medium shape and config tests.
- [ ] Design a separate streaming frontend state ABI using ONNX
      `streaming_config.json` as admission evidence.
