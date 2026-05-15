# Transformers Audit: kyutai_speech_to_text

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from transformers
Model id: kyutai/stt-2.6b-en-trfs
Config source: HF config/preprocessor/tokenizer/generation JSON plus local Transformers defaults
Source files inspected: kyutai_speech_to_text config/modeling/modular/processor/feature_extractor/converter; Mimi config/modeling; Moshi config/modeling neighbor source
Any missing files or assumptions: no weights were loaded; no model imports or execution; 1B and original 2.6B Kyutai repos are external Moshi-runtime configs, not Transformers-native KyutaiSpeechToText configs
```

Primary source links:

- Transformers source basis: `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- HF model: [kyutai/stt-2.6b-en-trfs](https://huggingface.co/kyutai/stt-2.6b-en-trfs)
- Transformers docs: [Kyutai Speech-To-Text](https://huggingface.co/docs/transformers/model_doc/kyutai_speech_to_text)
- Kyutai project page: [Kyutai STT](https://kyutai.org/stt)
- Neighbor/external runtime models: [kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en), [kyutai/stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr)

## 2. High-level architecture

Kyutai STT in Transformers is not a Whisper-like encoder-decoder. It is a Mimi audio codec plus a Moshi-like causal decoder over time-aligned text and audio-token streams.

```text
CPU audio normalization/padding -> Mimi codec encode -> packed text+audio token stream -> causal decoder prefill/decode -> text logits -> tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: raw waveform validation, 24 kHz check, right/left zero padding, optional chunk-size-aligned padding.
- Independently cacheable codec stage: Mimi causal Conv1d encoder, Mimi transformer, downsample, residual vector quantizer, producing discrete audio codes.
- Prefix/input construction: generation starts from BOS text input, pairs every text step with 32 audio codebook tokens, and injects audio BOS at position 0.
- Main decode: causal decoder with RoPE and sliding-window cache predicts text tokens.
- Output decode: tokenizer decodes text token ids; no audio waveform generation path is used for STT.

The codec stage, embedding stitch, main decoder, and tokenizer decode can be validated independently. End-to-end parity depends strongly on feature extractor delay/prefix padding and generation cache/session state.

## 3. Important config dimensions

Representative Transformers checkpoint: `kyutai/stt-2.6b-en-trfs`.

| Field | Value | Source |
|---|---:|---|
| dtype | `bfloat16` weights plus some F32 tensors | HF metadata/config |
| parameters | 2,696,431,425 total | HF safetensors metadata |
| text vocab size | 4001 | HF config |
| codebook vocab size | 2049 | HF config |
| codebooks | 32 | HF config |
| hidden size | 2048 | HF config |
| decoder layers | 48 | HF config |
| attention heads / KV heads | 32 / 32 | HF config |
| head dim | 64 | HF config |
| FFN dim | 11264 | HF config |
| activation | `silu` gated MLP | HF config/source |
| max positions | 750 | HF config |
| sliding window | 375 | HF config |
| RoPE theta | 100000.0 | HF config |
| RMSNorm eps | 1e-8 | HF config |
| cache | main `use_cache=true`, generation `sliding_window` | HF config/generation config |
| audio sample rate | 24000 Hz | preprocessor/codec config |
| feature channels | mono, `[batch, 1, samples]` after extractor | preprocessor/source |
| frame size | 1920 samples, 80 ms at 24 kHz | HF config/Mimi source property |
| audio delay/prefix | 2.5 s right delay plus 1.0 s left silence | preprocessor config |
| Mimi hidden size | 512 | nested codec config |
| Mimi transformer layers | 8 encoder + 8 decoder in Mimi model; STT encode uses encoder transformer | nested codec/source |
| Mimi heads / KV heads / head dim | 8 / 8 / 64 | nested codec config |
| Mimi sliding window | 250 | nested codec config |
| Mimi quantizers/codebook size | 32 / 2048 | nested codec config |

Representative checkpoint sweep:

| Model | Runtime family | Text vocab/card | Layers | Heads | Context/window | Delay | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `kyutai/stt-2.6b-en-trfs` | Transformers `kyutai_speech_to_text` | 4001 | 48 | 32 | max 750, slide 375 | 2.5 s + 1.0 s prefix | Direct target; ungated HF repo |
| `kyutai/stt-2.6b-en` | external `moshi` `model_type="stt"` | 4000 | 48 | 32 | context 375 | 2.5 s + 1.0 s prefix | Same family, different config schema/source runtime |
| `kyutai/stt-1b-en_fr` | external `moshi` `model_type="stt"` | 8000 | 16 | 16 | context 750 | 0.5 s + 0.0 s prefix | English/French, lower latency, non-Transformers-native |

## 3a. Family variation traps

- Do not infer text vocab from tokenizer alone. `tokenizer_config.json` leaves BOS/EOS/PAD ids null, while model/generation config uses `bos_token_id=48000` and `pad_token_id=3`; text logits are size 4001 for the direct checkpoint.
- Text and audio tokens share one embedding table with offsets. Audio pad id `69569` is outside the per-codebook local vocab and is used as the table padding index.
- The input to the decoder during generation is rank 3: `[batch, seq, 33]`, not ordinary `[batch, seq]`.
- `num_codebooks=32` equals Mimi `num_quantizers=32` for the target, but external 1B/2.6B configs use `n_q/card/text_card` fields and are not drop-in Transformers configs.
- The main decoder is MHA for the target (`num_key_value_heads == num_attention_heads`), but source supports `num_key_value_heads` as a separate field.
- Sliding-window behavior is config-sensitive: main decoder generation uses `cache_implementation="sliding_window"`, while the codec cache is also copied via prefixed `codec_*` generation config fields.
- `generate()` caps token count from raw padded input sample length and intentionally does not use `padding_mask`; variable-length batched audio can over-generate padding unless caller/postprocessing handles it.
- The Kyutai project page describes semantic VAD in the Rust server; the inspected Transformers source has no VAD head.
- External Kyutai production implementations include PyTorch/Rust/MLX. Transformers parity should not assume Rust server scheduling, websocket state, or VAD outputs exist in this model class.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank conversion for audio: feature extractor returns `[batch, channels, samples]`.
- Text/audio packing: `input_ids.unsqueeze(2)` + audio code tensor concat to `[B, T, 33]`.
- Audio code layout transpose: Mimi returns `[B, Q, F]`; Kyutai decoder consumes `[B, F, Q]`.
- Per-codebook offset add with pad preservation, embedding lookup, sum over stream axis.
- Slice/copy/update for `audio_tokens[:, frame_idxs, :]`, `current_window`, and `logits_to_keep`.

Neural primitives:

- Bias-free Linear: decoder Q/K/V/O, MLP `2048 -> 11264`, gated halve, `5632 -> 2048`, LM head `2048 -> 4001`.
- RMSNorm in fp32 accumulation with output cast back to input dtype.
- Gated SiLU MLP: split `ffn_dim` into two halves, `silu(x0) * x1`.
- Mimi causal Conv1d, ConvTranspose1d for codec decode if full Mimi parity is later needed, ELU, residual blocks, optional weight norm conversion compatibility.
- Mimi vector quantization: Euclidean nearest-codebook lookup, residual quantizer accumulation, split semantic/acoustic RVQ.

Attention primitives:

- Causal self-attention with RoPE on Q/K, MHA/GQA capable, fp32 softmax in eager path.
- SDPA path with `is_causal` dispatch and sliced causal mask.
- FlashAttention2 path with `[B, T, H, D]` layout and explicit sliding-window parameter.
- Mimi encoder transformer uses sliding-window causal mask and RoPE.

Position/rotary ops:

- Default RoPE with `theta=100000.0` for main decoder, `theta=10000.0` for Mimi codec.
- Dynamic RoPE update decorator is present; target config uses default RoPE parameters.

Generation/cache ops:

- Main decoder `Cache` over 48 layers, key/value shape before repeat `[B, 32, T, 64]` for target.
- Mimi encoder transformer cache over 8 layers, key/value shape before repeat `[B, 8, T, 64]`.
- Conv padding cache with one cached tensor per causal Mimi Conv1d plus downsample layer.
- Sliding-window cache admission, DynamicCache fallback when none is provided, FlashAttention2 rejects StaticCache.

Preprocessing-coupled ops:

- 24 kHz sampling-rate guard.
- Zero left/right padding based on `audio_silence_prefix_seconds` and `audio_delay_seconds + 1.0`.
- Optional chunk-length/overlap padding alignment.
- Mono/stereo shape validation, though target config is mono.

Discrete codebook/tokenizer ops:

- Mimi codebook size 2048 plus audio BOS id 2048 per codebook.
- Text tokenizer is `PreTrainedTokenizerFast`, converted from Kyutai/Moshi SentencePiece in the conversion script.
- Final text decode is outside the neural graph but required for ASR parity.

## 5. Layer/block breakdown

Main decoder, repeated 48 times:

```text
x: [B, T, 2048]
r = x
x = RMSNorm_fp32(x, eps=1e-8)
q = Linear(2048 -> 2048, no bias)(x).view(B,T,32,64).transpose(1,2)
k = Linear(2048 -> 2048, no bias)(x).view(B,T,32,64).transpose(1,2)
v = Linear(2048 -> 2048, no bias)(x).view(B,T,32,64).transpose(1,2)
q,k = RoPE(q,k, position_ids, theta=100000)
k,v = cache.update(k,v, layer)
attn = causal/sliding-window attention(q,k,v)
x = r + Linear(2048 -> 2048, no bias)(attn)
r = x
x = RMSNorm_fp32(x, eps=1e-8)
h = Linear(2048 -> 11264, no bias)(x).view(B,T,2,5632)
x = r + Linear(5632 -> 2048, no bias)(silu(h[...,0,:]) * h[...,1,:])
```

Embedding front-end:

```text
packed_ids: [B, T, 33]  # text slot + 32 audio-code slots
offsets = [0, vocab_size + 0*2049, ..., vocab_size + 31*2049]
ids = where(ids == audio_pad_id, ids, ids + offsets)
x = Embedding(vocab_size + 32*2049 + 1, 2048, padding_idx=69569)(ids).sum(axis=2)
```

Mimi encode stage:

```text
input_values: [B, 1, samples]
Conv1d/ELU/Resnet/strided Conv1d encoder -> [B, 512, frames_at_25Hz]
Mimi transformer on [B, frames, 512] with RoPE/sliding causal attention
downsample Conv1d stride 2 -> [B, 512, frames_at_12.5Hz]
split residual vector quantizer -> codes [B, 32, frames]
```

## 6. Attention requirements

Main decoder:

- Causal self-attention only; no cross-attention.
- Target checkpoint is MHA: 32 query heads, 32 KV heads, head dim 64. Source supports GQA via `num_key_value_heads` and `repeat_kv`.
- Q/K/V widths are each 2048 for target; output projection is 2048.
- RoPE is applied before cache update, so cached keys are post-RoPE.
- Eager path computes attention weights as `matmul(q, k.T) * 1/sqrt(64)`, adds mask, softmaxes in fp32, drops out in training, then multiplies by V.
- SDPA path repeats KV before calling `torch.nn.functional.scaled_dot_product_attention`.
- FlashAttention2 path rejects StaticCache and transposes cache layout to `[B, T, H, D]`; it passes `sliding_window`.
- Source mask construction uses `create_causal_mask` for main Kyutai decoder; generation config requests sliding-window cache. Admission should preserve the cache/window combination rather than using an unbounded dense causal mask by default.

Mimi codec attention:

- Mimi encoder transformer is also causal self-attention with 8 heads, 8 KV heads, head dim 64, hidden size 512.
- Mimi source explicitly calls `create_sliding_window_causal_mask`.
- Mimi cached keys are post-RoPE as well.

Cache types:

- Main autoregressive text/audio-token KV cache: owned by `KyutaiSpeechToTextModel`.
- Codec encoder transformer KV cache: passed as `encoder_past_key_values` into `codec_model.encode`.
- Codec causal convolution padding cache: owned by `KyutaiSpeechToTextConv1dPaddingCache`, separate from KV cache.
- Processor-derived padded audio and tokenizer decode are not KV caches.

## 7. Position encoding and custom math

Main RoPE is standard Llama-style default RoPE with a family-specific base.

```python
def kyutai_rope(q, k, position_ids, head_dim=64, theta=100000.0):
    inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = outer(position_ids.float(), inv_freq)
    emb = cat([freqs, freqs], dim=-1)
    cos, sin = cos(emb), sin(emb)
    return q * cos[:, None] + rotate_half(q) * sin[:, None], \
           k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Mimi uses the same shape of RoPE math with codec config `rope_theta=10000.0`, hidden size 512, 8 heads, head dim 64. Cos/sin are computed in fp32 under autocast-disabled scope and cast back.

Precomputable: inverse frequencies and static cos/sin tables up to max window. Dynamic input: `position_ids` offset by cache sequence length.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Raw audio must match 24 kHz or the feature extractor raises.
- Target is mono: raw input becomes `float32`, batched, padded/truncated, then transposed to `[B, channels, samples]`.
- Default padding is enabled if `padding` is `None`; simultaneous padding and truncation is rejected.
- Target preprocessor adds 1.0 s left silence and `(2.5 + 1.0) = 3.5 s` right zero padding. At 24 kHz, that is 24,000 left samples and 84,000 right samples.
- `padding_mask` is padded in parallel when padding is active, but generation's max-token cap ignores it.

Runtime/GPU graph:

- `generate()` computes `max_audio_frames = input_values.shape[-1] // 1920`.
- `_prepare_model_inputs()` allocates `audio_tokens: [B, audio_window_size, 32]` and `current_window: [B, 2]`.
- With representative `generation_config.audio_window_size=1`, generation encodes 1920 audio samples per new text step.
- At each generation step, if the current position has crossed the loaded audio window, Mimi encodes the next raw-audio slice and copies new codes into `audio_tokens`.
- The current text ids are concatenated with current audio codes into packed ids `[B, step, 33]`.

This means tokenizer/feature extractor settings are part of model parity, not merely input convenience.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed stream embedding

Source pattern:

```text
where(ids == audio_pad, ids, ids + offsets) -> Embedding -> sum(stream_axis)
```

Replacement:

```text
TextEmbedding(text_ids) + sum_i AudioEmbedding_i(audio_code_i)
```

Preconditions:

- `num_codebooks`, `codebook_vocab_size`, `vocab_size`, and `audio_pad_token_id` match the checkpoint.
- Audio embedding slices remain logical aliases of the single source table, or the loader records a deterministic split.
- Text slot is always slot 0 and audio slots are ordered codebook 0..31.

Failure cases: resized embeddings, changed audio pad id, external Moshi runtime checkpoints without the Transformers-converted table.

Parity test sketch: compare packed-table output against split-table sum for random text ids, random codebook ids, and audio pad positions.

### Rewrite: gated MLP to fused SwiGLU-style op

Source pattern:

```text
Linear(2048 -> 11264) -> view(..., 2, 5632) -> silu(first_half) * second_half -> Linear(5632 -> 2048)
```

Replacement: fused gated-MLP epilogue.

Preconditions: `hidden_act == "silu"`, `ffn_dim` even, no bias, split order exactly first-half activation times second-half gate.

Failure cases: different activation, odd FFN dim, flexible per-codebook linear path.

### Rewrite: Mimi non-overlap downsampling Conv1d to GEMM/im2col where useful

Source pattern: Mimi `Conv1d` with causal/asymmetric padding and stride from config.

Replacement: guarded im2col/GEMM for strided Conv1d.

Preconditions: preserve causal left padding, extra right padding equation, dilation, stride, pad mode, and streaming padding-cache semantics. For downsample layer, `pad_mode="replicate"` and cache behavior must be represented.

Failure cases: streaming incremental calls with hidden padding state unless cache is first-class; reflect/replicate modes without exact boundary parity.

### Rewrite: last-token-only logits

Source pattern: `logits_to_keep` slices hidden states before LM head.

Replacement: only run LM head for requested positions during decode.

Preconditions: no loss computation needing all logits; generation path uses last-token logits.

## 10. Kernel fusion candidates

Highest priority:

- Packed stream embedding plus sum: central nonstandard input ABI; avoids materializing 33 embedding tensors per step.
- RMSNorm: 48 main decoder layers plus Mimi transformer layers.
- QKV projection + RoPE + cache update: decode bottleneck, cache layout-sensitive.
- Sliding-window attention with KV cache: main runtime requirement for long streaming audio.
- Gated SiLU MLP: large `2048 -> 11264 -> 2048` block dominates dense compute.

Medium priority:

- Mimi causal Conv1d with padding-cache update: required for streaming codec parity.
- Mimi RVQ nearest-neighbor encode: required when Dinoml owns audio tokenization instead of consuming precomputed codes.
- Last-token-only LM head: generation only needs text logits for current step.

Lower priority:

- Mimi decoder/vocoder path: not required for STT output.
- FlashAttention2-specific layout path: useful later, but SDPA/eager parity can establish correctness first.

## 11. Runtime staging plan

Stage 1: config and static graph admission.

- Parse Kyutai config and nested Mimi config.
- Load weights without running model.
- Admit text-only packed-token decoder calls using supplied `input_ids: [B,T,33]`.

Stage 2: packed embedding and one-block parity.

- Implement packed stream embedding and main decoder block.
- Validate RMSNorm, RoPE, attention, gated MLP, LM head.

Stage 3: main decoder prefill/decode cache.

- Implement Dynamic/SlidingWindow cache with post-RoPE K storage.
- Add last-token logits.

Stage 4: Mimi encode parity.

- Implement Mimi causal Conv1d encoder, Mimi transformer, downsample, RVQ encode.
- Treat codec output `[B,Q,F]` to `[B,F,Q]` transpose as an ABI boundary.

Stage 5: end-to-end generation controller.

- Reproduce audio windowing, `current_window`, `audio_tokens.copy_`, BOS audio injection, and max-token cap.
- Stub semantic VAD; it is not present in Transformers source.

Stage 6: optimized kernels.

- Fuse embedding, RMSNorm, gated MLP, attention/cache, and Mimi conv/cache paths.

## 12. Parity and validation plan

- Random tensor tests:
  - packed embedding with pad positions and offsets.
  - RoPE for main decoder and Mimi against source equations.
  - RMSNorm fp32 accumulation and cast-back.
  - gated MLP split order.
  - Mimi Conv1d output length and padding equations.
- Single-layer parity:
  - main decoder layer with eager attention and no cache.
  - main decoder layer with cache update and `T=1`.
  - Mimi transformer layer with sliding-window mask.
- Codec parity:
  - short waveform through Mimi encode, compare audio codes and encoded length.
  - streaming two-slice encode vs single full encode where source semantics allow.
- Prefill/decode parity:
  - packed ids built from known text/audio tokens.
  - generated logits for one and several decode steps.
- End-to-end ASR parity:
  - small 24 kHz clip through processor and `generate`, compare token ids and decoded transcript.
  - batched variable-length clips to expose the padding-mask/max-token-gap behavior.
- Tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5` for hidden states.
  - bf16/fp16: use logits/top-token agreement first, then relaxed hidden tolerances around attention and RVQ boundaries.

## 13. Performance probes

- Processor throughput: waveform validation, padding, transpose, batching.
- Mimi encode throughput: Conv1d stack, transformer, downsample, RVQ separately.
- Decode-only tokens/sec with precomputed audio codes.
- End-to-end requests/hour for representative audio lengths.
- Batch-size sweep for concurrent streams.
- Audio length sweep, including short clips where fixed left/right padding dominates.
- Sliding-window cache memory for main decoder and Mimi encoder separately.
- RVQ codebook lookup bandwidth and temporary memory.
- Attention backend comparison: eager, SDPA, FlashAttention-like sliding window.
- Cache/session overhead probe: per-step `audio_window_size=1` versus larger windows.

## 14. Skip/defer list

- Training and gradient checkpointing.
- Mimi audio decode/vocoder path.
- Semantic VAD from Kyutai Rust server; absent in inspected Transformers model.
- External Moshi-runtime `stt` repos as direct first targets; use them only as variation references.
- Beam search and speculative decoding.
- Multi-GPU tensor parallel beyond loading metadata; source has TP/PP hints only for `lm_head`.
- Stereo audio for first target; target preprocessor is mono.
- Non-default/dynamic RoPE variants unless a future checkpoint requires them.

## 15. Final implementation checklist

- [ ] Parse `KyutaiSpeechToTextConfig` plus nested Mimi config and generation config.
- [ ] Load shared packed embedding table and preserve/split text/audio alias contract.
- [ ] Implement packed `[B,T,33]` text/audio id embedding with audio pad handling.
- [ ] Implement main decoder RMSNorm, RoPE, MHA/GQA attention, gated SiLU MLP, LM head.
- [ ] Implement sliding-window KV cache with post-RoPE key storage.
- [ ] Implement Mimi causal Conv1d encoder with padding-cache state.
- [ ] Implement Mimi transformer encode path and downsample Conv1d.
- [ ] Implement Mimi split residual vector quantizer encode.
- [ ] Implement generation controller: audio window slicing, codec encode, audio token copy, BOS audio codes, max-token cap.
- [ ] Add parity tests for packed embedding, RoPE, RMSNorm, gated MLP, cache update, Mimi Conv1d length, RVQ encode.
- [ ] Add precomputed-audio-code decoder parity before full waveform parity.
- [ ] Add end-to-end 24 kHz ASR token/transcript parity.
- [ ] Benchmark processor, Mimi encode, decode-only, and full streaming paths separately.
