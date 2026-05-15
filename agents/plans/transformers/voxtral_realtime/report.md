# Transformers Audit: voxtral_realtime

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: mistralai/Voxtral-Mini-4B-Realtime-2602
Config source: official HF config.json, generation_config.json, processor_config.json; tiny-random ONNX config; RedHat mirror; MLX 4-bit mirror
Primary runtime target: realtime / streaming automatic speech recognition with text-token generation
DinoML assumptions: inference-only first, CUDA GPU target, stateful streaming sessions, explicit cache/session ABI, CPU/data-pipeline audio preprocessing acceptable initially
```

Source files inspected:

- `transformers/src/transformers/models/voxtral_realtime/configuration_voxtral_realtime.py`
- `transformers/src/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py`
- `transformers/src/transformers/models/voxtral_realtime/modular_voxtral_realtime.py`
- `transformers/src/transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py`
- `transformers/src/transformers/models/voxtral_realtime/processing_voxtral_realtime.py`
- `transformers/src/transformers/models/voxtral_realtime/convert_voxtral_realtime_weights_to_hf.py`
- Neighbor/composed source: `voxtral`, `auto/modeling_auto.py`, `auto/processing_auto.py`, `auto/feature_extraction_auto.py`, `auto/tokenization_auto.py`

`modeling_voxtral_realtime.py` is generated from `modular_voxtral_realtime.py`; treat the modular file as authoritative for future upstream source edits. Source notes and config snapshots are summarized in `_sources/source_notes.md`.

Missing/assumptions: the official repo uses `processor_config.json`, not `preprocessor_config.json`; `tokenizer_config.json`, `chat_template.json`, and `audio_encoder.json` returned 404. The processor requires `mistral-common` and tokenizer state from `tekken.json`, so full prompt/token packing is an external tokenizer/processor ABI, not just ordinary HF tokenizer JSON.

## 2. High-level architecture

Voxtral Realtime is an audio encoder plus causal text decoder, but it is not an encoder-decoder cross-attention model. The audio stream is encoded into text-width embeddings and added to the text token embeddings. Generation then runs a conditioned causal LM.

```text
raw audio chunks -> processor/tokenizer + log-mel features
  -> causal conv audio embedder -> causal sliding-window audio encoder
  -> group by downsample_factor -> multimodal projector
  -> add to text embeddings + delay-time conditioning
  -> causal text decoder prefill/decode -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: mono waveform validation, chunking, `mistral-common` transcription request/token packing, STFT/mel/log normalization.
- Audio frontend: two causal Conv1d layers with mutable left-padding cache.
- Audio encoder: causal GQA/MHA transformer with RoPE, sliding-window mask, and its own KV cache.
- Projector: reshape every 4 audio-encoder frames into one text-token feature, then `Linear(5120 -> 3072) -> GELU -> Linear(3072 -> 3072)`.
- Prefix/decode coupling: audio embeddings are added to text embeddings, not scattered into placeholder positions in the main forward path.
- Text decoder: 26-layer production decoder with GQA, RoPE, sliding-window causal attention, SwiGLU MLP, RMSNorm, Ada RMS delay conditioning, tied embedding/LM head.

Independently validatable pieces: feature extractor, conv streaming cache, audio encoder cache parity, projector shape mapping, additive embedding coupling, text decoder prefill/decode, and generation length/stream-exhaustion behavior.

## 3. Important config dimensions

Production dimensions from official `mistralai/Voxtral-Mini-4B-Realtime-2602` `config.json`.

| Field | Audio encoder | Text decoder / top-level |
| --- | ---: | ---: |
| dtype | bfloat16 | bfloat16 |
| hidden_size | 1280 | 3072 |
| layers | 32 | 26 |
| attention heads | 32 | 32 |
| KV heads | 32 | 8 |
| head_dim | 64 | 128 |
| intermediate_size | 5120 | 9216 |
| vocab_size | 131072 | 131072 |
| max_position_embeddings | 1500 | 131072 |
| rope | default, theta 1e6 | default, theta 1e6 |
| sliding_window | 750 | 8192 |
| norm eps | 1e-5 | 1e-5 |
| activation | SwiGLU MLP, GELU conv/projector | SwiGLU MLP, GELU Ada RMS MLP |
| cache support | audio encoder KV + conv padding cache | text decoder KV |
| audio_length_per_tok | n/a | 8 |
| default_num_delay_tokens | n/a | 6 |
| downsample_factor | n/a | 4 |

Representative sweep:

| Model/config | Native HF schema? | Audio dims | Text dims | Operator-significant notes |
| --- | --- | --- | --- | --- |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | yes | 32L, H=1280, 32 KV heads, SW=750 | 26L, H=3072, GQA 32/8, SW=8192 | Production BF16, 4.43B params from HF safetensors metadata |
| `RedHatAI/Voxtral-Mini-4B-Realtime-2602` | yes | same as official | same as official | Mirror; useful availability fallback, not new architecture |
| `onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration` | yes | 2L, H=64, heads=4, KV=2 | 2L, H=64, heads=4, KV=2 | Debug parser/export config; dtype float32; not quality representative |
| `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | no | same logical production dims in MLX schema | same logical production dims in MLX schema | Quantized MLX format, 4-bit affine group size 64; route to separate loader audit |

## 3a. Family variation traps

- `hidden_size != num_heads * head_dim` for the audio encoder: H=1280 but Q width is `32 * 64 = 2048`; output projection maps `2048 -> 1280`.
- Audio attention has Q/V/O biases but K has no bias. Text attention has no Q/K/V/O biases.
- Audio encoder production config is MHA (`KV heads = heads`), while text is GQA (`KV heads = 8`, repeat factor 4).
- Both audio and text use sliding-window causal masks; dense full causal attention is not the production path unless `sliding_window=None`.
- Audio encoder is causal despite being described as Whisper-like. Do not import offline Whisper encoder assumptions.
- Realtime state is three-part: text KV cache, audio encoder KV cache, and causal Conv1d padding cache.
- `cache_implementation` for the audio encoder admits only `static` and `offloaded_static`; other generation cache modes are rejected by source.
- The official config omits `audio_token_id`, while `get_placeholder_mask` references it. The main source path uses additive audio/text embeddings; do not admit generic masked scatter for first integration.
- Processor chunking changes STFT centering: first chunk uses `center=True`, later streaming chunks use `center=False`.
- MLX/GGUF/ONNX mirrors are useful deployment evidence but may have non-native config schemas or extra quantized ABI. Native Transformers admission should require the nested HF `audio_config`/`text_config` schema unless a separate loader handles the mirror.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab=131072, dim=3072)` and tied `Linear(3072 -> 131072)`.
- `Conv1d` over `[B, C, T]`, causal left pad/cache, kernel 3, strides 1 and 2.
- `permute [B,C,T] -> [B,T,C]`, reshape/group `[B,T_audio,1280] -> [B,T_text,5120]` with `downsample_factor=4`.
- Elementwise add for audio/text embedding coupling, residual adds, multiply by `1 + condition`.
- `arange`, position-id construction, cache-length-derived offsets, dynamic slicing of `encoder_inputs_embeds`.

Neural primitives:

- RMSNorm with fp32 variance and dtype restore.
- Linear/GEMM families:
  - audio Q `1280 -> 2048` bias, K `1280 -> 2048` no bias, V `1280 -> 2048` bias, O `2048 -> 1280` bias.
  - audio MLP `1280 -> 5120` gate/up no bias, `5120 -> 1280` down bias.
  - text Q `3072 -> 4096`, K/V `3072 -> 1024`, O `4096 -> 3072`, all no bias.
  - text MLP `3072 -> 9216` gate/up/down no bias.
  - Ada RMS conditioning MLP `3072 -> 32 -> 3072`, GELU, no bias.
  - projector `5120 -> 3072 -> 3072`, GELU, no bias.
- GELU and SiLU/SwiGLU activation multiply.

Attention primitives:

- Causal sliding-window self-attention for audio and text.
- RoPE applied before cache update.
- GQA repeat for text KV heads.
- SDPA/Flash/Flex compatible dispatch in source, with eager matmul/softmax fallback.

Generation/cache ops:

- Text decoder KV cache per layer: `[B, 8, T_text, 128]` before repeat in production.
- Audio encoder KV cache per layer: `[B, 32, T_audio_encoded, 64]`.
- Conv padding cache:
  - `conv1`: `[B, 128, 2]` left context.
  - `conv2`: `[B, 1280, 1]` left context because kernel 3 stride 2 yields left pad 1.
- Static-address mutable cache updates with copy semantics.
- Generator-backed `input_features` stream and stream-exhaustion stop condition.

Preprocessing-coupled ops:

- Mono waveform conversion, padding/truncation, optional waveform attention mask.
- STFT with Hann window, `n_fft=400`, `hop_length=160`, center flag per chunk.
- Power magnitude, mel filterbank, log10 clamp, global max clamp, affine log scaling.
- `mistral-common` transcription request encoding and delay/right-pad token policy.

Quantized/packed weight metadata:

- Native official config is BF16 safetensors. Quantized MLX/GGUF mirrors should be deferred to separate loader/provider contracts; do not infer native HF quantization from those configs.

## 5. Layer/block breakdown

Audio embedder:

```text
features [B,128,T]
  -> causal Conv1d(128 -> 1280, k=3, s=1, bias) + GELU
  -> causal Conv1d(1280 -> 1280, k=3, s=2, bias) + GELU
  -> permute to [B,T',1280]
```

Audio encoder block, repeated 32 times:

```text
x = x + Attention(RMSNorm(x), sliding causal mask, RoPE, audio_cache)
x = x + Linear_down(SiLU(Linear_gate(RMSNorm(x))) * Linear_up(RMSNorm(x)))
```

Audio attention projection widths:

```text
q: [B,T,1280] -> [B,32,T,64] logically, stored width 2048
k/v: [B,T,1280] -> [B,32,T,64]
o: [B,T,2048] -> [B,T,1280]
```

Audio-to-text projector:

```text
audio_states [B,T_audio,1280]
  -> reshape [B,T_audio/4,5120]
  -> Linear(5120 -> 3072, no bias)
  -> GELU
  -> Linear(3072 -> 3072, no bias)
```

Text decoder block, repeated 26 times:

```text
x = x + Attention(RMSNorm(x), sliding causal mask, RoPE, text_cache)
y = RMSNorm(x)
y = y * (1 + AdaRmsNorm(t_cond))
x = x + SwiGLU_MLP(y)
```

Text logits:

```text
hidden [B,T,3072] -> RMSNorm -> tied LM head [3072,131072]
```

## 6. Attention requirements

Both branches use causal self-attention, not cross-attention.

Audio attention:

- causal self-attention with sliding window 750 in production.
- MHA in production config: 32 Q heads, 32 KV heads, head dim 64.
- Q/K receive RoPE before cache update; cached keys are post-RoPE.
- Cache stores un-repeated K/V with shape `[B, 32, T, 64]`.
- Eager fallback does `QK^T * head_dim^-0.5`, adds mask, softmax in fp32, casts back, dropout only in training, then `AV`.

Text attention:

- causal self-attention with sliding window 8192 in production.
- GQA: 32 Q heads, 8 KV heads, head dim 128; repeat factor 4 occurs inside eager attention.
- Cache stores K/V before repeat with shape `[B, 8, T, 128]`.
- Prefill/decode must preserve cache position when slicing audio embeddings and when building RoPE position ids.

Packed/varlen support: source delegates optimized attention to Transformers attention interfaces. The family source itself does not expose a cu_seqlens ABI. DinoML can start with dense padded sliding-window masks, then add packed attention only behind explicit processor/runtime metadata.

## 7. Position encoding and custom math

RoPE is default theta 1e6 for both branches:

```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat(freqs, freqs, dim=-1)
q = q * cos(emb) + rotate_half(q) * sin(emb)
k = k * cos(emb) + rotate_half(k) * sin(emb)
```

Delay-token time conditioning:

```python
inv = exp(-log(theta) * arange(dim // 2) / (dim // 2))
t_cond = cat(cos(num_delay_tokens * inv), sin(num_delay_tokens * inv))
hidden = hidden * (1 + linear2(gelu(linear1(t_cond))))
```

RoPE cos/sin can be precomputed per branch/head_dim/theta for static windows, but position ids depend on cache length during streaming. `t_cond` is scalar per request/chunk and can be cached for a fixed `num_delay_tokens`.

## 8. Preprocessing and input packing

Waveform contract:

- input is mono 16 kHz; multi-channel tensors are averaged to mono by the feature extractor.
- defaults: `feature_size=128`, `n_fft=400`, `win_length=400`, `hop_length=160`, `global_log_mel_max=1.5`.
- output `input_features` shape is `[B, 128, T_mel]`, dtype float32 before model/device conversion.
- optional audio attention mask is sampled from the padded waveform mask at `win_length - 1 :: hop_length`.

Streaming chunk contract from processor:

- First chunk: `is_streaming=True`, `is_first_audio_chunk=True`; processor calls `mistral-common`, emits `input_ids`, `attention_mask`, `input_features`, and `num_delay_tokens`.
- Subsequent chunks: `is_first_audio_chunk=False`; processor emits only audio features plus `num_delay_tokens`; text encoding is skipped.
- First chunk STFT uses `center=True`; later chunks use `center=False`.
- Helper formulas:
  - first mel frames: `(num_delay_tokens + 1) * audio_length_per_tok`.
  - first samples: `(first_mel_frames - 1) * hop_length + win_length // 2`.
  - later samples per chunk: `audio_length_per_tok * hop_length + win_length`.
- With official defaults, later chunks are `8 * 160 + 400 = 1680` samples, about 105 ms at 16 kHz.

Embedding stitch:

- Main path is additive: `inputs_embeds += projected_audio_embeds`.
- There is a placeholder-mask helper, but it depends on missing `audio_token_id` and is not called by the main realtime forward path. First DinoML integration should reject arbitrary placeholder scatter for this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: grouped audio frames -> projector GEMM

Source pattern:

```text
audio_hidden [B,T,1280] -> reshape [B,T/4,5120] -> Linear(5120 -> 3072)
```

Replacement: view/group contiguous audio time frames then GEMM. Preconditions: `T % downsample_factor == 0`, hidden size matches audio config, source layout `[B,T,H]` contiguous after encoder. Failure cases: streaming chunks that do not produce enough frames for a full group need buffering or rejection.

### Rewrite: causal Conv1d small-kernel cache

Source pattern: left-pad/cache then Conv1d k=3 over `[B,C,T]`.

Replacement: specialized causal 1D conv kernel or im2col+GEMM for larger batches. Preconditions: kernel 3, dilation 1, stride 1 or 2, contiguous NCT, cache tensors available with exact left-pad width. Failure cases: non-first streaming chunk with missing cache, changed batch size, or mismatched dtype/device.

### Rewrite: Ada delay conditioning hoist

Source pattern: `hidden * (1 + Linear2(GELU(Linear1(t_cond))))` repeated every text layer.

Replacement: compute per-layer scale vector once per request/chunk, then fuse multiply into MLP input epilogue. Preconditions: `num_delay_tokens` scalar and unchanged for the decode step; no training gradients. Failure cases: per-token or per-batch heterogeneous delay tokens.

### Rewrite: last-token logits

Source pattern: `logits_to_keep` slices hidden states before LM head.

Replacement: for decode, run LM head only on final token. Preconditions: generation only needs next-token logits. Failure cases: training loss, full logits requested, or custom logits processors needing full sequence logits.

### Layout guard

Source audio frontend is NCT Conv1d, then BTH transformer. Do not apply channel-last/NHWC-style rewrites globally. A layout pass may optimize the controlled conv frontend, but it must preserve the immediate `permute(0,2,1)` boundary and transformer axis semantics.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for audio/text BF16.
- Sliding-window GQA attention with RoPE and KV cache, especially text decode.
- Conv1d causal streaming frontend with explicit padding cache.
- SwiGLU MLP GEMM fusion for text and audio.
- Last-token LM head for decode.

Medium priority:

- Projector `Linear + GELU + Linear`.
- Ada RMS delay scale precompute and multiply fusion.
- Audio encoder sliding-window attention prefill over chunk-sized inputs.
- STFT/mel preprocessing GPU path only if CPU preprocessing becomes bottleneck.

Lower priority:

- Offloaded static audio encoder cache parity.
- Quantized MLX/GGUF mirror loading.
- Full placeholder-mask/scatter path, unless a future native config actually uses `audio_token_id`.

## 11. Runtime staging plan

Stage 1: parse native nested config and processor config; reject non-native MLX/GGUF schemas unless routed to a separate loader.

Stage 2: implement text-only `VoxtralRealtimeTextForCausalLM` parity with RMSNorm, RoPE, GQA sliding-window attention, SwiGLU, Ada delay conditioning, tied LM head, and `logits_to_keep`.

Stage 3: implement audio feature tensor path from precomputed `input_features`; start with non-streaming cache disabled, then add causal conv padding cache.

Stage 4: add audio encoder KV cache and projector parity; validate audio embeddings independently.

Stage 5: integrate additive audio/text embedding coupling and generation length rules for offline audio.

Stage 6: add generator/chunk streaming session ABI: input chunk feed, text KV cache, audio KV cache, conv padding cache, stream exhaustion.

Stage 7: optimize attention/MLP/conv kernels and optionally add packed/quantized deployment formats.

## 12. Parity and validation plan

- Feature extractor parity: fixed sine/noise waveforms, first chunk center true and later chunk center false; compare mel features to Transformers within fp32 tolerance.
- Causal Conv1d cache parity: split one waveform feature tensor into chunks and compare cached conv outputs against one-shot causal-pad output.
- Single audio layer parity: random `[B,T,1280]`, RoPE positions, sliding mask, cache/no-cache paths.
- Single text layer parity: random `[B,T,3072]`, GQA cache update, `t_cond` conditioning.
- Projector parity: random audio hidden states with `T % 4 == 0`.
- Prefill logits parity: official/tiny config weights when available, no sampling.
- Decode parity: one-token stepping with text KV and audio encoder cache.
- Streaming parity: generator of chunks stops generation when exhausted and preserves all three caches.
- Recommended tolerances: fp32 `1e-4`/`1e-5`; BF16 use source-like BF16 matmul tolerances, roughly `1e-2` absolute on logits before sampling-sensitive checks.

## 13. Performance probes

- CPU feature extraction ms per second of audio, first chunk versus later chunks.
- Audio frontend Conv1d throughput per chunk and cache update overhead.
- Audio encoder chunk latency with sliding window 750 and KV cache enabled.
- Projector latency and memory traffic for grouped frames.
- Text prefill latency for first chunk prompt/audio token sequence.
- Decode tokens/sec with text KV cache and last-token logits.
- End-to-end streaming latency at official later-chunk size, including processor and generation controller.
- Cache memory by batch: text KV, audio KV, conv padding cache.
- Attention backend comparison: dense sliding mask versus optimized sliding-window attention.
- Quantized mirror probe, separately: MLX/GGUF dequant/load time and steady-state GEMM provider cost.

## 14. Skip/defer list

- Training, labels/loss, dropout, gradient checkpointing.
- Beam search and multi-GPU tensor parallel beyond preserving weight names/plans.
- `offloaded_static` audio cache until static cache parity is stable.
- General masked scatter / placeholder-token insertion.
- Non-native MLX/GGUF quantized configs and external vLLM runtime behavior.
- GPU STFT/mel extraction unless CPU preprocessing is proven bottleneck.
- Full `mistral-common` tokenizer implementation inside DinoML; first integration can consume processor-produced tensors.

## 15. Final implementation checklist

- [ ] Parse `VoxtralRealtimeConfig` with nested `audio_config` and `text_config`.
- [ ] Parse `processor_config.json` and admit only matching audio feature parameters.
- [ ] Load tied embeddings / LM head without duplicating logical weight identity.
- [ ] Implement BF16 RMSNorm.
- [ ] Implement default RoPE theta 1e6 for both branches.
- [ ] Implement text GQA sliding-window attention with KV cache.
- [ ] Implement audio causal sliding-window attention with encoder KV cache.
- [ ] Implement causal Conv1d k=3 stride 1/2 with explicit padding cache ABI.
- [ ] Implement audio frame grouping and projector.
- [ ] Implement additive audio/text embedding coupling.
- [ ] Implement `num_delay_tokens` sinusoidal conditioning and Ada RMS multiply.
- [ ] Implement generation cache/session ABI carrying text KV, audio KV, and conv padding cache.
- [ ] Implement offline generation length clamp from audio feature length.
- [ ] Implement streaming chunk feed and stream-exhaustion stop condition.
- [ ] Reject missing/unsupported placeholder `audio_token_id` scatter path for first integration.
- [ ] Add feature extractor, block, prefill, decode, and streaming parity tests.
- [ ] Benchmark preprocessing, audio encoder, text decode, and cache memory separately.
