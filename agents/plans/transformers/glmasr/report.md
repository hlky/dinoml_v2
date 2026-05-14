# Transformers Audit: glmasr

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: zai-org/GLM-ASR-Nano-2512
Config source: https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/main/config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/glmasr/configuration_glmasr.py
  X:/H/transformers/src/transformers/models/glmasr/modeling_glmasr.py
  X:/H/transformers/src/transformers/models/glmasr/modular_glmasr.py
  X:/H/transformers/src/transformers/models/glmasr/processing_glmasr.py
  X:/H/transformers/src/transformers/models/glmasr/convert_glmasr_weights_to_hf.py
  X:/H/transformers/docs/source/en/model_doc/glmasr.md
  X:/H/transformers/src/transformers/models/llama/modeling_llama.py
  X:/H/transformers/src/transformers/models/llama/configuration_llama.py
  X:/H/transformers/src/transformers/models/whisper/feature_extraction_whisper.py
  X:/H/transformers/src/transformers/models/audioflamingo3/{modeling,processing}_audioflamingo3.py
Any missing files or assumptions: only one current official native checkpoint was found; older Hub revision used remote-code-style config fields and is treated as an admission trap, not the native source basis.
```

`modeling_glmasr.py` and `processing_glmasr.py` are generated from
`modular_glmasr.py`; the generated files are the runtime files inspected for the
current in-library implementation. Hub metadata for the official model reports
public access, MIT license tag, and BF16 safetensors with 2,257,843,200
parameters. Snapshot notes are in `_sources/`.

## 2. High-level architecture

Primary DinoML target: ASR-oriented conditional generation.

```text
raw mono audio + prompt
-> processor splits audio into 30s windows and computes Whisper log-mel features
-> GLM-ASR audio encoder
-> 4-frame merge + MLP projector
-> replace expanded <|pad|> audio placeholder token embeddings
-> Llama causal LM prefill/decode
-> logits/sampling/decoded transcript
```

Stage boundaries:

- CPU/data pipeline: audio loading/resampling, 30s windowing, Whisper log-mel feature extraction, chat template, placeholder expansion, tokenization.
- GPU/runtime encoder: Conv1d audio stem, 32 noncausal Transformer encoder blocks, final LayerNorm.
- Independently cacheable prefix work: projected audio embeddings and the full multimodal prefill KV cache can be cached for decode.
- Decode: Llama-only causal generation after the first iteration; `prepare_inputs_for_generation` forwards `input_features` only on the first iteration or when cache is disabled.

## 3. Important config dimensions

| Field | Current official value | Source |
| --- | ---: | --- |
| dtype | `bfloat16` | `config.json` |
| audio input mel bins | 128 | `audio_config` / processor |
| audio max feature frames/window | 3000 | processor config |
| audio encoder hidden | 1280 | `audio_config` |
| audio encoder layers | 32 | `audio_config` |
| audio attention heads / KV heads | 20 / 20 | `audio_config` |
| audio head dim | 64 | `audio_config` |
| audio partial rotary factor | 0.5 | `audio_config.rope_parameters` |
| audio intermediate | 5120 | `audio_config` |
| projector | 5120 -> 4096 -> 2048 | source + config |
| text LM type | Llama causal LM | `text_config.model_type` |
| text hidden | 2048 | `text_config` |
| text layers | 28 | `text_config` |
| text attention heads / KV heads | 16 / 4 | `text_config` |
| text head dim | 128 | `text_config` |
| text intermediate | 6144 | `text_config` |
| text RoPE theta/type | 10000 / default | `text_config.rope_parameters` |
| text max positions | 8192 | `text_config` |
| vocab size | 59264 | `config.json` |
| audio token id | 59260 | `config.json` |
| EOS ids | `[59246, 59253, 59255]` | `generation_config.json` / `text_config` |
| cache support | text decoder `use_cache=true`; audio encoder no KV cache | source + config |

Representative sweep:

| Checkpoint/config | Access | Operator-significant variation |
| --- | --- | --- |
| `zai-org/GLM-ASR-Nano-2512` main | public | Current native `audio_config` + `text_config`, BF16, Llama text decoder, Whisper feature extractor processor. |
| `eustlb/GLM-ASR-Nano-2512` main | public mirror | Same operator dimensions; historical `architectures` spelling differs and should not drive class selection. |
| `zai-org/GLM-ASR-Nano-2512` revision `fdc39709...` | public historical | Remote-code-style `lm_config`/`whisper_config`, `merge_factor`, `adapter_type`, `use_rope`, and `attn_implementation`; route through compatibility conversion or reject for native audit unless normalized. |

## 3a. Family variation traps

- Only the current native config shape is audited. Historical configs advertise fields the native `GlmAsrConfig` does not read directly (`lm_config`, `whisper_config`, `merge_factor`, `max_whisper_length`, `mlp_adapter_act`).
- `hidden_size != audio projector input width`: the projector consumes `audio_config.intermediate_size=5120` because source reshapes groups of four 1280-wide encoder frames.
- Audio encoder attention is noncausal and source passes `attention_mask=None`; the mask is used for valid projected rows after encoding, not for attention.
- Audio RoPE is partial: only 32 of 64 head dimensions rotate. Text Llama RoPE rotates all 128 head dims.
- Audio attention projection biases are asymmetric: Q/V/O have bias; K has no bias.
- Text decoder uses GQA (`num_key_value_heads=4` for 16 query heads). Cache storage is KV-head-shaped, not query-head-expanded.
- Processor default `max_audio_len=655` floors to `max_windows=int(655 // 30)=21`, i.e. 630 seconds of 30s chunks, despite doc text saying 655 seconds.
- Placeholder token is `<|pad|>`, but tokenizer `pad_token` is `<|endoftext|>` in tokenizer config. Do not infer audio placeholders from normal padding.
- `masked_scatter` is broad PyTorch syntax; the processor creates a stricter row-copy pattern by expanding one audio placeholder into exactly the projected audio-token count.
- Source layout is audio `[B, mel, frames]` for Conv1d and text `[B, seq, hidden]`; no NHWC/channel-last translation is relevant for first integration. Axis-sensitive ops: Conv1d time axis, transpose `(1,2)`, attention softmax `dim=-1`, LayerNorm/RMSNorm last dim, token placeholder mask over sequence axis.

## 4. Operator coverage checklist

Tensor/layout ops:

- Reshape/view: audio `[flat_windows, 1500, 1280] -> [original_batch, -1, 5120]`; attention QKV `[B,T,H] -> [B,T,n_heads,head_dim] -> [B,n_heads,T,head_dim]`; Llama logits slicing for `logits_to_keep`.
- Transpose/contiguous for attention and Conv1d output.
- Boolean mask creation, sum over mask, arange compare, row filtering.
- Guarded indexed row copy to replace audio placeholders in text embeddings.

Neural primitives:

- Conv1d `128 -> 1280`, kernel 3, stride 1, padding 1.
- Conv1d `1280 -> 1280`, kernel 3, stride 2, padding 1.
- LayerNorm over 1280 for audio encoder.
- RMSNorm over 2048 for Llama decoder.
- Linear shapes:
  - Audio attention Q `1280 -> 1280` bias, K `1280 -> 1280` no bias, V `1280 -> 1280` bias, O `1280 -> 1280` bias.
  - Audio MLP `1280 -> 5120 -> 1280` with GELU.
  - Projector `5120 -> 4096 -> 2048` with GELU.
  - Text attention Q `2048 -> 2048`, K/V `2048 -> 512`, O `2048 -> 2048`, no bias in current config.
  - Text SwiGLU MLP gate/up `2048 -> 6144`, down `6144 -> 2048`, no bias.
  - LM head `2048 -> 59264`, tied to token embeddings by Llama source contract.
- GELU, SiLU, elementwise multiply, residual add, softmax fp32 accumulation.

Attention primitives:

- Audio encoder dense noncausal self-attention, MHA, no attention mask, partial RoPE.
- Text decoder causal self-attention, GQA 16Q/4KV, causal mask, optional SDPA/FlashAttention-compatible backend, KV cache.

Position/cache ops:

- RoPE cos/sin generation in fp32, cast to model dtype.
- Audio position ids are `arange(audio_seq_len)[None, :]`.
- Llama position ids default to `arange(current_len) + past_seen_tokens`.
- Llama `Cache.update` per decoder layer; cache reorder/reset semantics should follow Transformers cache utilities for generation.

Preprocessing-coupled ops:

- Whisper log-mel: STFT `n_fft=400`, hop 160, Hann window, power 2, 128 Slaney mel bins, log10 clamp, dynamic range clamp to max - 8, affine `(x + 4) / 4`.
- 30s audio chunking and padding/truncation.
- Attention-mask rescale from sample mask to feature-frame mask by `::hop_length`.
- Prompt/chat template and audio placeholder expansion.

## 5. Layer/block breakdown

Audio preprocessing:

```text
raw mono waveform at 16 kHz
-> split into flat 30s windows, max 21 windows/sample by default
-> pad/truncate each window to 480000 samples
-> Whisper log-mel features [flat_windows, 128, 3000]
-> input_features_mask [flat_windows, 3000]
```

Audio encoder:

```text
x: [W, 128, 3000]
x = GELU(Conv1d(128, 1280, k=3, pad=1, stride=1)(x))   # [W,1280,3000]
x = GELU(Conv1d(1280,1280, k=3, pad=1, stride=2)(x))   # [W,1280,1500]
x = transpose(x, 1, 2)                                 # [W,1500,1280]
pos = RoPE(arange(1500), rotary_dim=32)
repeat 32:
  y = LayerNorm(x)
  q = Linear(1280,1280,bias=True)(y).view(W,T,20,64).transpose(1,2)
  k = Linear(1280,1280,bias=False)(y).view(W,T,20,64).transpose(1,2)
  v = Linear(1280,1280,bias=True)(y).view(W,T,20,64).transpose(1,2)
  q,k = partial_RoPE(q,k,pos, first 32 dims)
  y = Attention(q,k,v, mask=None, scale=1/sqrt(64))
  x = x + Linear(1280,1280,bias=True)(y)
  y = LayerNorm(x)
  y = Linear(5120,1280)(GELU(Linear(1280,5120)(y)))
  x = x + y
x = LayerNorm(x)
```

Project/stitch:

```text
x = x.reshape(original_batch, -1, 5120)
audio_embeds = Linear(4096,2048)(GELU(Linear(5120,4096)(x)))
post_lengths = conv_length_formula(input_features_mask.sum(-1), conv1, conv2)
post_lengths = (post_lengths - 4) // 4 + 1
audio_features = audio_embeds[arange(max_len) < post_lengths[:, None]]
inputs_embeds = token_embedding(input_ids)
inputs_embeds[audio_token_mask] = audio_features  # row-major guarded copy
```

Text decoder block, repeated 28 times:

```text
x = x + CausalGQAAttention(RMSNorm(x), RoPE, cache)
x = x + down_proj(SiLU(gate_proj(RMSNorm(x))) * up_proj(RMSNorm(x)))
```

## 6. Attention requirements

Audio encoder attention:

- Noncausal self-attention only.
- MHA, 20 heads, 20 KV heads, head dim 64.
- Q/K/V sequence lengths are equal after conv stride: max 1500 frames per 30s window.
- Source passes no mask to the attention backend. Padded/silent frames can attend; valid output rows are selected after projection.
- RoPE is applied before attention to first 32 dimensions of Q/K.
- No KV cache and no generation state.

Text decoder attention:

- Causal self-attention via Llama.
- GQA: 16 query heads, 4 KV heads, 4 query heads per KV head.
- Q width 2048; K/V width 512 before repeat; head dim 128.
- Cache stores per-layer K/V after RoPE and before repeat, logically `[B, 4, S, 128]`.
- Attention backends can be eager, SDPA, or FlashAttention through Transformers dispatch; first DinoML parity can use dense causal attention plus cache, then replace with optimized GQA attention.
- Cross-attention is absent. The audio condition is injected by embedding replacement before Llama prefill.

## 7. Position encoding and custom math

Audio partial RoPE:

```python
def glmasr_audio_rope(q, k, position_ids, theta=10000.0):
    # q/k: [B, 20, T, 64]; only first 32 dims rotate.
    inv = 1.0 / (theta ** (arange(0, 32, 2).float() / 32))
    freqs = outer(position_ids, inv)
    cos = cat([freqs, freqs], dim=-1).cos()
    sin = cat([freqs, freqs], dim=-1).sin()
    q_rot, q_pass = q[..., :32], q[..., 32:]
    k_rot, k_pass = k[..., :32], k[..., 32:]
    return cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], -1), \
           cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], -1)
```

Conversion-specific gap: `convert_glmasr_weights_to_hf.py` permutes original
checkpoint Q/K weights because the source implementation uses non-interleaved
partial RoPE order. DinoML loaders should consume already-converted HF weights,
or explicitly apply the same permutation when admitting historical raw weights.

Llama text RoPE is standard full-head default RoPE with theta 10000, generated
from `position_ids` and cache offset.

## 8. Preprocessing and input packing

Waveform contract:

- Mono audio only; WhisperFeatureExtractor rejects stereo-like rank > 2 batched arrays.
- Sampling rate must be 16 kHz when provided.
- Processor accepts paths/URLs through chat template loading, NumPy arrays, Torch tensors, or lists.
- Feature extraction is normally CPU/data-pipeline work. WhisperFeatureExtractor has a Torch STFT path that can run on CUDA, but the GLM-ASR processor defaults to normal feature-extractor invocation, not model graph execution.

Feature tensor contract:

- `input_features`: `[flat_windows, 128, 3000]`, float32 from processor, cast to model dtype by caller examples.
- `input_features_mask`: `[flat_windows, 3000]`, values 0/1.
- `input_ids` and `attention_mask`: tokenizer outputs with left padding.

Chunking/reassembly:

- Each user audio sample becomes `n_win = ceil(num_samples / 480000)`, clamped to at most 21 by default.
- Windows are flattened across batch. The model reconstructs original sample grouping only through `input_features.shape[0]` and the post-conv valid mask calculation.
- Model output is normal text generation; there is no CTC path or timestamp reassembly in the model source.

Placeholder ABI:

- Chat template inserts one `<|pad|>` between audio begin/end markers.
- Processor replaces that one token string with `<|pad|>` repeated by `_get_audio_token_length`.
- Model verifies the count and fills embeddings in row-major mask order.
- DinoML should reject caller-provided `input_ids` with mismatched audio-token count, ambiguous audio tokens outside the controlled template, or missing `input_features_mask`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: audio placeholder masked_scatter -> guarded row copy

Source pattern:

```text
inputs_embeds = embedding(input_ids)
mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, audio_features)
```

Replacement:

```text
positions = nonzero(input_ids == audio_token_id) in row-major order
assert len(positions) == audio_features.shape[0]
copy each audio_features[row] into inputs_embeds[positions[row]]
```

Preconditions: processor-controlled template, exact count check, hidden width 2048, no mixed arbitrary boolean scatter uses. Failure cases: caller supplies `inputs_embeds` without `input_ids`, multiple uncontrolled `<|pad|>` audio tokens, or count mismatch.

Parity test sketch: compare hidden embeddings before Llama for random audio features and token IDs with one and multiple audio chunks per batch row.

### Rewrite: 4-frame merge reshape -> strided grouped projection

Source pattern:

```text
[B, 1500 * windows, 1280] -> reshape [B, 375 * windows, 5120] -> Linear(5120,4096)
```

Replacement: treat every 4 consecutive encoder frames as one concatenated row and feed a GEMM. Preconditions: audio sequence after conv2 is divisible by 4 for all admitted windows; current fixed 3000-frame windows produce 1500 frames, divisible by 4 only after valid length formula is applied to rows selected from the reshaped max tensor. Failure cases: nonstandard feature length where reshape cannot group by 4.

### Rewrite: Conv1d stem -> GEMM/im2col or direct Conv1d provider

Conv1d kernels are overlapping (`kernel_size=3`, stride 1/2), so this is not a non-overlap linear patch rewrite. Use a real Conv1d lowering or im2col+GEMM. Layout guard: source semantic layout is `[B, C, T]`; any channel-last optimization must rewrite Conv1d, GELU, and transpose together.

### Rewrite: Llama GQA attention -> FlashAttention/GQA cache kernel

Preconditions: causal self-attention, no attention bias, RoPE applied before cache update, KV cache stored before repeat, mask compatible with left padding and generation. Failure cases: non-default attention backend requiring different mask semantics, `num_key_value_heads` not dividing query heads, or cache object not in static layout.

## 10. Kernel fusion candidates

Highest priority:

- Whisper log-mel preprocessing throughput, if DinoML owns preprocessing. It is often CPU-bound and includes STFT, mel GEMM, log/clamp/normalize.
- Audio Conv1d + GELU stem. It runs over every 30s window and is a fixed small audio layout.
- Audio encoder LayerNorm + QKV projections + partial RoPE + noncausal attention.
- Llama RMSNorm + GQA attention with KV cache for decode.
- Llama SwiGLU MLP GEMM fusion.
- Last-token-only LM head using `logits_to_keep` to avoid full-sequence vocab projection during decode.

Medium priority:

- Projector 4-frame merge + Linear/GELU/Linear.
- Placeholder row-copy kernel fused with token embedding materialization.
- Audio encoder MLP GELU fusion.
- Prefill whole-sequence Llama attention optimized separately from decode.

Lower priority:

- Training labels/loss.
- General boolean scatter.
- Dynamic RoPE variants beyond current default.
- GPU audio feature extraction unless preprocessing becomes the bottleneck.

## 11. Runtime staging plan

1. Parse current native config and reject historical remote-code configs unless normalized to `audio_config`/`text_config`.
2. Load tokenizer/processor metadata and implement CPU preprocessing parity for fixed 16 kHz mono audio.
3. Implement audio encoder stem and one encoder block parity with synthetic log-mel features.
4. Implement full audio encoder + projector and validate projected audio feature lengths/masks.
5. Implement guarded placeholder row-copy into Llama input embeddings.
6. Compose Llama prefill parity with the multimodal prefix, using dense causal attention first.
7. Add Llama decode KV cache with GQA cache layout `[layers][K,V][B,4,S,128]`.
8. Enable optimized attention/GEMM fusions and last-token LM head.
9. Add end-to-end ASR generation parity for short and multi-window audio.

Initially stubbable: audio file loading/URL fetch, chat templating beyond the transcription request, labels/loss, return attentions/hidden states, and pipeline postprocessing beyond token decode.

## 12. Parity and validation plan

- Whisper feature extraction: compare `input_features` and `input_features_mask` for 1s, 30s, 31s, and near-limit audio; fp32 tolerance around `1e-5` for Torch/NumPy feature path.
- Audio length math: assert token counts for full window (`3000 -> 1500 -> 375`) and partial windows, including batched multi-window samples.
- Audio encoder block: random tensors, fp32 then bf16 tolerances; validate partial RoPE only changes first 32 dims/head.
- Projector/stitch: compare flattened valid audio rows and embedding replacement for batches with different audio token counts.
- Llama text parity: reuse Llama audit tests for prefill logits and decode token parity with GQA cache.
- End-to-end: short single audio transcription, two-sample batch, and multi-window long audio. Compare generated token ids under greedy decode before decoded text.
- Recommended tolerances: fp32 `1e-4` for encoder/logits slices, bf16 `2e-2` to `5e-2` depending on attention backend and long sequence length.

## 13. Performance probes

- Processor throughput: waveform decode/resample, chunking, STFT/mel/log separately.
- Audio encoder throughput by number of 30s windows and batch size.
- Projector and placeholder copy overhead by total audio tokens.
- Llama prefill latency with long audio-token prefixes near 8192 text positions.
- Decode tokens/sec with batch size sweep and KV cache memory usage.
- FlashAttention/SDPA/eager attention comparison for audio encoder and text prefill.
- Conv1d provider comparison: direct Conv1d vs im2col+GEMM.
- LM head cost with `logits_to_keep=1` versus full sequence projection.
- BF16 versus fp16/fp32 accumulation probes for audio attention and MLP.

## 14. Skip/defer list

- Training, labels, and loss.
- Gradient checkpointing.
- Beam search and sampling beyond greedy first parity.
- General chat/multiturn prompt handling beyond controlled transcription request.
- General boolean `masked_scatter`.
- Historical remote-code configs unless normalized.
- Quantized weights and packed storage formats; current Hub weights are BF16 safetensors.
- Multi-GPU tensor parallel plans; GLM-ASR disables `_tp_plan`/`_pp_plan` at wrapper level.
- Output attentions/hidden states capture.
- Timestamp extraction or CTC-style decoding; not present in the source path.

## 15. Final implementation checklist

- [ ] Parse native `GlmAsrConfig` and nested Llama/audio configs.
- [ ] Reject or normalize historical `lm_config`/`whisper_config` configs.
- [ ] Load BF16 safetensors with Llama tied embedding/LM-head alias preserved.
- [ ] Implement Whisper log-mel preprocessing contract or define CPU pipeline boundary.
- [ ] Implement 30s chunking, masks, and audio-token expansion.
- [ ] Implement Conv1d audio stem.
- [ ] Implement audio LayerNorm, partial RoPE, noncausal MHA, and GELU MLP.
- [ ] Implement 4-frame audio merge and projector.
- [ ] Implement guarded placeholder row copy.
- [ ] Compose Llama causal prefill with RoPE and GQA.
- [ ] Implement Llama KV cache decode.
- [ ] Add parity tests for audio lengths, one block, full encoder/projector, prefill logits, and greedy decode.
- [ ] Benchmark preprocessing, encoder, prefill, decode, LM head, and KV memory.
