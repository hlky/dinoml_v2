# MusicFlamingo Transformers Audit

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nvidia/music-flamingo-2601-hf; nvidia/music-flamingo-think-2601-hf for representative variation
Config source: public HF config/processor/tokenizer/generation JSON plus local source defaults
Source files inspected:
  X:/H/transformers/src/transformers/models/musicflamingo/configuration_musicflamingo.py
  X:/H/transformers/src/transformers/models/musicflamingo/modeling_musicflamingo.py
  X:/H/transformers/src/transformers/models/musicflamingo/processing_musicflamingo.py
  X:/H/transformers/src/transformers/models/musicflamingo/modular_musicflamingo.py
  X:/H/transformers/src/transformers/models/musicflamingo/convert_musicflamingo_to_hf.py
  X:/H/transformers/docs/source/en/model_doc/musicflamingo.md
  X:/H/transformers/src/transformers/models/audioflamingo3/configuration_audioflamingo3.py
  X:/H/transformers/src/transformers/models/audioflamingo3/modeling_audioflamingo3.py
  X:/H/transformers/src/transformers/models/qwen2/configuration_qwen2.py
  X:/H/transformers/src/transformers/models/qwen2/modeling_qwen2.py
Any missing files or assumptions:
  No model imports, tests, weight loads, or model execution were run. Public HF repos are not gated, but the original
  source repository named by the converter is private. preprocessor_config.json is absent; processor_config.json
  embeds WhisperFeatureExtractor settings.
```

Representative URLs:

- [nvidia/music-flamingo-2601-hf](https://huggingface.co/nvidia/music-flamingo-2601-hf), HF API SHA `6b5be086d52f65a1e204cb0faf70bf54e2741ecd`
- [nvidia/music-flamingo-think-2601-hf](https://huggingface.co/nvidia/music-flamingo-think-2601-hf), HF API SHA `cbd8dec3066752db700a473d8869b8759e7437b8`
- [Transformers Music Flamingo docs](https://huggingface.co/docs/transformers/model_doc/musicflamingo)
- Source notes: `H:/dinoml_v2/agents/plans/transformers/musicflamingo/_sources/source_notes.md`

## 2. High-Level Architecture

MusicFlamingo is an audio/text-to-text multimodal causal LM. The neural path is:

```text
raw waveform -> WhisperFeatureExtractor log-mel windows -> AudioFlamingo3/Whisper-style encoder
  -> MusicFlamingo rotary time embedding -> MLP multimodal projector
  -> masked scatter into Qwen2 token embeddings -> Qwen2 prefill/decode -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: chat template construction, audio loading/decoding, 16 kHz waveform normalization, 30-second windowing, Whisper log-mel extraction, placeholder expansion, tokenization, padding.
- Audio encoder stage: independently cacheable for a fixed audio clip and feature mask. Input is `(num_windows_total, 128, mel_frames)`, output after pooling is `(num_windows_total, post_pool_frames, 1280)`.
- Music time embedding/projector stage: applies RoTE from token-derived timestamps, then projects 1280 audio hidden size to 3584 Qwen2 embedding width. This can be cached with the audio encoder result if the same prompt audio placement/window order is reused.
- Prefix construction: Qwen2 token embeddings are created, `<sound>` token positions are replaced with flattened projected audio embeddings, and `<|sound_bos|>`/`<|sound_eos|>` remain ordinary token embeddings.
- Prefill: Qwen2 causal LM consumes the stitched embedding sequence.
- Decode: Qwen2-only autoregressive decode. Audio features are only forwarded on the first generation iteration when cache is used.

This source is for music understanding/reasoning, not audio synthesis. There is no codec decoder, discrete music token stream, diffusion, or vocoder generation branch.

## 3. Important Config Dimensions

| Dimension | Public MusicFlamingo value | Source/default note |
|---|---:|---|
| dtype | `bfloat16` | From HF config/safetensors metadata |
| parameters | 8,267,215,360 BF16 | HF API safetensors metadata |
| audio model type | `audioflamingo3_encoder` | Composed via `AutoModel.from_config` |
| audio input bins | 128 | Whisper log-mel feature size |
| audio hidden size | 1280 | Also `MusicFlamingoConfig.head_dim` |
| audio layers | 32 | Bidirectional encoder |
| audio attention heads | 20 | Head dim 64 in audio encoder |
| audio FFN | 5120 | GELU MLP |
| audio max source positions | 1500 | After Conv1d stride-2 for a 30 s window |
| audio pooling | AvgPool1d kernel 2, stride 2 | Final post-pool frame count is 750 for full window |
| text model type | `qwen2` | Causal LM |
| text hidden size | 3584 | Projector output and token embedding width |
| text layers | 28 | All `full_attention` in public configs |
| text heads / KV heads | 28 / 4 | GQA, 7 Q heads per KV head |
| text head_dim | 128 inferred | `hidden_size / num_attention_heads` |
| text FFN | 18944 | SwiGLU Qwen2 MLP |
| vocab size | 151672 | Includes sound tokens |
| text max positions | 32768 | Public config and tokenizer |
| Qwen2 RoPE theta | 1,000,000 | In `text_config.rope_parameters` |
| Music RoTE theta | 1200 | `rope_parameters.rope_theta` |
| Music RoTE partial | 0.2 | Rotates 256 of 1280 audio hidden dims |
| projector | Linear 1280->3584, GELU, Linear 3584->3584 | Bias enabled |
| generation max_new_tokens | 2048 | `generation_config.json` |
| default text `use_cache` | false | Important for generation performance |

Representative checkpoint sweep:

| Model | Runtime-significant deltas | Tokenizer delta | Access/license notes |
|---|---|---|---|
| `nvidia/music-flamingo-2601-hf` | Baseline values above | `Qwen2Tokenizer`; `model_max_length=32768` | Public, not gated; license metadata `other` |
| `nvidia/music-flamingo-think-2601-hf` | No operator-significant config delta found | `TokenizersBackend` tokenizer class string | Public, not gated; same BF16 parameter count |

## 3a. Family Variation Traps

- Source defaults are unsafe for token IDs: `MusicFlamingoConfig.audio_token_id` defaults to `151669`, but public checkpoints use `<sound>` id `151667`. Read checkpoint config/tokenizer.
- `<|sound_bos|>` and `<|sound_eos|>` are inserted by the processor but are not replaced with audio embeddings. Only `<sound>` positions are replaced.
- Audio placeholder count must exactly equal the sum of post-pool audio frames. The model checks `n_audio_tokens == post_lengths.sum()`.
- `preprocessor_config.json` is absent. Feature extractor settings live inside `processor_config.json`.
- The converter source mentions a private original repo, `nvidia/music-flamingo-2601`; public HF assets do not provide the original component tree for recomputing pre-conversion expected outputs.
- The converter has stale-looking fields versus public config: `audio_rotary_dim=256` is passed but not declared in the inspected config class; public configs express this as `partial_rotary_factor=0.2`. Converter `model_max_length=8192` also differs from public `32768`.
- HF API `transformersInfo.auto_model` says `AutoModelForSeq2SeqLM`; source and config architecture are causal LM wrapper plus Qwen2.
- `use_cache=false` in public `text_config`. A Dinoml integration should not assume fast decode cache unless it explicitly admits `use_cache=True`.
- Qwen2 has GQA: cached KV heads are 4, not 28.
- Public configs have no sliding-window layers despite Qwen2 source support for them.
- No codec/token coupling beyond placeholder IDs exists. Do not introduce MusicGen-style codebook/token operators.
- Audio tower tensor layout changes are axis-sensitive: Conv1d uses `(B, C, T)`, transformer uses `(B, T, C)`, AvgPool1d returns to `(B, C, T)`.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- Embedding lookup for Qwen2 token embeddings and fixed audio positional embedding.
- `masked_scatter` or indexed copy from flat projected audio embeddings into text embeddings.
- Boolean masks, `torch.diff`, pad, `where`, `searchsorted`, `cumsum`, arange comparisons for timestamp and validity construction.
- Reshape/view, transpose, permute, contiguous, expand, broadcast, flatten, repeat_interleave, cat, stack.
- Left/right padding handling: processor uses common `padding_side="left"` but feature extractor pads audio right.

Neural network primitives:

- Conv1d `128 -> 1280`, kernel 3, padding 1, stride 1.
- Conv1d `1280 -> 1280`, kernel 3, padding 1, stride 2.
- GELU after both audio Conv1d layers and in projector/audio FFN.
- AvgPool1d kernel 2, stride 2 over time.
- LayerNorm on audio encoder and audio blocks.
- Qwen2 RMSNorm.
- Linear projections:
  - Audio attention Q `1280 -> 1280` bias, K `1280 -> 1280` no bias, V `1280 -> 1280` bias, O `1280 -> 1280` bias.
  - Audio FFN `1280 -> 5120 -> 1280`.
  - Projector `1280 -> 3584 -> 3584` with bias.
  - Qwen2 attention Q `3584 -> 3584`, K/V `3584 -> 512`, O `3584 -> 3584`.
  - Qwen2 MLP gate/up `3584 -> 18944`, down `18944 -> 3584`.
  - LM head `3584 -> 151672`, bias false.
- Dropout exists in source but public configs set dropout values to 0.0.

Attention primitives:

- Audio encoder dense bidirectional MHA, 20 heads, head dim 64, attention mask additive over `(B, 1, T, T)`.
- Qwen2 causal GQA, 28 query heads, 4 KV heads, head dim 128, repeat KV to 28 heads for eager attention.
- FlashAttention/SDPA interfaces are advertised by source for both encoder and decoder paths.

Position/rotary/custom math:

- Audio encoder learned absolute positions of length 1500 added after Conv1d stack.
- MusicFlamingo RoTE applied to audio hidden states before projector.
- Qwen2 default RoPE over causal LM q/k with theta 1e6.

Generation/cache ops:

- Qwen2 `DynamicCache` per layer when `use_cache=True`.
- `prepare_inputs_for_generation` must preserve `input_features` and `input_features_mask` only for first iteration or no-cache generation.
- `logits_to_keep` slicing in Qwen2 causal LM.

Preprocessing-coupled ops:

- 16 kHz Whisper feature extraction, 30-second chunking, 128 mel bins, hop 160, n_fft 400.
- Placeholder expansion based on feature attention mask sums and exact conv/pool length formulas.
- Optional label masking for training: audio placeholder/boundary tokens and pad tokens become `-100`.

Discrete codebook / tokenizer ops:

- None for audio content. Tokenizer still owns special text IDs: `<sound>`, `<|sound_bos|>`, `<|sound_eos|>`, `[BOS]`, `[PAD]`, `<|im_end|>`.

## 5. Layer/Block Breakdown

Processor, per sample:

```text
waveform length N samples
window_size = sampling_rate * chunk_length = 16000 * 30 = 480000
n_windows = ceil(N / window_size), clamped to max_audio_len / chunk_length = 40
flat_chunks -> WhisperFeatureExtractor -> input_features[num_windows, 128, 3000], input_features_mask[num_windows, 3000]
post_pool_len = (((valid_frames - 1) // 2 + 1) - 2) // 2 + 1
text "<sound>" -> "<|sound_bos|>" + "<sound>" * sum(post_pool_len) + "<|sound_eos|>"
```

Audio encoder:

```text
input_features: [W, 128, Tmel]
lengths_conv = (mask.sum(-1) - 1) // 2 + 1
x = GELU(Conv1d(128, 1280, k=3, pad=1, stride=1)(input_features))
x = GELU(Conv1d(1280, 1280, k=3, pad=1, stride=2)(x))
x = permute(x, [0, 2, 1])                         # [W, Tconv, 1280]
x = x + embed_positions.weight                     # source assumes max length-compatible Tconv
for 32 layers:
  residual = x
  x = LayerNorm(x)
  x = dense_bidirectional_attention_20h(x, mask)
  x = residual + x
  residual = x
  x = LayerNorm(x)
  x = Linear(1280, 5120)(x)
  x = GELU(x)
  x = Linear(5120, 1280)(x)
  x = residual + x
  if fp16: clamp to finite range
x = AvgPool1d(k=2, stride=2)(permute(x, [0, 2, 1]))
x = LayerNorm(permute(x, [0, 2, 1]))               # [W, Tpost, 1280]
```

MusicFlamingo audio feature path:

```text
hidden = audio_tower(input_features, input_features_mask).last_hidden_state
post_lengths = audio_tower._get_feat_extract_output_lengths(mask.sum(-1))[1]
timestamps = build_audio_timestamps(input_ids, post_lengths, max_post_length=hidden.shape[-2])
cos, sin = MusicFlamingoRotaryEmbedding(timestamps, seq_len=hidden.shape[-2])
hidden = apply_rotary_time_emb(hidden, cos, sin)   # float64 internal rotation
audio_embeds = Linear(1280, 3584)(hidden)
audio_embeds = GELU(audio_embeds)
audio_embeds = Linear(3584, 3584)(audio_embeds)
audio_embeds = audio_embeds[valid_mask]            # flattened [sum(post_lengths), 3584]
```

Qwen2 decoder block, repeated 28 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(3584, 3584, bias=True)(x).view(B, S, 28, 128).transpose(1, 2)
k = Linear(3584, 512,  bias=True)(x).view(B, S, 4, 128).transpose(1, 2)
v = Linear(3584, 512,  bias=True)(x).view(B, S, 4, 128).transpose(1, 2)
q, k = Qwen2RoPE(q, k, position_ids)
if cache: k, v = cache.update(k, v, layer_idx)
x = GQA_causal_attention(q, k, v, repeat_kv=7)
x = Linear(3584, 3584, bias=False)(x)
x = residual + x
residual = x
x = RMSNorm(x)
x = Linear(18944, 3584, bias=False)(Silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

Final LM:

```text
x = RMSNorm(x)
logits = Linear(3584, 151672, bias=False)(x[:, slice(-logits_to_keep, None), :])
```

The Qwen2 source declares tied weight keys for `lm_head.weight` and `model.embed_tokens.weight`, but public config has `tie_word_embeddings=false`. Treat them as separate loaded parameters unless the loader explicitly ties them.

## 6. Attention Requirements

Audio encoder attention:

- Noncausal bidirectional self-attention.
- MHA, 20 heads, key/query/value width 1280, head_dim 64.
- Attention mask comes from `create_bidirectional_mask` using post-conv audio mask. It is additive and shaped for attention backends.
- No KV cache. This branch can be cached as an encoder/projector result, not as autoregressive KV.
- Source-specific math: Q is multiplied by `head_dim**-0.5` before the backend call, then `scaling=1.0` is passed. Preserve this order for parity.
- FlashAttention and SDPA interfaces are source-supported, with eager fallback.

Text decoder attention:

- Causal self-attention.
- GQA: 28 Q heads, 4 KV heads, head_dim 128. Eager path repeats KV by factor 7 after cache update.
- Cache shape before repeat is per layer:

```text
keys:   [batch, 4, cached_seq, 128]
values: [batch, 4, cached_seq, 128]
```

- Cached keys are stored after Qwen2 RoPE is applied.
- Public configs set all 28 `layer_types` to `full_attention`, with `use_sliding_window=false` and `sliding_window=null`.
- Eager attention softmax upcasts to float32 and casts back to query dtype.
- DynamicCache is created by Qwen2 only if `use_cache=True` and no cache is provided. The public checkpoint config default is `use_cache=false`, so first integration can run no-cache prefill/decode while optimized integration should add explicit cache admission.

Multimodal interaction with attention:

- Audio is fused only by replacing token embeddings before Qwen2. There is no decoder cross-attention and no separate audio KV cache.
- Audio features must be present for prefill. During cached generation, `prepare_inputs_for_generation` forwards `input_features` only on the first iteration.

## 7. Position Encoding and Custom Math

Audio encoder learned positions:

- `embed_positions.weight` length is `max_source_positions=1500`.
- It is added directly to the post-conv sequence. For full 30-second windows, `Tconv=1500`, so shape matches. Shorter windows are padded by the feature extractor, so the source usually still runs fixed 3000 mel frames per chunk.

MusicFlamingo RoTE:

```python
def musicflamingo_audio_rope(hidden, timestamps, inv_freq, position_angles, seq_len, audio_frame_step):
    # Source-derived sketch, omitting device/dtype plumbing.
    window_starts = timestamps[:, 0]
    window_duration = audio_frame_step * 4 * seq_len
    window_positions = round(window_starts / window_duration) / 1200
    window_freqs = repeat_interleave(window_positions[:, None] * inv_freq[None, :], 2, dim=-1)
    time_freqs = position_angles[:seq_len]
    freqs = concat(broadcast(window_freqs[:, None, :], time_freqs[None, :, :]), dim=-1)
    angle = -timestamps * 2 * pi
    cos, sin = cos(freqs * angle[..., None]), sin(freqs * angle[..., None])
    return rotate_prefix_float64_then_cast_back(hidden, cos, sin)
```

Important details:

- Public configs use `audio_frame_step=0.01` seconds and final audio embedding step is `0.04` seconds after Conv1d plus avg-pool downsampling.
- `partial_rotary_factor=0.2` over `head_dim=1280` gives `dim=256` inverse-frequency dimensions, then MusicFlamingo concatenates window and time axes, so `rot_dim=512` in `apply_rotary_time_emb`.
- Rotation internally casts hidden states to float64 and returns original dtype.
- `position_angles` for the time axis can be precomputed from config. Timestamps depend on input_ids, audio segment starts/ends, window indices, and post_lengths.

Qwen2 RoPE:

- Default RoPE with theta `1_000_000.0`, head_dim 128.
- Position IDs default to `arange(seq_len) + past_seen_tokens`.
- Cos/sin are computed in float32 autocast-disabled context and cast to model dtype.

## 8. Preprocessing and Input Packing

CPU/data pipeline:

- The processor requires `return_tensors="pt"`.
- Audio and text are 1:1 at the processor call boundary.
- Waveforms are split into 30-second chunks at 16 kHz. Over 1200 seconds is truncated to 40 chunks.
- `WhisperFeatureExtractor` settings from `processor_config.json`: feature_size 128, sampling_rate 16000, hop_length 160, n_fft 400, chunk_length 30, n_samples 480000, nb_max_frames 3000, right audio padding, return attention mask true.
- Chat template emits one `<sound>` marker when any audio item is present in a user message, followed by concatenated text content.
- `_expand_audio_tokens` replaces that marker with boundary and repeated audio tokens. The repeated count is source-coupled to the audio tower length formulas.

GPU/runtime packing:

- `input_features` shape is `(total_windows, 128, 3000)` for padded full chunks.
- `input_features_mask` shape is `(total_windows, 3000)`.
- `input_ids` includes text, `<sound>` placeholders, and boundary tokens.
- The model embeds all input IDs, computes audio embeddings, builds a boolean mask where `input_ids == audio_token_id`, expands it over hidden width, checks exact element count, then uses `masked_scatter`.
- The scattered sequence length is unchanged. There is no packed varlen descriptor in the source; padding and attention masks carry sequence validity.

Codec/token coupling:

- No audio codec, vector quantizer, EnCodec, MusicGen codebook, vocoder, or audio token generation path appears in this family.
- The only coupling is between placeholder token IDs and continuous projected audio frame count.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: fixed Whisper Conv1d stack to layout-aware temporal convolution kernels

Source pattern:

```text
Conv1d(128, 1280, k=3, pad=1, stride=1) -> GELU
Conv1d(1280, 1280, k=3, pad=1, stride=2) -> GELU
```

Replacement:

```text
Native Conv1d or im2col/GEMM with explicit NCT layout -> GELU fusion
```

Preconditions:

- Input layout remains `(B, C, T)`.
- Padding is symmetric one element, dilation 1, groups 1.
- Conv2 stride is exactly 2.
- Dynamic T must keep positional embedding length and mask length consistent.

Failure cases:

- Do not translate to NTC without also rewriting Conv1d axes and subsequent permutes.
- Do not assume unpadded short chunks unless feature extractor settings change.

Parity test sketch:

- Random `(2, 128, 3000)` and masks with varied valid frame counts; compare after both conv/GELU layers and length formula.

### Rewrite: audio encoder attention backend

Source pattern:

```text
LayerNorm -> Q scaled before backend -> dense noncausal MHA -> output projection
```

Replacement:

```text
Fused LayerNorm plus FlashAttention/SDPA-compatible MHA, preserving pre-scaled Q and scaling=1.0
```

Preconditions:

- Dense bidirectional mask only.
- Dropout 0 for inference.
- Head dim 64, 20 heads.

Failure cases:

- Applying another `1/sqrt(d)` inside fused attention changes logits.
- Treating this as causal attention is wrong.

### Rewrite: projector MLP fusion

Source pattern:

```text
Linear(1280, 3584, bias) -> GELU -> Linear(3584, 3584, bias)
```

Replacement:

```text
GEMM+bias+GELU epilogue -> GEMM+bias
```

Preconditions:

- Activation is checkpoint `projector_hidden_act`.
- Bias flag must match config.
- Input is flattened valid audio rows or dense `[W, Tpost, 1280]`.

### Rewrite: multimodal scatter to indexed copy

Source pattern:

```text
special_audio_mask = input_ids == audio_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[..., None], flat_audio_embeds)
```

Replacement:

```text
positions = nonzero(input_ids == audio_token_id)
indexed_copy(inputs_embeds, positions, flat_audio_embeds)
```

Preconditions:

- Position order must match row-major flatten order used by `masked_scatter`.
- Placeholder count equals audio feature count.

Failure cases:

- Including BOS/EOS boundary tokens in replacement is wrong.
- Reordering batched positions breaks audio-window alignment.

### Rewrite: Qwen2 GQA decode kernels

Source pattern:

```text
RMSNorm -> Q/K/V linears -> RoPE -> KV cache update -> causal GQA -> O projection
```

Replacement:

```text
Fused RMSNorm/QKV where profitable, RoPE+cache update, paged GQA attention
```

Preconditions:

- KV heads 4, Q heads 28, head_dim 128.
- Cached keys are post-RoPE.
- No sliding-window layers for public configs.

## 10. Kernel Fusion Candidates

Highest priority:

- Whisper Conv1d+GELU front-end: fixed, large, repeated over up to 40 windows.
- Audio encoder LayerNorm + dense MHA with pre-scaled Q: 32 layers over up to 1500 conv frames per window.
- Audio projector GEMM+GELU and second GEMM: direct bridge to Qwen2 hidden size.
- Qwen2 RMSNorm + GQA attention with KV cache: required for usable decode speed once cache is admitted.
- Indexed audio embedding stitch: avoids expensive boolean masked scatter over long multimodal prompts.

Medium priority:

- Qwen2 SwiGLU MLP fusion.
- Qwen2 last-token-only logits via `logits_to_keep`.
- Audio RoTE custom kernel, especially the float64 rotation cost if parity allows a lower precision bounded mode later.
- Audio valid-mask compaction/gather after projector.

Lower priority:

- Training dropout/layerdrop paths.
- Sliding-window Qwen2 attention, since public configs disable it.
- Tensor parallel plans; source has Qwen2 plans but MusicFlamingo wrapper `_tp_plan=None`.

## 11. Runtime Staging Plan

Stage 1: config and processor metadata admission.

- Parse MusicFlamingo config, nested AudioFlamingo3 encoder config, nested Qwen2 config, processor_config, tokenizer special IDs, and generation config.
- Reject if token IDs or post-pool length formulas cannot be resolved.

Stage 2: audio preprocessing parity outside compiled graph.

- Implement or bridge Whisper log-mel extraction and windowing on CPU.
- Verify placeholder expansion and labels masking without running model weights.

Stage 3: audio tower block parity.

- Lower Conv1d/GELU, learned positions, bidirectional attention, audio FFN, avg pool, final LayerNorm.
- Validate one layer, then full 32-layer encoder against source.

Stage 4: MusicFlamingo RoTE and projector.

- Implement timestamp construction, RoTE, projector MLP, valid-row flattening.
- Validate audio embeddings for varied durations and batch/window counts.

Stage 5: multimodal prefill.

- Implement Qwen2 embedding stitch and no-cache causal LM prefill.
- Compare logits for audio+text and text-only prompts.

Stage 6: explicit Qwen2 decode cache.

- Admit `use_cache=True` as an optimized mode despite checkpoint default false.
- Define KV cache ABI `[layers=28][K/V: batch, 4, seq, 128]`, post-RoPE keys.

Stage 7: optimized backends.

- Add FlashAttention/SDPA-like audio encoder attention, paged GQA decode, and fused MLP/norm kernels.

Initial stubs:

- Training loss and label masking can be staged after inference.
- Beam search and sampling processors can use host generation controller initially.
- Original private source checkpoint parity can be skipped unless access is available.

## 12. Parity and Validation Plan

Processor tests:

- Waveforms of 1 sample, 1 second, exactly 30 seconds, 30 seconds plus 1 sample, and over 1200 seconds.
- Assert window counts, truncation, feature mask shape, and `<sound>` expansion count.
- Assert labels mask `<sound>`, `<|sound_bos|>`, `<|sound_eos|>`, and pad tokens to `-100`.

Custom op tests:

- Audio length formulas: `conv_len = (L - 1) // 2 + 1`, `post_len = (conv_len - 2) // 2 + 1`.
- Timestamp builder with multiple samples and uneven window counts; compare starts/ends/searchsorted behavior.
- MusicFlamingo RoTE on random BF16/FP32 hidden states; use FP64 internal reference.
- Masked scatter versus indexed copy preserving row-major placeholder order.

Neural parity:

- Single audio encoder layer with random weights, fp32 tolerance around `1e-5`.
- Full audio tower on short and full window masks.
- Projector output parity for dense and flattened valid rows.
- Qwen2 single decoder layer with no cache, then with cache.
- Full prefill logits for text-only prompt.
- Full prefill logits for audio+text prompt with synthetic tiny audio length if source allows mocked features, otherwise real processor features.
- Decode token parity for greedy generation with `use_cache=false` first, then `use_cache=true`.

Recommended tolerances:

- FP32 custom math: `rtol=1e-5`, `atol=1e-5`.
- BF16 end-to-end logits: start with `rtol=2e-2`, `atol=2e-2`, tighten per subgraph.
- RoTE should get a separate tolerance because source rotates in float64 then casts back.

## 13. Performance Probes

- CPU audio loading and WhisperFeatureExtractor throughput: seconds of audio per second of preprocessing.
- Window-count sweep: 1, 2, 8, 16, 40 windows.
- Audio tower throughput alone: batch windows/sec at 3000 mel frames.
- Audio RoTE plus projector throughput and memory bandwidth.
- Multimodal stitch time versus number of placeholders, comparing masked_scatter and indexed_copy.
- Qwen2 prefill tokens/sec for stitched sequence lengths including 750, 1500, 7500, and 30000 audio placeholders.
- Qwen2 decode tokens/sec with `use_cache=false` versus explicit cache.
- KV cache memory: `28 layers * 2 * batch * 4 KV heads * seq * 128 * dtype_bytes`.
- Last-token-only logits speedup via `logits_to_keep=1`.
- Attention backend comparison: eager, SDPA, FlashAttention for audio encoder and Qwen2.
- End-to-end requests/hour split into preprocessing, audio tower/projector, prefill, decode.

## 14. Skip/Defer List

- Training, gradient checkpointing, dropout/layerdrop stochastic behavior.
- Original private source repo conversion parity unless access to `nvidia/music-flamingo-2601` is available.
- Audio synthesis, vocoder, codec/codebook generation: not part of this source family.
- Beam search, speculative decoding, assistant generation, and advanced generation processors.
- Sliding-window Qwen2 attention for first public-checkpoint integration.
- Tensor parallel and pipeline parallel wrapper plans.
- Quantized or packed weights; public metadata reports BF16 safetensors, no source-coupled quantization path.
- Multi-audio-per-message semantics beyond the processor's current “audio found” marker behavior.

## 15. Final Implementation Checklist

- [ ] Parse MusicFlamingo, nested AudioFlamingo3 encoder, nested Qwen2, processor, tokenizer, and generation configs.
- [ ] Validate public checkpoint token IDs instead of source defaults.
- [ ] Implement/bridge Whisper 16 kHz log-mel preprocessing and 30-second windowing.
- [ ] Implement audio placeholder expansion with BOS/EOS boundary tokens and exact post-pool frame counts.
- [ ] Lower AudioFlamingo3 Conv1d/GELU front-end with NCT layout guards.
- [ ] Lower audio learned positional embedding and bidirectional attention mask.
- [ ] Implement AudioFlamingo3 encoder layer, final AvgPool1d, and LayerNorm.
- [ ] Implement MusicFlamingo timestamp construction.
- [ ] Implement MusicFlamingo RoTE with checkpoint-compatible rotated prefix and dtype behavior.
- [ ] Implement multimodal projector `1280 -> 3584 -> 3584`.
- [ ] Implement ordered audio embedding stitch into Qwen2 embeddings.
- [ ] Lower Qwen2 causal LM prefill with GQA and RoPE.
- [ ] Define optional Qwen2 KV cache ABI for `use_cache=True`.
- [ ] Add no-cache prefill parity tests for text-only and audio+text.
- [ ] Add cached decode parity tests once cache is admitted.
- [ ] Benchmark preprocessing, audio tower/projector, prefill, decode, and cache memory separately.
