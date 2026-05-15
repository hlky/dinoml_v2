# Voxtral Transformers Audit

Primary target: native Transformers `VoxtralForConditionalGeneration` inference for audio-text-to-text generation: log-mel audio chunks are encoded, projected into the Llama hidden space, stitched into text embeddings at audio placeholder token positions, then consumed by a causal text decoder for prefill/decode.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  mistralai/Voxtral-Mini-3B-2507 and mistralai/Voxtral-Small-24B-2507
Config source:
  HF config.json, preprocessor_config.json, generation_config.json, HF model API search metadata
Source files inspected:
  transformers/src/transformers/models/voxtral/configuration_voxtral.py
  transformers/src/transformers/models/voxtral/modular_voxtral.py
  transformers/src/transformers/models/voxtral/modeling_voxtral.py
  transformers/src/transformers/models/voxtral/processing_voxtral.py
  transformers/src/transformers/models/voxtral/convert_voxtral_weights_to_hf.py
  transformers/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py
  transformers/src/transformers/models/llama/modeling_llama.py
  transformers/src/transformers/models/whisper/feature_extraction_whisper.py
  transformers/src/transformers/masking_utils.py
  transformers/src/transformers/modeling_utils.py
  transformers/src/transformers/integrations/{sdpa_attention,flash_attention,flex_attention}.py
Any missing files or assumptions:
  processor_config.json was not present for the official Mini repo. Processor behavior is taken from source plus preprocessor_config.json. Voxtral Realtime and Voxtral TTS repos are adjacent families and out of scope for this native model_type=voxtral report.
```

Primary links:

- [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)
- [Transformers Voxtral docs](https://huggingface.co/docs/transformers/model_doc/voxtral)
- [Generated source at commit](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/voxtral/modeling_voxtral.py)
- [Authoritative modular source at commit](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/voxtral/modular_voxtral.py)
- [Processor source at commit](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/voxtral/processing_voxtral.py)

Local notes are under `agents/plans/transformers/voxtral/_sources/`. `modeling_voxtral.py` is generated from `modular_voxtral.py`; future Transformers edits should target the modular file, while this audit inspected both.

## 2. High-level architecture

Voxtral is an audio encoder + multimodal projector + causal text decoder:

```text
audio/text request -> Mistral tokenizer + Whisper log-mel features
  -> Voxtral audio encoder over NCL mel chunks
  -> group audio sequence positions and project 5120 -> text hidden
  -> replace <audio> token embeddings by projected audio rows
  -> Llama causal decoder prefill
  -> autoregressive decode with KV cache
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: audio load/resample/mono conversion, Mistral chat or transcription request templating, audio placeholder token insertion, Whisper log-mel extraction.
- Audio encoder stage: `input_features[B_audio_chunks, 128, 3000] -> last_hidden_state[B_audio_chunks, 1500, 1280]`. This is independently cacheable for repeated prompts over the same audio.
- Projector stage: group four consecutive audio encoder positions to `[B_audio_chunks * 375, 5120]`, then `Linear(5120 -> text_hidden) -> GELU -> Linear(text_hidden -> text_hidden)`.
- Prefix construction: text token embeddings `[B_text, S, text_hidden]`; source uses `masked_scatter` to replace every audio placeholder token row with projected audio rows in flattened row-major order.
- Decoder prefill/decode: delegated to `AutoModelForCausalLM` from `text_config`, normally native Llama. `input_features` are passed only on the first generation iteration or when cache is disabled.

First useful DinoML target: preprocessed `input_features`, `input_ids`, and `attention_mask` into prefill logits plus decode with KV cache. The raw waveform-to-log-mel pipeline can remain CPU-owned initially, but its exact shape contract must be honored.

## 3. Important config dimensions

Representative config sweep:

| Checkpoint | Source | Audio tower | Text decoder | Operator-significant notes |
| --- | --- | --- | --- | --- |
| `tiny-random/voxtral` | public debug config | H=64, FFN=256, L=2, heads=2, head_dim=32, max_source_positions=1500 | H=64, FFN=128, L=2, Q heads=2, KV heads=1, head_dim=32 | Good shape-smoke target; still uses 128 mel bins and 131072 vocab. |
| `mistralai/Voxtral-Mini-3B-2507` | official config | H=1280, FFN=5120, L=32, heads=20, head_dim=64 | H=3072, FFN=8192, L=30, Q heads=32, KV heads=8, head_dim=128 | Common target; BF16; GQA decoder; 131072 vocab/context. |
| `MohamedRashad/Voxtral-Mini-3B-2507-transformers` | open mirror config | same as official Mini | same as official Mini | Useful mirror; label provenance if used. |
| `mistralai/Voxtral-Small-24B-2507` | official config | same as official Mini | H=5120, FFN=32768, L=40, Q heads=32, KV heads=8, head_dim=128 | Production-scale text decoder; audio tower unchanged. |
| `VincentGOURBIN/voxtral-small-8bit` | quantized mirror config | same as official Small | same as official Small | Adds a large `quantization` map; native source does not read it. |
| `mzbac/voxtral-mini-3b-4bit-mixed` | quantized mirror config | same as official Mini | same as official Mini | Mixed 4/6-bit map; external provider/loading metadata only for native source basis. |

Core dimensions:

| Field | Mini | Small | Source/effective meaning |
| --- | ---: | ---: | --- |
| `audio_config.num_mel_bins` | 128 | 128 | `Conv1d` input channels; NCL feature layout. |
| `audio_config.max_source_positions` | 1500 | 1500 | Encoder sequence length after `conv2` stride 2. |
| Mel chunk frames | 3000 | 3000 | Processor `max_source_positions`; encoder input length must be `1500 * 1 * 2`. |
| `audio_config.hidden_size` | 1280 | 1280 | Audio encoder hidden width. |
| `audio_config.intermediate_size` | 5120 | 5120 | Audio FFN width and projector grouped input width. |
| `audio_config.num_hidden_layers` | 32 | 32 | Whisper/Qwen2Audio-style encoder layers. |
| `audio_config.num_attention_heads` | 20 | 20 | Audio encoder MHA, head dim 64. |
| `text_config.hidden_size` | 3072 | 5120 | Decoder hidden and projector output. |
| `text_config.intermediate_size` | 8192 | 32768 | Llama SwiGLU FFN width. |
| `text_config.num_hidden_layers` | 30 | 40 | Decoder layers. |
| `text_config.num_attention_heads` | 32 | 32 | Query heads. |
| `text_config.num_key_value_heads` | 8 | 8 | GQA KV heads; cache stores 8 heads, not 32. |
| `text_config.head_dim` | 128 | 128 | Projection width per head; Q width 4096 for Mini despite hidden 3072. |
| `text_config.max_position_embeddings` | 131072 | 131072 | Long context. |
| `text_config.rope_theta` | 100000000.0 | 100000000.0 | Llama RoPE base. |
| `audio_token_id` | 24 | 24 | Placeholder token id. |
| `projector_hidden_act` | gelu | gelu | Projector activation. |
| `torch_dtype` | bfloat16 | bfloat16 | From configs. |

Projector grouping contract: `get_audio_features` reshapes `audio_outputs.last_hidden_state` from `[B_chunks, 1500, 1280]` to `[-1, 5120]` before `linear_1`. For official configs this groups four consecutive post-conv audio positions into one decoder audio embedding, so one 30-second chunk produces 375 placeholder rows. DinoML should validate `audio_config.intermediate_size % audio_config.hidden_size == 0` and sequence-length divisibility before lowering this reshape.

Processor dimensions from official Mini `preprocessor_config.json` and source defaults:

| Field | Value |
| --- | ---: |
| `feature_extractor_type` | `WhisperFeatureExtractor` |
| `sampling_rate` | 16000 |
| `chunk_length` | 30 seconds |
| `n_samples` | 480000 |
| `feature_size` | 128 |
| `n_fft` | 400 |
| `hop_length` | 160 |
| `nb_max_frames` | 3000 |
| `return_attention_mask` | false |
| `pad_to_multiple_of` | 480000 by VoxtralProcessor default |
| `truncation` | false by VoxtralProcessor default |

## 3a. Family variation traps

- `modeling_voxtral.py` is generated. Treat `modular_voxtral.py` as the source-edit authority.
- Native Voxtral composes `AutoModel.from_config(audio_config)` and `AutoModelForCausalLM.from_config(text_config)`. The decoder operator surface is owned by the nested `text_config.model_type`, normally `llama`.
- Voxtral Realtime (`voxtral_realtime`) and Voxtral TTS (`voxtral_tts`) repos are not covered by this native `model_type=voxtral` audit.
- Audio preprocessing chunks one user audio sample into multiple model examples. The model graph does not consume reassembly metadata; only the processor uses the chunking/count relationship.
- `VoxtralProcessor` has two different `max_source_positions` meanings: processor mel chunk frames default 3000, encoder config post-conv sequence positions 1500.
- The encoder ignores `attention_mask`; silence/padding behavior is handled by log-mel features, not a model-side mask.
- Audio encoder uses NCL `Conv1d` input `[B, mel_bins, frames]`, then permutes to sequence `[B, frames/2, hidden]`. Do not silently translate to NHWC/NLC without rewriting conv axes.
- Source declares an unused `avg_pooler = AvgPool1d(2, stride=2)` and `_get_feat_extract_output_lengths` applies two reductions, but `VoxtralEncoder.forward` does not call `avg_pooler`; first integration should follow forward plus the later projector reshape grouping, not helper intent.
- Audio attention scales Q before the backend call and then passes `scaling=1.0`; Llama decoder attention passes unscaled Q and `scaling=head_dim**-0.5`. Fused attention parity depends on preserving this ordering.
- Audio encoder Q/K/V projections are not packed; K projection is bias-free, Q/V/out projections have bias by source default.
- Text decoder has GQA with `num_key_value_heads < num_attention_heads`; cache stores unexpanded KV heads.
- Mini has `hidden_size=3072`, `num_attention_heads=32`, `head_dim=128`, so Q/O attention width is 4096, not equal to hidden size. Do not infer Q projection width from hidden size.
- Official configs set `sliding_window=null`; source Llama masking can support sliding attention generally, but native Voxtral official configs do not require it.
- `audio_token_id=24` is hardcoded in both config/converter and processor source. Tokenizer chat/transcription templates own placeholder count and order.
- `masked_scatter` replacement is broad PyTorch surface; processor-derived prompts should allow a stricter row-copy lowering with guards on placeholder count and flattened order.
- Quantized mirror configs advertise `quantization` maps, but native Voxtral source does not read those fields. Treat them as external loading/provider contracts, not neural graph ops.

## 4. Operator coverage checklist

Tensor/layout ops:

- Audio feature tensor ingest: `input_features[B_audio_chunks, 128, 3000]`, contiguous NCL.
- Conv1d with padding/stride:
  - `conv1`: `Conv1d(128 -> 1280, kernel=3, stride=1, padding=1)`.
  - `conv2`: `Conv1d(1280 -> 1280, kernel=3, stride=2, padding=1)`.
- GELU after both convs.
- Permute `NCL -> NLC`: `[B, 1280, 1500] -> [B, 1500, 1280]`.
- Learned/fixed audio position embedding add: `[1500, 1280]` broadcast over batch.
- Reshape/group audio output: `[B_chunks, 1500, 1280] -> [B_chunks * 375, 5120]`.
- Text embedding lookup: `[B, S] -> [B, S, text_hidden]`.
- Placeholder mask: compare `input_ids == 24`, expand to embedding shape, validate element count.
- `masked_scatter` / replacement row copy from flattened audio embeds into text embeddings.
- Decoder views/transposes for Q/K/V, cache append, attention output reshape, and `logits_to_keep` slicing.

Neural network primitives:

- Audio LayerNorm over 1280.
- Audio MHA:
  - Q `Linear(1280 -> 1280, bias=True)`.
  - K `Linear(1280 -> 1280, bias=False)`.
  - V `Linear(1280 -> 1280, bias=True)`.
  - O `Linear(1280 -> 1280, bias=True)`.
- Audio FFN: `Linear(1280 -> 5120) -> GELU -> Linear(5120 -> 1280)`, both with bias.
- Audio residual adds and optional fp16 clamp.
- Projector:
  - Reshape/group: `[B_chunks, 1500, 1280] -> [B_chunks * 375, 5120]`.
  - `Linear(5120 -> text_hidden, bias=False) -> GELU -> Linear(text_hidden -> text_hidden, bias=False)`.
  - Admission guard: verify `audio_config.intermediate_size == group_size * audio_config.hidden_size` and `1500 % group_size == 0`.
- Llama decoder primitives:
  - RMSNorm over text hidden.
  - Q/K/V/O projections with `attention_bias=false` in official configs.
  - SwiGLU MLP: `silu(gate_proj(x)) * up_proj(x) -> down_proj`.
  - LM head `Linear(text_hidden -> 131072, bias=False)`.

Attention primitives:

- Audio encoder noncausal full self-attention, MHA, no cache, no RoPE.
- Text decoder causal self-attention, GQA, RoPE, KV cache, long context.
- Source advertises eager, SDPA, FlashAttention, and FlexAttention attention interfaces.
- No encoder-decoder cross-attention: audio features are stitched into token embeddings before the decoder.

Position/rotary ops:

- Audio learned embedding table `embed_positions[1500, 1280]`, `requires_grad=False`, kept in fp32 by strict module list.
- Text Llama RoPE with theta `1e8` over `head_dim=128`; cos/sin computed from `position_ids`, applied to Q/K before cache update.

Generation/cache ops:

- `prepare_inputs_for_generation` drops `input_features` after the first cached iteration.
- Decoder cache is ordinary Llama self-attention cache:
  - Mini/Small key update input per layer: `[B, 8, new_tokens, 128]`.
  - Query heads: 32; repeat factor 4 inside attention backend/eager path.
- `logits_to_keep` is forwarded to the language model and should support last-token logits.

Preprocessing-coupled ops:

- Audio load/resample/mono conversion for string/path/base64/transcription requests.
- Whisper STFT/log-mel extraction:
  - Hann window, `n_fft=400`, `hop_length=160`.
  - Power spectrogram, Slaney mel filters, log10 clamp at `1e-10`.
  - Dynamic range clamp to `max - 8`, then `(x + 4) / 4`.
- Pad raw audio to multiples of 480000 samples, then split mel features into chunks of 3000 frames.
- Mistral-common chat/transcription tokenization and placeholder insertion.

Quantized/packed weight metadata ops:

- None in native source. Quantized mirror `quantization` maps must be a separate provider/loading admission path if targeted.

## 5. Layer/block breakdown

Audio encoder input stem:

```text
input_features: [B_chunks, 128, 3000]
x = GELU(Conv1d(128 -> 1280, k=3, pad=1, stride=1)(input_features))
x = GELU(Conv1d(1280 -> 1280, k=3, pad=1, stride=2)(x))  # [B_chunks, 1280, 1500]
x = x.permute(0, 2, 1)                                   # [B_chunks, 1500, 1280]
x = x + embed_positions[1500, 1280]
```

Audio encoder layer, repeated 32 times for official checkpoints:

```text
residual = x
x_norm = LayerNorm_1280(x)
q = Linear(1280 -> 1280, bias=True)(x_norm) * (64 ** -0.5)
k = Linear(1280 -> 1280, bias=False)(x_norm)
v = Linear(1280 -> 1280, bias=True)(x_norm)
q,k,v = view [B, 1500, 20, 64] -> transpose [B, 20, 1500, 64]
attn = attention(q, k, v, mask=None, scaling=1.0)
x = residual + Linear(1280 -> 1280, bias=True)(merge_heads(attn))

residual = x
x = LayerNorm_1280(x)
x = Linear(1280 -> 5120)(x) -> GELU -> Linear(5120 -> 1280)(x)
x = residual + x
if dtype == fp16: clamp to finite fp16 range minus 1000
```

Audio projector and stitch:

```text
audio_hidden = audio_tower(input_features).last_hidden_state             # [B_chunks, 1500, 1280]
audio_hidden = audio_hidden.reshape(-1, audio_config.intermediate_size)  # [B_chunks * 375, 5120]
audio_embeds = Linear(5120 -> text_hidden, no bias)(audio_hidden)
audio_embeds = GELU(audio_embeds)
audio_embeds = Linear(text_hidden -> text_hidden, no bias)(audio_embeds)

inputs_embeds = token_embedding(input_ids)
mask = input_ids == audio_token_id
validate mask.sum() == audio_embeds.shape[0]
inputs_embeds = masked_scatter(mask.expand_as(inputs_embeds), audio_embeds)
```

Llama decoder layer, repeated 30 Mini / 40 Small:

```text
residual = x
x = RMSNorm(x)
q = Linear(text_hidden -> 32 * 128)(x)
k = Linear(text_hidden -> 8 * 128)(x)
v = Linear(text_hidden -> 8 * 128)(x)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v, layer_idx)
attn = causal_gqa_attention(q, k, v, attention_mask, scale=128 ** -0.5)
x = residual + Linear(32 * 128 -> text_hidden)(attn)

residual = x
x = RMSNorm(x)
x = down_proj(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

Decoder output:

```text
x = final RMSNorm(x)
logits = lm_head(x[:, selected_positions, :])
```

## 6. Attention requirements

Audio encoder attention:

| Property | Requirement |
| --- | --- |
| Type | noncausal self-attention |
| Heads | 20 |
| Head dim | 64 |
| Q/K/V width | 1280 each |
| Cache | none |
| Mask | source accepts `attention_mask` but Voxtral passes/uses none in normal path |
| Position | learned position added before layers; no RoPE |
| Scaling | Q is multiplied by `head_dim**-0.5` before backend call; backend scaling is forced to 1.0 |
| Backends | eager/SDPA/Flash/Flex through `ALL_ATTENTION_FUNCTIONS` |

Text decoder attention:

| Property | Mini / Small requirement |
| --- | --- |
| Type | causal self-attention |
| Q heads | 32 |
| KV heads | 8 |
| Head dim | 128 |
| KV repeat | 4x for eager attention or backend GQA equivalent |
| Q width | 4096 |
| K/V width | 1024 each |
| O input width | 4096 |
| Cache storage | per layer `[B, 8, cache_seq, 128]` for K and V before repeat |
| RoPE placement | apply to Q/K before cache update |
| Mask | `create_causal_mask` from generic Transformers masking; additive/eager masks or backend-specific mask forms |
| Sliding/local | official configs set `sliding_window=null`; reject non-null until separately validated |
| Packed/varlen | source integrations can use Flash/Flex conventions, but native model does not create `cu_seqlens` itself |

Cache distinction:

- Audio encoder/projector outputs are independently cacheable prefix features but not KV cache.
- Decoder self-attention KV cache is the only autoregressive cache.
- There is no cross-attention cache because audio enters as embeddings, not encoder hidden states.

## 7. Position encoding and custom math

Audio position add:

```python
def voxtral_audio_positions(inputs_embeds, embed_positions):
    # inputs_embeds: [B_chunks, 1500, 1280]
    return (inputs_embeds + embed_positions.weight).to(inputs_embeds.dtype)
```

Audio attention scaling:

```python
def voxtral_audio_qkv(x, q_proj, k_proj, v_proj, num_heads=20, head_dim=64):
    b, t, _ = x.shape
    q = q_proj(x) * (head_dim ** -0.5)
    k = k_proj(x)
    v = v_proj(x)
    q = q.view(b, t, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(b, t, num_heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(b, t, num_heads, head_dim).transpose(1, 2).contiguous()
    return q, k, v  # backend scaling must be 1.0 for parity
```

Whisper log-mel math from `WhisperFeatureExtractor`:

```python
mel = mel_filters.T @ (abs(stft(waveform, n_fft=400, hop=160, hann)) ** 2)
log_spec = log10(clamp(mel, min=1e-10))
log_spec = maximum(log_spec, max(log_spec) - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Text RoPE is standard Llama full-head RoPE with theta `100000000.0`:

```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
cos, sin = cat([freqs, freqs], dim=-1).cos(), cat([freqs, freqs], dim=-1).sin()
q = q * cos[:, None] + rotate_half(q) * sin[:, None]
k = k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Precompute opportunities:

- Audio `embed_positions` is fixed length 1500 and can be stored as a constant.
- Log-mel preprocessing can be CPU-owned first; GPU STFT/log-mel is a later optimization.
- Text RoPE cos/sin can be precomputed/bucketed, but decode needs cache-position offsets.

## 8. Preprocessing and input packing

Raw audio contract:

- Expected sampling rate: 16 kHz.
- Mono audio only for feature extraction; string/path URLs are loaded with `force_mono=True`. Array stereo inputs in transcription requests are averaged to mono before writing a buffer.
- Voxtral processor pads to nearest 480000 raw samples by default. This is exactly 30 seconds at 16 kHz.
- `truncation=false` by default, so long audio can become multiple 30-second chunks rather than being clipped.

Feature tensor contract:

- `WhisperFeatureExtractor` emits `input_features` shaped `[batch, 128, num_frames]`, where one 30-second chunk has 3000 frames.
- `VoxtralProcessor._retrieve_input_features` reshapes each audio sample to `[128, -1, 3000]`, transposes to `[num_chunks, 128, 3000]`, and concatenates chunks from all audio samples along batch dimension.
- After the audio tower, source groups each 30-second chunk's 1500 encoder positions into 375 projected audio embeddings. Template placeholder count must match projected rows, not raw mel frames or post-conv positions.
- The neural model receives only the concatenated chunks. It does not receive per-user-audio chunk mapping metadata.

Text/audio packing:

- Chat template and transcription request logic are delegated to `mistral_common`.
- `audio_token_id=24`; processor also stores `audio_token = tokenizer.convert_ids_to_tokens(24)`.
- `VoxtralProcessor.__call__` rejects literal audio placeholder tokens in plain text and requires `apply_chat_template` or `apply_transcription_request` for audio.
- The model validates only total placeholder row count: `n_audio_tokens == audio_features.shape[0]` after comparing element counts through expanded masks.
- Source flatten order for replacement is row-major over `[batch, seq]`: all `True` placeholder rows in `input_ids` order are replaced by `audio_embeds` row order.

Generation-controller behavior:

- During generation, `input_features` should be supplied at prefill only. `prepare_inputs_for_generation` removes it from cached decode iterations unless `is_first_iteration` or `use_cache=false`.
- `generation_config.json` from official Mini contains `bos_token_id=1`, `eos_token_id=2`, `pad_token_id=11`; chat/transcription token templates still define much of end-to-end control.

## 9. Graph rewrite / lowering opportunities

### Rewrite: placeholder masked_scatter -> guarded row copy

Source pattern:

```text
mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, audio_embeds)
```

Replacement:

```text
Embedding(input_ids) -> IndexedRowCopy(audio_embeds into rows where input_ids == 24)
```

Preconditions:

- `input_ids` is present; avoid the `inputs_embeds == embedding(audio_token)` fallback for first integration.
- `audio_embeds.shape[0] == count(input_ids == 24)`.
- `audio_embeds.shape[1] == text_hidden`.
- Placeholder traversal order is row-major `[batch, seq]`.
- Output shape remains `[B, S, text_hidden]`.

Failure cases:

- Caller supplies only `inputs_embeds`.
- Placeholder count mismatch.
- Nonstandard tokenizer/template changes placeholder ID or expects grouped audio metadata.

Parity sketch: synthesize `input_ids` with placeholders at prefix, middle, and multiple rows; compare full `inputs_embeds` after replacement to HF.

### Rewrite: audio Conv1d stem -> NCL GEMM/im2col

Source pattern:

```text
Conv1d(128 -> 1280, k=3, pad=1, stride=1) -> GELU
Conv1d(1280 -> 1280, k=3, pad=1, stride=2) -> GELU
```

Replacement:

```text
NCL pad/window_extract -> GEMM(weight_flat.T) -> bias -> GELU
```

Preconditions:

- Input is NCL `[B, C, L]`.
- Kernel=3, dilation=1, groups=1.
- `conv2` output length uses PyTorch floor formula with stride 2 and padding 1, producing 1500 from 3000.
- Preserve zero padding at both ends.

Layout constraints:

- A channel-last/NLC optimization is possible only for this local stem if window extraction and weight flatten order are rewritten together.
- The downstream encoder sequence is NLC, so the post-conv `permute` is a natural boundary.

### Rewrite: audio encoder attention QKV packing

Source pattern:

```text
q_proj(x) * scale, k_proj(x), v_proj(x)
```

Replacement:

```text
packed Linear(1280 -> 3840) split [Q,K,V], then apply scale to Q
```

Preconditions:

- Same input `x`.
- Bias handling preserves source asymmetry: Q and V have bias, K is bias-free. A packed bias must insert zeros for K or use separate bias add.
- Split order must be all-Q, all-K, all-V.

Failure cases:

- Provider cannot represent mixed bias in packed projection.
- Attention backend also applies scaling; Voxtral audio requires backend scaling 1.0 after pre-scaled Q.

### Rewrite: audio encoder/projector cache

Source pattern:

```text
get_audio_features(input_features) inside prefill forward
```

Replacement:

```text
run audio_tower + projector once -> cache audio_embeds rows -> reuse for prompts over same audio
```

Preconditions:

- `input_features` unchanged.
- Same model weights/projector dtype.
- Placeholder count in each prompt matches cached projected row count: 375 rows per 30-second chunk for official configs.

Failure cases:

- Any request changes audio chunking, dtype, or projector weights.

### Rewrite: Llama decoder QKV packing and GQA attention

Source pattern:

```text
q_proj, k_proj, v_proj with output widths [32*128, 8*128, 8*128]
```

Replacement:

```text
packed GEMM split [Q=4096, K=1024, V=1024] -> RoPE Q/K -> GQA attention
```

Preconditions:

- `text_config.model_type == "llama"` or separately audited equivalent.
- `num_attention_heads % num_key_value_heads == 0`.
- Cache stores K/V before repeat expansion.
- `sliding_window is None` for official first path.

Failure cases:

- Non-Llama nested decoder, non-null sliding window, attention bias variants without coverage, or advanced RoPE scaling.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
for decode/logits_to_keep=1, run LM head only on [B,1,H]
```

Preconditions:

- No training loss.
- Caller does not request full prompt logits.
- Slice is trailing/token-index tensor supported by the runtime.

## 10. Kernel fusion candidates

Highest priority:

- Audio placeholder row-copy lowering: avoids admitting general boolean scatter for a processor-guaranteed pattern.
- Decoder GQA FlashAttention/SDPA equivalent with KV cache: dominant prefill/decode path, with 32 Q heads and 8 KV heads.
- RMSNorm + Llama SwiGLU MLP fusions for Mini/Small text decoder.
- Audio Conv1d stem and audio MHA/FFN kernels: the encoder processes 1500 tokens per 30-second chunk over 32 layers.

Medium priority:

- Audio encoder QKV packing with mixed-bias handling.
- Audio encoder/projector output cache for repeated prompting or transcription variants over the same audio.
- Projector GEMM/activation fusion once the four-position grouping path has parity coverage.
- Last-token-only LM head for 131072 vocab decode.
- RoPE precompute/bucketing for long context and decode offsets.

Lower priority:

- GPU Whisper STFT/log-mel extraction; keep CPU/data-pipeline first unless preprocessing dominates.
- FlexAttention-specific lowering; official configs do not require sparse/block masks.
- Quantized mirror config maps; treat as later provider/load work.
- The unused `avg_pooler` path; do not optimize until a source path uses it.

## 11. Runtime staging plan

1. Parse `VoxtralConfig` and nested `audio_config`/`text_config`; reject unsupported nested `text_config.model_type` values until separately audited.
2. Establish a CPU preprocessing boundary: accept already computed `input_features[B_chunks,128,3000]`, `input_ids`, and `attention_mask`.
3. Implement audio Conv1d stem and one encoder layer parity for debug/native configs.
4. Implement projector grouping `[chunks,1500,1280] -> [chunks*375,5120]` with divisibility guards and checkpoint weight-shape validation.
5. Implement full audio tower + projector to produce audio embedding rows.
6. Lower placeholder replacement as guarded indexed row copy with strict count/order validation.
7. Compose Llama decoder prefill with input embeddings, causal mask, RoPE, GQA, and `logits_to_keep`.
8. Add decode with KV cache and ensure `input_features` are not rerun after first cached iteration.
9. Add optimized attention/fusions: decoder GQA, audio MHA, SwiGLU/RMSNorm, audio Conv1d stem.
10. Optional later: own GPU log-mel preprocessing, quantized mirrors, continuous batching with cached audio embeddings.

Initially stub/defer raw audio loading, Mistral-common templating, sampling controllers beyond greedy/logit parity, and quantized mirror loading.

## 12. Parity and validation plan

- Processor parity:
  - Compare Whisper log-mel features from a short waveform padded to 480000 samples.
  - Verify multi-minute audio splits to `[num_chunks, 128, 3000]`.
  - Verify sample-rate mismatch and stereo-to-mono behavior at the CPU boundary.
- Audio stem parity:
  - Random `[B,128,3000]` through `conv1/GELU/conv2/GELU/permute`.
  - Check exact output length 1500.
- Audio attention parity:
  - One layer, no mask, both eager and selected backend if supported.
  - Explicitly test pre-scaled Q with backend scaling 1.0.
- Projector parity:
  - Validate actual `linear_1.weight.shape == [text_hidden, 5120]`.
  - Test reshape grouping plus `Linear -> GELU -> Linear` for Mini and Small hidden widths.
- Placeholder replacement parity:
  - Prefix-only audio placeholders, interleaved text/audio placeholders, multiple batch rows.
  - Count mismatch must fail before decoder.
- Decoder parity:
  - One Llama layer with GQA and RoPE for Mini dimensions.
  - Prefill logits for short text-only prompt.
  - Audio+text prefill logits with a tiny/debug config if available.
  - Decode one token at a time with cache vs full recompute.
- End-to-end:
  - Compare HF `generate` greedy output for a known audio prompt after processor/template parity is available.

Suggested tolerances:

- FP32 unit ops: `rtol=1e-4`, `atol=1e-5`.
- BF16 full blocks/logits: start with `rtol=2e-2`, `atol=2e-2`, then tighten per op.
- STFT/log-mel: source says CPU/GPU feature extraction should be similar around `1e-5`; use a slightly looser tolerance if different FFT backends are used.

No DinoML imports, tests, or model execution were run for this docs-only audit.

## 13. Performance probes

- Audio preprocessing throughput: waveform load/resample/log-mel/split, seconds of audio per second.
- Audio encoder throughput per 30-second chunk: batch chunks 1/2/4/8, sequence 1500, BF16.
- Projector throughput and memory traffic for `B_chunks * 375` grouped rows.
- Placeholder stitch cost: general scatter vs guarded row copy for long prompts with many audio placeholders.
- Decoder prefill split: text-only vs audio+text, with audio encoder included/excluded.
- Decode tokens/sec sweep: active batch size, cache length, Mini vs Small.
- KV cache memory: layers x 2 x `[B, 8, T, 128]` x dtype.
- Audio embedding cache memory: `[B_chunks * 375, text_hidden]` x dtype.
- Attention backend comparison: eager, SDPA, FlashAttention, DinoML GQA for decoder; separate audio MHA probe for 1500-token encoder.
- LM head probe: full logits vs `logits_to_keep=1`.
- Optional preprocessing placement: CPU log-mel vs GPU STFT/log-mel if CPU pipeline dominates.

## 14. Skip/defer list

- Training, gradients, layerdrop/dropout behavior, and gradient checkpointing.
- Raw audio decode/resample inside DinoML runtime; keep it in the data pipeline first.
- Full Mistral-common tokenizer/template implementation inside the graph.
- Beam search, speculative decoding, and advanced sampling controllers.
- Voxtral Realtime and Voxtral TTS model families.
- Quantized mirror configs and GGUF conversions until a concrete provider/loading target is chosen.
- Non-null decoder `sliding_window`, alternative nested decoder families, and advanced RoPE variants not present in official configs.
- General boolean `masked_scatter`; use guarded row copy for placeholders.
- NHWC/channel-last translation for audio stem until a local NCL-to-NLC rewrite is explicitly proven.

## 15. Final implementation checklist

- [ ] Parse `VoxtralConfig` and nested `VoxtralEncoderConfig` / Llama text config.
- [ ] Load processor metadata for 16 kHz, 128 mel bins, 3000-frame chunks.
- [ ] Define CPU boundary for audio load/resample/log-mel and Mistral-common templating.
- [ ] Implement NCL Conv1d stem with exact 3000 -> 1500 length.
- [ ] Implement audio LayerNorm, MHA with pre-scaled Q, GELU FFN, residuals, and fp16 clamp.
- [ ] Preserve audio learned position embedding constant and dtype behavior.
- [ ] Implement and guard projector grouping from four 1280-wide audio positions into one 5120-wide row.
- [ ] Implement projector `Linear -> GELU -> Linear`.
- [ ] Implement placeholder count/order validation and row-copy replacement for `audio_token_id=24`.
- [ ] Compose nested Llama decoder with GQA, RoPE theta `1e8`, causal mask, KV cache, and `logits_to_keep`.
- [ ] Ensure generation prefill consumes `input_features` and cached decode does not rerun the audio tower.
- [ ] Add parity tests for processor, audio stem, one audio layer, projector, placeholder stitch, decoder prefill, and decode cache.
- [ ] Add benchmarks for preprocessing, audio encoder, projector, prefill, decode, KV memory, and row-copy stitch.
