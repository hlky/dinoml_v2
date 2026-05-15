# Qwen2-Audio Transformers Family Audit

Primary target: `Qwen2AudioForConditionalGeneration` audio/text-to-text generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited and no DinoML tests were run.

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen2_audio
Primary task: multimodal audio/text causal LM generation
Local source root: transformers
```

Source files inspected:

- `src/transformers/models/qwen2_audio/configuration_qwen2_audio.py`
- `src/transformers/models/qwen2_audio/modeling_qwen2_audio.py`
- `src/transformers/models/qwen2_audio/processing_qwen2_audio.py`
- Supporting text decoder source: `src/transformers/models/qwen2/configuration_qwen2.py`, `src/transformers/models/qwen2/modeling_qwen2.py`
- Supporting audio preprocessor source: `src/transformers/models/whisper/feature_extraction_whisper.py`
- Local tests: `tests/models/qwen2_audio/test_modeling_qwen2_audio.py`, `tests/models/qwen2_audio/test_processing_qwen2_audio.py`

Representative configs inspected:

- [Qwen/Qwen2-Audio-7B `config.json`](https://huggingface.co/Qwen/Qwen2-Audio-7B/blob/main/config.json)
- [Qwen/Qwen2-Audio-7B-Instruct `config.json`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/blob/main/config.json)
- [Qwen/Qwen2-Audio-7B `preprocessor_config.json`](https://huggingface.co/Qwen/Qwen2-Audio-7B/raw/main/preprocessor_config.json)
- [Qwen/Qwen2-Audio-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- [Qwen/Qwen2-Audio-7B-Instruct `generation_config.json`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/raw/main/generation_config.json)
- Tokenizer config was inspected for special audio tokens; it is very large because it includes timestamp and other added tokens.

Missing files or assumptions:

- Only two official in-library Qwen2-Audio checkpoints were found for this family: base and instruct. The checkpoint sweep is therefore two official checkpoints plus source defaults/test mini configs where useful.
- No `qwen2_audio` tokenizer implementation exists; tokenization is Qwen2 tokenizer plus Qwen2-Audio special-token conventions.
- The public checkpoint configs are from Transformers `4.38.1` and omit several current `Qwen2Config` fields. Treat current source defaults as effective for omitted text fields, but verify load-time config normalization before weight import.
- `text_config.use_mrope=false` appears in checkpoint JSON, but the pinned Qwen2 text source does not consume `use_mrope`; Qwen2-Audio should not require M-RoPE in this source path.

## 2. High-Level Architecture

Qwen2-Audio is an audio encoder plus text decoder:

```text
raw mono audio + prompt text
  -> WhisperFeatureExtractor log-mel features
  -> Qwen2AudioEncoder conv + bidirectional Transformer encoder + avg pool
  -> Linear audio projector
  -> replace <|AUDIO|> token embeddings in Qwen2 token stream
  -> Qwen2 causal LM prefill/decode
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: audio loading/resampling, mono waveform validation, log-mel feature extraction, chat template, tokenizer, audio-token expansion.
- Audio runtime stage: `input_features [num_audios, 128, 3000]` through audio encoder to projected audio embeddings `[num_audios, <=750, hidden_size]`.
- Prefix construction: audio embeddings are masked/indexed into the text embedding tensor at `<|AUDIO|>` positions.
- Prefill: Qwen2 causal LM consumes the fused audio/text embedding sequence and builds text-decoder KV cache.
- Decode: only new text token IDs are passed; `prepare_inputs_for_generation` intentionally stops passing `input_features` after the first cached iteration.

Independently cacheable stages: the audio encoder/projector output for a fixed audio clip can be cached before LM prefill. The text decoder KV cache after fused prefill is the main generation cache.

## 3. Important Config Dimensions

Source defaults from `Qwen2AudioEncoderConfig`:

| Field | Default / behavior |
| --- | --- |
| `num_mel_bins` | 128 |
| `d_model` | 1280 |
| `encoder_layers` | 32 |
| `encoder_attention_heads` | 20 |
| `encoder_ffn_dim` | 5120 |
| `activation_function` | `gelu` |
| `max_source_positions` | 1500 |
| `dropout`, `attention_dropout`, `activation_dropout` | 0.0 |
| `scale_embedding` | false |

Source defaults from current `Qwen2Config` for omitted checkpoint text fields:

| Field | Default / behavior |
| --- | --- |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 32 if omitted/`None` |
| `head_dim` | `hidden_size // num_attention_heads` unless explicit |
| `hidden_act` | `silu` |
| `use_cache` | true |
| `tie_word_embeddings` | false |
| `use_sliding_window` | false, so `sliding_window` becomes `None` in current config post-init |
| `attention_dropout` | 0.0 |

Official checkpoint sweep:

| Model id | Audio encoder | Text decoder fields present in JSON | Effective text inference from current source |
| --- | --- | --- | --- |
| `Qwen/Qwen2-Audio-7B` | 128 mel, 32 layers, 20 heads, 1280 dim, max source positions 1500 | `intermediate_size=11008`, `max_position_embeddings=8192`, `rms_norm_eps=1e-5`, `rope_theta=10000`, `torch_dtype=bfloat16`, `vocab_size=156032`, `sliding_window=32768` | likely 32-layer Qwen2 decoder, 4096 hidden, 32 Q heads, 32 KV heads, no active sliding window unless `use_sliding_window` is true after normalization |
| `Qwen/Qwen2-Audio-7B-Instruct` | same as base | same as base | same as base; generation config adds sampling defaults |
| Source/test mini configs | tests use tiny `ALMModelTester` shapes; `feat_seq_length=60`, `max_source_positions=30` | synthetic Qwen2 mini dimensions from test harness | only useful for parity scaffolding, not production dimensions |

Derived production shapes:

- Feature extractor emits `input_features [A, 128, 3000]` for 30 seconds at 16 kHz.
- Audio encoder requires `input_features.shape[-1] == max_source_positions * conv1.stride * conv2.stride = 3000`.
- Conv2 halves the feature time length to 1500, then final avg pool halves it to at most 750 audio embeddings.
- Processor token expansion computes placeholder count as `((feature_attention_mask.sum(-1) - 1) // 2 + 1 - 2) // 2 + 1`, matching conv2 plus avg-pool output length.

## 3a. Family Variation Traps

- The model is multimodal but the decoder is standard Qwen2; audio support is an embedding stitch, not cross-attention.
- Public configs omit `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `num_key_value_heads`; do not infer these from audio config.
- Checkpoint JSON contains `rope_theta`, while current Qwen2 source reads `config.rope_parameters["rope_theta"]`. Weight import should confirm the config loader normalizes legacy `rope_theta`.
- `sliding_window=32768` appears in JSON, but current `Qwen2Config` disables sliding attention unless `use_sliding_window=True`; this family should not assume local/sliding attention from that field alone.
- Audio encoder attention is bidirectional full self-attention with LayerNorm, not Qwen2 RMSNorm/RoPE causal attention.
- Qwen2 text attention has bias on Q/K/V projections and no bias on O projection; audio encoder Q/V/O have bias, K has no bias.
- Qwen2 text MLP is gated SwiGLU (`silu(gate) * up`), while audio encoder FFN is ungated GELU.
- The processor must expand one `<|AUDIO|>` marker into many `<|AUDIO|>` tokens before modern model forward. The model still has a legacy merge path for unexpanded prompts, but source warns that expansion belongs in processing.
- `masked_scatter` stitching requires exact equality between number of audio placeholder tokens and flattened projected audio features.
- Batched legacy merge tries to infer left/right padding from `attention_mask`; mixed left and right padding is rejected.
- Audio feature layout is channel/time `[batch, mel_bins, frames]`, i.e. Conv1d NCL. No NHWC/NCHW image-style layout translation applies. The only layout-sensitive ops are 1D channel/time permutes around Conv1d/AvgPool1d.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- Embedding lookup for Qwen2 tokens.
- `masked_scatter` or indexed copy from flattened audio features into token embeddings.
- Boolean masks, `torch.where`, `arange`, comparisons, `sum`, `cumsum`, reshape/view, transpose/permute, contiguous.
- Dynamic sequence-length arithmetic for audio output lengths and fused text/audio positions.

Audio preprocessing-coupled ops:

- Mono waveform padding/truncation to 480000 samples for default 30s chunks.
- STFT with Hann window, `n_fft=400`, `hop_length=160`, power spectrum.
- Slaney mel filter bank to 128 bins, log10 clamp at `1e-10`, dynamic range clamp to `max - 8`, affine normalization `(x + 4) / 4`.
- Optional zero-mean/unit-var waveform normalization if requested, although Qwen2-Audio processor defaults do not force it.

Audio encoder primitives:

- Conv1d `128 -> 1280`, kernel 3, padding 1, stride 1, bias true.
- GELU.
- Conv1d `1280 -> 1280`, kernel 3, padding 1, stride 2, bias true.
- Learned positional embedding `[1500, 1280]`, non-trainable in source.
- 32 bidirectional Transformer encoder blocks:
  - LayerNorm over 1280.
  - MHA with 20 heads, head_dim 64.
  - Q projection `1280 -> 1280` bias true.
  - K projection `1280 -> 1280` bias false.
  - V projection `1280 -> 1280` bias true.
  - O projection `1280 -> 1280` bias true.
  - softmax attention with additive bidirectional padding mask.
  - FFN `Linear(1280 -> 5120)` + GELU + `Linear(5120 -> 1280)`.
  - residual adds; fp16 clamp to finite range after block.
- AvgPool1d kernel 2 stride 2 over time after permuting back to NCL.
- Final LayerNorm over 1280.
- Projector `Linear(1280 -> text_hidden_size)` bias true.

Text decoder primitives:

- Qwen2 causal LM stack: token embeddings, repeated RMSNorm, GQA/MHA causal attention, RoPE, SwiGLU MLP, final RMSNorm, `lm_head`.
- Production inferred shapes: hidden 4096, layers 32, Q heads 32, KV heads 32, head_dim 128, MLP intermediate 11008, vocab 156032.
- `lm_head` is bias false. `tie_word_embeddings=false`, so preserve separate LM head and token embedding weights.

Generation/cache ops:

- Dynamic KV cache per text decoder layer.
- Position ID creation from fused attention mask during prefill; during decode Qwen2 computes position IDs from cache length if not supplied.
- `logits_to_keep` in Qwen2 LM can avoid full-sequence logits if integration routes it.
- Generation config for Instruct: `do_sample=true`, `top_k=20`, `top_p=0.5`, `temperature=0.7`, `repetition_penalty=1.1`, `eos_token_id=[151643,151645]`, `pad_token_id=151643`.

## 5. Layer/Block Breakdown

Audio encoder front end:

```text
input_features: [A, 128, 3000]
x = GELU(Conv1d(128 -> 1280, k=3, pad=1, stride=1)(input_features))      # [A, 1280, 3000]
x = GELU(Conv1d(1280 -> 1280, k=3, pad=1, stride=2)(x))                  # [A, 1280, 1500]
x = permute(x, [0, 2, 1])                                                # [A, 1500, 1280]
x = x + embed_positions[0:1500]
```

Audio encoder block, repeated 32 times:

```text
residual = x
x = LayerNorm(x)
q = Linear(1280 -> 1280, bias=True)(x) * (64 ** -0.5)
k = Linear(1280 -> 1280, bias=False)(x)
v = Linear(1280 -> 1280, bias=True)(x)
x = MHA(q, k, v, bidirectional_padding_mask)
x = residual + Linear(1280 -> 1280, bias=True)(x)
residual = x
x = LayerNorm(x)
x = Linear(5120 -> 1280)(GELU(Linear(1280 -> 5120)(x)))
x = residual + x
if fp16: x = clamp(x, +/- (finfo.max - 1000))
```

Audio pooling/projector:

```text
x = permute(x, [0, 2, 1])        # [A, 1280, 1500]
x = AvgPool1d(k=2, stride=2)(x)  # [A, 1280, 750]
x = permute(x, [0, 2, 1])        # [A, 750, 1280]
x = LayerNorm(x)
audio_features = Linear(1280 -> 4096, bias=True)(x)
```

Text decoder block, repeated by Qwen2:

```text
residual = x
x = RMSNorm(x)
q = Linear(hidden -> n_q_heads * head_dim, bias=True)(x)
k = Linear(hidden -> n_kv_heads * head_dim, bias=True)(x)
v = Linear(hidden -> n_kv_heads * head_dim, bias=True)(x)
q, k = RoPE(q, k, position_ids)
k, v = cache.update(k, v, layer_idx) if cache is enabled
x = causal_attention(q, k, v, mask, optional sliding_window)
x = residual + Linear(n_q_heads * head_dim -> hidden, bias=False)(x)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

## 6. Attention Requirements

Audio encoder attention:

- Bidirectional self-attention, no causal mask.
- Full MHA: 20 heads, KV heads equal Q heads, head_dim 64 for production audio config.
- Query is scaled before attention backend call; the call receives `scaling=1.0` to preserve Whisper-like numerical order.
- Attention mask is additive, produced by `create_bidirectional_mask` from frame-validity after conv2 time reduction.
- Source declares FlashAttention/SDPA support through the shared attention interface, but eager fallback is explicit.
- No KV cache in the audio encoder.

Text decoder attention:

- Causal self-attention from Qwen2.
- Production official configs infer MHA, not GQA, because omitted `num_key_value_heads` defaults to `num_attention_heads=32`.
- Cache tensors per layer before repeat expansion are `[batch, num_key_value_heads, cache_seq, head_dim]`, likely `[B, 32, S, 128]` for official 7B.
- Cached keys are stored after RoPE application because `past_key_values.update` happens after `apply_rotary_pos_emb`.
- Eager attention repeats KV heads only when `num_key_value_groups > 1`, computes `q @ k.T * scale`, adds mask, softmaxes in fp32, casts back to query dtype, then multiplies by V.
- Optimized backend path is `ALL_ATTENTION_FUNCTIONS` with Qwen2 passing `sliding_window=self.sliding_window`. For the official configs, sliding should be inactive unless normalized config sets `use_sliding_window`.

Prefill/decode behavior:

- First prefill with audio uses fused `inputs_embeds`; text decoder receives no `input_ids`.
- Decode with cache should pass only new text token IDs/embeddings and no `input_features`.
- If `use_cache=False`, `prepare_inputs_for_generation` allows `input_features` again, so a no-cache generation loop may recompute the audio prefix.

## 7. Position Encoding and Custom Math

Audio encoder uses learned absolute position embeddings over 1500 post-conv feature frames before avg pooling. These are fixed parameters in source (`requires_grad_(False)`).

Text decoder uses Qwen2 RoPE:

```python
def apply_qwen2_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotate_half = torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

RoPE cos/sin are computed in fp32 from `position_ids` and then cast to hidden dtype. They can be precomputed for static max positions for default RoPE, but dynamic/advanced RoPE types in shared Qwen2 source can update buffers and need a guard.

Audio length math that must match processor/model:

```python
conv2_len = (feature_frames - 1) // 2 + 1
pooled_len = (conv2_len - 2) // 2 + 1
```

For a default 30s chunk: `feature_frames=3000`, `conv2_len=1500`, `pooled_len=750`.

## 8. Preprocessing and Input Packing

Waveform contract:

- Input must be mono audio, one float per timestep. Batched NumPy arrays with rank > 2 are rejected.
- Expected sampling rate is 16000 Hz; passing a different `sampling_rate` raises.
- Default chunk length is 30 seconds, `n_samples=480000`.
- Processor forces audio kwargs `return_attention_mask=True` and `padding="max_length"` when audio is provided.

Feature tensor contract:

- `WhisperFeatureExtractor` returns `input_features [num_audios, 128, 3000]` for default config and `feature_attention_mask [num_audios, 3000]`.
- Feature extraction is CPU/data-pipeline work initially. The source has a Torch STFT path that can run on CUDA if requested, but DinoML should not make GPU STFT part of the first compiled model graph.

Audio-token packing:

- Processor counts occurrences of `<|AUDIO|>` in text and requires that count to equal the number of audio arrays.
- For each audio, it computes output audio-token count from `feature_attention_mask.sum(-1)` and replaces one textual `<|AUDIO|>` placeholder with repeated `<|AUDIO|>` tokens.
- If the placeholder is not already surrounded by `<|audio_bos|>` and `<|audio_eos|>`, processor inserts them.
- Token IDs from tokenizer config: `<|AUDIO|>` is 151646, `<|audio_bos|>` is 151647, `<|audio_eos|>` is 151648. Config `audio_token_index` is 151646.

Runtime stitch:

```text
input_ids -> token embeddings [B, S, H]
audio_features -> projector -> [A, T_audio_max, H]
mask valid audio rows by audio_output_lengths
flatten valid audio features
masked_scatter into positions where input_ids == audio_token_id
```

Generation controller:

- `prepare_inputs_for_generation` strips `input_features` on cached decode iterations.
- Chat template is controller/tokenizer work. The in-library default exists only as fallback; the Hub tokenizer config carries a ChatML template.
- Instruct generation config supplies sampling defaults and multi-EOS behavior. Beam search, speculative decoding, and server-side batching are outside the model forward graph.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Audio Conv1d Stride-1 Front End

Source pattern:

```text
Conv1d(128 -> 1280, k=3, pad=1, stride=1) -> GELU
```

Replacement: dedicated Conv1d or im2col/GEMM only if Conv1d support is missing.

Preconditions:

- NCL layout `[batch, channels, time]`.
- Padding exactly 1, dilation 1, groups 1.
- Preserve zero-padding at boundaries.

Failure cases: do not reinterpret as NHWC/channel-last; this is audio NCL, and downstream Conv1d expects channels dimension 1.

Parity test sketch: random `[2,128,3000]` fp32/fp16 against PyTorch Conv1d+GELU.

### Rewrite: Conv1d Stride-2 Length Guard

Source pattern:

```text
Conv1d(1280 -> 1280, k=3, pad=1, stride=2) -> GELU
```

Replacement: Conv1d kernel or guarded GEMM-window lowering.

Preconditions:

- Input time length exactly 3000 for official checkpoints.
- Output length equation `(L - 1) // 2 + 1`.
- Preserve PyTorch Conv1d cross-correlation weight layout `[out_channels, in_channels, kernel]`.

Failure cases: dynamic lengths must still be padded to `max_source_positions * 2`; the encoder raises otherwise.

### Rewrite: Audio Embedding Stitch

Source pattern:

```text
special_audio_mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, flat_audio_features)
```

Replacement: explicit indexed copy into embedding buffer.

Preconditions:

- Number of true mask elements divided by hidden size equals flattened valid audio feature rows.
- Processor-expanded prompts only, so placeholder count already matches audio output lengths.
- Input and audio feature hidden dims match text hidden size.

Failure cases: legacy unexpanded prompt path changes sequence length and recomputes position IDs; route that path to preprocessing or reject for first integration.

Parity test sketch: use local integration expected prompt with 101 audio tokens and compare fused embeddings to PyTorch masked scatter.

### Rewrite: Audio Encoder Attention

Source pattern: bidirectional MHA over `[A, 1500, 1280]`.

Replacement: SDPA/FlashAttention noncausal if backend supports additive padding mask and query pre-scaling order.

Preconditions:

- No dropout in inference.
- Preserve query scaling before backend call or prove numerical equivalence within tolerance.
- Full sequence attention, not causal.

Failure cases: backend that assumes causal or fuses scale in a different order may drift.

### Rewrite: Last-Token LM Head

Source pattern: Qwen2 `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: during decode or sampling-only prefill, project only final token(s).

Preconditions:

- No full-sequence logits requested.
- Loss is not computed.
- Sampling controller only needs last position.

## 10. Kernel Fusion Candidates

Highest priority:

- Qwen2 decoder RMSNorm + QKV projections + RoPE + cached attention. This dominates prefill/decode once audio embeddings are stitched.
- Qwen2 SwiGLU MLP: `silu(gate) * up` plus down projection; large GEMMs and elementwise fusion.
- Audio embedding stitch as an indexed copy, avoiding general `masked_scatter` overhead and dynamic graph complexity.
- Audio encoder LayerNorm + MHA + FFN kernels for encoder-only throughput.

Medium priority:

- Conv1d+GELU front end, especially the large `1280 -> 1280` stride-2 convolution.
- AvgPool1d plus final LayerNorm over audio time.
- Last-token-only logits for generation.
- Audio projector GEMM `1280 -> 4096`.

Lower priority:

- GPU STFT/log-mel preprocessing. Useful eventually for high-throughput serving, but CPU/data-pipeline parity is simpler first.
- Legacy unexpanded audio merge path. Prefer modern processor-expanded prompts.
- Sampling/logits processors; these are controller-side and can initially remain outside DinoML compiled graph.

## 11. Runtime Staging Plan

1. Parse configs and load weights, with explicit checks for legacy text fields (`rope_theta`, `use_mrope`, omitted Qwen2 dimensions).
2. Implement processor-equivalent offline preparation: waveform to `input_features`, `feature_attention_mask`, expanded prompt `input_ids`, and `attention_mask`.
3. Validate audio encoder alone on fixed `[1,128,3000]` features through convs, bidirectional blocks, avg pool, and final LayerNorm.
4. Add audio projector and embedding stitch parity, using precomputed text embeddings if helpful.
5. Run Qwen2 text decoder prefill with fused `inputs_embeds` and no audio recomputation.
6. Add cached decode, ensuring `input_features` are absent after first iteration.
7. Enable optimized attention/GEMM/fusions and last-token logits.
8. Add batching with multiple audio clips per text batch, preserving flattened-audio ordering.

Initially stub or keep outside compiled graph: audio file I/O, resampling, chat templates, tokenizer, sampling controller, beam search.

## 12. Parity and Validation Plan

- Feature extractor parity: known mono waveforms to `input_features` and `feature_attention_mask`, fp32 tolerance around `1e-5` for Torch/NumPy STFT paths.
- Audio length parity: random `feature_attention_mask` lengths through processor formula and `_get_feat_extract_output_lengths`.
- Conv front-end parity: one and two audio examples, fp32 and fp16/bf16 if supported.
- Single audio encoder block parity with random hidden states and additive bidirectional masks.
- Full audio encoder/projector parity on synthetic padded features.
- Stitch parity: compare explicit indexed copy against PyTorch `masked_scatter` for one audio, multi-audio single batch, and two-batch left-padding cases.
- Qwen2 decoder prefill parity with fused embeddings and attention mask.
- Decode parity: one-token cached decode with `past_key_values`; verify no audio encoder invocation after prefill.
- End-to-end smoke: reproduce the local integration shape expectation where glass-breaking audio yields 101 audio tokens in the expanded prompt.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 encoder+decoder `rtol=1e-2, atol=1e-2`, with stricter component tests for preprocessing in fp32.

## 13. Performance Probes

- CPU feature extraction throughput: clips/sec for 1s, 10s, and 30s audio.
- Audio encoder throughput: `[A,128,3000]` batch sweep, separated from text decoder.
- Audio projector + stitch time versus number of audio tokens.
- Prefill throughput versus fused sequence length: text-only, 1 audio, 2 audio clips, batch mixed lengths.
- Decode tokens/sec with populated KV cache and no audio branch.
- KV cache memory per batch/layer/sequence length.
- Attention backend comparison: eager, SDPA, FlashAttention for audio encoder noncausal and text decoder causal.
- Last-token-only logits versus full logits in prefill/decode.
- End-to-end requests/hour with CPU preprocessing overlapped with GPU decode.

## 14. Skip/Defer List

- Training and loss path, including label rewrite in legacy merge.
- Gradient checkpointing and LayerDrop.
- Legacy unexpanded `<|AUDIO|>` merge as a first-class compiled path.
- GPU log-mel/STFT preprocessing.
- Beam search, speculative decoding, and advanced generation controllers.
- Tensor parallel plans and multi-GPU placement.
- Sliding-window Qwen2 attention unless a config explicitly enables `use_sliding_window`.
- Remote-code or fine-tuned derivatives that alter architecture.
- Timestamp/audio special-token generation semantics beyond preserving tokenizer IDs.

## 15. Final Implementation Checklist

- [ ] Parse `Qwen2AudioConfig`, `Qwen2AudioEncoderConfig`, and nested current `Qwen2Config`.
- [ ] Normalize/validate legacy checkpoint fields: `rope_theta`, omitted Qwen2 dimensions, `use_mrope=false`.
- [ ] Load Qwen2-Audio weights with separate audio encoder, projector, token embedding, and untied LM head.
- [ ] Implement/own CPU data-pipeline contract for Whisper log-mel features.
- [ ] Add Conv1d NCL lowering or guarded GEMM-window fallback for audio front end.
- [ ] Add audio encoder LayerNorm, bidirectional MHA, GELU FFN, fp16 clamp, AvgPool1d.
- [ ] Add projector `Linear(1280 -> hidden_size)`.
- [ ] Add processor-compatible audio-token expansion or require pre-expanded prompts.
- [ ] Implement explicit indexed audio embedding stitch.
- [ ] Reuse Qwen2 decoder coverage for RMSNorm, RoPE, causal attention cache, SwiGLU, LM head.
- [ ] Ensure `prepare_inputs_for_generation` semantics: audio only on first/no-cache iteration.
- [ ] Add component parity tests for feature lengths, audio encoder, stitch, prefill, and one-token decode.
- [ ] Benchmark preprocessing, audio encoder, prefill, decode, and end-to-end mixed-audio batches.
