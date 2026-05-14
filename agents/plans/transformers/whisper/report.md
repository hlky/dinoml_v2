# Transformers Whisper Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: openai/whisper-large-v3.
  Additional sizing references: openai/whisper-tiny, openai/whisper-base,
  openai/whisper-small, openai/whisper-medium, openai/whisper-large-v3-turbo.

Config source:
  https://huggingface.co/openai/whisper-large-v3/raw/main/config.json
  https://huggingface.co/openai/whisper-large-v3/raw/main/preprocessor_config.json
  Additional configs and preprocessors fetched from Hugging Face repos listed above.

Source files inspected:
  X:/H/transformers/src/transformers/models/whisper/modeling_whisper.py
  X:/H/transformers/src/transformers/models/whisper/configuration_whisper.py
  X:/H/transformers/src/transformers/models/whisper/feature_extraction_whisper.py
  X:/H/transformers/src/transformers/models/whisper/processing_whisper.py
  X:/H/transformers/src/transformers/models/whisper/tokenization_whisper.py
  X:/H/transformers/src/transformers/models/whisper/generation_whisper.py

Any missing files or assumptions:
  No remote-code files are required for standard OpenAI Whisper checkpoints.
  This report targets WhisperForConditionalGeneration for ASR/transcription and
  speech translation. Audio classification, training SpecAugment, long-form
  timestamp postprocessing, and speculative decoding are optional or deferred.
  Dinoml assumptions: inference-only first, CUDA GPU target, batch throughput
  prioritized, and layout translation handled through guarded layout/fusion
  passes rather than default semantic graph translation.
```

## 2. High-level architecture

Whisper is an audio encoder plus text decoder seq2seq model. Raw mono audio is converted to fixed-length log-mel features. A two-layer Conv1d frontend downsamples 3000 mel frames to 1500 encoder tokens, an encoder transformer produces audio states, and a causal decoder attends to both its own text prefix and the encoder states.

```text
raw mono audio -> log-mel feature extractor -> Conv1d audio frontend -> encoder
  -> decoder prefill with forced prompt tokens -> decode with self/cross cache -> tied LM logits
```

Stage decomposition:

- CPU/data pipeline: resample to 16 kHz if needed outside the feature extractor, mono validation, pad/truncate raw audio, STFT, mel projection, log/clamp/normalize.
- Audio frontend: source input layout `[B, mel, frames]`, Conv1d/GELU/Conv1d/GELU, then `permute(0, 2, 1)` to token layout `[B, 1500, H]`.
- Encoder stage: noncausal MHA transformer blocks with LayerNorm and GELU MLP.
- Decoder stage: token embedding plus learned positions, causal self-attention with cache, cross-attention over encoder outputs, GELU MLP.
- Generation controller: forced language/task/timestamp prompt tokens, suppress-token logits processors, optional timestamp and long-form segmentation logic.

## 3. Important config dimensions

Worked example: `openai/whisper-large-v3`.

| Field | Value | Source |
|---|---:|---|
| primary runtime target | ASR / speech translation seq2seq LM | report scope |
| d_model / H | 1280 | config.json |
| encoder_layers | 32 | config.json |
| decoder_layers | 32 | config.json |
| encoder_attention_heads | 20 | config.json |
| decoder_attention_heads | 20 | config.json |
| head_dim / D | 64 | inferred `H / heads` |
| encoder_ffn_dim | 5120 | config.json |
| decoder_ffn_dim | 5120 | config.json |
| vocab_size | 51866 | config.json |
| num_mel_bins | 128 | config.json/preprocessor |
| max_source_positions | 1500 | config.json |
| required feature frames | 3000 | `max_source_positions * conv strides` |
| max_target_positions | 448 | config.json |
| activation_function | gelu | config.json |
| layer_norm_eps | source default from PyTorch LayerNorm | source |
| conv1 | Conv1d(128 -> 1280, kernel=3, stride=1, padding=1) | source + config |
| conv2 | Conv1d(1280 -> 1280, kernel=3, stride=2, padding=1) | source + config |
| q/k/v/out projection | Linear(1280 -> 1280), k bias false, q/v/out bias true | source |
| LM head | Linear(1280 -> 51866), bias false, tied to decoder embeddings | source/config |
| cache support | EncoderDecoderCache with decoder self-cache and cross-cache | source |
| feature extractor sampling_rate | 16000 | preprocessor_config/source default |
| feature extractor n_fft / hop | 400 / 160 | preprocessor_config/source default |
| feature extractor chunk_length | 30 seconds | preprocessor_config/source default |
| preprocessor return_attention_mask | false by default | preprocessor_config |

Representative checkpoint sweep:

| Checkpoint | H | enc layers | dec layers | heads | FFN | mel bins | vocab | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| openai/whisper-tiny | 384 | 4 | 4 | 6 | 1536 | 80 | 51865 | small/debug shape |
| openai/whisper-base | 512 | 6 | 6 | 8 | 2048 | 80 | 51865 | small production shape |
| openai/whisper-small | 768 | 12 | 12 | 12 | 3072 | 80 | 51865 | mid-size |
| openai/whisper-medium | 1024 | 24 | 24 | 16 | 4096 | 80 | 51865 | larger 80-mel family |
| openai/whisper-large-v3 | 1280 | 32 | 32 | 20 | 5120 | 128 | 51866 | production large; changed feature bins/vocab |
| openai/whisper-large-v3-turbo | 1280 | 32 | 4 | 20 | 5120 | 128 | 51866 | asymmetric encoder-heavy variant |

## 3a. Family variation traps

- `num_mel_bins` is not fixed across real usage: older checkpoints use 80 bins, while large-v3 and large-v3-turbo use 128. The first Conv1d weight shape and feature extractor mel bank must follow config/preprocessor, not source defaults.
- `large-v3-turbo` keeps the large 32-layer encoder but uses only 4 decoder layers. Do not infer decoder depth from encoder depth.
- Whisper uses MHA only; there is no GQA/MQA, but encoder and decoder head counts are separate config fields.
- K projection has `bias=False`; Q, V, and output projections have bias. Packed-QKV rewrites need zero-bias handling for K.
- The model applies query scaling immediately after `q_proj` and passes `scaling=1.0` into the attention backend. Fusing attention with the usual backend scale must preserve this order or fold the scale into Q.
- Encoder input length is fixed by source validation to `max_source_positions * conv1.stride * conv2.stride`, normally 3000 frames. Long-form generation segments longer features into 3000-frame windows in generation code.
- Encoder `attention_mask` is accepted but not used by `WhisperEncoder.forward`; feature masks still matter for generation/timestamp bookkeeping.
- Generation correctness depends on tokenizer/generation config: forced decoder prompt IDs, suppress tokens, language/task tokens, and timestamp tokens are not visible in the core module graph.

## 4. Operator coverage checklist

### Tensor/layout ops

- Pad/truncate raw audio to `chunk_length * sampling_rate` samples, normally 480000 samples for 30 seconds at 16 kHz.
- Feature tensor layout `[B, mel, frames]`, usually `[B,80,3000]` or `[B,128,3000]`.
- Conv frontend output transpose `permute(0,2,1)` from `[B,H,1500]` to `[B,1500,H]`.
- Token layout `[B,T,H]` for encoder and decoder transformer blocks.
- Reshape/view/transposes for attention: `[B,T,H] -> [B,heads,T,D] -> [B,T,H]`.
- Gather/select last-token logits for optimized generation, if Dinoml chooses to avoid full-sequence LM projection.

### Neural network primitives

- Conv1d with padding and stride:
  - large-v3 `Conv1d(128 -> 1280, kernel=3, stride=1, padding=1)`.
  - large-v3 `Conv1d(1280 -> 1280, kernel=3, stride=2, padding=1)`.
- GELU after each Conv1d and in every encoder/decoder MLP.
- LayerNorm over hidden dimension in pre-norm blocks and final encoder/decoder norms.
- Linear projections:
  - Attention Q/V/O: `Linear(H -> H, bias=True)`.
  - Attention K: `Linear(H -> H, bias=False)`.
  - MLP: `Linear(H -> 4H, bias=True)` then `Linear(4H -> H, bias=True)` for standard sizes.
  - LM head: `Linear(H -> vocab_size, bias=False)`, tied to token embedding.
- Embedding lookup for decoder tokens and learned decoder positions.
- Fixed sinusoidal encoder position embedding table addition.

### Attention primitives

- Encoder noncausal self-attention, MHA, sequence length 1500.
- Decoder causal self-attention, MHA, sequence length up to `max_target_positions`.
- Decoder cross-attention, MHA, query length decoder T, key/value length encoder S=1500.
- Additive causal mask for decoder self-attention.
- Optional output attentions force eager paths for token timestamps; this is a slow-path feature.

### Position/relative-bias ops

- Encoder fixed sinusoidal table generation and addition.
- Decoder learned absolute position embedding with offset from past cache length.
- No RoPE, ALiBi, relative bias, or sliding-window attention in the standard model.

### Generation/cache ops

- `EncoderDecoderCache(DynamicCache, DynamicCache)` with per-layer self-attention K/V and cross-attention K/V.
- Self-cache append for decoder generated tokens.
- Cross-cache populate once per decoder layer from encoder outputs, then reuse.
- Forced decoder IDs for language/task/no-timestamps prefix.
- SuppressTokens and SuppressTokensAtBegin logits processors.
- Optional timestamp logits processor and token timestamp extraction.

### Preprocessing-coupled ops

- STFT with Hann window, `n_fft=400`, `hop_length=160`, power spectrogram.
- Mel filter-bank matmul, Slaney mel scale, 80 or 128 filters.
- `log10`, clamp floor to max minus 8, affine normalize `(log_spec + 4) / 4`.
- Attention mask rescale from raw samples to feature frames by slicing every `hop_length`.
- Optional zero-mean/unit-variance raw waveform normalization before feature extraction.

## 5. Layer/block breakdown

Audio frontend:

```text
input_features: [B, M, 3000] where M = num_mel_bins
x = GELU(Conv1d(M -> H, kernel=3, stride=1, padding=1, bias=True))(input_features)
x = GELU(Conv1d(H -> H, kernel=3, stride=2, padding=1, bias=True))(x)
x: [B, H, 1500]
x = permute(x, [0, 2, 1])
x = x + sinusoidal_position_table[0:1500]
```

Encoder block, repeated `encoder_layers` times:

```text
residual = x
x = LayerNorm(x)
q = Linear(H -> H, bias=True)(x) * head_dim**-0.5
k = Linear(H -> H, bias=False)(x)
v = Linear(H -> H, bias=True)(x)
x = MHA(q, k, v, noncausal, scaling=1.0)
x = residual + Linear(H -> H, bias=True)(x)
residual = x
x = LayerNorm(x)
x = Linear(H -> I, bias=True)(x)
x = GELU(x)
x = Linear(I -> H, bias=True)(x)
x = residual + x
```

Decoder block, repeated `decoder_layers` times:

```text
residual = x
x = LayerNorm(x)
q = Linear(H -> H, bias=True)(x) * head_dim**-0.5
k,v = self K/V projections, append/read decoder self-cache
x = causal MHA(q, k, v, cache, scaling=1.0)
x = residual + Linear(H -> H, bias=True)(x)

residual = x
x = LayerNorm(x)
q = Linear(H -> H, bias=True)(x) * head_dim**-0.5
k,v = cross K/V projections of encoder states, cached after first decode step
x = cross MHA(q, k, v, scaling=1.0)
x = residual + Linear(H -> H, bias=True)(x)

residual = x
x = LayerNorm(x)
x = GELU(Linear(H -> I, bias=True)(x))
x = Linear(I -> H, bias=True)(x)
x = residual + x
```

Head:

```text
decoder_hidden: [B, T, H]
logits = MatMul(decoder_hidden, tied_embedding_weight.T)  # [B,T,V]
```

## 6. Attention requirements

Whisper requires three MHA variants:

| Site | Causality | Source states | Heads | KV heads | Head dim | Cache | Mask |
|---|---|---|---:|---:|---:|---|---|
| Encoder self-attn | noncausal | encoder tokens | Aenc | Aenc | `H/Aenc` | none | usually none |
| Decoder self-attn | causal | decoder tokens | Adec | Adec | `H/Adec` | append K/V | causal + optional decoder padding |
| Decoder cross-attn | noncausal | encoder tokens | Adec | Adec | `H/Adec` | reusable cross K/V | source passes `None` in normal path |

For large-v3, all head dims are 64 (`1280 / 20`). Cache tensors are naturally stored as `[B, heads, seq, head_dim]` after projection and transpose. Decoder self-cache grows from prompt length to generated length. Cross-cache stores K/V for the encoder sequence, normally `[B, 20, 1500, 64]` per decoder layer for large-v3.

The source dispatches through `ALL_ATTENTION_FUNCTIONS` for configured attention backends and falls back to eager attention. Parity should first reproduce eager semantics:

```text
attn_weights = MatMul(q, k^T) * scaling
attn_weights += additive_mask if present
attn_probs = Softmax(attn_weights, dim=-1)
out = MatMul(attn_probs, v)
```

Whisper-specific nuance: `q` is already multiplied by `head_dim**-0.5`, and the backend is called with `scaling=1.0`. FlashAttention/SDPA compatibility is good for normal inference if this scaling is folded consistently. Token timestamp extraction can force eager attention because it needs attention weights; that path is likely too slow for first optimized runtime.

## 7. Position encoding and custom math

Encoder positions are fixed sinusoidal embeddings initialized into a frozen table:

```python
def whisper_sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    inv = exp(-(log(max_timescale) / (channels // 2 - 1)) * arange(channels // 2))
    scaled = arange(length)[:, None] * inv[None, :]
    return concat([sin(scaled), cos(scaled)], dim=1)
```

This can be precomputed per `(max_source_positions, d_model)` and loaded as a constant. It does not depend on batch inputs.

Decoder positions are learned absolute embeddings. Position IDs default to:

```python
position_ids = arange(current_decoder_length) + past_key_values_length
```

Whisper attention scaling should be reproduced as:

```python
def whisper_project_q(q_proj_out, head_dim):
    return q_proj_out * (head_dim ** -0.5)
```

If Dinoml folds this into weights, scale both Q weight and Q bias, and keep attention backend scale at 1.0 for parity.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Accept mono raw audio arrays; stereo/multichannel input is rejected by the feature extractor.
- Require or verify sampling rate 16 kHz. If source audio differs, resampling happens outside the inspected feature extractor.
- Pad/truncate to `chunk_length * sampling_rate`, normally 480000 samples.
- Optional `do_normalize` applies zero-mean/unit-variance normalization using sample-level attention mask.
- Compute log-mel features with STFT and mel filters, returning `[B, num_mel_bins, 3000]`.
- Optional `attention_mask` is rescaled from sample mask to frame mask by `mask[:, ::hop_length]`.

GPU/runtime work:

- Initial runtime can accept precomputed `input_features` and treat the feature extractor as an external pipeline.
- If feature extraction is moved to GPU, implement STFT/Hann/magnitude/mel/log/clamp/normalize as a separate validated pipeline, not as part of the transformer graph.
- Decoder input packing is generation-controller work: prefix tokens may include start-of-transcript, language, task, and no-timestamps/timestamp controls.

Whisper generation has strong tokenizer coupling. `get_decoder_prompt_ids(task, language, no_timestamps=True)` builds forced token IDs from prefix tokens except the start token. Runtime graph parity does not require tokenizer internals, but end-to-end ASR parity does require the same forced IDs and suppress-token processors.

Layout note: the source frontend is channel-first `[B, mel, time]` for Conv1d. Dinoml should keep the semantic translation faithful initially. A layout/fusion pass may lower the frontend to a time-major or channel-last internal layout only if it rewrites Conv1d axes, weight layout, padding/stride semantics, and the following `permute` consumer contract together. The feature extractor and Conv1d stack are good candidates for a conceptual `no_layout_translation()` guard until such a local pass owns all consumers.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Query-scale fold into Q projection

Source pattern:

```text
q = Linear(x, Wq, bq) * head_dim**-0.5
attention(q, k, v, scaling=1.0)
```

Replacement:

```text
q = Linear(x, Wq * scale, bq * scale)
attention(q, k, v, scaling=1.0)
```

Preconditions:

- `scale = head_dim**-0.5` is constant for the layer.
- Q projection has no dynamic post-linear consumers before scaling.
- Attention backend does not also apply `head_dim**-0.5`.

Failure cases:

- Debug/output paths that expose raw Q before scaling.
- Backend APIs that force their own scaling unless disabled.

Parity test sketch: compare eager attention output and logits for one encoder layer and one decoder layer before and after folding in fp32 and bf16/fp16.

### Rewrite: Packed self-attention QKV with mixed K bias

Source pattern:

```text
q = Linear(H -> H, bias=True)(x)
k = Linear(H -> H, bias=False)(x)
v = Linear(H -> H, bias=True)(x)
```

Replacement:

```text
qkv = Linear(H -> 3H, bias=True)(x)
split q,k,v
```

Weight transform:

```python
Wqkv = concat([Wq * q_scale, Wk, Wv], dim=0)
bqkv = concat([bq * q_scale, zeros(H), bv], dim=0)
```

Preconditions:

- Self-attention only; cross-attention K/V source is encoder states, not decoder states.
- Packed projection split order is fixed and tested.
- Zero K bias is supported exactly.

Failure cases:

- Output attentions/debug paths that expect separate module names.
- Tensor parallel or weight-loading code that assumes separate parameters.

### Rewrite: Cross-attention K/V precompute

Source pattern:

```text
for each decoder layer:
  k_cross = k_proj(encoder_hidden_states)
  v_cross = v_proj(encoder_hidden_states)
```

Replacement:

```text
precompute per layer cross K/V once after encoder
reuse for all generated tokens
```

Preconditions:

- Encoder outputs are fixed for the request.
- Cross-attention mask is absent or fixed for the request.
- Cache layout `[B,heads,S,D]` is stable across prefill/decode.

Failure cases:

- Chunked long-form generation where encoder segment changes.
- Batch reduction/reordering in long-form generation unless cache is reordered with batch map.

### Rewrite: Conv1d frontend layout-local lowering

Source pattern:

```text
[B,M,T] -> Conv1d(k=3,pad=1,stride=1) -> GELU -> Conv1d(k=3,pad=1,stride=2) -> GELU -> permute [B,T/2,H]
```

Replacement options:

```text
Option A: keep Conv1d provider kernels and fuse Conv1d+GELU epilogues.
Option B: lower Conv1d windows to GEMM/im2col in a local time-major layout.
```

Preconditions:

- Layout pass owns the entire frontend through the final token layout handoff.
- Padding semantics match PyTorch Conv1d.
- Axis-sensitive ops are rewritten together; source channel dim `1` becomes the optimized channel axis only inside the guarded region.

Weight transform:

```python
# conceptual for GEMM lowering of a Conv1d layer
W_flat = conv.weight.reshape(out_channels, in_channels * kernel_width)
```

Failure cases:

- Dynamic feature lengths that violate the source fixed-length check.
- Partial layout conversion where the following transformer consumes an unexpected token order.

### Rewrite: Last-token LM projection for decode

Source pattern:

```text
logits = MatMul(decoder_hidden[:, :, :], embedding_weight.T)
```

Replacement:

```text
decode_logits = MatMul(decoder_hidden[:, -1:, :], embedding_weight.T)
```

Preconditions:

- Generation step only needs the newest token logits.
- Public forward parity requiring full `[B,T,V]` logits is not being served by this lowered graph.

Failure cases:

- Training/loss path.
- APIs returning full-sequence logits or token timestamp logic that consumes historical scores.

## 10. Kernel fusion candidates

Highest priority:

- Decoder self-attention with KV cache: the steady-state decode bottleneck, MHA shape `[B,A,1,past,D]`.
- Decoder cross-attention with precomputed K/V: every generated token attends to 1500 encoder positions; efficient cross-cache layout matters.
- LayerNorm + QKV projection and MLP fusions: repeated across up to 64 total transformer blocks in large-v3.
- Q scaling + attention backend integration: required for both parity and FlashAttention/SDPA efficiency.
- Last-token-only logits: vocab is about 52K, so avoiding full-sequence projection in decode saves substantial bandwidth/compute.

Medium priority:

- Conv1d + GELU frontend fusion: small relative to the transformer, but easy to isolate and useful for batch throughput.
- Packed QKV self-attention projection with mixed K bias handling.
- Cross-attention K/V projection precompute immediately after encoder.
- Feature-extractor GPU path for high-throughput batch ASR, if CPU preprocessing becomes the bottleneck.

Lower priority:

- Timestamp attention-weight extraction kernels; important for feature completeness but expensive and not needed for first short-form transcription.
- Long-form segmentation scheduling and dynamic batch shrinking.
- Audio classification head fusions.

## 11. Runtime staging plan

Stage 1: config and weight loading

- Parse Whisper config and preprocessor config.
- Load encoder Conv1d, encoder/decoder transformer, tied embeddings, and LM head.
- Stub generation controller with explicit provided `decoder_input_ids`.

Stage 2: feature input contract and audio frontend parity

- Accept precomputed `input_features` `[B,M,3000]`.
- Implement Conv1d/GELU/Conv1d/GELU/permute and sinusoidal position add.

Stage 3: encoder parity

- Run encoder-only parity for one block, then all blocks, using fixed 1500-token outputs.
- Encoder attention mask can be stubbed as unsupported/no-op for primary path.

Stage 4: decoder prefill parity

- Implement decoder embeddings, learned positions, causal mask, self-attn, cross-attn, MLP, and tied LM head.
- Use full decoder prompt tokens supplied by caller.

Stage 5: decode with cache

- Implement `EncoderDecoderCache`: append self K/V and reuse cross K/V.
- Validate one-token decode logits against Transformers after prefill.

Stage 6: generation controller features

- Add forced decoder IDs, language/task/no-timestamps prompt construction, suppress tokens, and begin suppress tokens.
- Defer long-form timestamps until short-form generation is stable.

Stage 7: optimized attention and fusions

- Enable FlashAttention/SDPA-compatible kernels with Q scaling preserved.
- Add packed projections, cross K/V precompute, and last-token logits.

Stage 8: preprocessing integration

- Keep CPU feature extraction as baseline.
- Add optional GPU STFT/mel path if performance probes justify it.

## 12. Parity and validation plan

- Feature extractor parity:
  - Compare CPU/Numpy or Torch STFT log-mel features against Transformers for fixed waveforms.
  - Tolerance: fp32 `atol=1e-5, rtol=1e-5` for feature extraction; looser if using approximate GPU FFT.
- Custom math tests:
  - Sinusoid table parity for `(1500,H)`.
  - Query-scale fold parity for random tensors.
- Frontend parity:
  - Conv1d stack output after `permute` for `[B,M,3000]`.
- Encoder parity:
  - One encoder layer, then all encoder layers, fp32 first.
- Decoder parity:
  - One decoder layer self-attn without cache, cross-attn, then full decoder prefill.
- Cache parity:
  - Prefill then one-token decode, comparing self-cache length growth and cross-cache reuse.
- Logits parity:
  - Full prefill logits and last-token decode logits.
- End-to-end short-form ASR:
  - Same processor output, same forced decoder IDs, greedy decode, compare token IDs and decoded text.
- Recommended tolerances:
  - fp32: `atol=1e-4, rtol=1e-4` for hidden states/logits after full model.
  - fp16/bf16: `atol=2e-2, rtol=2e-2` initially; tighten per kernel after stable parity.

## 13. Performance probes

- CPU feature-extraction throughput: raw seconds processed per second, batched and single-file.
- Optional GPU feature-extraction throughput: isolate STFT/mel/log pipeline cost.
- Audio frontend throughput: Conv1d stack only for `[B,M,3000]`.
- Encoder-only throughput: `[B,1500,H]` for tiny through large-v3.
- Decoder prefill throughput: prompt lengths 4, 16, 64, 224, 448.
- Decode tokens/sec: cross-attn over S=1500, batch-size sweep, with and without cross K/V precompute.
- End-to-end requests/hour for 30-second clips.
- Sequence sweep for long-form segments if timestamp support is enabled.
- KV cache memory usage:
  - self-cache per layer grows with generated tokens.
  - cross-cache per decoder layer is fixed at `[B,A,1500,D]`.
- Attention backend comparison: eager, SDPA, FlashAttention-equivalent; include timestamp/output-attention slow path separately.
- LM head cost: full-sequence logits versus last-token-only logits.

## 14. Skip/defer list

- Training and SpecAugment.
- Gradient checkpointing and LayerDrop behavior.
- Audio classification head and weighted layer-sum classifier.
- Beam search, sampling variants, and speculative decoding beyond greedy first path.
- Long-form transcription with dynamic chunk loops and batch shrinking.
- Word-level timestamps and attention-weight median filtering.
- Full tokenizer implementation; use HF tokenizer or precomputed prompt IDs initially.
- Quantization.
- Multi-GPU tensor parallel.

## 15. Final implementation checklist

- [ ] Parse Whisper config and preprocessor config.
- [ ] Load Conv1d, encoder, decoder, tied embedding/LM-head weights.
- [ ] Accept `input_features` `[B,num_mel_bins,3000]`.
- [ ] Implement Conv1d frontend and final `[B,1500,H]` token handoff.
- [ ] Implement fixed encoder sinusoidal position table.
- [ ] Implement encoder noncausal MHA blocks.
- [ ] Implement decoder learned positions and causal mask.
- [ ] Implement decoder self-attention cache.
- [ ] Implement decoder cross-attention and reusable cross K/V cache.
- [ ] Preserve Whisper query scaling order or fold scale into Q projection.
- [ ] Implement GELU MLP and LayerNorm primitives.
- [ ] Implement tied LM head and last-token decode projection.
- [ ] Add forced decoder prompt IDs and suppress-token logits processors.
- [ ] Add optional CPU feature-extractor parity harness.
- [ ] Add one-block, encoder, decoder prefill, and cached decode parity tests.
- [ ] Benchmark feature extraction, encoder, prefill, decode, cross-attn cache, and LM head.
