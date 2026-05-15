# Transformers MusicGen Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: facebook/musicgen-small.
  Representative configs: facebook/musicgen-small, facebook/musicgen-medium,
  facebook/musicgen-large, facebook/musicgen-stereo-small,
  facebook/musicgen-stereo-medium, facebook/musicgen-stereo-large.

Config source:
  Official Hugging Face repo raw files fetched under
  H:/dinoml_v2/agents/plans/transformers/musicgen/_sources/.
  Files fetched per checkpoint: config.json, generation_config.json,
  preprocessor_config.json, tokenizer_config.json.

Source files inspected:
  transformers/src/transformers/models/musicgen/modeling_musicgen.py
  transformers/src/transformers/models/musicgen/configuration_musicgen.py
  transformers/src/transformers/models/musicgen/processing_musicgen.py
  transformers/src/transformers/models/musicgen/convert_musicgen_transformers.py
  transformers/src/transformers/models/encodec/modeling_encodec.py
  transformers/src/transformers/models/encodec/configuration_encodec.py
  transformers/src/transformers/models/encodec/feature_extraction_encodec.py
  transformers/tests/models/musicgen/test_modeling_musicgen.py
  transformers/tests/models/musicgen/test_processing_musicgen.py
  transformers/tests/models/encodec/test_modeling_encodec.py

Any missing files or assumptions:
  No remote-code files are required for the standard facebook/musicgen checkpoints.
  This report targets MusicgenForConditionalGeneration for text-conditioned and
  audio-prompt-conditioned music generation, including EnCodec decode to waveform.
  Text encoder internals are T5 and should compose the separate T5 audit for full
  operator coverage; this report owns MusicGen-specific decoder, audio codebook,
  EnCodec coupling, generation controller, and staged runtime boundaries.
  DinoML assumptions: inference-only first, CUDA GPU target, batch throughput
  prioritized, and layout translation only through guarded local rewrites.
```

## 2. High-level architecture

MusicGen is a multi-stage audio generation system:

```text
text/audio preprocessing
  -> T5 text encoder and optional EnCodec audio-prompt encoder
  -> MusicGen causal decoder over delayed EnCodec RVQ token streams
  -> logits processors / sampling
  -> delay-pattern unmasking
  -> EnCodec decoder
  -> waveform
```

Stage decomposition:

- CPU/data pipeline: T5 tokenization, EnCodec feature extraction, waveform shape validation, padding, optional prompt audio truncation/padding.
- Text conditioning: T5 encoder returns `[B, text_seq, 768]`; if the decoder hidden size differs, MusicGen adds `enc_to_dec_proj: Linear(768 -> H)`.
- Audio prompt conditioning: EnCodec encodes `input_values [B, C, samples]` to audio codes. For generation, chunking must produce exactly one frame.
- Decoder prefill/decode: input token ids are shaped `[B * num_codebooks, T]`, reshaped to `[B, num_codebooks, T]`, embedded by one embedding table per codebook, summed, position-embedded, and decoded autoregressively.
- Generation controller: builds a delayed codebook pattern mask, applies classifier-free guidance by batch duplication and a logits processor, restricts generation modes to greedy or sampling, and filters pad/BOS tokens out of final code streams.
- Audio decode: generated codes are reshaped to `[1, B, num_codebooks, T]` and decoded by EnCodec. Stereo models split interleaved codebooks into left/right mono decode passes and concatenate channels.

Independently cacheable or testable stages:

- T5 encoder output can be cached independently of decoder generation.
- EnCodec prompt encoding can be cached as decoder prompt ids.
- Cross-attention K/V for text encoder states are cached after first decode step.
- EnCodec waveform decode can be validated from generated or fixture audio codes without running the text encoder or decoder.

## 3. Important config dimensions

Worked example: `facebook/musicgen-small`.

| Field | Value | Source |
|---|---:|---|
| primary target | text/audio-conditioned music generation | report scope |
| text encoder | T5 encoder, `t5-base`-style | config.json |
| text hidden / layers / heads | 768 / 12 / 12 | config.json |
| text FFN / activation | 3072 / ReLU | config.json |
| text vocab / max length | 32128 / 512 tokenizer max | config/tokenizer |
| audio encoder | EnCodec 32 kHz | config.json |
| audio sampling rate | 32000 | config/preprocessor |
| processor waveform layout | mono `[samples]` or stereo `[2, samples]`; batched to `[B, C, samples]` | feature extractor |
| EnCodec hidden size | 128 | config.json |
| EnCodec codebook size | 2048 | config.json |
| EnCodec bits per codebook | 11 | inferred from codebook size |
| EnCodec upsampling ratios / hop length | `8,5,4,4` / 640 | config + source property |
| EnCodec frame rate | 50 fps | inferred `ceil(32000 / 640)` |
| EnCodec target bandwidth | 2.2 kbps | config.json |
| EnCodec quantizers used | 4 | inferred `floor(2200 / (50 * 11))` |
| decoder hidden / layers / heads | 1024 / 24 / 16 | config.json |
| decoder head dim | 64 | inferred `hidden_size / heads` |
| decoder FFN / activation | 4096 / GELU | config/source |
| decoder vocab size | 2048 audio codes | config.json |
| decoder embedding rows | 2049 per codebook | source `vocab_size + 1` |
| decoder pad/BOS/start token | 2048 | config/generation_config |
| decoder codebooks | 4 mono, 8 stereo | config sweep |
| max decoder positions | 2048 | config.json |
| generation max length | 1500 audio tokens | generation_config |
| default guidance scale | 3.0 | generation_config |
| default generation mode | sampling | generation_config |
| cache support | `EncoderDecoderCache(DynamicCache, DynamicCache)` | source |

Representative checkpoint sweep:

| Model id | Decoder H | Layers | Heads | FFN | Codebooks | Decoder channels | Processor channels | Generation length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| facebook/musicgen-small | 1024 | 24 | 16 | 4096 | 4 | default mono | 1 | 1500 |
| facebook/musicgen-medium | 1536 | 48 | 24 | 6144 | 4 | default mono | 1 | 1500 |
| facebook/musicgen-large | 2048 | 48 | 32 | 8192 | 4 | default mono | 1 | 1500 |
| facebook/musicgen-stereo-small | 1024 | 24 | 16 | 4096 | 8 | 2 | 2 | 1500 |
| facebook/musicgen-stereo-medium | 1536 | 48 | 24 | 6144 | 8 | 2 | 2 | 1500 |
| facebook/musicgen-stereo-large | 2048 | 48 | 32 | 8192 | 8 | 2 | 2 | 1500 |

Config/default notes:

- Checkpoint `decoder.audio_channels` is omitted for mono configs; source default is `1`.
- Checkpoint `audio_encoder.num_quantizers` is not serialized; source computes it from target bandwidth, frame rate, and codebook bits.
- Checkpoint `decoder.add_cross_attention` is false in config, but `MusicgenDecoderLayer` always constructs a cross-attention module and `MusicgenForConditionalGeneration` passes encoder states. Do not use that flag to remove cross-attention in this class.
- Checkpoint `decoder.decoder_start_token_id` can be null; generation_config supplies `decoder_start_token_id=2048`.

## 3a. Family variation traps

- Mono vs stereo is a structural decoder difference: mono uses 4 codebooks; stereo uses 8 interleaved codebooks `[L0, R0, L1, R1, ...]`.
- The EnCodec audio encoder config remains mono (`audio_channels=1`) even for stereo checkpoints. Stereo generation encodes/decode left and right channels separately using the same mono codec.
- Decoder token ids include real code ids `0..2047` plus pad/BOS/start id `2048`; embedding tables have 2049 rows, but LM heads project only to 2048 logits.
- Input ids are flattened as `[B * num_codebooks, T]` at public decoder boundaries, but almost every MusicGen-specific rule operates on `[B, num_codebooks, T]`.
- Delay-pattern masking changes sequence layout. A naive autoregressive loop over unshifted codebooks will not match source behavior.
- CFG is controller-level batch duplication plus `ClassifierFreeGuidanceLogitsProcessor`; the model graph itself does not have a CFG branch.
- MusicGen restricts generation to greedy or sampling; beam search is rejected by the overridden generate methods.
- Audio prompt encode in `MusicgenForConditionalGeneration` requires `frames == 1`; chunked EnCodec prompt inputs are rejected for this runtime path.
- Text encoder is T5; full text coverage includes T5 relative position bias and T5 layer norm semantics, but those are composed from the T5 family rather than MusicGen-owned.
- EnCodec uses 1D convolution, transposed convolution, ELU, LSTM, residual blocks, weight norm, optional reflect padding, RVQ embedding lookup/sum, and optional overlap-add. This is not a transformer branch.
- Layout translation should be guarded. EnCodec source tensors are NCL `[B, C, T]`; decoder/text tensors are token-major `[B, T, H]`. Do not apply a blanket NCHW/NHWC-style pass across the waveform codec boundary.

## 4. Operator coverage checklist

Tensor/layout ops:

- Reshape/view `[B * Cb, T] <-> [B, Cb, T]`.
- Transpose/permute for EnCodec LSTM `[B, C, T] -> [T, B, C] -> [B, C, T]`.
- Codebook interleave/deinterleave for stereo: assign even rows to left, odd rows to right.
- Boolean masks, triangular masks, nonzero/min for delay-pattern start detection.
- `torch.where` for delay-pattern application.
- Index select/gather for sinusoidal positions.
- Pad, slice, concatenate, repeat, repeat_interleave, stack.
- Final filtering `output_ids != pad_token` followed by reshape to `[B, codebooks, T]`.

Neural network primitives:

- T5 encoder primitives: token embedding, T5 layer norm, self-attention with relative position bias, ReLU FFN. Owned by T5 audit.
- Optional `enc_to_dec_proj: Linear(768 -> H)` when text hidden size differs from decoder hidden size.
- Decoder embedding table per codebook: `num_codebooks` x `Embedding(2049 -> H)`, summed.
- Sinusoidal positional embedding add.
- Decoder LayerNorm, residual add, dropout disabled at inference.
- Decoder MLP: `Linear(H -> 4H, bias=False) -> GELU -> Linear(4H -> H, bias=False)`.
- LM heads: one `Linear(H -> 2048, bias=False)` per codebook, stacked and reshaped to `[B * Cb, T, 2048]`.
- EnCodec Conv1d/ConvTranspose1d with weight norm, ELU, GroupNorm if `norm_type=time_group_norm`, LSTM, residual blocks.
- EnCodec RVQ: Euclidean distance argmax for encode, embedding lookup per quantizer, residual subtraction, quantized sum for decode.

Attention primitives:

- Decoder causal self-attention, MHA only, no GQA/MQA.
- Decoder cross-attention over text encoder states.
- Attention projections are bias-free in decoder layers.
- SDPA/Flash/Flex attention are advertised by source flags; eager path is matmul, additive mask, softmax, dropout, matmul.

Generation/cache ops:

- `EncoderDecoderCache` with separate self-attention and cross-attention `DynamicCache`.
- Cross-attention cache update-once per layer via `is_updated[layer_idx]`.
- Delay-pattern mask build/apply.
- Classifier-free guidance logits processor.
- Greedy/sample generation controller, top-k/top-p/temperature inherited from generation utilities if configured.

Preprocessing-coupled ops:

- EnCodec feature extraction validates sampling rate and mono/stereo shape, pads by longest/max length, emits `input_values [B, C, T]` and `padding_mask`.
- MusicgenProcessor delegates text to T5 tokenizer and audio to EnCodec feature extractor.
- `batch_decode(audio=...)` strips padded waveform samples based on padding mask.

## 5. Layer/block breakdown

Text conditioning stage:

```text
input_ids [B, S_text] + attention_mask [B, S_text]
  -> T5 encoder -> encoder_hidden_states [B, S_text, 768]
  -> optional Linear(768 -> H)
  -> masked zeroing with attention_mask[..., None]
```

Decoder input embedding:

```text
decoder_input_ids [B * Cb, T]
  -> reshape [B, Cb, T]
  -> sum over codebook embeddings Embedding(2049 -> H)
  -> add sinusoidal positions [T, H]
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = LayerNorm(x)
x = causal self-attention(q/k/v/o Linear(H -> H), bias=False, cache)
x = residual + x

residual = x
x = LayerNorm(x)
x = cross-attention(q from x, k/v from encoder states, bias=False, cache)
x = residual + x

residual = x
x = LayerNorm(x)
x = Linear(H -> ffn_dim, bias=False)
x = GELU(x)
x = Linear(ffn_dim -> H, bias=False)
x = residual + x
```

LM head:

```text
hidden [B, T, H]
  -> per-codebook Linear(H -> 2048, bias=False)
  -> stack [B, Cb, T, 2048]
  -> reshape [B * Cb, T, 2048]
```

EnCodec decode from generated tokens:

```text
audio_codes [1, B, Cb, T]
  mono: decode all codebooks with mono EnCodec
  stereo: decode even codebooks as left, odd codebooks as right, concat channel dim
  -> waveform [B, audio_channels, samples]
```

## 6. Attention requirements

Decoder attention:

- Type: causal self-attention plus noncausal cross-attention.
- Heads: MHA; `num_key_value_heads` does not exist.
- Head dim: 64 for all swept checkpoints.
- Self-attention cache shape per layer before any backend packing: keys and values `[B, heads, T_cache, 64]`.
- Cross-attention cache shape per layer: keys and values `[B, heads, S_text, 64]`; stored after projection of text encoder states and reused after first update.
- Cached keys are plain projected keys; no RoPE or ALiBi is applied.
- Masking: self-attention uses `create_causal_mask`; cross-attention uses `create_bidirectional_mask` from Transformers masking utilities. Source eager attention adds mask before softmax.
- SDPA/Flash/Flex can be selected through Transformers attention interfaces; parity should preserve scaling `head_dim ** -0.5`, additive mask order, softmax dim, and output projection.

T5 text encoder attention:

- Required for text-conditioned generation but not MusicGen-specific.
- Uses T5 encoder self-attention with relative position bias; no KV cache is needed for encoder-only use.

No sliding-window/local attention, RoPE, ALiBi, GQA, MQA, or packed varlen metadata is used by MusicGen decoder source.

## 7. Position encoding and custom math

MusicGen decoder uses fixed sinusoidal positions, not RoPE:

```python
def musicgen_sinusoidal_positions(seq_len, hidden, past_len):
    half = hidden // 2
    inv = exp(arange(half) * -(log(10000) / (half - 1)))
    pos = arange(seq_len) + past_len
    angles = pos[:, None] * inv[None, :]
    emb = concat([cos(angles), sin(angles)], dim=1)
    if hidden % 2:
        emb = concat([emb, zeros(seq_len, 1)], dim=1)
    return emb
```

The position table can be precomputed up to `max_position_embeddings`, but the source expands it if a longer sequence is requested. Position ids depend on `past_key_values.get_seq_length()` during decode.

Delay-pattern codebook layout is the main custom math:

```python
def apply_musicgen_delay(ids, mask):
    return where(mask[..., : ids.shape[-1]] == -1, ids, mask[..., : ids.shape[-1]])
```

The full mask construction is shape/control-flow heavy: it shifts each codebook by its codebook index, pads BOS/EOS regions with token 2048, duplicates rows interleaved for stereo, and trims the decoder prompt to the first `-1` prediction slot.

EnCodec RVQ decode custom math:

```python
quantized = 0
for i, code_indices in enumerate(codes):  # quantizer axis
    quantized = quantized + embedding(code_indices, codebook[i])
```

EnCodec encode uses Euclidean nearest-codebook argmax; first integration can avoid encode by accepting decoder prompt ids or generated ids, but audio-prompt parity needs it.

## 8. Preprocessing and input packing

Text preprocessing:

- Tokenizer: T5Tokenizer, pad token `<pad>`, EOS `</s>`, unk `<unk>`, model max length 512.
- Processor forwards text args to tokenizer and audio args to EnCodec feature extractor.
- GPU graph input for text-conditioned generation is `input_ids [B, S_text]` and `attention_mask [B, S_text]`.

Audio preprocessing:

- Feature extractor type: `EncodecFeatureExtractor`.
- Sampling rate: 32000 for inspected MusicGen checkpoints.
- Mono feature size 1 for mono models, stereo feature size 2 for stereo models.
- Raw mono input accepted as `(num_samples,)`; stereo as `(2, num_samples)`. Batched examples are converted and returned as `[B, C, T]`.
- Default padding is enabled when `padding is None`; padding mask is returned as `padding_mask`.
- No STFT/mel extraction is used; this is waveform codec preprocessing.
- Chunking is null in inspected MusicGen configs. If chunking is enabled, the MusicGen generation path rejects prompt audio when EnCodec returns more than one frame.

Audio-prompt packing:

- Mono: `audio_encoder.encode(input_values)` returns `audio_codes [frames, B, 4, T]`; generation requires `frames == 1`; decoder prompt ids are `audio_codes[0].reshape(B * 4, T)`.
- Stereo: source requires `input_values.shape[1] == 2`, encodes left and right separately with mono EnCodec, allocates `[frames, B, 8, T]`, assigns left to `::2`, right to `1::2`, then flattens to `[B * 8, T]`.
- Conditional text and audio prompt can be staged separately: text encoder output conditions cross-attention, audio prompt ids seed the decoder sequence.

Generation-controller behavior required for parity:

- `decoder_start_token_id`, `bos_token_id`, and `pad_token_id` are 2048 in generation configs.
- `guidance_scale=3.0` by default; CFG duplicates decoder inputs and appends a null text encoder state batch.
- Only greedy and sampling are accepted. Beam search should be rejected or routed to a separate controller.
- Final ids are delay-mask-applied, pad tokens filtered, reshaped to codebooks, and decoded to waveform.

## 9. Graph rewrite / lowering opportunities

### Rewrite: codebook embeddings -> summed embedding gather

Source pattern:

```text
sum(Embedding_i(input[:, i]) for i in codebooks)
```

Replacement:

```text
Cb independent embedding gathers -> reduction sum over Cb
```

Preconditions:

- `input_ids` are reshaped to `[B, Cb, T]`.
- Each codebook has a distinct embedding table of shape `[2049, H]`.
- Do not tie or merge tables unless weights are proven identical.

Shape equations:

- Output `[B, T, H]`.

Failure cases:

- Decoder inputs supplied as `inputs_embeds`; then gathers are bypassed.

Parity test sketch:

- Random ids in `0..2048`; compare separate gathers plus sum to source decoder embedding path.

### Rewrite: per-codebook LM heads -> batched GEMM family

Source pattern:

```text
stack([Linear_i(hidden) for i in Cb], dim=1)
```

Replacement:

```text
grouped or strided batched GEMM over Cb independent weights
```

Preconditions:

- All heads are bias-free `Linear(H -> 2048)`.
- Same hidden input `[B, T, H]` feeds every codebook head.
- Preserve output orientation `[B, Cb, T, 2048]` before flattening.

Failure cases:

- Weight aliasing is not implied; each head has separate parameters.

Parity test sketch:

- Compare grouped implementation against PyTorch stack for mono Cb=4 and stereo Cb=8.

### Rewrite: decoder attention to fused SDPA/FlashAttention

Source pattern:

```text
q,k,v = Linear(x)
scores = q @ k.T * scale + mask
probs = softmax(scores)
out = probs @ v
out = Linear(out)
```

Replacement:

```text
fused attention backend with cache-aware self/cross modes
```

Preconditions:

- MHA only, `H % heads == 0`.
- No RoPE or relative bias in MusicGen decoder.
- Additive causal/cross masks must match Transformers masking utilities.
- Cross-attention cache is update-once and then reused.

Failure cases:

- T5 encoder attention is different and must use T5-specific relative bias handling.

Parity test sketch:

- Single-layer decoder with fixed masks, then prefill/decode cache parity over multiple steps.

### Rewrite: EnCodec RVQ decode -> fused codebook gather-sum

Source pattern:

```text
for quantizer: embedding(indices, codebook_i); sum
```

Replacement:

```text
fused multi-codebook embedding gather and sum to [B, hidden, T]
```

Preconditions:

- Decode-only path.
- `codes` axis order normalized to `[num_quantizers, B, T]` per mono channel.
- Codebook size and dim fixed from config.

Failure cases:

- Encode path needs Euclidean nearest-neighbor quantization and residual subtraction, not just gather.

Parity test sketch:

- Random code ids, compare quantized embeddings before EnCodec decoder conv stack.

### Rewrite: EnCodec Conv1d regions to optimized NCL 1D kernels

Source pattern:

```text
reflect/asymmetric pad -> weight-norm Conv1d/ConvTranspose1d -> optional GroupNorm -> ELU
```

Replacement:

```text
pre-materialize weight-norm weights for inference, then NCL Conv1d kernels
```

Preconditions:

- Inference-only; weight-norm parametrization can be folded into static conv weights.
- Preserve source NCL layout `[B, C, T]`.
- Padding math must match causal/noncausal, reflect handling for short inputs, and transposed-conv trimming.

Failure cases:

- Dynamic tiny input lengths where reflect padding inserts temporary right zeros need exact handling.
- General Conv1d-to-Linear rewrite is not safe because kernels are overlapping and padded.

Parity test sketch:

- Per-layer EnCodec conv parity on varied lengths, including short length <= pad.

## 10. Kernel fusion candidates

Highest priority:

- Decoder LayerNorm + GEMM boundaries: repeated 24/48-layer decoder hot path.
- Causal self-attention with KV cache: decode tokens/sec bottleneck.
- Cross-attention cache projection: project text encoder K/V once and reuse.
- Per-codebook LM heads as grouped GEMM: 4 or 8 independent projections over the same hidden states.
- Delay-pattern mask apply/filter as controller/runtime primitive: required for correct generated code layout.

Medium priority:

- Codebook embedding gather-sum: small but called every prefill/decode step.
- EnCodec RVQ gather-sum and decoder Conv1d/ConvTranspose1d stack: important for end-to-end waveform latency after token generation.
- T5 encoder GEMM/attention/norm via T5 coverage: amortized over request, cacheable across decode.
- `enc_to_dec_proj` fused with encoder mask zeroing when `H != 768`.

Lower priority:

- Audio prompt EnCodec encode: optional for first text-to-music path, but needed for audio continuation.
- EnCodec overlap-add/chunking: inspected configs do not use chunking for MusicGen generation.
- Training loss and codebook-wise cross entropy.

## 11. Runtime staging plan

Stage 1: config and controller skeleton.

- Parse composite `MusicgenConfig`, nested T5, EnCodec, and decoder configs.
- Load/generate input fixtures with `encoder_outputs` supplied, bypassing T5 and EnCodec initially.
- Implement delay-pattern mask build/apply on CPU or runtime helper for parity.

Stage 2: decoder-only token parity.

- Run `MusicgenForCausalLM` style decoder from audio token ids to logits.
- Validate codebook embeddings, sinusoidal positions, LayerNorm, attention, cross-attention, MLP, LM heads.
- Support prefill without optimized cache first.

Stage 3: cached decode parity.

- Implement self-attention KV cache and cross-attention update-once cache.
- Add greedy and sampling-loop parity with logits processors stubbed or delegated.

Stage 4: text-conditioned composite.

- Compose T5 encoder audit implementation or accept precomputed encoder states.
- Add `enc_to_dec_proj` for hidden-size mismatch and attention-mask zeroing.
- Add CFG batch duplication and `ClassifierFreeGuidanceLogitsProcessor` parity.

Stage 5: EnCodec decode to waveform.

- Implement RVQ decode gather-sum and EnCodec decoder Conv1d/ConvTranspose1d/LSTM stack.
- Validate mono waveform decode from generated codes.
- Add stereo split/decode/concat.

Stage 6: audio-prompt continuation.

- Implement EnCodec encode path or allow precomputed audio codes as prompt input.
- Add mono/stereo prompt packing and one-frame rejection behavior.

Stage 7: optimized kernels and batching.

- Add fused attention, grouped LM heads, codebook gather-sum, and staged caching.
- Add batch-size and codebook-count scheduling for mono/stereo.

## 12. Parity and validation plan

- Config parsing tests for all six swept checkpoints, including omitted mono `decoder.audio_channels` default and computed EnCodec quantizers.
- Delay-pattern unit tests for mono Cb=4 and stereo Cb=8, with prompt length 1, prompt length >1, and `max_length < 2 * channel_codebooks - 1`.
- Codebook embedding-sum parity against source random ids.
- Single decoder layer parity with random hidden states, encoder states, masks, and no cache.
- Full decoder prefill logits parity for small test config and `facebook/musicgen-small` shapes where feasible.
- Cached decode parity: one prefill plus several single-token decode steps, checking self-cache length and cross-cache reuse.
- CFG parity: compare conditional/unconditional batch duplication and final guided logits from the processor.
- EnCodec RVQ decode parity from random valid audio codes, before and after decoder waveform stack.
- Mono end-to-end text generation smoke: short `max_new_tokens`, compare code ids or waveform within tolerance.
- Stereo end-to-end smoke: verify interleaved codebook split and output shape `[B, 2, samples]`.

Recommended tolerances:

- fp32 decoder logits: `rtol=1e-4`, `atol=1e-4` for unfused eager parity.
- fp16/bf16 optimized decoder: start `rtol=1e-2`, `atol=1e-2`, tighten per kernel.
- Waveform decode: use both sample-wise tolerance and perceptual/aggregate checks because ConvTranspose/LSTM accumulated differences can grow.

## 13. Performance probes

- Processor throughput: T5 tokenization plus EnCodec feature extraction separately from GPU runtime.
- T5 encoder throughput by text length and batch size.
- Decoder prefill latency by prompt token length, codebook count 4 vs 8, and hidden size 1024/1536/2048.
- Decode tokens/sec with KV cache by batch size and codebook count.
- Per-codebook LM head time and grouped-GEMM speedup.
- CFG overhead: guidance off vs guidance scale 3.0.
- EnCodec decode latency from `[1, B, Cb, T]` codes to waveform for mono/stereo.
- KV cache memory by model size and generation length 1500.
- End-to-end requests/hour split into text encode, token generation, and waveform decode so codec cost does not hide decoder bottlenecks.

## 14. Skip/defer list

- Training, labels, and codebook-wise cross entropy.
- Gradient checkpointing and LayerDrop behavior.
- Beam search; source generation rejects non-greedy/non-sampling modes.
- Chunked EnCodec prompt generation and overlap-add unless targeting nonstandard configs.
- EnCodec audio encode for first text-to-music milestone; allow precomputed prompt ids or no prompt.
- Full T5 implementation inside this report; compose T5 family coverage.
- Quantization, speculative decoding, multi-GPU tensor parallel, and continuous batching policy.
- Remote-code or MusicGen Melody variants; audit separately if targeted.

## 15. Final implementation checklist

- [ ] Parse composite MusicGen config and nested T5/EnCodec/decoder configs.
- [ ] Load decoder weights: codebook embeddings, decoder blocks, per-codebook LM heads.
- [ ] Load or compose T5 encoder weights and optional `enc_to_dec_proj`.
- [ ] Implement flattened `[B * codebooks, T]` decoder boundary.
- [ ] Implement MusicGen sinusoidal positional embedding with cache offset.
- [ ] Implement delay-pattern mask build/apply and final pad-token filtering.
- [ ] Implement decoder prefill attention and cross-attention.
- [ ] Implement `EncoderDecoderCache` semantics, including cross-attention update-once reuse.
- [ ] Implement classifier-free guidance controller and logits processor parity.
- [ ] Implement grouped per-codebook LM head lowering.
- [ ] Implement EnCodec RVQ decode gather-sum.
- [ ] Implement EnCodec decoder Conv1d/ConvTranspose1d/LSTM stack for waveform output.
- [ ] Add stereo codebook interleave/deinterleave and left/right decode.
- [ ] Add audio prompt packing from precomputed or EnCodec-produced codes.
- [ ] Add parity tests for delay masks, decoder logits, cached decode, CFG, RVQ decode, and waveform decode.
- [ ] Benchmark text encoder, prefill, decode, LM heads, EnCodec decode, and end-to-end generation.
