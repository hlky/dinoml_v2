# Transformers family audit: higgs_audio_v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: bosonai/higgs-audio-v2-generation-3B-base
Config source: HF raw config.json plus source defaults
Primary target: autoregressive audio-code generation conditioned by text and optional prompt audio
DinoML assumptions: inference-only first, CUDA target, prioritize generator prefill/decode, stage codec separately
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/higgs_audio_v2/configuration_higgs_audio_v2.py`
- `X:/H/transformers/src/transformers/models/higgs_audio_v2/modeling_higgs_audio_v2.py`
- `X:/H/transformers/src/transformers/models/higgs_audio_v2/generation_higgs_audio_v2.py`
- `X:/H/transformers/src/transformers/models/higgs_audio_v2/processing_higgs_audio_v2.py`
- `X:/H/transformers/src/transformers/models/higgs_audio_v2/modular_higgs_audio_v2.py`
- Coupled tokenizer/codec files:
  - `X:/H/transformers/src/transformers/models/higgs_audio_v2_tokenizer/configuration_higgs_audio_v2_tokenizer.py`
  - `X:/H/transformers/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`
  - `X:/H/transformers/src/transformers/models/dac/feature_extraction_dac.py`

HF configs inspected:

- [generation config.json](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/config.json)
- [processor_config.json](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/processor_config.json)
- [generation_config.json](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/generation_config.json)
- [tokenizer_config.json](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/tokenizer_config.json)
- [audio tokenizer config.json](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/config.json)
- [audio tokenizer preprocessor_config.json](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/preprocessor_config.json)

Missing files or assumptions:

- No small/debug checkpoint with `model_type="higgs_audio_v2"` was found in the quick HF/source search. Source defaults are listed separately from checkpoint facts.
- `modeling_higgs_audio_v2.py` and `configuration_higgs_audio_v2.py` are generated from `modular_higgs_audio_v2.py`; future source edits should use the modular file as authoritative.
- End-to-end waveform parity requires a separate audit of `higgs_audio_v2_tokenizer`, DAC, and HuBERT. This report owns the generator and documents the codec boundary.

## 2. High-level architecture

Higgs Audio V2 generation is a text/audio-token causal decoder. It consumes text tokens plus optional prompt audio codes, replaces audio placeholder token embeddings with summed multi-codebook audio embeddings, runs a Llama-like decoder with GQA/RoPE/RMSNorm/SwiGLU, and predicts the next frame of 8 audio codebooks.

Dataflow:

```text
chat/template + optional waveform prompt
  -> text tokenizer + DAC feature extractor + Higgs audio tokenizer encode
  -> processor delay-pattern packing and placeholder expansion
  -> text/audio embedding stitch
  -> causal decoder prefill/decode with KV cache
  -> audio_lm_head logits [B, T, num_codebooks * codebook_size]
  -> per-codebook masking/warping/sampling
  -> generated delayed audio code frames [B, T_audio, num_codebooks]
  -> delay-pattern revert
  -> Higgs audio tokenizer decode/DAC vocoder
  -> waveform
```

Stage decomposition:

- CPU/data pipeline: chat template, text tokenization, waveform input validation, audio feature extraction wrapper, placeholder expansion.
- Separately cacheable prompt-audio stage: waveform -> audio codec codes -> delayed `audio_input_ids`; codes can be cached outside the decoder.
- Generator prefill: stitched text/audio embeddings and causal attention over the full prompt.
- Generator decode: one text placeholder/delay token per audio frame, plus one generated vector of 8 codebook IDs per step.
- Postprocess/vocoder: collect generated audio codes, locate BOS/EOS, revert delay pattern, decode through the audio tokenizer.

First useful DinoML target: text-conditioned audio-code generation through `HiggsAudioV2ForConditionalGeneration.forward` and custom generation loop, returning discrete audio codes. Waveform decode is a later stage unless DinoML explicitly owns the codec.

## 3. Important config dimensions

Main generator dimensions:

| Field | Source default | bosonai generation config | Notes |
| --- | ---: | ---: | --- |
| `vocab_size` | 128256 | 128256 | Text embedding size |
| `hidden_size` | 3072 | 3072 | Decoder width |
| `intermediate_size` | 8192 | 8192 | SwiGLU gate/up width |
| `num_hidden_layers` | 28 | 28 | Decoder blocks |
| `num_attention_heads` | 24 | 24 | Query heads |
| `num_key_value_heads` | 8 | 8 | GQA, 3 query groups per KV head |
| `head_dim` | 128 | 128 | `24 * 128 == 3072` |
| `max_position_embeddings` | 2048 | 2048 | RoPE cache/source max |
| `rope_type` | `llama3` | `llama3` | From `rope_parameters` |
| `rope_theta` | 500000.0 | 500000.0 | Llama-3 RoPE |
| `hidden_act` | `silu` | `silu` | SwiGLU |
| `rms_norm_eps` | 1e-5 | 1e-5 | RMSNorm |
| `attention_bias` | false | false | Q/K/V/O bias absent |
| `mlp_bias` | false | false | FFN bias absent |
| `dtype` | not set | `bfloat16` | HF config fact |
| `use_cache` | true | true | DynamicCache supported |
| `num_codebooks` | 8 | 8 | Audio codebooks generated per step |
| `codebook_size` | 1024 | 1026 | Production config includes stream BOS/EOS IDs |
| `audio_token_id` | 128016 | 128016 | Placeholder |
| `audio_delay_token_id` | 128014 | 128014 | Delay placeholder |
| `audio_stream_bos_id` | 1024 | 1024 | Per-codebook control ID |
| `audio_stream_eos_id` | 1025 | 1025 | Per-codebook control ID |

Representative checkpoint/config sweep:

| Basis | Purpose | Key variation |
| --- | --- | --- |
| Source defaults | random/debug construction only | `codebook_size=1024`; otherwise same topology |
| `bosonai/higgs-audio-v2-generation-3B-base` | production generator | `dtype=bfloat16`, `codebook_size=1026`, `use_text_head=true` in generation config |
| `bosonai/higgs-audio-v2-tokenizer` | required codec checkpoint | separate `model_type=higgs_audio_v2_tokenizer`; sample rate 24000, RVQ codebook size 1024, codec hidden sizes below |

Audio tokenizer dimensions from `bosonai/higgs-audio-v2-tokenizer`:

| Field | Value | Source |
| --- | ---: | --- |
| `sample_rate` | 24000 | tokenizer config |
| `semantic_sample_rate` | 16000 | tokenizer config |
| DAC hop/downsample product | 960 | `[8,5,4,2,3]` |
| `downsample_factor` | 320 | tokenizer config |
| RVQ `codebook_size` | 1024 | tokenizer config |
| `num_quantizers` | 8 inferred | `target_bandwidths[-1]=2`, frame rate 25, 10 bits/code |
| semantic model | HuBERT, 12 layers, hidden 768 | nested config |
| acoustic model | DAC, hidden 256, decoder hidden 1024 | nested config |

## 3a. Family variation traps

- Generator `codebook_size` can differ from codec `codebook_size`: production generator uses 1026 logits per codebook so IDs 1024/1025 are valid stream BOS/EOS controls, while the codec codebook has 1024 real codes.
- The decoder has two FFN/norm paths per layer: text FFN and audio FFN. Lowering must preserve `audio_token_mask` selection or admit an audio-only specialization.
- `audio_token_mask=None` is an audio-only mode. In that case every token uses the audio norms/MLP; this is not the same graph as mixed text/audio prompts.
- Placeholder replacement uses broad `masked_scatter`, but the processor guarantees a stricter ordered expansion pattern. DinoML should lower it under count/order guards, not admit general boolean scatter as the core primitive.
- HF source doc says placeholder count is checked in `get_placeholder_mask`, but the implementation only returns the mask. Count/order validation comes mostly from the processor.
- Attention is GQA with `num_key_value_heads < num_attention_heads`.
- `head_dim` is explicit; do not infer projection widths only from `hidden_size`.
- Only greedy/sample generation modes are supported; beam and most text logits processors are rejected.
- `logits_to_keep=0` means all logits in Python slice semantics. Decode should use `logits_to_keep=1` or an equivalent last-token-only head to avoid full prefill logits.
- Source supports FlashAttention/SDPA/FlexAttention backend dispatch through Transformers, but eager semantics are the parity baseline.
- The audio tokenizer requires `torchaudio` for resampling and composes HuBERT + DAC + RVQ; route codec support to a separate audit if waveform output is in scope.
- No NHWC/NCHW image layout issue exists in the generator. Codec Conv1d tensors are channel-first `[B,C,T]`; layout translation should be guarded and local only.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids) -> [B,S,3072]`.
- Audio embedding: add per-codebook offsets `[0, codebook_size, ...]`, embedding over `8 * codebook_size`, then sum over codebook axis: `[B,T_audio,8] -> [B,T_audio,3072]`.
- Boolean masks from equality against `audio_token_id` and `audio_delay_token_id`.
- Guarded ordered embedding stitch replacing placeholder rows. Source uses `masked_scatter`.
- Boolean indexing/scatter for dual norm/FFN paths in mixed text/audio prompts.
- `view`, `reshape`, `transpose`, `contiguous`, `cat`, `slice`, `arange`, `cumsum`, `pad`, `stack`.
- Last-token or selected-token slicing for logits.

Neural primitives:

- RMSNorm over 3072 with fp32 variance and output cast back to input dtype.
- Linear Q: `3072 -> 3072`, no bias.
- Linear K/V: `3072 -> 1024`, no bias.
- Linear O: `3072 -> 3072`, no bias.
- Text MLP: gate/up `3072 -> 8192`, SiLU, multiply, down `8192 -> 3072`, no bias.
- Audio MLP: same shape as text MLP, separate weights.
- Final RMSNorm.
- Audio LM head: `3072 -> 8 * codebook_size` = 8208 for production config.
- Optional text LM head exists only when `use_text_head=True`; not required for audio generation.

Attention primitives:

- Causal self-attention with GQA: Q heads 24, KV heads 8, head dim 128.
- RoPE on Q/K before cache update.
- KV cache update per layer; cached K/V shape conceptually `[B, 8, T_cache, 128]`.
- Repeat/interleave KV to 24 heads for eager attention, or native GQA backend for optimized attention.
- Additive causal/padding mask from `create_causal_mask`.
- Softmax in fp32 over keys, cast to query dtype.

Position/rotary:

- Llama-3 RoPE via Transformers `ROPE_INIT_FUNCTIONS["llama3"]` for production config.
- Dynamic RoPE update decorator may update cached inverse frequencies for advanced rope types.

Generation/cache ops:

- `DynamicCache(config=config)` creation when `use_cache=True`.
- `prepare_inputs_for_generation` masks already cached prompt audio IDs using cumulative valid counts.
- Decode specialization can pass only latest `audio_input_ids[:, -1:, :]` and omit text `input_ids` when prior audio IDs are cached.
- Delay-pattern logits processor reshapes flat logits to `[B, 8, codebook_size]`, masks stream BOS/EOS by per-codebook delay state, then flattens to `[B*8, codebook_size]` for warpers.
- Supported warpers: temperature, top-k, top-p, inf/nan removal.
- Sampling uses per-codebook softmax + multinomial or argmax.
- RAS repetition avoidance optionally resamples repeated codebook tokens in a recent window.

Preprocessing-coupled ops:

- Text tokenizer with left padding by processor default.
- DAC feature extraction: mono waveform, 24000 Hz, right padding to hop multiple.
- Audio tokenizer encode/decode boundary, described below.

Optional codec/diffusion/vocoder generation ops:

- No diffusion model is in this source.
- Codec decode uses Higgs audio tokenizer, DAC acoustic decoder, and ConvTranspose1d. This is required for waveform parity but can be deferred for first audio-code generation.

## 5. Layer/block breakdown

Input construction:

```text
text_embeds = Embedding(input_ids)                         # [B,S,3072]
audio_embeds = sum(Embedding(audio_ids + offsets), axis=-2) # [B,T_audio,3072]
inputs_embeds[placeholder_mask] = audio_embeds[valid_audio_mask]
position_ids = arange(S_current) + past_seen_tokens
cos,sin = rotary_emb(inputs_embeds, position_ids)
causal_mask = create_causal_mask(...)
```

Decoder block, repeated 28 times:

```text
residual = x
if audio_token_mask is None:
    x_norm = audio_input_rmsnorm(x)
else:
    x_norm[audio] = audio_input_rmsnorm(x[audio])
    x_norm[text] = input_rmsnorm(x[text])

q = Linear(3072 -> 24*128)(x_norm).view(B,S,24,128).transpose(1,2)
k = Linear(3072 ->  8*128)(x_norm).view(B,S, 8,128).transpose(1,2)
v = Linear(3072 ->  8*128)(x_norm).view(B,S, 8,128).transpose(1,2)
q,k = RoPE(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx)
attn = causal_attention_gqa(q,k,v,mask)
x = residual + Linear(3072 -> 3072)(attn)

if audio_token_mask is None:
    x = x + audio_mlp(audio_post_rmsnorm(x))
else:
    x[text] += text_mlp(post_attention_rmsnorm(x[text]))
    x[audio] += audio_mlp(audio_post_attention_rmsnorm(x[audio]))
```

Output head:

```text
x = final_rmsnorm(x)
logits = audio_lm_head(x[:, slice_indices, :]) # [B,T_keep,8*codebook_size]
```

All generator Linear layers are bias-free for the production config.

## 6. Attention requirements

- Variant: causal self-attention only.
- Head structure: GQA, 24 query heads, 8 KV heads, 3 query groups per KV head.
- Head dim: 128.
- Q width: 3072. K/V width: 1024 each. Attention output width before O projection: 3072.
- Mask: Transformers `create_causal_mask` combines causal and caller attention mask.
- Position: RoPE applied to Q/K before cache update.
- Cache: autoregressive KV cache, one K and one V per layer. Store post-RoPE K, unrotated V.
- Prefill: rectangular full prompt attention with causal mask.
- Decode: query length 1 when cache is enabled; source can omit `input_ids` and feed only newest audio codes in decode.
- Backend: source advertises FlashAttention, SDPA, and FlexAttention support. Eager path repeats KV to query heads and uses matmul/softmax/matmul.
- Dropout: `attention_dropout=0.0` in config and inference uses 0.

For DinoML, the optimized target is native GQA causal attention with KV cache and RoPE. A correct fallback can repeat KV to 24 heads, but that is a memory/bandwidth tax.

## 7. Position encoding and custom math

RoPE basis:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Production config uses `rope_type="llama3"` with factor 32, low/high frequency factors 0.125/0.5, original max positions 1024, and theta 500000. DinoML should use Transformers Llama-3 RoPE equations, not only the default snippet above, for the production checkpoint.

RMSNorm:

```python
variance = mean(x.float() ** 2, dim=-1, keepdim=True)
y = weight * (x.float() * rsqrt(variance + eps)).to(input_dtype)
```

Delay pattern packing:

```python
new_len = seq_len + num_codebooks - 1
output[bos lower triangle] = audio_stream_bos_id
output[data diagonal band] = input_codes.flatten()
output[eos upper triangle] = audio_stream_eos_id
```

Delay revert extracts diagonal slices per codebook and concatenates them back to `[T_codes, num_codebooks]`.

## 8. Preprocessing and input packing

Text/control ABI:

- Processor wraps `DacFeatureExtractor`, `AutoTokenizer`, and `HiggsAudioV2TokenizerModel`.
- Text defaults: `padding=True`, `padding_side="left"`, `return_tensors="pt"` required.
- Special strings: `<|AUDIO_OUT|>`, `<|audio_out_bos|>`, `<|audio_eos|>`, `<|reserved_special_token_6|>`.
- Special token IDs in production: audio placeholder 128016, audio BOS 128013, audio delay 128014, EOS 128009, pad 128001.

Prompt audio encode ABI:

- Raw audio must be mono and sampled at 24000 Hz.
- `DacFeatureExtractor` pads right, padding value 0.0, hop length 960 in the tokenizer preprocessor config.
- Processor currently encodes each audio sample one at a time, removes `padding_mask`, moves features to the audio tokenizer device, and calls `audio_tokenizer.encode`.
- Encoded codec output shape from tokenizer: `[B, num_quantizers, codes_length]`, then processor adds BOS/EOS along code time, applies delay pattern, and transposes to `[delayed_length, num_codebooks]` per prompt audio.
- For each `<|AUDIO_OUT|>` in text, the processor expands text to:
  - `num_audio_tokens - (num_codebooks - 1)` copies of `<|AUDIO_OUT|>`
  - followed by `num_codebooks - 1` delay tokens.
- Batched prompt audio IDs are padded with `audio_stream_eos_id`; `audio_input_ids_mask` marks valid delayed frames.

Embedding stitch contract:

- Placeholder mask is `(input_ids == audio_token_id) | (input_ids == audio_delay_token_id)`.
- Source uses `inputs_embeds.masked_scatter(mask[..., None].expand_as(inputs_embeds), audio_embeds)`.
- Safe DinoML lowering: require processor-produced row-major placeholder positions and require `sum(placeholder_mask) == sum(audio_input_ids_mask)` per batch; replace by ordered row copy/gather-scatter, not generic boolean scatter.

Postprocess:

- `batch_decode` finds the last full-codebook stream BOS row, slices after it, finds first full-codebook stream EOS per batch, reverts delay pattern, clips codes to `[0, audio_stream_bos_id - 1]`, transposes to `[1, num_codebooks, T]`, and calls audio tokenizer decode.

Codec boundary:

- `HiggsAudioV2TokenizerModel.encode` resamples 24000 -> 16000 for semantic HuBERT, pads 160 samples, averages all hidden states, optionally downsamples semantic features, runs semantic Conv1d blocks, runs DAC acoustic encoder, concatenates acoustic+semantic features, projects, and RVQ-encodes by Euclidean nearest-codebook search.
- `decode` RVQ-decodes, projects to acoustic hidden size, and runs DAC decoder with adjusted ConvTranspose1d output padding and no final Tanh.
- This is GPU-runtime work only if DinoML chooses to own waveform codec parity. For first generator integration, accept precomputed `audio_input_ids` and return generated `audio_sequences`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ordered audio embedding stitch

Source pattern:

```text
audio_embeds = audio_embed(audio_input_ids)
audio_embeds = audio_embeds[audio_input_ids_mask]
inputs_embeds.masked_scatter(audio_token_mask[...,None].expand_as(inputs_embeds), audio_embeds)
```

Replacement:

```text
Embedding(text) -> EmbeddingSum(audio) -> OrderedRowsCopy(audio rows into placeholder row indices)
```

Preconditions:

- `input_ids` is present and processor-generated.
- Placeholder IDs are only `audio_token_id` or `audio_delay_token_id`.
- Per batch, placeholder count equals valid audio frame count.
- Placeholder order matches flattened valid `audio_input_ids` order.
- Hidden size is contiguous last dimension.

Failure cases:

- Caller supplies arbitrary `inputs_embeds`.
- Missing or inconsistent `audio_input_ids_mask`.
- Audio-only mode with no `input_ids`; route to audio-only specialization.

Parity sketch:

- Random text IDs with synthetic placeholder spans and random audio IDs.
- Compare source masked scatter against indexed copy for mixed batch lengths.

### Rewrite: audio-only decode graph specialization

Source pattern:

```text
if cached and only latest audio frame is needed:
    audio_input_ids = audio_input_ids[:, -1:, :]
    input_ids omitted
    audio_token_mask = None
```

Replacement:

```text
AudioEmbeddingSum(latest_codes) -> all-audio decoder block -> last-token head
```

Preconditions:

- `past_key_values` present.
- Decode step after prefill.
- No new text tokens besides generator-maintained placeholder/delay token.
- `audio_input_ids` has shape `[B,1,8]`.

Failure cases:

- Mixed prompt still being prefetched.
- Cache length/audio mask mismatch.

Parity sketch:

- Prefill once, then decode N steps with source `prepare_inputs_for_generation`; compare logits from full source and specialized input path.

### Rewrite: dual FFN/norm mask split to two compact GEMM batches

Source pattern:

```text
x[text_mask] -> text RMSNorm/MLP
x[audio_mask] -> audio RMSNorm/MLP
```

Replacement:

```text
Gather text rows -> text RMSNorm+SwiGLU -> scatter-add
Gather audio rows -> audio RMSNorm+SwiGLU -> scatter-add
```

Preconditions:

- Static hidden/intermediate dims.
- Row order can be recovered from boolean mask.
- Empty text or audio partition is allowed and skips its branch.

Failure cases:

- Need deterministic row order matching PyTorch boolean indexing.
- General masked updates should remain rejected outside this pattern.

Parity sketch:

- Mixed prompts with all-text, all-audio, and interleaved audio spans; compare per-layer outputs.

### Rewrite: last-token-only audio head

Source pattern:

```text
audio_lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
select last hidden row -> GEMM(3072, 8*codebook_size)
```

Preconditions:

- In generation, only next-token logits are consumed.
- `logits_to_keep=1` or equivalent generation loop guarantee.

Failure cases:

- Training/loss or diagnostics requesting full prefill logits.

Parity sketch:

- Compare last row logits with full head sliced at `[:, -1, :]`.

### Rewrite: per-codebook logits processing

Source pattern:

```text
flat_logits [B, 8*V] -> reshape [B,8,V] -> delay masks -> [B*8,V] -> top-k/top-p/temp
```

Replacement:

```text
structured codebook logits tensor with codebook axis retained
```

Preconditions:

- Generation mode is greedy or sample.
- Logits processors are limited to delay pattern, temperature, top-k, top-p, inf/nan removal.

Failure cases:

- Beam search, repetition penalty, or text-token processors.

Parity sketch:

- Fixed delay-state examples with BOS/EOS positions; verify masked logits and sampled/argmax IDs under seeded RNG.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm, including separate text/audio weights.
- Fused QKV projection for GQA where weight packing preserves split order Q then K then V; source stores separate Linear modules.
- RoPE + GQA causal attention with KV cache.
- SwiGLU MLP: gate/up GEMMs, SiLU, multiply, down GEMM.
- Last-token-only audio LM head.

Medium priority:

- Ordered audio embedding stitch and audio embedding sum over codebooks.
- Dual-branch text/audio FFN row partitioning to avoid general masked scatter.
- Per-codebook logits reshape/mask/top-k/top-p for generation.
- KV cache memory layout optimized for `[layers, B, kv_heads, T, head_dim]`.

Lower priority:

- Full codec encode/decode kernels: HuBERT, DAC, Conv1d/ConvTranspose1d, RVQ nearest-neighbor.
- RAS repetition-avoidance resampling inside generation.
- Optional text LM head for training loss.

## 11. Runtime staging plan

Stage 1: config/weight loader

- Parse generator config, including production `codebook_size=1026`.
- Load text embeddings, audio embeddings, 28 decoder layers, final norm, audio head.
- Reject unsupported remote-code or nonmatching `model_type`.

Stage 2: block-level parity

- Implement RMSNorm, RoPE, GQA attention, SwiGLU, residuals.
- Validate one decoder block for text-only, audio-only, and mixed masks.

Stage 3: generator prefill parity

- Accept precomputed `input_ids`, `audio_input_ids`, `audio_input_ids_mask`.
- Implement ordered audio embedding stitch and full causal prefill logits.

Stage 4: decode with KV cache

- Implement `DynamicCache` equivalent, position offset by cache length, and audio-only latest-frame decode.
- Return last-step audio logits.

Stage 5: generation controller

- Implement greedy/sample, delay-pattern logits processor, top-k/top-p/temperature, EOS/pad behavior, and generated `audio_sequences`.
- Defer beam search because source rejects it.

Stage 6: codec boundary

- Initially accept/return discrete audio codes.
- Later compose `higgs_audio_v2_tokenizer` for waveform decode after its own parity audit.

Stage 7: optimized kernels

- Enable fused RMSNorm, fused SwiGLU, GQA FlashAttention, last-token head, and structured codebook logits.

## 12. Parity and validation plan

- Config parsing tests:
  - Source-default config and production config.
  - Assert generator `codebook_size` and tokenizer `codebook_size` are separate.
- Custom op tests:
  - RMSNorm fp32/fp16/bf16 tolerance.
  - Llama-3 RoPE against Transformers.
  - Audio embedding sum with codebook offsets.
  - Delay-pattern build/revert and logits masking.
- Single-layer parity:
  - Random hidden states for text-only, audio-only, and mixed masks.
  - Compare after attention and after MLP.
- Prefill parity:
  - Synthetic prompt with placeholder spans and precomputed audio IDs.
  - Compare final hidden state and full/last audio logits.
- Decode parity:
  - Prefill prompt, then run 3-8 decode steps with cache.
  - Compare generated code IDs under greedy; compare logits before sampling for stochastic mode.
- End-to-end audio-code parity:
  - Processor-created prompt with fixed precomputed audio prompt codes.
  - Compare `audio_sequences` to HF greedy output.
- Waveform parity, later:
  - Decode known code sequences through Higgs tokenizer; compare waveform length and values.

Suggested tolerances: fp32 `1e-4` absolute for block math; bf16/fp16 `1e-2` to `3e-2` around attention/MLP unless fused attention changes accumulation. Generation ID parity should use greedy first.

## 13. Performance probes

- Processor throughput: text tokenization, placeholder expansion, and audio prompt tokenization separately.
- Prefill throughput by prompt length and number of prompt audio frames.
- Decode tokens/sec by batch size, where one token means one 8-codebook audio frame.
- KV cache memory: 28 layers * 2 * B * 8 KV heads * T * 128 * dtype bytes.
- Attention backend comparison: eager repeat-KV vs native GQA FlashAttention/SDPA.
- Audio head cost: full prefill logits vs last-token-only logits.
- Dual FFN mask distribution: all-audio decode, mixed prompt prefill, text-heavy prompt.
- Per-codebook sampling overhead: top-k/top-p/temperature and RAS on/off.
- Codec probes, later: audio tokenizer encode, RVQ nearest-neighbor, DAC decode, waveform samples/sec.

## 14. Skip/defer list

- Training losses, gradient checkpointing, and text LM head loss.
- Beam search and unsupported text logits processors.
- Multi-GPU tensor parallel; config only provides TP plan metadata.
- Full waveform codec parity for first generator milestone.
- Audio prompt waveform encode inside DinoML; accept precomputed prompt audio codes first.
- RAS repetition avoidance can be disabled for first greedy parity, then added for sampling parity.
- FlexAttention-specific behavior beyond source eager/SDPA parity.
- General boolean scatter/gather public ops beyond the guarded audio stitch and dual-FFN patterns.

## 15. Final implementation checklist

- [ ] Parse `HiggsAudioV2Config` and reject non-`higgs_audio_v2` configs.
- [ ] Keep generator and codec `codebook_size` contracts separate.
- [ ] Load text embedding, audio embedding, decoder, final norm, and audio head weights.
- [ ] Implement audio codebook offset embedding sum.
- [ ] Implement guarded ordered placeholder audio embedding stitch.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement Llama-3 RoPE for production config.
- [ ] Implement GQA causal attention with KV cache.
- [ ] Implement text/audio dual FFN branches with guarded row partitioning.
- [ ] Implement last-token-only audio head for decode.
- [ ] Implement delay-pattern logits processor.
- [ ] Implement greedy generation over `[B, num_codebooks]` audio frames.
- [ ] Add block parity tests for text-only, audio-only, and mixed masks.
- [ ] Add prefill logits parity with synthetic audio placeholders.
- [ ] Add decode KV-cache parity for several generated frames.
- [ ] Add generation parity for greedy audio code IDs.
- [ ] Stage a separate `higgs_audio_v2_tokenizer` audit before waveform decode parity.
