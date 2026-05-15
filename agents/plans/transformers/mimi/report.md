# Transformers Mimi Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary native checkpoint: kyutai/mimi.
  Additional native-shape config snapshots: onnx-community/kyutai-mimi-ONNX,
  jingyaogong/mimi, shreeharsha/mimi_kyutai, FILM6912/encodec_24khz,
  Cnam-LMSSC/mimi_throat_microphone, biubiu2/HEAR-mimi, Cronkllr/mimi.

Config source:
  Official and mirror Hugging Face raw files snapshotted under
  H:/dinoml_v2/agents/plans/transformers/mimi/_sources/.
  Files fetched where available: config.json, preprocessor_config.json, README.md.

Source files inspected:
  transformers/src/transformers/models/mimi/modeling_mimi.py
  transformers/src/transformers/models/mimi/configuration_mimi.py
  transformers/src/transformers/models/mimi/convert_mimi_checkpoint_to_pytorch.py
  transformers/docs/source/en/model_doc/mimi.md
  transformers/tests/models/mimi/test_modeling_mimi.py
  transformers/src/transformers/models/encodec/feature_extraction_encodec.py

Any missing files or assumptions:
  There is no separate Mimi feature extractor in Transformers; Mimi examples load
  EncodecFeatureExtractor with 24 kHz mono waveform settings. No trust_remote_code
  file is required for kyutai/mimi. This report targets inference-only audio codec
  encode/decode for CUDA, with CPU/data-pipeline waveform padding retained outside
  the first GPU graph. Search results include many downstream "mimi" speech/text
  systems and ONNX exports; those are not separate native Mimi topologies unless
  their config uses model_type="mimi" and MimiModel.
```

## 2. High-level architecture

Mimi is a neural audio codec, not an autoregressive text model:

```text
raw waveform preprocessing
  -> causal SEANet Conv1d encoder [B,C,T] -> [B,512,T/960]
  -> causal transformer encoder over frames [B,T,512]
  -> stride-2 downsample Conv1d to 12.5 Hz
  -> split residual vector quantizer -> audio codes [B,K,T_codes]
  -> RVQ decode / embedding sum
  -> grouped ConvTranspose1d upsample to 25 Hz
  -> causal transformer decoder over frames [B,T,512]
  -> SEANet ConvTranspose1d decoder -> reconstructed waveform [B,C,T]
```

Stage decomposition:

- CPU/data pipeline: sample-rate check, mono/stereo shape validation, padding/truncation, and `padding_mask` creation by `EncodecFeatureExtractor`.
- Encoder codec stack: NCL Conv1d/residual/ELU region, independently testable from waveform to continuous frame embeddings.
- Encoder transformer: causal sliding-window self-attention over `[B,T,H]`, with optional KV cache for streaming encode.
- Quantizer: split semantic/acoustic RVQ, independently testable from continuous embeddings to codes and back.
- Decoder stack: RVQ decode, upsample, transformer, and NCL ConvTranspose1d waveform synthesis.
- Streaming state: causal convolution padding cache plus optional transformer KV cache. These are not interchangeable.

## 3. Important config dimensions

Worked example: `kyutai/mimi`.

| Field | Value | Source |
|---|---:|---|
| primary target | neural audio codec encode/decode | report scope |
| sampling_rate | 24000 | config/preprocessor |
| waveform channels | 1 mono | config/preprocessor |
| source tensor layout | `[B,C,T]` waveform/codec; `[B,T,H]` transformer | source |
| hidden_size | 512 | config.json |
| num_hidden_layers | 8 per transformer stack, two stacks | config.json |
| num_attention_heads / KV heads | 8 / 8 | config.json |
| head_dim | 64 | config.json |
| intermediate_size | 2048 | config.json |
| activation | GELU in transformer MLP; ELU in codec conv blocks | config/source |
| attention bias | false effective default | source/default; explicit in newer mirrors |
| attention dropout | 0.0 | config.json |
| max_position_embeddings | 8000 | config.json |
| RoPE theta | 10000.0 | config.json historical field; current source uses `rope_parameters` |
| sliding_window | 250 frames | config.json |
| num_filters | 64 | config.json |
| upsampling_ratios | `[8,6,5,4]` | config.json |
| EnCodec-style hop before final Mimi downsample | 960 samples, 25 Hz | inferred from ratios/source |
| Mimi frame_size / frame_rate | 1920 samples / 12.5 Hz | config property or explicit `frame_rate` |
| codebook_size / bits | 2048 / 11 | config + inference |
| codebook_dim | 256 | config.json |
| num_quantizers | 32 effective | source default; omitted by several configs |
| semantic/acoustic quantizers | 1 semantic + 31 acoustic | config/source default |
| VQ projection | Conv1d(512 -> 256), Conv1d(256 -> 512), bias=false | source |
| upsample groups | 512 grouped ConvTranspose1d | config/source |
| cache support | DynamicCache in both transformer stacks; conv padding cache for streaming encode | source |

Representative config sweep:

| Repo/config | Native model_type | H | Layers | Heads/KV | Codebooks | dtype | Notes |
|---|---|---:|---:|---:|---:|---|---|
| kyutai/mimi | mimi | 512 | 8 | 8 / 8 | 32 effective | float32 | official native checkpoint; `num_quantizers` omitted, source default applies |
| onnx-community/kyutai-mimi-ONNX | mimi | 512 | 8 | 8 / 8 | 32 | float32 | ONNX export mirror; not a distinct PyTorch topology |
| jingyaogong/mimi | mimi | 512 | 8 | 8 / 8 | 32 effective | float16 | mirror; omits `num_quantizers` |
| FILM6912/encodec_24khz | mimi | 512 | 8 | 8 / 8 | 32 | float16 | MimiModel config named encodec_24khz |
| Cnam-LMSSC/mimi_throat_microphone | mimi | 512 | 8 | 8 / 8 | 32 | float32 | fine-tune-like config; no preprocessor_config found |

Preprocessor sweep:

| Repo/config | Extractor | feature_size | sampling_rate | chunking | padding |
|---|---|---:|---:|---|---|
| kyutai/mimi and inspected mirrors | EncodecFeatureExtractor | 1 | 24000 | `chunk_length_s=null`, `overlap=null` | right, value 0, returns mask |

## 3a. Family variation traps

- Public native configs are mostly one topology. Do not infer a size ladder from unrelated repos tagged `mimi`.
- Several configs omit `num_quantizers`; current `MimiConfig` supplies `32`, while docs mention 16 codebooks in prose. DinoML should trust the loaded config plus source defaults, not the prose.
- Current config class uses `rope_parameters`; older configs carry `rope_theta`. Loading must normalize this through Transformers config behavior or synthesize default `{"rope_type":"default","rope_theta":10000.0}`.
- `frame_rate` is backward-compatible hidden config state. If absent, source computes `sampling_rate / frame_size = 12.5`; if present, it overrides. This controls whether the extra downsample/upsample modules exist.
- `frame_rate != encodec_frame_rate` is true for standard configs: a stride-2 Conv1d downsample and grouped ConvTranspose1d upsample are part of the graph.
- `num_key_value_heads` can differ from `num_attention_heads` in source, even though swept configs use MHA. GQA lowering needs `repeat_kv`.
- `attention_bias` is source-default false and may be omitted by older configs.
- The source TODO says the encoder currently does not convert `padding_mask` into an attention mask. Padding masks are used for defaults/trimming, not for encoder transformer masking in `_encode_frame`.
- Streaming encode requires both convolution padding cache and encoder transformer KV cache. Decode cache is transformer-only; the decoder ConvTranspose stack has no streaming padding cache.
- NCL layout is semantic for all Conv1d/ConvTranspose1d regions. Transformer regions are `[B,T,H]`. A blanket channel-last rewrite would break transposes, residual blocks, quantizer permutes, and Conv1d axis contracts.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCL Conv1d input/output `[B,C,T]`, asymmetric pad, slice, concat, transpose `[B,C,T] <-> [B,T,C]`.
- `view`, `reshape`, `permute`, `contiguous`, `stack`, `cat`, `flip`, arange/cumsum-like position id creation, bool masks.
- Quantizer axis handling: encode returns `[K,B,T]`, model returns `[B,K,T]`, decode consumes `[B,K,T]`.
- Cache update and append for tensors `[B,KV_heads,T,D]`; repeat KV to `[B,heads,T,D]`.

Neural network primitives:

- Causal/asymmetric `Conv1d`, `ConvTranspose1d`, grouped ConvTranspose1d, optional Conv1d shortcut.
- ELU, GELU, LayerNorm over last dim, residual add, learned layer scale `[H]`.
- Transformer MLP `Linear(512 -> 2048, bias=false) -> GELU -> Linear(2048 -> 512, bias=false)`.
- Attention projections: Q `Linear(512 -> 512)`, K/V `Linear(512 -> 512)`, O `Linear(512 -> 512)`, bias=false for standard configs.
- VQ projection Conv1d `512 -> 256` and `256 -> 512`, kernel 1, bias=false.
- Codebook lookup and Euclidean nearest-centroid quantization over `codebook_size=2048`, `codebook_dim=256`.

Attention primitives:

- Causal self-attention only; no cross-attention.
- RoPE on Q/K, sliding-window causal mask, MHA or source-capable GQA.
- Eager path: matmul, additive mask, fp32 softmax, dropout, matmul, output projection.
- SDPA and FlashAttention2 dispatch paths exist; FlashAttention2 rejects `StaticCache`.

Generation/cache ops:

- `DynamicCache(config)` per transformer stack when `use_cache=True`.
- Encoder transformer cache used when `use_streaming=True` during encode.
- Decoder transformer cache can be passed to `decode` for chunked code decoding.
- Conv padding cache stores per-conv left context for streaming encode.

Preprocessing-coupled ops:

- EncodecFeatureExtractor sample-rate check, mono/stereo validation, batch padding, `padding_mask`.
- No STFT, FFT, mel, normalization, or tokenizer coupling for native Mimi.

## 5. Layer/block breakdown

Encoder Conv stack, standard config:

```text
input_values [B,1,T]
Conv1d(1 -> 64, k=7, stride=1, causal pad)
for ratios reversed [4,5,6,8]:
  ResnetBlock at channels 64,128,256,512:
    residual = x
    ELU -> Conv1d(C -> C/2, k=3, dilation=1)
    ELU -> Conv1d(C/2 -> C, k=1)
    x = residual + branch
  ELU -> Conv1d(C -> 2C, k=2*ratio, stride=ratio)
ELU -> Conv1d(1024 -> 512, k=3)
```

Encoder transformer, repeated 8 times:

```text
x [B,T,512] = transpose(codec_embeddings)
residual = x
x = LayerNorm(x)
q = Linear(512 -> 512)(x).view(B,T,8,64).transpose(1,2)
k,v = Linear(512 -> 512)(x).view(B,T,8,64).transpose(1,2)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v) if cache is enabled
x_attn = causal/sliding attention(q,k,v)
x = residual + LayerScale(0.01) * Linear(512 -> 512)(x_attn)
residual = x
x = LayerNorm(x)
x = residual + LayerScale(0.01) * MLP(512 -> 2048 -> 512)
```

Quantizer:

```text
embeddings [B,512,T25]
downsample Conv1d(512 -> 512, k=4, stride=2, pad_mode=replicate) -> [B,512,T12.5]
semantic RVQ:
  Conv1d(512 -> 256,k=1) -> nearest codebook -> residual subtract
acoustic RVQ:
  same residual encode for requested remaining codebooks
return codes [B,K,T_codes]
```

Decoder:

```text
codes [B,K,T_codes]
semantic/acoustic codebook gather-sum -> [B,256,T_codes]
Conv1d(256 -> 512,k=1) output projection
grouped ConvTranspose1d(512 -> 512,k=4,stride=2,groups=512)
decoder transformer over [B,T,512]
Conv1d(512 -> 1024,k=7)
for ratios [8,6,5,4]:
  ELU -> ConvTranspose1d(C -> C/2,k=2*ratio,stride=ratio)
  ResnetBlock(C/2)
ELU -> Conv1d(64 -> 1,k=3)
```

## 6. Attention requirements

| Site | Causal | Heads | KV heads | Head dim | Positional math | Cache |
|---|---|---:|---:|---:|---|---|
| encoder_transformer | yes, sliding window | 8 | 8 | 64 | RoPE | optional DynamicCache for streaming encode |
| decoder_transformer | yes, sliding window | 8 | 8 | 64 | RoPE | optional DynamicCache for chunked decode |

Cache tensors are stored before `repeat_kv` and after RoPE update as `[B,num_key_value_heads,T_cache,head_dim]` for keys and values. For standard configs no repeat is needed; if `num_key_value_heads < num_attention_heads`, repeat happens after cache update.

Masking uses `create_sliding_window_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids)`. Eager attention adds the resulting mask to scores before fp32 softmax. SDPA slices the causal mask to current key length and uses `is_causal=True` only when no mask is passed and `q_len > 1`. FlashAttention2 transposes to `[B,T,heads,D]`, passes `sliding_window`, and has a top-left-mask compatibility guard.

No packed varlen metadata, ALiBi, relative bias, encoder-decoder cross-attention, or text-generation logits are required for native Mimi.

## 7. Position encoding and custom math

Default RoPE:

```python
def mimi_default_rope(position_ids, head_dim=64, theta=10000.0):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = inv[None, :, None] @ position_ids[:, None, :].float()
    emb = cat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1)
    return emb.cos(), emb.sin()
```

Apply to projected Q/K:

```python
def apply_mimi_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    rotate = lambda x: cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)
```

RVQ encode/decode:

```python
def rvq_encode(x, codebooks):
    residual, codes = x, []
    for table in codebooks:
        idx = cdist(residual.transpose(1, 2).reshape(-1, D), table).argmin(-1)
        q = embedding(idx, table).view(B, T, D).transpose(1, 2)
        residual = residual - q
        codes.append(idx.view(B, T))
    return stack(codes)  # [K,B,T]
```

RoPE cos/sin can be precomputed up to `max_position_embeddings` for static contexts, but dynamic RoPE update support exists in the source for non-default `rope_parameters`. RVQ nearest-neighbor encode depends on runtime embeddings; decode is pure gather-sum.

## 8. Preprocessing and input packing

Feature extractor contract:

- `EncodecFeatureExtractor`, not model-specific Mimi preprocessing.
- Raw mono input shape `(samples,)`; stereo shape `(2,samples)` is supported by the generic extractor, but inspected Mimi configs use `feature_size=1`.
- Required sampling rate is 24000 if supplied; mismatch raises.
- Default `padding=None` becomes `padding=True`; `padding_mask` is returned when padding.
- Output to model is `input_values [B,C,T]` and `padding_mask [B,T]` or batch-compatible mask from the sequence feature extractor. Mimi encode docs describe `[B,C,T]`; forward defaults to `torch.ones_like(input_values).bool()` if absent.
- No STFT/mel/FFT/window/hop extraction. Waveform samples are passed directly.
- Chunking fields are null in inspected preprocessors. Generic EnCodec chunking can alter padded length, but native Mimi tests stream by slicing model `frame_size=1920` samples and passing cache state.

Packing:

- Encode input: `input_values [B,C,T]` -> codes `[B,K,T_codes]`.
- Decode input: `audio_codes [B,K,T_codes]` -> waveform `[B,C,T_out]`; if `padding_mask` is supplied and shorter than decoder output, output is truncated to mask length.
- `get_audio_codes_mask` derives code lengths from `padding_mask.sum(-1)` through the Conv1d length formulas and can flip for left padding.

## 9. Graph rewrite / lowering opportunities

### Rewrite: static causal Conv1d padding + Conv1d

Source pattern:

```text
extra_padding = ceil length guard
causal: pad(left=kernel_size-stride, right=extra_padding) -> Conv1d
noncausal: asymmetric pad -> Conv1d
```

Replacement:

```text
Pad1dExact -> Conv1dNCL
```

Preconditions:

- Preserve NCL layout `[B,C,T]`.
- Use source `_get_extra_padding_for_conv1d` length math.
- Respect `pad_mode` and causal/noncausal switch.
- For streaming encode, replace left pad with `cat(cache, current)` and update cache.

Failure cases: reflect padding on tiny inputs has special right-zero insertion in `_pad1d`; streaming cache is unsupported for noncausal convolutions.

Parity test sketch: per-layer random NCL tensors with lengths below/equal/above padding size, then full encoder conv stack.

### Rewrite: RVQ decode -> fused multi-codebook gather-sum

Source pattern:

```text
for codebook i: embedding(codes[:, i]) -> permute -> sum
```

Replacement:

```text
fused gather over K codebooks -> reduce-sum -> optional Conv1d(256 -> 512,k=1)
```

Preconditions:

- Decode-only path.
- Normalize codes to `[B,K,T]`; first `num_semantic_quantizers` use semantic tables, the rest acoustic tables.
- Preserve distinct physical codebook tables.

Failure cases: encode requires residual nearest-neighbor search and cannot be lowered to gather.

Parity test sketch: random codes for K in `{1,8,32}`, compare quantized continuous embeddings before decoder upsample.

### Rewrite: transformer QKV packing with RoPE

Source pattern:

```text
q = Linear(x); k = Linear(x); v = Linear(x); q,k = RoPE(q,k)
```

Replacement:

```text
packed Linear(512 -> 1536) -> split Q/K/V -> RoPE -> attention
```

Preconditions:

- Self-attention only, same hidden input for Q/K/V.
- Bias handling matches `attention_bias`; standard configs are bias-free.
- For GQA variants, packed K/V output rows are `num_key_value_heads * head_dim`, not full head count.

Weight transform:

```python
Wqkv = cat([Wq, Wk, Wv], dim=0)
bqkv = None if all biases are None else cat([bq, bk, bv], dim=0)
```

Failure cases: separate weight loading/debug views that require original module identities, or nonstandard `head_dim` causing `hidden_size != heads * head_dim` for Q output.

### Rewrite: sliding-window causal attention to fused backend

Replacement: fused SDPA/FlashAttention with RoPE-applied Q/K, cache-aware K/V, and sliding-window mask.

Preconditions:

- Preserve score scale `1/sqrt(head_dim)`.
- Eager parity uses fp32 softmax before downcast.
- Cached keys are already RoPE-positioned.
- FlashAttention static cache is rejected; route StaticCache to SDPA/eager.

Failure cases: backend without sliding-window support, or mask alignment differences for q_len != k_len.

### Rewrite: ConvTranspose1d upsample path

Source pattern:

```text
ConvTranspose1d -> trim left/right fixed padding
```

Replacement:

```text
ConvTranspose1dNCL -> Slice
```

Preconditions:

- Preserve source padding trim: causal standard uses all fixed trim on right for `trim_right_ratio=1.0`.
- Standard Mimi middle upsample uses `groups=512`, kernel 4, stride 2.
- SEANet decoder upsample layers use normal groups=1 with ratio-dependent kernels.

Failure cases: treating ConvTranspose1d as simple non-overlap linear is unsafe; kernels overlap after trimming.

## 10. Kernel fusion candidates

Highest priority:

- NCL Conv1d/ConvTranspose1d with exact causal padding and trimming. Codec stacks dominate end-to-end encode/decode and have many small temporal convolutions.
- RVQ decode gather-sum plus 1x1 projection. This is simple, frequent, and isolates codebook layout.
- Sliding-window RoPE attention with cache. Both encoder and decoder transformers use the same block, and streaming depends on cache correctness.
- LayerNorm + Linear/MLP GEMM around `H=512`, `I=2048`.

Medium priority:

- Packed QKV projection and RoPE application.
- Learned layer-scale multiply fused into residual add.
- Streaming conv padding cache update and concat for frame-size chunks.
- Euclidean codebook search for encode, using batched distance or equivalent `x^2 + e^2 - 2xe`.

Lower priority:

- Generic EncodecFeatureExtractor chunking; inspected preprocessors do not enable it.
- Noncausal convolution path and `use_conv_shortcut=True`; source supports them but standard configs do not use them.
- Dynamic/non-default RoPE variants; no inspected native checkpoint requires them.

## 11. Runtime staging plan

Stage 1: config and shape loader.

- Parse MimiConfig including omitted defaults, `frame_rate`, `rope_theta`/`rope_parameters`, and preprocessor metadata.
- Load weights for conv stacks, transformers, VQ projections, and codebooks.
- Accept prepared `input_values [B,1,T]` and `audio_codes [B,K,T]`.

Stage 2: decoder-only from codes to waveform.

- Implement RVQ decode, grouped stride-2 upsample, decoder transformer without cache, and SEANet decoder.
- Truncate waveform by `padding_mask` length.

Stage 3: encoder-only from waveform to codes.

- Implement causal Conv1d encoder, encoder transformer, downsample, Euclidean RVQ encode.
- Validate code sums and code shapes against Transformers fixtures.

Stage 4: streaming encode/decode state.

- Add `MimiConv1dPaddingCache` equivalent.
- Add DynamicCache for encoder streaming and decoder chunk decode.
- Validate chunk-by-frame-size parity.

Stage 5: optimized attention and conv providers.

- Add SDPA/FlashAttention dispatch with sliding-window guards.
- Add optimized NCL Conv1d/ConvTranspose1d and packed QKV.

Stage 6: broader config guards.

- Add explicit rejection or fallback for stereo, noncausal conv, conv shortcuts, dynamic RoPE, and `frame_rate == encodec_frame_rate` until validated.

## 12. Parity and validation plan

- Config tests: load all snapshotted native configs, verify effective defaults for omitted `num_quantizers`, `attention_bias`, and frame-rate-derived downsample/upsample presence.
- Feature extractor parity: raw mono arrays to `input_values [B,1,T]` and `padding_mask`, sample-rate mismatch rejection, padding/truncation behavior.
- Conv unit tests: `MimiConv1d._get_output_length`, causal pad, streaming pad cache, ConvTranspose trim.
- Transformer unit tests: one layer with random `[B,T,512]`, RoPE, sliding mask, fp32 eager attention, then all 8 layers.
- Cache parity: prefill plus chunked continuation for encoder and decoder; verify cache shape `[B,8,T,64]`.
- Quantizer tests: Euclidean nearest-codebook encode, residual subtraction order, semantic/acoustic split, decode gather-sum.
- End-to-end encode/decode smoke with `kyutai/mimi`: use integration-test style RMSE/code-sum checks for `num_quantizers=8` and `32`.
- Streaming encode parity: split input by `frame_size=1920`, pass padding cache and encoder KV cache, compare concatenated codes to full encode.
- Recommended tolerances: fp32 hidden parity `atol=1e-5` to `1e-4`; fp16/bf16 optimized attention/conv start at `rtol=2e-2, atol=2e-2`; waveform parity should include RMSE because transposed-conv accumulation may differ.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- CPU preprocessing throughput: audio seconds/sec for padding and tensor conversion.
- Encoder Conv1d stack throughput by waveform length and batch size.
- Encoder transformer throughput by frame count, with and without sliding-window fused attention.
- RVQ encode cost by codebook count `1,8,32`, separating distance search from residual update.
- Decoder RVQ gather-sum and waveform decoder throughput from random valid codes.
- Chunked streaming encode latency for `frame_size=1920` samples.
- Decode cache benchmark: full-code decode vs split decode with `decoder_past_key_values`.
- NCL Conv1d provider comparison: source-faithful PyTorch-equivalent vs optimized kernels.
- KV cache memory by `B`, frame count, two transformer stacks, and `H=512`.
- End-to-end encode/decode requests/hour split into preprocessing, conv encoder, transformer, quantizer, transformer decoder, and conv decoder.

## 14. Skip/defer list

- Training behavior, gradient checkpointing, weight-norm application/removal, and checkpoint conversion.
- Remote-code downstream systems that use Mimi tokens with Qwen, TTS, ASR, or ONNX streaming wrappers.
- Stereo Mimi unless a native `audio_channels=2` checkpoint is targeted.
- Noncausal convolution, reflect padding, and `use_conv_shortcut=True` optimized paths beyond guarded fallback.
- Dynamic/non-default RoPE variants.
- GPU implementation of EncodecFeatureExtractor padding/chunking.
- StaticCache with FlashAttention2; source rejects it.
- Quantized/ONNX export parity; use separate audit if DinoML targets those artifacts.

## 15. Final implementation checklist

- [ ] Parse `MimiConfig` with omitted defaults and backward-compatible `frame_rate`.
- [ ] Parse `EncodecFeatureExtractor` metadata for 24 kHz mono waveform input.
- [ ] Load Conv1d/ConvTranspose1d, LayerNorm, Linear, layer-scale, VQ projection, and codebook weights.
- [ ] Implement exact NCL `MimiConv1d` padding and output-length math.
- [ ] Implement `MimiConv1dPaddingCache` for streaming encode.
- [ ] Implement SEANet encoder and decoder residual blocks.
- [ ] Implement Mimi RoPE and sliding-window causal attention.
- [ ] Implement transformer DynamicCache shapes `[B,KV_heads,T,head_dim]`.
- [ ] Implement Mimi transformer block with LayerNorm, attention, layer scale, and GELU MLP.
- [ ] Implement split RVQ encode with Euclidean nearest-codebook search.
- [ ] Implement split RVQ decode gather-sum and 1x1 output projection.
- [ ] Add guarded fused QKV/RoPE/attention lowering.
- [ ] Add guarded NCL Conv1d/ConvTranspose1d optimized lowering.
- [ ] Add parity tests for conv padding, RoPE, attention cache, RVQ, full encode/decode, and streaming encode.
- [ ] Benchmark preprocessing, conv stacks, transformer stacks, quantizer encode/decode, cache memory, and end-to-end codec throughput.
