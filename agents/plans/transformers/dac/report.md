# DAC Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: descript/dac_16khz, descript/dac_24khz, descript/dac_44khz
Config source: HF Hub config.json and preprocessor_config.json for the three official descript/* DAC checkpoints
Source files inspected: configuration_dac.py, feature_extraction_dac.py, modeling_dac.py, convert_dac_checkpoint.py, __init__.py
Any missing files or assumptions: no gated official config gaps; source is in-library, not remote-code-only
```

Primary runtime target for this report: inference audio codec encode/decode. The useful first DinoML target is `decode(audio_codes) -> waveform`, followed by `encode(waveform) -> audio_codes` and full reconstruction. Training losses, quantizer dropout, and checkpoint conversion are deferred.

Exact local source is under `transformers/src/transformers/models/dac` at the commit above. Hub config snapshots are recorded in `_sources/snapshots/hf_config_sweep.json`; source notes are in `_sources/source_notes.md`.

## 2. High-level architecture

DAC is a convolutional neural audio codec, not an attention or text-generation model.

```text
raw mono waveform -> feature extractor pad/pack -> Conv1d encoder -> residual vector quantizer
  -> audio codes / quantized latent -> ConvTranspose1d decoder -> tanh waveform -> crop to input length
```

Stage decomposition:

| Stage | Owner | Runtime contract |
|---|---|---|
| Audio load/resample | CPU/data pipeline | Caller supplies mono float waveform at checkpoint sampling rate. |
| Feature extraction | CPU/data pipeline first | Validate sampling rate, right pad/truncate, pad to `hop_length`, emit `[B,1,T]`; `padding_mask` is produced but model forward does not consume it. |
| Encoder | DinoML graph | NCL Conv1d stack with Snake activations, residual units, strided downsampling. |
| RVQ encode | DinoML graph or staged helper | Per-codebook 1x1 Conv1d projection, L2 normalization, codebook distance/top-1, residual update, code stacking. |
| RVQ decode | DinoML graph or staged helper | Codebook embedding gather per codebook, 1x1 Conv1d out projection, sum codebook contributions. |
| Decoder | DinoML graph | NCL Conv1d + ConvTranspose1d upsampling stack, residual units, final tanh. |
| Postprocess | Runtime/output wrapper | Squeeze channel and crop to original input length for full forward. |

`decode(audio_codes)` can be validated independently because it bypasses the expensive nearest-codebook search and accepts a compact integer ABI. `encode` adds codebook argmax and residual update requirements.

## 3. Important config dimensions

Source defaults from `DacConfig`:

| Field | Source default |
|---|---:|
| `encoder_hidden_size` | 64 |
| `decoder_hidden_size` | 1536 |
| `downsampling_ratios` | `[2, 4, 8, 8]` |
| `upsampling_ratios` | reverse of downsampling |
| `hidden_size` | `encoder_hidden_size * 2 ** len(downsampling_ratios)` = 1024 |
| `hop_length` | product of downsampling ratios |
| `n_codebooks` | 9 |
| `codebook_size` | 1024, must be power of two |
| `codebook_dim` | 8 |
| `sampling_rate` | 16000 |
| `quantizer_dropout` | 0.0 |
| `dtype` | checkpoint configs report `float32` |
| cache support | none; no KV cache or autoregressive state |

Representative checkpoint sweep:

| Checkpoint | Sampling rate | Ratios | Effective hop | Preprocessor hop | Codebooks | Frame rate inference | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| `descript/dac_16khz` | 16000 | `[2,4,5,8]` | 320 | 320 | 12 | `ceil(16000/320)=50` | Hub config also stores stale `hop_length: 512`; do not trust that field over ratios/preprocessor. |
| `descript/dac_24khz` | 24000 | `[2,4,5,8]` | 320 | 320 | 32 | `ceil(24000/320)=75` | Same stale config `hop_length: 512` issue. |
| `descript/dac_44khz` | 44100 | `[2,4,8,8]` | 512 | 512 | 9 | `ceil(44100/512)=87` | Default-like ratio family. |

## 3a. Family variation traps

- This is a codec family with no attention, no MLP blocks, no logits, no tokenizer vocab, no text generation, and no KV cache.
- Source layout is NCL audio tensors, `[batch, channels, time]`. Do not silently rewrite to channel-last unless the entire local convolution region and every axis-sensitive op is guarded.
- `ConvTranspose1d` output length for stride 5 is not the simple inverse of strided `Conv1d` under the source `padding=ceil(stride/2)`. Keep exact PyTorch length equations and crop behavior.
- The 16 kHz and 24 kHz Hub configs contain a stale `hop_length` field. Effective source behavior derives hop length from `downsampling_ratios`; the preprocessor agrees with the product.
- `n_quantizers` is runtime-controllable for inference encode. If provided, source stops after that many quantizers; output codebook count changes.
- `quantizer_dropout` is training-only in the inspected forward path. Reject or ignore it for inference rather than lowering random dropout.
- Codebook size must be a power of two; source checks this during model init.
- Converted checkpoints have weight norm removed before HF save. DinoML should expect ordinary dense Conv1d/ConvTranspose1d weights in saved HF weights, not live weight-normalized parameters.
- `padding_mask` is emitted by the feature extractor when padding, but DAC model forward ignores it. Length/crop metadata is still needed for end-to-end reconstruction parity.

## 4. Operator coverage checklist

Tensor/layout ops:

- Static and dynamic rank-3 NCL tensor shape validation.
- `reshape`, `permute(0,2,1)`, `transpose(1,2)`, channel concatenation, `stack(dim=1)`, slicing/cropping `[..., :length]`.
- Center crop in residual unit when conv output length differs.
- Squeeze channel from `[B,1,T]` to `[B,T]`.

Neural network primitives:

- `Conv1d` with bias, including kernels 1, 3, 7, and `2 * stride`; strides 1, 2, 4, 5, 8; dilation 1, 3, 9; padding as source.
- `ConvTranspose1d` with bias for decoder upsampling, same stride set and `kernel_size=2*stride`.
- Elementwise add/sub/mul, reciprocal, sin, square/pow(2), tanh.
- Snake activation: learnable `alpha` with shape `[1,C,1]`.
- Residual add and residual subtraction in RVQ.
- MSE losses are training-only and optional for inference.

RVQ/codebook ABI:

- Integer `audio_codes`: source documents `[B, num_codebooks, Tq]`, code dtype long/int64.
- Codebook embedding weights per quantizer: `[codebook_size, codebook_dim]`, usually `[1024,8]`.
- Encode-side code search: L2-normalize projected latents and codebook; compute score over `[B*Tq, codebook_size]`; take argmax.
- Decode-side code lookup: gather embedding rows per codebook, transpose to `[B, codebook_dim, Tq]`, per-codebook `Conv1d(8 -> 1024,k=1)`, sum.
- `projected_latents`: concatenated `[B, n_codebooks_used * codebook_dim, Tq]`.

Preprocessing-coupled ops:

- Sampling-rate equality check.
- Mono waveform validation only; stereo is explicitly rejected.
- Right padding with value 0.0.
- Pad to multiple of feature extractor `hop_length`.
- Optional truncation or padding, but source rejects both true simultaneously.

Postprocess ops:

- Full forward crops decoded waveform to original input length.
- End-to-end decode should return `audio_values` as `[B,T]`.

Attention/generation/cache ops:

- Not applicable. No attention masks, packed varlen attention metadata, positional encodings, RoPE, logits, sampling, or KV cache are required.

## 5. Layer/block breakdown

Encoder, for input `[B,1,T]`:

```text
x = Conv1d(1 -> 64, k=7, padding=3)(input)
for stride_index i=1..len(ratios), stride s in downsampling_ratios:
  C_in = encoder_hidden_size * 2 ** (i - 1)
  C_out = encoder_hidden_size * 2 ** i
  x = ResidualUnit(C_in, dilation=1)(x)
  x = ResidualUnit(C_in, dilation=3)(x)
  x = ResidualUnit(C_in, dilation=9)(x)
  x = Snake(C_in)(x)
  x = Conv1d(C_in -> C_out, k=2*s, stride=s, padding=ceil(s/2))(x)
x = Snake(hidden_size)(x)
x = Conv1d(hidden_size -> hidden_size, k=3, padding=1)(x)
```

Residual unit:

```text
y = Snake(C)(x)
y = Conv1d(C -> C, k=7, dilation=d, padding=3*d)(y)
y = Snake(C)(y)
y = Conv1d(C -> C, k=1)(y)
if lengths differ: x = center_crop(x)
return x + y
```

Residual vector quantizer:

```text
residual = encoder_latent
quantized_sum = 0
for i in selected_codebooks:
  z_i = Conv1d(hidden_size -> codebook_dim, k=1)(residual)
  codes_i = nearest_codebook(normalize(z_i), normalize(codebook_i))
  q_i = embedding_i(codes_i).transpose(1, 2)
  q_hidden_i = Conv1d(codebook_dim -> hidden_size, k=1)(q_i)
  quantized_sum += q_hidden_i
  residual -= q_hidden_i
audio_codes = stack(codes_i, dim=1)
projected_latents = cat(z_i, dim=1)
```

Decoder from quantized `[B,1024,Tq]`:

```text
x = Conv1d(1024 -> 1536, k=7, padding=3)(quantized)
for stride_index i, stride s in upsampling_ratios:
  C_in = decoder_hidden_size // 2 ** i
  C_out = decoder_hidden_size // 2 ** (i + 1)
  x = Snake(C_in)(x)
  x = ConvTranspose1d(C_in -> C_out, k=2*s, stride=s, padding=ceil(s/2))(x)
  x = ResidualUnit(C_out, dilation=1)(x)
  x = ResidualUnit(C_out, dilation=3)(x)
  x = ResidualUnit(C_out, dilation=9)(x)
x = Snake(final_channels)(x)
x = Conv1d(final_channels -> 1, k=7, padding=3)(x)
x = tanh(x)
```

All conv modules use bias in PyTorch defaults. Checkpoint conversion removes weight norm after loading original DAC weights, so HF inference weights are normal conv weights.

## 6. Attention requirements

No attention is required. DAC has no self-attention, cross-attention, MHA/MQA/GQA, masks, sliding windows, ALiBi/RoPE, FlashAttention/SDPA, generation controller, or KV cache.

The only cache-like optimization opportunity is reusing decoded codebook embeddings or quantized latents for repeated decode calls. That is not a source ABI cache and should be treated as a DinoML-side optional memoization, not model state.

## 7. Position encoding and custom math

No position encoding is present. Temporal location is represented implicitly through convolutional receptive fields and strided down/up sampling.

Custom Snake activation:

```python
def snake1d(x, alpha):
    # x: [B, C, T], alpha: [1, C, 1]
    return x + (1.0 / (alpha + 1e-9)) * sin(alpha * x) ** 2
```

Encode-side codebook search:

```python
def dac_decode_latents(z, codebook):
    # z: [B, D, T], codebook: [N, D]
    enc = normalize(z.permute(0, 2, 1).reshape(B * T, D))
    cb = normalize(codebook)
    score = -(enc.pow(2).sum(1, keepdim=True) - 2 * enc @ cb.T) + cb.pow(2).sum(1, keepdim=True).T
    codes = score.argmax(dim=1).reshape(B, T)
    return embedding(codes).transpose(1, 2), codes
```

Because both operands are normalized, score ranking is effectively cosine-similarity ranking plus constants, but parity should preserve the source expression or prove the simplification under fp tolerance.

## 8. Preprocessing and input packing

Feature extractor contract:

- Input raw audio can be a single array/list or a batch of arrays.
- Mono only: feature size 1; stereo currently raises an error.
- Expected sample rate must equal preprocessor `sampling_rate`.
- Default padding is enabled when `padding is None`.
- Padding and truncation cannot both be enabled.
- Padding uses right side and `padding_value=0.0`.
- `pad_to_multiple_of=hop_length`.
- Tensor output shape for model execution is `[B,1,T_padded]`.
- `padding_mask` is produced when padding is enabled but is not consumed by `DacModel.forward`.

For end-to-end parity, DinoML should keep original unpadded length in wrapper metadata so final decoded audio can be cropped to the caller-visible length. For model-only parity, pass already packed `[B,1,T]` and validate only graph outputs.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d(k=1) -> per-time Linear/GEMM

Source pattern: quantizer `in_proj`, quantizer `out_proj`, residual `conv2`.

Replacement:

```text
[B,C,T] -> transpose/flatten [B*T,C] -> GEMM(weight.T) + bias -> reshape/transpose [B,C_out,T]
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Dense contiguous NCL or a guarded layout-aware equivalent.
- Bias shape `[C_out]`.

Weight transform:

```python
w_linear = conv.weight[:, :, 0]  # [C_out, C_in]
```

Failure cases: grouped conv, nonzero padding, non-unit stride, non-contiguous layout without an accessor-aware GEMM path.

Parity test sketch: random `[B,C,T]`, compare Conv1d and GEMM lowering for several `T`, fp32 tight tolerance and fp16 relaxed tolerance.

### Rewrite: fixed Conv1d -> im2col + GEMM

Source pattern: encoder/decoder/residual `Conv1d` with kernel 3, 7, or `2*stride`.

Replacement:

```text
Pad1d -> SlidingWindowExtract [B*T_out, C_in*K] -> GEMM(weight_flat.T) + bias -> reshape [B,C_out,T_out]
```

Preconditions:

- `groups == 1`.
- Source NCL layout preserved or fully rewritten with axis guards.
- Exact PyTorch output length equation for stride, dilation, and padding.
- For dynamic `T`, runtime shape guards must compute `T_out` exactly.

Weight transform:

```python
w = conv.weight.reshape(C_out, C_in * K)
```

Failure cases: attempting to treat strided conv as non-overlap patch GEMM when `kernel_size != stride`; dilation handling omitted; boundary padding mismatch.

### Rewrite: ConvTranspose1d -> upsample-scatter GEMM or provider op

Source pattern: decoder upsampling blocks.

Replacement options:

- Prefer a native ConvTranspose1d provider first.
- Later lower to zero-insertion/scatter plus Conv1d or a specialized transposed-conv kernel.

Preconditions:

- Preserve PyTorch formula:
  `T_out = (T_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1`.
- Here `kernel=2*stride`, `padding=ceil(stride/2)`, `output_padding=0`.
- Must support stride 5 for 16/24 kHz checkpoints.

Failure cases: assuming decoder output length is exactly `T_in * stride`; this is false for odd stride 5 under source padding.

### Rewrite: RVQ from_codes -> batched gather + 1x1 GEMM sum

Source pattern: `DacResidualVectorQuantizer.from_codes`.

Replacement:

```text
for each codebook i:
  GatherRows(codebook_i, codes[:, i, :]) -> [B,Tq,D]
  transpose -> [B,D,Tq]
  1x1 Conv/GEMM out_proj_i -> [B,H,Tq]
sum over i
```

Preconditions:

- `audio_codes` int64/int32 values in `[0, codebook_size)`.
- Runtime `num_codebooks <= config.n_codebooks`.
- Codebook axis order exactly `[B,Q,Tq]`.

Failure cases: arbitrary codebook count beyond weights, wrong code dtype, flattening `[Q,T]` in the wrong order.

### Rewrite: encode nearest-codebook search -> normalized top1 GEMM

Source pattern: `decode_latents` score matrix and `dist.max(1)`.

Replacement:

```text
NormalizeRows(z_flat) -> NormalizeRows(codebook) -> GEMM(z, codebook.T) -> ArgMax
```

Preconditions:

- Normalize semantics and epsilon match PyTorch `F.normalize` defaults.
- If removing constant terms from score, prove equivalence with normalized operands and acceptable tie behavior.
- Argmax returns first maximal index, matching PyTorch.

Failure cases: tie-breaking drift, lower precision changing code index, using squared-distance minimization without matching source score ordering.

### Rewrite: layout-local NCL conv pipeline

Source pattern: long NCL convolution stack.

Replacement: keep semantic NCL graph initially. Later, a provider can run internal channel-last or vectorized layout inside a fused region.

Preconditions:

- Region boundaries are Conv1d/Snake/residual only.
- Axis-sensitive ops are rewritten together: channel axis 1, time axis 2, concat/stack code axes, crop `[..., time]`.
- Outputs returned to source NCL ABI before RVQ or public outputs unless the consumer is also inside the fused region.

Failure cases: exposing channel-last tensors to RVQ code that expects channel dimension at axis 1.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d/ConvTranspose1d provider coverage for NCL audio. The decoder is dominated by transposed convolutions; fallback im2col may be too memory-heavy.
- Snake + Conv1d fusion. Snake is repeated before most convs and is elementwise with per-channel alpha.
- RVQ decode from codes. Codebook gather plus per-codebook 1x1 projection plus sum is the narrowest useful audio-code decoder path.

Medium priority:

- Conv1d(k=1) GEMM lowering using existing GEMM/CUTLASS infrastructure.
- ResidualUnit fusion: Snake -> dilated Conv1d -> Snake -> 1x1 Conv -> residual add.
- Encode codebook top-1: normalized projection + codebook GEMM + argmax, especially for large `B*Tq*Q`.

Lower priority:

- Full Conv1d im2col fusion for small kernels.
- Preprocessor padding on GPU. CPU/data-pipeline preprocessing is sufficient first.
- Training loss fusion. Not needed for inference.

## 11. Runtime staging plan

Stage 1: Config and weight loading audit.

- Parse `DacConfig`, normalize effective hop length from ratios, and reject config/source inconsistencies only when they affect graph shape.
- Load Conv1d, ConvTranspose1d, Snake alpha, codebook embeddings, and quantizer projection weights.

Stage 2: Decode-only from audio codes.

- Implement `from_codes` ABI and decoder stack.
- Accept `audio_codes [B,Q,Tq]`; produce `[B,T_audio]`.
- Stub encode-side nearest-codebook search.

Stage 3: Encoder-only latent parity.

- Implement Conv1d/residual/Snake downsampling stack and validate latent shape/value parity before RVQ.

Stage 4: Encode with RVQ.

- Add normalized codebook top-1, residual subtraction, code stacking, and optional `n_quantizers`.

Stage 5: Full reconstruction.

- Wire feature-extractor-compatible input, full forward crop, and end-to-end waveform parity.

Stage 6: Optimized providers and fusions.

- Replace reference Conv1d/ConvTranspose1d lowering with provider-backed kernels and fuse Snake/residual regions where profitable.

## 12. Parity and validation plan

- Snake activation random tensor parity for `[B,C,T]` with broadcast alpha.
- Conv1d shape/value parity for every DAC kernel/stride/dilation combination, including stride 5.
- ConvTranspose1d shape/value parity for strides 2, 4, 5, 8.
- ResidualUnit parity with odd/even `T` values that exercise center crop.
- `from_codes` parity using random valid integer codes for `Q=1`, partial `Q`, and full checkpoint `Q`.
- `decode(audio_codes)` checkpoint parity against Transformers for short clips.
- Encoder latent parity before RVQ for padded lengths equal to one and several hops.
- RVQ encode parity: verify `audio_codes`, `projected_latents`, and quantized representation. Use fp32 first because argmax code selection is precision-sensitive.
- Full forward parity: reconstructed waveform cropped to original length, recommended initial fp32 tolerance `rtol=1e-4`, `atol=1e-4`; relaxed tolerance after provider/fp16 paths.

No DinoML tests or imports were run for this report, per user scope.

## 13. Performance probes

- CPU preprocessing throughput: raw audio validation, pad to hop multiple, tensor packing.
- Decode-only throughput by `B`, `Q`, and `Tq`, separating RVQ gather/projection from decoder ConvTranspose1d.
- Encoder-only throughput by sample rate and waveform seconds.
- Conv1d versus im2col+GEMM versus provider Conv1d for residual blocks.
- ConvTranspose1d provider comparison for strides 2, 4, 5, 8.
- RVQ encode codebook search throughput: sweep `B*Tq`, `n_codebooks`, `codebook_size=1024`, `codebook_dim=8`.
- End-to-end reconstruction latency for 1 s, 5 s, 30 s audio at 16/24/44.1 kHz.
- Memory probe for temporary im2col buffers if Conv1d lowering uses window extraction.

## 14. Skip/defer list

- Training losses: commitment loss and codebook loss.
- Quantizer dropout and random training-time codebook selection.
- Stereo audio: source feature extractor rejects it.
- Checkpoint conversion scripts and original Descript weight-norm loading.
- Remote-code mirrors such as non-official wrappers; official in-library source is sufficient for this audit.
- Streaming/chunked overlap-add decode. The inspected HF source does not implement a chunk reassembly ABI.
- Mixed precision and quantized weights. Checkpoint configs report F32; optimize later after fp32 parity.

## 15. Final implementation checklist

- [ ] Parse `DacConfig` and compute effective `hidden_size`, `upsampling_ratios`, `hop_length`, and frame rate from source equations.
- [ ] Load HF DAC weights for Conv1d, ConvTranspose1d, Snake alpha, codebooks, and quantizer projections.
- [ ] Implement NCL `Conv1d` coverage for DAC kernel/stride/dilation/padding set.
- [ ] Implement NCL `ConvTranspose1d` coverage for DAC decoder strides, including stride 5.
- [ ] Implement Snake activation.
- [ ] Implement residual unit center-crop and add semantics.
- [ ] Implement `from_codes(audio_codes)` RVQ decode ABI with gather, 1x1 projection, and sum.
- [ ] Add decode-only parity tests for `descript/dac_16khz`, `descript/dac_24khz`, and `descript/dac_44khz`.
- [ ] Implement encoder stack parity tests.
- [ ] Implement normalized codebook top-1 and RVQ encode residual loop.
- [ ] Add full forward reconstruction parity with original-length crop.
- [ ] Add safe rewrites for 1x1 Conv1d to GEMM and RVQ decode gather/projection/sum.
- [ ] Benchmark Conv1d, ConvTranspose1d, RVQ encode, RVQ decode, and end-to-end reconstruction separately.
