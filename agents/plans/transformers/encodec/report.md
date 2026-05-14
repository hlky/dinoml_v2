# EnCodec Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/encodec_24khz, facebook/encodec_48khz, facebook/encodec_32khz
Config source: Hugging Face config.json and preprocessor_config.json for the three model ids above
Source files inspected:
  X:/H/transformers/src/transformers/models/encodec/configuration_encodec.py
  X:/H/transformers/src/transformers/models/encodec/feature_extraction_encodec.py
  X:/H/transformers/src/transformers/models/encodec/modeling_encodec.py
Any missing files or assumptions:
  No remote code is required for the audited in-library EncodecModel path.
  This report targets inference encode/decode parity, not training losses or discriminator code from the original project.
```

Primary source URLs used for representative configs:

- https://huggingface.co/facebook/encodec_24khz/resolve/main/config.json
- https://huggingface.co/facebook/encodec_24khz/resolve/main/preprocessor_config.json
- https://huggingface.co/facebook/encodec_48khz/resolve/main/config.json
- https://huggingface.co/facebook/encodec_48khz/resolve/main/preprocessor_config.json
- https://huggingface.co/facebook/encodec_32khz/resolve/main/config.json
- https://huggingface.co/facebook/encodec_32khz/resolve/main/preprocessor_config.json

## 2. High-level architecture

EnCodec is a neural audio codec, not a Transformer attention model. The runtime contract is a VQ-VAE-like waveform encoder/decoder with residual vector quantization.

```text
raw waveform preprocessing -> SEANet Conv1d encoder -> RVQ code selection -> RVQ embedding sum -> SEANet ConvTranspose1d decoder -> waveform
```

Stage decomposition:

- CPU/data pipeline: validate sampling rate, convert mono/stereo waveform arrays to float32, pad/truncate, emit `input_values` and `padding_mask`.
- Encoder graph: NCL waveform `float` tensor `[batch, channels, samples]` through causal or noncausal Conv1d/residual/LSTM stack to latent `[batch, hidden_size, frames]`.
- Quantizer graph: bandwidth-dependent number of RVQ codebooks; nearest-code search emits integer codes.
- Decoder graph: codebook embedding lookup/sum to `[batch, hidden_size, frames]`, Conv1d/LSTM/ConvTranspose1d stack to waveform.
- Chunking graph: optional chunk loop and decode overlap-add. The 48 kHz model requires chunking for standard preprocessing; 24 kHz and 32 kHz configs do not.

Encode-only tokenization, decode-only synthesis from provided codes, and full encode-decode reconstruction can be validated independently. There is no autoregressive cache, prefill, decode-token loop, or attention head.

## 3. Important config dimensions

| Field | Source default | 24 kHz | 48 kHz | 32 kHz |
|---|---:|---:|---:|---:|
| `sampling_rate` | 24000 | 24000 | 48000 | 32000 |
| `audio_channels` / `feature_size` | 1 | 1 mono | 2 stereo | 1 mono |
| `hidden_size` | 128 | 128 | 128 | 128 |
| `num_filters` | 32 | 32 | 32 | 64 |
| `num_residual_layers` | 1 | 1 | 1 | 1 |
| `upsampling_ratios` | `[8,5,4,2]` | `[8,5,4,2]` | `[8,5,4,2]` | `[8,5,4,4]` |
| Derived `hop_length` | 320 | 320 | 320 | 640 |
| Derived `frame_rate` | 75 | 75 | 150 | 50 |
| `num_lstm_layers` | 2 | 2 | 2 | 2 |
| `norm_type` | `weight_norm` | `weight_norm` | `time_group_norm` | `weight_norm` |
| `use_causal_conv` | true | true | false | false |
| `normalize` | false | false | true | false |
| `chunk_length_s` | null | null | 1.0 | null |
| Derived `chunk_length` | null | full input | 48000 | full input |
| `overlap` | null | null | 0.01 | null |
| Derived `chunk_stride` | null | full input | 47520 | full input |
| `codebook_size` | 1024 | 1024 | 1024 | 2048 |
| Derived `codebook_nbits` | 10 | 10 | 10 | 11 |
| Derived max `num_quantizers` | 32 | 32 | 16 | 4 |
| `target_bandwidths` | `[1.5,3,6,12,24]` | `[1.5,3,6,12,24]` | `[3,6,12,24]` | `[2.2]` |
| `use_conv_shortcut` | true | default true, omitted in config | default true, omitted in config | false |
| dtype | config/HF metadata | float32 | float32 | float32 |
| cache support | source-derived | none | none | none |

Bandwidth selects active RVQ layers as `max(1, floor(bandwidth * 1000 / (log2(codebook_size) * frame_rate)))`. That yields 24 kHz quantizer counts `{1.5: 2, 3: 4, 6: 8, 12: 16, 24: 32}`, 48 kHz counts `{3: 2, 6: 4, 12: 8, 24: 16}`, and 32 kHz count `{2.2: 4}`.

## 3a. Family variation traps

- The family is Conv1d/LSTM/RVQ based. Do not force attention, RoPE, KV-cache, or token-generation assumptions.
- Source tensors are NCL: `[batch, channels, time]` for audio and `[batch, hidden, frames]` for latents. Add NCL layout guards before any layout translation.
- The 48 kHz variant is stereo, noncausal, normalized, chunked, and uses `GroupNorm(1, channels)` after convs. This changes both operators and boundary math.
- The 24 kHz variant is mono, causal, unchunked, and uses PyTorch weight norm parametrization on Conv1d/ConvTranspose1d.
- The 32 kHz variant has `num_filters=64`, `codebook_size=2048`, `upsampling_ratios=[8,5,4,4]`, `use_conv_shortcut=false`, and only one target bandwidth.
- `use_conv_shortcut` is omitted by 24 kHz and 48 kHz configs, so the effective default is true from the current config class.
- The feature extractor transposes numpy inputs internally. For batched stereo arrays it expects external shape `(2, samples)` but model input becomes `[batch, 2, samples]`.
- `padding_mask` may be `[batch, samples]` from preprocessing and is reshaped in the model to `[batch, 1, samples]`; broadcast against stereo inputs is source behavior.
- Chunking is a model/runtime behavior, not just preprocessing. Encode loops over chunks; decode performs linear overlap-add.
- RVQ output shape is frame-major: `audio_codes` `[nb_frames, batch, nb_quantizers, frame_len]`, not batch-major.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCL contiguous Conv1d layout, no NHWC/NLC default translation.
- `permute(2,0,1)` and `permute(1,2,0)` around LSTM.
- `transpose(0,1)` for codebook-frame axes.
- `reshape/view`, `stack`, slicing, pad, mask multiply, scalar broadcast multiply/divide.
- Dynamic loops over chunks and active quantizers.

Neural network primitives:

- Conv1d with dynamic asymmetric/causal pre-padding.
- ConvTranspose1d plus deterministic trim.
- WeightNorm-wrapped Conv1d/ConvTranspose1d, preferably folded to plain conv weights for inference.
- GroupNorm with one group for `time_group_norm`.
- ELU activation.
- LSTM with `input_size=hidden_size`, `hidden_size=hidden_size`, `num_layers=2`, no explicit state input/output in public API.
- Residual add, optional 1x1 Conv1d shortcut.

Quantizer and code ops:

- Codebook distance: `x^2 - 2*x@embed.T + embed^2`, argmax over codebook after negated distance.
- Integer code tensors, `torch.stack`, `nn.functional.embedding`.
- Residual quantization loop: subtract decoded quantized vector after each active codebook; decoder sums active quantized vectors.

Preprocessing-coupled ops:

- Sampling-rate validation, mono/stereo shape validation, right padding with zeros, optional truncation.
- Chunk-length alignment to `(nb_step - 1) * chunk_stride + chunk_length`.
- Padding mask production and mask-based zeroing before encode.

Attention/cache/position ops:

- None. No causal attention, cross-attention, RoPE, ALiBi, packed sequence metadata, or KV cache is present.

## 5. Layer/block breakdown

Source `EncodecConv1d`:

```text
input [B, Cin, T]
extra_padding = ceil((T - effective_kernel + padding_total) / stride + 1) - 1 transformed back to ideal length
if causal: pad left=padding_total, right=extra_padding
else: pad left=padding_total - floor(padding_total/2), right=floor(padding_total/2) + extra_padding
Conv1d(Cin -> Cout, kernel, stride, dilation)
optional GroupNorm(1, Cout) for time_group_norm
```

Source `EncodecConvTranspose1d`:

```text
input [B, Cin, T]
ConvTranspose1d(Cin -> Cout, kernel, stride)
optional GroupNorm(1, Cout)
trim fixed padding:
  causal: right = ceil((kernel - stride) * trim_right_ratio)
  noncausal: right = floor((kernel - stride) / 2)
  left = (kernel - stride) - right
output[..., left:end]
```

Encoder, repeated by downsampling ratio in reversed `upsampling_ratios`:

```text
x = EncodecConv1d(audio_channels -> num_filters, kernel=7)
for ratio in reversed(upsampling_ratios):
  for residual layer:
    y = ELU(x)
    y = EncodecConv1d(C -> C/compress, residual_kernel_size=3, dilation=dilation_growth_rate**j)
    y = ELU(y)
    y = EncodecConv1d(C/compress -> C, kernel=1)
    x = shortcut(x) + y
  x = ELU(x)
  x = EncodecConv1d(C -> 2C, kernel=2*ratio, stride=ratio)
x = LSTM(C_final -> C_final, num_layers=2) + residual
x = ELU(x)
x = EncodecConv1d(C_final -> hidden_size, last_kernel_size=7)
```

Decoder:

```text
x = EncodecConv1d(hidden_size -> num_filters * 2**len(upsampling_ratios), kernel=7)
x = LSTM(C -> C, num_layers=2) + residual
for ratio in upsampling_ratios:
  x = ELU(x)
  x = EncodecConvTranspose1d(C -> C/2, kernel=2*ratio, stride=ratio)
  residual blocks at C/2 as above
x = ELU(x)
x = EncodecConv1d(num_filters -> audio_channels, last_kernel_size=7)
```

For 24/48 kHz, channel ladder is `32 -> 64 -> 128 -> 256 -> 512 -> hidden_size 128`. For 32 kHz, it is `64 -> 128 -> 256 -> 512 -> 1024 -> hidden_size 128`. Conv modules include bias by PyTorch default.

## 6. Attention requirements

No attention is required. The following are not applicable for the primary EnCodec target: causal self-attention, encoder-decoder cross-attention, MHA/MQA/GQA, attention masks, sliding-window attention, ALiBi/RoPE, FlashAttention/SDPA, packed attention metadata, and KV cache.

The only cache-like reusable artifacts are non-autoregressive: precomputed audio codes for decode-only synthesis, and optionally preprocessed/padded waveform batches. These are not KV caches.

## 7. Position encoding and custom math

No learned or analytic position encoding is present. Temporal position enters through convolution padding/stride and LSTM recurrence over latent frames.

Custom math that needs exact parity:

```python
def rvq_quantize(x, embed):
    # x: [B*T, codebook_dim], embed: [codebook_size, codebook_dim]
    dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed.t() + embed.t().pow(2).sum(0, keepdim=True))
    return dist.max(dim=-1).indices
```

```python
def bandwidth_to_quantizers(bandwidth, codebook_size, frame_rate, max_quantizers):
    if bandwidth is None or bandwidth <= 0:
        return max_quantizers
    bw_per_q = math.log2(codebook_size) * frame_rate
    return int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
```

```python
def overlap_weight(frame_length, dtype, device):
    t = torch.linspace(0, 1, frame_length + 2, dtype=dtype, device=device)[1:-1]
    return 0.5 - (t - 0.5).abs()
```

The RVQ codebook embeddings are model buffers, not train-time optimizer state for inference. Distance terms involving `embed.pow(2).sum(...)` can be precomputed per codebook for a fixed checkpoint.

## 8. Preprocessing and input packing

Feature extractor contract:

- Raw mono input can be a 1D array/list of length `samples`; raw stereo input is documented as `(2, samples)`.
- It validates `sampling_rate` when supplied and warns if absent.
- It defaults to padding unless `padding=False` or truncation is selected; padding and truncation cannot both be active.
- It emits `input_values` with model layout `[batch, channels, samples]`.
- If padding is active, it emits `padding_mask` with right-padding semantics and model input name `padding_mask`.
- For chunked configs and no explicit `max_length`, padding extends to `(ceil(max_input/chunk_stride) - 1) * chunk_stride + chunk_length`; truncation uses floor instead.

Runtime encode contract:

- `input_values`: `[B, C, T]`, `C` must be 1 or 2.
- `padding_mask`: reshaped to `[B, -1, T]`, then sliced per chunk and multiplied into the waveform.
- If `config.normalize`, scale is computed from mono mix RMS over the frame: `sqrt(mean(mono**2)) + 1e-8`; decode multiplies by this scale.
- If `chunk_length_s` is null, one frame covers the full input.
- If chunking is active, offsets are `range(0, input_length, chunk_stride)`.

Decode contract:

- `audio_codes`: `[nb_frames, B, active_quantizers, frame_len]`.
- `audio_scales`: list length `nb_frames`; entries are `[B,1]` or `None`.
- Last encoded frame may be padded in code space so frames can be stacked; `last_frame_pad_length` removes that code padding before decoding.
- Chunked decode overlap-adds decoded waveform frames with triangular weights and then truncates to `padding_mask.shape[-1]` when needed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fold WeightNorm Conv1d for inference

Source pattern:

```text
weight_norm(Conv1d/ConvTranspose1d)(x)
```

Replacement:

```text
Conv1d/ConvTranspose1d(x, folded_weight, bias)
```

Preconditions:

- Inference-only, no mutation of `weight_g`/`weight_v`.
- The PyTorch parametrization has been materialized using the same formula as the checkpoint loader.
- Preserve Conv1d NCL layout, stride, dilation, groups, and bias.

Failure cases: dynamic retraining/fine-tuning, unsupported parametrization metadata, grouped convs beyond source defaults.

Parity test sketch: compare each folded conv module against the original module for random NCL tensors at short, exact, and non-divisible lengths.

### Rewrite: static RVQ codebook distance GEMM

Source pattern:

```text
dist = -(x2 - 2 * x @ embed.T + embed2)
argmax(dist)
```

Replacement:

```text
scores = x @ (2 * embed.T) - precomputed_embed_norm
argmax(scores)
```

Preconditions:

- Codebook embedding fixed for the compiled artifact.
- Preserve argmax tie behavior as PyTorch as far as DinoML supports it.
- `x2` can be dropped because it is constant across codebook choices for a row.

Shape equations:

- Encoder latent before VQ is `[B, D, F]`; VQ uses `[B*F, D] @ [D, K] -> [B*F, K]`.
- `D=codebook_dim`, `K=codebook_size`.

Failure cases: exact tie parity requirements, dynamic codebook mutation, unsupported integer index output.

### Rewrite: chunked encode as static-shape frame batch

Source pattern:

```text
for offset in range(0, input_length, stride):
  frame = input[..., offset:offset+chunk_length]
  encode_frame(frame)
```

Replacement:

```text
FrameExtract/Pad -> reshape [B*nb_frames, C, chunk_length] -> encoder -> regroup frames
```

Preconditions:

- Known `chunk_length` and `chunk_stride`.
- Padding has already aligned length as the feature extractor specifies.
- Preserve per-frame normalization and per-frame scale output.

Failure cases: ragged unpadded inputs, user bypasses feature extractor, dynamic input lengths without a chosen max bucket.

### Rewrite: no-layout-translation guard for codec core

Source pattern:

```text
NCL Conv1d/LSTM codec region
```

Replacement: none initially; keep source layout.

Preconditions for future NLC/channel-last optimization:

- Rewrite every Conv1d, ConvTranspose1d, GroupNorm channel axis, LSTM input permutation, residual add, RVQ permute, and codebook GEMM boundary consistently.
- Consumers must remain inside the rewritten region until returning to public NCL contract.

Failure cases: partial layout conversion around padding/slicing or GroupNorm axis will silently break parity.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d/ELU/residual blocks in NCL. These dominate encoder/decoder runtime and appear many times.
- ConvTranspose1d with trim. Decoder throughput depends on this and exact trimming is part of waveform parity.
- RVQ distance GEMM plus argmax. Encoding bandwidth and tokenization latency are dominated by repeated codebook search.
- LSTM with residual in latent time layout. DinoML needs a clear backend story or an explicit fallback.

Medium priority:

- WeightNorm folding at load/compile time. This removes parametrization overhead and simplifies conv kernels.
- GroupNorm(1, C) for 48 kHz time-group-norm path.
- Chunk extraction and overlap-add kernels for 48 kHz stereo.
- Normalize/unscale RMS kernels for `normalize=true`.

Lower priority:

- Specialized mono/stereo padding-mask multiply.
- Codebook embedding lookup/sum fusion for decode-only.
- Full frame-batch lowering for chunked encode once scalar-loop parity is stable.

## 11. Runtime staging plan

Stage 1: parse config and preprocessor config; load weights and expose the exact NCL input/output contracts.

Stage 2: implement unchunked 24 kHz encode-decode with PyTorch fallback or reference kernels for LSTM/RVQ while Conv1d padding and output shapes are validated.

Stage 3: fold weight norm and lower Conv1d/ConvTranspose1d/ELU/residual blocks; add exact NCL layout guards.

Stage 4: implement RVQ encode/decode with bandwidth-dependent active quantizer count and integer code outputs.

Stage 5: add 48 kHz path: stereo inputs, `normalize=true`, `time_group_norm`, chunking, last-frame code padding, and overlap-add.

Stage 6: add 32 kHz path: wider channel ladder, 2048-entry codebooks, 11-bit codebook math, altered hop length, and identity residual shortcut.

Stage 7: optimize RVQ GEMM/argmax, chunk batching, and fused codec blocks.

Initially stubbable: audio file IO, resampling, streaming APIs, and training-only codebook update buffers.

## 12. Parity and validation plan

- Feature extractor tests for mono/stereo shapes, sampling-rate mismatch, padding/truncation rejection, chunk-aligned max length, and padding mask orientation.
- Conv1d padding tests for causal and noncausal modes at short inputs, exact stride multiples, and non-divisible lengths.
- ConvTranspose1d trim tests for causal and noncausal modes.
- Resnet block parity with and without conv shortcut.
- LSTM wrapper parity: NCL input -> sequence-major LSTM -> NCL output plus residual.
- RVQ quantizer tests for active quantizer counts per target bandwidth and code shapes.
- Single encoder and single decoder parity against Transformers for 24/48/32 kHz random tensors.
- End-to-end encode-decode parity for short waveforms and chunk-boundary waveforms.
- Decode-only parity from captured `audio_codes`, `audio_scales`, and `last_frame_pad_length`.

Recommended tolerances: fp32 conv/LSTM/RVQ decode should target close absolute parity around `1e-5` to `1e-4` for module-level tests. End-to-end waveform parity may need relaxed tolerances if backend LSTM or conv algorithms differ. RVQ integer code parity is stricter: code indices should match exactly unless ties are detected and explicitly tolerated.

## 13. Performance probes

- Feature extractor throughput: samples/sec for padding and chunk alignment.
- Encoder-only throughput by sample length and batch size.
- Decoder-only throughput from codes by frame length and active quantizer count.
- RVQ encode throughput split by codebook size and quantizer count.
- Chunked 48 kHz throughput with and without frame batching.
- Overlap-add bandwidth for stereo chunk counts.
- LSTM share of encoder/decoder latency.
- Memory footprint for codebook scores `[B*F, codebook_size]` and alternatives that tile the codebook.
- End-to-end encode-decode latency for 1 s, 5 s, and 30 s clips at batch sizes 1, 4, and 16.

No benchmark observations are included here; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training losses, discriminator/adversarial components, and codebook EMA updates.
- Audio resampling and file decoding; require caller or processor pipeline to provide the configured sampling rate.
- Streaming encode/decode APIs beyond the chunked batch behavior implemented in Transformers.
- Distributed or tensor-parallel execution.
- Autoregressive generation, beam search, speculative decoding, KV cache, and attention backends.
- Non-HF remote-code variants unless separately audited.
- Aggressive NLC/channel-last layout translation until NCL parity is established.

## 15. Final implementation checklist

- [ ] Parse `EncodecConfig` including derived `hop_length`, `frame_rate`, `codebook_nbits`, `num_quantizers`, `chunk_length`, and `chunk_stride`.
- [ ] Parse `EncodecFeatureExtractor` config and validate `input_values`/`padding_mask` layout.
- [ ] Load Conv1d, ConvTranspose1d, LSTM, GroupNorm, and codebook weights/buffers.
- [ ] Implement NCL Conv1d asymmetric/causal padding with reflect-mode short-input behavior.
- [ ] Implement ConvTranspose1d fixed trim.
- [ ] Implement or route LSTM wrapper with NCL input/output and residual add.
- [ ] Fold WeightNorm conv parameters for inference.
- [ ] Implement GroupNorm(1, C) for `time_group_norm`.
- [ ] Implement residual blocks with optional 1x1 conv shortcut and identity shortcut.
- [ ] Implement RVQ bandwidth-to-quantizer selection, codebook GEMM/argmax, residual subtraction, embedding lookup, and quantized sum.
- [ ] Implement encode output shape `[nb_frames, batch, active_quantizers, frame_len]`.
- [ ] Implement decode from provided codes/scales and `last_frame_pad_length`.
- [ ] Implement chunk loop and linear overlap-add for 48 kHz.
- [ ] Add no-layout-translation guards for the codec core.
- [ ] Add 24 kHz, 48 kHz, and 32 kHz module parity tests.
- [ ] Add performance probes for Conv1d stack, LSTM, RVQ, and overlap-add.
