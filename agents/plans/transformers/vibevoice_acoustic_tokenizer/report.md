# Transformers Audit: vibevoice_acoustic_tokenizer

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
  2026-05-11, "fix(minicpmv4_6): skip invalid failing tests (#45836)"
Model id:
  microsoft/VibeVoice-AcousticTokenizer for the standalone tokenizer target.
  microsoft/VibeVoice-1.5B for the original composite config that the converter normalizes.
Config source:
  https://huggingface.co/microsoft/VibeVoice-AcousticTokenizer/blob/main/config.json
  https://huggingface.co/microsoft/VibeVoice-AcousticTokenizer/raw/main/preprocessor_config.json
  https://huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json
  https://huggingface.co/microsoft/VibeVoice-1.5B/raw/main/preprocessor_config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/vibevoice_acoustic_tokenizer/configuration_vibevoice_acoustic_tokenizer.py
  X:/H/transformers/src/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py
  X:/H/transformers/src/transformers/models/vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py
  X:/H/transformers/src/transformers/models/vibevoice_acoustic_tokenizer/modular_vibevoice_acoustic_tokenizer.py
  X:/H/transformers/src/transformers/models/vibevoice_acoustic_tokenizer/convert_vibevoice_acoustic_tokenizer_to_hf.py
Any missing files or assumptions:
  modeling_vibevoice_acoustic_tokenizer.py is generated from modular_vibevoice_acoustic_tokenizer.py; future upstream edits should target the modular file.
  No small/debug architectural variant was found. Available public configs are the same architecture or legacy-key mirrors.
  The report did not download full safetensors weights.
```

Auxiliary evidence snapshot: `agents/plans/transformers/vibevoice_acoustic_tokenizer/config_sweep.md`.

## 2. High-level architecture

Primary DinoML runtime target: inference reconstruction/tokenization for `VibeVoiceAcousticTokenizerModel`, with deterministic `sample=False` first. This is not an autoregressive text model and has no attention, logits, token vocabulary, RoPE, or KV cache. It is a causal 1D convolutional VAE-style acoustic tokenizer:

```text
mono waveform preprocessing -> [B,1,T] input_values
  -> encoder causal Conv1d/ConvNeXt stack -> latents [B,L,64]
  -> optional VAE noise sampling
  -> decoder causal Conv1d/ConvTranspose1d/ConvNeXt stack -> reconstructed audio [B,1,T']
```

Stage decomposition:

| Stage | Owner | Shape/layout | Independently stageable |
| --- | --- | --- | --- |
| Audio normalization, padding, channel insertion | CPU/data pipeline feature extractor | mono list to padded `input_values [B,1,T]`; `padding_mask [B,T]` emitted but not consumed by model source | Yes; keep outside GPU graph initially. |
| Encoder | GPU/runtime candidate | NCL conv body, BLC latent output | Yes; validate encoder-only latent parity with `sample=False`. |
| Latent sampling | Runtime or generation controller | `latents += scalar_per_batch * randn_like(latents)` | Defer or disable for deterministic parity. |
| Decoder | GPU/runtime candidate | BLC latent input permuted to NCL, output NCL audio | Yes; validate decoder-only with random latents. |
| Streaming cache | Runtime session state | per-conv left-padding state `[B,C,left_pad]` | Defer after full-sequence parity. |

## 3. Important config dimensions

Normalized official standalone config:

| Field | Value | Source / runtime effect |
| --- | ---: | --- |
| `channels` | 1 | Mono waveform input and output. |
| `hidden_size` | 64 | Latent width; encoder head output and decoder stem input. |
| `kernel_size` | 7 | Stem/head and ConvNeXt depthwise conv kernel. |
| `num_filters` | 32 | Base channel width. |
| `downsampling_ratios` | `[2,2,4,5,5,8]` | Encoder stride ratios; product/hop length `3200`. |
| `depths` | `[3,3,3,3,3,3,8]` | Encoder stem depth plus per-downsample-stage ConvNeXt depths; decoder reverses this list. |
| `ffn_expansion` | 4 | ConvNeXt MLP width is `4 * C`. |
| `hidden_act` | `gelu` | Ungated FFN activation. |
| `rms_norm_eps` | `1e-5` | RMSNorm epsilon. |
| `layer_scale_init_value` | `1e-6` | Per-channel residual scale parameters `gamma` and `ffn_gamma`. |
| `vae_std` | `0.625` | Stochastic latent sampling scale. |
| `dtype` | `bfloat16` | Checkpoint config dtype; source ops also run fp32/fp16/bf16 depending weights/inputs. |
| `cache support` | Optional `use_cache` conv padding cache | Not KV cache; per-layer causal convolution left state. |

Representative checkpoint sweep:

| Checkpoint/config | Normalized fields? | Operator-significant variation |
| --- | --- | --- |
| `microsoft/VibeVoice-AcousticTokenizer` | Yes | Baseline standalone tokenizer. |
| `bezzam/VibeVoice-AcousticTokenizer` | Yes | Mirror with the same normalized fields. |
| `microsoft/VibeVoice-1.5B` nested `acoustic_tokenizer_config` | No, original composite names | Converter maps `encoder_ratios=[8,5,5,4,2,2]` to `downsampling_ratios=[2,2,4,5,5,8]`, `encoder_depths` string to list, `vae_dim` to `hidden_size`, and `fix_std/0.8` to `vae_std=0.625`. |
| `vibevoice/VibeVoice-Audio-Tokenizer` | No | Open mirror with legacy/original fields only and no preprocessor config observed. |
| `mrfakename/VibeVoice-Acoustic-Tokenizer` | No | Same legacy/original field pattern as above. |

## 3a. Family variation traps

- This family has no transformer attention despite living under Transformers. Do not force prefill/decode, logits, or KV-cache abstractions onto the primary target.
- Source uses NCL `[B,C,T]` for Conv1d/ConvTranspose1d and BLC `[B,T,C]` for RMSNorm/Linear FFN. Layout passes must guard every `transpose(1,2)`/`permute(0,2,1)` boundary.
- The feature extractor emits `padding_mask`, but the model forward ignores it. Padded audio therefore still enters convolution; parity must match source behavior.
- `sample=True` is the default for `forward()` and `encode()`, adding random latent noise. First deterministic runtime should call/admit `sample=False`.
- Legacy configs advertise fields such as `causal`, `conv_bias`, `pad_mode`, `mixer_layer`, `layernorm`, `disable_last_norm`, and `decoder_ratios`; the current native source does not read these fields after normalized config construction. Treat them as conversion-time metadata, not runtime feature switches.
- The decoder reverses `depths` and `downsampling_ratios` through `decoder_config`/`upsampling_ratios`; naive encoder/decoder symmetry from raw config order is wrong.
- Streaming cache is static-address mutable state per convolution layer, not an autoregressive KV cache. Batch size, left-pad length, and in-channel width are part of its ABI.

## 4. Operator coverage checklist

Tensor/layout ops:

- Static and dynamic shape validation for rank-3 tensors `[B,C,T]` and `[B,T,C]`.
- `permute(0,2,1)`, `transpose(1,2)`, contiguous or materialized layout copies as needed.
- `pad` on the last dimension with left-only constant zero padding.
- `cat` on time axis for streaming cache.
- Slice/truncate on time axis: `[..., :-padding_total]`, `[:, :, -expected_new_output:]`, and `hidden_states[:, :, -left_pad:]`.
- Broadcast multiply/add with per-channel parameters shaped `[C]` or `[C,1]`.

Neural network primitives:

- Conv1d NCL with bias, groups=1:
  - encoder stem `1 -> 32`, kernel 7, stride 1.
  - encoder downsampling stages `32 -> 64`, `64 -> 128`, `128 -> 256`, `256 -> 512`, `512 -> 1024`, `1024 -> 2048`, kernels `[4,4,8,10,10,16]`, strides `[2,2,4,5,5,8]`.
  - encoder head `2048 -> 64`, kernel 7, stride 1.
  - decoder stem `64 -> 2048`, kernel 7, stride 1.
  - decoder head `32 -> 1`, kernel 7, stride 1.
- Depthwise Conv1d NCL with bias, groups=C, kernel 7, stride 1 for each ConvNeXt mixer at C in `{32,64,128,256,512,1024,2048}`.
- ConvTranspose1d NCL with bias, groups=1:
  - decoder upsampling stages `2048 -> 1024`, `1024 -> 512`, `512 -> 256`, `256 -> 128`, `128 -> 64`, `64 -> 32`, kernels `[16,10,10,8,4,4]`, strides `[8,5,5,4,2,2]`, followed by right trim of `kernel-stride`.
- RMSNorm over last axis of BLC tensors, fp32 variance accumulation, learned weight, no bias.
- Linear BLC:
  - per ConvNeXt block `Linear(C -> 4C)` + GELU + `Linear(4C -> C)`, both with bias.
- Elementwise: `pow`, `mean`, `rsqrt`, GELU, multiply, add, abs/max for preprocessing, optional random normal.

Attention primitives:

- None required.

Preprocessing-coupled ops:

- Mono audio validation, sampling-rate guard, optional RMS amplitude normalization to target dB FS, max-absolute clamp to `[-1,1]` if needed, right padding, channel insertion.

Recurrent/state ops:

- Optional convolution padding cache: allocate/update fixed tensors `[B,in_channels,left_pad]`, copy current padding state, concatenate old padding with current chunk.

## 5. Layer/block breakdown

Encoder full-sequence path, default config:

```text
input_values: [B,1,T] NCL
stem:
  x = left_pad_zeros(6) -> Conv1d(1 -> 32, k=7, s=1)      # [B,32,T]
  repeat 3 ConvNeXt blocks at C=32
for ratios [2,2,4,5,5,8] and channels [64,128,256,512,1024,2048]:
  x = left_pad_zeros(r) -> Conv1d(Cin -> Cout, k=2r, s=r) # time floor-divides by r
  repeat depth ConvNeXt blocks at Cout
head:
  x = left_pad_zeros(6) -> Conv1d(2048 -> 64, k=7, s=1)
  latents = permute NCL -> BLC                                # [B,L,64]
```

ConvNeXt block at channel width C:

```text
residual = x                                                # [B,C,T]
y = RMSNorm(transpose(x, BCT -> BTC))
y = transpose(y, BTC -> BCT)
y = causal depthwise Conv1d(C -> C, k=7, groups=C)
x = residual + y * gamma[:, None]
residual = x
y = RMSNorm(transpose(x, BCT -> BTC))
y = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
y = transpose(y, BTC -> BCT)
x = residual + y * ffn_gamma[:, None]
```

Decoder full-sequence path:

```text
latents: [B,L,64] BLC
x = permute BLC -> NCL                                      # [B,64,L]
stem:
  x = left_pad_zeros(6) -> Conv1d(64 -> 2048, k=7, s=1)
  repeat 8 ConvNeXt blocks at C=2048
for ratios [8,5,5,4,2,2] and channels [1024,512,256,128,64,32]:
  x = ConvTranspose1d(Cin -> Cout, k=2r, s=r)
  x = x[..., :-(k-r)]                                       # output length becomes old_T * r
  repeat depth ConvNeXt blocks at Cout
head:
  audio = left_pad_zeros(6) -> Conv1d(32 -> 1, k=7, s=1)    # [B,1,T']
```

For an input length divisible by `3200`, encoder latent length is `T/3200` and decoder reconstructs `T` samples. For non-divisible lengths, each encoder stage applies sequential floor division by its ratio; exact reconstruction length then follows from the floored latent length.

## 6. Attention requirements

No attention is required for the primary target. The following are not applicable: causal self-attention, cross-attention, MHA/GQA/MQA, attention masks, packed/varlen attention, RoPE/ALiBi, FlashAttention/SDPA, and autoregressive KV cache.

The only stateful inference feature is convolution padding cache. Cache entries are keyed by module string names such as `encoder_stem`, `encoder_layer_i`, `convnext_layer_j`, `decoder_layer_i`, and `decoder_head`. Each layer stores the previous left context as `[B,in_channels,left_pad]`. Cached keys are not expanded/repeated by heads and do not grow with sequence length.

## 7. Position encoding and custom math

No positional encoding is present. Causal temporal behavior comes only from left padding and convolution kernels.

RMSNorm custom math:

```python
def vibevoice_rms_norm(x, weight, eps):
    # x is BLC; normalize over C.
    y = x.float()
    var = (y * y).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(var + eps)
    return weight * y.to(dtype=x.dtype)
```

Causal Conv1d padding:

```python
left_pad = (kernel_size - 1) * dilation - (stride - 1)
x = pad_last_dim_left(x, left_pad)
y = conv1d(x, weight, bias, stride=stride, dilation=dilation, groups=groups)
```

Causal ConvTranspose1d trim:

```python
y = conv_transpose1d(x, weight, bias, stride=r, kernel_size=2 * r)
y = y[..., :-(kernel_size - stride)]  # if positive
```

## 8. Preprocessing and input packing

Feature extractor contract:

- Input must be mono audio; each example is converted to `torch.float32` and must be rank 1.
- Sampling rate defaults to and is checked against `24000`.
- Only `return_tensors="pt"` is supported.
- If `normalize_audio=True`, RMS is computed as `sqrt(mean(audio**2))`; waveform is scaled by `10 ** (target_dB_FS / 20) / (rms + eps)`. If max absolute value exceeds `1.0`, it divides by `max_val + eps`.
- Padding defaults to longest/right padding with value `0.0`; `pad_to_multiple_of` and `max_length` are delegated to the sequence feature extractor.
- The output model tensor is `input_values [B,1,T]`; `padding_mask [B,T]` may be returned but is not consumed by the model forward.

CPU/data-pipeline first integration should own all preprocessing. GPU graph admission should require already-normalized/padded `input_values [B,1,T]` and ignore or reject `padding_mask` until there is an explicit model consumer.

## 9. Graph rewrite / lowering opportunities

### Rewrite: full-sequence causal Conv1d to pad plus Conv1d provider

Source pattern:

```text
left_pad_zeros -> nn.Conv1d(NCL)
```

Replacement:

```text
PadLastDimLeft -> provider Conv1d
```

Preconditions:

- Input rank is 3 NCL.
- Padding mode is constant zero.
- `left_pad == (kernel_size - 1) * dilation - (stride - 1)` and is non-negative.
- Groups are either `1` or exactly `C` for depthwise ConvNeXt mixers.

Failure cases: streaming cache enabled, nonzero right padding, non-source config claiming a different pad mode.

Parity test sketch: compare one stem conv, one downsample conv per ratio, and one depthwise mixer against PyTorch over odd and divisible sequence lengths.

### Rewrite: non-streaming ConvTranspose1d to provider op plus right trim

Source pattern:

```text
ConvTranspose1d(k=2r, stride=r) -> slice last dim removing r samples
```

Replacement:

```text
ConvTranspose1d provider -> DynamicSlice
```

Preconditions:

- `kernel_size == 2 * stride`.
- `padding == 0`, `output_padding == 0`, `dilation == 1`, `groups == 1`.
- Right trim is exactly `kernel_size - stride`.

Failure cases: streaming cache path, non-default ConvTranspose1d attrs, dynamic trim not represented in shape metadata.

### Rewrite: BCT/BTC layout island fusion

Source pattern:

```text
transpose -> RMSNorm -> transpose -> depthwise Conv1d
transpose -> RMSNorm -> Linear -> GELU -> Linear -> transpose
```

Opportunity:

- Keep source-faithful graph initially.
- Later fuse the norm/MLP BLC island while preserving NCL conv providers.

Layout constraints:

- `RMSNorm` and `Linear` reduce/project the channel axis as last dimension in BLC.
- Conv1d consumes channel axis as dim 1 in NCL.
- A channel-last/NHWC-style pass must rewrite axes and guarantee all consumers in the local island agree.

Failure cases: streaming cache with NCL state tensors, external ABI expecting NCL audio, or arbitrary user-provided strided tensors.

### Rewrite: deterministic encoder admission

Source pattern:

```text
if sample:
  latents += stochastic noise
```

Replacement:

```text
Reject sample=True for deterministic runtime or route to Python/Torch fallback.
```

Preconditions: `sample=False` is supplied or a graph-level deterministic mode is selected.

Failure cases: user expects stochastic VAE sampling parity, seeded randomness, or distributional output tests.

## 10. Kernel fusion candidates

Highest priority:

- Causal Conv1d and depthwise Conv1d provider coverage. The model is dominated by 1D convolution over long audio sequences; Conv1d/depthwise Conv1d is a key missing operator family for this target.
- ConvTranspose1d provider coverage with exact trim semantics. Decoder parity depends on this; no attention work is useful before this lands.
- RMSNorm over BLC with fp32 accumulation. It appears twice per ConvNeXt block.

Medium priority:

- Linear(C -> 4C) + GELU + Linear(4C -> C) with residual scale. These are small-width but numerous; GEMM epilogue/fused elementwise can reduce launch count.
- Depthwise causal Conv1d + layer scale + residual add fusion for each ConvNeXt mixer.
- BCT/BTC transpose elimination inside local ConvNeXt blocks, guarded by layout contracts.

Lower priority:

- Streaming cache update kernel/ring buffer. Useful for chunked audio, but full-sequence reconstruction parity can land first.
- Stochastic latent sampling. This is a controller/RNG parity issue, not a core deterministic inference requirement.

## 11. Runtime staging plan

Stage 1: config and preprocessing admission.

- Parse normalized `VibeVoiceAcousticTokenizerConfig`.
- Reject legacy/original key-only configs unless a converter-normalization pass is added.
- Accept `input_values [B,1,T]` directly; leave waveform normalization/padding in Python.

Stage 2: operator micro-parity.

- Add/validate Conv1d NCL, depthwise Conv1d, ConvTranspose1d, RMSNorm, GELU FFN, pad/slice/transpose patterns.
- Test default widths and kernels from this family.

Stage 3: decoder-only parity.

- Run random latents `[B,L,64]` through decoder, no cache.
- Validate exact output length `L * 3200` for default ratios.

Stage 4: encoder-only parity.

- Run padded waveform `[B,1,T]` through encoder with `sample=False`.
- Cover lengths divisible by `3200` and non-divisible floor behavior.

Stage 5: end-to-end deterministic reconstruction.

- `forward(input_values, sample=False, use_cache=False)`.
- Compare `latents` and reconstructed `audio`.

Stage 6: streaming cache.

- Define explicit session-state ABI for every conv cache tensor.
- Validate chunked inference equals full-sequence inference within tolerance.

Stage 7: optimization.

- Add provider/fusion plans for Conv1d/ConvTranspose1d, RMSNorm, FFN, and transpose elimination.

## 12. Parity and validation plan

- Feature extractor parity: mono validation, sampling-rate rejection, RMS normalization, clipping, padding, and output `[B,1,T]`.
- RMSNorm unit tests: fp32/fp16/bf16 inputs, tolerance `1e-5` fp32, `1e-2` fp16/bf16 initially.
- Causal Conv1d unit tests: each default ratio/kernel plus depthwise groups=C. Include short sequences smaller than left pad.
- ConvTranspose1d unit tests: ratios `[8,5,5,4,2,2]`, verify source trim and output length.
- ConvNeXt block parity at C in `{32,64,2048}`.
- Encoder-only after each stage: compare stage outputs to local Transformers/PyTorch.
- Decoder-only after each upsampling stage.
- End-to-end with `sample=False`, `use_cache=False`.
- Streaming cache parity: split one waveform into chunks; compare chunked decode/encode output against full sequence after handling boundary semantics.
- Do not include stochastic `sample=True` in first deterministic parity gate; add seeded RNG tests only if DinoML owns that path.

## 13. Performance probes

- CPU feature-extractor throughput: seconds of audio normalized/padded per second.
- Encoder-only throughput by input seconds and batch size.
- Decoder-only throughput by latent length and batch size.
- End-to-end reconstruction throughput for 1 s, 10 s, 60 s, and long-form chunks.
- Conv1d provider comparison: direct convolution versus im2col/GEMM for downsampling layers.
- Depthwise Conv1d specialized kernel versus generic grouped convolution.
- ConvTranspose1d provider comparison and trim overhead.
- Transpose/materialization overhead inside ConvNeXt blocks.
- Streaming cache overhead: cache update/cat cost per chunk size.
- Memory usage by largest channel stage C=2048 and long sequence length.

## 14. Skip/defer list

- Training, gradients, checkpoint conversion, and weight initialization parity.
- Stochastic `sample=True` latent noise for first deterministic runtime.
- Streaming `use_cache=True` until full-sequence parity is stable.
- Legacy/original key-only config admission unless a converter-normalization pass is in scope.
- Composite VibeVoice LLM, semantic tokenizer, and diffusion head. Those belong to separate audits.
- Attention, RoPE, KV cache, token logits, beam search, language control, and text generation; they are not part of this family target.
- Quantization and safetensors weight metadata inspection beyond normal dense loading.

## 15. Final implementation checklist

- [ ] Parse normalized `VibeVoiceAcousticTokenizerConfig`.
- [ ] Admit/preprocess `input_values [B,1,T]`; keep waveform normalization in CPU pipeline.
- [ ] Reject or normalize legacy/original acoustic config keys.
- [ ] Load dense Conv1d, ConvTranspose1d, RMSNorm, and Linear weights.
- [ ] Implement NCL causal Conv1d with left zero padding.
- [ ] Implement depthwise Conv1d with `groups=C`.
- [ ] Implement ConvTranspose1d with source right-trim semantics.
- [ ] Implement RMSNorm over BLC with fp32 accumulation.
- [ ] Implement ConvNeXt block parity.
- [ ] Add decoder-only parity tests.
- [ ] Add encoder-only parity tests with `sample=False`.
- [ ] Add end-to-end reconstruction parity with `use_cache=False`.
- [ ] Design optional convolution padding cache ABI.
- [ ] Add streaming chunk parity only after full-sequence parity passes.
- [ ] Benchmark Conv1d, depthwise Conv1d, ConvTranspose1d, RMSNorm, FFN, and transpose overhead separately.
