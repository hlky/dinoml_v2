# UnivNet audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: dg845/univnet-dev
Config source: https://huggingface.co/dg845/univnet-dev/raw/main/config.json
Source files inspected:
- transformers/src/transformers/models/univnet/configuration_univnet.py
- transformers/src/transformers/models/univnet/modeling_univnet.py
- transformers/src/transformers/models/univnet/feature_extraction_univnet.py
- transformers/src/transformers/models/univnet/convert_univnet.py
- transformers/tests/models/univnet/test_modeling_univnet.py
- transformers/tests/models/univnet/test_feature_extraction_univnet.py
Any missing files or assumptions: only one public Transformers-native HF config was found. `my3bikaht/univnet-RU` has no config.json and is treated as a non-native checkpoint gap, not a source of runtime requirements.
```

Small snapshots are stored beside this report:

- `dg845_univnet-dev_config.json`
- `dg845_univnet-dev_preprocessor_config.json`
- `hf_config_availability.md`

The source basis is the in-library generator/vocoder. The training discriminator from UnivNet GAN is not implemented in Transformers and is out of scope for inference.

## 2. High-level architecture

UnivNet in Transformers is a neural vocoder, not a transformer. It maps a log-mel spectrogram plus a Gaussian noise sequence to a mono waveform.

```text
raw waveform preprocessing -> log-mel spectrogram + Gaussian noise
  -> Conv1d noise stem
  -> repeated LVC upsampling blocks conditioned by the full mel spectrogram
  -> Conv1d waveform head + tanh
  -> optional CPU batch_decode trim by waveform_lengths
```

Stage decomposition:

- CPU/data pipeline: mono waveform validation, right padding/truncation, optional end padding, reflect-padded STFT, mel filterbank, log compression, optional Tacotron-style normalization, noise generation.
- GPU/runtime target: consume `input_features [B, T_mel, num_mel_bins]`, optional `noise_sequence [B, T_mel, model_in_channels]`, optional sample-level `padding_mask [B, T_samples]`; emit waveform `[B, T_mel * prod(strides)]`.
- Independently optimizable region: each LVC block has a static transposed convolution and a mel-conditioned kernel predictor; the expensive custom op is location-variable convolution, which can be validated independently from preprocessing.

## 3. Important config dimensions

| Field | Source default | `dg845/univnet-dev` | Operator impact |
|---|---:|---:|---|
| `model_in_channels` | 64 | 64 | Noise channel count and `conv_pre` input channels. |
| `model_hidden_channels` | 32 | 32 | Main residual channel width; LVC generated output is `2 * hidden`. |
| `num_mel_bins` | 100 | 100 | Mel ABI last dimension before model transpose. |
| `resblock_kernel_sizes` | `[3,3,3]` | `[3,3,3]` | LVC generated kernel width and residual Conv1d kernel width per upsample block. |
| `resblock_stride_sizes` | `[8,8,4]` | `[8,8,4]` | ConvTranspose1d upsampling; product is 256 samples per mel frame. |
| `resblock_dilation_sizes` | `[[1,3,9,27]] * 3` | same | Four LVC residual layers per block. |
| `kernel_predictor_num_blocks` | 3 | 3 | Conv1d residual blocks inside each kernel predictor. |
| `kernel_predictor_hidden_channels` | 64 | 64 | Kernel predictor hidden width. Must match residual-block conv channel assumption for current source. |
| `kernel_predictor_conv_size` | 3 | 3 | Predictor residual/output Conv1d kernel size. |
| `kernel_predictor_dropout` | 0.0 | 0.0 | Dropout is identity in inference/eval. |
| `leaky_relu_slope` | 0.2 | 0.2 | All LeakyReLU activations. |
| `torch_dtype` | not in class default | float32 | Checkpoint metadata. |

Representative config sweep:

| Config/checkpoint | Basis | Native HF config? | Key dims | Notes |
|---|---|---:|---|---|
| `dg845/univnet-dev` | HF config + preprocessor | yes | in 64, hidden 32, mel 100, strides 8/8/4 | Common public Transformers checkpoint; matches source defaults. |
| Source default `UnivNetConfig()` | `configuration_univnet.py` | n/a | same as above | Tortoise/c32-style default. |
| Local tiny tester | `test_modeling_univnet.py` | no | in 8, hidden 8, mel 20, predictor hidden 8, default strides | Useful for one-block/operator parity shape tests. |
| `my3bikaht/univnet-RU` | HF API files only | no | unknown | `.pt` original checkpoint only; requires conversion and config inference before admission. |

Preprocessor dimensions for `dg845/univnet-dev`:

| Field | Value | Runtime relevance |
|---|---:|---|
| `sampling_rate` | 24000 | Raw input must be 24 kHz. |
| `hop_length` | 256 | Mel frames map to 256 waveform samples. |
| `win_length` / `filter_length` | 1024 / 1024 | STFT and mel extraction. |
| `num_mel_bins` | 100 | Must match model config. |
| `fmin` / `fmax` | 0 / 12000 | Full-band mel up to Nyquist. |
| `center` | false | Custom reflect padding is applied before STFT. |
| `compression_clip_val` | 1e-5 | `log(clip(mel, min) * factor)`. |
| `do_normalize` | false | Optional CPU-side normalization, not default for checkpoint. |
| `max_length_s` | 10 | Default waveform pad/truncation cap. |
| `pad_end_length` | 10 frames | Optional end-padding for artifact reduction. |

## 3a. Family variation traps

- This family is not attention-based; no KV cache, RoPE, logits, tokenizer, or text decode should be admitted.
- The model source accepts unbatched `input_features` and/or `noise_sequence` and inserts a batch dimension. DinoML can initially require explicit rank-3 batched inputs for a simpler ABI.
- Batch broadcast is asymmetric: batch-1 spectrogram can be repeated to multi-noise batch and batch-1 noise can be repeated to multi-spectrogram batch. Initial DinoML can reject mixed batch sizes except exact equality.
- The kernel predictor residual block uses `config.model_in_channels` as its channel width, while the preceding `input_conv` outputs `kernel_predictor_hidden_channels`. Source defaults keep both at 64; non-matching configs should be rejected unless source behavior is intentionally mirrored and tested.
- `resblock_kernel_sizes`, `resblock_stride_sizes`, and `resblock_dilation_sizes` only validate equal outer length. The source does not validate odd LVC kernels, but local convolution padding and same-length assumptions are cleanest when kernels are odd.
- LVC `location_variable_convolution` has a `dilation` parameter but callers never pass the residual block dilation into it; the custom local convolution always uses dilation 1 in current source. The static Conv1d before it uses the configured dilation.
- The feature extractor returns `input_features [B,T_mel,M]` and `noise_sequence [B,T_mel,C]`, but the model immediately transposes both to channel-first `[B,M,T]` and `[B,C,T]`.
- `padding_mask` from the feature extractor is sample-level raw waveform length, not mel-frame attention. The model only sums it for output trim metadata; it does not mask neural computation.
- `Conv1d(..., padding_mode="reflect")` appears in `conv_pre` and `conv_post`; most internal Conv1d uses zero padding.
- Source uses `torch.channels_last_3d` inside the LVC temporary before bias add and flatten. Treat as a local layout optimization opportunity, not a semantic requirement for the public ABI.
- Converted original checkpoints may contain weight-norm parameterization names (`weight_g`, `weight_v`); conversion applies weight norm, loads, then removes it for inference. DinoML should load dense folded weights after conversion, or implement the same fold explicitly.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank guards and optional unsqueeze for unbatched `[T,M]` and `[T,C]` inputs, or an initial stricter rank-3 ABI.
- `transpose(2,1)` from `[B,T,C]` to `[B,C,T]`.
- `repeat` for batch-1 broadcast, if admitted.
- `view`/`contiguous` for generated kernel/bias reshape.
- `slice` for generated kernel layers and gated split.
- `squeeze(1)` for final waveform.
- `sum(padding_mask, dim=1)` for trim metadata.
- `unfold` along temporal axes, `transpose`, `contiguous`, and flatten inside LVC.
- Constant and reflect padding.

Neural network primitives:

- Conv1d NCT: `conv_pre 64 -> 32, kernel 7, stride 1, reflect pad 3, bias`.
- ConvTranspose1d NCT per LVC block: `32 -> 32, kernel 2*stride, stride stride, padding stride//2 + stride%2, output_padding stride%2, bias`. Default strides produce exact temporal lengths `T -> 8T -> 64T -> 256T`.
- Kernel predictor per LVC block:
  - `input_conv 100 -> 64, kernel 5, zero pad 2, bias`.
  - 3 residual predictor blocks: dropout, Conv1d `64 -> 64, kernel 3, pad 1`, LeakyReLU, Conv1d `64 -> 64`, LeakyReLU, residual add.
  - `kernel_conv 64 -> 24576, kernel 3, pad 1, bias` for default `32 * 64 * 3 * 4`.
  - `bias_conv 64 -> 256, kernel 3, pad 1, bias` for default `64 * 4`.
- LVC residual block repeated per dilation:
  - LeakyReLU.
  - dilated Conv1d `32 -> 32, kernel 3, dilation in {1,3,9,27}, zero pad dilation, bias`.
  - LeakyReLU.
  - location-variable convolution from `32` input channels to `64` output channels.
  - gated activation `sigmoid(first 32 channels) * tanh(last 32 channels)`.
  - residual add.
- `conv_post 32 -> 1, kernel 7, reflect pad 3, bias`, then `tanh`.

Custom/dynamic-kernel ops:

- Location-variable convolution:
  - Input `hidden_states [B, Cin=32, L]`.
  - Kernel `kernel [B, Cin=32, Cout=64, K=3, T_mel]`.
  - Bias `bias [B, Cout=64, T_mel]`.
  - Guard `L == T_mel * hop_size`, where hop sizes are cumulative strides `[8,64,256]`.
  - Source computes window extraction and `einsum("bildsk,biokl->bolsd")`, adds bias, and flattens to `[B,Cout,L]`.

Preprocessing-coupled ops:

- Mono waveform validation, right padding/truncation, optional end padding.
- Reflect padding by `(n_fft - hop_length)/2 = 384` samples each side for default config.
- STFT with Hann window, `n_fft=1024`, `hop=256`, `center=false`.
- Magnitude `sqrt(real^2 + imag^2 + mel_floor)`, mel matrix multiply, `log(clip(mel, min=1e-5) * compression_factor)`.
- Optional affine normalization to `[-1,1]`.
- Standard Gaussian noise generation `[T_mel, model_in_channels]`.

Attention, position, cache, tokenizer, quantized metadata, distributed ops:

- Not required for primary target. There is no attention, RoPE, KV cache, logits head, tokenizer, embedding table, or source-coupled quantized weight format in the inspected Transformers source.

## 5. Layer/block breakdown

Default forward path with `B` batch, `T` mel frames:

```text
input_features: [B,T,100]
noise_sequence: [B,T,64] or generated randn

noise = transpose -> [B,64,T]
mel = transpose -> [B,100,T]

x = reflect Conv1d(64 -> 32, k=7, pad=3)(noise)       # [B,32,T]

LVC block 0, stride 8, hop 8:
  x = LeakyReLU(x)
  x = ConvTranspose1d(32 -> 32, k=16, stride=8, pad=4)(x)  # [B,32,8T]
  kernels,biases = KernelPredictor(mel)
  for dilation in [1,3,9,27]:
    residual = x
    x = LeakyReLU(x)
    x = Conv1d(32 -> 32, k=3, dilation=dilation, pad=dilation)(x)
    x = LeakyReLU(x)
    x = LVC(x, kernel_i [B,32,64,3,T], bias_i [B,64,T], hop=8)
    x = sigmoid(x[:, :32, :]) * tanh(x[:, 32:, :])
    x = residual + x

LVC block 1, stride 8, hop 64:
  same pattern, output [B,32,64T]

LVC block 2, stride 4, hop 256:
  same pattern, output [B,32,256T]

x = LeakyReLU(x)
x = reflect Conv1d(32 -> 1, k=7, pad=3)(x)
waveform = tanh(x).squeeze(1)                         # [B,256T]
waveform_lengths = sum(padding_mask, dim=1), optional
```

Every convolution has bias in source. Weight norm is not a runtime op for converted inference checkpoints after `remove_weight_norm()`.

## 6. Attention requirements

No attention is required. UnivNet is a convolutional neural vocoder with mel-conditioned dynamic local convolution. Causal/noncausal attention, self/cross attention, MHA/MQA/GQA, masks, packed/varlen attention, sliding windows, ALiBi/RoPE, SDPA/FlashAttention, and KV cache are not applicable.

The only "mask" input is `padding_mask`, which is not an attention mask in the neural graph. It is summed after waveform generation to support CPU-side trimming.

## 7. Position encoding and custom math

No position encoding is present. Temporal location enters through Conv1d/ConvTranspose1d, STFT frame order, and LVC's per-frame generated kernels.

Short LVC math sketch:

```python
def lvc(x, kernel, bias, hop):
    # x: [B, Cin, T_mel * hop]
    # kernel: [B, Cin, Cout, K, T_mel]
    pad = (K - 1) // 2
    x = zero_pad_time(x, pad, pad)
    windows = unfold_time(x, size=hop + 2 * pad, step=hop)
    windows = unfold_last(windows, size=1, step=1)[:, :, :, :, :hop]
    windows = windows.transpose(3, 4)
    patches = unfold_last(windows, size=K, step=1)
    y = einsum("bildsk,biokl->bolsd", patches, kernel)
    y = y + bias[:, :, :, None, None]
    return y.contiguous().view(B, Cout, T_mel * hop)
```

In current source, the LVC function's own `dilation` argument remains default `1`; only the preceding static Conv1d uses configured dilation.

## 8. Preprocessing and input packing

Raw-audio contract:

- Input must be mono waveform samples. Batched 2-D numpy arrays are accepted; arrays with more than 2 dims are rejected.
- Caller should provide `sampling_rate`; mismatch with configured `24000` raises.
- Padding defaults to max length `max_length_s * sampling_rate = 240000` samples with right padding and truncation enabled.
- Optional `pad_end=True` appends `pad_length * hop_length`, default `10 * 256 = 2560`, before normal padding.

Feature tensor contract:

- `input_features`: log-mel spectrogram `[B,T_mel,100]`, float32 by feature extractor, same dtype/device as caller if manually supplied to model.
- `noise_sequence`: standard Gaussian `[B,T_mel,64]`; generated by feature extractor by default, or generated inside `forward` with `torch.randn` if omitted.
- `padding_mask`: sample-level `[B,T_samples]` int/bool mask from waveform padding. It is only summed to produce trim lengths.

CPU/data pipeline versus GPU/runtime:

- STFT/mel/noise generation can remain CPU/data pipeline for first integration.
- First DinoML runtime should accept precomputed `input_features` and explicit `noise_sequence`, avoiding RNG in graph. Later integration can add deterministic RNG or keep noise generation outside runtime.
- `batch_decode` is postprocessing: detach/copy waveform to CPU and trim each sample to `waveform_lengths`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ConvTranspose1d exact upsample lowering

Source pattern:

```text
ConvTranspose1d(C -> C, kernel=2*s, stride=s, padding=s//2+s%2, output_padding=s%2)
```

Replacement:

```text
provider ConvTranspose1d, or zero-insert upsample -> Conv1d equivalent
```

Preconditions:

- 1-D NCT contiguous input.
- Static stride from config.
- `kernel_size == 2 * stride`.
- `padding == stride//2 + stride%2`.
- `output_padding == stride%2`.
- groups = 1, dilation = 1.

Shape equation:

```text
L_out = (L_in - 1) * s - 2p + (2s - 1) + output_padding + 1 = L_in * s
```

Failure cases: odd/custom transposed-conv settings, non-contiguous layout, dynamic config not matching source formula.

Parity test sketch: compare each default stride block against PyTorch for random `[B,32,L]`, including `s=8` and `s=4`.

### Rewrite: predictor Conv1d to temporal GEMM/im2col

Source pattern: small-kernel Conv1d over `[B,C,T]` with padding preserving `T`.

Replacement:

```text
Pad1d -> temporal im2col [B*T, C*K] -> GEMM(weight.T) -> reshape [B,Cout,T]
```

Preconditions: static kernel, stride 1, dilation 1 for predictor convs, groups 1, bias present, NCT dense. Weight transform is `w.reshape(out_channels, in_channels * kernel_size)`.

Failure cases: residual LVC Conv1d uses dilation, which needs a dilation-aware im2col guard; reflect padding in `conv_pre`/`conv_post` is not the same as zero-pad predictor convs.

### Rewrite: LVC unfold+einsum to dynamic local-conv kernel

Source pattern:

```text
pad -> unfold by hop -> optional unfold dilation=1 -> transpose -> unfold K -> einsum -> bias -> flatten
```

Replacement:

```text
custom CUDA local convolution:
  for b, out_c, frame, sample_in_hop:
    sum over in_c,k hidden[b,in_c,frame*hop+sample+offset(k)] * kernel[b,in_c,out_c,k,frame] + bias[b,out_c,frame]
```

Preconditions:

- `in_length == T_mel * hop`.
- `K` odd for symmetric same padding in the specialized kernel.
- LVC dilation argument is 1, as in current caller.
- Hidden and kernel dense contiguous or provider-declared layouts.

Shape equations:

```text
kernel: [B,Cin,Cout,K,T]
bias: [B,Cout,T]
output: [B,Cout,T*hop]
```

Layout constraints: source public ABI is NCT after transpose. A custom kernel can choose an internal NHWC/NLC-like layout only inside the LVC region if adjacent Conv1d/activation consumers are also rewritten.

Failure cases: caller-provided LVC dilation other than 1, non-default kernel parity, very large hop requiring excessive temporary materialization, batch broadcasting not normalized before launch.

Parity test sketch: compare custom kernel to PyTorch `location_variable_convolution` for default hops `[8,64,256]`, random kernels/biases, and small `T`.

### Rewrite: gated activation fusion

Source pattern:

```text
sigmoid(x[:, :H, :]) * tanh(x[:, H:, :])
```

Replacement:

```text
split + fused sigmoid/tanh/mul kernel, optionally fused into LVC epilogue
```

Preconditions: LVC output channel count exactly `2H`; split dimension is channel axis 1 in NCT layout.

Failure cases: layout pass changes channel axis without rewriting slices.

### Layout guidance

Initial semantic lowering should preserve NCT inside model because source uses `Conv1d`, `ConvTranspose1d`, channel-axis splits, and `padding_mode` semantics. Candidate optimized regions:

- Kernel predictor Conv1d chain can be internally lowered to `[B,T,C]` GEMM-like layout if all pads/conv/residual ops in that chain are controlled and output reshape is preserved.
- LVC custom kernel may prefer frame-major layout for generated kernels and output, but it must preserve the source flatten order from `[B,Cout,T,hop,1] -> [B,Cout,T*hop]`.
- `channels_last_3d` in source is local to LVC temporary bias addition. Treat it as an implementation hint, not a public layout ABI.

No-layout-translation guard regions:

- `conv_pre` and `conv_post` with reflect padding.
- Channel split for gated activation.
- `padding_mask` sum: axis is sample axis 1, unrelated to mel/channel layout.

## 10. Kernel fusion candidates

Highest priority:

- LVC dynamic local convolution: this is the unique expensive op and current PyTorch eager path materializes large unfold temporaries.
- Conv1d/ConvTranspose1d NCT provider coverage with reflect and zero padding: required for any runtime parity.
- LVC epilogue fusion: dynamic local convolution + bias + sigmoid/tanh gate + residual add.

Medium priority:

- Predictor Conv1d chains lowered to temporal GEMM or fused small Conv1d kernels.
- ConvTranspose1d + following LeakyReLU/Conv1d scheduling to reduce memory traffic.
- Batch-static mel-conditioned kernel predictor caching when the same mel spectrogram is used with different noise samples. This is useful because `kernels,biases` depend only on mel and block config.

Lower priority:

- CPU STFT/mel acceleration inside DinoML. It is useful for end-to-end serving but not required for model graph parity.
- Weight-norm load-time folding. Conversion already removes weight norm for the HF checkpoint.
- Mixed precision tuning. Start with fp32 parity, then test fp16 only after Conv1d/LVC accumulation behavior is defined.

## 11. Runtime staging plan

Stage 1: parse `UnivNetConfig` and load dense converted HF weights for `dg845/univnet-dev`. Reject non-native `.pt` blobs without an explicit conversion path.

Stage 2: implement model ABI with explicit batched `input_features [B,T,100]` and `noise_sequence [B,T,64]`. Defer in-graph RNG and batch-1 broadcast.

Stage 3: add Conv1d, ConvTranspose1d, LeakyReLU, sigmoid, tanh, add, transpose, reshape/slice, and reflect/constant pad parity for one tiny block.

Stage 4: implement LVC as a reference helper or custom provider, with strict guards on `K`, hop, shapes, and source flatten order.

Stage 5: full generator parity for `dg845/univnet-dev` using precomputed mel/noise.

Stage 6: add feature extractor parity as a CPU/data-pipeline component: STFT/mel/log compression/noise/padding metadata.

Stage 7: optimize LVC and Conv1d/ConvTranspose1d layouts/fusions; then evaluate fp16 or mixed precision.

Initial stubs: discriminator/training, RNG inside compiled graph, batch broadcasting, `batch_decode` trimming, and original `.pt` conversion can be outside the first compiled runtime.

## 12. Parity and validation plan

- Config admission tests:
  - accept source default and `dg845/univnet-dev`.
  - reject `kernel_predictor_hidden_channels != model_in_channels` until source-compatible behavior is proven.
  - reject missing `config.json` original checkpoints unless converted.
- Operator tests:
  - Conv1d zero and reflect padding against PyTorch.
  - ConvTranspose1d default stride shapes and values.
  - LVC reference against source for small `B,T,C,K,hop` and default hops.
  - gated activation split along channel axis.
- Block parity:
  - kernel predictor output shapes `[B,4,32,64,3,T]` and `[B,4,64,T]` for default config.
  - one `UnivNetLvcResidualBlock`.
  - one `UnivNetLvcBlock` for each stride/hop.
- Full model parity:
  - local tiny tester config with `B=2,T=7,in=8,hidden=8,mel=20` should output `[2,1792]`.
  - `dg845/univnet-dev` random mel/noise smoke should match the Transformers integration statistics/slices.
  - end-to-end feature extractor + model using librispeech dummy should match output mean/std/slice if the data dependency is available.
- Tolerances:
  - fp32: `rtol=1e-4, atol=1e-4` for full model; tighter for isolated simple ops.
  - fp16: start with `rtol=1e-2, atol=1e-2` only after accumulation policy is explicit.

## 13. Performance probes

- Feature extraction throughput: raw seconds/sec for STFT + mel + log compression.
- Generator throughput split by stage:
  - `conv_pre`.
  - each ConvTranspose1d.
  - each kernel predictor.
  - each LVC residual block.
  - `conv_post`.
- LVC probe sweep over `B`, `T_mel`, hop `{8,64,256}`, `Cin=32`, `Cout=64`, `K=3`.
- Temporary memory probe comparing unfold/einsum materialization versus custom kernel.
- Batch sweep with same mel and multiple noise samples to quantify kernel predictor caching opportunity.
- Sequence length sweep for 1 s, 5 s, 10 s audio at 24 kHz.
- fp32 versus fp16/bf16 conv and LVC accumulation comparison.

## 14. Skip/defer list

- Training and discriminator.
- Multi-resolution spectrogram discriminator and GAN losses.
- Autoregressive generation controller, beam search, sampling logits, tokenizer behavior.
- KV cache and attention kernels.
- In-graph RNG for `noise_sequence`; require explicit noise first.
- Original `.pt` checkpoint conversion beyond documented weight-key mapping.
- General dynamic local convolution API for arbitrary dilation or unusual kernels.
- Multi-GPU/tensor parallel.
- Full audio preprocessing on GPU.
- Batch-1 broadcast and unbatched convenience ABI, unless needed by an early caller.

## 15. Final implementation checklist

- [ ] Parse `UnivNetConfig` and `UnivNetFeatureExtractor` configs.
- [ ] Admit only `model_type="univnet"` and `architectures=["UnivNetModel"]` for native HF checkpoints.
- [ ] Add config guards for matched predictor hidden width and stride/kernel/dilation lengths.
- [ ] Load dense folded Conv1d/ConvTranspose1d weights from converted HF checkpoints.
- [ ] Implement explicit batched ABI: `input_features [B,T,M]`, `noise_sequence [B,T,C]`.
- [ ] Implement/validate Conv1d with zero and reflect padding.
- [ ] Implement/validate ConvTranspose1d shape-exact upsampling.
- [ ] Implement LeakyReLU, sigmoid, tanh, add, mul, transpose, slice, view/contiguous semantics.
- [ ] Implement LVC reference helper with source shape guards.
- [ ] Add LVC custom CUDA/provider kernel plan.
- [ ] Add LVC epilogue fusion: bias + gate + residual.
- [ ] Add one-block and full-generator parity tests against Transformers.
- [ ] Add optional CPU feature-extractor parity for STFT/mel/noise/padding metadata.
- [ ] Benchmark LVC, kernel predictor, ConvTranspose1d, and full generator.
