# EfficientNet DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/efficientnet-b0, google/efficientnet-b3, google/efficientnet-b7
Config source: official HF config.json and preprocessor_config.json snapshots
Source files inspected:
- X:/H/transformers/src/transformers/models/efficientnet/configuration_efficientnet.py
- X:/H/transformers/src/transformers/models/efficientnet/modeling_efficientnet.py
- X:/H/transformers/src/transformers/models/efficientnet/image_processing_efficientnet.py
- X:/H/transformers/src/transformers/models/efficientnet/image_processing_pil_efficientnet.py
Any missing files or assumptions:
- Primary target is image classification with EfficientNetForImageClassification.
- No attention, generation, tokenizer, cache, remote-code, or custom autoload path is required for the inspected in-library source.
- google/efficientnet-lite0 and google/efficientnet-lite4 returned 401 for config/preprocessor fetches. Links: [lite0](https://huggingface.co/google/efficientnet-lite0), [lite4](https://huggingface.co/google/efficientnet-lite4). Access would confirm whether lite variants only change config dimensions/preprocessing or require separate operator coverage.
```

Snapshots were written under `agents/plans/transformers/efficientnet/_sources/`.

## 2. High-level architecture

EfficientNet is a CNN image encoder with MBConv-style blocks, squeeze-excitation, final convolution, global-like pooling, dropout, and a linear classifier. Source tensors are NCHW throughout the model graph.

```text
CPU/image processor -> pixel_values[N,C,H,W] -> stem Conv/BN/Swish
  -> repeated MBConv blocks with depthwise Conv + SE + projection/skip
  -> top 1x1 Conv/BN/Swish -> global-like pool -> dropout -> classifier logits
```

Stage decomposition:

- CPU/data pipeline: resize, optional center crop, rescale, normalize, optional EfficientNet extra std normalization, output `pixel_values`.
- GPU/runtime encoder: NCHW conv stack, BN, activation, squeeze-excitation, residual additions.
- Head: pool to `[batch, hidden_dim]`, dropout is inactive in eval, linear classifier to `num_labels`.

## 3. Important config dimensions

Source defaults are B7-like: `image_size=600`, `width_coefficient=2.0`, `depth_coefficient=3.1`, `hidden_dim=2560`, `dropout_rate=0.5`.

| Field | B0 | B3 | B7/source default |
|---|---:|---:|---:|
| `image_size` | 224 | 300 | 600 |
| input channels | 3 | 3 | 3 |
| width coefficient | 1.0 | 1.2 | 2.0 |
| depth coefficient | 1.0 | 1.4 | 3.1 |
| rounded top hidden dim | 1280 | 1536 | 2560 |
| MBConv blocks after depth rounding | 16 | 26 | 55 |
| base stage repeats | `[1,2,2,3,3,4,1]` | same | same |
| strides by stage | `[1,2,2,2,1,2,1]` | same | same |
| kernel sizes | `[3,3,5,3,5,5,3]` | same | same |
| expand ratios | `[1,6,6,6,6,6,6]` | same | same |
| squeeze ratio | 0.25 | 0.25 | 0.25 |
| hidden activation | `swish` | `swish` | `swish` |
| batch norm eps/momentum | `0.001/0.99` | `0.001/0.99` | `0.001/0.99` |
| drop connect/dropout | `0.2/0.2` | `0.2/0.3` | `0.2/0.5` |
| pooling type | `mean` | `mean` | `mean`, source also supports `max` |
| dtype from config | float32 | float32 | float32 |
| cache support | none | none | none |

Representative stage sweep after source `round_filters` and `ceil(depth_coefficient * repeats)`:

| Checkpoint | Blocks | Top channels | Stages |
|---|---:|---:|---|
| `google/efficientnet-b0` | 16 | 1280 | `s1 1x 32->16 k3`, `s2 2x 16->24 k3 s2`, `s3 2x 24->40 k5 s2`, `s4 3x 40->80 k3 s2`, `s5 3x 80->112 k5`, `s6 4x 112->192 k5 s2`, `s7 1x 192->320 k3` |
| `google/efficientnet-b3` | 26 | 1536 | `s1 2x 40->24 k3`, `s2 3x 24->32 k3 s2`, `s3 3x 32->48 k5 s2`, `s4 5x 48->96 k3 s2`, `s5 5x 96->136 k5`, `s6 6x 136->232 k5 s2`, `s7 2x 232->384 k3` |
| `google/efficientnet-b7` | 55 | 2560 | `s1 4x 64->32 k3`, `s2 7x 32->48 k3 s2`, `s3 7x 48->80 k5 s2`, `s4 10x 80->160 k3 s2`, `s5 10x 160->224 k5`, `s6 13x 224->384 k5 s2`, `s7 4x 384->640 k3` |

Preprocessor sweep:

| Checkpoint | Resize | Center crop | Rescale/normalize | Notes |
|---|---:|---:|---|---|
| B0 | 224x224 | false | `1/255`, ImageNet mean, EfficientNet std | `include_top` behavior exists in processor class but snapshot omits field, so class default `include_top=True` applies |
| B3 | 300x300 | false | same | `depthwise_padding=[5,18]` changes stride-2 depthwise padding adjustment |
| B7 | 600x600 | false | same | `depthwise_padding=[18]` |

## 3a. Family variation traps

- Source semantic layout is NCHW. NHWC should be a guarded optimization region, especially for Conv/BN/activation/SE blocks; public `pixel_values`, hidden states, and classifier path should remain faithful unless all consumers are controlled.
- `depthwise_padding` does not mean extra padding locations directly; it toggles `adjust_padding = curr_block_num not in depthwise_padding`, changing right/bottom asymmetric padding for stride-2 depthwise convolutions.
- The first block of each stage has `id_skip=True`, which disables residual addition in `EfficientNetFinalBlockLayer`; later repeated stride-1 blocks set `id_skip=False` and perform dropout plus residual add.
- Dropout in MBConv projection is the source implementation of drop-connect/stochastic depth, but in inference/eval it is identity. Keep the module for training parity, but first inference can treat it as no-op.
- `num_hidden_layers = sum(num_block_repeats) * 4` in config is not the actual depth-scaled block count; runtime uses `ceil(depth_coefficient * repeat)` in the encoder.
- `hidden_dim` must equal `round_filters(config, 1280)` for built-in checkpoints; the source top BN is constructed with `config.hidden_dim`, while top conv uses rounded 1280. Reject mismatches.
- `pooling_type` changes `AvgPool2d` vs `MaxPool2d`; unsupported values raise.
- Preprocessor `include_top=True` applies a second divide-by-std normalization after mean/std normalization. If the snapshot omits this field, the class default still applies.
- Lite variants were gated in this run; do not assume their preprocessing or dimensions until configs are available.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW `pixel_values` and feature tensors.
- Explicit asymmetric zero padding: stem `(left=0,right=1,top=0,bottom=1)`; stride-2 depthwise uses `correct_pad(k, adjust)` producing `(k//2-1,k//2,k//2-1,k//2)` or symmetric `(k//2,k//2,k//2,k//2)`.
- Reshape `[B,C,1,1] -> [B,C]` after pool.
- Optional hidden-state tuple materialization for each block, all NCHW.
- NHWC layout-pass candidates need axis rewrites for channel dim `1` to `-1`, BN feature axis, concat/reductions if introduced, and pool spatial axes.

### Neural network primitives

- Conv2d NCHW:
  - stem `Conv2d(3 -> round_filters(32), k=3, stride=2, bias=False)` after asymmetric pad.
  - expansion `1x1 Conv2d(in -> in*expand_ratio, bias=False)` when `expand_ratio != 1`.
  - depthwise `Conv2d(C -> C, groups=C, k=3 or 5, stride=1 or 2, bias=False)`.
  - SE reduce/expand `1x1 Conv2d(C -> max(1,int(input_stage_channels*0.25)) -> C, bias=True)`.
  - projection `1x1 Conv2d(expanded -> out, bias=False)`.
  - top `1x1 Conv2d(last_stage_out -> hidden_dim, bias=False)`.
- BatchNorm2d with eval affine/running mean/variance folding support; eps `0.001`.
- Swish/SiLU activation from `ACT2FN["swish"]`.
- Sigmoid for SE gating and elementwise multiply.
- Residual add for non-first, stride-1 repeats.
- Dropout modules are eval no-ops for inference.
- AvgPool2d/MaxPool2d with kernel size `hidden_dim`, `ceil_mode=True`; for expected final spatial maps this yields `[B,C,1,1]`.
- Linear classifier `Linear(hidden_dim -> num_labels)`, typically `1280/1536/2560 -> 1000`.

### Attention / position / generation / distributed ops

- No attention, no masks, no relative/RoPE/ALiBi position encoding, no KV cache, no generation, no tensor parallel logic in this family.

### Preprocessing-coupled ops

- Bicubic resize to checkpoint size; processor `resample=0` corresponds to PIL bicubic enum in snapshots.
- Optional center crop exists but is false for inspected configs.
- Rescale by `1/255`; optional `rescale_offset` subtracts 1 after rescale if enabled.
- Normalize by ImageNet mean/std, then if `include_top=True`, normalize again with mean 0 and same std.
- Torchvision backend groups images by shape before resize/normalize for batching, then reorders outputs; this is CPU/data-pipeline behavior, not model graph.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: [B,3,H,W] NCHW
x = ZeroPad2d(0,1,0,1)
x = Conv2d(3 -> stem_C, k=3, stride=2, valid, bias=False)
x = BatchNorm2d(stem_C, eps=0.001)
x = Swish(x)
```

MBConv block, repeated by stage:

```text
input x: [B,in_C,H,W]
residual = x
if expand_ratio != 1:
  x = Conv2d(in_C -> in_C*expand_ratio, k=1, same, bias=False)
  x = BatchNorm2d(in_C*expand_ratio)
  x = Swish(x)

if stride == 2:
  x = ZeroPad2d(correct_pad(kernel_size, adjust=block_index not in depthwise_padding))
x = DepthwiseConv2d(C -> C, groups=C, k=3 or 5, stride=1 or 2, same/valid, bias=False)
x = BatchNorm2d(C)
x = Swish(x)

se = AdaptiveAvgPool2d(1)(x)
se = Conv2d(C -> se_C, k=1, bias=True)(se)
se = Swish(se)
se = Conv2d(se_C -> C, k=1, bias=True)(se)
se = Sigmoid(se)
x = x * se

x = Conv2d(C -> out_C, k=1, same, bias=False)
x = BatchNorm2d(out_C)
if stride == 1 and not id_skip:
  x = Dropout(drop_rate)(x)  # identity in eval
  x = x + residual
return x
```

Top and classifier:

```text
x = Conv2d(last_out_C -> hidden_dim, k=1, same, bias=False)
x = BatchNorm2d(hidden_dim)
x = Swish(x)
pooled = AvgPool2d or MaxPool2d(kernel_size=hidden_dim, ceil_mode=True)(x)
pooled = reshape([B,hidden_dim,1,1] -> [B,hidden_dim])
pooled = Dropout(dropout_rate)(pooled)  # identity in eval
logits = Linear(hidden_dim -> num_labels)(pooled)
```

Projection biases: all core convs are `bias=False` except SE `reduce` and `expand`, which use default `bias=True`. The classifier linear has bias by default.

## 6. Attention requirements

No attention is required. EfficientNet has no self-attention, cross-attention, causal masks, packed/varlen attention, local/sliding attention, RoPE/ALiBi/relative bias, KV cache, FlashAttention, or SDPA path.

## 7. Position encoding and custom math

There is no explicit position embedding. Spatial position enters only through convolution, padding, stride, and pooling.

Important source-specific math:

```python
def round_filters(num_channels, width_coefficient, divisor):
    x = num_channels * width_coefficient
    new_dim = max(divisor, int(x + divisor / 2) // divisor * divisor)
    if new_dim < 0.9 * x:
        new_dim += divisor
    return int(new_dim)

def correct_pad(kernel_size, adjust=True):
    c = kernel_size // 2
    if adjust:
        return (c - 1, c, c - 1, c)  # left, right, top, bottom
    return (c, c, c, c)
```

`round_filters` and rounded repeat counts are static from config and can be precomputed at load/compile time. `correct_pad` depends on block index, kernel size, and config `depthwise_padding`, not batch data.

SE math:

```python
def squeeze_excite(x):
    gate = adaptive_avg_pool2d(x, output_size=1)
    gate = swish(conv1x1_reduce(gate))
    gate = sigmoid(conv1x1_expand(gate))
    return x * gate
```

## 8. Preprocessing and input packing

CPU/data pipeline:

- Accepts images, resizes to the checkpoint `size` with bicubic.
- Optional center crop is implemented but false for B0/B3/B7 snapshots.
- Rescales pixel values by `1/255`; with `rescale_offset=True`, subtracts 1 after scaling.
- Normalizes with ImageNet mean `[0.485,0.456,0.406]` and EfficientNet std `[0.47853944,0.4732864,0.47434163]`.
- If `include_top=True`, applies an additional normalization by mean `0` and the same std. Source class default is true.
- Emits `pixel_values` as a batch tensor for model input. The model expects NCHW `[B,3,H,W]`.

GPU/runtime:

- No input packing beyond dense NCHW image tensor.
- No masks, position ids, token ids, grid metadata, or variable-length records.
- End-to-end classification parity requires mapping logits to `id2label`; no NMS or structured postprocessing is present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d eval folding

Source pattern:

```text
Conv2d(bias=False or bias=True) -> BatchNorm2d(eps=0.001, affine=True, eval)
```

Replacement:

```text
Conv2d(folded_weight, folded_bias)
```

Preconditions:

- Inference/eval mode with frozen BN running mean/variance, gamma, beta.
- No observer/fake-quant module between Conv and BN.
- Preserve NCHW semantic output or apply a separately validated NHWC weight layout transform.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = w * scale.reshape(out_channels, 1, 1, 1)
b_fold = beta + (conv_bias - running_mean) * scale
```

Failure cases:

- Training mode BN, dynamic BN stats, or missing running stats.
- `hidden_dim` mismatch causing top BN channel count to differ from top conv output.

Parity test sketch: compare one folded stem, one folded depthwise, one folded projection, and full encoder logits in fp32.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern:

```text
NCHW Conv2d(Cin -> Cout, k=1, stride=1, padding=same)
```

Replacement:

```text
NCHW/NHWC flatten spatial -> MatMul([B*H*W,Cin] x [Cin,Cout]) -> reshape
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `dilation == 1`, `groups == 1`.
- Padding must not change spatial shape.
- Bias handling must match source.
- For NHWC optimized lowering, all adjacent consumers in the local region must accept NHWC or be part of the same guarded pass.

Weight transform:

```python
w_gemm = conv.weight.reshape(Cout, Cin).T
```

Failure cases:

- Depthwise/grouped conv, stride-2 stem/depthwise, or consumer requires public NCHW hidden states.

### Rewrite: depthwise Conv2d NHWC local region

Source pattern:

```text
optional ZeroPad2d -> depthwise Conv2d(groups=C) -> BN -> Swish
```

Replacement:

```text
NHWC pad -> NHWC depthwise kernel -> affine BN -> Swish
```

Preconditions:

- Local block region is fully owned; either no `output_hidden_states` materialization at the internal boundary or NCHW materialization is restored.
- Axis rewrite: channel dim `1` becomes `-1`; BN parameters index channel-last.
- Preserve asymmetric `correct_pad` exactly. For stride-2, padding is explicit before `valid` conv; do not replace with generic SAME unless parity proves identical for odd/even sizes and `depthwise_padding`.
- Depth multiplier is 1 in this source.

Failure cases:

- Public hidden-state outputs requested after each block.
- Unknown lite variant changes depth multiplier or padding contract.

### Rewrite: SE block fused channel gate

Source pattern:

```text
AdaptiveAvgPool2d(1) -> 1x1 Conv -> Swish -> 1x1 Conv -> Sigmoid -> Mul(input)
```

Replacement:

```text
ChannelReduceMean(H,W) -> two small dense/1x1 ops -> Sigmoid -> broadcast multiply
```

Preconditions:

- Dense NCHW or NHWC feature map with known channel axis.
- `output_size=1`; no spatial mask.
- Preserve SE reduce width `max(1, int(stage_input_channels * squeeze_ratio))`, not `int(expanded_channels * ratio)`.

Failure cases:

- Layout pass fails to update reduction axes or broadcast shape.

### Rewrite: eval dropout/drop-connect removal

Source pattern:

```text
Dropout(p=drop_rate) before residual or classifier
```

Replacement:

```text
Identity
```

Preconditions:

- Inference/eval target only.

Failure cases:

- Training parity or stochastic-depth analysis.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BN + Swish fusion for stem, expansion, depthwise, projection/top where applicable. This dominates runtime and removes memory traffic.
- Depthwise Conv2d NHWC kernel with explicit asymmetric padding support. EfficientNet spends many layers in depthwise k3/k5 convs; NHWC is likely the best production layout if boundaries are guarded.
- SE channel gate fusion: reduce mean plus small 1x1/dense ops plus sigmoid/broadcast multiply; frequent and memory-sensitive.

Medium priority:

- 1x1 Conv2d GEMM path for expansion/projection/top/SE pointwise convs.
- Residual add fused with projection BN output for repeated stride-1 blocks.
- Preprocessor affine fusion for rescale/normalize/include_top when running image preprocessing on GPU or batched tensor backend.

Lower priority:

- Pool + reshape + classifier fusion for classification-only throughput.
- Max-pool global variant, because inspected official configs use mean pooling.
- Hidden-state output materialization optimizations; optional for classification inference.

## 11. Runtime staging plan

1. Parse EfficientNet config and implement `round_filters`, rounded repeats, `correct_pad`, stage construction, and config validation that `hidden_dim == round_filters(1280)`.
2. Load weights for stem, one MBConv block, top conv, and classifier; implement NCHW faithful path first.
3. Add full encoder parity for B0 using fp32 eval with dropout as identity.
4. Add B3/B7 parity to cover deeper repeat counts, `depthwise_padding`, and larger hidden dims.
5. Add image processor parity for resize/rescale/normalize/include_top, or declare CPU pipeline dependency.
6. Add graph rewrites: BN folding, eval dropout removal, 1x1 Conv GEMM.
7. Add guarded NHWC fused regions for Conv/BN/Swish and depthwise/SE blocks while preserving public NCHW boundaries and `output_hidden_states`.
8. Benchmark batch/resolution sweeps and choose provider dispatch policy.

Stubbable initially: training losses, dropout randomness, hidden-state tuple outputs, max pooling variant, gated lite checkpoints until accessible.

## 12. Parity and validation plan

- Unit tests for `round_filters`, rounded repeats, and `correct_pad` across B0/B3/B7 configs, including `depthwise_padding=[5,18]` and `[18]`.
- Random tensor parity for stem, expansion, depthwise stride-1, depthwise stride-2 asymmetric pad, SE, projection residual add, and top block.
- BN-folded vs unfused parity in fp32 for each conv type.
- Single MBConv block parity for expand ratio 1 and expand ratio 6.
- After-N-layer parity at stage boundaries for B0 and B3.
- Full encoder last hidden state and pooled output parity.
- Classification logits parity for `google/efficientnet-b0`, then B3/B7.
- Preprocessor parity on a fixed image for torch and PIL backends if both are admitted.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-5`; fp16/bf16 fused kernels `atol=2e-2, rtol=2e-2`, tightened per provider after measurement.

## 13. Performance probes

- CPU image preprocessing images/sec for resize/normalize/include_top by resolution.
- Encoder-only throughput for B0/B3/B7 over batch sizes 1, 8, 32, 64.
- Resolution sweep 224, 300, 600 to expose memory bandwidth and depthwise scaling.
- Conv backend comparison: NCHW faithful vs guarded NHWC fused Conv/BN/Swish.
- Depthwise k3/k5 stride-1/stride-2 microbenchmarks with asymmetric padding.
- SE block microbenchmark by channel count and batch size.
- 1x1 Conv GEMM throughput by stage channel widths.
- End-to-end classification requests/sec including processor and logits.
- Hidden-state materialization overhead when `output_hidden_states=True`.

## 14. Skip/defer list

- Training, labels/loss, and stochastic dropout behavior.
- Quantization and packed-weight formats; no source-coupled packed weights are present.
- Multi-GPU/tensor parallel; not relevant to this CNN source.
- Attention/cache/generation features; not applicable.
- EfficientNet-Lite configs until gated access is available.
- Max-pooling checkpoints unless a representative config requires `pooling_type="max"`.

## 15. Final implementation checklist

- [ ] Parse `EfficientNetConfig` and validate `hidden_dim`.
- [ ] Implement `round_filters`, rounded repeats, and `correct_pad`.
- [ ] Load Conv2d, BatchNorm2d, SE, top conv, and classifier weights.
- [ ] Implement NCHW stem Conv/BN/Swish.
- [ ] Implement MBConv expansion/depthwise/SE/projection/residual block.
- [ ] Implement top Conv/BN/Swish, global-like pool, reshape, dropout identity, classifier.
- [ ] Implement image processor contract or define CPU preprocessing dependency.
- [ ] Add BN folding rewrite.
- [ ] Add eval dropout/drop-connect removal.
- [ ] Add guarded 1x1 Conv -> GEMM rewrite.
- [ ] Add guarded NHWC Conv/BN/Swish and depthwise/SE fusion pass.
- [ ] Add B0/B3/B7 parity tests.
- [ ] Benchmark preprocessing, encoder, depthwise, SE, and end-to-end classification.
