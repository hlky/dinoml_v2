# Transformers `bit` DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/bit-50
Config source: https://huggingface.co/google/bit-50/resolve/main/config.json
Source files inspected:
- X:/H/transformers/src/transformers/models/bit/configuration_bit.py
- X:/H/transformers/src/transformers/models/bit/modeling_bit.py
- X:/H/transformers/src/transformers/models/bit/image_processing_bit.py
- X:/H/transformers/src/transformers/models/bit/image_processing_pil_bit.py
- X:/H/transformers/src/transformers/models/bit/convert_bit_to_pytorch.py
Any missing files or assumptions: no gated files encountered. Only one native public `model_type="bit"` checkpoint was found in the official Google namespace. `google/vit-hybrid-base-bit-384` was inspected as a BiT-backbone consumer, not as the primary `bit` runtime target.
```

Snapshots were written under `agents/plans/transformers/bit/_sources/`, including HF configs for:

- `google/bit-50`
- `google/vit-hybrid-base-bit-384`

Primary DinoML target: image classification plus reusable NCHW backbone feature extraction. Dinoml assumptions: inference-only first, CUDA GPU target, NHWC/channel-last preferred only inside guarded local fusion/layout regions, faithful PyTorch-axis translation at graph boundaries.

## 2. High-level architecture

BiT in Transformers is a CNN/ResNetV2-style vision encoder, not an attention model. The primary checkpoint is `BitForImageClassification`: image processor -> NCHW pixel tensor -> weight-standardized convolutional stem -> four preactivation bottleneck stages -> final GroupNorm+ReLU -> adaptive average pool -> linear classifier.

```text
CPU image resize/crop/rescale/normalize -> pixel_values[B,3,H,W] ->
WSConv7x7 stem + pad + maxpool -> ResNetV2/BiT stages ->
final GroupNorm/ReLU -> adaptive avg pool -> flatten -> logits[B,num_labels]
```

The backbone path uses the same encoder and returns selected image-like NCHW feature maps from `stem`, `stage1`, `stage2`, `stage3`, and `stage4` according to `out_features`/`out_indices`.

## 3. Important config dimensions

`google/bit-50` uses `crop_size=448x448`; source defaults use `224x224` image processor defaults, but the checkpoint preprocessor overrides them.

| Field | `google/bit-50` | Source default / notes |
|---|---:|---|
| `model_type` | `bit` | `bit` |
| primary architecture | `BitForImageClassification` | `BitModel`, `BitForImageClassification`, `BitBackbone` |
| `num_channels` | 3 | 3 |
| `embedding_size` | 64 | 64 |
| `hidden_sizes` | `[256,512,1024,2048]` | same |
| `depths` | `[3,4,6,3]` | same |
| total bottleneck blocks | 16 | `sum(depths)` |
| `layer_type` | `preactivation` | `preactivation`; can be `bottleneck` |
| `hidden_act` | `relu` | `relu` |
| `num_groups` | 32 | 32 for GroupNorm |
| `global_padding` | `null` | `None`; source also supports `"SAME"` and `"VALID"` |
| `embedding_dynamic_padding` | `false` | false |
| `drop_path_rate` | 0.0 | 0.0; inference identity even if nonzero |
| `output_stride` | 32 | 32 |
| `width_factor` | 1 | multiplies stage channels via `make_div` |
| classifier labels | 1000 ImageNet labels | from checkpoint config |
| dtype | `float32` | checkpoint config |
| attention/cache | none | not applicable |

Representative checkpoint/config sweep:

| Config | Scope | Operator-significant details |
|---|---|---|
| `google/bit-50` | native `bit` image classifier | preactivation blocks, `global_padding=null`, static WSConv padding, stem pool without dynamic padding, four stages to 2048 channels, 448 preprocessor |
| source default `BitConfig()` | native random/config basis | same architecture dimensions, but processor defaults are 224 unless checkpoint processor overrides |
| `google/vit-hybrid-base-bit-384` nested `backbone_config` | BiT consumed by `vit-hybrid`, not native `bit` target | `layer_type="bottleneck"`, `global_padding="SAME"`, `embedding_dynamic_padding=true`, `depths=[3,4,9]`, `out_features=["stage3"]`, 384 processor; this changes block ordering and dynamic padding behavior |

## 3a. Family variation traps

- BiT is CNN-only despite living in Transformers; do not plan attention, RoPE, KV cache, prefill, or decode support for this family.
- `WeightStandardizedConv2d` standardizes weights at every forward before convolution. This is not ordinary Conv2d unless the standardized weights are pre-materialized under fixed inference weights.
- `conv_layer` appears in `google/bit-50` config and the conversion script, but the inspected modeling source does not read `config.conv_layer`; DinoML should not treat it as an operator switch for this source basis.
- `layer_type` changes block semantics. `preactivation` uses GN+ReLU before each residual conv and a final model GN+ReLU. `bottleneck` uses conv -> GN+ReLU blocks with activation after residual add and no final model norm.
- `global_padding="SAME"` may trigger `DynamicPad2d` for stride/dilation cases that cannot be represented as static symmetric padding. `global_padding=null` uses PyTorch-style static symmetric padding from `get_padding_value`.
- The stem has a source-specific `ConstantPad2d(1)` before maxpool unless `global_padding=="SAME"`, in which case the pad is identity.
- `output_stride` can convert later stage strides into dilation. This changes spatial shapes and the 3x3 conv dilation contract.
- `width_factor` rescales output channels with `make_div`; do not hardcode 256/512/1024/2048 when loading arbitrary configs.
- Backbone feature outputs are NCHW feature maps. NHWC should be a guarded internal layout/fusion choice, not a public ABI change.
- The vit-hybrid nested BiT config uses three depths with four hidden sizes. The native `BitEncoder` zips depths with hidden sizes, so only the first three stages are built for that config; `out_features=["stage3"]` is the relevant consumed feature.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input/output tensors: `pixel_values[B,3,H,W]`, feature maps `[B,C,Hs,Ws]`.
- Static and dynamic 2D padding, including asymmetric SAME padding: `[left,right,top,bottom]`.
- NCHW concat is not used in native Bit blocks; residual add requires identical NCHW shape.
- Flatten pooled `[B,C,1,1] -> [B,C]`.
- Backbone hidden-state tuple selection by stage name/index.

Neural network primitives:

- Weight-standardized Conv2d, all `bias=False` by default in source:
  - stem `WSConv2d(3 -> 64, k=7, stride=2, eps=1e-8)`.
  - bottleneck `1x1`, `3x3`, `1x1` WSConv2d.
  - downsample `1x1` WSConv2d when first layer of each stage.
- GroupNorm over channel axis with 32 groups, affine weights, eps `1e-5`.
- ReLU activation from `hidden_act="relu"`.
- MaxPool2d `k=3, stride=2`, with optional dynamic pad.
- AdaptiveAvgPool2d to `(1,1)`.
- Linear classifier `Linear(2048 -> 1000)` for `google/bit-50`.
- DropPath is identity in inference; training stochastic path can be deferred.

Attention primitives:

- None required.

Position/cache/generation ops:

- None required.

Preprocessing-coupled ops:

- Convert image to RGB.
- Resize shortest edge to 448 for `google/bit-50`, bicubic `resample=2`.
- Center crop to 448x448.
- Rescale by `1/255`.
- Normalize with mean/std `[0.5,0.5,0.5]`.
- Emit NCHW float pixel tensor.

## 5. Layer/block breakdown

For `google/bit-50`, input shape after preprocessing is typically `[B,3,448,448]`.

Stem:

```text
x = WSConv2d(3 -> 64, k=7, stride=2, padding=3, bias=False)(pixel_values)  # [B,64,224,224]
x = ConstantPad2d(1)(x)                                                     # [B,64,226,226]
x = MaxPool2d(k=3, stride=2, padding=0)(x)                                  # [B,64,112,112]
```

Preactivation bottleneck block, repeated by stage:

```text
pre = GroupNorm32+ReLU(x)
shortcut = WSConv1x1(in -> out, stride)(pre) if first block else x
y = WSConv1x1(in -> mid)(pre)
y = WSConv3x3(mid -> mid, stride/dilation)(GroupNorm32+ReLU(y))
y = WSConv1x1(mid -> out)(GroupNorm32+ReLU(y))
out = y + shortcut
```

Stage shapes for 448x448 and `output_stride=32`:

| Stage | Blocks | First-block convs | Output shape |
|---|---:|---|---|
| stage1 | 3 | downsample `64 -> 256`, residual mid 64, stride 1 | `[B,256,112,112]` |
| stage2 | 4 | downsample `256 -> 512`, residual mid 128, stride 2 | `[B,512,56,56]` |
| stage3 | 6 | downsample `512 -> 1024`, residual mid 256, stride 2 | `[B,1024,28,28]` |
| stage4 | 3 | downsample `1024 -> 2048`, residual mid 512, stride 2 | `[B,2048,14,14]` |

Model tail:

```text
x = GroupNorm32+ReLU(x)        # only for layer_type="preactivation"
pooled = AdaptiveAvgPool2d(1)(x)
logits = Linear(2048 -> 1000)(Flatten(pooled))
```

`layer_type="bottleneck"` variant:

```text
shortcut = WSConv1x1(in -> out, stride) + optional GroupNorm(no activation) if first block
y = WSConv1x1(in -> mid)(x); y = GroupNorm32+ReLU(y)
y = WSConv3x3(mid -> mid, stride/dilation)(y); y = GroupNorm32+ReLU(y)
y = WSConv1x1(mid -> out)(y); y = GroupNorm32(no activation)(y)
out = ReLU(y + shortcut)
```

## 6. Attention requirements

No attention is required. There is no causal/noncausal self-attention, cross-attention, mask handling, packed/varlen metadata, RoPE/ALiBi/relative bias, FlashAttention/SDPA path, or KV cache in `modeling_bit.py`. Generation sections are not applicable for the primary target.

## 7. Position encoding and custom math

There is no learned or analytic position encoding. Spatial position is represented only by convolution, pooling, padding, stride, and dilation.

Custom math that DinoML must reproduce or pre-materialize:

```python
def weight_standardize_conv_weight(w, eps=1e-8):
    # Source uses batch_norm(w.reshape(1, out_channels, -1), training=True).
    flat = w.reshape(w.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    var = flat.var(axis=1, keepdims=True, unbiased=False)
    return ((flat - mean) / (var + eps) ** 0.5).reshape(w.shape)
```

Because this depends only on static weights during inference, DinoML may precompute standardized weights at load time if the graph is inference-only and weights are immutable. If weights can mutate or fine-tuning is in scope, it must remain in graph.

Dynamic SAME padding:

```python
def same_pad_2d(h, k, stride, dilation):
    return max((ceil(h / stride) - 1) * stride + (k - 1) * dilation + 1 - h, 0)
```

The dynamic padding amounts depend on runtime spatial size.

## 8. Preprocessing and input packing

CPU/data pipeline for `google/bit-50`:

- Convert input image to RGB.
- Resize with shortest edge 448, bicubic interpolation.
- Center crop to `448x448`.
- Rescale to `[0,1]` by `1/255`.
- Normalize: `(x - 0.5) / 0.5` per channel.
- Return `pixel_values` as float NCHW `[B,3,448,448]`.

GPU/runtime graph begins at `pixel_values`. No token packing, masks, grid metadata, or variable-length descriptors are used.

Classification postprocessing is simple top-k/argmax over logits. Backbone output is a tuple of NCHW feature maps selected by `out_features`/`out_indices`; no NMS, box conversion, mask resizing, or structured postprocess is implemented in this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: inference WeightStandardizedConv2d -> Conv2d with pre-standardized weight

Source pattern:

```text
weight_std = batch_norm(weight.reshape(1, out_channels, -1), training=True, eps)
Conv2d(input, weight_std.reshape_as(weight), bias=None, stride, padding, dilation, groups)
```

Replacement:

```text
Conv2d(input, precomputed_weight_std, bias=None, stride, padding, dilation, groups)
```

Preconditions:

- Inference-only immutable weights.
- `eps` preserved per module; source uses `1e-8` for BiT convs and `1e-6` default only if directly constructing `WeightStandardizedConv2d` without override.
- Standardization over each output channel and all input-channel/kernel/group elements exactly as source reshape produces.
- Conv attributes preserved exactly, including dynamic padding applied before conv.

Shape equations:

- `w: [Cout, Cin/groups, Kh, Kw]`
- `flat: [Cout, (Cin/groups)*Kh*Kw]`
- output shape follows regular Conv2d after any source pad.

Failure cases:

- Mutable weights, training, custom grouped conv variants without parity verification, or provider kernels that use a different variance definition.

Parity sketch:

- Compare standardized weight tensor and Conv2d output against source for stem and each bottleneck conv in fp32; then repeat with bf16/fp16 accumulation policy.

### Rewrite: static SAME/null padding + Conv2d -> fused padded Conv2d

Source pattern:

```text
optional DynamicPad2d or static Conv2d padding -> WSConv2d/Conv2d
```

Replacement:

```text
Conv2d with provider-native explicit padding or fused pad+conv kernel
```

Preconditions:

- For static case, padding from `get_padding_value` is known and symmetric.
- For dynamic SAME, runtime shape-dependent left/right/top/bottom padding must match source `floor/ceil` split.
- `global_padding="VALID"` maps to zero padding.

Layout constraints:

- Source axes are NCHW. NHWC lowering must rewrite channel axis and pad axes only inside a controlled region.

Failure cases:

- Unknown spatial size without dynamic pad support, asymmetric padding unsupported by provider, or consumers expecting materialized padded tensor.

### Rewrite: 1x1 WSConv2d -> GEMM

Source pattern:

```text
NCHW [B,Cin,H,W] -> WSConv2d(Cin -> Cout, k=1, stride=s)
```

Replacement:

```text
if stride == 1: flatten spatial -> MatMul([B*H*W,Cin], Wstd.T) -> reshape
if stride > 1: strided gather spatial -> MatMul -> reshape
```

Preconditions:

- Standardized weights precomputed or standardization fused.
- `kernel_size=1`, `padding=0` or statically known equivalent, `dilation=1`.
- Bias absent or explicitly added if a future config uses bias.

Failure cases:

- Dynamic padding before the conv, non-unit dilation, or stride handling that changes sampling positions.

### Rewrite: AdaptiveAvgPool2d(1) + Flatten + Linear -> spatial mean + GEMM

Source pattern:

```text
AdaptiveAvgPool2d((1,1)) -> Flatten -> Linear(C -> num_labels)
```

Replacement:

```text
ReduceMean over H,W -> MatMul(weight.T) + bias
```

Preconditions:

- Output pool size is exactly `(1,1)`.
- Source layout axis is NCHW; NHWC pass must rewrite reduction axes from `(2,3)` to `(1,2)` only inside controlled layout regions.

Failure cases:

- Backbone mode that needs unpooled feature maps.

### Rewrite: guarded NCHW -> NHWC residual stage fusion

Source pattern:

```text
GroupNorm(channel dim=1) -> ReLU -> WSConv2d -> residual add
```

Replacement:

```text
NHWC internal layout with channel-last GroupNorm/Conv kernels, materialize NCHW only at public outputs
```

Preconditions:

- Entire local stage region and both residual branches are translated together.
- Axis-sensitive ops are rewritten: GroupNorm channel axis `1 -> -1`, Conv weights OIHW -> HWIO/provider layout, spatial pads `(H,W)` preserved, pool/reduce axes updated.
- `BackboneOutput.feature_maps` and `last_hidden_state` are converted back to NCHW if exposed.

Failure cases:

- Partial translation across external consumers, hidden-state capture inside a stage without layout materialization, or provider GroupNorm using a different grouping/channel order. Wrap public backbone extraction points in a conceptual `no_layout_translation()` guard unless materialization is explicit.

## 10. Kernel fusion candidates

Highest priority:

- WSConv2d inference materialization/fusion. Every convolution uses weight standardization; precomputing or fusing it removes a per-forward batch-norm-like weight path.
- GroupNorm+ReLU over NCHW/NHWC. BiT replaces BatchNorm with GroupNorm throughout; this is hot in every bottleneck.
- WSConv1x1/3x3 + adjacent GroupNorm+ReLU scheduling. Residual blocks are almost entirely conv/norm/activation chains.

Medium priority:

- Pad+WSConv fusion for static and dynamic SAME padding. Dynamic padding appears in BiT hybrid configs and can otherwise allocate temporary padded tensors.
- Residual add fusion with final activation for `layer_type="bottleneck"`.
- Adaptive average pool + classifier GEMM for classifier-only inference.

Lower priority:

- DropPath handling can be identity-folded in inference.
- Backbone feature tuple assembly is not compute-heavy but should avoid unnecessary layout copies.
- Width-factor specialization for uncommon wider BiT variants is useful after baseline parity.

## 11. Runtime staging plan

1. Parse `BitConfig`, reject unsupported `layer_type`/`global_padding` values, and record output feature contract.
2. Load weights and implement WSConv2d standardization as a reference path.
3. Run stem parity on NCHW pixel tensors, including static pad and maxpool.
4. Run one preactivation bottleneck block parity, then stage parity for `google/bit-50`.
5. Add final GroupNorm/ReLU, adaptive pool, and classifier logits parity.
6. Add `BitBackbone` feature extraction parity for selected `out_features`.
7. Add `layer_type="bottleneck"` and dynamic SAME padding coverage using the vit-hybrid nested config as a source-derived variant.
8. Enable optimized lowering: pre-standardized conv weights, pad+conv fusion, GroupNorm+ReLU, and guarded NHWC internal stages.

Initially stub/defer labels/loss and training DropPath. For classifier parity, postprocessing can be limited to logits and top-k.

## 12. Parity and validation plan

- Random tensor tests for `weight_standardize_conv_weight` against source math in fp32.
- Dynamic padding tests for odd/even `H,W`, stride 1/2, dilation 1/2.
- Stem parity for `[1,3,224,224]` and `[2,3,448,448]`.
- Single preactivation bottleneck parity for first and non-first blocks.
- Stage parity after stages 1, 2, 3, and 4 with source checkpoint weights.
- Full classifier logits parity for `google/bit-50` on fixed preprocessed pixel tensors.
- Backbone parity for `out_features` selections, verifying NCHW shape and ordering.
- Variant parity for `layer_type="bottleneck"` with `global_padding="SAME"` and `embedding_dynamic_padding=true`.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=5e-2, atol=5e-2` unless conv/norm accumulation is pinned to fp32.

No DinoML tests were run for this audit, per scope.

## 13. Performance probes

- Image preprocessing throughput for 224, 384, and 448 crops.
- Stem-only throughput and pad allocation count.
- Per-stage throughput and activation memory for batch-size sweep.
- WSConv2d reference vs pre-standardized-weight Conv2d.
- GroupNorm+ReLU kernel throughput on NCHW vs guarded NHWC internal layout.
- Dynamic SAME padding overhead on odd/even image sizes.
- Classifier-only tail timing: adaptive avg pool + linear.
- Backbone mode feature extraction memory and layout materialization cost.
- End-to-end image classification images/sec for batch sizes 1, 8, 32, 128.

## 14. Skip/defer list

- Training losses and label handling.
- DropPath stochastic behavior in training.
- Gradient checkpointing and autograd.
- Multi-GPU/tensor parallelism.
- Quantization.
- Remote/timm conversion script execution.
- ViT-hybrid transformer head integration; only its nested BiT backbone config was used as a variation reference.

## 15. Final implementation checklist

- [ ] Parse `BitConfig` including `layer_type`, `global_padding`, `embedding_dynamic_padding`, `output_stride`, `width_factor`, `out_features`
- [ ] Load `google/bit-50` weights and preserve classifier weight/bias
- [ ] Implement `WeightStandardizedConv2d` reference
- [ ] Add inference pre-standardized-weight Conv2d rewrite
- [ ] Implement `DynamicPad2d` and `BitMaxPool2d`
- [ ] Implement GroupNorm32+ReLU with channel-axis guards
- [ ] Implement preactivation bottleneck block
- [ ] Implement bottleneck variant block
- [ ] Implement output-stride stride/dilation schedule
- [ ] Implement final adaptive avg pool + classifier
- [ ] Implement `BitBackbone` stage feature selection
- [ ] Add guarded NHWC internal stage layout/fusion pass
- [ ] Add parity tests for WSConv, padding, stem, one block, stages, logits, and backbone outputs
- [ ] Benchmark preprocessing, WSConv lowering, GroupNorm fusion, batch sweep, and backbone materialization
