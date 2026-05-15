# MobileNetV1 DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/mobilenet_v1_1.0_224; google/mobilenet_v1_0.75_192
Config source: HF config.json/preprocessor_config.json snapshots plus MobileNetV1Config defaults
Source files inspected:
- transformers/src/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py
- transformers/src/transformers/models/mobilenet_v1/configuration_mobilenet_v1.py
- transformers/src/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py
- transformers/src/transformers/models/mobilenet_v1/image_processing_pil_mobilenet_v1.py
- transformers/src/transformers/models/mobilenet_v1/convert_original_tf_checkpoint_to_pytorch.py
Any missing files or assumptions:
- No remote code is required.
- google/mobilenet_v1_0.5_160 and google/mobilenet_v1_0.25_128 returned 401 for raw config/preprocessor files:
  [0.5_160](https://huggingface.co/google/mobilenet_v1_0.5_160), [0.25_128](https://huggingface.co/google/mobilenet_v1_0.25_128).
- Widths for gated 0.5/0.25 variants are inferred from source/conversion naming rules, not confirmed config snapshots.
```

Snapshots are under `agents/plans/transformers/mobilenet_v1/_sources/`.

## 2. High-level architecture

MobileNetV1 is a non-transformer CNN image encoder plus optional image-classification head. The primary target here is image classification; feature extraction is also implemented by `MobileNetV1Model`.

```text
image CPU preprocessing -> NCHW pixel_values -> conv stem -> 13 depthwise-separable conv blocks -> global average pool -> dropout -> Linear classifier -> logits
```

The modeling source uses NCHW tensors throughout: `[batch, channels, height, width]`. NHWC is an optimization/layout-pass candidate inside controlled convolution blocks, not the default semantic graph translation.

## 3. Important config dimensions

| Field | Source/default | Confirmed values / notes |
| --- | ---: | --- |
| `num_channels` | config | 3 |
| `image_size` | config | 224 for `1.0_224`; 192 for `0.75_192`; source default 224 |
| `depth_multiplier` | config | 1.0, 0.75 confirmed; 0.5/0.25 inferred from checkpoint names |
| `min_depth` | config | 8 |
| `hidden_act` | config | `relu6` |
| `tf_padding` | config | `true` in confirmed configs; dynamic TensorFlow SAME padding before conv |
| `classifier_dropout_prob` | config | source default 0.999; confirmed checkpoints use 0.001 |
| `layer_norm_eps` | config name used as BN eps | 0.001 in source default and confirmed configs |
| `num_labels` | config/HF labels | 1001 for official ImageNet checkpoints, with background class at index 0 |
| `dtype` | config.json | `float32` |
| cache / attention | source | none |

Representative checkpoint sweep:

| Checkpoint | Access | Resolution | Resize shortest edge | Depth multiplier | Stem channels | Final channels | Final map for native crop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/mobilenet_v1_1.0_224` | confirmed | 224 | 256 | 1.0 | 32 | 1024 | 7x7 |
| `google/mobilenet_v1_0.75_192` | confirmed | 192 | 224 | 0.75 | 24 | 768 | 6x6 |
| `google/mobilenet_v1_0.5_160` | 401 gap | 160 inferred | 192 inferred from conversion | 0.5 inferred | 16 inferred | 512 inferred | 5x5 inferred |
| `google/mobilenet_v1_0.25_128` | 401 gap | 128 inferred | 160 inferred from conversion | 0.25 inferred | 8 inferred | 256 inferred | 4x4 inferred |

Channel schedule for confirmed `1.0_224` pointwise outputs:

```text
stem 32
blocks: 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024
```

For other multipliers, source computes `max(int(base_channels * depth_multiplier), min_depth)`.

## 3a. Family variation traps

- This family is not attention-based and has no KV cache, position encoding, sequence packing, or generation path.
- `tf_padding=True` is dynamic shape-dependent padding computed from input `H/W`, stride, and kernel size before each conv. A static `padding=1` rewrite is only valid for guarded shapes where it matches TensorFlow SAME exactly.
- Source uses the config field name `layer_norm_eps` as `BatchNorm2d.eps`; there is no LayerNorm.
- The source output stride is fixed at 32. Original MobileNet output-stride/dilation variants are not implemented by this Transformers source.
- `classifier_dropout_prob` is 0.999 in config class defaults but 0.001 in official converted checkpoints. Use checkpoint config for loaded models.
- Official checkpoints predict 1001 labels, not 1000, because label index 0 is background.
- Hidden-state output returns every layer in `self.layer`, not the conv stem and not stage-selected features: 26 tensors for 13 depthwise + 13 pointwise conv layers.
- Source tensors are NCHW; axis-sensitive operations include BatchNorm/channel axis, `torch.flatten(..., start_dim=1)`, adaptive average pool over spatial axes, and any concat/reduction introduced by lowering.
- Quantized original TensorFlow checkpoints are explicitly rejected by the conversion script and are out of scope for this native source.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input/output tensors.
- Dynamic shape reads of `H/W`.
- Constant zero pad with asymmetric per-side pads `(left, right, top, bottom)`.
- Flatten from `[B, C, 1, 1]` to `[B, C]`.
- Optional hidden-state tuple accumulation of 26 NCHW tensors.

Neural network primitives:

- `Conv2d(3 -> C0, kernel=3, stride=2, groups=1, bias=False)`.
- Depthwise `Conv2d(C -> C, kernel=3, stride in {1,2}, groups=C, bias=False)`.
- Pointwise `Conv2d(Cin -> Cout, kernel=1, stride=1, groups=1, bias=False)`.
- `BatchNorm2d(C, eps=0.001, momentum=0.9997, affine=True, running stats=True)`.
- `ReLU6`.
- `AdaptiveAvgPool2d((1, 1))`.
- Inference dropout identity for classifier path.
- `Linear(final_channels -> num_labels)` with bias.

Attention / position / generation:

- Not required. No attention masks, positional encodings, caches, sampling, or decode operators.

Preprocessing-coupled ops:

- Resize shortest edge with bilinear resampling.
- Center crop to checkpoint resolution.
- Rescale by `1/255`.
- Normalize by mean/std, confirmed official preprocessors use mean `[0.5, 0.5, 0.5]` and std `[0.5, 0.5, 0.5]`.
- Produce `pixel_values` in NCHW by default.

Layout optimization candidates:

- Conv/BN/ReLU6 regions can run NHWC internally if all conv, BN, activation, and padding consumers in the region are translated together.
- Public model inputs, hidden states, `last_hidden_state`, and classifier flatten semantics should be protected by a no-layout-translation guard unless the consumer ABI is also updated.

## 5. Layer/block breakdown

Base model:

```text
pixel_values: [B, 3, H, W]
x = tf_same_pad_if_enabled(x, Conv3x3 stride=2)
x = Conv2d(3 -> stem_channels, k=3, s=2, bias=False)
x = BatchNorm2d(stem_channels, eps=0.001)
x = ReLU6(x)

for i in 0..12:
  s = [1,2,1,2,1,2,1,1,1,1,1,2,1][i]
  residual_in_channels = current_channels
  x = tf_same_pad_if_enabled(x, depthwise Conv3x3 stride=s)
  x = DepthwiseConv2d(C -> C, k=3, s=s, groups=C, bias=False)
  x = BatchNorm2d(C, eps=0.001)
  x = ReLU6(x)
  hidden_state append after depthwise layer if requested
  x = Conv2d(C -> next_channels, k=1, s=1, bias=False)
  x = BatchNorm2d(next_channels, eps=0.001)
  x = ReLU6(x)
  hidden_state append after pointwise layer if requested

last_hidden_state: [B, final_channels, ceil32(H), ceil32(W)] when tf SAME input path is used
pooled = AdaptiveAvgPool2d(1)(last_hidden_state) -> [B, final_channels, 1, 1]
pooler_output = flatten(pooled, start_dim=1) -> [B, final_channels]
```

For native crops, final shapes are `[B, 1024, 7, 7]` for `1.0_224` and `[B, 768, 6, 6]` for `0.75_192`.

Classification head:

```text
pooler_output: [B, final_channels]
x = Dropout(p=classifier_dropout_prob, inplace=True)  # identity in eval
logits = Linear(final_channels -> num_labels, bias=True)(x)
```

## 6. Attention requirements

No attention is required. The primary image-classification target has:

- no causal or noncausal attention,
- no self-attention or cross-attention,
- no masks,
- no packed/varlen sequence metadata,
- no sliding-window/local attention,
- no ALiBi/RoPE/relative bias,
- no KV cache or FlashAttention/SDPA path.

## 7. Position encoding and custom math

There is no explicit position encoding. The custom math DinoML must reproduce is TensorFlow SAME padding for convolutions when `tf_padding=True`.

```python
def mobilenet_v1_tf_same_pad(h, w, kernel_h, kernel_w, stride_h, stride_w):
    if h % stride_h == 0:
        pad_h = max(kernel_h - stride_h, 0)
    else:
        pad_h = max(kernel_h - (h % stride_h), 0)
    if w % stride_w == 0:
        pad_w = max(kernel_w - stride_w, 0)
    else:
        pad_w = max(kernel_w - (w % stride_w), 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom
```

Pads depend on dynamic input spatial dimensions and per-layer stride/kernel. They can be precomputed only for fixed admitted resolutions.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Convert image to processor backend representation; RGB conversion default is backend-driven because `do_convert_rgb=None`.
- Resize shortest edge: 256 for `1.0_224`, 224 for `0.75_192`; conversion script uses `image_size + 32`.
- Center crop to `image_size x image_size`.
- Rescale by `0.00392156862745098`.
- Normalize with mean/std `[0.5, 0.5, 0.5]`.
- Emit `pixel_values` as `[B, 3, image_size, image_size]`.

GPU/runtime:

- Consumes only `pixel_values`.
- No token IDs, masks, image patch metadata, or packed sequence descriptors.
- Postprocessing for ImageNet classification is `argmax`/top-k over 1001 logits with label index 0 as background. No NMS, boxes, masks, or segmentation resize rules.

## 9. Graph rewrite / lowering opportunities

### Rewrite: BatchNorm fold into Conv2d

Source pattern:

```text
Conv2d(bias=False) -> BatchNorm2d(eps=0.001, affine=True, running stats=True)
```

Replacement:

```text
Conv2d(bias=True) with folded weights/bias
```

Preconditions:

- Inference/eval mode only.
- BatchNorm running mean/variance available and frozen.
- Preserve groups for depthwise conv.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = conv_w * scale.reshape(-1, 1, 1, 1)
b_fold = beta - running_mean * scale
```

Layout constraints:

- Weight remains OIHW for NCHW conv lowering; NHWC kernels need matching OIHW-to-provider transform.

Failure cases:

- Training mode, missing running stats, or unfrozen BN.

Parity test sketch:

- Compare one folded stem and one folded depthwise block against source for random NCHW input in fp32.

### Rewrite: Conv + BatchNorm + ReLU6 fusion

Source pattern:

```text
tf_same_pad -> Conv2d -> BatchNorm2d -> ReLU6
```

Replacement:

```text
FusedPadConvBiasReLU6 or Pad -> FusedConvBiasReLU6
```

Preconditions:

- BN already folded or fused kernel consumes BN parameters.
- Pad semantics match TensorFlow SAME.
- Activation is exactly `relu6`.

Shape equations:

- With TF SAME, output spatial is `ceil(input / stride)` for stride 1/2 3x3 convs.
- Pointwise conv keeps spatial size.

Layout constraints:

- Good NHWC candidate only inside a controlled conv region. Public NCHW outputs and hidden states need materialization or a no-layout-translation guard.

Failure cases:

- `tf_padding=False`, non-ReLU6 `hidden_act`, training dropout/BN, or consumers expecting intermediate NCHW hidden states.

Parity test sketch:

- Fixed sizes 224, 192, odd input dimensions, and non-divisible H/W to verify asymmetric padding.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0, groups=1, bias=False)
```

Replacement:

```text
NCHW/NHWC flatten spatial -> MatMul([B*H*W, Cin] x [Cin, Cout]) -> reshape
```

Preconditions:

- Kernel 1x1, stride 1, dilation 1, groups 1.
- Correct activation flatten order for selected layout.
- Optional BN fold handled after GEMM as bias/scale or folded into weights.

Weight transform:

```python
w = conv.weight.reshape(cout, cin).T
```

Failure cases:

- Depthwise convs, non-1x1 kernels, or hidden-state consumers that require exact NCHW intermediate materialization.

### Rewrite: depthwise 3x3 conv optimized kernel

Source pattern:

```text
tf_same_pad -> Conv2d(C -> C, kernel=3, groups=C, stride=1 or 2)
```

Replacement:

```text
DepthwiseConv2d3x3 specialized kernel, optionally NHWC
```

Preconditions:

- `groups == in_channels == out_channels`.
- Kernel 3x3, dilation 1, bias false before BN fold.
- SAME padding exactly reproduced.

Layout constraints:

- NHWC usually improves memory coalescing for depthwise conv but requires channel-axis rewrites for BN/ReLU6 and NCHW materialization at external boundaries.

Failure cases:

- Any future dilation/output-stride variant; current source does not implement those variants.

## 10. Kernel fusion candidates

Highest priority:

- `tf_same_pad + depthwise Conv3x3 + folded BN + ReLU6`: repeated 13 times and dominates MobileNetV1's operator character.
- `1x1 Conv + folded BN + ReLU6`: repeated 13 times and accounts for most channel mixing work.
- NCHW boundary with NHWC internal conv region: useful for depthwise/pointwise fusion, but only behind guarded layout regions.

Medium priority:

- Stem `3x3 Conv + BN + ReLU6`.
- Adaptive average pool to `[B, C]` fused with flatten.
- Classifier `Dropout(eval identity) + Linear` simplification.

Lower priority:

- Hidden-state tuple materialization optimizations, because feature extraction with all 26 intermediate tensors is optional and memory-heavy.
- Generic Conv2d-to-GEMM for 3x3 kernels; specialized depthwise kernels are more important than im2col materialization for this family.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `MobileNetV1ForImageClassification`; reject quantized original TF checkpoints and unsupported output-stride/dilation configs.

Stage 2: implement faithful NCHW graph with dynamic TF SAME padding, Conv2d, BatchNorm2d, ReLU6, adaptive avg pool, flatten, and classifier.

Stage 3: add source parity for one conv layer, one depthwise/pointwise pair, full base model, and logits for accessible checkpoints.

Stage 4: fold BN into convs and validate full-logit parity.

Stage 5: add specialized depthwise and pointwise fused kernels.

Stage 6: add guarded NHWC internal layout regions with NCHW materialization for public outputs and hidden states.

Stage 7: add production benchmark/admission profiles for batch and image-size sweeps.

Initial stubs: labels/loss paths, training dropout, and all hidden-state materialization can be deferred for first image-classification logits parity.

## 12. Parity and validation plan

- Random tensor test for `mobilenet_v1_tf_same_pad` against source formula for divisible and non-divisible H/W.
- Single `MobileNetV1ConvLayer` parity for stem, depthwise stride 1, depthwise stride 2, and pointwise conv.
- BN folding parity for fp32 with tolerances around `rtol=1e-5, atol=1e-5`.
- Full base model parity for `[1,3,224,224]` and `[1,3,192,192]`.
- Hidden-state parity: confirm 26 outputs and exact NCHW shapes.
- Classification parity: logits shape `[B,1001]` for official checkpoints and top-k agreement after processor preprocessing.
- Suggested tolerances: fp32 `1e-4` end-to-end logits; fp16/bf16 `5e-3` to `1e-2` after fused kernels, with top-k agreement checks.

No DinoML tests were run for this report.

## 13. Performance probes

- CPU preprocessing throughput: resize + center crop + normalize.
- NCHW baseline encoder throughput for batch sizes 1, 8, 32, 64.
- Depthwise-only kernel timing by channel count and spatial size.
- Pointwise 1x1 GEMM/conv timing by channel schedule.
- BN-folded versus unfused graph comparison.
- NHWC internal fused region versus faithful NCHW graph.
- Image-size sweep: 128, 160, 192, 224, plus odd/non-divisible dimensions for TF padding overhead.
- Hidden-state materialization memory/time overhead for feature extraction mode.
- End-to-end images/sec and classifier latency with 1001-class head.

## 14. Skip/defer list

- Training, losses, and gradients.
- Quantized original TensorFlow checkpoints and FakeQuantization paths.
- Unsupported original MobileNet output-stride/dilation variants.
- Alternative heads beyond image classification and base feature extraction.
- Multi-GPU/tensor parallelism.
- Attention/cache/generation features; not applicable.
- Full hidden-state materialization in the first optimized path.

## 15. Final implementation checklist

- [ ] Parse `MobileNetV1Config`, including `depth_multiplier`, `min_depth`, `tf_padding`, `layer_norm_eps`, and checkpoint `classifier_dropout_prob`.
- [ ] Load conv, BN, and classifier weights; preserve depthwise group semantics.
- [ ] Implement TF SAME dynamic padding helper.
- [ ] Implement NCHW Conv2d, BatchNorm2d, ReLU6, AdaptiveAvgPool2d, flatten, and Linear path.
- [ ] Implement 13 depthwise-separable block schedule with source channel equations.
- [ ] Add processor contract for resize/crop/rescale/normalize to NCHW `pixel_values`.
- [ ] Add BN-folding rewrite for standard and depthwise convs.
- [ ] Add fused PadConvBNReLU6/depthwise and Conv1x1BNReLU6 kernels.
- [ ] Add guarded NHWC internal layout optimization with NCHW output materialization.
- [ ] Add parity tests for padding, single blocks, full encoder, hidden states, and logits.
- [ ] Benchmark preprocessing, depthwise kernels, pointwise kernels, NCHW vs NHWC, and batch/image-size sweeps.
