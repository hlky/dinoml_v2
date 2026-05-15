# MobileNetV2 DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/mobilenet_v2_1.0_224 as primary image-classification checkpoint; google/deeplabv3_mobilenet_v2_1.0_513 for segmentation coverage.
Config source: HF config.json/preprocessor_config.json snapshots under _sources/hf_configs.
Source files inspected:
- transformers/src/transformers/models/mobilenet_v2/configuration_mobilenet_v2.py
- transformers/src/transformers/models/mobilenet_v2/modeling_mobilenet_v2.py
- transformers/src/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py
- transformers/src/transformers/models/mobilenet_v2/image_processing_pil_mobilenet_v2.py
Any missing files or assumptions: no gated/401/403 files observed for sampled official Google configs. No Transformers import/execution or DinoML tests were run.
```

Source snapshots are stored in `agents/plans/transformers/mobilenet_v2/_sources/`.

Source URLs at the inspected commit:

- `configuration_mobilenet_v2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilenet_v2/configuration_mobilenet_v2.py
- `modeling_mobilenet_v2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilenet_v2/modeling_mobilenet_v2.py
- `image_processing_mobilenet_v2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py
- `image_processing_pil_mobilenet_v2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilenet_v2/image_processing_pil_mobilenet_v2.py

## 2. High-level architecture

MobileNetV2 is a non-attention CNN encoder with inverted residual blocks. Primary DinoML target should be image classification first; semantic segmentation is source-supported and can stage after the base encoder.

```text
CPU/PIL/torchvision image preprocessing -> NCHW pixel_values
-> conv stem with depthwise separable stage
-> 16 inverted residual blocks
-> final 1x1 conv
-> adaptive average pool -> dropout -> linear classifier
```

Segmentation path:

```text
image preprocessing -> MobileNetV2 encoder without pooler
-> final hidden state before final 1x1 conv via hidden_states[-1]
-> DeepLabV3+-style pooling/ASPP 1x1 head
-> segmentation logits -> optional bilinear resize + argmax postprocess
```

There is no autoregressive prefill/decode, no KV cache, and no text generation controller. The source tensor contract is NCHW throughout modeling code. NHWC should be a guarded layout/fusion optimization inside fully controlled convolution regions, not a default graph translation.

## 3. Important config dimensions

Source defaults from `MobileNetV2Config`:

| Field | Default | Source/runtime meaning |
|---|---:|---|
| `num_channels` | 3 | Input image channels, NCHW model input `[B, 3, H, W]`. |
| `image_size` | 224 | Config metadata and processor target for common classifier checkpoints. |
| `depth_multiplier` | 1.0 | Width multiplier applied through `make_divisible(round(base * multiplier), 8, 8)`. |
| `depth_divisible_by` | 8 | Channel rounding divisor. |
| `min_depth` | 8 | Minimum rounded channel count. |
| `expand_ratio` | 6.0 | Inverted residual expansion width before depthwise conv. |
| `output_stride` | 32 | Encoder spatial stride cap; `8`/`16` replace later strides with dilation. |
| `first_layer_is_expansion` | `true` | Stem skips a separate 1x1 expansion after first 3x3 conv. |
| `finegrained_output` | `true` | Keeps final output at 1280 when `depth_multiplier < 1.0`. |
| `hidden_act` | `relu6` | Activation after conv+BN except projection layers. |
| `tf_padding` | `true` | Runtime computes TensorFlow SAME padding before conv. |
| `classifier_dropout_prob` | 0.8 | Source default; classifier checkpoints override to 0.2. |
| `layer_norm_eps` | 0.001 | Misnamed for this CNN: used as BatchNorm2d eps. |
| `semantic_loss_ignore_index` | 255 | Training loss only; processor also uses 255 for reduced labels. |

Representative checkpoint sweep:

| Checkpoint | Task/head | `depth_multiplier` | `image_size` | Processor resize/crop | `output_stride` | Final conv channels | Labels |
|---|---|---:|---:|---|---:|---:|---:|
| `google/mobilenet_v2_0.35_96` | classification | 0.35 | 96 | shortest edge 128, crop 96 | 32 | 1280 due `finegrained_output` | 1001 |
| `google/mobilenet_v2_1.0_224` | classification | 1.0 | 224 | shortest edge 256, crop 224 | 32 | 1280 | 1001 |
| `google/mobilenet_v2_1.4_224` | classification | 1.4 | 224 | shortest edge 256, crop 224 | 32 | 1792 | 1001 |
| `google/deeplabv3_mobilenet_v2_1.0_513` | semantic segmentation | 1.0 | 513 | shortest edge 545, crop 513 | 8 | encoder head consumes 320-channel pre-final feature | 21 |

Width schedules after source rounding:

| `depth_multiplier` | Stem expanded channels | Projection schedule for stem output + 16 blocks | Final conv |
|---:|---:|---|---:|
| 0.35 | 16 | 8, 8, 8, 16, 16, 16, 24, 24, 24, 24, 32, 32, 32, 56, 56, 56, 112 | 1280 |
| 1.0 | 32 | 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320 | 1280 |
| 1.4 | 48 | 24, 32, 32, 48, 48, 48, 88, 88, 88, 88, 136, 136, 136, 224, 224, 224, 448 | 1792 |

## 3a. Family variation traps

- `depth_multiplier` changes almost every conv channel through `make_divisible`; do not infer channels from checkpoint names without config rounding.
- `finegrained_output=true` keeps the final conv at 1280 for multiplier `< 1.0`, so tiny classifiers still have a wide pooled classifier input.
- Source `tf_padding=true` computes dynamic asymmetric SAME padding from runtime H/W, stride, kernel, and dilation. Replacing it with static symmetric padding is only valid when the equations match.
- `output_stride=8` or `16` changes stride/dilation schedule in depthwise blocks; segmentation uses `output_stride=8`.
- `layer_norm_eps` config field is used as BatchNorm eps, not LayerNorm.
- Projection 1x1 layers in inverted residuals have no activation; expansion/depthwise/final convs use `hidden_act` unless overridden.
- Segmentation head consumes `encoder_hidden_states[-1]`, the 320-channel pre-final projection feature for multiplier 1.0, not `last_hidden_state` after final 1x1.
- Modeling code is NCHW. `torch.cat(..., dim=1)`, BatchNorm channel axis, adaptive pool, flatten `start_dim=1`, and segmentation `argmax(dim=1 or 0 after batch slice)` need no-layout-translation guards unless a layout pass rewrites all axes and consumers.
- No dedicated Transformers `BackboneMixin` wrapper is implemented here. Backbone-like staging is via `MobileNetV2Model(..., output_hidden_states=True)`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor input/output, rank-4 shape propagation.
- Dynamic TensorFlow SAME padding: `Pad2d(left, right, top, bottom)` with pads depending on spatial dims, stride, kernel, dilation.
- Reshape/flatten for pooled classifier: `AdaptiveAvgPool2d(1,1)` -> `flatten(start_dim=1)`.
- Tuple accumulation for optional hidden states.
- NCHW concat along channel axis for segmentation head: `cat([pool_branch, aspp_branch], dim=1)`.

Neural network primitives:

- `Conv2d` with bias false for encoder convs, including regular 3x3, pointwise 1x1, and depthwise 3x3 with `groups=channels`.
- `BatchNorm2d(eps=config.layer_norm_eps, momentum=0.997, affine=true, running_stats=true)`.
- `ReLU6` for main encoder when `hidden_act="relu6"`; segmentation head overrides to plain `relu`.
- Residual add when stride is 1 and `in_channels == out_channels`.
- Dropout for classifier, Dropout2d for segmentation; inference can compile these as identity.
- Linear classifier, e.g. `Linear(1280 -> 1001)` for 1.0/0.35 ImageNet checkpoints and `Linear(1792 -> 1001)` for 1.4.
- Bilinear interpolate for segmentation head pool branch and postprocess/loss resize.
- Cross-entropy training losses are implemented but not required for inference.

Attention primitives:

- None. No self-attention, cross-attention, masks, RoPE, relative bias, FlashAttention, varlen, or KV cache is required.

Preprocessing-coupled ops:

- RGB conversion, resize shortest edge, center crop, rescale by `1/255`, normalize by mean/std, channel-first tensor output.
- Segmentation labels: nearest resize/crop, optional reduce-label remapping, no rescale/normalize, squeeze channel, int64.
- Postprocess semantic segmentation: optional bilinear logits resize with `align_corners=False`, then argmax.

## 5. Layer/block breakdown

For `depth_multiplier=1.0`, classification input after preprocessing is typically `[B, 3, 224, 224]`.

Stem:

```text
x = Conv3x3(3 -> 32, stride=2, groups=1, bias=false, tf_same_pad) -> BN -> ReLU6
if first_layer_is_expansion is false:
  x = Conv1x1(32 -> 32, bias=false) -> BN -> ReLU6
x = DepthwiseConv3x3(32 -> 32, stride=1, groups=32, bias=false, tf_same_pad) -> BN -> ReLU6
x = Conv1x1(32 -> 16, bias=false) -> BN
```

Inverted residual block, repeated 16 times with config-derived widths:

```text
residual = x
expanded = make_divisible(round(in_channels * expand_ratio), depth_divisible_by, min_depth)
x = Conv1x1(in_channels -> expanded, bias=false) -> BN -> ReLU6
x = DepthwiseConv3x3(expanded -> expanded, stride=1 or 2, groups=expanded,
                     dilation=1 or output_stride-derived, bias=false, tf_same_pad) -> BN -> ReLU6
x = Conv1x1(expanded -> out_channels, bias=false) -> BN
if stride == 1 and in_channels == out_channels:
  x = residual + x
```

Base encoder tail:

```text
x = Conv1x1(320 -> 1280, bias=false) -> BN -> ReLU6      # multiplier 1.0
pooled = AdaptiveAvgPool2d(x, output_size=(1,1))
pooled = flatten(pooled, start_dim=1)                     # [B, 1280]
```

Classifier:

```text
logits = Linear(final_channels -> num_labels)(Dropout(pooled))
```

Segmentation head for `google/deeplabv3_mobilenet_v2_1.0_513`, input feature `[B, 320, H/8, W/8]`:

```text
pool = AdaptiveAvgPool2d(1)(features)
pool = Conv1x1(320 -> 256) -> BN(eps=1e-5) -> ReLU
pool = bilinear_interpolate(pool, size=features.spatial, align_corners=True)
aspp = Conv1x1(320 -> 256) -> BN(eps=1e-5) -> ReLU
x = concat([pool, aspp], dim=1)                           # [B, 512, H/8, W/8]
x = Conv1x1(512 -> 256) -> BN(eps=1e-5) -> ReLU
x = Dropout2d(x)
logits = Conv1x1(256 -> num_labels, bias=true, no BN/activation)
```

## 6. Attention requirements

No attention is required. MobileNetV2 has no causal or noncausal attention, no attention masks, no packed/varlen metadata, no local/sliding-window pattern, no position bias, and no KV-cache generation path.

The only cache-like optimization worth considering is reusing preprocessed image tensors or encoder features across downstream heads. That is not a KV cache and should be represented as ordinary feature caching if needed.

## 7. Position encoding and custom math

There is no position embedding. Spatial structure is represented by convolution, stride, dilation, and padding.

DinoML must reproduce two small source math helpers:

```python
def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
```

```python
def tf_same_pad_2d(h, w, stride_h, stride_w, kernel_h, kernel_w, dilation_h, dilation_w):
    pad_h = max(kernel_h - stride_h, 0) if h % stride_h == 0 else max(kernel_h - (h % stride_h), 0)
    pad_w = max(kernel_w - stride_w, 0) if w % stride_w == 0 else max(kernel_w - (w % stride_w), 0)
    top = (pad_h // 2) * dilation_h
    bottom = (pad_h - pad_h // 2) * dilation_h
    left = (pad_w // 2) * dilation_w
    right = (pad_w - pad_w // 2) * dilation_w
    return left, right, top, bottom
```

Channel schedules can be precomputed from config. TF padding depends on dynamic input spatial dims and conv attributes.

## 8. Preprocessing and input packing

Image classification checkpoints sampled use:

```text
convert RGB -> resize shortest edge -> center crop -> rescale by 1/255
-> normalize mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
-> pixel_values [B, 3, crop_h, crop_w]
```

Processor configs:

| Checkpoint | Resize shortest edge | Crop | Resample | Mean/std |
|---|---:|---|---:|---|
| `google/mobilenet_v2_0.35_96` | 128 | 96x96 | 2 / bilinear | 0.5 / 0.5 |
| `google/mobilenet_v2_1.0_224` | 256 | 224x224 | 2 / bilinear | 0.5 / 0.5 |
| `google/mobilenet_v2_1.4_224` | 256 | 224x224 | 2 / bilinear | 0.5 / 0.5 |
| `google/deeplabv3_mobilenet_v2_1.0_513` | 545 | 513x513 | 2 / bilinear | 0.5 / 0.5 |

Segmentation maps are CPU/data-pipeline inputs, not model graph inputs for inference. When provided for training/eval preprocessing, they are nearest-resized/cropped, optionally label-reduced (`0 -> 255`, subtract 1, `254 -> 255`), squeezed to `[H, W]`, and converted to int64.

Postprocessing for end-to-end semantic segmentation parity:

```text
if target_sizes provided:
  per image: bilinear interpolate logits[None] to target_sizes[i], align_corners=False
  semantic_map = argmax(class_dim)
else:
  semantic_map = logits.argmax(dim=1)
```

No token packing, masks, metadata grids, or placeholder stitching exists.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d folding

Source pattern:

```text
Conv2d(bias=false or true) -> BatchNorm2d(affine=true, running_stats=true)
```

Replacement:

```text
Conv2d(weight_fused, bias_fused)
```

Preconditions:

- Inference mode using frozen running mean/variance and affine parameters.
- BatchNorm eps equals source value: usually config `layer_norm_eps=0.001`; segmentation head overrides `1e-5`.
- No training/dropout semantics.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fused = conv_w * scale.reshape(out_channels, 1, 1, 1)
b_fused = beta + (conv_b_or_zero - running_mean) * scale
```

Failure cases: training mode, unfrozen BN stats, missing BN buffers, wrong eps.

Parity sketch: random NCHW input through each conv+BN layer before/after folding in eval mode with fp32 tolerance around `1e-5`.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0 after tf_padding no-op, dilation=1, groups=1)
```

Replacement:

```text
NCHW/NHWC local layout view -> MatMul(Cin -> Cout) -> optional bias -> restore layout
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `dilation == 1`, `groups == 1`.
- For `tf_padding=true`, SAME padding for 1x1/stride1 is all zero.
- Consumer layout is controlled or layout conversion is explicitly inserted.

Weight transform:

```python
w_gemm = conv.weight.reshape(out_channels, in_channels)
```

Layout constraints:

- NCHW direct lowering needs channel gather/strided access or temporary NHWC.
- NHWC fusion is safe only within a guarded region where BN/ReLU6/residual/next depthwise consumer axes are also rewritten.

Failure cases: depthwise/grouped conv, stride > 1, dynamic consumer expecting NCHW.

Parity sketch: compare standalone 1x1 projection and a full inverted residual block after axis-aware layout rewrite.

### Rewrite: depthwise 3x3 Conv2d specialized kernel

Source pattern:

```text
Pad2d(tf_same) -> Conv2d(C -> C, kernel=3, groups=C, stride=1/2, dilation=1/2/4...)
```

Replacement:

```text
DepthwiseConv2dNHWC or NCHW specialized kernel -> folded BN -> ReLU6
```

Preconditions:

- `in_channels == out_channels == groups`.
- Padding exactly matches source `apply_tf_padding`, including dilation scaling.
- Stride/dilation schedule from `output_stride` is preserved.
- Layout pass rewrites channel axis and residual/concat consumers if using NHWC.

Failure cases: asymmetric dynamic padding not represented, channel count mismatch, training BN.

Parity sketch: sweep H/W divisible and non-divisible by stride; test `output_stride` 8, 16, and 32.

### Rewrite: global average pool + flatten + Linear -> classifier GEMM

Source pattern:

```text
AdaptiveAvgPool2d((1,1)) -> flatten(start_dim=1) -> Dropout(identity in eval) -> Linear(C -> labels)
```

Replacement:

```text
SpatialMean(H,W) over axes 2,3 -> MatMul(C -> labels) -> BiasAdd
```

Preconditions:

- Inference mode so Dropout is identity.
- Input remains NCHW or axes are correctly rewritten to NHWC spatial axes.

Failure cases: non-1x1 adaptive output, training dropout, wrong channel axis after layout pass.

Parity sketch: classifier logits parity on random final feature maps for NCHW and guarded NHWC variants.

### Rewrite: no-layout-translation guard around public NCHW interfaces

Source pattern:

```text
processor emits NCHW; model returns NCHW last_hidden_state/logits; segmentation postprocess expects class dim=1.
```

Replacement:

```text
no_layout_translation(model boundary, hidden_states tuple, segmentation logits/postprocess)
```

Preconditions:

- Public output parity is required.
- Downstream consumer has not opted into NHWC contract.

Failure cases: silently changing `cat(dim=1)`, `argmax(dim=1)`, BatchNorm channel axis, or `flatten(start_dim=1)` without full rewrite.

Parity sketch: compare output shapes and values for classification and segmentation with and without internal NHWC fused regions.

## 10. Kernel fusion candidates

Highest priority:

- Inverted residual fused block: `1x1 conv -> BN -> ReLU6 -> depthwise 3x3 -> BN -> ReLU6 -> 1x1 projection -> BN -> optional residual`. This is the dominant encoder pattern.
- Depthwise 3x3 with dynamic TF SAME padding and dilation. Segmentation `output_stride=8` depends on dilated depthwise convs.
- 1x1 conv as GEMM in channel-last memory. MobileNetV2 has many pointwise convs; NHWC can improve locality if contained.
- Conv+BN folding for all inference convs. This removes many small BN kernels.

Medium priority:

- ReLU6 clamp fusion into conv epilogues.
- Final `adaptive_avg_pool -> flatten -> linear` classifier fusion.
- Segmentation head pool branch: global pool + 1x1 + upsample + concat. Useful for segmentation but not first classifier target.
- Layout-pass elimination of NCHW/NHWC transposes inside fully contained conv stacks.

Lower priority:

- Dropout/Dropout2d identity elimination in inference.
- Preprocessor batching/grouping throughput. It is CPU/data-pipeline work unless DinoML owns image preprocessing.
- Training loss kernels and segmentation label preprocessing.

## 11. Runtime staging plan

Stage 1: parse `MobileNetV2Config`, implement `make_divisible`, build width/stride/dilation schedule, load weights.

Stage 2: implement base NCHW conv stack with TF SAME padding, BatchNorm inference, ReLU6, depthwise conv, residuals; validate one stem and one inverted residual.

Stage 3: run full `MobileNetV2Model` parity for `last_hidden_state`, `pooler_output`, and optional hidden-state shapes.

Stage 4: add image classification head parity for 0.35, 1.0, and 1.4 checkpoints.

Stage 5: fold BN and add depthwise/pointwise conv kernels; preserve public NCHW outputs.

Stage 6: add segmentation head and postprocess parity for `google/deeplabv3_mobilenet_v2_1.0_513`.

Stage 7: add guarded NHWC fused regions for inverted residual blocks, with `no_layout_translation()` boundaries around model inputs/outputs and postprocess.

Initial stubs: losses, training dropout, label reduction, and postprocess can be stubbed for classifier-only inference.

## 12. Parity and validation plan

- Unit test `make_divisible` against known schedules for multipliers 0.35, 1.0, 1.4.
- Unit test TF SAME padding for odd/even H/W, stride 1/2, dilation 1/2/4.
- Random tensor parity for `MobileNetV2ConvLayer` with BN eval and ReLU6.
- Single inverted residual parity for residual and non-residual cases.
- Full encoder parity at `output_stride=32` and `output_stride=8`.
- Classification logits parity for 0.35/1.0/1.4 checkpoint shapes.
- Segmentation logits parity and postprocess map parity with and without `target_sizes`.
- Layout rewrite parity: NCHW baseline versus guarded NHWC fused blocks, including axis-sensitive ops.

Recommended tolerances: fp32 `atol=1e-5, rtol=1e-5` before fusions; fp16/bf16 `atol=1e-2, rtol=1e-2` after epilogue fusion, with tighter per-layer diagnostics when drift appears.

## 13. Performance probes

- Processor throughput: resize/crop/normalize images per second for 96, 224, and 513 crops.
- Encoder-only throughput for multipliers 0.35, 1.0, 1.4.
- Batch-size sweep for 224x224 classification and 513x513 segmentation.
- `output_stride` sweep 32 vs 16 vs 8 to isolate dilation cost.
- Pointwise 1x1 GEMM throughput before/after NHWC local layout.
- Depthwise 3x3 kernel throughput with TF SAME padding fused vs separate pad.
- Conv+BN folded vs unfused graph.
- Classification head latency/throughput separately from encoder.
- Segmentation head throughput and bilinear resize/postprocess cost.
- Memory bandwidth probe for hidden-state tuple materialization when `output_hidden_states=True`.

## 14. Skip/defer list

- Training losses and gradient behavior.
- Gradient checkpointing; source marks `supports_gradient_checkpointing=False`.
- Label preprocessing for training segmentation maps.
- Quantization and packed weights; no source-coupled quantized format is implemented here.
- Multi-GPU/tensor parallelism.
- Generic Backbone API wrapper; this source only exposes base/hidden-state outputs.
- Segmentation head and postprocess can be deferred for classifier-first integration.

## 15. Final implementation checklist

- [ ] Parse `MobileNetV2Config` and validate `depth_multiplier > 0`.
- [ ] Implement `make_divisible` and width schedule construction.
- [ ] Implement output-stride stride/dilation schedule.
- [ ] Load Conv2d, BatchNorm2d, Linear, and classifier/segmentation weights.
- [ ] Implement dynamic TF SAME padding.
- [ ] Implement Conv2d NCHW including depthwise/grouped conv.
- [ ] Implement BatchNorm2d inference and Conv+BN folding.
- [ ] Implement ReLU6 and ReLU epilogues.
- [ ] Implement inverted residual residual-add guard.
- [ ] Implement adaptive average pool, flatten, and classifier Linear.
- [ ] Add 1x1 Conv2d-to-GEMM rewrite with layout guards.
- [ ] Add depthwise 3x3 specialized kernel/fusion.
- [ ] Add `no_layout_translation()` boundaries around public NCHW outputs and axis-sensitive postprocess.
- [ ] Add classifier parity tests for 0.35/1.0/1.4 checkpoints.
- [ ] Add segmentation head and postprocess parity tests for DeepLabV3 MobileNetV2.
- [ ] Benchmark pointwise, depthwise, full encoder, classifier, and segmentation probes.
