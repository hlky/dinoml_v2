# DINOv3 ConvNeXt DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/dinov3-convnext-tiny-pretrain-lvd1689m and size variants from the official conversion script
Config source: configuration_dinov3_convnext.py defaults plus convert_dinov3_convnext_to_hf.py variant table
Source files inspected:
- transformers/src/transformers/models/dinov3_convnext/configuration_dinov3_convnext.py
- transformers/src/transformers/models/dinov3_convnext/modeling_dinov3_convnext.py
- transformers/src/transformers/models/dinov3_convnext/convert_dinov3_convnext_to_hf.py
- transformers/src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py
Any missing files or assumptions:
- Official raw config/preprocessor files for tiny/small/base/large returned 401, so checkpoint JSON snapshots are unavailable until HF access is granted.
- No remote code is required by the in-library source.
- Primary runtime target here is base image feature extraction/backbone output, not supervised classification.
```

Snapshots were written under `agents/plans/transformers/dinov3_convnext/_sources/`.

Gated/access gaps:

- [facebook/dinov3-convnext-tiny-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m): `config.json`, `preprocessor_config.json`, and `image_processor_config.json` returned 401.
- [facebook/dinov3-convnext-small-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m): same raw-file 401 gap.
- [facebook/dinov3-convnext-base-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m): same raw-file 401 gap.
- [facebook/dinov3-convnext-large-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m): same raw-file 401 gap.

## 2. High-level architecture

DINOv3 ConvNeXt is a convolutional image encoder with ConvNeXt blocks and a DINO-style final token output. It has no transformer attention, no text branch, no generation cache, and no classification head in this family source.

```text
image preprocessing -> NCHW pixel_values -> 4-stage ConvNeXt encoder
  -> final NCHW feature map
  -> adaptive average pooled "CLS" token + flattened patch tokens
  -> final LayerNorm over tokens -> pooler_output / last_hidden_state
```

Backbone path:

```text
image preprocessing -> NCHW pixel_values -> 4-stage ConvNeXt encoder
  -> selected NCHW feature maps from captured stage outputs
```

Stageable pieces:

- CPU/data pipeline: image rescale, resize with bilinear antialiased interpolation, ImageNet normalization.
- GPU/runtime: all convolution, layer norm, GELU, linear, scale, residual, pooling, flatten/transpose/concat.
- Independently optimizable: stage blocks, downsample layers, final tokenization, and backbone feature-map extraction.

## 3. Important config dimensions

Source defaults:

| Field | Default/effective value | Source |
|---|---:|---|
| `model_type` | `dinov3_convnext` | config |
| `num_channels` | 3 | config |
| `hidden_sizes` | `[96, 192, 384, 768]` | config |
| `depths` | `[3, 3, 9, 3]` | config |
| total ConvNeXt blocks | 18 | derived from `depths` |
| `hidden_act` | `gelu` | config |
| `layer_norm_eps` | `1e-6` | config |
| `layer_scale_init_value` | `1e-6` | config |
| `drop_path_rate` | `0.0` default | config |
| `image_size` | 224 | config |
| stem | `Conv2d(3 -> C0, kernel=4, stride=4)` | modeling |
| downsample stages | `LayerNorm(Cin, channels_first) -> Conv2d(Cin -> Cout, kernel=2, stride=2)` | modeling |
| block depthwise conv | `Conv2d(C -> C, kernel=7, padding=3, groups=C)` | modeling |
| block channel MLP | `Linear(C -> 4C) -> GELU -> Linear(4C -> C)` | modeling |
| GRN | absent | modeling |
| attention / cache | absent | modeling |

Representative checkpoint sweep from `convert_dinov3_convnext_to_hf.py`:

| Variant | HF model id | `hidden_sizes` | `depths` | Blocks | Final C | 224x224 final map | Output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| tiny | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | `[96, 192, 384, 768]` | `[3, 3, 9, 3]` | 18 | 768 | 7x7 | 50 |
| small | `facebook/dinov3-convnext-small-pretrain-lvd1689m` | `[96, 192, 384, 768]` | `[3, 3, 27, 3]` | 36 | 768 | 7x7 | 50 |
| base | `facebook/dinov3-convnext-base-pretrain-lvd1689m` | `[128, 256, 512, 1024]` | `[3, 3, 27, 3]` | 36 | 1024 | 7x7 | 50 |
| large | `facebook/dinov3-convnext-large-pretrain-lvd1689m` | `[192, 384, 768, 1536]` | `[3, 3, 27, 3]` | 36 | 1536 | 7x7 | 50 |

For an input size `(H, W)` divisible by 32, encoder stage feature maps are:

```text
stage0: [B, C0, H/4,  W/4]
stage1: [B, C1, H/8,  W/8]
stage2: [B, C2, H/16, W/16]
stage3: [B, C3, H/32, W/32]
model last_hidden_state: [B, 1 + (H/32 * W/32), C3]
pooler_output: [B, C3]
```

For non-divisible image sizes, PyTorch convolutions use floor output-size rules. DinoML should not assume token count is exactly `1 + ceil(H/32) * ceil(W/32)`.

## 3a. Family variation traps

- DINOv3 ConvNeXt is not DINOv3 ViT: no patch embeddings, positional embeddings, register tokens, attention, or token sequence until after the final ConvNeXt feature map.
- It is also not plain `convnext`/`convnextv2`: this source has no `ForImageClassification` head, no ConvNeXtV2 GRN, and its model output is a DINO token sequence made from global average plus flattened final feature map.
- The official converter maps original keys `dwconv -> depthwise_conv`, `pwconv -> pointwise_conv`, `norm -> layer_norm`, `stages.i.j -> stages.i.layers.j`, and `downsample_layers.i.j -> stages.i.downsample_layers.j`.
- Raw official checkpoint configs were gated, so fields like `drop_path_rate`, `out_features`, and processor JSON values should be reconfirmed after access. The conversion script only establishes architecture sizes and default processor behavior.
- Modeling is NCHW at public boundaries and between stages. Inside each block, the source explicitly permutes to NHWC for LayerNorm and `Linear(C -> 4C -> C)`, then permutes back to NCHW for the residual add.
- `DINOv3ConvNextLayerNorm(data_format="channels_first")` is an axis-sensitive wrapper that permutes NCHW to NHWC, applies channel LayerNorm, and permutes back. A layout pass must rewrite normalized axis and consumer expectations rather than blindly changing tensor layout.
- Backbone feature maps are returned as captured NCHW stage outputs without the extra per-stage output normalizers used by classic ConvNext/ConvNextV2 backbones.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW image tensors at model/backbone entry and backbone output.
- `permute(0, 2, 3, 1)` and `permute(0, 3, 1, 2)` inside every block.
- `flatten(2)`, `transpose(1, 2)`, `cat(dim=1)` for final token output.
- Dynamic shape arithmetic for convolution output sizes and token count.
- Backbone hidden-state capture/filter by stage name.

### Neural network primitives

- `Conv2d(3 -> C0, kernel=4, stride=4, bias=True)`.
- `Conv2d(Ci -> C{i+1}, kernel=2, stride=2, bias=True)` for downsample stages 1-3.
- Depthwise `Conv2d(C -> C, kernel=7, padding=3, groups=C, bias=True)`.
- `LayerNorm(C, eps=1e-6, affine=True)` in both channels-first wrapper and channels-last form.
- `Linear(C -> 4C, bias=True)`, `GELU`, `Linear(4C -> C, bias=True)`.
- Learned layer scale `gamma: [C]`, applied as elementwise multiply on NHWC block output.
- Residual add.
- `AdaptiveAvgPool2d(output_size=1)`.

Concrete channel MLP sizes:

| Variant | Stage channels | Per-block MLP shapes |
|---|---|---|
| tiny/small | 96, 192, 384, 768 | `96->384->96`, `192->768->192`, `384->1536->384`, `768->3072->768` |
| base | 128, 256, 512, 1024 | `128->512->128`, `256->1024->256`, `512->2048->512`, `1024->4096->1024` |
| large | 192, 384, 768, 1536 | `192->768->192`, `384->1536->384`, `768->3072->768`, `1536->6144->1536` |

### Attention primitives

- None required.

### Position/rotary/relative-bias ops

- None required. There is no positional embedding in this source; spatial structure comes from convolutions.

### Generation/cache ops

- None required. This is an image encoder/backbone with no autoregressive path.

### Preprocessing-coupled ops

- Rescale by `1/255` when enabled.
- Resize to 224x224 by default with bilinear interpolation and antialiasing.
- Normalize with ImageNet mean/std.
- Processor emits `pixel_values` in NCHW.

### Distributed/tensor-parallel ops

- None source-required.

## 5. Layer/block breakdown

Stem stage, once:

```text
pixel_values: [B, 3, H, W]
x = Conv2d(3 -> C0, kernel=4, stride=4, bias=True)(pixel_values)
x = LayerNorm(C0, eps=1e-6, channels_first wrapper)(x)
for block in stage0 depth:
  x = DINOv3ConvNextLayer(C0)(x)
```

Downsample stage `i > 0`, repeated for stages 1-3:

```text
x: [B, C{i-1}, Hi, Wi]
x = LayerNorm(C{i-1}, eps=1e-6, channels_first wrapper)(x)
x = Conv2d(C{i-1} -> Ci, kernel=2, stride=2, bias=True)(x)
for block in stage_i depth:
  x = DINOv3ConvNextLayer(Ci)(x)
```

ConvNeXt block, repeated according to `depths`:

```text
residual = x                                      # [B, C, Hs, Ws], NCHW
y = DepthwiseConv2d(C -> C, kernel=7, padding=3, groups=C, bias=True)(x)
y = permute(y, NCHW -> NHWC)                     # [B, Hs, Ws, C]
y = LayerNorm(C, eps=1e-6, affine=True)(y)
y = Linear(C -> 4C, bias=True)(y)
y = GELU(y)
y = Linear(4C -> C, bias=True)(y)
y = y * gamma[C]
y = permute(y, NHWC -> NCHW)
x = residual + DropPath_or_identity(y)
```

Model head-like output path:

```text
x = encoder(pixel_values).last_hidden_state       # [B, C3, H/32, W/32]
cls = AdaptiveAvgPool2d(1)(x)                     # [B, C3, 1, 1]
cls = flatten(2).transpose(1, 2)                  # [B, 1, C3]
patch = flatten(2).transpose(1, 2)                # [B, Hf*Wf, C3]
tokens = cat([cls, patch], dim=1)                 # [B, 1+Hf*Wf, C3]
tokens = LayerNorm(C3, eps=1e-6)(tokens)
pooler_output = tokens[:, 0]                      # [B, C3]
```

Backbone output path:

```text
captured hidden_states from stages are already NCHW
feature_maps = tuple(hidden_states[stage] for stage in out_features)
```

## 6. Attention requirements

No attention is required for the primary target. There is no causal or noncausal MHA, no cross-attention, no masks, no packed/varlen attention metadata, no ALiBi/RoPE/relative bias, and no KV cache.

The only sequence-like tensor is produced after convolution by flattening final spatial features plus a pooled token. It should not be treated as a transformer token stream requiring attention kernels.

## 7. Position encoding and custom math

No explicit position encoding is implemented. Spatial information is encoded through convolutions and downsampling.

Custom math that DinoML should reproduce:

```python
def channel_first_layer_norm(x, weight, bias, eps):
    # x: [B, C, H, W]
    y = x.permute(0, 2, 3, 1)
    y = layer_norm_last_dim(y, weight, bias, eps)
    return y.permute(0, 3, 1, 2)
```

```python
def dinov3_convnext_tokens(x, ln_weight, ln_bias, eps):
    # x: [B, C, Hf, Wf]
    cls = adaptive_avg_pool2d(x, (1, 1)).flatten(2).transpose(1, 2)
    patch = x.flatten(2).transpose(1, 2)
    tokens = concat([cls, patch], axis=1)
    tokens = layer_norm_last_dim(tokens, ln_weight, ln_bias, eps)
    return tokens, tokens[:, 0]
```

The final token count depends on runtime image size after convolution output-size rules. The layer scale `gamma` is static per channel and can be folded only with care into the second pointwise linear weight/bias if layout and DropPath identity assumptions hold for inference.

## 8. Preprocessing and input packing

The source image processor is shared with DINOv3 ViT. It emits `pixel_values` for image inputs:

```text
input images -> grouped by shape
if do_rescale: rescale
if do_resize: resize(size={height: 224, width: 224}, bilinear, antialias=True)
if do_center_crop: center crop, but default is disabled
if do_normalize: normalize with ImageNet mean/std
return pixel_values: [B, 3, H, W]
```

Important ordering: DINOv3 overrides the generic processor order to perform `rescale -> resize -> normalize`. The conversion script checks parity against torchvision `ToTensor() -> Resize((224,224), antialias=True) -> Normalize(mean/std)`.

CPU/data-pipeline work:

- Image decode/convert to RGB.
- Rescale, resize, optional crop, normalize, grouping/reordering.

GPU/runtime work:

- Model starts at `pixel_values`; no masks, token type IDs, grids, or packed sequence metadata are consumed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: stem patch Conv2d to GEMM

Source pattern:

```text
Conv2d(3 -> C0, kernel=4, stride=4, padding=0, groups=1)
```

Replacement:

```text
WindowFlatten([4,4], stride=4, NCHW/NHWC-aware) -> MatMul(weight_flat.T) -> BiasAdd -> Reshape
```

Preconditions:

- `kernel_size == stride == 4`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W divisible by 4, or fallback exactly matches PyTorch floor-window semantics.
- Weight flatten order preserves PyTorch OIHW layout.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * 4 * 4)
```

Layout constraints:

- NCHW source flatten order must be preserved.
- NHWC direct translation is safe only for this fully-contained window extraction if all consumers are controlled.

Failure cases:

- Dynamic sizes where floor-window drop behavior is not supported.
- Any non-default padding/dilation/groups.

Parity sketch:

- Compare stem output for random `[B,3,H,W]` where H/W are divisible and non-divisible by 4.

### Rewrite: stride-2 downsample Conv2d to GEMM

Source pattern:

```text
LayerNorm(channels_first) -> Conv2d(Cin -> Cout, kernel=2, stride=2, padding=0)
```

Replacement:

```text
ChannelLayerNorm -> WindowFlatten([2,2], stride=2) -> MatMul(weight_flat.T) -> BiasAdd -> Reshape
```

Preconditions:

- `kernel_size == stride == 2`, `padding == 0`, `dilation == 1`, `groups == 1`.
- H/W after prior stage are at least 2 and use PyTorch floor semantics.
- Axis rewrite for LayerNorm is explicit if a layout pass keeps NHWC.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * 2 * 2)
```

Failure cases:

- Attempting to fuse through unknown external backbone consumers.
- Wrong axis rewrite for `channels_first` norm.

### Rewrite: ConvNeXt block NHWC island

Source pattern:

```text
DepthwiseConv2d(NCHW) -> permute NCHW->NHWC -> LayerNorm -> Linear -> GELU -> Linear -> gamma
  -> permute NHWC->NCHW -> residual add
```

Replacement:

```text
DepthwiseConv2d_NHWC_or_NCHW -> ChannelLayerNorm -> PointwiseGEMM -> GELU -> PointwiseGEMM
  -> ChannelScale -> ResidualAdd
```

Preconditions:

- The whole block is owned by the layout pass.
- Residual branch and depthwise conv agree on memory layout.
- LayerNorm normalized axis is channel (`C`) after layout rewrite.
- Final output layout matches the next stage or a controlled consumer.

Layout constraints:

- Good candidate for NHWC fusion because pointwise `Linear` already operates on NHWC in source.
- Protect public model/backbone boundaries and feature-map outputs with a conceptual `no_layout_translation()` guard unless downstream consumers are also lowered.

Failure cases:

- Returning `hidden_states`/`feature_maps` to public API in unexpected layout.
- Fusing across final token flatten without rewriting `flatten(2)`/`transpose(1,2)` semantics.

### Rewrite: final adaptive pool + flatten tokenization

Source pattern:

```text
AdaptiveAvgPool2d(1) + flatten/transpose + concat pooled token and patch tokens + LayerNorm
```

Replacement:

```text
SpatialMean(H,W) -> prepend pooled row -> FlattenSpatialToTokens -> LayerNorm
```

Preconditions:

- Output is final model token output, not backbone feature maps.
- Flatten order matches PyTorch NCHW `flatten(2).transpose(1,2)`: token order is row-major over `Hf, Wf`.
- `dim=1` concat is preserved as token-axis concat.

Failure cases:

- Layout pass changes flatten order without explicit correction.

## 10. Kernel fusion candidates

### Highest priority

- Depthwise 7x7 Conv2d + NHWC LayerNorm + pointwise MLP island: repeated in every block and dominates runtime, especially small/base/large stage2 with 9 or 27 blocks.
- Channels-first LayerNorm wrapper: avoid physical NCHW<->NHWC permutes by fusing channel-axis normalization into adjacent convolution/layout regions.
- Pointwise `Linear(C -> 4C) -> GELU -> Linear(4C -> C)` on NHWC: map to two GEMMs with activation fusion or to 1x1 convolution kernels in a channel-last block implementation.

### Medium priority

- Stem/downsample non-overlap Conv2d-to-GEMM lowering: useful for unifying convolutional patch extraction with GEMM provider paths.
- Final tokenization fusion: adaptive average pool + flatten + concat + LayerNorm is small but common for feature extraction parity.
- Backbone output materialization guard: avoid unnecessary layout conversion when only final `pooler_output` is requested, but preserve NCHW for requested feature maps.

### Lower priority

- DropPath elimination for inference: source already uses identity when `training=False`; just constant-fold.
- Layer-scale folding into the second pointwise linear: possible for inference, but keep separate until parity is stable.

## 11. Runtime staging plan

1. Parse `DINOv3ConvNextConfig` and conversion-script size presets; reject unknown/gated configs only when required fields are absent.
2. Load weights with converter key aliases if consuming original DINOv3 checkpoint names; load native HF names directly when available.
3. Implement one stage with NCHW semantics and channels-first LayerNorm parity.
4. Implement one ConvNeXt block parity for each channel width used by tiny/small/base/large.
5. Implement full encoder parity for `DINOv3ConvNextModel`.
6. Implement final token output: pooled token, patch-token flatten order, final LayerNorm, `pooler_output`.
7. Implement `DINOv3ConvNextBackbone` selected feature maps with NCHW output contracts.
8. Add guarded NHWC fusion/layout passes for internal blocks.
9. Add stem/downsample GEMM rewrites and performance probes.

Initially stub/defer:

- Training DropPath stochastic behavior.
- Original checkpoint conversion if native HF weights are used.
- Public `hidden_states` capture beyond the backbone selections.

## 12. Parity and validation plan

- Random tensor tests for `channel_first_layer_norm` against PyTorch-style permute+LayerNorm+permute.
- Random tensor tests for final tokenization, including non-square final maps.
- Single-block parity at C = 96, 128, 192, 384, 768, 1024, 1536.
- Per-stage parity for downsample output shapes and values.
- Full encoder parity at image sizes 224x224 and a non-divisible size that exercises floor convolution semantics.
- Backbone parity for each `out_features` combination; verify returned maps are NCHW and unnormalized by extra output norms.
- End-to-end image processor plus model output parity after official raw configs are accessible.

Recommended tolerances:

- fp32: `atol=1e-5`, `rtol=1e-5` for blocks; `1e-4` full model.
- fp16/bf16: start with `atol=2e-2`, `rtol=2e-2`, then tighten per backend.

No DinoML tests were run for this report.

## 13. Performance probes

- Image preprocessing throughput: decode/rescale/resize/normalize to NCHW.
- Encoder-only throughput by variant: tiny/small/base/large.
- Stage-level timing, especially stage2 depth 9 vs 27.
- Depthwise 7x7 kernel throughput in NCHW vs guarded NHWC layout.
- Pointwise MLP GEMM throughput for each channel width.
- Layout conversion overhead with and without NHWC block fusion.
- Backbone feature-map extraction overhead for different `out_features`.
- Final tokenization overhead for pooled+patch token output.
- Batch-size sweep at 224x224 and larger feature-extraction resolutions.
- Memory bandwidth probes for repeated permute/materialization in unfused source-equivalent lowering.

## 14. Skip/defer list

- Training, gradients, stochastic DropPath behavior.
- Supervised image classification head; not implemented in this family source.
- Attention/FlashAttention/KV cache; not applicable.
- Quantization-specific loading/provider paths; no source-coupled quantized format inspected.
- Multi-GPU tensor parallel.
- Original Meta checkpoint conversion can be deferred if loading native HF snapshots.
- Gated official config/preprocessor reconfirmation until HF access is available.

## 15. Final implementation checklist

- [ ] Parse `DINOv3ConvNextConfig`.
- [ ] Add tiny/small/base/large architecture presets from official conversion script.
- [ ] Load native HF weights and/or converter-renamed original weights.
- [ ] Implement NCHW Conv2d stem/downsample/depthwise ops.
- [ ] Implement channels-first LayerNorm wrapper.
- [ ] Implement ConvNeXt block with NHWC channel MLP island.
- [ ] Implement layer-scale multiply and inference DropPath identity.
- [ ] Implement pooled-token + patch-token flatten + final LayerNorm output.
- [ ] Implement backbone `out_features` NCHW feature-map contract.
- [ ] Add guarded NHWC block fusion with public-output layout guards.
- [ ] Add stem/downsample Conv2d-to-GEMM rewrites with shape guards.
- [ ] Add parity tests for block, stage, encoder, model token output, and backbone maps.
- [ ] Benchmark NCHW baseline vs NHWC fused block lowering.
