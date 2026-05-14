# ResNet Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/resnet-50 as the common reference; sweep includes microsoft/resnet-18, -34, -50, -101, and -152.
Config source: official Hugging Face config.json and preprocessor_config.json files fetched from microsoft/* repos.
Source files inspected:
- X:/H/transformers/src/transformers/models/resnet/modeling_resnet.py
- X:/H/transformers/src/transformers/models/resnet/configuration_resnet.py
- X:/H/transformers/src/transformers/models/resnet/convert_resnet_to_pytorch.py
- X:/H/transformers/src/transformers/models/convnext/image_processing_convnext.py
- X:/H/transformers/src/transformers/models/convnext/image_processing_pil_convnext.py
- X:/H/transformers/src/transformers/models/auto/image_processing_auto.py
- X:/H/transformers/src/transformers/backbone_utils.py
Any missing files or assumptions: no gated/401/403 gaps were observed for the sampled official Microsoft repos. ResNet has no family-local image processor; AutoImageProcessor maps `model_type="resnet"` to the shared ConvNext image processor classes. This report targets native in-library ResNet, not DETR/MaskFormer heads that may compose ResNet as a backbone.
```

Pinned source URLs:

- [modeling_resnet.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/resnet/modeling_resnet.py)
- [configuration_resnet.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/resnet/configuration_resnet.py)
- [convert_resnet_to_pytorch.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/resnet/convert_resnet_to_pytorch.py)
- [image_processing_convnext.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/convnext/image_processing_convnext.py)
- [image_processing_auto.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/image_processing_auto.py)

Local snapshots are under [agents/plans/transformers/resnet/_sources](H:/dinoml_v2/agents/plans/transformers/resnet/_sources).

## 2. High-level architecture

ResNet is a convolutional vision encoder, not a Transformer/attention model. The primary DinoML target for this report is image classification inference with `ResNetForImageClassification`. `ResNetModel` feature extraction and `ResNetBackbone` NCHW feature-map output are optional but important because other Transformers families can consume ResNet as a backbone.

```text
CPU/image pipeline -> pixel_values NCHW -> 7x7 stem Conv/BN/ReLU -> 3x3 max pool
-> 4 residual stages -> adaptive global average pool -> flatten -> classifier Linear -> logits
```

Stage decomposition:

- CPU/data pipeline: ConvNext image processor resize/crop/rescale/normalize, then emits `pixel_values` in channels-first layout.
- Stem: `Conv2d(Cin=num_channels, Cout=embedding_size, kernel=7, stride=2, padding=3, bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(kernel=3, stride=2, padding=1)`.
- Encoder: four residual stages. Stage 1 optionally downsamples if `downsample_in_first_stage=True`; stages 2-4 downsample in the first block with stride 2. Blocks are `basic` or `bottleneck` depending on config.
- Classification head: `AdaptiveAvgPool2d((1,1)) -> Flatten -> Linear(hidden_sizes[-1] -> num_labels)`.
- Backbone output: selected hidden states named `stem`, `stage1`, `stage2`, `stage3`, `stage4`; returned tensors are NCHW feature maps.

## 3. Important config dimensions

Source defaults from `ResNetConfig`:

| Field | Default | Source/runtime effect |
|---|---:|---|
| `model_type` | `resnet` | Auto model dispatch. |
| `num_channels` | 3 | Checked against `pixel_values.shape[1]`. |
| `embedding_size` | 64 | Stem output channels before max pool. |
| `hidden_sizes` | `[256, 512, 1024, 2048]` | Stage output channels. |
| `depths` | `[3, 4, 6, 3]` | Residual block count per stage. |
| `layer_type` | `bottleneck` | Chooses `ResNetBottleNeckLayer`; valid values are `basic`, `bottleneck`. |
| `hidden_act` | `relu` | Activation after Conv/BN except final conv inside a block, then after residual add. |
| `downsample_in_first_stage` | `false` | If true, first stage first block uses stride 2. |
| `downsample_in_bottleneck` | `false` | If true, bottleneck stride moves from 3x3 conv to first 1x1 conv. |
| `_out_features` / `_out_indices` | last stage by default | Backbone feature selection via `BackboneConfigMixin`. |
| `torch_dtype` | checkpoint-specific | Sampled configs say `float32`; source is dtype-polymorphic. |
| cache support | none | No KV cache, recurrent state, or generation path. |

Representative checkpoint sweep from official `config.json` and `preprocessor_config.json`:

| Model id | Block type | Depths | Stage channels | Stem channels | Labels | Processor size | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) | `basic` | 2,2,2,2 | 64,128,256,512 | 64 | 1000 | 224 | Small/debug-friendly; no bottleneck reduction. |
| [microsoft/resnet-34](https://huggingface.co/microsoft/resnet-34) | `basic` | 3,4,6,3 | 64,128,256,512 | 64 | 1000 | 224 | Same stage widths as ResNet-18, deeper. |
| [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) | `bottleneck` | 3,4,6,3 | 256,512,1024,2048 | 64 | 1000 | 224 | Common reference; bottleneck reduced width is output/4. |
| [microsoft/resnet-101](https://huggingface.co/microsoft/resnet-101) | `bottleneck` | 3,4,23,3 | 256,512,1024,2048 | 64 | 1000 | 224 | Deeper stage 3. |
| [microsoft/resnet-152](https://huggingface.co/microsoft/resnet-152) | `bottleneck` | 3,8,36,3 | 256,512,1024,2048 | 64 | 1000 | 224 | Deepest sampled official variant. |

All sampled preprocessors use `feature_extractor_type="ConvNextFeatureExtractor"`, `do_resize=true`, `crop_pct=0.875`, `resample=3`, ImageNet mean/std, and `size=224`. The source config supplies omitted defaults such as `downsample_in_bottleneck=False`.

For 224x224 inputs with default stride placement and `downsample_in_first_stage=False`, spatial sizes are:

| Boundary | Stride from input | ResNet-18/34 channels | ResNet-50/101/152 channels | Spatial size for 224 |
|---|---:|---:|---:|---:|
| stem after max pool | 4 | 64 | 64 | 56x56 |
| stage1 | 4 | 64 | 256 | 56x56 |
| stage2 | 8 | 128 | 512 | 28x28 |
| stage3 | 16 | 256 | 1024 | 14x14 |
| stage4 | 32 | 512 | 2048 | 7x7 |

## 3a. Family variation traps

- `basic` and `bottleneck` checkpoints have different operator structure and channel equations. Do not infer ResNet-18/34 shapes from ResNet-50 defaults.
- In bottleneck blocks, `reduces_channels = out_channels // 4`. For ResNet-50 stage1, the first block is `1x1 64->64`, `3x3 64->64`, `1x1 64->256` plus shortcut `1x1 64->256`.
- `downsample_in_bottleneck` changes where stride 2 occurs inside bottleneck blocks. The sampled Microsoft configs omit it, so effective source default is `False`: stride is on the 3x3 conv, not the first 1x1 conv.
- `downsample_in_first_stage` changes output strides and spatial sizes. Sampled official configs use `False`; a custom config with `True` needs separate shape guards.
- Source model boundaries and backbone outputs are NCHW. NHWC/channel-last should be a guarded layout/fusion optimization only. Axis-sensitive ops include channel check `shape[1]`, BatchNorm channel axis, concatenation none, pooling over H/W, `Flatten` after `(N,C,1,1)`, and feature-map consumers expecting NCHW.
- `ResNetConvLayer` always performs Conv2d -> BatchNorm2d -> activation. The final conv in a residual branch uses `activation=None`, then the block applies activation after residual add.
- The source type annotation allows tuple `kernel_size`, but `padding=kernel_size // 2` only works for integer kernel sizes. Native ResNet paths use integer kernels 1, 3, and 7; a tuple-kernel custom config should be rejected or routed through a source-compatibility check.
- Backbone hidden states include the pre-stage embedding output as `stem`; `ResNetEncoder` records hidden state before each stage plus the final state.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor contract for `pixel_values` `[B, 3, H, W]`.
- Channel-dimension shape check against `num_channels`.
- NCHW `Flatten` after global pool: `[B, C, 1, 1] -> [B, C]`.
- Tuple/list hidden-state assembly for optional backbone/model outputs.
- Guarded NCHW <-> NHWC only inside layout/fusion passes, not direct semantic translation.

Neural network primitives:

- `Conv2d` static weights, bias usually false:
  - Stem: `Conv2d(3 -> 64, kernel=7, stride=2, padding=3, bias=False)`.
  - Basic block: `Conv2d(Cin -> Cout, kernel=3, stride=s, padding=1, bias=False)`, then `Conv2d(Cout -> Cout, kernel=3, stride=1, padding=1, bias=False)`.
  - Bottleneck block: `Conv2d(Cin -> Cout/4, kernel=1, stride=s_or_1, padding=0, bias=False)`, `Conv2d(Cout/4 -> Cout/4, kernel=3, stride=1_or_s, padding=1, bias=False)`, `Conv2d(Cout/4 -> Cout, kernel=1, stride=1, padding=0, bias=False)`.
  - Shortcut projection: `Conv2d(Cin -> Cout, kernel=1, stride=s, padding=0, bias=False)` when `Cin != Cout` or stride != 1.
- `BatchNorm2d(C)` in inference mode using affine weight/bias and running mean/var.
- ReLU activation.
- Residual add, then ReLU.
- `MaxPool2d(kernel=3, stride=2, padding=1)`.
- `AdaptiveAvgPool2d((1,1))`.
- `Linear(C_last -> num_labels)` classifier; sampled ImageNet heads are `512 -> 1000` for basic and `2048 -> 1000` for bottleneck.

Attention primitives:

- None required.

Position/rotary/relative-bias ops:

- None required.

Generation/cache ops:

- None required. Branch embeddings or backbone feature maps can be cached by a calling application, but ResNet has no internal KV/recurrent cache ABI.

Preprocessing-coupled ops:

- Resize with ConvNext processor semantics: if requested `shortest_edge < 384`, resize shortest edge to `int(shortest_edge / crop_pct)` while preserving aspect ratio, then center crop to `shortest_edge`; otherwise direct square resize.
- Rescale by `1/255` when enabled by processor defaults.
- Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- Emit channels-first `pixel_values`.

Distributed/tensor-parallel ops:

- None in source.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: [B, 3, H, W]
x = Conv2d(3 -> 64, k=7, stride=2, pad=3, bias=False)(pixel_values)
x = BatchNorm2d(64)(x)
x = ReLU(x)
x = MaxPool2d(k=3, stride=2, pad=1)(x)
```

Basic residual block, repeated according to `depths` in ResNet-18/34:

```text
input: [B, Cin, H, W]
residual = input
x = Conv2d(Cin -> Cout, k=3, stride=s, pad=1, bias=False)(input)
x = BatchNorm2d(Cout)(x)
x = ReLU(x)
x = Conv2d(Cout -> Cout, k=3, stride=1, pad=1, bias=False)(x)
x = BatchNorm2d(Cout)(x)
if Cin != Cout or s != 1:
    residual = Conv2d(Cin -> Cout, k=1, stride=s, bias=False)(residual)
    residual = BatchNorm2d(Cout)(residual)
x = ReLU(x + residual)
```

Bottleneck residual block, repeated according to `depths` in ResNet-50/101/152:

```text
input: [B, Cin, H, W]
Cr = Cout // 4
residual = input
if downsample_in_bottleneck:
    x = Conv2d(Cin -> Cr, k=1, stride=s, bias=False)(input)
else:
    x = Conv2d(Cin -> Cr, k=1, stride=1, bias=False)(input)
x = BatchNorm2d(Cr)(x)
x = ReLU(x)
if downsample_in_bottleneck:
    x = Conv2d(Cr -> Cr, k=3, stride=1, pad=1, bias=False)(x)
else:
    x = Conv2d(Cr -> Cr, k=3, stride=s, pad=1, bias=False)(x)
x = BatchNorm2d(Cr)(x)
x = ReLU(x)
x = Conv2d(Cr -> Cout, k=1, stride=1, bias=False)(x)
x = BatchNorm2d(Cout)(x)
if Cin != Cout or s != 1:
    residual = Conv2d(Cin -> Cout, k=1, stride=s, bias=False)(residual)
    residual = BatchNorm2d(Cout)(residual)
x = ReLU(x + residual)
```

Stages:

```text
stage1: first block stride = 2 if downsample_in_first_stage else 1; remaining blocks stride=1
stage2..stage4: first block stride = 2; remaining blocks stride=1
```

Classification:

```text
last_hidden_state: [B, C_last, H_last, W_last]
pooled = AdaptiveAvgPool2d((1, 1))(last_hidden_state)  # [B, C_last, 1, 1]
logits = Linear(C_last -> num_labels)(Flatten(pooled))
```

Backbone:

```text
hidden_states = (stem_output, stage1_output, stage2_output, stage3_output, stage4_output)
feature_maps = tuple(hidden_states[idx] for selected stage names)
```

## 6. Attention requirements

No attention is required for the primary target. There is no causal/noncausal self-attention, cross-attention, MHA/MQA/GQA, attention mask, RoPE/ALiBi/relative bias, packed/varlen attention metadata, sliding window, FlashAttention/SDPA backend, or KV cache. Any report or lowering path that tries to model ResNet as prefill/decode should be rejected for this family.

## 7. Position encoding and custom math

No position encoding is required. Spatial structure is represented only by convolution, pooling, padding, and stride.

The only custom math DinoML should reproduce carefully is inference BatchNorm folding/fusion and ConvNext-style preprocessing resize:

```python
def fold_conv_bn(w, b, gamma, beta, running_mean, running_var, eps):
    if b is None:
        b = zeros_like(running_mean)
    scale = gamma / sqrt(running_var + eps)
    w_folded = w * scale.reshape(-1, 1, 1, 1)
    b_folded = (b - running_mean) * scale + beta
    return w_folded, b_folded
```

```python
def convnext_resnet_resize_shape(h, w, shortest_edge, crop_pct):
    if shortest_edge < 384:
        resize_shortest = int(shortest_edge / crop_pct)
        # Preserve aspect ratio so min(new_h, new_w) == resize_shortest,
        # then center-crop to shortest_edge x shortest_edge.
        return (shortest_edge, shortest_edge)
    return (shortest_edge, shortest_edge)
```

BatchNorm constants can be pre-folded into Conv weights for inference after weights are loaded. Resize/crop output shape depends on processor config and input image aspect ratio; final sampled tensor shape is `[B, 3, 224, 224]`.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- AutoImageProcessor maps ResNet to the shared ConvNext image processor.
- Sampled `preprocessor_config.json` uses `size=224`, `crop_pct=0.875`, bicubic/PIL resample id `3`, ImageNet mean/std, resize, rescale, and normalize enabled.
- For `size < 384`, ConvNext processor resizes so shortest edge becomes `int(size / crop_pct)` and then center-crops to `size x size`. For size 224, this is `int(224 / 0.875) = 256`, then crop to 224.
- Output is `pixel_values` in channels-first `[B, 3, 224, 224]` for sampled configs.

GPU/runtime work:

- The model consumes only `pixel_values`; no masks, positions, token ids, placeholder tokens, or packed metadata.
- Backbone output contracts are NCHW feature maps. Feature names are `stem`, `stage1`, `stage2`, `stage3`, `stage4`. Default `out_features` is the last stage if not overridden.

Postprocessing:

- Image classification returns logits only. Label mapping is metadata in `id2label`/`label2id`; softmax/top-k is pipeline/application behavior, not in the model forward.
- Backbone returns feature maps only. Detection/segmentation postprocessing belongs to the downstream model family report, not this ResNet report.

## 9. Graph rewrite / lowering opportunities

### Rewrite: inference Conv2d + BatchNorm2d -> Conv2d with folded weights

Source pattern:

```text
Conv2d(bias=False or bias=True) -> BatchNorm2d(Cout)
```

Replacement:

```text
Conv2d(weight_folded, bias_folded)
```

Preconditions:

- Inference mode only; BatchNorm uses stored running mean/variance.
- BatchNorm affine parameters exist and have channel count equal to Conv2d output channels.
- No consumer reads intermediate pre-BN activation.
- Preserve source dtype behavior or define an explicit fp32 fold-then-cast policy.

Shape equations:

- Conv weight `[Cout, Cin/groups, Kh, Kw]`.
- BN vectors `[Cout]`.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w2 = w * scale[:, None, None, None]
b2 = (conv_bias_or_zero - running_mean) * scale + beta
```

Layout constraints:

- Works for NCHW and NHWC runtime kernels if the weight transform is applied in OIHW storage and runtime layout is tracked separately.
- A channel-last fusion pass must rewrite BN channel axis from NCHW `C=1` to NHWC `C=-1`.

Failure cases:

- Training mode, live batch statistics, missing BN running stats, or custom downstream use of pre-BN values.

Parity test sketch:

- Compare original Conv/BN/ReLU block to folded Conv/ReLU over random NCHW tensors for representative 7x7, 3x3, and 1x1 shapes.

### Rewrite: Conv2d + BatchNorm2d + ReLU -> fused ConvBNReLU kernel

Source pattern:

```text
ResNetConvLayer(..., activation="relu")
```

Replacement:

```text
FusedConv2dBiasRelu(weight_folded, bias_folded)
```

Preconditions:

- Same Conv+BN folding preconditions.
- Activation is exactly ReLU.
- The block-final convs with `activation=None` must not be fused with ReLU until after residual add.

Shape equations:

- Stem: `[B,3,H,W] -> [B,64,ceil(H/2),ceil(W/2)]`.
- Stage convs follow normal Conv2d output formula with padding `kernel//2`.

Weight transform:

- Same as Conv+BN folding.

Layout constraints:

- Best NHWC candidate region is a complete local chain where both branch tensors are in channel-last and residual add/ReLU consumers are controlled.
- Protect public model input/output and backbone outputs with a conceptual `no_layout_translation()` guard unless the downstream consumer also accepts NHWC.

Failure cases:

- Non-ReLU `hidden_act`, final residual ReLU placement, custom configs with nonstandard activations, or requested hidden-state outputs at internal unfused points.

Parity test sketch:

- Compare a full basic block and bottleneck block before/after fusion with stride 1 and stride 2 projection shortcuts.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1 or 2, padding=0, bias=False)
```

Replacement:

```text
Optional strided spatial sample -> MatrixMultiply([B*H'*W', Cin] x [Cin, Cout]) -> reshape
```

Preconditions:

- `kernel_size == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- For stride 2, sampling semantics must match Conv2d floor/offset behavior.
- Bias is absent in source, but folded BN introduces bias.

Shape equations:

- Output spatial: `H' = floor((H - 1) / stride + 1)`, `W' = floor((W - 1) / stride + 1)`.
- Matmul rows: `B * H' * W'`.

Weight transform:

```python
w_gemm = conv.weight.reshape(Cout, Cin).T
```

Layout constraints:

- NHWC makes row flatten natural as `[B,H,W,C]`. NCHW lowering requires an im2col/transpose or a native NCHW 1x1 kernel.

Failure cases:

- Groups, dilation, padding, non-1x1 kernels, or layout consumers not controlled.

Parity test sketch:

- Compare bottleneck 1x1 reduce/expand and shortcut projection paths for stride 1 and stride 2.

### Rewrite: global AdaptiveAvgPool2d((1,1)) + Flatten -> spatial mean

Source pattern:

```text
AdaptiveAvgPool2d((1,1)) -> Flatten
```

Replacement:

```text
ReduceMean over H,W -> [B,C]
```

Preconditions:

- Output size is exactly `(1,1)`.
- Tensor rank is NCHW or an explicitly tracked NHWC equivalent.
- No consumer requires the intermediate `[B,C,1,1]`.

Shape equations:

- NCHW: `[B,C,H,W] -> [B,C]`.
- NHWC: `[B,H,W,C] -> [B,C]`.

Weight transform:

- None.

Layout constraints:

- Axis rewrite required: NCHW reduce axes `(2,3)` become NHWC axes `(1,2)`.

Failure cases:

- Caller requests `ResNetModel.pooler_output` specifically as `[B,C,1,1]`; classification can use flattened mean directly, but base model parity may need the 4D pooled output.

Parity test sketch:

- Compare pooled output and classifier logits for non-square H/W as well as 224x224.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BatchNorm2d folding for all ResNet convs. This removes every BatchNorm runtime op in inference and is required for competitive throughput.
- ConvBNReLU fused kernels for stem and non-final residual branch convs. ResNet is dominated by these static convolution chains.
- Residual add + ReLU fusion for basic and bottleneck block exits. This is the critical elementwise path after the final BN and shortcut projection.

Medium priority:

- NHWC/channel-last fused residual blocks under guard. This can improve convolution memory access, but source/public contracts remain NCHW, so use a fusion/layout pass with axis rewrites rather than direct model translation.
- 1x1 Conv2d GEMM specialization for bottleneck reduce/expand and shortcut paths. ResNet-50+ has many 1x1 convs, especially in deep stage 3.
- Global average pool + flatten + classifier fusion for classification heads, optionally producing logits directly from the last feature map.

Lower priority:

- General 3x3/7x7 Conv2d im2col-to-GEMM lowering. Useful as a fallback but likely inferior to tuned convolution kernels.
- Backbone output materialization scheduling. Only needed when downstream model requests multiple feature maps.
- Processor resize/rescale/normalize GPU fusion. CPU preprocessing is acceptable initially; GPU preprocessing matters for high-throughput batched image services.

## 11. Runtime staging plan

1. Parse `ResNetConfig`, including `layer_type`, `depths`, `hidden_sizes`, `downsample_in_first_stage`, `downsample_in_bottleneck`, and backbone `out_features/out_indices`.
2. Load weights for `ResNetModel` and `ResNetForImageClassification`; preserve BatchNorm running stats and affine params.
3. Implement unfused NCHW stem, basic block, bottleneck block, shortcut, max pool, global pool, flatten, and classifier parity.
4. Add `ResNetBackbone` feature-map output parity for selected stages.
5. Add inference Conv+BN folding and ConvBNReLU/residual-add fusions.
6. Add guarded NHWC/channel-last block regions with explicit no-layout-translation guards around public inputs, hidden-state/backbone outputs, and downstream NCHW consumers.
7. Add processor parity or document CPU preprocessing handoff for ImageNet-style inference.

Initially stub training losses, labels/loss computation, non-ReLU activations, and uncommon custom stride variants if not required by target checkpoints.

## 12. Parity and validation plan

- Config parsing tests for sampled ResNet-18/34/50/101/152 configs, checking stage names, channel widths, depths, and omitted default fields.
- Random tensor tests for Conv+BN folding across stem 7x7, basic 3x3, bottleneck 1x1, and shortcut projection shapes.
- Single-block parity:
  - Basic block stride 1 identity shortcut.
  - Basic block stride 2 projection shortcut.
  - Bottleneck block stride 1 identity/projection cases.
  - Bottleneck block stride 2 with `downsample_in_bottleneck` false and true.
- Stage parity after each stage for ResNet-18 and ResNet-50 at `[B,3,224,224]`.
- `ResNetModel` parity for `last_hidden_state`, 4D `pooler_output`, and optional `hidden_states`.
- `ResNetForImageClassification` logits parity for official checkpoints.
- `ResNetBackbone` parity for `out_features=["stage1","stage2","stage3","stage4"]` and negative `out_indices`.
- End-to-end processor plus logits parity using the official preprocessor config.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=5e-2, atol=5e-2` for full-model logits after convolution/BN fusion, with tighter per-op thresholds where accumulation is fp32.

## 13. Performance probes

- CPU preprocessing throughput for resize/crop/rescale/normalize at batch sizes 1, 8, 32, 128.
- Stem throughput separately: 7x7 stride-2 ConvBNReLU plus max pool.
- Per-stage throughput for stage1-stage4, separated by block type and depth.
- Bottleneck 1x1 versus 3x3 convolution time breakdown for ResNet-50/101/152.
- Batch-size sweep for classification logits throughput.
- Resolution sweep for arbitrary caller-provided image sizes, especially non-224 and non-square tensors.
- NCHW baseline versus guarded NHWC block fusion comparison.
- Conv+BN folded versus unfused backend comparison.
- Backbone multi-output materialization cost when returning 1, 2, or 4 feature maps.
- Memory bandwidth and activation footprint probe for deep stage3 in ResNet-101/152.

## 14. Skip/defer list

- Training losses and label handling.
- BatchNorm training-mode statistics.
- Gradient checkpointing and autograd.
- Multi-GPU/tensor parallelism.
- Quantization and packed-weight formats; no source-coupled packed format is present.
- DETR/MaskFormer detection or segmentation heads that use ResNet as a backbone; audit those in their owning families.
- Non-ReLU custom activations until a real checkpoint requires them.
- Direct model-wide NHWC translation. Prefer guarded fusion/layout passes with explicit axis rewrites.

## 15. Final implementation checklist

- [ ] Parse `ResNetConfig` and backbone output settings.
- [ ] Load Conv2d, BatchNorm2d, and Linear weights plus BN running stats.
- [ ] Implement NCHW stem Conv/BN/ReLU + MaxPool.
- [ ] Implement `ResNetBasicLayer`.
- [ ] Implement `ResNetBottleNeckLayer` with both stride-placement modes.
- [ ] Implement shortcut projection Conv/BN.
- [ ] Implement adaptive global average pool, flatten, classifier Linear.
- [ ] Implement `ResNetBackbone` feature-map output selection.
- [ ] Add Conv+BN folding rewrite.
- [ ] Add ConvBNReLU and residual-add-ReLU fusion.
- [ ] Add guarded NHWC block-region optimization with axis rewrite checks.
- [ ] Add processor contract fixture for ConvNext-style ImageNet preprocessing.
- [ ] Add block, stage, model, classifier, and backbone parity tests.
- [ ] Benchmark preprocessing, per-stage throughput, classification throughput, and NCHW versus NHWC fusion.
