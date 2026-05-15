# UPerNet Transformers Audit

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: openmmlab/upernet-convnext-tiny as common reference; sweep includes OpenMMLab ConvNeXT and Swin variants.
Config source: official Hugging Face config.json, preprocessor_config.json, and repo_info.json snapshots from openmmlab/* repos.
Source files inspected:
- transformers/src/transformers/models/upernet/modeling_upernet.py
- transformers/src/transformers/models/upernet/configuration_upernet.py
- transformers/src/transformers/backbone_utils.py
- transformers/src/transformers/models/segformer/image_processing_segformer.py
- transformers/tests/models/upernet/test_modeling_upernet.py
Any missing files or assumptions: UPerNet has no family-local image processor; official checkpoints use SegformerImageProcessor. No gated/401/403 gaps were observed for sampled official OpenMMLab UPerNet repos. This report owns the UPerNet decode/auxiliary heads and composes previously audited ConvNeXT/Swin backbone coverage.
```

Pinned source URLs:

- [modeling_upernet.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/upernet/modeling_upernet.py)
- [configuration_upernet.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/upernet/configuration_upernet.py)
- [backbone_utils.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/backbone_utils.py)
- [image_processing_segformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/segformer/image_processing_segformer.py)

Local snapshots are under [agents/plans/transformers/upernet/_sources](H:/dinoml_v2/agents/plans/transformers/upernet/_sources).

## 2. High-Level Architecture

UPerNet is semantic segmentation with a nested vision backbone and a convolutional multi-scale decode head, not a generation model.

```text
Segformer-style image preprocessing -> pixel_values NCHW
  -> AutoBackbone feature maps [stage1..stage4], NCHW
  -> pyramid pooling on last feature
  -> FPN lateral/top-down fusion
  -> segmentation classifier logits
  -> bilinear resize to input tensor size
  -> optional post_process_semantic_segmentation resize + argmax
```

| Stage | Runtime contract | Independently testable? |
|---|---|---|
| CPU/data pipeline | Resize RGB image to 512x512 in sampled processors, rescale by `1/255`, normalize ImageNet mean/std, emit NCHW `pixel_values`. | Yes. |
| Backbone | `load_backbone(config)` calls `AutoBackbone.from_config(backbone_config)`. Official configs use ConvNeXT or Swin with `out_features=["stage1","stage2","stage3","stage4"]`. | Yes, through prior ConvNeXT/Swin audits. |
| Decode head | Four NCHW feature maps, PSP on stage4, FPN lateral/top-down adds, concat four 512-channel maps, classify to `num_labels`. | Yes. |
| Auxiliary head | Source computes it by default even with `labels=None`, but inference output drops it. | Yes; can be skipped for DinoML inference output parity. |
| Postprocess | Optional bilinear resize of logits to target size, then `argmax(dim=0)`. | Yes. |

Primary DinoML target: `UperNetForSemanticSegmentation` inference logits and semantic-map postprocess. Training loss and auxiliary loss are deferred.

## 3. Important Config Dimensions

Source defaults from `UperNetConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `backbone_config` | default ResNet, stages 1-4 | Official checkpoints override with ConvNeXT or Swin. |
| `hidden_size` | 512 | Decode-head working channels. |
| `pool_scales` | `(1,2,3,6)` | Adaptive-average pooling branches on last feature map. |
| `use_auxiliary_head` | `True` | Instantiates FCN auxiliary head; source forward computes it. |
| `auxiliary_in_channels` | `None` | If unset, uses backbone channel at feature index 2. |
| `auxiliary_channels` | 256 | Auxiliary conv width. |
| `auxiliary_num_convs` | 1 | Auxiliary Conv-BN-ReLU count. |
| `auxiliary_concat_input` | `False` | Optional auxiliary concat path. |
| `loss_ignore_index` | 255 | Training-only CrossEntropy ignore index. |

Representative checkpoint sweep:

| Model id | Repo SHA | Backbone | Backbone shape config | Head | Aux in | Labels | Processor |
|---|---|---|---|---:|---:|---:|---|
| [openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) | `876ffc56` | ConvNeXT | depths `3/3/9/3`, widths `96/192/384/768`, patch 4 | 512 | 384 | 150 | Segformer 512x512, resample 2, ImageNet mean/std. |
| [openmmlab/upernet-convnext-small](https://huggingface.co/openmmlab/upernet-convnext-small) | `550b68d2` | ConvNeXT | depths `3/3/27/3`, widths `96/192/384/768`, patch 4 | 512 | 384 | 150 | Same. |
| [openmmlab/upernet-convnext-base](https://huggingface.co/openmmlab/upernet-convnext-base) | `13a9c410` | ConvNeXT | depths `3/3/27/3`, widths `128/256/512/1024`, patch 4 | 512 | 512 | 150 | Same. |
| [openmmlab/upernet-convnext-large](https://huggingface.co/openmmlab/upernet-convnext-large) | `b029b4fb` | ConvNeXT | depths `3/3/27/3`, widths `192/384/768/1536`, patch 4 | 512 | 768 | 150 | Same. |
| [openmmlab/upernet-convnext-xlarge](https://huggingface.co/openmmlab/upernet-convnext-xlarge) | `c636a3c6` | ConvNeXT | depths `3/3/27/3`, widths `256/512/1024/2048`, patch 4 | 512 | 1024 | 150 | Same. |
| [openmmlab/upernet-swin-tiny](https://huggingface.co/openmmlab/upernet-swin-tiny) | `dc8e8c94` | Swin | depths `2/2/6/2`, embed 96, heads `3/6/12/24`, window 7 | 512 | 384 | 150 | Same. |
| [openmmlab/upernet-swin-small](https://huggingface.co/openmmlab/upernet-swin-small) | `bbf0728d` | Swin | depths `2/2/18/2`, embed 96, heads `3/6/12/24`, window 7 | 512 | 384 | 150 | Same. |
| [openmmlab/upernet-swin-base](https://huggingface.co/openmmlab/upernet-swin-base) | `63132e0c` | Swin | depths `2/2/18/2`, embed 128, heads `4/8/16/32`, window 12 | 512 | 512 | 150 | Same. |
| [openmmlab/upernet-swin-large](https://huggingface.co/openmmlab/upernet-swin-large) | `e7abf954` | Swin | depths `2/2/18/2`, embed 192, heads `6/12/24/48`, window 12 | 512 | 768 | 150 | Same. |

For 512x512 inputs and patch/stem stride 4, expected feature-map strides are approximately `4, 8, 16, 32`: `[B,C1,128,128]`, `[B,C2,64,64]`, `[B,C3,32,32]`, `[B,C4,16,16]`.

## 3a. Family Variation Traps

- UPerNet is composite. The head requires four image-like NCHW feature maps; it does not consume token sequences.
- Head channels come from `self.backbone.channels`, not UPerNet config alone.
- Official configs use `out_features=["stage1","stage2","stage3","stage4"]` in order. Missing, reordered, duplicate, or stem-only features should be rejected.
- Source layout is NCHW throughout the UPerNet head. NHWC is a local fusion optimization only. Axis-sensitive sites: `cat(dim=1)`, BatchNorm2d, Conv2d weight layout, `shape[2:]`, and interpolation `size=(H,W)`.
- A conceptual `no_layout_translation()` guard should cover the public backbone feature-map contract and final logits/postprocess contract.
- All UPerNet-head bilinear resizes use `align_corners=False`.
- Adaptive pooling scales `(1,2,3,6)` are fixed by config, but feature spatial sizes are dynamic; parity should include small maps where output scale can exceed input size.
- `use_auxiliary_head=True` costs inference compute in native source even though `auxiliary_logits` are not returned. DinoML inference can skip it when `labels is None`.
- `auxiliary_concat_input=True` and `auxiliary_num_convs=0` are source-supported even though sampled configs do not use them.
- Processor `do_reduce_labels=False` for sampled ADE20K repos; other datasets may vary.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- NCHW tensor contract for `pixel_values`, backbone feature maps, decode-head features, and logits.
- Shape reads from `pixel_values.shape[2:]` and feature-map `shape[2:]`.
- Channel-axis `cat`: PSP `C4 + 4*512`; FPN `4*512`.
- Elementwise add for top-down laterals.
- Optional per-image list output in postprocess.

Neural network primitives:

- Conv2d bias false -> BatchNorm2d -> ReLU in `UperNetConvModule`.
- Classifier Conv2d 1x1 bias true: `512 -> num_labels`.
- PSP: `AdaptiveAvgPool2d(scale)` -> Conv2d 1x1 `C4 -> 512` -> BN -> ReLU -> bilinear resize.
- Decode bottleneck Conv2d 3x3 `(C4 + 2048) -> 512`.
- Lateral Conv2d 1x1 `Ci -> 512` for stages 1-3.
- FPN Conv2d 3x3 `512 -> 512` for stages 1-3.
- FPN bottleneck Conv2d 3x3 `2048 -> 512`.
- Auxiliary Conv2d 3x3 `C3 -> 256`, BN, ReLU, then Conv2d 1x1 `256 -> num_labels`.

Attention primitives:

- None in UPerNet head.
- Swin variants require backbone-owned noncausal windowed self-attention, shifted-window masks, and relative position bias.
- ConvNeXT variants require no attention.

Position/relative-bias ops:

- None in UPerNet head; Swin relative position bias is owned by nested Swin coverage.

Generation/cache ops:

- Not applicable. No KV cache, prefill, decode, beam search, or sampling.

Preprocessing-coupled ops:

- SegformerImageProcessor resize/rescale/normalize/channels-first output.
- Semantic segmentation postprocess resize and class-axis argmax.

Distributed/tensor-parallel ops:

- Not required by source.

## 5. Layer/Block Breakdown

```text
pixel_values [B,3,H,W]
  -> AutoBackbone.forward_with_filtered_kwargs(...)
  -> feature_maps = [f1, f2, f3, f4]

For official 512x512 configs:
  ConvNeXT tiny/small: [96@128x128, 192@64x64, 384@32x32, 768@16x16]
  ConvNeXT base:       [128@128x128, 256@64x64, 512@32x32, 1024@16x16]
  ConvNeXT large:      [192@128x128, 384@64x64, 768@32x32, 1536@16x16]
  ConvNeXT xlarge:     [256@128x128, 512@64x64, 1024@32x32, 2048@16x16]
  Swin tiny/small:     [96@128x128, 192@64x64, 384@32x32, 768@16x16]
  Swin base:           [128@128x128, 256@64x64, 512@32x32, 1024@16x16]
  Swin large:          [192@128x128, 384@64x64, 768@32x32, 1536@16x16]
```

UPerNet ConvModule:

```text
y = Conv2d(in_channels -> out_channels, kernel, padding, dilation, bias=False)(x)
y = BatchNorm2d(out_channels)(y)
y = ReLU(y)
```

Decode head:

```text
for scale in [1,2,3,6]:
  p = AdaptiveAvgPool2d(scale)(f4)
  p = ConvModule(C4 -> 512, kernel=1)(p)
  p = BilinearResize(p, size=f4.shape[2:], align_corners=False)
psp = Cat([f4, p1, p2, p3, p6], dim=1)
psp = ConvModule(C4 + 2048 -> 512, kernel=3, padding=1)(psp)

l1,l2,l3 = ConvModule(Ci -> 512, kernel=1)(f1,f2,f3)
l4 = psp
for i = 4..2:
  l[i-1] = l[i-1] + BilinearResize(l[i], size=l[i-1].shape[2:], align_corners=False)
o1,o2,o3 = ConvModule(512 -> 512, kernel=3, padding=1)(l1,l2,l3)
o4 = l4
o2,o3,o4 = BilinearResize(..., size=o1.shape[2:], align_corners=False)
x = Cat([o1,o2,o3,o4], dim=1)
x = ConvModule(2048 -> 512, kernel=3, padding=1)(x)
logits = Conv2d(512 -> num_labels, kernel=1, bias=True)(x)
logits = BilinearResize(logits, size=pixel_values.shape[2:], align_corners=False)
```

Auxiliary head:

```text
a = feature_maps[2]
a = ConvModule(aux_in_channels -> 256, kernel=3, padding=1)(a)
if auxiliary_concat_input:
  a = ConvModule(aux_in_channels + 256 -> 256, kernel=3, padding=1)(Cat([feature_maps[2], a], dim=1))
aux_logits = Conv2d(256 -> num_labels, kernel=1, bias=True)(a)
aux_logits = BilinearResize(aux_logits, size=pixel_values.shape[2:], align_corners=False)
```

## 6. Attention Requirements

The UPerNet head has no attention, no causal mask, no KV cache, no SDPA/FlashAttention path, and no generation decode.

Swin backbones require noncausal windowed self-attention with shifted-window masks, per-stage MHA heads, and relative position bias. These are encoder-only features with no autoregressive cache. ConvNeXT UPerNet variants require no attention.

## 7. Position Encoding and Custom Math

UPerNet head-specific custom math is pyramid pooling plus top-down feature fusion:

```python
def upernet_ppm_top_feature(f4, pool_scales, conv1x1, bottleneck):
    pooled = []
    for scale, conv in zip(pool_scales, conv1x1):
        p = adaptive_avg_pool2d(f4, output_size=(scale, scale))
        p = conv_bn_relu(p)
        p = bilinear_resize(p, size=f4.shape[-2:], align_corners=False)
        pooled.append(p)
    return bottleneck(cat([f4, *pooled], axis=1))
```

```python
def upernet_fpn(laterals, fpn_convs, fpn_bottleneck):
    for i in range(len(laterals) - 1, 0, -1):
        laterals[i - 1] = laterals[i - 1] + bilinear_resize(
            laterals[i], size=laterals[i - 1].shape[-2:], align_corners=False
        )
    outs = [conv(laterals[i]) for i, conv in enumerate(fpn_convs)]
    outs.append(laterals[-1])
    outs = [outs[0]] + [bilinear_resize(o, size=outs[0].shape[-2:], align_corners=False) for o in outs[1:]]
    return fpn_bottleneck(cat(outs, axis=1))
```

Precomputable: fixed pool scales and static weight transforms. Dynamic: feature-map sizes, resize target sizes, postprocess target sizes.

## 8. Preprocessing and Input Packing

Sampled official repos use `SegformerImageProcessor`:

- RGB images; no tokenizer, multimodal packing, masks, boxes, or metadata.
- Resize `size={"height":512,"width":512}`, `resample=2`.
- Rescale by `0.00392156862745098`.
- Normalize with mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- `do_reduce_labels=false`.
- Emit NCHW float `pixel_values`, typically `[B,3,512,512]`.

Postprocess:

```text
if target_sizes is provided:
  resized_logits = bilinear_resize(logits[i][None], target_sizes[i], align_corners=False)
  semantic_map = resized_logits[0].argmax(dim=0)
else:
  semantic_map = logits.argmax(dim=1), split per batch element
```

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Conv-BatchNorm-ReLU Inference Fusion

Source pattern: `Conv2d(bias=False) -> BatchNorm2d -> ReLU`.

Replacement: `Conv2d(weight_fused, bias_fused) -> ReLU`.

Preconditions:

- Eval/inference mode.
- BatchNorm running stats and affine parameters are constants.
- No consumer reads intermediate pre-BN or pre-ReLU tensors.

Shape equations: `[B,Cin,H,W] -> [B,Cout,Hout,Wout]`; applies to PSP, lateral, FPN, bottleneck, and auxiliary ConvModules.

Weight transform:

```python
scale = bn.weight / sqrt(bn.running_var + bn.eps)
w_fused = conv.weight * scale[:, None, None, None]
b_fused = bn.bias - bn.running_mean * scale
```

Layout constraints: source is NCHW. NHWC kernels require explicit channel-axis rewrites and Conv2d weight transforms inside a controlled fused region.

Failure cases: training mode, mutable BN stats, or requested intermediate activations.

Parity test sketch: random NCHW tensors over multiple spatial sizes, compare unfused vs fused fp32/fp16/bf16 outputs.

### Rewrite: Classifier 1x1 Conv2d -> Per-Pixel Linear

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Consumer layout is restored to NCHW before postprocess or fully controlled.

Replacement:

```text
NHWC/view pixels -> MatMul(weight.T) -> BiasAdd -> restore NCHW logits
```

Shape equations: `[B,C,H,W] -> [B*H*W,C] -> [B*H*W,num_labels]`.

Weight transform:

```python
w = conv.weight.reshape(num_labels, C).T
```

Failure cases: grouped/non-unit convs or logits consumers requiring untouched NCHW memory layout.

### Rewrite: Fuse FPN Resize-Cat-Bottleneck Region

Source pattern: resize lower-resolution FPN outputs to `o1.shape[2:]`, concatenate on channel axis, then Conv-BN-ReLU 3x3.

Replacement: fused resize-concat-conv plan or a layout-aware subgraph.

Preconditions:

- Four feature levels in ascending resolution.
- All resize ops are bilinear `align_corners=False`.
- All resized tensors are `[B,512,H1,W1]`.
- No consumer reads individual resized outputs.

Layout constraints: candidate for NHWC/channel-last execution only inside the region. `cat(dim=1)` must become channel-last concat if translated.

Failure cases: nonstandard feature count, intermediate taps, or auxiliary consumers.

### Rewrite: Skip Auxiliary Head for Inference Output Parity

Preconditions:

- Inference-only target, `labels is None`.
- Returned contract is logits/hidden_states/attentions only.
- No profiling/debug hook depends on auxiliary module side effects.

Replacement: do not execute `auxiliary_head`.

Failure cases: training/loss parity, explicit auxiliary-head debugging, or non-eval BatchNorm side effects.

## 10. Kernel Fusion Candidates

Highest priority:

- Decode-head Conv-BN-ReLU fusion, because the UPerNet head is mostly ConvModule chains.
- Bilinear resize plus channel concat plus bottleneck planning, because FPN alignment creates heavy activation traffic.
- ConvNeXT/Swin backbone reuse, because UPerNet production value depends on composing optimized backbones with a thin head.

Medium priority:

- AdaptiveAvgPool2d fixed-scale PSP kernels for scales 1, 2, 3, 6.
- Guarded NHWC internal conv-heavy regions, while keeping public backbone/logits contracts NCHW.
- Auxiliary-head inference pruning.

Lower priority:

- 1x1 Conv2d to GEMM for classifier/lateral convs.
- Postprocess resize+argmax fusion.
- Training loss and auxiliary loss kernels.

## 11. Runtime Staging Plan

1. Parse UPerNet config and nested `backbone_config`; reject unsupported feature-map contracts.
2. Load weights and validate a stubbed backbone-output ABI using synthetic NCHW feature maps.
3. Implement UPerNet decode head parity: PSP, lateral convs, FPN top-down adds, resize-cat-bottleneck, classifier, final logits resize.
4. Compose ConvNeXT-backed UPerNet using existing ConvNeXT lowering.
5. Compose Swin-backed UPerNet using existing Swin lowering and window-attention coverage.
6. Add SegformerImageProcessor-compatible preprocessing and semantic segmentation postprocess.
7. Add inference pruning for auxiliary head when `labels=None`.
8. Add guarded Conv-BN-ReLU, FPN resize-cat, and local NHWC fusion passes.
9. Broaden to default ResNet or other AutoBackbone configs after separate backbone audits.

Initially stub/defer: labels/loss, auxiliary loss, output attentions for conv-only backbones, and non-OpenMMLab custom backbone repos.

## 12. Parity and Validation Plan

- Random tensor tests for `UperNetConvModule` and Conv-BN-ReLU fusion.
- PSP parity with feature maps both larger and smaller than pool scale 6.
- FPN head parity with four synthetic feature maps matching ConvNeXT/Swin channel/stride patterns.
- End-to-end head parity using captured backbone feature maps from ConvNeXT tiny and Swin tiny.
- Full logits parity for `openmmlab/upernet-convnext-tiny` and `openmmlab/upernet-swin-tiny`.
- Postprocess parity for `target_sizes=None`, fixed target size, and variable per-image target sizes.
- Auxiliary skip parity: returned inference logits unchanged when `labels=None`.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 start with `rtol=5e-3, atol=5e-3`.

No DinoML tests were run for this report.

## 13. Performance Probes

- Processor throughput for 512x512 resize/rescale/normalize.
- Backbone-only throughput for ConvNeXT tiny/base/large and Swin tiny/base/large.
- Decode-head-only throughput from synthetic feature maps.
- PSP microbenchmark by pool scale.
- FPN resize/add/concat/bottleneck bandwidth benchmark.
- Batch-size sweep for 1, 2, 4, 8, 16.
- Resolution sweep for 512, 640, 768, and odd caller-provided sizes.
- Auxiliary-head on/off inference benchmark.
- NCHW provider versus guarded NHWC fused-region comparison.
- End-to-end logits/sec and semantic-maps/sec, separating postprocess argmax.

## 14. Skip/Defer List

- Training loss and auxiliary loss.
- Gradient checkpointing and stochastic-depth training semantics inside backbones.
- Multi-GPU tensor parallelism.
- Quantization and packed-weight formats.
- Non-OpenMMLab fine-tuned custom backbones unless their `backbone_config` maps to an audited family.
- Timm backbone loading path unless a checkpoint explicitly requires it.
- Returning dense attention tensors from Swin backbones unless requested for diagnostics.

## 15. Final Implementation Checklist

- [ ] Parse `UperNetConfig` and nested `backbone_config`.
- [ ] Validate `out_features=["stage1","stage2","stage3","stage4"]` or equivalent four-level image-map ABI.
- [ ] Load UPerNet head weights and preserve Conv2d/BatchNorm parameters.
- [ ] Compose ConvNeXT backbone lowering for ConvNeXT UPerNet repos.
- [ ] Compose Swin backbone lowering for Swin UPerNet repos.
- [ ] Implement PSP adaptive pooling branches with bilinear `align_corners=False`.
- [ ] Implement FPN lateral convs, top-down resize/add, final resize/cat/bottleneck.
- [ ] Implement classifier logits resize to `pixel_values.shape[2:]`.
- [ ] Implement SegformerImageProcessor preprocessing contract.
- [ ] Implement semantic segmentation postprocess resize and argmax.
- [ ] Add inference graph prune for auxiliary head when `labels=None`.
- [ ] Add Conv-BN-ReLU fusion with eval-mode guards.
- [ ] Add guarded NHWC/local layout fusion candidates and no-layout guards at backbone/logits boundaries.
- [ ] Add synthetic head parity tests.
- [ ] Add ConvNeXT tiny and Swin tiny end-to-end logits parity tests.
- [ ] Benchmark backbone-only, head-only, auxiliary on/off, and postprocess throughput.
