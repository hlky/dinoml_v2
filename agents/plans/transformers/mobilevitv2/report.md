# MobileViTV2 DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: apple/mobilevitv2-1.0-imagenet1k-256; apple/mobilevitv2-2.0-imagenet1k-256; apple/mobilevitv2-1.0-voc-deeplabv3; apple/mobilevitv2-1.5-voc-deeplabv3
Config source: official HF config.json and preprocessor_config.json snapshots under _sources/
Source files inspected:
- transformers/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py
- transformers/src/transformers/models/mobilevitv2/configuration_mobilevitv2.py
- transformers/src/transformers/models/mobilevitv2/convert_mlcvnets_to_pytorch.py
- transformers/src/transformers/models/mobilevit/image_processing_mobilevit.py
- transformers/docs/source/en/model_doc/mobilevitv2.md
Source URLs:
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilevitv2/configuration_mobilevitv2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mobilevit/image_processing_mobilevit.py
HF config URLs:
- https://huggingface.co/apple/mobilevitv2-1.0-imagenet1k-256
- https://huggingface.co/apple/mobilevitv2-2.0-imagenet1k-256
- https://huggingface.co/apple/mobilevitv2-1.0-voc-deeplabv3
- https://huggingface.co/apple/mobilevitv2-1.5-voc-deeplabv3
Any missing files or assumptions: no gated files; no native MobileViTV2 image processor exists, AutoImageProcessor maps this family to MobileViTImageProcessor. No Transformers imports, model execution, or DinoML tests were run.
```

Primary DinoML target: image classification backbone/head first, then semantic segmentation with DeepLabV3/ASPP. Assumptions: inference-only, CUDA GPU target, NHWC preferred as a guarded optimization/fusion layout rather than default semantic translation.

## 2. High-level architecture

MobileViTV2 is a CNN-style vision encoder with MobileNetV2 inverted residual blocks plus MobileViTV2 global blocks. It is not an autoregressive generation model and has no tokenizer, logits sampling, or KV cache.

```text
CPU image preprocessing -> NCHW pixel_values -> conv stem -> MobileNet blocks ->
MobileViTV2 unfold/linear-attention blocks -> final feature map ->
global average pool + classifier OR DeepLabV3 ASPP segmentation logits
```

Stage decomposition:

- CPU/data pipeline: resize shortest edge, center crop, rescale, optional channel flip, channel-first tensor. Segmentation labels use nearest interpolation and no rescale/channel flip.
- Encoder: NCHW conv stem and five encoder stages; hidden states are image-like feature maps.
- Classification head: mean over spatial axes `[-2, -1]`, then `Linear(C -> num_labels)`.
- Segmentation head: consumes encoder hidden states, uses last hidden state only, ASPP pyramid, dropout, `Conv2d(aspp_out_channels -> num_labels)`.
- Postprocessing: semantic maps are `interpolate(logits, target_size, bilinear, align_corners=False)` then `argmax(dim=0)` per image.

## 3. Important config dimensions

Source defaults:

| Field | Default / source behavior |
|---|---|
| `model_type` | `mobilevitv2` |
| `num_channels` | 3 |
| `image_size` | 256 |
| `patch_size` | 2, used as both unfold kernel and stride |
| `conv_kernel_size` | 3 for local depthwise conv in MobileViTV2 blocks |
| `hidden_act` | `swish` |
| `expand_ratio` | 2.0 for inverted residual expansion |
| `output_stride` | 32; segmentation checkpoints use 16 |
| `n_attn_blocks` | `[2, 4, 3]` in stages 3/4/5 |
| `base_attn_unit_dims` | `[128, 192, 256]`, scaled by `width_multiplier` and `make_divisible(..., 8)` |
| `ffn_multiplier` | 2, rounded down to multiple of 16 |
| `layer_norm_eps` | 1e-5, used by `GroupNorm(num_groups=1)` |
| `attn_dropout`, `ffn_dropout` | 0.0 in official configs |
| `aspp_out_channels` | 512 |
| `atrous_rates` | `[6, 12, 18]` in official HF configs inspected |
| `dtype` | official configs say `torch_dtype=float32` |
| Cache support | none |

Representative checkpoint sweep:

| Checkpoint | Task | Image/crop | Labels | Width multiplier | Output stride | Stage channels L0/L1/L2/L3/L4/L5 | Attention dims | Segmentation fields |
|---|---:|---:|---:|---:|---:|---|---|---|
| `apple/mobilevitv2-1.0-imagenet1k-256` | classification | 256 | 1000 | 1.0 | 32 | 32/64/128/256/384/512 | 128/192/256 | unused by head |
| `apple/mobilevitv2-2.0-imagenet1k-256` | classification | 256 | 1000 | 2.0 | 32 | 64/128/256/512/768/1024 | 256/384/512 | unused by head |
| `apple/mobilevitv2-1.0-voc-deeplabv3` | segmentation | 512 | 21 | 1.0 | 16 | 32/64/128/256/384/512 | 128/192/256 | ASPP 512, rates 6/12/18 |
| `apple/mobilevitv2-1.5-voc-deeplabv3` | segmentation | 512 | 21 | 1.5 | 16 | 48/96/192/384/576/768 | 192/288/384 | ASPP 512, rates 6/12/18 |

Inference: width multipliers beyond these official native checkpoints are structurally supported by source, but should be admitted by config-driven channel computation rather than hardcoded tables.

## 3a. Family variation traps

- MobileViTV2 uses separable/linear self-attention, not standard MHA. There are no heads, QK matmul, attention masks, RoPE, relative bias, or KV cache.
- Source tensor contract is NCHW throughout public model boundaries. `GroupNorm`, `BatchNorm2d`, `Conv2d`, `torch.split(..., dim=1)`, `torch.cat(..., dim=1)`, and pooling/mean axes are channel/spatial-axis sensitive.
- Unlike MobileViT v1, MobileViTV2 source does not interpolate before unfold. `nn.functional.unfold` uses `kernel_size=stride=patch_size` and no padding; non-divisible feature-map edges are not a safe optimization target. Official preprocessing produces 256/512 crops divisible by all stage/patch sizes.
- `output_stride=8/16` changes layer 4/5 stride-vs-dilation behavior for segmentation. Classification checkpoints use stride 32.
- Configs include historical `mlp_ratio: 2.0`, but inspected current source uses `ffn_multiplier`, not `mlp_ratio`; treat `mlp_ratio` as ignored for this source basis.
- Official configs list architecture strings like `MobileViTv2ForImageClassification` with lower-case `v`; native class names are `MobileViTV2...`. Loaders rely on `model_type`, not exact report-time class spelling.
- Segmentation uses only `hidden_states[-1]` despite requesting all hidden states. Other feature levels are not consumed by this DeepLabV3 head.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW tensors, shape queries, `reshape`, `unfold`, `fold`, `split(dim=1)`, `cat(dim=1)`, `sum(dim=-1, keepdim=True)`, `mean(dim=[-2,-1])`, broadcast/expand.
- Guarded NHWC candidates require axis rewrites: channel split/cat `dim=1 -> dim=-1`, GroupNorm/BatchNorm channel axis, spatial mean `[-2,-1] -> [1,2]` if fully NHWC, segmentation `argmax(dim=1 -> -1)` if logits are channel-last.

Neural network primitives:

- `Conv2d` 3x3 stem `3 -> L0`, stride 2, BN, swish.
- Inverted residual: 1x1 expand `C -> make_divisible(C*2,8)`, depthwise 3x3 stride 1/2, 1x1 reduce, BN after each conv, swish except final reduce.
- MobileViTV2 local: depthwise 3x3 `C -> C`, then 1x1 `C -> D` without BN/activation.
- 1x1 qkv projection `D -> 1 + 2D`, bias true, no BN/activation.
- 1x1 attention output `D -> D`, bias true.
- FFN 1x1 `D -> floor(D*ffn_multiplier/16)*16`, swish, dropout, 1x1 back to `D`, dropout.
- `GroupNorm(num_groups=1, num_channels=D, eps=1e-5)` before attention, before FFN, and after stage transformer.
- Classification `Linear(L5 -> num_labels)`.
- Segmentation ASPP: parallel 1x1, three dilated 3x3, adaptive avg pool + 1x1 + bilinear upsample, channel concat of 5 branches, 1x1 project, dropout, 1x1 classifier.

Attention primitives:

- MobileViTV2 linear self-attention over unfolded patch tensors `[B, D, patch_area, Npatch]`.
- Softmax over `Npatch`, elementwise multiply with keys, sum over `Npatch`, ReLU on value, broadcast multiply, 1x1 output conv.

Position/rotary/relative-bias ops:

- None. Spatial location is carried by convolutional structure and unfold/fold ordering.

Generation/cache ops:

- Not applicable. No causal decode, no beam reorder, no KV cache.

Preprocessing-coupled ops:

- Resize shortest edge, center crop, rescale by `1/255`, channel flip if `do_flip_channel_order=true`, channel-first output. Segmentation maps use nearest interpolation and int64 labels.
- Semantic postprocess bilinear resize with `align_corners=False`, then argmax.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: [B, 3, H, W]
x = Conv2d(3 -> L0, k=3, stride=2, padding=1, bias=false)
x = BatchNorm2d(L0, eps=1e-5)
x = swish(x)
```

Inverted residual block:

```text
residual = x
x = Conv2d(C -> E, k=1, bias=false) + BN + swish
x = DepthwiseConv2d(E -> E, k=3, stride=s, padding=dilation, dilation=d, groups=E, bias=false) + BN + swish
x = Conv2d(E -> C_out, k=1, bias=false) + BN
out = residual + x only when stride == 1 and C == C_out
```

MobileViTV2 block, repeated for stages 3/4/5:

```text
optional downsample inverted residual if stride == 2
x = DepthwiseConv2d(C -> C, k=conv_kernel_size, groups=C) + BN + swish
x = Conv2d(C -> D, k=1, bias=false, no BN, no act)
patches = unfold(x, kernel=patch_size, stride=patch_size) -> [B, D, patch_area, Npatch]
repeat n_attn_blocks:
  y = GroupNorm(num_groups=1, channels=D)(patches)
  qkv = Conv2d(D -> 1+2D, k=1, bias=true)(y)
  query, key, value = split(qkv, [1, D, D], dim=1)
  scores = softmax(query, dim=-1)
  context = sum(key * scores, dim=-1, keepdim=True)
  y = Conv2d(D -> D, k=1, bias=true)(relu(value) * expand(context))
  patches = patches + y
  y = GroupNorm(num_groups=1, channels=D)(patches)
  y = Conv2d(D -> F, k=1, bias=true)(y) + swish + dropout
  y = Conv2d(F -> D, k=1, bias=true)(y) + dropout
  patches = patches + y
patches = GroupNorm(num_groups=1, channels=D)(patches)
x = fold(patches, output_size=(H_stage, W_stage), kernel=patch_size, stride=patch_size)
x = Conv2d(D -> C_current, k=1, bias=false) + BN
```

Classification head:

```text
last_hidden_state: [B, L5, H/output_stride, W/output_stride]
pooled = mean(last_hidden_state, dim=[-2, -1]) -> [B, L5]
logits = Linear(L5 -> num_labels, bias=true)
```

Segmentation head:

```text
features = hidden_states[-1]
branches = [1x1, 3x3 dilation r1, 3x3 dilation r2, 3x3 dilation r3, global_pool_1x1_upsample]
x = cat(branches, dim=1) -> [B, 5*aspp_out_channels, Hs, Ws]
x = Conv2d(5*aspp_out_channels -> aspp_out_channels, k=1) + BN + relu
logits = Conv2d(aspp_out_channels -> num_labels, k=1, bias=true)
```

## 6. Attention requirements

Required attention variant: noncausal encoder-only separable linear self-attention over patch columns. It is not MHA/MQA/GQA and should not be lowered to FlashAttention.

Details:

- Input to attention is `[B, D, P, N]`, where `P = patch_height * patch_width` and `N = floor(H/P_h) * floor(W/P_w)` from unfold.
- Q/K/V are packed by channel after a 1x1 conv: split order `[query_1, key_D, value_D]`.
- Softmax axis is `N` (`dim=-1`) for a single-channel query map.
- Context vector shape is `[B, D, P, 1]`, broadcast across all patches.
- No masks, no dropout in official configs, no position bias, no cache.
- Eager fallback risk is not attention matmul; the heavy path is `unfold`/`fold` materialization plus many small 1x1/depthwise convs and GroupNorms.

## 7. Position encoding and custom math

No explicit position encoding exists. Custom math DinoML should reproduce is the separable attention and the width rounding.

```python
def mobilevitv2_linear_attention(x, qkv_w, qkv_b, out_w, out_b):
    # x: [B, D, patch_area, num_patches], NCHW-style channel axis
    qkv = conv1x1(x, qkv_w, qkv_b)          # [B, 1 + 2 * D, P, N]
    q, k, v = split(qkv, [1, D, D], axis=1)
    scores = softmax(q, axis=-1)
    context = sum(k * scores, axis=-1, keepdims=True)
    return conv1x1(relu(v) * context, out_w, out_b)
```

```python
def make_divisible(value, divisor=8, min_value=None):
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    return int(new_value + divisor if new_value < 0.9 * value else new_value)
```

The channel schedule can be precomputed from config. Unfold patch count depends on dynamic input feature-map shape.

## 8. Preprocessing and input packing

Official preprocessor snapshots:

- ImageNet checkpoints: resize shortest edge to 288, center crop 256x256, rescale by `0.00392156862745098`, `do_flip_channel_order=true`, output NCHW.
- VOC segmentation checkpoints: resize shortest edge to 544, center crop 512x512, same rescale/channel flip, output NCHW.
- Segmentation maps: processor disables rescale and channel flip, uses nearest interpolation, squeezes channel dimension, converts to int64 labels.

Runtime packing:

- There is no token sequence or metadata input. Patch packing happens inside the GPU graph via `unfold`.
- `patch_size=2` and official crops make feature maps divisible along the inspected paths. For arbitrary H/W, source uses PyTorch `unfold`/`fold` without a divisibility check; DinoML should either reproduce exact floor-window/uncovered-edge semantics or reject non-divisible runtime shapes.

Postprocessing:

- Classification outputs `[B, num_labels]` logits.
- Segmentation raw logits are `[B, num_labels, H/output_stride, W/output_stride]`.
- `post_process_semantic_segmentation(outputs, target_sizes)` resizes each logit tensor to `(target_height, target_width)` with bilinear `align_corners=False`, then computes `argmax` over class channel.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d -> channel GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0, groups=1)
```

Replacement:

```text
NCHW/NHWC flatten spatial positions -> MatMul(weight.T) -> BiasAdd optional -> restore layout
```

Preconditions: `kernel_size=1`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`; preserve BN/activation ordering. Weight transform for NCHW semantic translation is `w.reshape(Cout, Cin)`. For NHWC optimized regions, place channel as last logical dimension and rewrite channel-axis consumers.

Failure cases: depthwise convolutions, ASPP dilated 3x3 branches, stem/local 3x3, or convs followed by source-observable NCHW outputs unless a layout guard owns all consumers.

Parity sketch: compare random tensors for qkv/out/FFN/projection 1x1 convs across static and dynamic spatial sizes, with and without bias.

### Rewrite: eval Conv2d + BatchNorm2d -> fused Conv2d

Preconditions:

- inference/eval mode, frozen BN running mean/variance/weight/bias
- same channel axis as source or a fully guarded NHWC region with transformed weights

Replacement:

```text
Conv2d(weight, bias?) -> BatchNorm2d
=> Conv2d(fused_weight, fused_bias)
```

Failure cases: training, unfrozen statistics, or graph regions where BN outputs are externally observed before activation.

### Rewrite: unfold + 1x1 conv attention -> patch-column linear attention kernel

Source pattern:

```text
feature map -> unfold(k=p, stride=p) -> [B,D,P,N] ->
qkv 1x1 -> split -> softmax over N -> sum over N -> relu(value)*context -> out 1x1 -> fold
```

Replacement:

```text
PatchBlockView/Im2Col -> fused qkv linear-attention over patch columns -> Col2Im/Fold
```

Preconditions:

- `kernel_size == stride == patch_size`
- no overlap, no padding, no dilation in unfold/fold
- feature-map H/W divisible by patch size or exact PyTorch edge behavior reproduced
- split order exactly `[1, D, D]` on channel axis
- softmax/sum axis is patch-column axis `N`, not patch-area axis `P`

Layout constraints: this is a strong NHWC candidate only if the whole local block from `conv_1x1` through `fold` and `conv_projection` is owned. Otherwise protect with `no_layout_translation()` because `split(dim=1)`, `GroupNorm(num_channels=D)`, and `fold` assume source channel placement.

Failure cases: variable non-divisible image sizes, external observation of unfolded patches, or mixed layout consumers.

Parity sketch: random `[B,D,H,W]` where H/W are divisible by 2 and non-divisible negative tests; compare fused block against source math at fp32 and fp16/bf16 tolerances.

### Rewrite: ASPP branch concat in NHWC

Source pattern:

```text
parallel conv branches -> torch.cat(branches, dim=1) -> 1x1 project
```

Replacement: guarded NHWC branch execution with `cat(axis=-1)` and channel-last 1x1 projection.

Preconditions: all branches and the project conv stay inside one layout region; bilinear upsample branch uses NHWC-compatible implementation with identical `align_corners=False`; public logits are converted back to NCHW unless downstream postprocess is also translated.

Failure cases: source-visible branch outputs, labels/loss training path, or postprocess expecting class axis at dim 1.

## 10. Kernel fusion candidates

Highest priority:

- Conv+BN+swish/relu fusion for stem, inverted residuals, local depthwise, and ASPP branches. This dominates non-attention runtime.
- 1x1 qkv projection + linear-attention elementwise/reduction + 1x1 out projection over `[B,D,P,N]`. Avoiding materialized intermediate q/k/v traffic matters.
- Unfold/fold elimination or view-based patch blocking for divisible feature maps. Official shapes make non-overlap patches predictable.

Medium priority:

- GroupNorm(num_groups=1) + adjacent 1x1 conv fusion in the transformer patch domain.
- Inverted residual depthwise + pointwise scheduling in NHWC for channel-last memory performance.
- ASPP parallel branch scheduling and concat+project fusion.

Lower priority:

- Classification global average pool + classifier fusion for batch throughput.
- Segmentation postprocess resize+argmax fusion for end-to-end segmentation serving.
- Dropout removal in eval graphs.

## 11. Runtime staging plan

1. Parse `MobileViTV2Config`; compute channel schedule with source `make_divisible`.
2. Load weights for base encoder and classification head; support NCHW semantic graph first.
3. Implement and parity-test conv stem, inverted residual, and MobileNet stages.
4. Implement MobileViTV2 unfold/fold plus linear-attention block for patch-divisible shapes.
5. Add classification pooled logits parity for 1.0 and 2.0 ImageNet configs.
6. Add segmentation model with output_stride=16 dilation behavior and ASPP head.
7. Add preprocessing/postprocessing parity for official image sizes.
8. Add guarded NHWC/fusion passes: Conv+BN+act, 1x1 GEMM, patch attention, ASPP.

Can stub initially: training losses, gradient checkpointing, label losses, non-official non-divisible input sizes, and segmentation postprocess if raw logits parity is the first milestone.

## 12. Parity and validation plan

- Unit-test `make_divisible` channel schedules for width multipliers 1.0, 1.5, 2.0.
- Random tensor parity for `MobileViTV2LinearSelfAttention` with `D in {128,192,256,512}`, `P=4`, varied `N`.
- Random tensor parity for unfold/fold round trip on divisible H/W; add negative/admission tests for non-divisible H/W.
- Single inverted residual block parity for stride 1 residual and stride 2 no-residual cases.
- One MobileViTV2Layer parity for stage 3 with `n_attn_blocks=2`.
- Encoder last-hidden-state parity after all five stages for classification and segmentation output strides.
- Classification logits parity for `apple/mobilevitv2-1.0-imagenet1k-256` and `2.0`.
- Segmentation raw logits parity for VOC checkpoints; postprocess parity for target-size resize and argmax.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-4`; fp16/bf16 start at `atol=2e-2, rtol=2e-2`, tighten per fused kernel.

## 13. Performance probes

- Processor throughput: resize/crop/rescale/channel flip for 256 and 512 crops.
- Encoder throughput by stage, separating MobileNet stages from MobileViTV2 patch stages.
- Unfold/fold materialization bytes and time versus fused/view patch path.
- Linear-attention patch kernel time as batch, `D`, and `Npatch` vary.
- Conv backend comparison: NCHW direct, channel-last/NHWC guarded region, Conv+BN fused.
- Classification images/sec across width multipliers 1.0 and 2.0.
- Segmentation throughput at 512 crop with output_stride 16; ASPP branch breakdown.
- Memory peak for segmentation logits/postprocess target-size upsample.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Beam search, generation controllers, tokenizer logic, and KV cache; not applicable.
- Multi-GPU/tensor parallelism.
- Quantized/packed checkpoint formats; not present in inspected native configs.
- Non-official arbitrary input sizes until unfold/fold edge semantics are intentionally admitted.
- ADE20K configs from conversion script unless a native checkpoint is targeted later.

## 15. Final implementation checklist

- [ ] Parse `MobileViTV2Config` and compute width-derived channels.
- [ ] Load MobileViTV2 encoder/classifier weights.
- [ ] Implement NCHW Conv2d/BatchNorm/GroupNorm/swish/depthwise ops.
- [ ] Implement MobileViTV2 `unfold`/`fold` patch contract.
- [ ] Implement separable linear self-attention `[query_1, key_D, value_D]`.
- [ ] Implement classification mean-pool + linear head.
- [ ] Implement output_stride 16 dilation path for segmentation.
- [ ] Implement ASPP segmentation head and postprocess resize+argmax.
- [ ] Add Conv+BN+activation fusion.
- [ ] Add guarded NHWC/no-layout-translation annotations for axis-sensitive regions.
- [ ] Add patch attention fusion with divisibility guards.
- [ ] Add parity tests for one block, full encoder, classification logits, segmentation logits.
- [ ] Benchmark preprocessing, encoder stages, unfold/fold, linear attention, and ASPP.
