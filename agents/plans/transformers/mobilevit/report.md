# MobileViT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: apple/mobilevit-small as primary image-classification target; apple/deeplabv3-mobilevit-small for segmentation head coverage
Config source: official Hugging Face config.json and preprocessor_config.json snapshots under _sources/
Source files inspected:
- transformers/src/transformers/models/mobilevit/configuration_mobilevit.py
- transformers/src/transformers/models/mobilevit/modeling_mobilevit.py
- transformers/src/transformers/models/mobilevit/image_processing_mobilevit.py
- transformers/src/transformers/models/mobilevit/image_processing_pil_mobilevit.py
- transformers/src/transformers/models/mobilevit/convert_mlcvnets_to_pytorch.py
Any missing files or assumptions: no gated or 401/403 gaps encountered; no Transformers import/execution; source/config inspection only.
```

Local source snapshots are in `_sources/`. Representative config snapshots were fetched from:

- [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small)
- [apple/mobilevit-x-small](https://huggingface.co/apple/mobilevit-x-small)
- [apple/mobilevit-xx-small](https://huggingface.co/apple/mobilevit-xx-small)
- [apple/deeplabv3-mobilevit-small](https://huggingface.co/apple/deeplabv3-mobilevit-small)
- [apple/deeplabv3-mobilevit-x-small](https://huggingface.co/apple/deeplabv3-mobilevit-x-small)
- [apple/deeplabv3-mobilevit-xx-small](https://huggingface.co/apple/deeplabv3-mobilevit-xx-small)
- [hf-internal-testing/tiny-random-MobileViTModel](https://huggingface.co/hf-internal-testing/tiny-random-MobileViTModel)

Authoritative implementation points: config defaults are declared in `_sources/configuration_mobilevit.py:50`; convolution, inverted residual, attention, MobileViT block, encoder, base model, classification head, and segmentation head are in `_sources/modeling_mobilevit.py:54`, `:118`, `:189`, `:325`, `:494`, `:615`, `:697`, and `:867`. The current image processor is `_sources/image_processing_mobilevit.py:59`; the PIL backend mirrors it.

## 2. High-level architecture

MobileViT is an image-only CNN/Transformer hybrid, not an autoregressive model. The first useful DinoML runtime target is image classification inference for `MobileViTForImageClassification`; semantic segmentation is a close second target because it reuses the same backbone with different output stride and a DeepLabV3-style ASPP head.

```text
CPU/data preprocessing -> NCHW pixel_values -> conv stem
  -> MobileNetV2 inverted residual stages
  -> MobileViT stages: local conv -> 2x2 patch-token unfold -> noncausal MHA/MLP -> fold -> conv fusion
  -> classification: 1x1 expansion -> global average pool -> Linear logits
  -> segmentation: hidden states -> ASPP/DeepLabV3 head -> segmentation logits -> optional postprocess resize/argmax
```

Stage decomposition:

- CPU/data pipeline: resize, center crop, rescale, optional RGB/BGR channel flip, optional segmentation-label remap.
- Backbone runtime: NCHW convolution-heavy path with three local MobileViT transformer islands.
- Classification head: global average over spatial axes and `Linear(C -> num_labels)`.
- Segmentation head: consumes backbone hidden states, specifically the final feature map, and emits low-resolution logits; postprocessing upsamples logits to requested target sizes and takes `argmax`.

## 3. Important config dimensions

Source defaults from `MobileViTConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `num_channels` | 3 | input is NCHW `[B,3,H,W]` |
| `image_size` | 256 | config/preprocessor convention, not directly enforced in forward |
| `patch_size` | 2 | MobileViT unfold/fold patch height and width |
| `hidden_sizes` | `[144,192,240]` | token channel sizes for MobileViT stages 3/4/5 |
| `neck_hidden_sizes` | `[16,32,64,96,128,160,640]` | CNN feature widths and final classification expansion width |
| `num_attention_heads` | 4 | all MobileViT blocks use 4-head MHA |
| `mlp_ratio` | 2.0 | token MLP intermediate size is `hidden_size * 2` |
| `expand_ratio` | 4.0 | MobileNetV2 inverted residual expansion, except XXS uses 2.0 |
| `hidden_act` | `silu` | conv and MLP activation unless overridden with ReLU in ASPP |
| `conv_kernel_size` | 3 | local and fusion conv kernel |
| `output_stride` | 32 | classification default; segmentation checkpoints use 16 |
| `qkv_bias` | true | Q/K/V Linear layers include bias |
| `aspp_out_channels` | 256 | segmentation head branch/project width |
| `atrous_rates` | `[6,12,18]` | segmentation ASPP dilated conv rates |

Representative checkpoint sweep:

| Checkpoint | Head | Image/preprocess crop | Hidden sizes | Neck sizes | Output stride | Labels | Operator-significant notes |
|---|---|---:|---|---|---:|---:|---|
| `apple/mobilevit-small` | classification | 256 | 144/192/240 | 16/32/64/96/128/160/640 | 32 | 1000 | largest common MobileViT v1 family |
| `apple/mobilevit-x-small` | classification | 256 | 96/120/144 | 16/32/48/64/80/96/384 | 32 | 1000 | smaller transformer/channel widths |
| `apple/mobilevit-xx-small` | classification | 256 | 64/80/96 | 16/16/24/48/64/80/320 | 32 | 1000 | `expand_ratio=2.0`, `hidden_dropout_prob=0.05` |
| `apple/deeplabv3-mobilevit-small` | segmentation | 512 | 144/192/240 | 16/32/64/96/128/160/640 | 16 | 21 | dilated layer 5; no final classification expansion |
| `apple/deeplabv3-mobilevit-xx-small` | segmentation | 512 | 64/80/96 | 16/16/24/48/64/80/320 | 16 | 21 | segmentation plus XXS expansion/dropout changes |
| `hf-internal-testing/tiny-random-MobileViTModel` | bare model | 32 | 144/192/240 | 16/32/64/96/128/160/640 | 32 | default config | useful shape-smoke config only |

## 3a. Family variation traps

- Classification and segmentation share `model_type="mobilevit"` but differ structurally: segmentation constructs `MobileViTModel(expand_output=False)` and needs hidden states for the DeepLabV3 head.
- `output_stride` changes layer 4/5 stride-vs-dilation behavior. The source sets `dilate_layer_5=True` for stride 16 and both layer 4/5 dilation for stride 8.
- XXS changes `expand_ratio` from 4.0 to 2.0 and dropout from 0.1 to 0.05. Do not derive inverted-residual channel widths from checkpoint name alone.
- Current image processor uses `do_flip_channel_order`; older official preprocessor configs use `feature_extractor_type="MobileViTFeatureExtractor"` and `do_flip_channels`. Treat this as a processor/config compatibility issue, not a model graph op.
- Source is NCHW throughout. NHWC/channel-last is an optimization candidate only inside guarded conv/BN/activation regions and must rewrite `dim=1` concat/channel operations, BatchNorm channel axis, and all reshape/transposes in MobileViT unfold/fold.
- Attention hidden sizes are all divisible by 4 in inspected configs; source rejects non-divisible hidden size/head count.
- No positional embeddings exist. Adding synthetic position state for transformer islands would be a parity bug.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW contiguous tensor contract for `pixel_values`.
- `reshape`, `view`, `transpose`, `permute`, `contiguous` for patch-token unfold/fold.
- `torch.cat(..., dim=1)` for MobileViT fusion and ASPP branch concat.
- Spatial reductions: `mean(dim=[-2,-1])` for classification global average pooling.
- Bilinear `interpolate(..., align_corners=False)` for non-divisible patch repair and segmentation postprocess; adaptive average pool to `1x1` in ASPP pooling.

Neural network primitives:

- Conv2d NCHW with kernel 1 and 3, stride 1/2, zero padding, dilation 1/2/ASPP rates, groups 1 or depthwise `groups=expanded_channels`.
- BatchNorm2d inference with affine/running stats.
- Activations: SiLU for backbone, ReLU in ASPP.
- Linear projections for attention, MLP, and classifier.
- LayerNorm over last token channel.
- Dropout and Dropout2d are inference identities.

Attention primitives:

- Noncausal encoder self-attention over local patch-token sequences.
- Per MobileViT stage: `Linear(hidden -> hidden)` Q/K/V with optional bias, reshape to `[B*patch_area, heads, num_patches, head_dim]`, scaled QK^T, softmax on key dimension, AV, output projection.

Preprocessing-coupled ops:

- Resize shortest edge, center crop, rescale by `1/255` when enabled, optional RGB/BGR flip.
- Segmentation label preprocessing: nearest resize, no rescale/channel flip, squeeze channel, cast int64, optional reduce labels.

Structured-output postprocessing:

- Segmentation logits `[B,num_labels,h,w]` optionally bilinear-resized to per-image `target_sizes`, then `argmax(dim=0)` per image or `argmax(dim=1)` batched without target sizes.
- No NMS, box decode, masks, or variable-object filtering.

## 5. Layer/block breakdown

Backbone for classification checkpoints with crop 256 and output stride 32, using source NCHW axes:

```text
Input: pixel_values [B,3,256,256]
conv_stem: Conv3x3 stride2 3->C0 + BN + SiLU -> [B,C0,128,128]
layer1: 1 inverted residual, stride1 C0->C1
layer2: 3 inverted residuals, first stride2 C1->C2
layer3: MobileViTLayer stride2 C2->C3, hidden H0, 2 transformer layers
layer4: MobileViTLayer stride2 C3->C4, hidden H1, 4 transformer layers
layer5: MobileViTLayer stride2 C4->C5, hidden H2, 3 transformer layers
conv_1x1_exp: Conv1x1 C5->C6 + BN + SiLU
pool: mean over H,W -> [B,C6]
classifier: Linear(C6 -> num_labels)
```

Inverted residual:

```text
x -> Conv1x1 C->make_divisible(C*expand_ratio,8) + BN + SiLU
  -> depthwise Conv3x3 stride/dilation + BN + SiLU
  -> Conv1x1 expanded->out + BN
  -> residual add only when stride==1 and in_channels==out_channels
```

MobileViT layer:

```text
optional downsampling inverted residual
residual = x
x = ConvKxK(local) + BN + SiLU
x = Conv1x1(C -> hidden, no BN, no activation)
patches = unfold_2x2(x) -> [B*4, num_patches, hidden]
repeat N times:
  y = LayerNorm(patches)
  y = MHA(y) -> Linear output
  patches = patches + y
  y = LayerNorm(patches)
  y = Linear(hidden -> 2*hidden) -> SiLU -> Linear(2*hidden -> hidden)
  patches = patches + y
patches = LayerNorm(patches)
x = fold_2x2(patches) -> [B,hidden,H,W]
x = Conv1x1(hidden -> C) + BN + SiLU
x = ConvKxK(cat(residual, x), 2C -> C) + BN + SiLU
```

Segmentation head:

```text
backbone = MobileViTModel(expand_output=False), output_hidden_states=True
feature = encoder_hidden_states[-1]  # [B,C5,H/16,W/16] for output_stride=16
ASPP branches:
  Conv1x1 C5->256
  Conv3x3 dilation 6 C5->256
  Conv3x3 dilation 12 C5->256
  Conv3x3 dilation 18 C5->256
  AdaptiveAvgPool2d(1) -> Conv1x1 C5->256 -> bilinear resize to feature spatial size
cat dim=1 -> Conv1x1 1280->256 -> Dropout identity in inference
classifier Conv1x1 256->num_labels with bias, no BN/activation
```

## 6. Attention requirements

MobileViT uses noncausal self-attention only inside encoder-style MobileViT blocks. There is no autoregressive prefill/decode, no KV cache, no causal mask, no cross-attention, no RoPE/ALiBi/relative bias, and no packed/varlen attention metadata.

Attention shape for one MobileViT stage:

```text
patches: [B * patch_area, Np, H]
patch_area = patch_size * patch_size = 4 for inspected configs
Np = ceil(feature_h / patch_h) * ceil(feature_w / patch_w)
heads = 4
head_dim = H / 4
q,k,v: [B*4, Np, H] -> view [B*4, Np, 4, head_dim] -> transpose [B*4, 4, Np, head_dim]
scores: [B*4, 4, Np, Np] / sqrt(head_dim)
softmax dim=-1
context: [B*4, 4, Np, head_dim] -> [B*4, Np, H]
```

For 256 classification inputs, inferred from source strides and `patch_size=2`, the three transformer islands have approximate sequence lengths `Np = 1024, 256, 64`. For 512 segmentation with `output_stride=16`, layer 5 is dilated rather than downsampled, so later transformer sequence lengths are larger than classification at the same input size.

SDPA/FlashAttention compatibility: the math is standard noncausal MHA with no mask and dropout disabled in eval. A fused attention primitive can replace QK/softmax/AV if it preserves scaling by `sqrt(head_dim)`, softmax axis, and `[B*patch_area, heads, Np, head_dim]` packing.

## 7. Position encoding and custom math

No position encoding is present. The model relies on local convolutions before and after the transformer blocks plus patch ordering.

Patch-token layout is the custom parity-critical math:

```python
def mobilevit_unfold_nchw(x, patch_h=2, patch_w=2):
    B, C, H, W = x.shape
    new_h = ceil(H / patch_h) * patch_h
    new_w = ceil(W / patch_w) * patch_w
    if (new_h, new_w) != (H, W):
        x = bilinear_interpolate(x, size=(new_h, new_w), align_corners=False)
    nph, npw = new_h // patch_h, new_w // patch_w
    patches = x.reshape(B * C * nph, patch_h, npw, patch_w)
    patches = patches.transpose(1, 2)
    patches = patches.reshape(B, C, nph * npw, patch_h * patch_w)
    patches = patches.transpose(1, 3)
    return patches.reshape(B * patch_h * patch_w, nph * npw, C)
```

Folding is the exact inverse sequence from `_sources/modeling_mobilevit.py:439`, with optional bilinear resize back to the original feature spatial size if unfold had to interpolate.

## 8. Preprocessing and input packing

The processor emits `pixel_values` for the model graph. Official classification preprocessor configs use resize size 288 and crop 256; segmentation uses resize size 544 and crop 512. The current image processor defaults to shortest edge 224 and crop 256, but official config snapshots override those values.

Processor contract:

- Input images are converted/prepared by the image backend, resized, center-cropped, rescaled, and channel-flipped by default.
- Output layout is channel-first image tensors compatible with model NCHW input.
- Source defaults include ImageNet mean/std constants but `do_normalize=None`; the MobileViT `_preprocess` implementation inspected applies resize, crop, rescale, channel flip, and label reduction, not normalization.
- Segmentation maps use expected 2D input, nearest interpolation, no rescale, no channel flip, squeeze channel, int64 labels.

CPU/data pipeline vs runtime: resize/crop/rescale/channel flip and label handling can initially remain outside DinoML compiled runtime. The runtime graph starts at `pixel_values`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: inference Conv2d + BatchNorm2d + activation fold

Source pattern: `MobileViTConvLayer` applies Conv2d, optional BatchNorm2d, optional activation.

Replacement: fold BN affine/running stats into Conv weights/bias at load time, then lower as Conv2d + activation.

Preconditions:

- Inference mode with frozen BN running statistics.
- Conv and BN are adjacent with no intervening consumer.
- Activation is SiLU or ReLU and can be fused by backend.

Failure cases: training mode, unfrozen BN, or need to expose intermediate hidden states before BN.

Parity test sketch: compare one folded `MobileViTConvLayer` and full backbone logits in fp32, then fp16 with relaxed tolerance.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern: stem expansion/projection and ASPP/project/classifier 1x1 convs.

Replacement:

```text
NCHW tensor -> channel-last or flattened [B*H*W, Cin] -> GEMM(Cin,Cout) -> restore layout
```

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`.
- Layout pass owns both producer and consumer or inserts explicit layout transforms.
- Weight transform from PyTorch Conv2d `[Cout,Cin,1,1]` to GEMM RHS `[Cin,Cout]`.

Failure cases: grouped/depthwise conv, stride/padding not identity, exposed NCHW consumer without restore.

### Rewrite: MobileViT unfold/fold as guarded layout op

Source pattern: reshape/transpose sequence in `unfolding` and inverse in `folding`.

Replacement: one explicit `mobilevit_patch_unfold` and `mobilevit_patch_fold` metadata/lowering pair, or a fused kernel around the transformer island.

Preconditions:

- `patch_size` static, inspected checkpoints use 2.
- Input feature map is dense NCHW and spatial dims are known or runtime-divisible/repairable by the source bilinear interpolation rule.
- Folding receives the original `info_dict` shape metadata.

Shape equations:

```text
new_h = ceil(H/ph) * ph
new_w = ceil(W/pw) * pw
Np = (new_h/ph) * (new_w/pw)
unfold: [B,C,H,W] -> [B*ph*pw,Np,C]
fold: [B*ph*pw,Np,C] -> [B,C,new_h,new_w] -> optional resize [B,C,H,W]
```

Failure cases: non-dense/strided tensors, non-static patch size, NHWC rewrite without exact axis/order update.

### Rewrite: local transformer island to batched GEMM/attention

Source pattern: independent Q/K/V linear layers and standard MHA over `[B*4,Np,H]`.

Replacement: combine Q/K/V into one packed projection followed by fused noncausal attention and output GEMM.

Preconditions:

- `qkv_bias` setting known; inspected configs use true.
- Packed weight layout must be `[Q;K;V]` or separately recorded with split order Q, K, V.
- No attention mask, no position encoding, dropout disabled.

Failure cases: hidden size not divisible by heads, custom config changing `qkv_bias`, or runtime requiring attention probabilities output.

### Rewrite: guarded NHWC/channel-last conv region

Source pattern: long Conv/BN/activation regions before and after patch islands.

Replacement: translate local regions to NHWC for conv kernels, preserving source graph boundaries at unfold/fold and `cat(dim=1)` unless the whole region is owned.

Required axis rewrites:

- Channel concat `dim=1` becomes `dim=-1`.
- BatchNorm2d channel axis becomes last.
- Spatial mean remains over spatial axes, but indices change from `[-2,-1]` in NCHW to `[1,2]` or equivalent in NHWC.
- MobileViT unfold/fold reshape order must be rederived, not blindly reused.

Failure cases: public hidden states expected in NCHW, segmentation head consuming hidden states, or mixed layout consumers.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BatchNorm2d + SiLU/ReLU inference fusion. This dominates the CNN backbone and is required before MobileViT looks competitive.
- Depthwise Conv3x3 + BN + SiLU. Inverted residual stages are rich in depthwise conv; a poor depthwise path will bottleneck small variants.
- MobileViT patch unfold/fold. The reshape/transpose chain is small but layout-sensitive and repeated three times; explicit lowering avoids accidental copies.

Medium priority:

- Packed QKV projection + fused noncausal attention for `[B*4,Np,H]`. Sequence lengths are modest, but this removes multiple GEMMs and transposes inside each transformer island.
- LayerNorm + Linear/MLP activation fusion for token MLPs.
- 1x1 Conv as GEMM for projection-heavy blocks and ASPP.

Lower priority:

- ASPP branch fusion/parallel scheduling for segmentation. Useful only after classification/backbone parity.
- Segmentation postprocess resize + argmax. It is end-to-end useful but not part of the core model forward unless DinoML owns postprocessing.

## 11. Runtime staging plan

1. Parse `MobileViTConfig`, load classification checkpoint weights, and admit NCHW `pixel_values`.
2. Implement/fuse inference `MobileViTConvLayer` and inverted residual blocks; validate through stem/layer1/layer2.
3. Implement explicit MobileViT unfold/fold and standard encoder MHA/MLP over patch tokens; validate one MobileViT block.
4. Run full `MobileViTModel` classification backbone with final 1x1 expansion and global average pooling.
5. Add `MobileViTForImageClassification` head and ImageNet logits parity.
6. Add segmentation path: `expand_output=False`, hidden-state capture, ASPP, classifier conv, and postprocess parity.
7. Add guarded NHWC/channel-last and QKV/attention fusions once source-faithful NCHW parity is stable.

Initially stub/drop:

- Training losses, labels, dropout behavior in train mode, gradient checkpointing, hidden-state outputs except where segmentation needs them.
- Processor execution inside compiled graph; keep it in CPU/data pipeline.

## 12. Parity and validation plan

- Unit parity for `make_divisible` channel expansion decisions from config values.
- Random tensor parity for Conv+BN+activation, inverted residual residual/no-residual cases, depthwise conv, and dilated conv.
- Random tensor parity for `mobilevit_unfold`/`fold`, including dimensions divisible by 2 and non-divisible dimensions that trigger bilinear repair.
- Single `MobileViTTransformerLayer` parity for `[B*4,Np,H]` with H values 64, 80, 96, 120, 144, 192, 240.
- One complete MobileViT layer parity for stage 3/4/5 shapes.
- Full classification logits parity for small/x-small/xx-small at official preprocessed crop 256.
- Full segmentation logits parity for DeepLabV3 small and XXS at crop 512, plus postprocess resize/argmax parity for target sizes.
- Recommended tolerances: fp32 max/rtol tight enough for Conv/Linear reorder checks; fp16/bf16 should use relaxed tolerances around attention softmax and bilinear interpolation.

## 13. Performance probes

- Processor throughput separately from model runtime: resize/crop/rescale/channel flip for 256 and 512 crops.
- Backbone-only throughput by variant: XXS/XS/S, batch sweep 1/8/32.
- Per-stage timings: stem/layer1/2, each MobileViT stage, final expansion/classifier.
- Patch-token attention probe by `Np` and hidden size: classification-like `1024/256/64` and segmentation-like larger maps.
- NCHW vs guarded NHWC conv-region benchmarks, with explicit layout conversion cost.
- Depthwise conv backend comparison.
- Segmentation ASPP throughput and memory bandwidth, especially five-branch concat/project.
- End-to-end classification images/sec and segmentation images/sec including optional postprocess.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout randomness; inference treats dropout as identity.
- Gated/private checkpoints: none encountered.
- MobileViTV2: separate `mobilevitv2` family, not covered by this report.
- Dynamic arbitrary image sizes beyond source-compatible bilinear patch repair can wait until fixed 256/512 parity works.
- Full NHWC translation by default; only guarded local rewrites after NCHW parity.
- Quantization, Core ML metadata, and original ml-cvnets conversion script execution.

## 15. Final implementation checklist

- [ ] Parse `MobileViTConfig` including `hidden_sizes`, `neck_hidden_sizes`, `output_stride`, `patch_size`, `expand_ratio`, `qkv_bias`, and ASPP fields.
- [ ] Load Conv2d, BatchNorm2d, Linear, LayerNorm weights with source names and preserve head-specific model class.
- [ ] Implement NCHW Conv2d coverage for standard, depthwise, stride 2, dilation, and 1x1 cases.
- [ ] Add inference Conv+BN folding and SiLU/ReLU fusion.
- [ ] Implement MobileViT inverted residual block parity.
- [ ] Implement `mobilevit_patch_unfold` and `mobilevit_patch_fold` with bilinear repair.
- [ ] Implement noncausal MHA for `[B*patch_area,Np,H]` with Q/K/V bias and softmax dim `-1`.
- [ ] Implement LayerNorm and SiLU MLP for token transformer layers.
- [ ] Implement global average pooling over spatial axes and classification Linear head.
- [ ] Implement segmentation hidden-state return path, ASPP branches, classifier Conv1x1, and postprocess resize/argmax.
- [ ] Add guarded QKV packing/fused attention rewrite.
- [ ] Add guarded 1x1 Conv -> GEMM rewrite.
- [ ] Add guarded NHWC/channel-last conv-region rewrite with explicit axis rewrites.
- [ ] Add parity tests for small/x-small/xx-small classification and small/xx-small segmentation configs.
- [ ] Benchmark processor, backbone, transformer islands, classification, and segmentation separately.
