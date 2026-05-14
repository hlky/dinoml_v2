# Mask2Former Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary inference target: Mask2FormerForUniversalSegmentation for semantic,
  instance, and panoptic segmentation from preprocessed images.
  Representative configs:
    facebook/mask2former-swin-tiny-coco-instance
    facebook/mask2former-swin-small-coco-instance
    facebook/mask2former-swin-small-ade-semantic
    facebook/mask2former-swin-base-coco-panoptic
    facebook/mask2former-swin-large-coco-panoptic

Config source:
  https://huggingface.co/facebook/mask2former-swin-small-coco-instance/raw/main/config.json
  https://huggingface.co/facebook/mask2former-swin-small-coco-instance/raw/main/preprocessor_config.json
  Same raw paths for the sweep checkpoints above.

Source files inspected:
  X:/H/transformers/src/transformers/models/mask2former/configuration_mask2former.py
  X:/H/transformers/src/transformers/models/mask2former/modeling_mask2former.py
  X:/H/transformers/src/transformers/models/mask2former/modular_mask2former.py
  X:/H/transformers/src/transformers/models/mask2former/image_processing_mask2former.py
  X:/H/transformers/src/transformers/models/mask2former/image_processing_pil_mask2former.py
  X:/H/transformers/src/transformers/models/swin/modeling_swin.py
  X:/H/transformers/src/transformers/backbone_utils.py

Any missing files or assumptions:
  No remote code is required for the inspected official checkpoints. The
  modeling and image-processing files are generated from modular sources in
  this checkout; future upstream edits should prefer modular_mask2former.py.
  This report targets CUDA inference first. CPU/torchvision preprocessing and
  Python postprocessing may remain outside the compiled graph initially.
  Training loss, Hungarian matching, point sampling, and auxiliary loss
  supervision are documented but deferred.
```

## 2. High-level architecture

Mask2Former is a universal image segmentation model: a Swin backbone produces multi-scale NCHW feature maps, a pixel decoder refines them with multi-scale deformable attention and FPN-style fusion, and a learned-query transformer decoder predicts a set of class logits plus dense masks.

```text
CPU image preprocessing -> pixel_values/pixel_mask
  -> Swin backbone feature pyramid
  -> pixel decoder: 1x1 projections + 6 deformable-attention encoder layers + FPN
  -> learned query decoder with masked cross-attention over multi-scale features
  -> class head + mask embedding head
  -> semantic/instance/panoptic postprocess
```

Stage decomposition:

- CPU/data pipeline: decode, resize, rescale, normalize, pad batch, emit `pixel_values [B,3,H,W]` and `pixel_mask [B,H,W]`.
- Backbone: Swin patch embedding, shifted-window self-attention, patch merging, and NCHW feature maps for `stage1..stage4`.
- Pixel decoder: projects the last 3 Swin stages to `feature_size=256`, flattens spatial maps, runs 6 multi-scale deformable attention encoder layers, then fuses one higher-resolution FPN level to produce `mask_features [B,256,H/4,W/4]` and 3 multi-scale decoder features.
- Transformer decoder: learned query content and learned query position embeddings; 9 masked-attention decoder layers for the source default `decoder_layers=10`, because the first prediction happens before the loop.
- Heads: `Linear(hidden_dim -> num_labels+1)` class head and a 3-layer mask MLP whose output is combined with pixel embeddings by `einsum("bqc,bchw->bqhw")`.
- Postprocessing: semantic, instance, and panoptic map construction from `[B,Q,C+1]` class logits and `[B,Q,Hm,Wm]` mask logits.

Independently stageable units: processor handoff, Swin backbone, pixel decoder deformable attention, query decoder without postprocess, raw class/mask heads, and each postprocess mode.

## 3. Important config dimensions

Worked example: `facebook/mask2former-swin-small-coco-instance`.

| Field | Value | Source |
| --- | ---: | --- |
| `model_type` | `mask2former` | config.json |
| primary task | universal segmentation | architecture/source |
| `hidden_dim` | 256 | config.json |
| `feature_size` / `mask_feature_size` | 256 / 256 | config.json |
| pixel decoder encoder layers | 6 | config.json |
| decoder layers field | 10 | config.json |
| actual masked-attention layers | 9 | source: `decoder_layers - 1` |
| decoder / pixel-decoder heads | 8 | config.json |
| head dim | 32 | inferred `256 / 8` |
| pixel decoder FFN | 1024 | config.json |
| query decoder FFN | 2048 | config.json |
| activation | ReLU | config/source |
| `num_queries` | 100 | config.json |
| class logits | `num_labels + 1` | source; extra no-object class |
| feature strides | `[4,8,16,32]` | config.json |
| transformer feature levels | 3 | source constant |
| deformable points per level | 4 | source constant |
| common stride | 4 | config.json |
| backbone | Swin | config/source support list |
| Swin patch size | 4 | backbone config |
| Swin depths | `[2,2,18,2]` | small config |
| Swin heads | `[3,6,12,24]` | small config |
| Swin window size | 7 | small config |
| processor resize | `384x384` in inspected configs | preprocessor_config |
| processor size divisor | 32 | preprocessor_config |
| processor mean/std | ImageNet mean/std | preprocessor_config |

Representative checkpoint sweep:

| Checkpoint | Labels | Queries | Swin embed | Swin hidden | Swin depths | Swin heads | Window | Target |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| `facebook/mask2former-swin-tiny-coco-instance` | 80 | 100 | 96 | 768 | `2,2,6,2` | `3,6,12,24` | 7 | COCO instance |
| `facebook/mask2former-swin-small-coco-instance` | 80 | 100 | 96 | 768 | `2,2,18,2` | `3,6,12,24` | 7 | COCO instance |
| `facebook/mask2former-swin-small-ade-semantic` | 150 | 100 | 96 | 768 | `2,2,18,2` | `3,6,12,24` | 7 | ADE semantic |
| `facebook/mask2former-swin-base-coco-panoptic` | 133 | 100 | 128 | 1024 | `2,2,18,2` | `4,8,16,32` | 12 | COCO panoptic |
| `facebook/mask2former-swin-large-coco-panoptic` | 133 | 200 | 192 | 1536 | `2,2,18,2` | `6,12,24,48` | 12 | COCO panoptic |

Defaults from `Mask2FormerConfig` that older configs may omit: `feature_size=256`, `mask_feature_size=256`, `encoder_layers=6`, `decoder_layers=10`, `num_attention_heads=8`, `num_queries=100`, `use_auxiliary_loss=True`, `output_auxiliary_logits=None`, `feature_strides=(4,8,16,32)`, `common_stride=4`, `dropout=0.0`, and default Swin backbone config `depths=[2,2,18,2]`, `drop_path_rate=0.3`, `out_features=["stage1","stage2","stage3","stage4"]`.

## 3a. Family variation traps

- The source ABI is NCHW. `pixel_values`, Swin patch Conv2d, backbone feature maps, pixel decoder Conv2d/GroupNorm/FPN, mask embeddings, and postprocess masks all use `[B,C,H,W]` or `[B,Q,H,W]`. NHWC/channel-last should be an internal optimized region with explicit boundary rewrites.
- Swin itself alternates sequence `[B,H*W,C]` and spatial NHWC-like `[B,H,W,C]` inside window partitioning, then backbone outputs are converted back to NCHW. Layout passes must not silently change the public model axes.
- Only Swin is listed as supported in config, though unsupported backbone configs may warn rather than hard fail. DinoML should initially admit official Swin checkpoints only.
- `decoder_layers=10` does not mean 10 masked-attention layer applications. Source creates `decoder_layers - 1` layers and emits an initial prediction before the first layer, giving 10 prediction stages for auxiliary loss.
- Large panoptic uses `num_queries=200`; most other inspected checkpoints use 100. Query count affects decoder attention, class logits, postprocess top-k, and mask memory.
- `num_labels` changes by task: 80 COCO instance, 133 COCO panoptic, 150 ADE semantic. The class head always has one extra no-object class.
- The current model forward accepts `pixel_mask` but does not pass it into the pixel level module; pixel decoder masks are internally all-valid zeros for projected feature maps. Processor `pixel_mask` is still part of the public tensor contract and may matter if future source changes restore padded-mask use.
- Pixel decoder deformable attention uses `grid_sample` over per-level NCHW values and learned sampling offsets. This is not standard dense attention.
- Query decoder cross-attention uses PyTorch `nn.MultiheadAttention` with boolean `attn_mask` shaped after flattening heads: `[B*heads,Q,H_l*W_l]`. Masked attention is derived from the previous mask prediction, thresholded after sigmoid, repeated over heads, detached, and then all-true rows are zeroed out.
- Position embeddings are DETR-style 2D sine embeddings generated from masks/shapes, not learned RoPE/ALiBi. Swin backbone additionally uses learned relative position bias inside shifted-window attention.
- Semantic/instance/panoptic postprocess contains Python loops and variable-size outputs. Initial runtime can return raw logits and leave postprocess outside the compiled graph.
- Training-only code constructs binary masks, samples points, computes focal/dice-like losses, and calls SciPy Hungarian matching. These should not gate inference integration.
- Axis-sensitive layout traps:
  - `pixel_values` is `[B,3,H,W]`; `pixel_mask` is `[B,H,W]`.
  - Swin patch projection is Conv2d OIHW with `kernel_size=stride=patch_size=4`.
  - Feature flattening is `x.flatten(2).transpose(1,2)`, preserving NCHW row-major spatial order.
  - Sine embedding cumsums height axis 1 and width axis 2 on `[B,H,W]` masks, then returns NCHW.
  - Pixel decoder `interpolate(..., size=current_fpn.shape[-2:], mode="bilinear", align_corners=False)` is spatial-axis sensitive.
  - Postprocess `target_sizes` are `(height,width)` and masks are resized on spatial `[-2:]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `pixel_values [B,3,H,W]`, padded to multiples of 32 by processor for inspected configs.
- Optional `pixel_mask [B,H,W]` int64 from processor; source forward currently defaults/ignores it downstream.
- Conv2d patch embedding, 1x1 projections, 3x3 FPN output convs, and 1x1 mask projection.
- Flatten spatial axes, transpose/permute, split by level lengths, reshape back to NCHW.
- `torch.cat`, `torch.stack`, `view`, `repeat`, `unsqueeze`, `flatten`, `contiguous`.
- Bilinear and nearest interpolation for FPN, mask attention creation, and postprocess resizing.
- `grid_sample` with bilinear, zeros padding, `align_corners=False` for deformable attention and training point sampling.
- Boolean mask creation, comparisons, `masked_fill`, detach, and per-image loops in postprocess.

Neural network primitives:

- Swin backbone: Conv2d patch embedding, LayerNorm, shifted-window MHA, learned relative position bias gather, GELU MLP, residual adds, patch merging by 2x2 channel concat plus Linear.
- Pixel decoder: Conv2d 1x1/3x3, GroupNorm(32), ReLU, LayerNorm, Linear, residual adds, softmax over deformable sampling points.
- Transformer decoder: learned embeddings, LayerNorm, Linear, `nn.MultiheadAttention` cross-attention, custom self-attention, ReLU FFN.
- Heads: `Linear(256 -> num_labels+1)` class predictor; 3-layer MLP `256 -> 256 -> 256 -> 256` mask embedder; mask logits via `einsum("bqc,bchw->bqhw")`.

Attention primitives:

- Swin local noncausal window attention with optional shifted-window additive masks and relative position bias.
- Pixel decoder multi-scale deformable attention with heads=8, levels=3, points=4, per-query learned offsets and weights.
- Query decoder self-attention over learned queries, MHA heads=8, head_dim=32, no causal mask/cache.
- Query decoder masked cross-attention over one of three feature levels per layer, cyclically selected by `idx % 3`.

Position/custom math:

- Swin relative position bias table lookup by static window index.
- Mask2Former sine position embedding for pixel decoder and query decoder feature levels.
- Deformable reference point generation from spatial shapes and valid ratios.
- Sigmoid-threshold mask attention generation and all-true mask-row suppression.

Postprocessing-coupled ops:

- Semantic: class softmax excluding no-object, mask sigmoid, `einsum("bqc,bqhw->bchw")`, optional bilinear resize, argmax over classes.
- Instance: class softmax excluding no-object, flatten class/query scores, `topk(num_queries)`, integer divide by `num_classes` to recover query ids, threshold masks, average mask probability, score filtering, optional nearest resize, optional binary maps or RLE.
- Panoptic: max class score/label, remove low-score/no-object queries, weight mask probabilities by scores, argmax over queries per pixel, overlap/area filtering, optional class fusion.

Training/loss deferred ops:

- Hungarian matching with SciPy `linear_sum_assignment`.
- Random point sampling, uncertainty top-k, sampled `grid_sample`, sigmoid cross entropy, dice loss, cross entropy with no-object weight, auxiliary per-stage losses.

## 5. Layer/block breakdown

Processor output:

```text
pixel_values: [B,3,Hpad,Wpad] float, NCHW, ImageNet normalized
pixel_mask:   [B,Hpad,Wpad] int64, 1 for valid image pixels and 0 for padding
```

Swin backbone, official checkpoints:

```text
x = Conv2d(3 -> embed_dim, kernel=4, stride=4)(pixel_values)
x = flatten spatial -> [B,H/4*W/4,embed_dim]
for each Swin stage:
  repeat depth_i times:
    y = LayerNorm(x)
    y = window_partition(y as [B,H,W,C])
    y = MHA(q,k,v) + relative_position_bias + optional shifted-window mask
    y = window_reverse(y)
    x = residual + y
    x = x + MLP(LayerNorm(x))                  # GELU MLP, ratio 4
  optional patch_merge: 2x2 channel concat -> LayerNorm(4C) -> Linear(4C -> 2C)
feature_maps = stage1..stage4 as NCHW maps
```

For a `384x384` processor output and patch size 4, Swin feature strides are approximately:

```text
stage1: [B, embed_dim,     96,96]  stride 4
stage2: [B, 2*embed_dim,   48,48]  stride 8
stage3: [B, 4*embed_dim,   24,24]  stride 16
stage4: [B, 8*embed_dim,   12,12]  stride 32
```

Pixel decoder setup:

```text
inputs = stage4, stage3, stage2                     # last 3 levels, low to high after reverse
for each level:
  p_l = Conv2d(C_l -> 256, 1x1) -> GroupNorm(32)
  pos_l = sine_position_embedding(shape(p_l))        # [B,256,H_l,W_l]
flat = cat(p_l.flatten(2).transpose(1,2), dim=1)    # [B,S,256]
pos_flat = cat(pos_l.flatten(2).transpose(1,2) + level_embed[l])
```

Pixel decoder encoder layer, repeated 6 times:

```text
residual = x
q = x + pos
offsets = Linear(256 -> heads*levels*points*2)(q)
weights = softmax(Linear(256 -> heads*levels*points)(q), dim=-1)
value = Linear(256 -> 256)(x).view(B,S,heads,32)
sampled = grid_sample(level_values, sampling_locations)
y = sum(sampled * weights)
y = Linear(256 -> 256)(y)
x = LayerNorm(residual + y)

residual = x
y = Linear(256 -> 1024) -> ReLU -> Linear(1024 -> 256)
x = LayerNorm(residual + y)
```

FPN/mask feature output:

```text
encoded levels -> reshape back to NCHW outputs
stage1 lateral = Conv2d(C_stage1 -> 256, 1x1, bias=False) -> GroupNorm(32)
out = lateral + interpolate(previous, size=stage1.HW, bilinear)
out = Conv2d(256 -> 256, 3x3, padding=1, bias=False) -> GroupNorm(32) -> ReLU
mask_features = Conv2d(256 -> 256, 1x1)(out)        # usually stride 4
multi_scale_features = first 3 decoder outputs
```

Transformer query module:

```text
for each of 3 multi_scale_features:
  src_l = optional Conv2d(256 -> 256, 1x1)(feature_l) + level_embedding[l]
  src_l = src_l.flatten(2).permute(2,0,1)           # [H_l*W_l,B,256]
  pos_l = sine_position_embedding(feature_l).flatten(2).permute(2,0,1)

query_pos = Embedding(num_queries,256).unsqueeze(1).repeat(1,B,1)
query_content = Embedding(num_queries,256).unsqueeze(1).repeat(1,B,1)
```

Masked-attention decoder:

```text
hidden = query_content                              # [Q,B,256]
normed = LayerNorm(hidden)
mask_logits, attn_mask = mask_predictor(normed, mask_features, feature_size_list[0])

for idx in range(decoder_layers - 1):
  level = idx % 3
  attn_mask = zero_rows_where_all_positions_masked(attn_mask)

  # post-norm default because pre_norm=False in inspected configs
  y = MultiheadAttention(
        query=hidden + query_pos,
        key=src_level + pos_level,
        value=src_level,
        attn_mask=attn_mask)                        # masked cross-attention first
  hidden = LayerNorm(hidden + y)

  y = self_attention(q=hidden+query_pos, k=hidden+query_pos, v=hidden)
  hidden = LayerNorm(hidden + y)

  y = Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)
  hidden = LayerNorm(hidden + y)

  normed = LayerNorm(hidden)
  mask_logits, attn_mask = mask_predictor(normed, mask_features, feature_size_list[(idx+1)%3])
```

Heads:

```text
for each normalized intermediate decoder state:
  class_logits = Linear(256 -> num_labels + 1)(state.transpose(0,1))

mask_embed = MLP(256 -> 256 -> 256 -> 256)(state.transpose(0,1))
mask_logits = einsum("bqc,bchw->bqhw", mask_embed, mask_features)

returned inference outputs:
  class_queries_logits: [B,Q,num_labels+1]
  masks_queries_logits: [B,Q,H/4,W/4] before postprocess resize
```

## 6. Attention requirements

- Swin attention: noncausal local self-attention inside windows. Q/K/V are separate biased Linear projections, attention scores are scaled, relative position bias is added, shifted-window masks may be added, softmax is fp32 in eager path, then output projection. No KV cache.
- Pixel decoder attention: multi-scale deformable self-attention over flattened image features. It is not expressible as standard QK attention without a specialized lowering. Required tensors include `value [B,S,heads,32]`, `sampling_offsets [B,S,heads,3,4,2]`, `attention_weights [B,S,heads,3,4]`, normalized reference points, and `grid_sample` over each feature level.
- Query decoder self-attention: noncausal MHA over `Q` learned queries, heads=8, head_dim=32. Q/K receive query position embeddings; V uses the unpositioned hidden state.
- Query decoder cross-attention: noncausal MHA with `target_len=Q`, source length equal to one selected feature level. Q receives query position; K receives spatial sine position; V is unpositioned feature level. Boolean `attn_mask` blocks background regions predicted by the previous mask.
- Masking style: no causal masks, no packed/varlen metadata, no KV cache. Cross-attention mask is boolean and shape-checked by PyTorch `nn.MultiheadAttention` semantics after flattening heads.
- FlashAttention/SDPA: Swin may dispatch through the generic Transformers attention interface. Query decoder uses `nn.MultiheadAttention`; masked cross-attention can potentially lower to dense attention with boolean additive masks, but deformable attention needs its own op or rewrite.

## 7. Position encoding and custom math

Mask2Former sine position embedding:

```python
def mask2former_sine_pos(shape, dtype, mask=None, num_feats=128, temperature=10000):
    # shape is [B,C,H,W]; mask is [B,H,W] bool where True means masked.
    if mask is None:
        mask = zeros([B,H,W], dtype=bool)
    not_mask = (~mask).to(dtype)
    y = cumsum(not_mask, axis=1)
    x = cumsum(not_mask, axis=2)
    y = y / (y[:, -1:, :] + 1e-6) * (2*pi)
    x = x / (x[:, :, -1:] + 1e-6) * (2*pi)
    dim = temperature ** (2 * floor(arange(num_feats) / 2) / num_feats)
    px = stack(sin(x[..., 0::2] / dim[0::2]), cos(x[..., 1::2] / dim[1::2])).flatten()
    py = stack(sin(y[..., 0::2] / dim[0::2]), cos(y[..., 1::2] / dim[1::2])).flatten()
    return concat([py, px], dim=-1).permute(0,3,1,2)
```

Deformable attention core:

```python
value_l = value_l.flatten(2).transpose(1, 2).reshape(B * heads, head_dim, H_l, W_l)
grid_l = 2 * sampling_locations[:, :, :, l] - 1
sampled_l = grid_sample(value_l, grid_l.transpose(1, 2).flatten(0, 1),
                        mode="bilinear", padding_mode="zeros", align_corners=False)
out = sum(stack(sampled_l) * attention_weights).view(B, heads * head_dim, queries)
```

Masked-attention mask generation:

```python
mask_logits = einsum("bqc,bchw->bqhw", mask_embed, pixel_embeddings)
mask_for_next_level = interpolate(mask_logits, size=target_hw, mode="bilinear", align_corners=False)
attn_mask = sigmoid(mask_for_next_level).flatten(2).unsqueeze(1).repeat(1, heads, 1, 1)
attn_mask = (attn_mask.flatten(0, 1) < 0.5).detach()
```

Precompute opportunities:

- Sine position embeddings can be cached per `(B,H,W,mask pattern,dtype)`; all-valid fixed-size batches can cache by shape.
- Swin relative position indices are static for a window size; learned bias tables are weights.
- Deformable reference points are shape/valid-ratio derived and cacheable for all-valid fixed buckets.
- Query content and query position embeddings are constant weights expanded over batch.

## 8. Preprocessing and input packing

Processor contract from `Mask2FormerImageProcessor`:

- Input images are prepared as channel-first tensors.
- Inspected official preprocessors use `size={"height":384,"width":384}`, `size_divisor=32`, bilinear resize, ImageNet mean/std, rescale factor `1/255`, and right/bottom padding if batching requires it.
- Default source class uses aspect-ratio resize `shortest_edge=800,longest_edge=1333`; checkpoint preprocessor configs override this to 384 square in the inspected official repos.
- `pixel_values` is `[B,3,H,W]`; `pixel_mask` is `[B,H,W]` int64 where 1 means valid pixel and 0 means padding.
- Training preprocessing can also emit unbatched `mask_labels` and `class_labels` from segmentation maps; this is not needed for inference.

Recommended first DinoML boundary:

```text
CPU processor owns image decode/resize/rescale/normalize/pad.
DinoML runtime accepts:
  pixel_values: [B,3,H,W] float32/float16 NCHW, contiguous
  pixel_mask:   [B,H,W] int64/bool-compatible, optional for current source parity
Runtime returns:
  class_queries_logits: [B,Q,C+1]
  masks_queries_logits: [B,Q,H/4,W/4]
CPU or separate runtime helper postprocesses to task outputs.
```

Postprocess contracts:

- Semantic: resize mask logits to `(384,384)`, softmax class logits excluding no-object, sigmoid masks, matrix combine to `[B,C,384,384]`, optional resize each image to `target_sizes[i]`, argmax class per pixel.
- Instance: resize mask logits to `(384,384)`, softmax classes excluding no-object, take top `Q` over flattened query/class scores, gather masks, threshold mask logits at `>0`, compute average sigmoid mask score over positive mask pixels, multiply by class score, keep scores `>= threshold`, assign segment ids sequentially. Optional nearest resize, binary maps, or RLE.
- Panoptic: resize to `(384,384)`, sigmoid masks, choose max class label/score including no-object then filter no-object/low-score, weight masks by score, per-pixel argmax over remaining queries, reject low-overlap/tiny segments, optionally fuse configured stuff labels.
- No NMS is present in source postprocessing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Swin patch Conv2d -> patch-flatten GEMM

Source pattern:

```text
Conv2d(3 -> embed_dim, kernel=4, stride=4, padding=0) -> flatten(2).transpose(1,2)
```

Replacement:

```text
Extract non-overlapping 4x4x3 patches in source NCHW row-major order
-> GEMM(weight_flat.T, bias)
-> [B,H/4*W/4,embed_dim]
```

Preconditions:

- `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input spatial dimensions are padded or divisible by patch size; source Swin pads when needed.
- Patch flatten order matches PyTorch OIHW Conv2d over NCHW.

Weight transform:

```python
w = conv.weight.reshape(embed_dim, 3 * 4 * 4)
b = conv.bias
```

Failure cases: non-divisible dynamic shapes without Swin-equivalent padding, changed patch size, or layout pass that changes patch element order.

Parity test sketch: compare patch projection output before the first Swin LayerNorm for random and processor-produced images.

### Rewrite: 1x1 Conv2d projections -> GEMM

Source pattern:

```text
Conv2d(C -> 256, kernel=1) on NCHW feature maps
```

Replacement:

```text
spatial rows [B*H*W,C] -> GEMM_RCR_Bias(C -> 256) -> reshape to NCHW or [B,S,256]
```

Preconditions: kernel 1, stride 1, padding 0, dilation 1, groups 1, source row-major spatial flatten preserved.

Weight transform:

```python
w = conv.weight.reshape(256, C)
```

Failure cases: GroupNorm immediately after projection may make Conv+GN fusion more useful than a bare GEMM in NCHW regions.

### Rewrite: mask-einsum -> batched GEMM

Source pattern:

```text
outputs_mask = einsum("bqc,bchw->bqhw", mask_embed, pixel_embeddings)
```

Replacement:

```text
for each batch:
  [Q,C] x [C,H*W] -> [Q,H*W] -> reshape [Q,H,W]
```

Preconditions:

- `mask_embed [B,Q,256]`, `pixel_embeddings [B,256,H,W]`.
- Preserve NCHW spatial flatten order.
- Batch size may use BMM or looped GEMM; Q is 100 or 200.

Failure cases: NHWC internal mask features require explicit transpose or weight/accessor-aware BMM.

### Rewrite: deformable attention specialized op

Source pattern:

```text
Linear offsets/weights + split value by feature level + grid_sample + weighted sum
```

Replacement:

```text
fused_ms_deform_attn(value_levels, offsets, weights, reference_points, spatial_shapes)
```

Preconditions:

- `num_levels=3`, `num_points=4`, `heads=8`, `head_dim=32` for standard configs.
- Bilinear interpolation, zero padding, `align_corners=False`.
- Feature levels are contiguous NCHW or a declared layout-specific equivalent.

Failure cases: changing `n_levels`, `n_points`, nonstandard valid ratios, or relying on PyTorch `grid_sample` edge behavior without parity tests.

### Rewrite: masked cross-attention to dense attention

Source pattern:

```text
nn.MultiheadAttention(query=hidden+query_pos, key=src+pos, value=src, attn_mask=bool_mask)
```

Replacement:

```text
Q/K/V Linear projections -> scaled dot product attention with boolean/additive mask -> output Linear
```

Preconditions:

- Preserve Q/K position additions and unpositioned V.
- Mask shape resolves to `[B*heads,Q,S_l]` and all-true mask rows have already been zeroed.
- Dropout disabled for inference.

Failure cases: naive packed QKV would incorrectly add position embeddings to V or merge different source tensors.

### Rewrite: guarded NCHW/NHWC backbone island

Source pattern: NCHW public ABI, Swin internally uses sequence and NHWC-like window layouts, then returns NCHW maps.

Replacement:

```text
NCHW input -> layout guard/transpose -> optimized channel-last Swin windows
-> NCHW feature-map boundary for pixel decoder/postprocess
```

Required axis rewrites: Conv2d weights OIHW to provider layout, LayerNorm over last channel stays last-channel in sequence/window form, window partition axes are H/W, and exported feature maps must return to NCHW before pixel decoder unless that entire region is translated.

No-layout-translation guards: public `pixel_values`, `pixel_mask`, processor outputs, postprocess masks, sine embedding cumsum axes, and any source-visible NCHW `interpolate`/`GroupNorm` region not fully translated.

## 10. Kernel fusion candidates

Highest priority:

- Swin backbone kernels: patch embedding, window partition/reverse, shifted-window attention with relative bias, patch merging, and LayerNorm/MLP. Backbone cost dominates and is required before Mask2Former-specific work is meaningful.
- Multi-scale deformable attention. This is the biggest nonstandard runtime blocker and should become an explicit op/provider candidate rather than decomposing into many slow `grid_sample` calls.
- Masked cross-attention and query self-attention for fixed `Q=100/200`, heads=8, hidden=256.
- Mask head BMM/einsum for `[B,Q,256] x [B,256,H/4*W/4]`.
- Interpolate + threshold path for attention-mask generation, because it appears before every decoder layer.

Medium priority:

- Conv2d 1x1/3x3 + GroupNorm + ReLU FPN fusion.
- LayerNorm + residual patterns in pixel decoder and query decoder.
- Sine position embedding/reference-point cache for fixed 384 buckets.
- Semantic postprocess combine `softmax -> sigmoid -> einsum -> argmax` if end-to-end semantic segmentation is the first product target.

Lower priority:

- Instance/panoptic Python postprocess acceleration, including RLE and segment fusion.
- Training point sampling and loss kernels.
- GPU image preprocessing.
- General unsupported-backbone abstraction.

## 11. Runtime staging plan

Stage 1: Processor/config handoff.

- Parse `Mask2FormerConfig` plus Swin `backbone_config`.
- Accept CPU-processor-produced `pixel_values [B,3,H,W]` and optional `pixel_mask`.
- Load weights and verify key tensor shapes for tiny/small first.

Stage 2: Swin backbone parity.

- Reuse or implement Swin backbone ops: patch Conv2d, shifted-window attention, relative bias, patch merging.
- Validate `stage1..stage4` NCHW feature maps for `384x384`.

Stage 3: Pixel decoder parity.

- Implement 1x1 projections, sine embeddings, reference points, and 6 deformable attention layers.
- Add FPN fusion and mask projection.
- Validate `multi_scale_features` and `mask_features`.

Stage 4: Query decoder raw parity.

- Implement learned query content/position expansion, masked cross-attention, self-attention, FFN, layernorm, and mask predictor.
- Validate final class and mask logits for one checkpoint.

Stage 5: Postprocess parity.

- Implement CPU helper first for semantic, then instance, then panoptic.
- Return raw outputs from compiled runtime and apply postprocess outside the graph.

Stage 6: Optimization.

- Add patch/1x1 Conv-to-GEMM rewrites, mask-einsum BMM, deformable-attention provider, and layout islands.

Stage 7: Larger variants.

- Sweep small/base/large, `Q=100/200`, label counts 80/133/150, and Swin window 7/12.

## 12. Parity and validation plan

Random/operator tests:

- Sine position embedding for all-valid and padded masks.
- Swin window partition/reverse and shifted-window masks.
- Swin relative position bias indexing for window 7 and 12.
- Patch merging 2x2 concat order.
- Multi-scale deformable attention against PyTorch `grid_sample` for small shapes.
- Masked-attention mask generation, including all-true row suppression.
- Mask-einsum BMM replacement.

Model slice tests:

- Processor fixture for one image: compare `pixel_values` and `pixel_mask`.
- Backbone feature maps for tiny and small checkpoints.
- Pixel decoder after input projections, after deformable encoder, and after FPN/mask projection.
- Single masked decoder layer, then full decoder.
- Raw `class_queries_logits` and `masks_queries_logits`.
- Semantic postprocess parity for `facebook/mask2former-swin-small-ade-semantic`.
- Instance postprocess parity for `facebook/mask2former-swin-small-coco-instance`.
- Panoptic postprocess parity for `facebook/mask2former-swin-base-coco-panoptic` or large after smaller variants pass.

Suggested tolerances:

- fp32 source-faithful slices: `rtol=1e-4`, `atol=1e-5`; deformable attention and full-model accumulated outputs may need `rtol=2e-4`.
- fp16/bf16 optimized paths: compare against reduced-precision PyTorch or mixed-precision baselines and inspect threshold-sensitive masks near `0.5` separately.

## 13. Performance probes

- CPU processor throughput for resize/rescale/normalize/pad at 384 square and aspect-ratio default.
- Swin backbone latency by variant: tiny, small, base, large.
- Pixel decoder deformable attention time by feature sizes and batch.
- Query decoder time for `Q=100` vs `Q=200`.
- Mask-einsum/BMM time for `[Q,H/4*W/4]`.
- Raw model throughput with postprocess excluded.
- Semantic, instance, and panoptic postprocess time separately, including variable output counts.
- Batch-size sweep and mixed-image padding waste.
- NCHW baseline versus guarded channel-last/Swin-window layout island.
- Deformable attention decomposition versus fused-provider implementation.

No benchmark measurements are included; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training mode, gradients, dropout, layerdrop, and gradient checkpointing.
- Hungarian matching, SciPy dependency, point sampling, focal/dice/cross-entropy losses, and auxiliary loss weighting.
- Processor-side segmentation-map conversion to `mask_labels`/`class_labels`.
- Unsupported non-Swin backbones.
- In-graph RLE conversion and Python-style variable-length records for first raw-logit parity.
- NMS, because Mask2Former source postprocessing does not use it.
- Autoregressive generation, KV cache, beam search, tokenization, and text machinery; not applicable.
- Quantization and multi-GPU tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `Mask2FormerConfig` and nested Swin `backbone_config`.
- [ ] Parse processor config: resize policy, mean/std, rescale, size divisor, and output tensor names.
- [ ] Load Swin backbone, pixel decoder, query decoder, class head, and mask MLP weights.
- [ ] Accept `pixel_values [B,3,H,W]` NCHW and optional `pixel_mask [B,H,W]`.
- [ ] Implement Swin patch embedding, shifted-window attention, relative position bias, patch merging, LayerNorm, GELU MLP.
- [ ] Implement NCHW multi-scale feature-map outputs for `stage1..stage4`.
- [ ] Implement Mask2Former sine position embedding and deformable reference-point generation.
- [ ] Implement pixel decoder 1x1 projections, GroupNorm, 6-layer multi-scale deformable attention, FPN fusion, and mask projection.
- [ ] Implement learned query content and query position expansion.
- [ ] Implement masked cross-attention with previous-mask boolean attention masks.
- [ ] Implement decoder self-attention, FFN, residual/LayerNorm ordering, and mask predictor.
- [ ] Implement class predictor and mask-einsum/BMM head.
- [ ] Return raw `class_queries_logits` and `masks_queries_logits`.
- [ ] Add CPU postprocess helper for semantic segmentation.
- [ ] Add CPU postprocess helper for instance segmentation.
- [ ] Add CPU postprocess helper for panoptic segmentation.
- [ ] Add parity tests for backbone, pixel decoder, query decoder, raw logits/masks, and postprocess outputs.
- [ ] Add variant coverage for tiny/small/base/large, label counts 80/133/150, window sizes 7/12, and `num_queries` 100/200.
- [ ] Add guarded Conv-to-GEMM, mask-einsum BMM, deformable-attention provider, and layout-island rewrites after source-faithful parity.
- [ ] Defer Hungarian/loss/training functionality until explicitly targeted.
