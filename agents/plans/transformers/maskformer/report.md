# MaskFormer Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary target: MaskFormerForInstanceSegmentation for semantic, instance,
  and panoptic image segmentation from preprocessed images.
  Representative configs:
    facebook/maskformer-swin-tiny-ade
    facebook/maskformer-swin-base-ade
    facebook/maskformer-swin-large-coco
    facebook/maskformer-resnet50-ade
    facebook/maskformer-resnet101-cityscapes

Config source:
  https://huggingface.co/facebook/maskformer-swin-tiny-ade/raw/main/config.json
  https://huggingface.co/facebook/maskformer-swin-base-ade/raw/main/config.json
  https://huggingface.co/facebook/maskformer-swin-large-coco/raw/main/config.json
  https://huggingface.co/facebook/maskformer-resnet50-ade/raw/main/config.json
  https://huggingface.co/facebook/maskformer-resnet101-cityscapes/raw/main/config.json
  Matching preprocessor_config.json files were inspected for each checkpoint.

Source files inspected:
  X:/H/transformers/src/transformers/models/maskformer/configuration_maskformer.py
  X:/H/transformers/src/transformers/models/maskformer/configuration_maskformer_swin.py
  X:/H/transformers/src/transformers/models/maskformer/modeling_maskformer.py
  X:/H/transformers/src/transformers/models/maskformer/modeling_maskformer_swin.py
  X:/H/transformers/src/transformers/models/maskformer/modular_maskformer.py
  X:/H/transformers/src/transformers/models/maskformer/image_processing_maskformer.py
  X:/H/transformers/src/transformers/models/maskformer/image_processing_pil_maskformer.py

Any missing files or assumptions:
  No remote code is required for the inspected official checkpoints. Raw
  image_processor_config.json paths returned 404, but the official repos use
  preprocessor_config.json, which was accessible. No gated/401/403 gaps were
  found in the sampled official checkpoints. The generated modeling files are
  the execution source; modular_maskformer.py is useful for future source edits.
  This report assumes inference-only CUDA runtime first, with CPU preprocessing
  and Python/CPU postprocessing allowed outside the compiled graph initially.
```

Snapshots are stored under `agents/plans/transformers/maskformer/_sources/`.

## 2. High-level architecture

MaskFormer is a structured-output vision segmentation model. It composes a Swin
or ResNet backbone, a simple FPN pixel decoder, a DETR-style learned-query
transformer decoder, and class/mask heads.

```text
CPU image preprocessing -> pixel_values/pixel_mask
  -> Swin or ResNet backbone feature maps
  -> FPN pixel decoder -> pixel embeddings
  -> DETR-style learned query decoder over final backbone map
  -> class head + mask embedding head
  -> semantic/instance/panoptic postprocess
```

Stage decomposition:

- CPU/data pipeline: image decode, resize, rescale, normalize, right/bottom pad, emit `pixel_values [B,3,H,W]` and `pixel_mask [B,H,W]`.
- Backbone: Swin or ResNet `AutoBackbone` emits NCHW feature maps for `stage1..stage4`.
- Pixel decoder: FPN starts from the final feature map and fuses higher-resolution lateral maps; final `mask_projection` emits pixel embeddings `[B,mask_feature_size,H/4,W/4]`.
- Query decoder: learned object-query embeddings plus zero query content cross-attend to the final backbone feature map `[B,C,H/32,W/32]` after a 1x1 projection to `d_model=256`.
- Heads: class logits `[B,Q,num_labels+1]`; mask embeddings `[B,Q,256]`; mask logits via `einsum("bqc,bchw->bqhw")`.
- Postprocessing: semantic, instance, and panoptic conversion with task-specific score/mask filtering and resizing.

MaskFormer differs from Mask2Former: this native MaskFormer path does not use
multi-scale deformable attention in its pixel decoder and does not use
Mask2Former's iterative masked cross-attention over multi-scale features.

## 3. Important config dimensions

Common source defaults:

| Field | Value | Source |
| --- | ---: | --- |
| `model_type` | `maskformer` | config |
| primary task | semantic/instance/panoptic segmentation | source |
| `fpn_feature_size` | 256 | `MaskFormerConfig` default/configs |
| `mask_feature_size` | 256 | `MaskFormerConfig` default/configs |
| decoder `d_model` | 256 | nested DETR config |
| decoder layers | 6 | nested DETR config |
| decoder attention heads | 8 | nested DETR config |
| decoder head dim | 32 | inferred from `256 / 8` |
| decoder FFN dim | 2048 | nested DETR config |
| decoder activation | ReLU | nested DETR config |
| `num_queries` | 100 | nested DETR config |
| class logits | `num_labels + 1` | source; extra null/no-object class |
| Swin patch size | 4 | Swin configs |
| Swin window size | 7 or 12 | checkpoint-dependent |
| cache support | none | non-autoregressive encoder/decoder |

Representative checkpoint sweep:

| Checkpoint | Labels | Backbone | Backbone dims | Depths | Heads | Image/processor size | Notes |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `facebook/maskformer-swin-tiny-ade` | 150 | Swin | embed 96, channels 96/192/384/768 | 2/2/6/2 | 3/6/12/24 | 512 | small/debug-like official ADE variant |
| `facebook/maskformer-swin-base-ade` | 150 | Swin | embed 128, channels 128/256/512/1024 | 2/2/18/2 | 4/8/16/32 | 640 | common ADE semantic checkpoint |
| `facebook/maskformer-swin-large-coco` | 133 | Swin | embed 192, channels 192/384/768/1536 | 2/2/18/2 | 6/12/24/48 | 800 | larger COCO panoptic-style label space |
| `facebook/maskformer-resnet50-ade` | 150 | ResNet | 256/512/1024/2048 | 3/4/6/3 | n/a | shortest 800, longest 1333, divisor 32 | ResNet backbone path |
| `facebook/maskformer-resnet101-cityscapes` | 19 | ResNet | 256/512/1024/2048 | 3/4/23/3 | n/a | shortest 800, longest 1333, divisor 32 | Cityscapes ignore index 65535 |

Effective defaults worth preserving when checkpoint configs omit fields:

- `MaskFormerConfig` default Swin backbone is base-like: `depths=[2,2,18,2]`, `embed_dim=128`, `num_heads=[4,8,16,32]`, `image_size=384`, `window_size=12`, `drop_path_rate=0.3`, and `out_features=["stage1","stage2","stage3","stage4"]`.
- `MaskFormerConfig` accepts `backbone_config.model_type` `resnet` or `swin`; unsupported backbones warn.
- `decoder_config` defaults to `MaskFormerDetrConfig`, whose `encoder_layers` fields mostly belong to DETR compatibility; MaskFormer's main query decoder uses `decoder_layers`, `decoder_attention_heads`, `decoder_ffn_dim`, `d_model`, and `num_queries`.

## 3a. Family variation traps

- The public runtime ABI is NCHW. `pixel_values`, backbone feature maps, FPN Conv2d/GroupNorm, pixel embeddings, and mask logits use `[B,C,H,W]` or `[B,Q,H,W]`.
- Swin internally alternates sequence `[B,H*W,C]` and NHWC-like `[B,H,W,C]` for window partitioning, but `MaskFormerSwinBackbone` materializes NCHW feature maps. Treat NHWC as a guarded layout/fusion island, not a default translation.
- Swin official configs may omit `out_features`; `MaskFormerPixelLevelModule` forcibly sets Swin `out_features=["stage1","stage2","stage3","stage4"]` for compatibility.
- ResNet-backed checkpoints are native `maskformer` configs. Their backbone operator surface should compose the separately audited ResNet family, but this report owns the consumed feature contract: 4 NCHW maps with channels 256/512/1024/2048.
- `pixel_mask` is produced by the processor and defaulted in `MaskFormerModel.forward`, but the current source does not pass it into the backbone, FPN, sine position embedding, or decoder. It remains a public input and should be retained for API parity/future-proofing.
- The transformer decoder cross-attends only to the final backbone feature map, not to FPN pixel embeddings.
- Query inputs are zero content tensors plus learned positional query embeddings. Q/K receive position additions; V does not.
- `num_labels` changes by task; class head always includes one extra null class. ADE has 150 labels, COCO sampled checkpoint has 133, Cityscapes has 19.
- Processor configs differ: Swin checkpoints use scalar resize sizes 512/640/800 in sampled repos; ResNet checkpoints use aspect-ratio `shortest_edge=800,longest_edge=1333` and `size_divisor=32`.
- Source contains training/loss/Hungarian code and DETR-compatible helper classes; first inference integration can ignore those unless training parity is explicitly targeted.
- Axis-sensitive layout traps:
  - FPN lateral fusion concatenates/fuses along channel axis 1 and upsamples to `left.shape[-2:]`.
  - GroupNorm uses NCHW channel axis semantics.
  - Sine position embedding cumsums on `[B,H,W]` mask axes, then returns `[B,C,H,W]`.
  - `einsum("bqc,bchw->bqhw")` assumes channel dimension is axis 1 in pixel embeddings.
  - Postprocess target sizes are `(height,width)`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input `pixel_values [B,3,H,W]`; optional `pixel_mask [B,H,W]`.
- NCHW feature maps from Swin or ResNet `AutoBackbone`.
- `view`, `reshape`, `flatten`, `transpose`, `permute`, `contiguous`, `repeat`, `unsqueeze`, `stack`, `cat`.
- Right/bottom padding in processor and Swin patch/window paths.
- Nearest interpolation in FPN; bilinear interpolation in postprocess and sine/mask-related resizing paths.
- Boolean/int mask creation, comparisons, `masked_fill`, top-k, argmax, gather/indexing for postprocess.

Neural network primitives:

- Swin path: Conv2d patch embedding, LayerNorm, window partition/reverse, shifted-window MHA, learned relative-position bias gather, GELU MLP, residual adds, patch merging `LayerNorm(4C) -> Linear(4C -> 2C)`.
- ResNet path: Conv2d, BatchNorm/activation/maxpool/residual bottleneck blocks as consumed from the ResNet family.
- FPN: `Conv2d(C_last -> 256, 3x3, bias=False) -> GroupNorm(32) -> ReLU`; lateral `Conv2d(C_lateral -> 256, 1x1, bias=False) -> GroupNorm(32)`; nearest upsample; residual add; 3x3 Conv/GN/ReLU; final `Conv2d(256 -> 256, 3x3, padding=1)`.
- Transformer decoder: learned embeddings, Linear Q/K/V/O with bias, LayerNorm, ReLU FFN `Linear(256 -> 2048) -> Linear(2048 -> 256)`, residual adds.
- Heads: `Linear(256 -> num_labels+1)` with bias; mask MLP `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 256)`; BMM/einsum for masks.

Attention primitives:

- Swin local shifted-window MHA with relative position bias for Swin backbones.
- DETR-style noncausal self-attention over object queries.
- DETR-style rectangular cross-attention from `Q=100` object queries to `S=H/32*W/32` final image tokens.
- No causal masks, no autoregressive KV cache, no packed/varlen metadata.

Position/custom math:

- Swin relative position index/table lookup.
- 2D sine/cosine position embedding for final image feature map.
- Learned query-position embedding table `[num_queries,256]`.

Preprocessing/postprocessing-coupled ops:

- Processor resize/rescale/normalize/pad; segmentation-map conversion only for training.
- Semantic postprocess: class softmax excluding null, mask sigmoid, class-mask einsum, optional bilinear resize, argmax.
- Instance postprocess: flattened class/query top-k, mask threshold, average mask score, score threshold, optional nearest resize/RLE/binary maps.
- Panoptic postprocess: no-object/score filtering, score-weighted masks, per-pixel argmax, overlap/area filtering, optional stuff-label fusion.

Distributed/tensor-parallel ops: none required for first integration.

## 5. Layer/block breakdown

Processor output:

```text
pixel_values: [B,3,Hpad,Wpad] float, NCHW, ImageNet normalized
pixel_mask:   [B,Hpad,Wpad] int64, 1 valid / 0 padded; retained for API parity
```

Swin backbone block, composed from `modeling_maskformer_swin.py`:

```text
x = Conv2d(3 -> embed_dim, kernel=4, stride=4)(pixel_values)
x = flatten spatial -> [B,H/4*W/4,embed_dim]
for stage i:
  repeat depth_i:
    y = LayerNorm(x)
    y = reshape [B,H,W,C], pad to window multiple
    y = optional cyclic shift
    y = window_partition -> [B*num_windows, window^2, C]
    y = MHA(q,k,v) + relative_position_bias + optional shifted mask
    y = window_reverse, undo shift, crop padding
    x = x + y
    x = x + Linear(GELU(Linear(LayerNorm(x))))
  if not final stage:
    x = patch_merge_2x2(x): concat rows/cols -> LayerNorm(4C) -> Linear(4C -> 2C, bias=False)
return feature maps stage1..stage4 as NCHW
```

For a `640x640` Swin-base input, approximate feature maps are:

```text
stage1: [B,128,160,160]
stage2: [B,256, 80, 80]
stage3: [B,512, 40, 40]
stage4: [B,1024,20, 20]
```

FPN pixel decoder:

```text
features = [stage1, stage2, stage3, stage4]
out = Conv3x3/GN/ReLU(stage4 -> 256)
for left in [stage3, stage2, stage1]:
  left = Conv1x1/GN(left -> 256)
  out = nearest_interpolate(out, size=left.HW) + left
  out = Conv3x3/GN/ReLU(out -> 256)
pixel_embeddings = Conv3x3(256 -> mask_feature_size=256)(out)
```

Transformer module:

```text
image_features = stage4                         # [B,C4,H32,W32]
image_features = Conv1x1(C4 -> 256)             # skipped only if C4 == 256
spatial_pos = sine_position_embedding(image_features.shape)  # [B,256,H32,W32]
queries_pos = Embedding(num_queries,256).repeat(B)           # [B,Q,256]
query_content = zeros_like(queries_pos)                      # [B,Q,256]

image_tokens = image_features.view(B,256,H32*W32).permute(0,2,1)
spatial_pos_tokens = spatial_pos.view(B,256,H32*W32).permute(0,2,1)
```

DETR decoder layer, repeated 6 times:

```text
residual = x
self_qk = x + queries_pos
x = MHA(q=self_qk, k=self_qk, v=x) -> Linear(256 -> 256)
x = LayerNorm(residual + dropout(x))

residual = x
cross_q = x + queries_pos
cross_k = image_tokens + spatial_pos_tokens
x = MHA(q=cross_q, k=cross_k, v=image_tokens) -> Linear(256 -> 256)
x = LayerNorm(residual + dropout(x))

residual = x
x = Linear(256 -> 2048) -> ReLU -> dropout -> Linear(2048 -> 256) -> dropout
x = LayerNorm(residual + x)
```

Final heads:

```text
decoder_out = LayerNorm(last_decoder_state)      # [B,Q,256]
class_logits = Linear(256 -> num_labels+1)(decoder_out)
mask_embed = MLP(256 -> 256 -> 256 -> 256)(decoder_out)
mask_logits = einsum("bqc,bchw->bqhw", mask_embed, pixel_embeddings)
```

Projection biases:

- Swin Q/K/V/O Linear projections are biased when `qkv_bias=True` in checkpoint configs.
- DETR decoder Q/K/V/O Linear projections are biased by source default.
- FPN Conv2d blocks use `bias=False` except `mask_projection`, which uses default bias.
- Class predictor and mask MLP Linear layers use default bias.

## 6. Attention requirements

Swin backbone attention:

- Noncausal self-attention inside fixed windows.
- MHA, not MQA/GQA.
- Heads vary by stage and checkpoint: tiny `3/6/12/24`, base `4/8/16/32`, large `6/12/24/48`.
- Head dim equals stage channels divided by heads: 32 for sampled Swin checkpoints.
- Optional shifted-window additive mask uses `0` and `-100.0`.
- Learned relative position bias is added before softmax.
- No KV cache and no generation decode.

MaskFormer DETR decoder self-attention:

- Noncausal self-attention over `Q=100` learned object queries.
- `num_heads=8`, `head_dim=32`, q/k/v width 256, output width 256.
- Q and K inputs add learned query-position embeddings; V uses raw hidden states.
- No query padding mask is used by the main MaskFormer path.

MaskFormer DETR decoder cross-attention:

- Noncausal rectangular cross-attention.
- Query length `Q=100`; key/value length `S=H32*W32`.
- Q source: decoder hidden state plus learned query-position embeddings.
- K source: final backbone feature tokens plus 2D sine position embeddings.
- V source: final backbone feature tokens without position embeddings.
- q/k/v width 256, 8 heads, head_dim 32.
- `encoder_attention_mask` is wired in the generic decoder, but MaskFormer calls it with `None`; processor `pixel_mask` is not consumed in this source path.

Backend compatibility:

- `MaskFormerDetrPreTrainedModel` advertises SDPA/Flash/Flex support through the generic attention interface for the custom DETR decoder attention classes.
- Swin attention in this local source is explicit matmul/softmax code, not a cache-oriented generation attention.
- Eager fallback is acceptable for parity but likely slow for large Swin images; window attention and final-map cross-attention are optimization targets.

## 7. Position encoding and custom math

2D sine position embedding:

```python
def maskformer_sine_position(shape, dtype, mask=None, num_feats=128):
    # shape: [B,C,H,W]; mask: [B,H,W] bool where True means masked
    if mask is None:
        mask = zeros([B, H, W], dtype=bool)
    not_mask = (~mask).to(dtype)
    y = cumsum(not_mask, axis=1)
    x = cumsum(not_mask, axis=2)
    y = y / (y[:, -1:, :] + 1e-6) * (2 * pi)
    x = x / (x[:, :, -1:] + 1e-6) * (2 * pi)
    dim = 10000 ** (2 * floor(arange(num_feats) / 2) / num_feats)
    pos_x = stack([sin(x[..., 0::2] / dim[0::2]), cos(x[..., 1::2] / dim[1::2])]).flatten()
    pos_y = stack([sin(y[..., 0::2] / dim[0::2]), cos(y[..., 1::2] / dim[1::2])]).flatten()
    return concat([pos_y, pos_x], axis=-1).permute(0, 3, 1, 2)
```

Swin window math:

```python
def relative_position_index(window_h, window_w):
    coords = meshgrid(arange(window_h), arange(window_w))
    rel = flatten(coords)[:, :, None] - flatten(coords)[:, None, :]
    rel[..., 0] += window_h - 1
    rel[..., 1] += window_w - 1
    rel[..., 0] *= 2 * window_w - 1
    return rel.sum(-1)
```

Mask combination:

```python
def mask_logits(mask_embed, pixel_embeddings):
    # [B,Q,C] x [B,C,H,W] -> [B,Q,H,W]
    return einsum("bqc,bchw->bqhw", mask_embed, pixel_embeddings)
```

Precompute/cache candidates:

- Swin relative-position indices are static per window size; bias tables are weights.
- All-valid sine embeddings can be cached per `(B,H,W,dtype)` because current source does not pass processor masks into the transformer module.
- Learned query embeddings are weights expanded along batch.

## 8. Preprocessing and input packing

Processor behavior from `MaskFormerImageProcessor` and sampled configs:

- Converts images to channel-first tensor form.
- Resize:
  - Swin sampled checkpoints use scalar `size` values: 512, 640, or 800.
  - ResNet sampled checkpoints use `{"shortest_edge":800,"longest_edge":1333}` and `size_divisor=32`.
  - Source class default is `shortest_edge=800,longest_edge=1333`, `size_divisor=32`.
- Rescale by `1/255`, normalize with ImageNet mean/std.
- Pad bottom/right to a common batch size; emits `pixel_mask [B,H,W]` where 1 means valid and 0 means padding.
- For training labels only, segmentation maps are resized with nearest-exact interpolation, converted to per-instance binary `mask_labels` and `class_labels`, with optional label reduction and ignore-index handling.

Recommended first DinoML boundary:

```text
CPU/data pipeline:
  images -> pixel_values [B,3,H,W], pixel_mask [B,H,W]

DinoML runtime:
  pixel_values -> class_queries_logits [B,Q,num_labels+1]
               -> masks_queries_logits [B,Q,Hmask,Wmask]

CPU/runtime helper:
  postprocess raw logits to semantic/instance/panoptic records.
```

Postprocessing contracts:

- Semantic: `softmax(class_logits)[..., :-1]`, `sigmoid(mask_logits)`, `einsum("bqc,bqhw->bchw")`, optional bilinear resize to `target_sizes[i]`, `argmax(dim=0)`.
- Instance: softmax classes excluding null, flatten query/class scores, `topk(num_queries)`, recover query ids by floor-dividing by `num_classes`, threshold masks with `mask_pred > 0`, compute average sigmoid score over positive pixels, multiply by class score, keep scores above threshold. Optional nearest resize, binary maps, or COCO RLE.
- Panoptic: sigmoid masks, max class score/label, remove low-score/no-object queries, score-weight masks, choose per-pixel segment by argmax, filter by overlap area, optionally fuse labels in `label_ids_to_fuse`.
- No NMS is implemented by source postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Swin patch Conv2d -> patch GEMM

Source pattern:

```text
Conv2d(3 -> embed_dim, kernel=patch_size, stride=patch_size)
-> flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW patches in Conv2d order) -> MatMul(weight_flat.T) -> BiasAdd -> [B,N,embed_dim]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Match source padding for non-divisible height/width before extraction.
- Preserve NCHW/OIHW patch flatten order.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases: dynamic shapes without Swin-equivalent padding, layout pass that changes patch element order, or nonstandard patch configs.

### Rewrite: FPN 1x1 Conv2d -> GEMM

Source pattern:

```text
Conv2d(C -> 256, kernel=1, bias=False) -> GroupNorm(32)
```

Replacement:

```text
[B,H,W,C] or [B*H*W,C] local island -> MatMul(C -> 256) -> GroupNorm-compatible layout boundary
```

Preconditions:

- Kernel 1, stride 1, padding 0, dilation 1, groups 1.
- If translated to NHWC, all consumers in the island must be controlled and GroupNorm axis semantics must be rewritten.
- Public FPN boundary remains source-faithful NCHW unless the full Conv/GN/ReLU/interpolate/add region is translated.

Failure cases: isolated Conv rewrite that leaves GroupNorm expecting NCHW channel axis.

### Rewrite: mask einsum -> batched GEMM

Source pattern:

```text
einsum("bqc,bchw->bqhw")
```

Replacement:

```text
for each batch: [Q,256] x [256,H*W] -> [Q,H*W] -> [Q,H,W]
```

Preconditions:

- `mask_embed` and `pixel_embeddings` share channel width `mask_feature_size=256`.
- Spatial flatten order is NCHW row-major.

Failure cases: NHWC pixel embeddings without explicit transpose/accessor-aware GEMM.

### Rewrite: DETR decoder attention canonicalization

Source pattern:

```text
q,k = Linear(hidden + position)
v = Linear(hidden_or_image_tokens)
attention(q,k,v)
```

Replacement:

```text
PositionAdd(Q/K only) -> QKV projections -> noncausal dense attention -> output projection
```

Preconditions:

- Do not add position embeddings to V.
- Self-attention Q/K positions come from learned query embeddings.
- Cross-attention K positions come from sine image positions; Q positions come from learned query embeddings.
- Dropout is disabled for inference.

Failure cases: fused QKV projection that assumes a single input tensor for q/k/v in cross-attention.

### Rewrite: guarded NCHW/NHWC layout islands

Candidate islands:

- Swin window attention internally already uses last-channel `[B,H,W,C]` and `[B,N,C]` forms; a channel-last provider can avoid some permutes inside the Swin island.
- FPN Conv/GN/ReLU blocks may become channel-last kernels if the entire local region and its interpolation/add consumers are translated together.

Required guards:

- Public `pixel_values`, `pixel_mask`, backbone feature-map ABI, `GroupNorm`, FPN concat/add axes, sine cumsum axes, `einsum("bqc,bchw->bqhw")`, and postprocess masks must be protected by a conceptual `no_layout_translation()` unless all consumers are rewritten together.

## 10. Kernel fusion candidates

Highest priority:

- Swin backbone kernels: patch embedding, window partition/reverse, shifted-window attention with relative bias, patch merging, LayerNorm/GELU MLP. Swin dominates compute for official Swin checkpoints.
- FPN Conv2d + GroupNorm + ReLU blocks. These are repeated across feature levels and are NCHW-sensitive.
- DETR query self-attention and final-map cross-attention for fixed `Q=100`, `hidden=256`, `heads=8`.
- Mask head BMM/einsum `[B,Q,256] x [B,256,H/4*W/4]`.

Medium priority:

- Sine position embedding cache for all-valid fixed-size buckets.
- Class softmax + mask sigmoid + semantic combine for semantic-only serving.
- ResNet backbone channel-last Conv/BN/activation fusions for ResNet checkpoints.
- Nearest upsample + add + Conv/GN/ReLU FPN fusion.

Lower priority:

- Instance/panoptic Python postprocess acceleration and RLE conversion.
- Training losses, Hungarian matching, and segmentation-map label conversion.
- General unsupported-backbone lowering.

## 11. Runtime staging plan

Stage 1: Config and processor handoff.

- Parse `MaskFormerConfig`, nested `backbone_config`, nested DETR `decoder_config`, and preprocessor config.
- Admit `swin` and `resnet` backbones separately; start with Swin tiny/base.
- Return raw logits/masks before postprocess.

Stage 2: Backbone parity.

- Compose existing Swin/ResNet coverage.
- Validate `stage1..stage4` NCHW feature maps and channel widths.

Stage 3: FPN pixel decoder parity.

- Implement FPN Conv/GN/ReLU, nearest upsample, lateral adds, and mask projection.
- Validate pixel embeddings.

Stage 4: Query decoder parity.

- Implement 1x1 projection of final backbone map, sine positions, learned queries, 6 DETR decoder layers, final LayerNorm.
- Validate query hidden states.

Stage 5: Heads and raw outputs.

- Implement class predictor, 3-layer mask MLP, and mask BMM/einsum.
- Validate `class_queries_logits` and `masks_queries_logits`.

Stage 6: Postprocess.

- Add semantic postprocess first, then instance and panoptic helpers.
- Keep variable-length records outside compiled graph initially.

Stage 7: Optimization.

- Add guarded patch-Conv GEMM, 1x1 Conv GEMM, FPN fusions, attention kernels, and layout islands after source-faithful parity.

## 12. Parity and validation plan

Random/operator tests:

- Swin patch embedding with padding and patch-flatten GEMM replacement.
- Swin window partition/reverse and relative position bias for window 7 and 12.
- Sine position embedding for all-valid masks and synthetic padded masks.
- FPN nearest upsample/add/Conv/GN/ReLU block.
- DETR self-attention and cross-attention with Q/K-only position additions.
- Mask MLP plus `einsum("bqc,bchw->bqhw")` BMM replacement.
- Semantic/instance/panoptic postprocess on synthetic logits.

Model slice tests:

- Processor snapshot parity for one Swin and one ResNet checkpoint.
- Backbone feature maps for `facebook/maskformer-swin-tiny-ade` and `facebook/maskformer-resnet50-ade`.
- Pixel embeddings after FPN.
- Single decoder layer, full 6-layer decoder, final class logits, final mask logits.
- End-to-end semantic map parity for `facebook/maskformer-swin-base-ade`.
- Panoptic/instance-style output parity for `facebook/maskformer-swin-large-coco`.
- Cityscapes ignore-index/postprocess smoke parity for `facebook/maskformer-resnet101-cityscapes`.

Suggested tolerances:

- fp32 source-faithful slices: `rtol=1e-4`, `atol=1e-5`.
- Full fp32 logits after Swin/ResNet: `rtol=2e-4`, `atol=2e-5` if backend attention/conv ordering differs slightly.
- fp16/bf16 optimized paths: compare to matching reduced-precision PyTorch baselines; test threshold-sensitive masks around `0.5` separately.

## 13. Performance probes

- CPU preprocessing throughput by resize policy: Swin scalar size versus ResNet aspect-ratio plus size divisor.
- Swin backbone latency by tiny/base/large and image size 512/640/800.
- ResNet50/101 backbone latency at aspect-ratio-resized inputs.
- FPN-only throughput and memory traffic at stride 4/8/16/32 feature sizes.
- Decoder-only throughput for `Q=100`, `S=H/32*W/32`, 6 layers.
- Mask BMM/einsum throughput at output stride 4.
- Raw model throughput with postprocess excluded.
- Semantic/instance/panoptic postprocess latency and variable output count.
- NCHW baseline versus guarded channel-last Swin and FPN islands.
- Batch-size sweep and padding waste for mixed image sizes.

No benchmark measurements are included; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training mode, gradients, dropout/layerdrop behavior, gradient checkpointing.
- Hungarian matching, pairwise dice/focal losses, cross-entropy loss, and auxiliary loss supervision.
- Processor-side segmentation-map conversion to `mask_labels`/`class_labels`.
- Unsupported backbones outside official Swin/ResNet configs.
- In-graph RLE conversion and variable-length instance/panoptic records for first raw-logit parity.
- NMS, because source postprocessing does not use it.
- Autoregressive generation, KV cache, beam search, tokenization, and text paths; not applicable.
- Quantization and multi-GPU tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `MaskFormerConfig`, nested `backbone_config`, and nested DETR `decoder_config`.
- [ ] Parse processor config: resize policy, rescale, normalize, pad, size divisor, ignore index.
- [ ] Load Swin/ResNet backbone, FPN, query decoder, class head, and mask MLP weights.
- [ ] Accept `pixel_values [B,3,H,W]` NCHW and optional `pixel_mask [B,H,W]`.
- [ ] Implement/compose Swin backbone feature-map output for Swin checkpoints.
- [ ] Implement/compose ResNet feature-map output for ResNet checkpoints.
- [ ] Implement FPN Conv/GN/ReLU, nearest upsample, lateral add, and mask projection.
- [ ] Implement MaskFormer sine position embedding.
- [ ] Implement learned query expansion and zero query-content initialization.
- [ ] Implement DETR decoder self-attention, cross-attention, FFN, residuals, and LayerNorm ordering.
- [ ] Implement class predictor and mask MLP.
- [ ] Lower mask `einsum("bqc,bchw->bqhw")` to BMM.
- [ ] Return raw `class_queries_logits` and `masks_queries_logits`.
- [ ] Add semantic segmentation postprocess helper.
- [ ] Add instance segmentation postprocess helper.
- [ ] Add panoptic segmentation postprocess helper.
- [ ] Add parity tests for processor, backbone, FPN, decoder, heads, and postprocess.
- [ ] Add checkpoint coverage for Swin tiny/base/large and ResNet50/101.
- [ ] Add guarded patch-Conv GEMM, 1x1 Conv GEMM, FPN fusion, attention fusion, and layout-island rewrites.
- [ ] Defer loss/Hungarian/training features until explicitly targeted.
