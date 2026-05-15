# LW-DETR Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary inference target: AnnaZhang/lwdetr_small_60e_coco,
  LwDetrForObjectDetection.
  Representative sweep:
    AnnaZhang/lwdetr_tiny_60e_coco
    AnnaZhang/lwdetr_small_60e_coco
    AnnaZhang/lwdetr_medium_60e_coco
    AnnaZhang/lwdetr_large_60e_coco
    AnnaZhang/lwdetr_xlarge_60e_coco
    AnnaZhang/lwdetr_small_30e_objects365

Config source:
  https://huggingface.co/AnnaZhang/lwdetr_small_60e_coco/raw/main/config.json
  https://huggingface.co/AnnaZhang/lwdetr_small_60e_coco/raw/main/preprocessor_config.json
  Same raw paths for the other AnnaZhang sweep checkpoints above.
  Local snapshots are in agents/plans/transformers/lw_detr/_sources/.

Source files inspected:
  transformers/src/transformers/models/lw_detr/configuration_lw_detr.py
  transformers/src/transformers/models/lw_detr/modeling_lw_detr.py
  transformers/src/transformers/models/lw_detr/modular_lw_detr.py
  transformers/src/transformers/models/lw_detr/convert_lw_detr_to_hf.py
  transformers/src/transformers/models/deformable_detr/image_processing_deformable_detr.py
  transformers/src/transformers/loss/loss_lw_detr.py
  transformers/src/transformers/models/auto/image_processing_auto.py
  transformers/src/transformers/models/auto/modeling_auto.py

Any missing files or assumptions:
  modeling_lw_detr.py and configuration_lw_detr.py are generated from
  modular_lw_detr.py; future upstream source edits should inspect the modular
  file first. There is no family-specific image processor: AutoImageProcessor
  maps lw_detr to DeformableDetrImageProcessor. All representative configs were
  accessible; no gated/401 gaps were encountered. This report targets CUDA
  inference for object detection from processor-produced image tensors first.
  Training losses, Hungarian matching, auxiliary loss materialization, and
  third-party custom kernels are documented but deferred.
```

## 2. High-level architecture

LW-DETR is a real-time object detector with a plain ViT-style image backbone,
a compact convolutional multi-scale projector, and a shallow fixed-query DETR
decoder. It is not an autoregressive model and has no KV cache or generation
loop.

```text
CPU image preprocessing
  -> pixel_values [B,3,H,W], pixel_mask [B,H,W]
  -> LW-DETR ViT patch backbone with window-major token organization
  -> selected same-resolution backbone feature maps
  -> scale projector: optional up/downsample + C2F conv block + channel LayerNorm
  -> flatten projected feature maps and masks
  -> encoder proposal generation + top-k mixed query selection
  -> decoder: query self-attention + multiscale deformable cross-attention
  -> class logits [B,Q,C] and normalized cxcywh boxes [B,Q,4]
  -> DeformableDetrImageProcessor postprocess to scores/labels/absolute xyxy boxes
```

Stage decomposition:

- CPU/data pipeline: image decode, resize to 640x640, rescale, ImageNet normalize, pad, and create `pixel_mask`.
- Backbone: NCHW `Conv2d` patch embedding with stride 16, optional absolute position table interpolation, then window-major ViT blocks.
- Projector: consumes selected backbone stages, resamples each selected stage per output scale, concatenates channels, runs C2F/RepVGG-like conv blocks, and emits one or two NCHW feature maps with `d_model` channels.
- Proposal/query setup: flatten all feature levels to `[B,S,d_model]`, compute valid ratios and grid proposals, score every feature token, select top-k proposals, and combine them with learned reference/query tables.
- Decoder: 3 layers in inspected configs, noncausal self-attention over object queries, deformable cross-attention over multi-level feature memory, and MLP.
- Detection heads: final class head and box MLP. HF postprocess is required for end-to-end parity and intentionally has no NMS.

Independently stageable units: processor handoff, ViT patch/window backbone, projector scale path, proposal/top-k query path, one decoder layer with deformable attention, final heads, and postprocess.

## 3. Important config dimensions

Worked example: `AnnaZhang/lwdetr_small_60e_coco`.

| Field | Value | Source |
| --- | ---: | --- |
| primary task | object detection | source/config |
| architecture | `LwDetrForObjectDetection` | config.json |
| `model_type` | `lw_detr` | config.json |
| labels | 91 COCO id labels | config `id2label` count |
| processor | `DeformableDetrImageProcessor` | preprocessor_config / auto map |
| processor size | 640x640 | preprocessor_config |
| processor normalize | ImageNet mean/std | preprocessor_config |
| `d_model` | 256 | config.json |
| `num_queries` | 300 | config.json |
| `decoder_layers` | 3 | config.json |
| decoder self-attention heads | 8 | config.json |
| decoder cross-attention heads | 16 | config.json |
| self-attention head dim | 32 | inferred `d_model / self_heads` |
| cross-attention head dim | 16 | inferred `d_model / cross_heads` |
| `decoder_n_points` | 2 | config.json |
| decoder FFN | 2048 | config.json |
| `projector_scale_factors` | `[1.0]` | config.json |
| `group_detr` | 13 | config.json; training only uses all groups |
| backbone type | `lw_detr_vit` | nested config |
| backbone hidden size | 192 | nested config |
| backbone layers / heads | 10 / 12 | nested config |
| patch size | 16 | nested config |
| backbone image size | 1024 | nested config; processor still emits 640 |
| absolute position embeddings | true | nested config |
| `num_windows` | 16, 4x4 windows | nested config |
| window attention blocks | `[0,1,3,6,7,9]` | nested config |
| selected backbone stages | `[3,5,6,10]` | nested config |
| cache/generation | not applicable | source |

Representative checkpoint sweep:

| Model id | Labels | Backbone width/layers | Selected stages | Projector scales | `d_model` | Queries | Decoder heads | Points |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| `AnnaZhang/lwdetr_tiny_60e_coco` | 91 | 192 / 6 | 2,4,6 | `[1.0]` | 256 | 100 | self 8, cross 16 | 2 |
| `AnnaZhang/lwdetr_small_60e_coco` | 91 | 192 / 10 | 3,5,6,10 | `[1.0]` | 256 | 300 | self 8, cross 16 | 2 |
| `AnnaZhang/lwdetr_medium_60e_coco` | 91 | 384 / 10 | 3,5,6,10 | `[1.0]` | 256 | 300 | self 8, cross 16 | 2 |
| `AnnaZhang/lwdetr_large_60e_coco` | 91 | 384 / 10 | 3,5,6,10 | `[2.0,0.5]` | 384 | 300 | self 12, cross 24 | 4 |
| `AnnaZhang/lwdetr_xlarge_60e_coco` | 91 | 768 / 10 | 3,5,6,10 | `[2.0,0.5]` | 384 | 300 | self 12, cross 24 | 4 |
| `AnnaZhang/lwdetr_small_30e_objects365` | 366 | 192 / 10 | 3,5,6,10 | `[1.0]` | 256 | 300 | self 8, cross 16 | 2 |

Source defaults that may be omitted or easy to miss: `attention_bias=True`,
`decoder_activation_function="relu"`, `activation_function="silu"` for projector
blocks, `hidden_expansion=0.5`, `c2f_num_blocks=3`, `disable_custom_kernels=True`,
`auxiliary_loss=True`, and `backbone_config` defaults to a small LW-DETR ViT if
not supplied.

## 3a. Family variation traps

- Source tensors are NCHW at the public ABI, patch embedding, projector, mask interpolation, and flatten boundary. Treat NHWC/channel-last as a guarded optimization only.
- The ViT backbone is not a standard flat sequence ViT at runtime. Patch tokens are reorganized into `num_windows=16` window-major batches. Window blocks attend within each window; non-window blocks reshape back to global image-token attention across all windows.
- Backbone selected stages all have the same patch-grid resolution before the projector. For 640x640 and patch 16, that is 40x40. Large/xlarge projector scale `2.0` creates 80x80 features and scale `0.5` creates 20x20 features.
- Absolute position embeddings are stored for pretrain image size 224 and include a dropped cls-token slot; source bicubic-interpolates the table to the current patch grid.
- Backbone attention uses Q and V bias if `qkv_bias=True`, but K is explicitly bias-free. Do not pack QKV as a homogeneous biased projection unless weights are transformed carefully.
- Projector scale behavior changes operator structure: scale `1.0` has no sampling layers, `2.0` uses `ConvTranspose2d` and may use a pre-upsample `1x1 ConvNorm` when channels exceed 512, and `0.5` uses stride-2 `3x3 ConvNorm`.
- `d_model` changes from 256 to 384 for large/xlarge, while medium keeps 384 backbone width but projects to 256.
- Decoder self-attention and deformable cross-attention use different head counts. Cross-attention head dim is `d_model / decoder_cross_attention_heads`.
- In inference, only the first `num_queries` learned query/reference embeddings are used. Training uses `num_queries * group_detr` and special batch/sequence reshaping in self-attention.
- Mixed query selection is not ordinary learned DETR queries: top-k encoder feature proposals refine learned reference points before the decoder.
- `disable_custom_kernels=True` appears in inspected configs, but source still wraps deformable attention with `use_kernel_forward_from_hub("MultiScaleDeformableAttention")`. DinoML should treat the custom kernel as optional and preserve the eager `grid_sample` semantics first.
- Class logits use independent sigmoid scores. There is no no-object logit in the HF postprocess for LW-DETR.
- The current source does not iteratively update decoder reference points inside
  each decoder layer. Final boxes refine the initial mixed-query reference
  points with the final head delta.
- `auxiliary_loss=True` affects training loss paths when labels are present, but inference raw outputs can ignore auxiliary materialization unless debugging requires intermediate states.
- Axis-sensitive layout traps:
  - `pixel_values` is `[B,C,H,W]`; channel axis is 1.
  - `pixel_mask` is `[B,H,W]`; mask downsampling uses feature-map `[-2:]`.
  - Backbone window packing reshapes `[B,H,W,C]` into `[B*4*4,10*10,C]` for 640 inputs; any layout pass must preserve spatial row/column order.
  - Projector C2F splits and concatenates channel axis `dim=1`; NHWC rewrites must change this to `dim=-1`.
  - Feature flatten is `source.flatten(2).transpose(1,2)`, preserving NCHW row-major spatial order for deformable attention level metadata.
  - Deformable attention fallback reshapes values to `[B*heads, head_dim, H, W]` and calls `grid_sample`; this region should be protected by a no-layout-translation guard unless a channel-last equivalent is implemented.
  - Postprocess target sizes are `(height,width)` but box scaling order is `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation for `pixel_values [B,3,H,W]`.
- Optional `pixel_mask [B,H,W]`; source synthesizes all-ones if absent.
- `Conv2d` patch embedding with kernel/stride 16.
- Bicubic `interpolate` for absolute position table resize; boolean/float `interpolate` for masks.
- Window-major reshape/permute patterns and inverse restore to NCHW feature maps.
- `flatten`, `transpose`, `permute`, `reshape`, `view`, `contiguous`.
- Multi-level `cat` over channel axis and sequence axis.
- `split`, `gather`, `topk`, `repeat`, `expand`, `stack`, `meshgrid`, `linspace`, `arange`, `cumsum`, `prod`.
- Boolean masks, `masked_fill` with zero and `-inf`, and variable-length postprocess filtering.

Neural network primitives:

- ViT backbone attention: `Linear(hidden -> hidden)` Q/K/V/O, GELU MLP `hidden -> 4*hidden -> hidden`, LayerNorm, learned CAE gamma scale, residual adds.
- Projector `Conv2d + BatchNorm2d + activation`, `ConvTranspose2d`, C2F split/concat, RepVGG-like two sequential 3x3 ConvNorm blocks, channel-first LayerNorm.
- Decoder Linear projections with bias for self-attention, deformable attention offsets/weights/value/output, FFNs, proposal heads, query position MLP, class heads, and box heads.
- Decoder MLP `d_model -> 2048 -> d_model` with ReLU by default.
- Box MLP `d_model -> d_model -> d_model -> 4`, final `exp` on width/height deltas through `refine_bboxes`.
- Activations: GELU, SiLU, ReLU, sigmoid, softmax, exp, log-like loss math only for training.
- Dropout exists but is inactive in inference.

Attention primitives:

- Backbone noncausal MHA over window-local or global image-token sequences.
- Decoder noncausal self-attention over fixed object queries.
- Multiscale deformable cross-attention with learned offsets and weights.
- No causal masks, no KV cache, no RoPE/ALiBi in attention.

Position/custom math:

- Absolute backbone position table interpolation from pretrain patch grid to runtime patch grid.
- Query position MLP fed by sinusoidal embedding of refined reference boxes.
- Grid proposal generation over each projected feature level.
- `refine_bboxes(reference, deltas)` using center delta scaled by reference width/height and `exp` width/height update.

Preprocessing/postprocessing-coupled ops:

- `DeformableDetrImageProcessor`: resize 640x640, rescale `1/255`, ImageNet normalize, pad, channel-first output.
- Postprocess sigmoid logits, global top-k over query-class pairs, gather boxes, convert cxcywh to xyxy, scale to target sizes, threshold.
- No NMS in source postprocess.

Training-only/deferred ops:

- Group-DETR full 13-group query path.
- Hungarian matching, IoU-aware BCE/class loss, L1/GIoU box loss, cardinality loss, auxiliary outputs, distributed loss normalization, and mask losses.

## 5. Layer/block breakdown

Processor output:

```text
pixel_values: [B,3,640,640] float, NCHW, rescaled and normalized
pixel_mask:   [B,640,640] int/bool-compatible, 1 for valid pixels
```

ViT backbone:

```text
x = Conv2d(3 -> H_backbone, kernel=16, stride=16)(pixel_values)
  # [B,H_backbone,40,40] for 640 inputs
if absolute_pos:
  pos = bicubic_resize(position_table_without_cls, 40, 40)
  x = x + pos.permute_to_NCHW()

x = x.permute(0,2,3,1)
x = window_pack_4x4(x)                         # [B*16,100,H_backbone]
for layer_idx in backbone layers:
  y = LayerNorm(x)
  if layer_idx not in window_block_indices:
    y = y.reshape(B, 16*100, H_backbone)       # global image-token attention
  y = MHA(y)                                   # K has no bias
  y = gamma_1 * y
  if global path:
    y = y.reshape(B*16,100,H_backbone)
  x = x + y
  x = x + gamma_2 * MLP(LayerNorm(x))
selected stages are unpacked back to [B,H_backbone,40,40]
```

Projector scale path:

```text
for scale in projector_scale_factors:
  for each selected backbone map:
    if scale == 2.0:
      maybe Conv1x1(C -> C/2) + ReLU, then ConvTranspose2d stride 2
    if scale == 0.5:
      Conv3x3 stride 2 + BN + ReLU
    if scale == 1.0:
      identity sampling
  x = cat(sampled_maps, dim=channel)
  x = Conv1x1(Cin -> 2*hidden) + BN + SiLU
  chunks = split(x, hidden, dim=channel)
  for 3 bottlenecks:
    y = Conv3x3 + BN + SiLU -> Conv3x3 + BN + SiLU
    append y
  x = cat(chunks_and_bottlenecks, dim=channel)
  x = Conv1x1((2+blocks)*hidden -> d_model) + BN + SiLU
  x = LayerNorm over channel axis
```

Decoder preparation:

```text
features = [(source_l [B,d_model,Hl,Wl], mask_l [B,Hl,Wl])]
source_flatten = cat(source_l.flatten(2).transpose(1,2), dim=1) # [B,S,d_model]
mask_flatten = cat(mask_l.flatten(1), dim=1)                    # [B,S]
spatial_shapes = [[H0,W0], ...]
level_start_index = cumsum(Hl*Wl)
valid_ratios = stack(valid_width/W, valid_height/H per level)

object_query, proposals, invalid_mask =
  gen_encoder_output_proposals(source_flatten, ~mask_flatten, spatial_shapes)
object_query = Linear(d_model -> d_model) + LayerNorm
proposal_logits = Linear(d_model -> num_labels)(object_query)
proposal_boxes = refine_bboxes(proposals, BoxMLP(object_query))
topk_idx = topk(max_class_score, num_queries)
topk_boxes = gather(proposal_boxes, topk_idx).detach()
topk_object_query = gather(object_query, topk_idx)

target = learned_query_feat[:num_queries].expand(B,Q,d_model)
reference_points = refine_bboxes(topk_boxes, learned_reference[:num_queries])
```

Decoder layer, repeated 3 times:

```text
query_sine = sine_embedding(reference_points, num_feats=d_model//2)
query_pos = MLP(2*d_model -> d_model -> d_model)(query_sine)

y = SelfAttention(q/k from x + query_pos, v from x)
x = LayerNorm(x + y)

y = DeformableAttention(
      query=x + query_pos,
      value=source_flatten,
      reference_points=reference_points[:, :, None, :] * valid_ratios,
      spatial_shapes,
      level_start_index)
x = LayerNorm(x + y)

x = LayerNorm(MLP_with_internal_residual(x))
```

Detection head:

```text
logits = Linear(d_model -> num_labels)(last_hidden_state)
box_delta = MLP(d_model -> d_model -> d_model -> 4)(last_hidden_state)
pred_boxes = refine_bboxes(initial_mixed_query_reference_points, box_delta)
```

## 6. Attention requirements

Backbone attention:

- Noncausal self-attention, MHA, no cache.
- Tiny/small backbone: 12 heads over hidden 192, head dim 16.
- Medium/large backbone: 12 heads over hidden 384, head dim 32.
- Xlarge backbone: 12 heads over hidden 768, head dim 64.
- Window blocks attend over `window_height * window_width` tokens per window. With 640 input and 4x4 windows, this is 100 tokens.
- Non-window blocks reshape to global attention over all 1600 patch tokens per image.
- Q and V bias follow `qkv_bias`; K is bias-free.

Decoder self-attention:

- Noncausal self-attention over `Q=100` for tiny or `Q=300` for other inspected inference configs.
- MHA, not GQA/MQA. Small/medium use 8 heads x 32; large/xlarge use 12 heads x 32.
- Q/K receive query position embeddings; V uses original hidden states.
- In training, group-DETR reshapes `Q*13` queries across the batch dimension. First inference should reject or ignore this branch.

Multiscale deformable cross-attention:

- Noncausal cross-attention from object queries to flattened multi-level image features.
- `value_proj(source_flatten)` -> `[B,S,n_heads,head_dim]`.
- `sampling_offsets(query)` -> `[B,Q,n_heads,num_levels,n_points,2]`.
- `attention_weights(query)` -> softmax over `num_levels * n_points`, then `[B,Q,n_heads,num_levels,n_points]`.
- For 4D reference points, sampling locations are:

```text
reference_xy + sampling_offsets / n_points * reference_wh * 0.5
```

- Fallback samples each feature level after reshaping values to `[B*heads, head_dim, H, W]`; grids use `2 * sampling_locations - 1`, bilinear mode, zero padding, and `align_corners=False`.
- This is not SDPA/FlashAttention-compatible. It needs a separate deformable-attention lowering/provider or a source-faithful eager composition.

Not applicable: causal masks, autoregressive prefill/decode, KV cache, RoPE, ALiBi, sliding-window text attention, packed varlen sequence metadata.

## 7. Position encoding and custom math

Absolute ViT position interpolation:

```python
def lw_detr_abs_pos(table, height, width):
    # table is [1, pretrain_patches + 1, C]; first slot is cls and is removed.
    table = table[:, 1:]
    size = int(sqrt(table.shape[1]))
    table = table.reshape(1, size, size, C).permute(0, 3, 1, 2)
    table = interpolate(table, size=(height, width),
                        mode="bicubic", align_corners=False)
    return table.permute(0, 2, 3, 1)
```

Query sine embedding from normalized boxes:

```python
def lw_detr_query_sine(pos, num_feats):
    dim = 10000 ** (2 * floor(arange(num_feats) / 2) / num_feats)
    parts = []
    for coord in unbind_last(pos):
        e = coord[..., None] * (2 * pi) / dim
        parts.append(stack([sin(e[..., 0::2]), cos(e[..., 1::2])]).flatten_last())
    parts[0], parts[1] = parts[1], parts[0]  # DETR y,x convention
    return concat(parts, dim=-1)
```

Grid proposal generation:

```python
grid_xy = (meshgrid_x_y + 0.5) / [valid_width, valid_height]
wh = 0.05 * (2.0 ** level)
proposal = concat([grid_xy, wh, wh], -1)
valid = all((proposal > 0.01) & (proposal < 0.99), dim=-1)
proposal = where(valid & not_padding, proposal, 0)
```

Box refinement:

```python
def refine_bboxes(reference, delta):
    cxcy = delta[..., :2] * reference[..., 2:] + reference[..., :2]
    wh = exp(delta[..., 2:]) * reference[..., 2:]
    return concat([cxcy, wh], dim=-1)
```

Precompute opportunities: absolute position tables per patch-grid size,
window packing metadata, full-valid masks per feature level, and grid proposals
per `(feature levels, valid extents, dtype)` bucket. Proposal valid ratios still
depend on padded image masks when batches contain mixed valid sizes.

## 8. Preprocessing and input packing

Processor contract from the inspected `DeformableDetrImageProcessor` configs:

- `do_resize=True`, `size={"height":640,"width":640}`.
- `do_rescale=True`, `rescale_factor=1/255`.
- `do_normalize=True`, ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- `do_pad=True`; emits `pixel_mask`.
- Output tensor layout is channel-first: `pixel_values [B,3,H,W]`.
- `format="coco_detection"` appears for annotation conversion; inference only needs image tensors and optional target sizes.

Recommended first DinoML ABI:

```text
pixel_values: contiguous NCHW float tensor [B,3,640,640]
pixel_mask:   bool/int tensor [B,640,640], optional all-ones default
target_sizes: optional CPU-side [B,2] height,width for postprocess
```

Postprocess contract:

```text
raw outputs:
  logits:     [B,Q,C] independent class logits
  pred_boxes: [B,Q,4] normalized cxcywh boxes

post_process_object_detection(threshold=0.5, top_k=100):
  scores = sigmoid(logits).reshape(B, Q*C)
  topk_scores, topk_indexes = topk(scores, min(top_k, Q*C))
  topk_boxes = floor(topk_indexes / C)
  labels = topk_indexes % C
  boxes = gather(cxcywh_to_xyxy(pred_boxes), topk_boxes)
  if target_sizes: boxes *= [width,height,width,height]
  filter each image by score > threshold
```

There is no NMS and no no-object class removal in LW-DETR postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> patch Linear/GEMM

Source pattern:

```text
Conv2d(3 -> H, kernel=16, stride=16, padding=0)
```

Replacement:

```text
WindowFlatten_NCHW([B,3,H,W], 16x16) -> GEMM(weight_flat.T) -> [B,Hpatch,Wpatch,H]
```

Preconditions:

- Kernel size equals stride, padding 0, dilation 1, groups 1.
- Input height/width divisible by patch size or explicitly cropped/rejected.
- Flatten order matches PyTorch OIHW convolution and NCHW row-major windows.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * 16 * 16)
```

Failure cases: dynamic image sizes without divisibility guards, NHWC path without matching window flatten order, or future non-16 patch configs.

### Rewrite: absolute position cache

Source pattern:

```text
drop cls pos -> reshape square table -> bicubic interpolate -> add to patch map
```

Replacement:

```text
CachedAbsPos[(patch_h, patch_w, dtype)] + patch_embeddings
```

Preconditions:

- Position table is constant.
- Runtime patch grid belongs to a known bucket.
- Bicubic `align_corners=False` parity is validated.

Failure cases: arbitrary dynamic image sizes or non-square source position table.

### Rewrite: window-major packing as metadata view

Source pattern:

```text
[B,H,W,C] -> reshape/permute -> [B*num_windows, tokens_per_window,C]
```

Replacement:

```text
layout-aware view or generated index map, avoiding physical transposes when possible
```

Preconditions:

- `num_windows` is a square and divides patch height/width into equal windows.
- Attention backend accepts strided/tiled token layout or the compiler emits a fused pack.

Failure cases: non-divisible patch grid, backend requiring dense contiguous `[B,T,C]`, or selected hidden states consumed as materialized feature maps.

### Rewrite: Conv2d + BatchNorm2d fold in projector

Source pattern:

```text
Conv2d(..., bias=False) -> BatchNorm2d(eps) -> activation
```

Replacement:

```text
Conv2d with folded weight/bias -> activation
```

Preconditions:

- Inference mode; BN running stats and affine weights are constants.
- Preserve `batch_norm_eps`.
- Consumer does not read pre-BN output.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = w * scale[:, None, None, None]
b_fold = beta - running_mean * scale
```

Failure cases: training, mutable BN stats, or unguarded channel-axis change under NHWC.

### Rewrite: 1x1 projector conv -> GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0)
```

Replacement:

```text
[B*H*W,Cin] GEMM -> reshape to [B,Cout,H,W]
```

Preconditions:

- Kernel 1, stride 1, padding 0, dilation 1, groups 1.
- NCHW flatten/restore order is preserved or the entire region is channel-last controlled.

Failure cases: stride-2/downsample convs, transposed convs, or arbitrary accessors.

### Rewrite: deformable attention provider

Source pattern:

```text
value split by levels -> grid_sample per level -> weight/sum -> output_proj
```

Replacement:

```text
fused multiscale deformable attention CUDA provider
```

Preconditions:

- Coordinate convention is `sampling_grid = 2 * sampling_locations - 1`.
- Bilinear sampling uses zero padding and `align_corners=False`.
- Level flatten order matches `spatial_shapes` and `level_start_index`.
- `num_heads`, `num_levels`, and `num_points` are represented in the provider manifest.

Failure cases: layout mismatch, 2D-reference branch untested for target configs, dynamic unsupported levels, or accumulated dtype differences around bilinear sampling.

### Rewrite: postprocess top-k/gather

Source pattern:

```text
sigmoid(logits).flatten -> topk -> div/mod labels -> gather boxes -> threshold
```

Replacement:

```text
fused top-k score selection plus box gather, with variable-output materialization later
```

Preconditions:

- Independent sigmoid class logits.
- No NMS and no no-object class.
- Top-k and class count known.

Failure cases: configs routed to a different detector postprocess, product-added NMS, or variable-output ABI not available.

## 10. Kernel fusion candidates

Highest priority:

- Multiscale deformable attention provider. The fallback uses reshapes, level splits, `grid_sample`, stack, multiply, and sum inside every decoder layer; it is the unique high-risk LW-DETR operator.
- Window/global ViT attention kernels for the backbone. Global blocks attend over 1600 tokens for 640 inputs, while window blocks have many small 100-token attention calls after batch-window packing.
- Projector Conv2d+BN folding and 1x1 conv lowering. The projector is launch-heavy and channel-concat-heavy before the decoder.
- Top-k proposal selection plus gather. This path sits on the critical bridge between image features and decoder queries.

Medium priority:

- Patch embedding Conv2d-to-GEMM rewrite.
- Absolute position and proposal grid precompute for fixed 640 buckets.
- Decoder self-attention and FFN fusion for small fixed query lengths.
- Box refinement fusion around MLP output, `exp`, center/size update, and optional gather.

Lower priority:

- GPU postprocess variable-output materialization. CPU/Python can preserve parity initially.
- Training loss kernels.
- GPU image preprocessing; useful later but not required for first model-graph parity.

## 11. Runtime staging plan

Stage 1: config, processor, and weight import.

- Parse `LwDetrConfig` and nested `LwDetrViTConfig`.
- Load tiny/small/large weights and validate expected tensor names/shapes.
- Keep `DeformableDetrImageProcessor` outside the compiled graph.

Stage 2: backbone parity.

- Implement NCHW patch embedding, absolute position interpolation, window pack/unpack, window/global ViT blocks, CAE gamma scaling, and selected stage extraction.
- Validate tiny 6-layer and small 10-layer variants.

Stage 3: projector parity.

- Implement scale `1.0` path first, then large/xlarge scale `2.0` and `0.5`.
- Validate C2F split/concat and channel-first LayerNorm.

Stage 4: decoder preparation parity.

- Implement mask downsampling, multi-level flatten metadata, valid ratios, proposal generation, top-k, gather, and reference-point refinement.

Stage 5: one decoder layer.

- Implement query sine embedding, query position MLP, decoder self-attention, deformable attention fallback/provider, and FFN.
- Validate one-layer then full 3-layer outputs.

Stage 6: detection heads and postprocess.

- Implement final class and box heads.
- Add source-compatible sigmoid top-k postprocess with no NMS, initially CPU/Python.

Stage 7: optimization.

- Add Conv+BN folding, patch/1x1 GEMM rewrites, deformable-attention provider, precompute caches, and guarded NHWC conv islands.

## 12. Parity and validation plan

Random/operator tests:

- Patch embedding Conv2d vs lowered patch GEMM for `[B,3,640,640]`.
- Absolute position interpolation for 40x40 and any dynamic bucket.
- Window pack/unpack round trip and global-window attention reshape.
- Backbone attention Q/V biased and K bias-free projection parity.
- Projector sampling for scales `1.0`, `2.0`, and `0.5`.
- Channel-first LayerNorm vs source permute-LayerNorm-permute.
- `refine_bboxes`, proposal grid generation, valid ratio, top-k/gather.
- Deformable attention fallback/provider for 4D reference points, multiple levels, fp32/fp16.
- Postprocess sigmoid/top-k/label/box gather/threshold with target-size scaling.

Model slice tests:

- Processor fixture: save HF `pixel_values` and `pixel_mask` for one image.
- Backbone selected feature-map parity for tiny, small, and xlarge widths.
- Projector output parity for small single-scale and large two-scale.
- Decoder prep parity: `source_flatten`, `spatial_shapes`, `level_start_index`, top-k indices, and initial references.
- Single decoder-layer parity, then full decoder parity.
- Raw `logits` and `pred_boxes` parity for `AnnaZhang/lwdetr_small_60e_coco`.
- End-to-end postprocessed detection parity: compare scores, labels, and boxes at fixed threshold and `top_k`.

Suggested tolerances:

- fp32 source-faithful blocks: `rtol=1e-4`, `atol=1e-5`, with slightly looser full-model boxes after deep accumulation.
- fp16/bf16 optimized paths: compare to reduced-precision PyTorch and use looser checks around bicubic interpolation, ConvTranspose2d, and deformable bilinear sampling.

## 13. Performance probes

- CPU preprocessing throughput for 640x640 resize/rescale/normalize/pad.
- Backbone-only throughput split by window blocks and global blocks.
- Window pack/unpack overhead and attention backend comparison for 100-token windows vs 1600-token global blocks.
- Projector-only latency for small/medium single-scale vs large/xlarge two-scale.
- Decoder preparation time: proposal generation, class scoring, top-k, gather, reference refinement.
- Deformable attention backend comparison: eager fallback vs fused provider, `n_points=2` vs 4, one vs two levels, fp32/fp16.
- Decoder-layer split: self-attention, deformable attention, FFN, box/query MLPs.
- Postprocess latency for sigmoid top-k/gather/threshold on CPU vs GPU.
- Batch-size sweep for B=1/4/8 and mixed masks.
- Layout probe: NCHW baseline vs guarded NHWC projector/backbone conv islands, including transpose cost and no-layout guards.

No benchmark measurements are included; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training mode, dropout, gradients, and gradient checkpointing.
- Group-DETR 13-group training query execution.
- Hungarian matching, IoU-aware BCE, L1/GIoU/cardinality losses, auxiliary loss outputs, and distributed loss normalization.
- Mask losses; LW-DETR inspected source target is object detection, not segmentation inference.
- Third-party Hub custom kernel as a required dependency. First parity should use source fallback semantics.
- NMS, because HF LW-DETR postprocess does not use it.
- Autoregressive generation, KV cache, beam search, and token sampling; not applicable.
- Arbitrary dynamic image resolutions; start with 640x640 buckets and explicit guards.
- GPU image preprocessing until model-graph parity is stable.

## 15. Final implementation checklist

- [ ] Parse `LwDetrConfig` and nested `LwDetrViTConfig`.
- [ ] Load AnnaZhang LW-DETR weights and verify backbone/projector/decoder/head tensor shapes.
- [ ] Accept `pixel_values [B,3,640,640]` and optional `pixel_mask [B,640,640]`.
- [ ] Implement NCHW patch embedding and absolute position interpolation.
- [ ] Implement window-major pack/unpack and selected backbone hidden-state extraction.
- [ ] Implement LW-DETR ViT attention with Q/V bias and bias-free K.
- [ ] Implement CAE gamma residual scaling, LayerNorm, and GELU MLP blocks.
- [ ] Implement projector sampling scales `1.0`, `2.0`, and `0.5`.
- [ ] Implement projector C2F/RepVGG-style ConvNorm blocks and channel-first LayerNorm.
- [ ] Implement mask downsampling, feature flattening, `spatial_shapes`, and `level_start_index`.
- [ ] Implement proposal grid generation, valid ratios, top-k scoring, and gather.
- [ ] Implement `refine_bboxes` and query/reference initialization.
- [ ] Implement query sinusoidal embedding and query position MLP.
- [ ] Implement decoder self-attention with Q/K position addition and plain V.
- [ ] Implement multiscale deformable attention fallback and plan a CUDA provider.
- [ ] Implement decoder FFN and final layernorm outputs.
- [ ] Implement class logits and box MLP heads.
- [ ] Implement HF-compatible sigmoid top-k object-detection postprocess with no NMS.
- [ ] Add parity tests for custom math, backbone, projector, decoder prep, decoder layer, raw outputs, and postprocess.
- [ ] Benchmark backbone, projector, decoder prep, deformable attention, and postprocess separately.
- [ ] Add guarded Conv+BN folding, patch/1x1 GEMM rewrites, and NHWC islands after source-faithful parity.
