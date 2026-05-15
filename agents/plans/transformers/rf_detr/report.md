# RF-DETR Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary target: stevenbucaille/rf-detr-base, RfDetrForObjectDetection.
  Representative sweep:
    stevenbucaille/rf-detr-nano
    stevenbucaille/rf-detr-small
    stevenbucaille/rf-detr-base
    stevenbucaille/rf-detr-medium
    stevenbucaille/rf-detr-large
    stevenbucaille/rf-detr-seg-small
    stevenbucaille/rf-detr-seg-xxlarge

Config source:
  https://huggingface.co/stevenbucaille/rf-detr-base/raw/main/config.json
  https://huggingface.co/stevenbucaille/rf-detr-base/raw/main/preprocessor_config.json
  Same raw paths for the other sweep checkpoints. Local snapshots are in
  agents/plans/transformers/rf_detr/_sources/.

Source files inspected:
  transformers/src/transformers/models/rf_detr/configuration_rf_detr.py
  transformers/src/transformers/models/rf_detr/modeling_rf_detr.py
  transformers/src/transformers/models/rf_detr/modular_rf_detr.py
  transformers/src/transformers/models/rf_detr/convert_rf_detr_weights_to_hf.py
  transformers/src/transformers/models/detr/image_processing_detr.py
  transformers/docs/source/en/model_doc/rf_detr.md
  transformers/tests/models/rf_detr/test_modeling_rf_detr.py

Any missing files or assumptions:
  configuration_rf_detr.py and modeling_rf_detr.py are generated from
  modular_rf_detr.py; future upstream edits should inspect modular_rf_detr.py
  first. RF-DETR uses DetrImageProcessor from the DETR family. Official
  stevenbucaille configs and preprocessor configs fetched without 401/403/404.
  No remote code is required for inspected official checkpoints.
```

This report targets CUDA inference for object detection first. Instance segmentation is documented as a later head. Training losses, Hungarian matching, random mask-point sampling, annotation conversion, and Group DETR multi-group training behavior can stay outside the compiled runtime initially.

## 2. High-level architecture

RF-DETR is an image detector/segmenter, not an autoregressive model. It combines a windowed DINOv2-like ViT backbone, a C2F convolutional projector, a shallow fixed-query DETR decoder, a two-stage top-k proposal bridge, and class/box heads.

```text
CPU image preprocessing
  -> pixel_values [B,3,H,W] and optional pixel_mask [B,H,W]
  -> RF-DETR DINOv2 backbone: patch conv, cls token, interpolated pos table, window/global ViT blocks
  -> selected backbone feature maps, NCHW [B,384,H/patch,W/patch]
  -> scale projector: concat selected maps on channel axis, C2F conv stack, LayerNorm
  -> flattened memory [B,S,256], masks, spatial metadata
  -> encoder proposal heads: per-pixel boxes/classes, top-k proposals
  -> learned query features and reference boxes
  -> decoder: query self-attention + one-level deformable cross-attention
  -> logits [B,Q,num_labels] and normalized cxcywh boxes [B,Q,4]
  -> DETR postprocess: thresholded scores, labels, absolute xyxy boxes
```

Stage decomposition:

- CPU/data pipeline: image decode, resize, rescale by `1/255`, ImageNet normalization, optional padding and `pixel_mask`.
- Backbone: channel-first patch embedding consumes `[B,3,H,W]`; tokens are mostly `[B*windows^2,T,C]` until selected hidden states are unpartitioned for feature output.
- Projector: selected feature maps have the same spatial size and channel width; they are concatenated along NCHW channel dimension, projected to `d_model=256`, and normalized.
- Proposal bridge: encoder memory is scored by group-specific class/box heads; inference uses only one group, selects top `num_queries`, and refines learned reference boxes.
- Decoder: fixed query sequence, noncausal self-attention, deformable cross-attention over flattened image memory, FFN, and per-layer normalized hidden states.
- Detection head: final decoder state predicts class logits and boxes using `refine_bboxes(reference_points, deltas)`.
- Segmentation head: optional later stage using detector features, per-layer query features, depthwise conv blocks, query/spatial matmul masks, and DETR mask postprocess.

Independently stageable units: DetrImageProcessor handoff, RF-DETR DINOv2 backbone, feature projector, top-k proposal bridge, one decoder layer with deformable attention, class/box heads, DETR object postprocess, and optional segmentation mask head.

## 3. Important config dimensions

Worked example: `stevenbucaille/rf-detr-base`.

| Field | Value | Source |
| --- | ---: | --- |
| architecture | `RfDetrForObjectDetection` | config.json |
| `model_type` | `rf_detr` | config.json |
| nested backbone `model_type` | `rf_detr_dinov2` | config.json |
| labels | 91 COCO detection labels | config.json |
| processor size | `560x560` | preprocessor_config |
| processor rescale/normalize | `1/255`, ImageNet mean/std | preprocessor_config |
| `d_model` | 256 | config.json |
| `num_queries` | 300 | config.json |
| `group_detr` | 13, but inference uses 1 group | config/source |
| `decoder_layers` | 3 | config.json |
| decoder self-attention heads | 8, head dim 32 | config/source |
| decoder cross-attention heads | 16, head dim 16 | config/source |
| `decoder_n_points` | 2 | config.json |
| `num_feature_levels` | 1 | config.json |
| decoder FFN width | 2048 | source default |
| backbone hidden size | 384 | config.json |
| backbone layers/heads | 12 / 6 | config.json |
| backbone patch size | 14 | config.json |
| backbone image size | 518 | config.json |
| backbone windows | 4 | config.json |
| backbone output stages | `stage2`, `stage5`, `stage8`, `stage11` | config.json |
| cache/generation | not applicable | source |

Representative checkpoint sweep:

| Model id | Head | Processor size | Backbone image/patch/windows | Backbone output stages | Decoder layers | Queries |
| --- | --- | --- | --- | --- | ---: | ---: |
| `stevenbucaille/rf-detr-nano` | object detection | `384x384` | `384 / 16 / 2` | `stage3/6/9/12` | 2 | 300 |
| `stevenbucaille/rf-detr-small` | object detection | `512x512` | `512 / 16 / 2` | `stage3/6/9/12` | 3 | 300 |
| `stevenbucaille/rf-detr-base` | object detection | `560x560` | `518 / 14 / 4` | `stage2/5/8/11` | 3 | 300 |
| `stevenbucaille/rf-detr-medium` | object detection | `576x576` | `576 / 16 / 2` | `stage3/6/9/12` | 4 | 300 |
| `stevenbucaille/rf-detr-large` | object detection | `704x704` | `704 / 16 / 2` | `stage3/6/9/12` | 4 | 300 |
| `stevenbucaille/rf-detr-seg-small` | instance segmentation | `384x384` | `384 / 12 / 2` | `stage3/6/9/12` | 4 | 100 |
| `stevenbucaille/rf-detr-seg-xxlarge` | instance segmentation | `768x768` | `768 / 12 / 2` | `stage3/6/9/12` | 6 | 300 |

Shared official-config facts:

- `d_model=256`, backbone hidden size 384, backbone layers 12, backbone heads 6.
- `decoder_self_attention_heads=8`, `decoder_cross_attention_heads=16`, `decoder_n_points=2`.
- `hidden_expansion=0.5`, `c2f_num_blocks=3`, and segmentation `intermediate_size=1024`.
- `num_feature_levels=1`, so official RF-DETR deformable cross-attention uses one projected feature map even though the operator is written as multiscale.
- Checkpoint JSONs contain `projector_scale_factors`, but current RF-DETR source does not use a multi-scale projector list; the source-owned projector concatenates selected backbone maps and applies one `RfDetrC2FLayer`.

## 3a. Family variation traps

- Source layout is NCHW for image tensors, backbone feature maps, projector convolutions, pixel masks, and segmentation spatial features. NHWC/channel-last should be only a guarded local optimization.
- The RF-DETR backbone is not stock DINOv2. It adds window partitioning, global-attention unpartition/repartition, changed positional interpolation settings, and stage outputs from selected hidden states.
- Backbone `image_size` can differ from processor size. Base uses backbone `image_size=518` but preprocessor `560x560`; source interpolates the position table to runtime `height // patch_size` and `width // patch_size`.
- Window partitioning assumes height and width are divisible by `patch_size * num_windows`. Official configs satisfy this for processor sizes; arbitrary dynamic sizes need guards or fallback.
- The source comments note a copied original bug around height/width order during feature-map reshape. DinoML should reproduce observed source semantics, not silently repair it.
- In inference, `group_detr` collapses to 1. Training uses 13 groups and group packing in self-attention, but this is not required for first inference parity.
- Encoder top-k proposal selection is numerically sensitive. Transformers tests skip flash-attention equivalence because tiny backbone differences can change top-k proposals and downstream boxes.
- Deformable cross-attention fallback uses `grid_sample`, `align_corners=False`, and zero padding. `disable_custom_kernels=True` in official configs does not block DinoML from adding a native provider if it matches fallback semantics.
- The decoder stores only the initial reference points in `intermediate_reference_points`; current source does not update references per decoder layer inside `RfDetrDecoder.forward`.
- Detection logits are `num_labels`, not `num_labels + 1`; DETR object postprocess calls `softmax(logits)[..., :-1]`, so the last label behaves as no-object/background by convention.
- Source object postprocess does not run NMS. Adding NMS would not be parity.
- Axis-sensitive layout traps:
  - `pixel_values` is `[B,C,H,W]`; channel axis is 1.
  - Backbone selected tokens drop cls via `hidden_state[:, 1:]`, reshape to patch grid, then `permute(0,3,1,2)`.
  - Projector concatenates selected features with `torch.cat(hidden_states, dim=1)`.
  - `pixel_mask` is `[B,H,W]`; projector mask uses `interpolate(pixel_mask[None].float(), size=features.shape[-2:]).to(bool)[0]`.
  - Memory flatten is `features.flatten(2).transpose(1,2)`, preserving NCHW row-major spatial order.
  - Deformable attention reshapes values to `[B*heads, head_dim, H, W]` before `grid_sample`; keep this a no-layout-translation boundary until a channel-last provider is proven.
  - Segmentation blocks intentionally permute NCHW to NHWC for LayerNorm/Linear and then back to NCHW.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `pixel_values [B,3,H,W]`, optional `pixel_mask [B,H,W]`, and synthesized all-ones mask when absent.
- `Conv2d`, including patch embedding `kernel=stride=patch`, standard 1x1/3x3 convolutions, and depthwise 3x3 conv for segmentation.
- `LayerNorm` over final dimension and RF-DETR channels-first wrapper via `permute`.
- `flatten`, `transpose`, `permute`, `reshape`, `view`, `contiguous`, `expand`, `repeat`, `split`, `chunk`, `cat`, `stack`.
- `interpolate` bicubic antialiased position-table resize, mask resize, and segmentation bilinear resize.
- `meshgrid`, `linspace`, `arange`, `cumsum`, `prod`, mask reductions for valid ratios.
- `masked_fill`, bool mask logic, `topk`, `gather`, max over class dimension.
- Variable-length Python postprocess outputs after score thresholding.

Neural network primitives:

- DINOv2-like patch embedding: `Conv2d(3 -> 384, kernel=patch_size, stride=patch_size)`.
- Backbone block: `LayerNorm -> noncausal MHA -> Linear output -> LayerScale -> residual -> LayerNorm -> MLP/SwiGLU -> LayerScale -> residual`.
- Official configs use MLP GELU, not SwiGLU, but source has a `use_swiglu_ffn` branch.
- Projector: concatenate 4 selected backbone maps, `1x1 ConvNormAct`, channel split, three RepVGG blocks, channel concat, `1x1 ConvNormAct`, channels-first LayerNorm.
- RepVGG projector block: two sequential `3x3 Conv2d + channels-first LayerNorm + activation`.
- Decoder self-attention: Q/K/V/O Linear projections with bias, noncausal MHA over queries.
- Deformable cross-attention: value projection, sampling-offset Linear, attention-weight Linear+softmax, bilinear grid sampling, weighted sum, output projection.
- Decoder FFN: `Linear(256 -> 2048) -> activation -> dropout -> Linear(2048 -> 256) -> dropout -> residual`, then LayerNorm.
- Proposal and prediction heads: `Linear(256 -> 256)`, LayerNorm, class Linear, 3-layer ReLU MLP for boxes, final class Linear.
- Segmentation head: depthwise conv, NHWC LayerNorm, Linear, activation, 1x1 conv projection, query MLP, matmul query by flattened spatial features, bias add.

Attention primitives:

- Backbone noncausal MHA: 6 heads, head dim 64 for 384-wide official configs.
- Decoder self-attention: 8 heads, head dim 32 for `d_model=256`.
- Decoder deformable cross-attention: 16 heads, head dim 16, 1 feature level, 2 sample points in official configs.
- No causal mask, no KV cache, no generation-time cache, no RoPE/ALiBi.

Position/custom math:

- Bicubic positional table interpolation with `align_corners=False`, `antialias=True`, fp32 interpolation, cast back to table dtype.
- Window partition/unpartition of patch tokens with per-window cls-token replication.
- Encoder proposals from normalized feature-grid centers and fixed width/height `0.05 * 2**level`.
- Sinusoidal query position embedding from normalized reference boxes, then MLP to query position.
- RF-DETR `refine_bboxes`: `new_xy = delta_xy * ref_wh + ref_xy`; `new_wh = exp(delta_wh) * ref_wh`.

Preprocessing/postprocessing-coupled ops:

- DetrImageProcessor resize, rescale, normalize, pad, and pixel-mask creation.
- Object postprocess: softmax over labels, drop last class, threshold, center-to-corners conversion, scale by target `(height,width)`.
- Segmentation postprocess from DETR processor for optional mask heads.

## 5. Layer/block breakdown

RF-DETR DINOv2 backbone:

```text
pixel_values [B,3,H,W]
  -> patch Conv2d(3 -> 384, kernel=patch, stride=patch)
  -> flatten patches to [B,N,384]
  -> optional bool_masked_pos replacement with mask token
  -> prepend cls token
  -> add interpolated absolute position embeddings
  -> if num_windows > 1: split patch grid into windows and repeat cls per window
  -> 12 transformer layers
  -> selected hidden states -> optional LayerNorm -> drop cls -> unpartition -> NCHW maps
```

Windowed/global backbone layer:

```text
if layer is global attention:
  hidden = window_unpartition_before_attention(hidden)
normed = LayerNorm(hidden)
attn = MHA(normed)
if layer is global attention:
  attn = window_partition_after_attention(original_shape, attn)
hidden = residual + LayerScale(attn)
mlp = MLP_or_SwiGLU(LayerNorm(hidden))
hidden = hidden + LayerScale(mlp)
```

Projector:

```text
features = tuple of 4 NCHW maps, each [B,384,Hf,Wf]
x = cat(features, dim=1)                         # [B,1536,Hf,Wf]
x = Conv1x1(1536 -> 256) + channels-first LN + SiLU
chunks = split x into two [B,128,Hf,Wf] tensors
for 3 bottlenecks:
  y = Conv3x3(128 -> 128) + LN + SiLU
  y = Conv3x3(128 -> 128) + LN + SiLU
  append y
x = cat(5 chunks, dim=1)                         # [B,640,Hf,Wf]
x = Conv1x1(640 -> 256) + channels-first LN + SiLU
x = channels-first LN                            # [B,256,Hf,Wf]
mask = interpolate(pixel_mask, [Hf,Wf]).bool()
```

Proposal bridge:

```text
memory = features.flatten(2).transpose(1,2)      # [B,S,256], S=Hf*Wf
padding_mask = ~mask.flatten(1)
grid proposals = [cx,cy,w,h] in normalized feature coordinates
invalid_mask = padding or proposals outside (0.01,0.99)
object_query = memory.masked_fill(invalid_mask, 0)
object_query = Linear(256 -> 256) + LayerNorm
scores = class Linear(object_query).masked_fill(invalid, -inf)
deltas = box MLP(object_query)
coords = refine_bboxes(proposals, deltas)
topk_idx = topk(max(scores, dim=-1), num_queries)
topk_coords = gather(coords, topk_idx).detach()
enc_outputs_class = gather(object_query, topk_idx)
```

Decoder:

```text
reference_points = learned_embedding[:Q]         # [Q,4] in inference
reference_points[:Q] = refine_bboxes(topk_coords, reference_points[:Q])
target = learned_query_feat[:Q].expand(B,Q,256)
valid_ratios = [valid_width/Wf, valid_height/Hf]
query_pos = MLP(sinusoidal_position(reference_points * valid_ratios))

repeat decoder_layers times:
  x = x + SelfAttention(q/k from x + query_pos, v from x)
  x = LayerNorm(x)
  x = x + DeformableCrossAttention(x + query_pos, memory, reference_points, spatial_shapes)
  x = LayerNorm(x)
  x = LayerNorm(MLP_residual(x))

last_hidden_state = LayerNorm output from final layer
```

Detection head:

```text
logits = Linear(256 -> num_labels)(last_hidden_state)
boxes_delta = MLP(256 -> 256 -> 256 -> 4)(last_hidden_state)
pred_boxes = refine_bboxes(reference_points, boxes_delta)   # normalized cxcywh
```

Segmentation head:

```text
spatial = bilinear_interpolate(backbone_features, image_size / mask_downsample_ratio)
for each decoder layer:
  spatial = depthwise Conv3x3 -> NHWC LayerNorm -> Linear -> activation -> NCHW + residual
  spatial_proj = Conv1x1(256 -> 256)
  query = LayerNorm -> Linear(256 -> 1024) -> activation -> Linear(1024 -> 256)
  query = Linear(256 -> 256)
  mask_logits = query @ spatial_proj.flatten(2) + scalar_bias
  mask_logits = view [B,Q,H/ratio,W/ratio]
```

## 6. Attention requirements

Backbone attention:

- Noncausal self-attention over cls+patch tokens.
- Official configs use MHA with 6 heads and 64-dimensional heads.
- Windowed layers attend inside local windows with replicated cls tokens. Global layers unpartition windows to attend over all window tokens together, then repartition.
- Source uses Transformers attention backend dispatch for eager/SDPA/Flash/Flex, but top-k proposal sensitivity means DinoML should validate with eager-equivalent math before enabling alternate kernels.

Decoder self-attention:

- Noncausal MHA over object queries, no mask in inference.
- Query and key inputs receive `position_embeddings`; value projection uses unpositioned query features.
- Official query length is usually 300, except segmentation small uses 100.
- No KV cache; the full fixed query set is processed every image.

Deformable cross-attention:

- Query source: detector query hidden states plus query position embeddings.
- Key/value source: flattened projected image feature memory `[B,S,256]`.
- Official configs use one spatial level and two sample points, but source supports `num_feature_levels` and both 2D/4D reference point modes.
- Value projection output shape is `[B,S,16,16]` for official configs.
- Sampling offsets shape is `[B,Q,16,1,2,2]`; attention weights shape is `[B,Q,16,1,2]`.
- For 2-coordinate reference points: `sampling_locations = reference + offsets / [width,height]`.
- For 4-coordinate reference points: `sampling_locations = ref_xy + offsets / n_points * ref_wh * 0.5`.
- Fallback samples each level by converting locations to `grid_sample` coordinates with `2 * loc - 1`, bilinear mode, zero padding, and `align_corners=False`.
- Returned cross-attention weights are deformable point weights, not dense `[Q,S]` attention matrices.

Not applicable: causal masks, sliding-window decoder attention, ALiBi/RoPE, packed varlen metadata, generation caches, or cache reorder.

## 7. Position encoding and custom math

Backbone positional interpolation:

```python
def interpolate_rf_detr_pos(pos_table, embeddings, height, width, patch_size):
    cls = pos_table[:, :1]
    patch = pos_table[:, 1:]
    dim = embeddings.shape[-1]
    src = int((patch.shape[1]) ** 0.5)
    patch = patch.reshape(1, src, src, dim).permute(0, 3, 1, 2)
    patch = interpolate(
        patch.float(),
        size=(height // patch_size, width // patch_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(pos_table.dtype)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return cat([cls, patch], dim=1)
```

Window partitioning:

```python
patches = tokens[:, 1:].view(B, Hp, Wp, C)
patches = patches.view(B, Wn, Wp_per, Wn, Hp_per, C).transpose(2, 3)
windowed = patches.reshape(B * Wn * Wn, Hp_per * Wp_per, C)
cls = tokens[:, :1].repeat(Wn * Wn, 1, 1)
tokens = cat([cls, windowed], dim=1)
```

Query sine embedding:

```python
def encode_sinusoidal_position_embedding(pos_tensor, num_pos_feats=128):
    dim_t = temperature ** (2 * floor(arange(num_pos_feats) / 2) / num_pos_feats)
    embeddings = []
    for coord in unbind(pos_tensor, dim=-1):
        e = coord[..., None] * (2 * pi) / dim_t
        embeddings.append(stack([sin(e[..., 0::2]), cos(e[..., 1::2])], -1).flatten(-2))
    embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return cat(embeddings, dim=-1).to(pos_tensor.dtype)
```

RF-DETR box refinement:

```python
def refine_bboxes(reference_points, deltas):
    xy = deltas[..., :2] * reference_points[..., 2:] + reference_points[..., :2]
    wh = exp(deltas[..., 2:]) * reference_points[..., 2:]
    return cat([xy, wh], dim=-1)
```

Precomputable pieces: backbone position interpolation for fixed processor buckets, proposal mesh grids for fixed feature shape, and query sine frequencies. Dynamic pieces: valid ratios from mask, top-k proposals, reference-point refinement, sampling offsets, and attention weights.

## 8. Preprocessing and input packing

Official RF-DETR checkpoints use `DetrImageProcessor`.

- `image_processor_type="DetrImageProcessor"`.
- `do_resize=True`; official sizes vary from `312x312` to `768x768`.
- `do_rescale=True`, `rescale_factor=1/255`.
- `do_normalize=True`, ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- `do_pad=True`; padded batches emit `pixel_mask`.
- Output tensor layout is channel-first `pixel_values [B,3,H,W]`.
- `pixel_mask [B,H,W]` uses 1 for valid pixels and 0 for padding. If model forward receives no mask, it creates an all-ones mask.

Object-detection postprocess:

- Inputs: `logits [B,Q,C]`, `pred_boxes [B,Q,4]` normalized `cx,cy,w,h`, optional `target_sizes [B,2]` in `(height,width)` order.
- Source applies `softmax(logits, -1)`, then drops the last class by taking `prob[..., :-1].max(-1)`.
- Boxes convert center-to-corners, then if `target_sizes` is present scale by `[width,height,width,height]`.
- Thresholding is per query and returns variable-length per-image lists of `scores`, `labels`, and `boxes`.
- No NMS is performed.

Segmentation postprocess:

- `pred_masks [B,Q,H/ratio,W/ratio]` are logits. Processor applies sigmoid, filters low/no-object predictions, computes segment maps, optionally resizes to target size, and can emit RLE.
- Segmentation parity can be deferred if first target is object detection.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> hidden, kernel=patch_size, stride=patch_size, padding=0)
flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten NCHW patches -> GEMM(weight_flat.T) -> bias -> [B,N,hidden]
```

Preconditions: `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`, input `H,W` divisible by patch size, and flatten order matches PyTorch NCHW convolution patch order.

Failure cases: dynamic image sizes without divisibility guard, non-RGB inputs, or layout pass that changes patch flatten order.

Parity test sketch: compare patch embeddings before cls/pos addition for base `patch=14` and segmentation `patch=12`.

### Rewrite: projector 1x1 Conv2d -> GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0) on [B,C,H,W]
```

Replacement: NCHW spatial flatten to `[B*H*W,Cin]`, GEMM with `[Cin,Cout]`, reshape to `[B,Cout,H,W]`.

Preconditions: kernel 1, stride 1, groups 1, and consumer layout is controlled. Channels-first LayerNorm/activation must be preserved or fused.

Failure cases: 3x3 projector convolutions and segmentation depthwise conv are not covered.

Parity test sketch: compare `RfDetrC2FLayer.conv1`, `conv2`, and segmentation `spatial_features_proj` on fixed feature maps.

### Rewrite: channels-first LayerNorm local NHWC region

Source pattern:

```text
NCHW -> permute NHWC -> LayerNorm(C) -> permute NCHW
```

Replacement: NCHW channels-first LayerNorm kernel, or a guarded NHWC conv/norm island.

Preconditions: normalized dimension is exactly channel count, and any NHWC island rewrites following concat/conv axes explicitly.

Failure cases: crossing into deformable attention or postprocess without restoring source layout.

Parity test sketch: compare `RfDetrLayerNorm(data_format="channels_first")` over projector and segmentation tensors.

### Rewrite: deformable attention provider

Source pattern:

```text
value split by level -> reshape to [B*heads,head_dim,H,W]
sampling_locations -> grid_sample bilinear
attention weights -> weighted sum over levels/points -> output projection
```

Replacement: fused one/multilevel deformable attention CUDA kernel.

Preconditions: coordinate transform is `grid = 2 * sampling_locations - 1`, bilinear interpolation uses zero padding and `align_corners=False`, memory flatten order and `spatial_shapes` match source, and dtype/accumulation policy is explicit.

Failure cases: NHWC/NCHW mismatch, wrong point normalization for 2D vs 4D reference points, or unsupported dynamic feature shapes.

Parity test sketch: random small tensors against source fallback for one-level official shape and synthetic multilevel shapes.

### Rewrite: proposal top-k + gather

Source pattern:

```text
class_scores.max(-1) -> topk(Q, dim=1) -> gather coords and query features
```

Replacement: fused per-image top-k over `S` proposals with parallel gather.

Preconditions: inference `group_detr=1`; `S=Hf*Wf` and `Q` are bounded by config bucket; tie behavior and numerical precision are validated.

Failure cases: training group packing, altered attention kernels changing scores near ties, or dynamic variable output.

Parity test sketch: compare top-k indices and gathered tensors for fixed random scores with tie cases.

### Rewrite: object postprocess as CPU or fixed-size GPU top-k

Source pattern:

```text
softmax(logits) -> drop last class -> max -> threshold -> box cxcywh_to_xyxy -> target-size scale
```

Replacement: CPU postprocess first; later fixed-size GPU score/label/box arrays before threshold compaction.

Preconditions: source softmax/no-object semantics are used; do not substitute sigmoid/focal RT-DETR postprocess. No NMS.

Parity test sketch: compare scores, labels, and boxes for random logits and target sizes, including all-below-threshold output.

## 10. Kernel fusion candidates

Highest priority:

- Deformable attention provider. The fallback `grid_sample` path is the unique decoder bottleneck and must preserve exact coordinate and interpolation semantics.
- Backbone/projector GEMM and conv/norm/activation kernels. RF-DETR spends substantial work in patch embedding, ViT MLPs, projector 1x1/3x3 convs, and LayerNorm.
- Query self-attention and backbone MHA. Official query lengths and window token lengths are modest, but many layers make launch/layout overhead visible.
- Proposal top-k + gather. It is a small but numerically decisive bridge; stable top-k parity matters more than raw speed at first.

Medium priority:

- Patch Conv2d-to-GEMM for fixed image buckets.
- Channels-first LayerNorm fused kernels for projector and segmentation head.
- RF-DETR box refine MLP plus `exp` plus multiply/add fusion.
- Precompute interpolated position embeddings and proposal grids per fixed size.

Lower priority:

- Segmentation mask matmul and postprocess acceleration.
- Dropout elimination and DropPath identity removal in inference graphs.
- CPU/Python object postprocess replacement, unless end-to-end latency shows it dominates.

## 11. Runtime staging plan

1. Config and processor handoff: parse `RfDetrConfig` plus nested `RfDetrDinov2Config`, load base weights, accept preprocessed NCHW tensors plus optional mask, and keep processor/postprocess in Python.
2. Backbone parity: implement patch embedding, cls/pos addition, positional interpolation, window partition/unpartition, and 12 RF-DETR DINOv2 layers.
3. Projector parity: implement NCHW feature concat, C2F conv stack, channels-first LayerNorm, and mask interpolation.
4. Proposal bridge parity: implement feature-grid proposals, valid ratios, invalid masks, class/box proposal heads, `topk`, `gather`, learned query/reference embeddings, and `refine_bboxes`.
5. Decoder parity: implement query sine embedding, decoder self-attention, deformable cross-attention fallback/provider, FFN, and LayerNorm.
6. Detection head and postprocess: return logits and normalized boxes; run source DETR postprocess externally.
7. Optimization: add guarded NHWC/channel-last islands only for local conv/norm/activation regions, then enable patch-conv GEMM, 1x1 conv GEMM, fused MHA, top-k/gather, and deformable attention provider.
8. Segmentation variant: add segmentation blocks, query/spatial mask matmul, mask logits, and DETR segmentation postprocess.

## 12. Parity and validation plan

- Config tests for the representative sweep, including nested backbone dimensions and checkpoint fields ignored by current source.
- Processor contract tests: resize/rescale/normalize/pad/pixel-mask behavior and model fallback when `pixel_mask` is omitted.
- Random custom math tests: positional interpolation, window partition/unpartition, `encode_sinusoidal_position_embedding`, `refine_bboxes`, proposal grid generation, and valid mask.
- Backbone parity: compare patch embeddings, selected hidden states, and final selected feature maps.
- Projector parity: compare C2F intermediate/final features and downsampled mask.
- Proposal parity: compare class scores, deltas, top-k indices, gathered query features, and refined reference points.
- Deformable attention parity: one-level official case plus synthetic multilevel/2D-reference cases against source fallback.
- Decoder parity: one-layer and full-decoder outputs for base and one segmentation variant.
- End-to-end raw output parity: `logits`, `pred_boxes`, and first few postprocessed scores/labels/boxes on the COCO fixture used by Transformers tests.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` before top-k; after top-k use exact index checks plus output tolerances. fp16/bf16 `rtol=1e-2, atol=1e-2`, with tighter checks around top-k scores when possible.

## 13. Performance probes

- Processor throughput by official size: 312, 384, 512, 560, 576, 704, 768.
- Backbone-only throughput split into patch embedding, windowed attention layers, global attention layers, and MLPs.
- Projector throughput: selected-map concat, 1x1 convs, 3x3 convs, channels-first LayerNorm.
- Proposal bridge throughput: grid/proposal generation, class/box heads, top-k, and gather.
- Decoder throughput split into query self-attention, deformable cross-attention, FFN, and prediction heads.
- Deformable attention backend comparison: source-like fallback vs fused provider, fp32/fp16, batch and query sweeps.
- Batch-size sweep for detection variants and segmentation variants.
- Memory probes: backbone hidden-state retention for selected stages, projector temporaries, deformable attention sampled values, segmentation mask logits.
- Layout probes: NCHW baseline vs guarded NHWC conv/norm islands, including transpose cost and no-layout guard boundaries.
- Postprocess latency: CPU DETR postprocess vs fixed-size GPU score/box preparation.

## 14. Skip/defer list

- Training losses, Hungarian matching, auxiliary-loss plumbing, random mask point sampling, and Group DETR multi-group training behavior.
- Roboflow conversion script execution and original `.pth` conversion; use already-converted HF safetensors/configs first.
- Segmentation head for first object-detection integration.
- Segmentation postprocess, RLE output, and mask annotation preprocessing.
- Remote custom kernels from Hub; source fallback is enough for first parity.
- NMS, because official source postprocess does not perform NMS.
- Arbitrary dynamic image sizes; start with official processor buckets and divisibility guards.
- `use_swiglu_ffn=True`, `num_windows=1`, or `num_feature_levels>1` variants unless a target checkpoint requires them.
- Training/eval differences in DropPath and dropout; inference should treat them as identity/no-op.

## 15. Final implementation checklist

- [ ] Parse `RfDetrConfig` and nested `RfDetrDinov2Config`.
- [ ] Load official stevenbucaille RF-DETR object-detection weights.
- [ ] Accept `pixel_values [B,3,H,W]` and optional `pixel_mask [B,H,W]`.
- [ ] Implement RF-DETR DINOv2 patch embedding and positional interpolation.
- [ ] Implement window partition/unpartition and global-attention layer handling.
- [ ] Implement backbone MHA, MLP, LayerScale, LayerNorm, and selected feature outputs.
- [ ] Implement projector NCHW concat, C2F conv stack, and channels-first LayerNorm.
- [ ] Implement mask downsampling and valid ratios.
- [ ] Implement proposal grid generation, invalid masking, class/box proposal heads, top-k, and gather.
- [ ] Implement learned query/reference embeddings and `refine_bboxes`.
- [ ] Implement query sine embedding and reference-point MLP.
- [ ] Implement decoder self-attention.
- [ ] Implement deformable cross-attention fallback and plan fused CUDA provider.
- [ ] Implement decoder FFN and final LayerNorm.
- [ ] Implement detection class and box heads.
- [ ] Implement DETR object postprocess parity externally or in runtime wrapper.
- [ ] Add parity tests for custom math, backbone, projector, proposal bridge, decoder layer, full model, and postprocess.
- [ ] Benchmark backbone, projector, proposal bridge, decoder, deformable attention, and postprocess separately.
- [ ] Later add RF-DETR instance segmentation head and mask postprocess.
