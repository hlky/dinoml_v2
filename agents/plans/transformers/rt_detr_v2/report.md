# RT-DETRv2 Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary inference target: PekingU/rtdetr_v2_r50vd, RTDetrV2ForObjectDetection.
  Representative sweep:
    PekingU/rtdetr_v2_r18vd
    PekingU/rtdetr_v2_r34vd
    PekingU/rtdetr_v2_r50vd
    PekingU/rtdetr_v2_r101vd

Config source:
  https://huggingface.co/PekingU/rtdetr_v2_r18vd/raw/main/config.json
  https://huggingface.co/PekingU/rtdetr_v2_r34vd/raw/main/config.json
  https://huggingface.co/PekingU/rtdetr_v2_r50vd/raw/main/config.json
  https://huggingface.co/PekingU/rtdetr_v2_r101vd/raw/main/config.json
  Same raw paths for preprocessor_config.json.

Source files inspected:
  X:/H/transformers/src/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py
  X:/H/transformers/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py
  X:/H/transformers/src/transformers/models/rt_detr_v2/modular_rt_detr_v2.py
  X:/H/transformers/src/transformers/models/rt_detr_v2/convert_rt_detr_v2_weights_to_hf.py
  X:/H/transformers/src/transformers/models/rt_detr/configuration_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/configuration_rt_detr_resnet.py
  X:/H/transformers/src/transformers/models/rt_detr/modeling_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/modeling_rt_detr_resnet.py
  X:/H/transformers/src/transformers/models/rt_detr/image_processing_rt_detr.py
  X:/H/transformers/tests/models/rt_detr_v2/test_modeling_rt_detr_v2.py

Local snapshots:
  H:/dinoml_v2/agents/plans/transformers/rt_detr_v2/_sources/

Any missing files or assumptions:
  modeling_rt_detr_v2.py and configuration_rt_detr_v2.py are generated from
  modular_rt_detr_v2.py; future source edits should inspect the modular file
  first. The V2 modular source imports RT-DETR base classes and overrides the
  decoder deformable attention; the generated modeling file also contains copied
  base code, so this report uses inherited RT-DETR behavior only where the V2
  source or modular inheritance proves it. No official checkpoint in the sweep
  was gated or unavailable. V2 checkpoints use the existing RTDetrImageProcessor
  configs, not a separate in-tree RTDetrV2 image processor implementation.
```

## 2. High-level architecture

RT-DETRv2 is a real-time object detector: a ResNet-style CNN backbone, a hybrid CNN/transformer encoder, and a fixed-query transformer decoder with V2 multiscale deformable cross-attention. It is not autoregressive and has no KV cache.

```text
CPU image preprocessing
  -> pixel_values [B,3,H,W] and optional pixel_mask [B,H,W]
  -> RT-DETR ResNet backbone, NCHW multi-scale feature maps
  -> 1x1 encoder input projections
  -> hybrid encoder: AIFI dense self-attention on selected levels + FPN/PAN conv fusion
  -> decoder input projections + multi-level flatten
  -> anchor/proposal generation + top-k query selection
  -> decoder: query self-attention + V2 multiscale deformable cross-attention
  -> class logits [B,Q,C] and normalized cxcywh boxes [B,Q,4]
  -> postprocess scores, labels, absolute xyxy boxes; no NMS
```

Stage decomposition:

- CPU/data pipeline: image decode, resize to 640x640 in official processors, rescale by `1/255`, optional pad/mask, and optional annotation transforms for training.
- Backbone: RT-DETR ResNet family through `load_backbone(config)` with selected `stage2`, `stage3`, `stage4` outputs.
- Encoder input projection: each backbone output becomes `encoder_hidden_dim` channels through `Conv2d(1x1, bias=False) + BatchNorm2d`.
- Hybrid encoder: `encode_proj_layers` levels pass through AIFI dense self-attention over spatial tokens; FPN and PAN remain NCHW convolutional regions.
- Decoder prep: encoded levels are projected to `d_model`, flattened in NCHW row-major spatial order, concatenated, and paired with `spatial_shapes` plus `level_start_index`.
- Query selection: anchors are generated from feature grids; encoder class scores choose top `num_queries`; gathered memory and reference boxes seed decoder queries.
- Decoder: each layer performs noncausal query self-attention, V2 deformable cross-attention, FFN, and iterative box refinement.
- Postprocess: common focal-loss path applies sigmoid, flattened top-k over `[queries, classes]`, box gather, thresholding, and target-size scaling. Source behavior intentionally does not include NMS.

## 3. Important config dimensions

Worked example: `PekingU/rtdetr_v2_r50vd`.

| Field | Value | Source |
| --- | ---: | --- |
| primary task | object detection | source/config |
| architecture | `RtDetrV2ForObjectDetection` | config.json |
| `model_type` | `rt_detr_v2` | config.json |
| `torch_dtype` | `float32` | config.json |
| labels | 80 COCO labels | config.json |
| image processor | `RTDetrImageProcessor` | preprocessor_config.json |
| processor size | `640x640` | preprocessor_config.json |
| processor rescale | `1/255` | preprocessor_config.json |
| processor normalize | `false` | preprocessor_config.json |
| `d_model` | 256 | config.json |
| `encoder_hidden_dim` | 256 | config.json |
| `encoder_layers` | 1 | config.json |
| `encoder_attention_heads` | 8 | config.json |
| AIFI head dim | 32 | inferred from config |
| `encoder_ffn_dim` | 1024 | config.json |
| `encoder_activation_function` | `gelu` | config.json |
| conv activation | `silu` | config.json |
| `decoder_layers` | 6 | config.json |
| `decoder_attention_heads` | 8 | config.json |
| decoder head dim | 32 | inferred from config |
| `decoder_ffn_dim` | 1024 | config.json |
| `decoder_activation_function` | `relu` | config.json |
| `num_queries` | 300 | config.json |
| `num_feature_levels` | 3 | config.json |
| `decoder_n_levels` | 3 | config.json |
| `decoder_n_points` | 4 | config.json |
| V2 `decoder_offset_scale` | 0.5 | config.json |
| V2 `decoder_method` | `default` | config.json |
| `feat_strides` | `[8,16,32]` | config.json |
| `encode_proj_layers` | `[2]` | config.json |
| `with_box_refine` | true | config.json |
| cache/generation | not applicable | source |

Representative checkpoint sweep:

| Model id | Backbone config | Encoder inputs | Encoder width/FFN | Decoder in | Decoder layers | Hidden expansion | V2 method | Labels | Processor |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `PekingU/rtdetr_v2_r18vd` | basic, hidden `64/128/256/512`, depths `2/2/2/2` | `128/256/512` | `256/1024` | `256/256/256` | 3 | 0.5 | `default` | 80 | `640x640`, rescale, no normalize |
| `PekingU/rtdetr_v2_r34vd` | basic, hidden `64/128/256/512`; depths omitted in downloaded config but conversion sets `3/4/6/3` | `128/256/512` | `256/1024` | `256/256/256` | 4 | 0.5 | `default` | 80 | same |
| `PekingU/rtdetr_v2_r50vd` | bottleneck ResNet defaults via nested config | `512/1024/2048` | `256/1024` | `256/256/256` | 6 | 1.0 | `default` | 80 | same |
| `PekingU/rtdetr_v2_r101vd` | bottleneck, depths `3/4/23/3` | `512/1024/2048` | `384/2048` | `384/384/384` | 6 | 1.0 | `default` | 80 | same |

Omitted fields supplied by source/default configs include `batch_norm_eps=1e-5`, `anchor_image_size=None`, `learn_initial_query=False`, `num_denoising=100`, `decoder_n_levels=3`, `decoder_offset_scale=0.5`, `decoder_method="default"`, and RT-DETR ResNet defaults for configs whose nested `backbone_config` omits hidden sizes/depths/layer type.

## 3a. Family variation traps

- V2 is not a pure alias of `rt_detr`: modular source subclasses RT-DETR but specifically replaces decoder cross-attention with `RTDetrV2MultiscaleDeformableAttention`.
- `decoder_n_levels` controls V2 deformable attention, while `num_feature_levels` controls decoder input levels. Official configs keep both at 3; DinoML should guard mismatches.
- V2 offsets are shaped as flattened `[n_levels * n_points, 2]` per head before sampling, then split by `num_points_list`. RT-DETR base attention uses a different offset layout. Treat V2 deformable attention as a distinct op contract unless a future source comparison proves a common lowering.
- Official configs set `decoder_method="default"`, which uses bilinear `grid_sample` with `sampling_grids = 2 * locations - 1`. Source also has a `discrete` gather/clamp branch; DinoML should reject or separately implement that branch if a checkpoint enables it.
- For 4D reference boxes, V2 scales offsets by `1 / n_points`, reference box width/height, and `decoder_offset_scale=0.5`. This is a parity-sensitive difference from a naive deformable attention formula.
- Source tensors are NCHW through preprocessing, backbone, mask interpolation, projections, FPN/PAN, and feature flattening. NHWC/channel-last should be introduced only as a guarded local optimization.
- `encode_proj_layers=[2]` means only the lowest-resolution projected feature map goes through dense AIFI self-attention by default. Do not assume a full transformer encoder over all feature levels.
- R101 changes `encoder_hidden_dim` to 384 and `decoder_in_channels` to `384/384/384`, while `d_model` remains 256. Decoder projections must bridge 384-channel encoder maps to 256-wide decoder memory.
- `learn_initial_query=False` in official configs: decoder query content is gathered from encoder memory and detached. The learned query embedding path is present but not required for the sweep.
- `with_box_refine=True` wires per-layer class/box heads and reference-point updates. The tied-weight keys indicate class and box head cloning/alias expectations should be preserved when loading.
- `pixel_mask` is optional. If missing, source creates all ones `[B,H,W]`. Official preprocessor configs use `do_pad=False`, so normal fixed-size batches often omit a real padding mask.
- Postprocessing has no NMS. Adding NMS would break source parity.
- Axis/layout traps:
  - `pixel_values` is `[B,C,H,W]`; channel axis is 1.
  - `pixel_mask` is `[B,H,W]`; masks are resized to each backbone feature map `[-2:]`.
  - Feature flatten is `source.flatten(2).transpose(1,2)`, preserving NCHW row-major `(h,w)` order.
  - AIFI restores with `.permute(0,2,1).reshape(B,C,H,W)`.
  - FPN/PAN concat uses `dim=1`; NHWC translation must rewrite to `dim=-1`.
  - Deformable attention samples per-level tensors as `[B*heads, head_dim, H, W]` through `grid_sample`; protect this region with a no-layout-translation guard unless a channel-last equivalent is implemented.
  - Postprocess target sizes are `(height,width)`, but scaling order is `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input `pixel_values [B,3,H,W]`; optional `pixel_mask [B,H,W]`.
- `Conv2d`, `BatchNorm2d`, frozen batch-norm affine, `MaxPool2d`, residual add.
- `interpolate` for mask downsampling and nearest FPN upsampling.
- `flatten`, `transpose`, `permute`, `reshape`, `view`, `contiguous`.
- `cat`/`concat` across channel and sequence dimensions; `stack`, `split`, `gather`, `topk`, `repeat`, `tile`, `unsqueeze`, `squeeze`.
- `meshgrid`, `arange`, `cumsum`, `prod`, integer tensors for `spatial_shapes` and `level_start_index`.
- Boolean masks, `masked_fill`, comparisons, `where`, `clamp`, `log`, `sigmoid`, `softmax`.
- Variable-length Python postprocess records after thresholding.

Neural network primitives:

- RT-DETR ResNet backbone from nested `rt_detr_resnet` config.
- Encoder input projections: `Conv2d(C_backbone -> encoder_hidden_dim, kernel=1, bias=False) + BatchNorm2d`.
- Hybrid encoder conv blocks: `Conv2d + BatchNorm2d + SiLU`, RepVGG-style `3x3 + 1x1` branch add, and CSPRep layers.
- Decoder projections: `Conv2d(C_encoder -> d_model, kernel=1, bias=False) + BatchNorm2d`, plus optional stride-2 `3x3` extra levels.
- Linear layers for self-attention Q/K/V/O, FFNs, deformable offsets/weights/value/output, proposal heads, query position MLP, class heads, and box heads.
- LayerNorm over final dimension.
- Activations: ReLU, GELU, SiLU, sigmoid, softmax.

Attention primitives:

- Dense noncausal MHA self-attention in AIFI and decoder queries.
- V2 multiscale deformable cross-attention with official defaults `heads=8`, `levels=3`, `points=4`, `head_dim=32`.
- `grid_sample` bilinear fallback with zero padding and `align_corners=False` for `decoder_method="default"`.
- Optional `decoder_method="discrete"` nearest-like integer gather branch if future configs enable it.

Position/custom math:

- 2D sine position embedding for AIFI.
- Anchor generation from spatial grids and feature-level `wh`.
- `inverse_sigmoid` and iterative box refinement.
- Center-to-corners and corners-to-center box conversion.

Preprocessing/postprocessing-coupled ops:

- Shared `RTDetrImageProcessor`: resize to 640x640, rescale, no normalize, optional padding mask.
- Focal-loss postprocess branch: sigmoid, flattened top-k, modulo/division label/query recovery, gather boxes, threshold.
- Softmax/no-object postprocess branch is implemented in processor but not used by official `use_focal_loss=True` sweep.

Training-only/deferred ops:

- Contrastive denoising query generation, random label/box perturbation, Hungarian matching, and losses.
- COCO annotation conversion and mask annotation transforms.

## 5. Layer/block breakdown

Backbone:

```text
pixel_values [B,3,H,W]
  -> RT-DETR ResNet stem/stages
  -> selected stage2/stage3/stage4 feature maps, NCHW
```

At 640x640, selected features are approximately strides 8/16/32: `[B,C2,80,80]`, `[B,C3,40,40]`, `[B,C4,20,20]`.

Encoder projection and AIFI:

```text
for each backbone feature:
  y = Conv1x1(C_in -> encoder_hidden_dim, bias=False)
  y = BatchNorm2d(y)

for enc_ind in encode_proj_layers:
  x = feature.flatten(2).permute(0,2,1)      # [B,H*W,C]
  pos = sine_2d_position(H,W,C)
  repeat encoder_layers:
    x = LayerNorm(x + SelfAttention(q,k from x+pos, v from x))
    x = LayerNorm(x + MLP(x))
  feature = x.permute(0,2,1).reshape(B,C,H,W)
```

Hybrid encoder FPN/PAN:

```text
top-down:
  lateral = ConvNormAct1x1(lowest_or_prior_top)
  up = nearest_interpolate(lateral, scale=2)
  fused = concat([up, higher_resolution_backbone_feature], channel_dim=1)
  out = CSPRepLayer(fused)

bottom-up:
  down = ConvNormAct3x3_stride2(previous_high_resolution)
  fused = concat([down, next_fpn_feature], channel_dim=1)
  out = CSPRepLayer(fused)
```

Decoder preparation:

```text
for each encoded feature:
  source = Conv/BN projection to d_model
  source_flat = source.flatten(2).transpose(1,2)
source_flatten = cat(source_flat, dim=1)      # [B,S,d_model]
spatial_shapes = [[H0,W0], [H1,W1], [H2,W2]]
level_start_index = cumsum(Hl*Wl)
anchors, valid_mask = generate_anchors(spatial_shapes)
memory = valid_mask.float() * source_flatten
output_memory = LayerNorm(Linear(memory))
enc_logits = Linear(output_memory)
enc_boxes_unact = BoxMLP(output_memory) + anchors
topk_ind = topk(max(enc_logits, class_dim), num_queries)
target = gather(output_memory, topk_ind).detach() unless learn_initial_query
reference_points_unact = gather(enc_boxes_unact, topk_ind).detach()
```

Decoder layer:

```text
reference_points = sigmoid(reference_points_unact)
query_pos = MLP(reference_points)             # 4 -> 2*d_model -> d_model

x = LayerNorm(x + SelfAttention(q/k from x + query_pos, v from x))
x = LayerNorm(x + RTDetrV2MultiscaleDeformableAttention(
      query=x + query_pos,
      memory=source_flatten,
      reference_points=reference_points[:, :, None, :],
      spatial_shapes,
      spatial_shapes_list))
x = LayerNorm(x + FFN(x))

box_delta = bbox_embed[layer](x)
reference_points = sigmoid(box_delta + inverse_sigmoid(reference_points)).detach()
logits = class_embed[layer](x)
```

Detection head:

```text
logits = intermediate_logits[:, -1]                 # [B,300,num_labels]
pred_boxes = intermediate_reference_points[:, -1]   # [B,300,4] normalized cxcywh
```

## 6. Attention requirements

Dense self-attention:

- Noncausal MHA, no cache.
- Used in AIFI selected spatial level(s) and decoder query sequence.
- Q/K projections consume `hidden_states + position_embeddings` when positions are supplied; V projection consumes unpositioned hidden states.
- Eager math is scaled dot product, additive mask when present, softmax over key dimension, dropout, value matmul, output projection.
- Official widths use `8` heads and `head_dim=32` for both AIFI 256-wide levels and decoder 256-wide queries. R101 AIFI uses `encoder_hidden_dim=384`, so AIFI `head_dim=48`.

V2 deformable cross-attention:

- Query-driven, noncausal cross-attention from fixed object queries to flattened multi-level image memory.
- `value_proj(memory)` -> `[B,S,heads,head_dim]`.
- Optional attention mask zeroes invalid memory positions before sampling.
- `sampling_offsets = Linear(query)` -> `[B,Q,heads,levels * points,2]`.
- `attention_weights = softmax(Linear(query), dim=-1)` over `levels * points`.
- For 2D reference points: `locations = ref_xy + offsets / [width,height]`.
- For 4D reference boxes: `offset = offsets * (1/n_points) * ref_wh * decoder_offset_scale`; `locations = ref_xy + offset`.
- `decoder_method="default"`: `sampling_grids = 2 * locations - 1`, split by level, bilinear `grid_sample(..., padding_mode="zeros", align_corners=False)`, multiply by attention weights, sum points/levels, then output projection.
- `decoder_method="discrete"`: source scales by `[width,height]`, adds `0.5`, casts to `int64`, clamps coordinates, gathers exact pixels, and then weights/sums. Official configs do not use this branch.

Not applicable: causal masks, autoregressive KV cache, RoPE/ALiBi, sliding windows, packed varlen attention, and generation controller logic.

## 7. Position encoding and custom math

2D sine position embedding used by AIFI:

```python
def rt_detr_v2_sine_2d(height, width, embed_dim, temperature=10000):
    pos_dim = embed_dim // 4
    omega = arange(pos_dim) / pos_dim
    omega = 1.0 / (temperature ** omega)
    y, x = meshgrid(arange(height), arange(width), indexing="ij")
    out_y = flatten(y).outer(omega)
    out_x = flatten(x).outer(omega)
    return cat([sin(out_y), cos(out_y), sin(out_x), cos(out_x)], dim=1)
```

This can be precomputed per `(height,width,encoder_hidden_dim,dtype)` bucket. It depends on actual feature-map size when dynamic image shapes are admitted.

Anchor generation:

```python
def generate_anchors(spatial_shapes, grid_size=0.05):
    anchors = []
    for level, (height, width) in enumerate(spatial_shapes):
        y, x = meshgrid(arange(height), arange(width), indexing="ij")
        xy = stack([x, y], -1) + 0.5
        xy[..., 0] /= width
        xy[..., 1] /= height
        wh = ones_like(xy) * grid_size * (2.0 ** level)
        anchors.append(concat([xy, wh], -1).reshape(1, height * width, 4))
    anchors = concat(anchors, dim=1)
    valid = ((anchors > 1e-2) & (anchors < 1 - 1e-2)).all(-1, keepdim=True)
    logits = log(anchors / (1 - anchors))
    return where(valid, logits, finfo_max), valid
```

V2 deformable offset math:

```python
if reference_points.shape[-1] == 2:
    normalizer = stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1)
    locations = ref[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
else:
    scale = n_points_scale.to(dtype).unsqueeze(-1)
    offset = offsets * scale * ref[:, :, None, :, 2:] * decoder_offset_scale
    locations = ref[:, :, None, :, :2] + offset
```

Box refinement:

```python
def inverse_sigmoid(x, eps=1e-5):
    x = clamp(x, 0, 1)
    return log(clamp(x, min=eps) / clamp(1 - x, min=eps))

reference_points = sigmoid(box_delta + inverse_sigmoid(reference_points))
```

## 8. Preprocessing and input packing

The official V2 checkpoints use `RTDetrImageProcessor`:

- `image_processor_type="RTDetrImageProcessor"`.
- `do_resize=True`, `size={"height":640,"width":640}`.
- `do_rescale=True`, `rescale_factor=0.00392156862745098`.
- `do_normalize=False`; mean/std may be present but are not applied.
- `do_pad=False`; if enabled, processor emits `pixel_mask`.
- Runtime tensor layout is channel-first: `pixel_values [B,3,H,W]`.

First DinoML integration should accept preprocessed NCHW tensors and keep image decode/resize/rescale outside the compiled graph. If `pixel_mask` is absent, source synthesizes an all-ones mask.

Postprocessing contract:

- Inputs: `logits [B,Q,C]`, `pred_boxes [B,Q,4]` normalized center boxes, optional `target_sizes [B,2]` in `(height,width)` order.
- Convert boxes from `cxcywh` to `xyxy`.
- If `target_sizes` is present, multiply by `[width,height,width,height]`.
- Focal-loss branch: `sigmoid(logits)`, flatten to `[B,Q*C]`, take `topk(Q)`, compute `label = index % C`, `query_index = index // C`, gather boxes, then threshold.
- Softmax branch: `softmax(logits)[:, :, :-1]`, max class per query, optional top-k, then threshold.
- Return one variable-length record per image: `scores`, `labels`, `boxes`.
- No NMS is performed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d Fold

Preconditions:

- Inference mode only.
- BatchNorm running stats and affine parameters are constants.
- No graph consumer observes the intermediate Conv output.
- Preserve `batch_norm_eps`.

Replacement:

```text
Conv2d(..., bias=False/True) -> BatchNorm2d(eps)
  => Conv2d with folded weight and bias
```

Parity sketch: compare backbone, projection, FPN, PAN, and CSP blocks before/after folding on random NCHW tensors.

### Rewrite: Frozen BatchNorm2d -> Per-channel Affine

Preconditions:

- Frozen BN buffers are constants.
- Channel axis is known. For NHWC regions, rewrite broadcast constants and axes.

Replacement:

```text
y = x * scale[1,C,1,1] + bias[1,C,1,1]
```

### Rewrite: RepVGG Branch Fusion

Source pattern:

```text
(Conv3x3+BN)(x) + (Conv1x1+BN)(x) -> activation
```

Replacement:

```text
single folded Conv3x3, with padded 1x1 kernel added into center, then activation
```

Preconditions: stride 1, groups 1, same channels, foldable BN, no branch output consumers.

### Rewrite: 1x1 Conv2d -> GEMM

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`.
- Input layout and flatten order are controlled.
- Weight layout is `[Cout,Cin,1,1]`.

Replacement:

```text
[B,Cin,H,W] -> [B*H*W,Cin] -> GEMM(weight.T) -> [B,Cout,H,W]
```

Failure cases: stride-2 extra levels, 3x3 convs, grouped convs, or unguarded layout translation.

### Rewrite: V2 Multiscale Deformable Attention Provider

Source pattern:

```text
value split by level -> [B*heads, head_dim, H, W]
locations -> grid_sample/gather per level
weighted sum over levels/points -> output projection
```

Replacement:

```text
fused V2 deformable-attention CUDA provider
```

Preconditions:

- Preserve flattened memory level order and `spatial_shapes`.
- Preserve `sampling_grids = 2 * locations - 1` and `align_corners=False` for `default`.
- Preserve `n_points_scale`, `decoder_offset_scale`, and 2D versus 4D reference formulas.
- Treat `discrete` as a separate kernel path or reject it.
- Guard NCHW level tensors used by `grid_sample`.

Parity sketch: random small tensors for 2D and 4D references, default and discrete branches, fp32/fp16, multiple levels.

### Rewrite: Fixed-size Detector Top-k

Preconditions:

- `use_focal_loss=True`.
- `Q` and `C` known or bounded.
- Output can remain padded fixed-size before Python thresholding.

Replacement:

```text
sigmoid + flattened topk + label/query decode + box gather
```

Failure cases: softmax/no-object branch, product-level NMS, or strict variable-length GPU output requirement.

## 10. Kernel fusion candidates

Highest priority:

- V2 deformable-attention provider. The source fallback uses per-level reshape, split, `grid_sample`/gather, concat, multiply, and sum inside each decoder layer.
- Conv/BN folding across backbone and hybrid encoder.
- 1x1 conv lowering to GEMM for projection-heavy regions.
- Dense self-attention kernels for AIFI and query self-attention, with small-to-medium sequence lengths.

Medium priority:

- RepVGG branch fusion.
- CSPRepLayer fusion around split 1x1 convs, RepVGG branch, concat/add, and final projection.
- Proposal top-k plus gather over encoder memory.
- Box MLP + inverse-sigmoid + sigmoid refinement fusion.
- Sine position and anchor precompute for fixed 640x640 buckets.

Lower priority:

- Postprocess top-k/threshold variable-output GPU path.
- Elementwise activation fusions around conv blocks when not absorbed into conv epilogues.
- Pixel-mask interpolation optimization, because official fixed-size inference commonly has no padded mask.

## 11. Runtime staging plan

Stage 1: config and processor handoff.

- Parse `RTDetrV2Config` and nested `rt_detr_resnet` config.
- Load R18 and R50 first; add R101 width bridge after baseline works.
- Accept preprocessed NCHW `pixel_values`; synthesize all-ones `pixel_mask` when absent.
- Keep preprocessing and postprocessing in Python initially.

Stage 2: backbone parity.

- Implement or compose RT-DETR ResNet stem/stages and selected outputs.
- Validate stage2/stage3/stage4 maps before optimizations.

Stage 3: hybrid encoder parity.

- Add encoder input projections, AIFI selected-level attention, FPN, PAN, CSPRep, and RepVGG blocks.
- Validate R18/R50, then R101 `encoder_hidden_dim=384`.

Stage 4: decoder prep parity.

- Implement decoder input projections, flatten/concat metadata, anchor generation, valid mask, encoder heads, top-k, and gathers.
- Validate `enc_topk_logits`, `enc_topk_bboxes`, initial target, and references.

Stage 5: V2 decoder parity.

- Implement query self-attention, query position MLP, V2 deformable attention fallback/provider, FFN, and box refinement.
- Validate one decoder layer, then full 3/4/6-layer variants.

Stage 6: raw detection output parity.

- Return `logits [B,300,80]` and `pred_boxes [B,300,4]`.
- Add auxiliary/intermediate outputs only as debugging or training-compatible optional outputs.

Stage 7: postprocess parity.

- Start with CPU/Python `RTDetrImageProcessor.post_process_object_detection` behavior.
- Later add fixed-size GPU top-k/gather and optional variable-output materialization.

Stage 8: optimization.

- Enable BN folding, RepVGG fusion, 1x1 GEMM lowering, cached sine/anchors, deformable attention provider, and guarded NHWC/channel-last conv regions.

## 12. Parity and validation plan

- Config load tests for R18/R34/R50/R101, including nested backbone defaults and omitted fields.
- Processor contract tests: 640x640 resize/rescale/no-normalize, optional pad/mask, and target-size box scaling order.
- Custom op tests:
  - 2D sine position embedding.
  - Anchor generation and valid mask.
  - `inverse_sigmoid` near clamp boundaries.
  - V2 deformable attention for 2D and 4D references.
  - V2 `default` and `discrete` branches if `discrete` is admitted.
- Backbone parity: selected stage outputs for R18 and R50 on small and 640-sized inputs.
- Hybrid encoder parity: projected features, AIFI level output, FPN/PAN outputs.
- Decoder prep parity: flattened memory, spatial metadata, anchors, top-k indices, gathered references.
- Single decoder-layer parity, then full decoder parity.
- End-to-end raw output parity against Transformers for `PekingU/rtdetr_v2_r18vd` and `PekingU/rtdetr_v2_r50vd`.
- Postprocess parity: compare sorted `scores`, `labels`, and scaled boxes; assert no-NMS behavior by preserving overlapping boxes when source preserves them.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 `rtol=1e-2, atol=1e-2`, with looser local tolerances around bilinear sampling and deep conv stacks.

## 13. Performance probes

- CPU preprocessing throughput: decode, resize, rescale to 640x640.
- Backbone-only throughput for R18/R34/R50/R101 and batch sizes 1/4/8/16.
- Hybrid encoder split: AIFI, FPN, PAN.
- Decoder prep split: projections, flatten/concat, anchor generation, top-k/gather.
- Decoder-layer split: self-attention, V2 deformable attention, FFN, box/class heads.
- Deformable attention backend comparison: source-like fallback vs fused provider; fp32/fp16; batch/query/level/point sweeps.
- End-to-end raw model latency excluding postprocess.
- Postprocess CPU vs GPU fixed-top-k latency.
- Memory probes: peak activations by variant and per-layer deformable attention temporaries.
- Layout probes: NCHW baseline vs guarded NHWC/channel-last conv regions, including transpose overhead and deformable-attention guard costs.

## 14. Skip/defer list

- Training losses, Hungarian matching, denoising query generation, and random label/box noise.
- Annotation preprocessing and segmentation-mask annotation support.
- Product-level NMS, because source postprocess has no NMS.
- `decoder_method="discrete"` can be rejected at first because official configs use `default`; add parity only if a target checkpoint enables it.
- Learned initial query path until a checkpoint with `learn_initial_query=True` is targeted.
- `use_focal_loss=False` postprocess branch unless a target checkpoint requires it.
- Dynamic-resolution production scheduler; start with fixed 640x640 or explicit buckets.
- Multi-GPU tensor parallel and distributed execution.

## 15. Final implementation checklist

- [ ] Parse `RTDetrV2Config` and nested `rt_detr_resnet` config.
- [ ] Load official PekingU R18/R50 weights, then R34/R101.
- [ ] Accept NCHW `pixel_values [B,3,H,W]` and optional `pixel_mask [B,H,W]`.
- [ ] Implement RT-DETR ResNet selected feature outputs.
- [ ] Implement/fold BatchNorm2d and FrozenBatchNorm2d.
- [ ] Implement encoder input projections.
- [ ] Implement AIFI 2D sine position embedding and dense self-attention.
- [ ] Implement FPN/PAN ConvNorm/CSPRep/RepVGG blocks.
- [ ] Implement decoder input projections and multi-level flatten metadata.
- [ ] Implement anchor generation, valid mask, encoder proposal heads, top-k, and gather.
- [ ] Implement decoder query self-attention and query position MLP.
- [ ] Implement V2 multiscale deformable attention `default` path.
- [ ] Reject or implement V2 `discrete` deformable attention path.
- [ ] Implement iterative box refinement and class/box heads.
- [ ] Implement CPU/Python postprocess parity with no NMS.
- [ ] Add custom math, deformable attention, backbone, encoder, decoder, full-model, and postprocess parity tests.
- [ ] Benchmark backbone, hybrid encoder, decoder prep, V2 deformable attention, and postprocess separately.
