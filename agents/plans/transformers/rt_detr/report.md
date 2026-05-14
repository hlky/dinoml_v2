# RT-DETR Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary inference target: PekingU/rtdetr_r50vd, RTDetrForObjectDetection.
  Representative sweep:
    PekingU/rtdetr_r18vd
    PekingU/rtdetr_r34vd
    PekingU/rtdetr_r50vd
    PekingU/rtdetr_r101vd
    PekingU/rtdetr_r50vd_coco_o365

Config source:
  https://huggingface.co/PekingU/rtdetr_r50vd/raw/main/config.json
  https://huggingface.co/PekingU/rtdetr_r50vd/raw/main/preprocessor_config.json
  Same raw paths for the other PekingU sweep checkpoints above.

Source files inspected:
  X:/H/transformers/src/transformers/models/rt_detr/configuration_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/configuration_rt_detr_resnet.py
  X:/H/transformers/src/transformers/models/rt_detr/modeling_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/modeling_rt_detr_resnet.py
  X:/H/transformers/src/transformers/models/rt_detr/image_processing_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/image_processing_pil_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/modular_rt_detr.py
  X:/H/transformers/src/transformers/models/rt_detr/convert_rt_detr_original_pytorch_checkpoint_to_hf.py
  X:/H/transformers/tests/models/rt_detr/test_modeling_rt_detr.py
  X:/H/transformers/tests/models/rt_detr/test_image_processing_rt_detr.py

Any missing files or assumptions:
  modeling_rt_detr.py, image_processing_rt_detr.py, and image_processing_pil_rt_detr.py
  are generated from modular_rt_detr.py; future upstream edits should inspect
  the modular file first. No remote code is required for inspected official
  checkpoints. This report targets CUDA inference for object detection from
  processor-produced image tensors first. CPU/PIL/torchvision preprocessing,
  Python postprocessing, training denoising, Hungarian matching, and losses can
  remain outside the compiled graph initially.
```

## 2. High-level architecture

RT-DETR is a real-time object detector with a CNN backbone, a hybrid CNN/transformer encoder, and a fixed-query transformer decoder. It is not an autoregressive language model and has no KV cache or token generation loop.

```text
CPU image preprocessing
  -> pixel_values [B,3,H,W] and optional pixel_mask [B,H,W]
  -> RT-DETR ResNet backbone, NCHW multi-scale features
  -> 1x1 encoder projections
  -> hybrid encoder: AIFI attention on selected feature level + FPN/PAN conv fusion
  -> decoder input projections + multi-level flatten
  -> top-k encoder proposals and reference boxes
  -> decoder: query self-attention + multiscale deformable cross-attention
  -> per-query class logits [B,Q,C] and normalized cxcywh boxes [B,Q,4]
  -> object-detection postprocess: scores, labels, absolute xyxy boxes
```

Stage decomposition:

- CPU/data pipeline: RGB image conversion, resize to configured shape, rescale by `1/255`, optional pad/mask, optional COCO annotation conversion for training.
- Backbone: NCHW RT-DETR-specific ResNet stem and stages; official configs request `stage2`, `stage3`, and `stage4`.
- Encoder projections: one `1x1 Conv2d + BatchNorm2d` per backbone output, producing `encoder_hidden_dim` channels.
- Hybrid encoder: AIFI transformer encoder runs only on levels in `encode_proj_layers` by default `[2]`; top-down FPN and bottom-up PAN remain NCHW convolutional regions.
- Decoder preparation: feature levels are projected to `d_model`, flattened to `[B,S,d_model]`, and paired with `spatial_shapes` and `level_start_index`.
- Proposal selection: anchors are generated from feature map grids, encoder class scores choose top `num_queries`, and gathered memory/reference boxes seed decoder queries.
- Decoder: fixed query count, noncausal self-attention, and multiscale deformable attention over the flattened multi-level memory.
- Detection head: final decoder layer output supplies logits and boxes; intermediate heads are present for refinement and optional auxiliary loss.
- Postprocessing: source does not run NMS. It applies sigmoid or softmax scoring, top-k/threshold filtering, center-to-corners conversion, and optional scaling to `target_sizes`.

Independently stageable units: processor handoff, ResNet backbone, encoder projection stack, AIFI-only level attention, FPN/PAN conv fusion, decoder preparation/top-k proposal selection, one decoder layer with deformable attention, detection heads, and postprocess.

## 3. Important config dimensions

Worked example: `PekingU/rtdetr_r50vd`.

| Field | Value | Source |
| --- | ---: | --- |
| primary task | object detection | architecture/source |
| architecture | `RTDetrForObjectDetection` | config.json |
| `model_type` | `rt_detr` | config.json |
| `torch_dtype` | `float32` | config.json |
| labels | 80 COCO labels | config `id2label` |
| input processor size | `640x640` | preprocessor_config |
| processor rescale | `1/255` | preprocessor_config |
| processor normalize | `false` | preprocessor_config |
| `d_model` | 256 | config.json |
| `encoder_hidden_dim` | 256 | config.json |
| `encoder_layers` | 1 | config.json/source default |
| `encoder_attention_heads` | 8 | config.json |
| AIFI head dim | 32 | inferred `encoder_hidden_dim / heads` |
| `encoder_ffn_dim` | 1024 | config.json |
| `encoder_activation_function` | `gelu` | config.json |
| `activation_function` for conv blocks | `silu` | config.json/source default |
| `decoder_layers` | 6 | config.json |
| `decoder_attention_heads` | 8 | config.json |
| decoder head dim | 32 | inferred `d_model / heads` |
| `decoder_ffn_dim` | 1024 | config.json |
| `decoder_activation_function` | `relu` | config.json |
| `num_queries` | 300 | config.json |
| `num_feature_levels` | 3 | config.json |
| `decoder_n_points` | 4 | config.json |
| `feat_strides` | `[8,16,32]` | config.json |
| `encode_proj_layers` | `[2]` | config.json |
| `with_box_refine` | true | config.json |
| `disable_custom_kernels` | true | config.json |
| cache/generation | not applicable | source |

Representative checkpoint sweep:

| Model id | Backbone config | Backbone channels | Decoder layers | Encoder width | Encoder FFN | Hidden expansion | Labels | Processor |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `PekingU/rtdetr_r18vd` | basic, depths `2/2/2/2` | `64/128/256/512`; encoder inputs `128/256/512` | 3 | 256 | 1024 | 0.5 | 80 | resize `640x640`, rescale, no normalize |
| `PekingU/rtdetr_r34vd` | basic, effective default depths `3/4/6/3` | `64/128/256/512`; encoder inputs `128/256/512` | 4 | 256 | 1024 | 0.5 | 80 | same |
| `PekingU/rtdetr_r50vd` | bottleneck defaults `3/4/6/3` | default `256/512/1024/2048`; encoder inputs `512/1024/2048` | 6 | 256 | 1024 | 1.0 | 80 | same |
| `PekingU/rtdetr_r101vd` | bottleneck, depths `3/4/23/3` | default `256/512/1024/2048`; encoder inputs `512/1024/2048` | 6 | 384 | 2048 | 1.0 | 80 | same |
| `PekingU/rtdetr_r50vd_coco_o365` | bottleneck defaults `3/4/6/3` | default `256/512/1024/2048`; encoder inputs `512/1024/2048` | 6 | 256 | 1024 | 1.0 | 80 | same |

Config fields omitted by some checkpoint JSONs but supplied by source defaults include ResNet hidden sizes, depths, `layer_type`, `decoder_ffn_dim=1024`, `decoder_activation_function="relu"`, `encoder_layers=1`, `num_denoising=100`, `learn_initial_query=False`, `anchor_image_size=None`, and `batch_norm_eps=1e-5`.

## 3a. Family variation traps

- Source tensors are NCHW through preprocessing, ResNet, mask interpolation, encoder projections, FPN/PAN, decoder projections, and feature flattening. NHWC/channel-last must be a guarded optimization, not the semantic import format.
- R18/R34 use ResNet `basic` blocks and smaller backbone channels. R50/R101 use `bottleneck` blocks and wider channels. `encoder_in_channels` must match the chosen backbone outputs.
- R101 changes `encoder_hidden_dim` to 384 and `encoder_ffn_dim` to 2048 while `d_model` remains 256. This creates a real width transition: hybrid encoder features are 384-channel, then decoder input projections produce 256-channel memory.
- `encode_proj_layers=[2]` means only the lowest-resolution projected feature map goes through AIFI attention by default. Do not assume a full transformer encoder over all feature levels.
- `num_feature_levels` can exceed the number of decoder input sources; source appends stride-2 `3x3 Conv2d + BatchNorm2d` levels from the final feature map.
- `learn_initial_query=False` in inspected configs, so decoder initial target features are gathered from encoder memory and detached. If true, an embedding table supplies learned query content.
- `with_box_refine=True` wires per-decoder-layer class and box heads and iteratively updates reference points with `sigmoid(box_delta + inverse_sigmoid(reference_points))`.
- `disable_custom_kernels=True` is present, but the in-library fallback still implements multiscale deformable attention with `grid_sample`. DinoML should treat custom kernel support as an optimization, not required for source parity.
- `use_focal_loss=True` changes postprocess semantics: source flattens sigmoid scores over `[queries, classes]`, takes top `num_queries`, then gathers boxes by query index. The softmax/no-object branch is a separate compatibility path.
- Source postprocessing intentionally has no NMS. Adding NMS would be a product-level change, not parity.
- `pixel_mask` is optional in the model forward. If absent, source uses all ones `[B,H,W]`; image processor default has `do_pad=False`, so normal 640x640 batches may not include a mask.
- Axis-sensitive layout traps:
  - `pixel_values` is `[B,C,H,W]`; source channel axis is 1.
  - `pixel_mask` is `[B,H,W]`; backbone mask downsampling calls interpolate to each feature map `[-2:]`.
  - Feature flatten uses `source.flatten(2).transpose(1,2)`, preserving NCHW row-major `(h,w)` order.
  - AIFI unflattens with `.permute(0,2,1).reshape(B,C,H,W)`.
  - FPN/PAN concatenates channel axis `dim=1`; an NHWC pass must rewrite this to `dim=-1`.
  - Deformable attention samples 4D feature grids as `[B*heads, head_dim, H, W]` using `grid_sample`; this region needs a no-layout-translation guard unless DinoML owns a channel-last equivalent.
  - Postprocess target sizes are `(height,width)`, but box scaling order is `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation for `pixel_values [B,3,H,W]`.
- Optional `pixel_mask [B,H,W]`, with all-ones default and bool/int64 conversion.
- NCHW `Conv2d`, `BatchNorm2d`, frozen batch-norm affine, `MaxPool2d`, `AvgPool2d`, residual add.
- `interpolate` for pixel-mask downsampling and FPN nearest upsampling.
- `flatten(2)`, `transpose`, `permute`, `reshape`, `view`, `contiguous`.
- Multi-level `torch.cat`/`torch.concat` over sequence dimension and channel dimension.
- `torch.stack`, `split`, `gather`, `topk`, `repeat`, `tile`, `unsqueeze`, `squeeze`.
- `meshgrid`, `arange`, `cumsum`, `prod`, and integer tensor metadata for spatial shapes and level starts.
- Boolean mask arithmetic and `masked_fill` for deformable attention value masking.
- Variable-length postprocess outputs after thresholding.

Neural network primitives:

- RT-DETR ResNet stem: three `3x3 Conv2d + BatchNorm2d + activation` layers, then `MaxPool2d(3, stride=2, padding=1)`.
- ResNet basic or bottleneck residual stages with `1x1`/`3x3` convs, batch norm, ReLU, optional avg-pool shortcut for stride-2 bottleneck shortcuts.
- Encoder projection: `Conv2d(C_backbone -> encoder_hidden_dim, kernel=1, bias=False) + BatchNorm2d`.
- Hybrid encoder conv blocks: `Conv2d + BatchNorm2d + SiLU`, RepVGG-style `3x3 + 1x1` branch add, CSP split/add/project.
- Decoder projection: `Conv2d(C_encoder -> d_model, kernel=1, bias=False) + BatchNorm2d`, plus optional `3x3 stride=2` extra levels.
- Linear layers with bias for Q/K/V/O projections, FFNs, deformable attention offsets/weights/value/output, proposal heads, query position MLP, class heads, and box heads.
- LayerNorm over the final dimension for encoder/decoder sublayers and proposal memory.
- Activations: ReLU, GELU, SiLU, sigmoid, softmax, log, clamp, reciprocal/rsqrt-like frozen BN math.
- Dropout is present but inactive in inference.

Attention primitives:

- Noncausal MHA self-attention in AIFI and decoder query self-attention.
- Multiscale deformable cross-attention over flattened feature memory with `num_heads=8`, `num_feature_levels=3`, and `n_points=4`.
- Attention backend dispatch may use eager, SDPA, FlashAttention, or flex attention for self-attention. Deformable attention is separate and not SDPA-compatible.

Position/custom math:

- 2D sine position embeddings for AIFI, generated per feature-map `height,width`.
- Anchor generation from multi-level grids: normalized center `xy`, level-scaled `wh`, valid mask, logit transform.
- `inverse_sigmoid` for iterative box refinement.
- Center-to-corners and corners-to-center box format helpers.

Preprocessing/postprocessing-coupled ops:

- Resize to configured size, rescale by `1/255`, optional normalization, optional padding and pixel mask.
- Postprocess sigmoid score flatten/top-k/gather path for focal-loss checkpoints.
- Postprocess softmax/no-object branch for `use_focal_loss=False` compatibility.

Training-only/deferred ops:

- Contrastive denoising query generation with random labels/boxes.
- Hungarian matching and detection losses.
- Annotation resize/normalize/pad and segmentation-mask annotation conversion.

## 5. Layer/block breakdown

RT-DETR ResNet backbone:

```text
pixel_values [B,3,H,W]
  -> stem: Conv3x3 s2, Conv3x3 s1, Conv3x3 s1, MaxPool3x3 s2
  -> stage1..stage4 residual blocks
  -> selected feature maps stage2/stage3/stage4, NCHW
```

For a 640x640 input, expected selected feature map sizes are approximately strides 8/16/32: `[B,C2,80,80]`, `[B,C3,40,40]`, `[B,C4,20,20]`. R50/R101 channels are usually `[512,1024,2048]` at these selected stages; R18/R34 use `[128,256,512]`.

Encoder projection and AIFI:

```text
for each backbone feature:
  feature = Conv1x1(C_in -> encoder_hidden_dim, bias=False)
  feature = BatchNorm2d(feature)

for level in encode_proj_layers:
  x = feature.flatten(2).permute(0,2,1)       # [B,Hl*Wl, encoder_hidden_dim]
  pos = sine_2d_position(Hl,Wl,encoder_hidden_dim)
  repeat encoder_layers:
    q,k = Linear(x + pos)
    v = Linear(x)
    x = LayerNorm(x + SelfAttention(q,k,v))
    x = LayerNorm(x + MLP(x))
  feature = x.permute(0,2,1).reshape(B,C,Hl,Wl)
```

Hybrid encoder FPN/PAN:

```text
top_down = [lowest_resolution_feature]
for higher level:
  top = ConvNormAct1x1(top)
  up = nearest_interpolate(top, scale=2)
  fused = concat([up, lateral_feature], channel_dim=1)
  out = CSPRepLayer(fused)

bottom_up = [highest_resolution_fpn]
for lower level:
  down = ConvNormAct3x3_stride2(prev)
  fused = concat([down, next_fpn], channel_dim=1)
  out = CSPRepLayer(fused)
```

Decoder preparation:

```text
for each encoded feature:
  source = Conv1x1(C_encoder -> d_model) + BatchNorm2d
  source_flat = source.flatten(2).transpose(1,2)
source_flatten = cat(source_flat, dim=1)      # [B,S,d_model]
spatial_shapes = [[H0,W0], [H1,W1], [H2,W2]]
level_start_index = cumsum(Hl*Wl)
anchors, valid_mask = generate_anchors(spatial_shapes)
memory = valid_mask.float() * source_flatten
output_memory = LayerNorm(Linear(memory))
enc_logits = Linear(output_memory)            # [B,S,num_labels]
enc_boxes_unact = BoxMLP(output_memory) + anchors
topk_ind = topk(max(enc_logits, class_dim), num_queries)
target = gather(output_memory, topk_ind).detach()
reference_points_unact = gather(enc_boxes_unact, topk_ind).detach()
```

Decoder layer, repeated `decoder_layers` times:

```text
reference_points = sigmoid(reference_points_unact)
query_pos = MLP(reference_points)             # 4 -> 2*d_model -> d_model

x = LayerNorm(x + SelfAttention(q/k from x + query_pos, v from x))
x = LayerNorm(x + DeformableAttention(
      query=x + query_pos,
      memory=source_flatten,
      reference_points=reference_points[:, :, None, :],
      spatial_shapes,
      level_start_index))
x = LayerNorm(x + MLP(x))

box_delta = bbox_embed[layer](x)
reference_points = sigmoid(box_delta + inverse_sigmoid(reference_points)).detach()
logits = class_embed[layer](x)
```

Detection head:

```text
logits = intermediate_logits[:, -1]           # [B,num_queries,num_labels]
pred_boxes = intermediate_reference_points[:, -1]  # [B,num_queries,4] in cxcywh normalized coordinates
```

## 6. Attention requirements

Self-attention:

- Noncausal, no KV cache, encoder/decoder style.
- MHA, not MQA/GQA. `num_heads=8`, `head_dim=32` for the common `d_model=256` and `encoder_hidden_dim=256` configs.
- AIFI attention runs on selected feature levels after flattening NCHW maps to `[B,H*W,C]`.
- Decoder self-attention runs on fixed query sequence length `num_queries=300` in inference.
- Position embeddings are added to hidden states before Q/K projections; values use unpositioned hidden states.
- Eager math is `matmul(q,k^T) * head_dim^-0.5`, additive mask if present, softmax over key dimension, dropout, then `matmul(weights,v)`.
- Source advertises SDPA, FlashAttention, flex attention backend support for this self-attention wrapper, but backend dispatch must preserve noncausal masks and Q/K positional addition.

Multiscale deformable cross-attention:

- Noncausal cross-attention from decoder queries to multi-level encoder memory.
- `value_proj(memory)` -> `[B,S,num_heads,head_dim]`.
- Optional attention mask zeroes invalid memory positions before sampling.
- `sampling_offsets = Linear(query)` -> `[B,Q,Hd,L,P,2]`.
- `attention_weights = softmax(Linear(query), dim=-1)` over `L*P`, then reshape to `[B,Q,Hd,L,P]`.
- If reference points have 2 coordinates, offsets are normalized by `[width,height]` per feature level.
- If reference points have 4 coordinates, offsets are scaled by box width/height and `0.5 / n_points`.
- Fallback implementation splits memory by level, reshapes each level to `[B*heads, head_dim, H, W]`, samples with bilinear `grid_sample(..., align_corners=False)`, weights, sums over levels/points, then applies `output_proj`.
- This is the highest-risk operator for DinoML parity and performance. A custom CUDA deformable-attention kernel should be treated as its own provider family with explicit layout, dtype, and coordinate conventions.

Not applicable: causal masks, decode-time cache shapes, RoPE/ALiBi in attention, sliding-window/local attention, packed varlen sequence metadata, and generation controller logic.

## 7. Position encoding and custom math

2D sine embedding used by AIFI:

```python
def rt_detr_sine_2d(height, width, embed_dim=256, temperature=10000):
    pos_dim = embed_dim // 4
    omega = arange(pos_dim, float64) / pos_dim
    omega = 1.0 / (temperature ** omega)
    grid_h, grid_w = meshgrid(arange(height), arange(width), indexing="ij")
    emb_h = flatten(grid_h).outer(omega)
    emb_w = flatten(grid_w).outer(omega)
    return cat([sin(emb_h), cos(emb_h), sin(emb_w), cos(emb_w)], dim=1)
```

This can be precomputed per `(height,width,embed_dim,dtype)` bucket when `eval_size` or input size is fixed. It depends on actual feature-map dimensions when shapes vary. `embed_dim` must be divisible by 4.

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

Box refinement:

```python
def inverse_sigmoid(x, eps=1e-5):
    x = clamp(x, 0, 1)
    return log(clamp(x, min=eps) / clamp(1 - x, min=eps))

new_ref = sigmoid(box_delta + inverse_sigmoid(old_ref))
```

Frozen BatchNorm2d:

```python
scale = weight.reshape(1,C,1,1) * rsqrt(running_var.reshape(1,C,1,1) + 1e-5)
bias = bias.reshape(1,C,1,1) - running_mean.reshape(1,C,1,1) * scale
y = x * scale + bias
```

For inference, ordinary BatchNorm2d in non-backbone projection/encoder blocks can also be folded into preceding convolutions when weights are fixed and running stats are available.

## 8. Preprocessing and input packing

Processor defaults for inspected PekingU checkpoints:

- `image_processor_type="RTDetrImageProcessor"`.
- `do_resize=True`, `size={"height":640,"width":640}`. This exact-size path does not preserve aspect ratio.
- `do_rescale=True`, `rescale_factor=1/255`.
- `do_normalize=False`, although ImageNet mean/std are present in config.
- `do_pad=False`; if enabled, source pads images to max batch shape or `pad_size` and emits `pixel_mask`.
- `model_input_names=["pixel_values","pixel_mask"]`.
- Output tensor layout is channel-first: `pixel_values [B,3,H,W]`.
- Optional `pixel_mask [B,H,W]` uses 1 for valid pixels and 0 for padding.

Runtime graph input contract for first integration:

- Accept preprocessed `pixel_values` as contiguous NCHW float tensor.
- Accept optional `pixel_mask`; if omitted, synthesize all-ones mask in the runtime or at import boundary.
- Keep preprocessing outside compiled graph initially. It uses image decode/resize and optional annotation logic better suited to CPU/data pipeline.

Postprocessing contract:

- Inputs: `outputs.logits [B,Q,C]`, `outputs.pred_boxes [B,Q,4]` normalized center `cx,cy,w,h`, optional `target_sizes [B,2]` in `(height,width)` order.
- Convert boxes to xyxy normalized coordinates.
- If `target_sizes` is supplied, multiply by `[width,height,width,height]` per image.
- Focal-loss branch: `scores = sigmoid(logits)`, flatten classes and queries, `topk(scores, Q)`, `labels = index % C`, `query_index = index // C`, gather boxes by query index, then threshold.
- Softmax branch: `scores = softmax(logits)[:, :, :-1]`, take max label per query, optional top-k if query dimension exceeds `Q`, then threshold.
- Return a Python-style list of per-image records with variable lengths: `scores`, `labels`, `boxes`.
- Source behavior has no NMS and no mask output.

Training-only preprocessing includes COCO annotation conversion, box conversion to normalized center format, mask resizing/thresholding, and padding annotation updates. These do not need first inference support.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d fold

Source pattern:

```text
Conv2d(..., bias=False or bias=True) -> BatchNorm2d(eps)
```

Replacement:

```text
Conv2d with folded weight and bias
```

Preconditions:

- Inference mode only.
- BatchNorm running mean/variance, gamma, beta are constants.
- No consumer reads the intermediate conv output.
- Preserve `batch_norm_eps`; projection blocks sometimes pass `config.batch_norm_eps`.

Shape equations:

- Input and output shapes are unchanged.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = w * scale[:, None, None, None]
b_fold = beta + (conv_bias - running_mean) * scale
```

Failure cases:

- Training mode, mutable BN stats, or missing BN buffers.

Parity test sketch:

- Compare each ResNet/projection/RepVGG/FPN/PAN block before and after folding on random NCHW tensors at fp32 and fp16 tolerances.

### Rewrite: FrozenBatchNorm2d -> per-channel affine

Source pattern:

```text
RTDetrFrozenBatchNorm2d(x)
```

Replacement:

```text
x * scale[1,C,1,1] + bias[1,C,1,1]
```

Preconditions:

- Frozen BN buffers are constants.
- NCHW channel axis is preserved, or a layout pass rewrites constants and broadcast axes.

Failure cases:

- Accidental NHWC translation without changing broadcast axis.

Parity test sketch:

- Run backbone feature maps through source frozen BN and folded affine for representative channel counts.

### Rewrite: RepVGG inference branch fusion

Source pattern:

```text
Conv3x3+BN branch + Conv1x1+BN branch -> SiLU
```

Replacement:

```text
single Conv3x3 with padded 1x1 kernel folded into center -> SiLU
```

Preconditions:

- Both branches have stride 1, same input/output channels, groups 1, same spatial shape.
- Both BN layers are foldable.
- Padding conventions match: 3x3 branch padding 1 and 1x1 branch padding 0.

Weight transform:

```python
w3, b3 = fold_bn(conv3, bn3)
w1, b1 = fold_bn(conv1, bn1)
w1_padded = pad_1x1_to_center_3x3(w1)
w = w3 + w1_padded
b = b3 + b1
```

Failure cases:

- Non-unit stride, grouped convolution, unfrozen BN, or consumer reads branch output.

Parity test sketch:

- Compare every `RTDetrRepVggBlock` output before and after fusion at R18 and R50 channel widths.

### Rewrite: 1x1 Conv2d on NCHW -> batched Linear/GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0) on [B,Cin,H,W]
```

Replacement:

```text
transpose/flatten to [B*H*W,Cin] -> GEMM(weight.T) -> reshape
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Layout pass either preserves NCHW or controls the flatten/transpose order.
- Weight is stored as `[Cout,Cin,1,1]`.

Shape equations:

- GEMM `M=B*H*W`, `K=Cin`, `N=Cout`.
- Output reshapes to `[B,Cout,H,W]`.

Failure cases:

- 3x3 convs, stride-2 extra levels, grouped convs, or unguarded NHWC/NCHW mismatch.

Parity test sketch:

- Test encoder and decoder projection layers on dynamic batch and fixed 640-derived H/W.

### Rewrite: AIFI flatten-attention-unflatten layout guard

Source pattern:

```text
NCHW feature -> flatten(2).permute(0,2,1) -> self-attention/MLP -> permute/reshape NCHW
```

Replacement:

```text
layout-aware token view over spatial rows -> attention -> layout-aware restore
```

Preconditions:

- Spatial token order remains row-major `(h,w)` from NCHW.
- 2D sine embedding uses the same flattened order.
- Downstream FPN/PAN receives NCHW, or all consumers are inside an NHWC-controlled region.

Failure cases:

- A layout pass changes spatial/channel order without rewriting sine embedding and reshape.

Parity test sketch:

- Compare AIFI-only level output for `[B,C,20,20]` and odd feature sizes.

### Rewrite: Deformable attention custom CUDA provider

Source pattern:

```text
value split by level -> reshape to [B*heads,head_dim,H,W]
sampling_locations -> grid_sample bilinear per level
weighted sum over levels and points -> output projection
```

Replacement:

```text
fused multiscale deformable attention kernel
```

Preconditions:

- Coordinate convention exactly matches `sampling_grids = 2 * sampling_locations - 1`.
- Bilinear sampling uses `align_corners=False` and zero padding.
- Input memory level flatten order matches `spatial_shapes`.
- Dtype and accumulation policy are explicit.
- `num_heads`, `num_levels`, and `num_points` are bounded or profiled.

Failure cases:

- NHWC/NCHW mismatch, different `align_corners`, wrong level start index, unsupported dynamic feature sizes, or invalid reference point coordinate mode.

Parity test sketch:

- Random small tensors against source fallback for both 2-coordinate and 4-coordinate reference points, multiple feature levels, fp32 and fp16.

### Rewrite: Postprocess top-k detector head

Source pattern:

```text
sigmoid(logits).flatten(1) -> topk(Q) -> modulo/divide label/query index -> gather boxes -> threshold
```

Replacement:

```text
fused score top-k + box gather, optional CPU variable-output materialization
```

Preconditions:

- `use_focal_loss=True`.
- `Q` and `C` known; official configs use `Q=300`, `C=80`.
- Output may remain padded fixed-size before Python thresholding for first GPU integration.

Failure cases:

- Softmax/no-object branch, custom class counts, or product requires NMS.

Parity test sketch:

- Compare score/label/box ordering before threshold for random logits and boxes.

## 10. Kernel fusion candidates

Highest priority:

- Deformable attention CUDA kernel/provider. The fallback uses many reshapes, per-level `grid_sample`, stack, multiply, sum, and transpose operations inside every decoder layer; this is the main unique performance-critical operator.
- Conv2d + BatchNorm folding across backbone/projection/FPN/PAN. The model is convolution-heavy before the decoder, and folding removes many per-channel normalization launches.
- 1x1 Conv2d lowering to GEMM/CUTLASS. Encoder/decoder projections and many bottleneck convolutions are 1x1-heavy.
- Fused self-attention for AIFI and decoder queries. Query length 300 and AIFI level around 20x20 for 640 input are small enough that launch overhead and layout movement matter.

Medium priority:

- RepVGG branch fusion in `RTDetrRepVggBlock`.
- CSPRepLayer fusion patterns around split `1x1` convs, branch add, and final projection.
- Proposal top-k + gather fusion over encoder memory.
- Box MLP + inverse-sigmoid + sigmoid refinement fusion in decoder loop.
- Sine position and anchor precompute for fixed 640x640/eval buckets.

Lower priority:

- Postprocess top-k/threshold variable-output kernel. It matters for end-to-end latency but can initially run on CPU/Python.
- SiLU/ReLU elementwise fusion around conv blocks when not absorbed by a library epilogue.
- Dynamic mask interpolation optimization, because default official inference often has no padding mask.

## 11. Runtime staging plan

Stage 1: config and processor handoff.

- Parse `RTDetrConfig` plus nested `RTDetrResNetConfig`.
- Load official R18/R50/R101 weights.
- Accept preprocessed NCHW `pixel_values`; synthesize all-ones `pixel_mask` if absent.
- Keep image resize/rescale and postprocess in Python.

Stage 2: backbone parity.

- Implement RT-DETR ResNet stem/stages in source-faithful NCHW.
- Fold frozen BN and normal BN only after unfused parity passes.
- Validate selected feature maps `stage2/stage3/stage4`.

Stage 3: hybrid encoder parity.

- Add encoder projection convs, AIFI on `encode_proj_layers`, and FPN/PAN conv fusion.
- Validate feature-level outputs for R18 and R50, then R101 width 384.

Stage 4: decoder preparation parity.

- Implement decoder input projections, flatten/concat, `spatial_shapes`, `level_start_index`, anchor generation, valid mask, encoder proposal heads, top-k, and gathers.
- Validate `enc_topk_logits`, `enc_topk_bboxes`, and decoder initial target/reference tensors.

Stage 5: one decoder layer parity.

- Implement decoder query self-attention, query position MLP, multiscale deformable attention fallback or provider, FFN, and box refinement.
- Validate one-layer and then full `decoder_layers` outputs.

Stage 6: raw detection head parity.

- Return `logits [B,300,80]` and `pred_boxes [B,300,4]`.
- Add intermediate output checks only if needed for debugging or auxiliary-loss compatibility.

Stage 7: optimization pass.

- Enable BN folding, 1x1 conv GEMM lowering, RepVGG fusion, precomputed sine/anchors, and deformable attention provider.
- Add guarded NHWC/channel-last regions only after NCHW parity is stable.

Stage 8: postprocess integration.

- Start with CPU postprocess using DinoML outputs.
- Later add fixed-size GPU top-k/gather path and optional variable-output materialization.

## 12. Parity and validation plan

- Config load tests for R18/R34/R50/R101, verifying nested backbone defaults and source-supplied omitted fields.
- Processor contract tests: 640x640 resize/rescale/no-normalize output, optional pad/mask behavior, and target-size box scaling order.
- Random op tests:
  - Frozen BN affine vs source.
  - `build_2d_sinusoidal_position_embedding` for odd/even H/W and dtype.
  - `generate_anchors` and valid mask for fixed spatial shapes.
  - `inverse_sigmoid` and box refinement near eps boundaries.
  - Multiscale deformable attention fallback/provider for 2D and 4D reference points.
- Backbone parity: compare selected feature maps after stem/stage outputs for R18 and R50 on small random images.
- Hybrid encoder parity: compare projected features, AIFI level, FPN outputs, and PAN outputs.
- Decoder prep parity: compare flattened memory, spatial metadata, top-k indices, gathered reference points, and initial target.
- Single decoder-layer parity, then full decoder parity for R18 3-layer and R50 6-layer variants.
- End-to-end raw output parity against Transformers for `PekingU/rtdetr_r50vd` on one 640x640 image.
- Postprocess parity: compare final `scores`, `labels`, and `boxes` before/after threshold for focal-loss branch; include no-NMS expectation.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for most blocks; fp16/bf16 `rtol=1e-2, atol=1e-2`, with looser checks around deformable bilinear sampling and deep conv stacks.

## 13. Performance probes

- CPU preprocessing throughput: decode + resize + rescale to 640x640.
- Backbone-only throughput by variant: R18/R34/R50/R101 and batch sizes 1/4/8/16.
- Hybrid encoder throughput split into AIFI, FPN, and PAN.
- Decoder preparation throughput: projections, flatten/concat, anchor generation, top-k, gather.
- Decoder-layer throughput split into self-attention, deformable attention, FFN, and box heads.
- Deformable attention backend comparison: source-like fallback vs fused provider, fp32/fp16, batch and query sweeps.
- End-to-end raw model latency and throughput for 640x640, excluding postprocess.
- Postprocess latency for sigmoid top-k/gather/threshold on CPU vs GPU.
- Memory probes: peak activation memory for R50 vs R101, and per-stage temporary allocations for deformable attention.
- Layout probes: NCHW baseline vs guarded NHWC/channel-last conv regions, including transpose overhead and deformable-attention guard costs.

## 14. Skip/defer list

- Training losses, Hungarian matching, denoising query generation, random label/box noise, and auxiliary supervision.
- Annotation preprocessing and segmentation-mask annotation support.
- Remote custom kernels from third-party hubs; first parity should use in-library source semantics.
- NMS, because source postprocess does not perform it.
- Multi-GPU tensor parallelism and distributed execution.
- General dynamic-resolution production scheduler; start with fixed 640x640 buckets.
- `use_focal_loss=False` postprocess branch can be second-wave unless a target checkpoint requires it.
- Learned initial query branch can be deferred until a checkpoint with `learn_initial_query=True` is targeted.
- `anchor_image_size` cached-anchor path can be deferred; inspected configs use dynamic generation.

## 15. Final implementation checklist

- [ ] Parse `RTDetrConfig` and nested `RTDetrResNetConfig`.
- [ ] Load PekingU checkpoint weights with nested backbone defaults.
- [ ] Accept/source `pixel_values [B,3,H,W]` and optional `pixel_mask [B,H,W]`.
- [ ] Implement RT-DETR ResNet stem, basic blocks, bottleneck blocks, pooling, and selected outputs.
- [ ] Implement/fold FrozenBatchNorm2d and inference BatchNorm2d.
- [ ] Implement encoder input projections.
- [ ] Implement 2D sine position embedding.
- [ ] Implement AIFI encoder layer.
- [ ] Implement FPN/PAN ConvNorm/CSPRep/RepVGG blocks.
- [ ] Implement decoder input projections and multi-level flatten metadata.
- [ ] Implement anchor generation and valid mask.
- [ ] Implement encoder proposal heads, `topk`, and `gather`.
- [ ] Implement decoder self-attention.
- [ ] Implement multiscale deformable attention fallback and plan fused provider.
- [ ] Implement query position MLP, box MLP, and iterative refinement.
- [ ] Implement class logits and final box outputs.
- [ ] Implement CPU/Python postprocess parity for sigmoid focal-loss branch.
- [ ] Add single-op parity tests for custom math and deformable attention.
- [ ] Add backbone, encoder, one-decoder-layer, full-model, and postprocess parity tests.
- [ ] Benchmark backbone, hybrid encoder, decoder, deformable attention, and postprocess separately.
