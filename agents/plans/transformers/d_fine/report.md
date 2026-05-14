# D-FINE Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: d_fine family; representative public checkpoints from ustc-community
Config source: local Transformers defaults plus fetched HF config/preprocessor JSON snapshots
Source files inspected:
- X:/H/transformers/src/transformers/models/d_fine/configuration_d_fine.py
- X:/H/transformers/src/transformers/models/d_fine/modeling_d_fine.py
- X:/H/transformers/src/transformers/models/d_fine/convert_d_fine_original_pytorch_checkpoint_to_hf.py
- X:/H/transformers/src/transformers/models/rt_detr/image_processing_rt_detr.py
- X:/H/transformers/src/transformers/backbone_utils.py
- X:/H/transformers/src/transformers/models/hgnet_v2/configuration_hgnet_v2.py
- X:/H/transformers/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py
Any missing files or assumptions: D-FINE has no local image processor; it uses RTDetrImageProcessor. PekingU/DFine_r50vd returned 401 in this environment.
```

Snapshots are under `_sources/`, including source copies and fetched configs for:

- [ustc-community/dfine-nano-coco](https://huggingface.co/ustc-community/dfine-nano-coco)
- [ustc-community/dfine-small-coco](https://huggingface.co/ustc-community/dfine-small-coco)
- [ustc-community/dfine-medium-coco](https://huggingface.co/ustc-community/dfine-medium-coco)
- [ustc-community/dfine-large-coco](https://huggingface.co/ustc-community/dfine-large-coco)
- [ustc-community/dfine-xlarge-coco](https://huggingface.co/ustc-community/dfine-xlarge-coco)
- [ustc-community/dfine-small-obj365](https://huggingface.co/ustc-community/dfine-small-obj365)

Gated/unavailable: [PekingU/DFine_r50vd](https://huggingface.co/PekingU/DFine_r50vd) returned 401 for config, preprocessor, and model API. Access would confirm its exact backbone/config lineage; report facts below use source defaults and reachable official `ustc-community` configs.

## 2. High-level architecture

D-FINE is an object-detection model: image preprocessing -> convolutional backbone -> hybrid feature encoder -> query decoder -> class logits and normalized boxes -> RT-DETR postprocessing. It is not an autoregressive generation model and has no KV cache.

```text
image(s) -> RTDetrImageProcessor -> pixel_values [B,3,H,W], optional pixel_mask [B,H,W]
  -> AutoBackbone feature maps (usually HGNetV2)
  -> 1x1 projections -> D-FINE hybrid encoder (AIFI + FPN + PAN)
  -> flatten multi-scale maps, generate anchors, top-k encoder proposals
  -> decoder queries with self-attention + multi-scale deformable cross-attention
  -> fine-grained distribution box refinement + class logits
  -> post_process_object_detection: cxcywh -> xyxy, scale, top-k, threshold, no NMS
```

Backbone ownership: D-FINE calls `load_backbone(config)` and consumes an `AutoBackbone` `feature_maps` tuple. The reachable configs use `backbone_config.model_type = "hgnet_v2"`. DinoML should compose HGNetV2 as a separately audited backbone body, while this report owns the D-FINE projections, hybrid encoder, decoder, box refinement, and postprocess.

Stageable boundaries:

- CPU/data pipeline: resize/rescale/pad/mask and target-size metadata.
- Backbone feature extraction: independently cacheable only for fixed image tensors.
- Hybrid encoder: image-like NCHW feature maps, no generation state.
- Decoder/head: query-driven noncausal detection decoder.
- Postprocess: required for end-to-end object-detection parity.

## 3. Important config dimensions

Source defaults from `DFineConfig`:

| Field | Default / source behavior |
| --- | --- |
| `model_type` | `d_fine` |
| `backbone_config` | default `hgnet_v2`, `out_indices=[2,3,4]` |
| `d_model` / `hidden_size` | 256 |
| `encoder_hidden_dim` | 256 |
| `encoder_in_channels` | `[512, 1024, 2048]` |
| `feat_strides` | `[8, 16, 32]` |
| `encoder_layers` | 1 AIFI layer per selected level |
| `encoder_attention_heads` | 8 |
| `encoder_ffn_dim` | 1024 |
| `encode_proj_layers` | `[2]` |
| `decoder_layers` | 6 |
| `decoder_attention_heads` | 8 |
| `head_dim` | `d_model // decoder_attention_heads` |
| `decoder_n_points` | 4 or list per feature level; source validates list length |
| `num_feature_levels` | 3 |
| `num_queries` | 300 |
| `max_num_bins` | 32, so bbox head outputs `4 * 33 = 132` bins |
| `reg_scale`, `up` | 4.0, 0.5 for distribution-to-distance weighting |
| `decoder_method` | `"default"` bilinear `grid_sample`; `"discrete"` gather path also implemented |
| `num_denoising` | 100, training-only when labels are provided |
| `use_focal_loss` | true; postprocess uses sigmoid multi-label top-k by default |
| `eval_idx` | -1, last decoder layer output for inference |

Representative config sweep from fetched official configs:

| Checkpoint | Backbone | `d_model` | Enc dim | Levels | Strides | Dec layers | Points | Enc in channels | Notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `dfine-nano-coco` | HGNetV2 | 128 | 128 | 2 | 16,32 | 3 | 6,6 | 512,1024 | `out_indices=[3,4]`, smaller decoder/feedforward dims |
| `dfine-small-coco` | HGNetV2 | 256 | 256 | 3 | 8,16,32 | 3 | 3,6,3 | 256,512,1024 | default-sized D-FINE head with shallow decoder |
| `dfine-medium-coco` | HGNetV2 | 256 | 256 | 3 | 8,16,32 | 4 | 3,6,3 | 384,768,1536 | wider backbone, `depth_mult=0.67` |
| `dfine-large-coco` | HGNetV2 | 256 | 256 | 3 | 8,16,32 | 6 | 3,6,3 | 512,1024,2048 | common production-like COCO config |
| `dfine-xlarge-coco` | HGNetV2 | 256 | 384 | 3 | 8,16,32 | 6 | 3,6,3 | 512,1024,2048 | encoder/decoder feature channels are 384 before decoder projection |
| `dfine-small-obj365` | HGNetV2 | 256 | 256 | 3 | 8,16,32 | 3 | 3,6,3 | 256,512,1024 | same structure as small, larger label map in config metadata |

All fetched preprocessors use `RTDetrImageProcessor`, resize to 640x640, rescale, `do_normalize=false`, `do_pad=false`.

## 3a. Family variation traps

- `backbone_config` controls a delegated AutoBackbone. Do not assume ResNet; reachable official configs use HGNetV2.
- Nano uses only 2 feature levels and `d_model=128`; most other variants use 3 levels and `d_model=256`.
- XLarge uses `encoder_hidden_dim=384` and `decoder_in_channels=[384,384,384]`, then projects to `d_model=256`.
- `decoder_n_points` can be an int or per-level list. The public configs use `[3,6,3]` or `[6,6]`, not just the default scalar 4.
- `decoder_method="discrete"` switches deformable cross-attention from bilinear `grid_sample` to integer clamped gather. Default configs use `"default"` unless changed.
- `num_denoising`, Hungarian matching, auxiliary losses, and many bbox loss fields are training-only for inference.
- `learn_initial_query=false` means inference queries are gathered from top-k encoder proposal features, not a learned query table.
- Source supports SDPA/Flash/Flex only for ordinary self-attention. Multi-scale deformable cross-attention remains a custom sampling op.
- Modeling is NCHW for images/features. NHWC is an optimization opportunity only inside controlled conv/projection regions; flatten/concat/order and grid sampling are axis-sensitive.
- Postprocess has no NMS. Adding NMS would change source parity.
- Weight tying keys intentionally alias `class_embed` and `bbox_embed` clones into decoder fields; preserve logical sharing.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors, feature-map tuples, `flatten(2)`, `transpose(1,2)`, `permute`, `reshape/view`, `contiguous`.
- Multi-level concat over sequence dimension and channel concat in FPN/PAN.
- Split/chunk for gate and CSP blocks.
- `gather` and `topk` for encoder proposal selection.
- Boolean masks, masked fill/multiply, `where`, repeats/tiles.
- `interpolate` nearest for FPN upsample and mask downsample.

Neural primitives:

- Conv2d 1x1 and 3x3, stride 1/2, padding 0/1, depthwise grouped conv in `DFineSCDown`.
- BatchNorm2d and FrozenBatchNorm2d affine fold opportunity.
- Linear, MLP with ReLU/GELU/SiLU, LayerNorm, dropout disabled in eval.
- Sigmoid gate: `LayerNorm(g1 * residual + g2 * cross_attn)`.

Attention primitives:

- Noncausal MHA self-attention in AIFI and decoder query self-attention.
- Query/key positional embeddings are added before Q/K projection; values use unpositioned hidden states.
- Custom multi-scale deformable cross-attention over flattened image features.

Position/custom math:

- 2D sinusoidal embedding `[sin_y, cos_y, sin_x, cos_x]` for AIFI.
- Anchor grid generation in normalized `cxcywh`, logit transform, valid mask.
- Fine-grained distribution refinement: softmax over bins, linear with nonuniform bin weights, distance-to-box update.
- Location Quality Estimator: per-corner distribution top-k, mean, MLP score offset.

Pre/postprocessing:

- RTDetrImageProcessor resize/rescale/pad/mask; model expects `[B,3,H,W]`.
- Postprocess converts normalized `cxcywh` to `xyxy`, scales by target sizes, sigmoid/softmax scoring, top-k, threshold. No NMS.

## 5. Layer/block breakdown

Backbone boundary:

```text
pixel_values [B,3,H,W]
  -> AutoBackbone(HGNetV2 in reachable configs)
  -> feature_maps selected by out_indices:
     nano: [B,512,H/16,W/16], [B,1024,H/32,W/32]
     small/large: [B,C8,H/8,W/8], [B,C16,H/16,W/16], [B,C32,H/32,W/32]
```

Hybrid encoder:

```text
for each backbone feature:
  source = Conv1x1(in_channels -> encoder_hidden_dim, bias=False) -> BatchNorm

for selected encode_proj_layers:
  x [B,C,H,W] -> flatten to [B,HW,C]
  pos = 2D sine embedding [1,HW,C] when training or eval_size is None
  repeat encoder_layers:
    q,k = Linear(x + pos), v = Linear(x)
    x = LayerNorm(residual + MHA(q,k,v))
    x = LayerNorm(residual + MLP(x))
  reshape back to [B,C,H,W]

top-down FPN:
  lateral Conv1x1 -> nearest upsample x2 -> concat channels -> RepNCSPELAN4

bottom-up PAN:
  depthwise/pointwise downsample -> concat with FPN level -> RepNCSPELAN4
```

Decoder preparation:

```text
sources = decoder_input_proj(each encoded map) and optional stride-2 extra levels
source_flatten = concat([B,Hl*Wl,d_model] per level)
anchors, valid_mask = generate_anchors(spatial_shapes)
memory = valid_mask * source_flatten
output_memory = Linear(d_model,d_model) -> LayerNorm
enc_class = Linear(d_model,num_labels)
enc_box_logits = MLP(d_model,d_model,4) + anchors
topk = topk(max_class_score, num_queries)
reference_points_unact = gather(enc_box_logits, topk)
target = gather(output_memory, topk).detach() unless learn_initial_query
```

Decoder layer, repeated `decoder_layers` plus extra clone logic after `eval_idx`:

```text
ref = sigmoid(reference_points)
query_pos = MLP(4 -> 2*d_model -> d_model)(ref).clamp(-10, 10)
x = LayerNorm(x + SelfAttention(q/k from x + query_pos, v from x))
cross = MSDeformableAttention(x + query_pos, source_flatten, ref)
x = LayerNorm(gate(residual, cross))
x = LayerNorm((x + MLP(x)).clamp(-65504, 65504))
pred_corners = bbox_embed[i](x + previous_x_detached) + previous_corners
box = distance2bbox(initial_ref, Integral(softmax(pred_corners), project), reg_scale)
score = class_embed[i](x); score += LQE(score, pred_corners)
```

Detection head:

- `class_embed[i]`: `Linear(d_model -> num_labels)`.
- `bbox_embed[i]`: `DFineMLP(hidden -> hidden -> hidden -> 4*(max_num_bins+1))`.
- Inference output uses `outputs.intermediate_logits[:, -1]` and `outputs.intermediate_reference_points[:, -1]`.

## 6. Attention requirements

Ordinary self-attention:

- Noncausal, no KV cache, no autoregressive mask.
- MHA with `decoder_attention_heads` or `encoder_attention_heads`; no GQA/MQA.
- Q/K/V widths equal hidden size; head dim must divide hidden size.
- Mask in decoder self-attention is only used for training denoising attention separation. In normal inference it is `None`.
- Source can route through eager/SDPA/Flash/Flex interfaces for this attention family.

Multi-scale deformable cross-attention:

- Query source: decoder object queries from top-k encoder proposals or learned table.
- Key/value source: concatenated flattened multi-scale feature maps.
- `value` reshaped to `[B, sequence_length, heads, d_model/heads]`.
- `spatial_shapes`: `[num_levels,2]` in `(height,width)` order.
- `sampling_offsets`: `Linear(d_model -> heads * sum(points_per_level) * 2)`.
- `attention_weights`: `Linear(d_model -> heads * sum(points_per_level))`, softmax over all level-points.
- Reference-point inference path uses shape `[B,num_queries,1,4]`; offsets are scaled by per-level `1/n_points`, reference width/height, and `decoder_offset_scale`.
- Default backend uses `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)` with normalized grid `2*location-1`.
- Discrete method uses clamped integer coordinate gather. Treat as a separate admission path.

No KV cache, packed varlen, causal decode, sliding window, RoPE, ALiBi, or relative-bias attention is required.

## 7. Position encoding and custom math

2D sine embedding:

```python
omega = 1.0 / temperature ** (arange(embed_dim // 4) / (embed_dim // 4))
grid_y, grid_x = meshgrid(arange(H), arange(W), indexing="ij")
pos = cat([sin(grid_y*omega), cos(grid_y*omega),
           sin(grid_x*omega), cos(grid_x*omega)], dim=-1)
```

Anchor generation:

```python
grid_xy = (meshgrid_xy + 0.5) / [width, height]
wh = grid_size * (2.0 ** level)
anchor = [cx, cy, w, h]
valid = all(anchor > 1e-2 and anchor < 1 - 1e-2)
anchor_logit = log(anchor / (1 - anchor))
anchor_logit = where(valid, anchor_logit, finfo(dtype).max)
```

Distribution refinement:

```python
project = weighting_function(max_num_bins, up, reg_scale)
prob = softmax(pred_corners.reshape(-1, max_num_bins + 1), dim=1)
dist = linear(prob, project).reshape(B, Q, 4)
box = distance2bbox(initial_ref_cxcywh, dist, reg_scale)
```

`weighting_function` is nonuniform and source-specific; do not replace with uniform DFL bin indices without parity tests.

## 8. Preprocessing and input packing

The processor is `RTDetrImageProcessor`, inherited by D-FINE examples. Fetched configs set:

- `size={"height":640,"width":640}`
- `do_resize=true`
- `do_rescale=true`
- `do_normalize=false`
- `do_pad=false`
- model inputs: `pixel_values`, optional `pixel_mask`

Runtime graph contract:

- `pixel_values`: `[B,3,H,W]`, channel-first, float after rescale.
- If no `pixel_mask` is passed, modeling creates all-ones `[B,H,W]`.
- If processor padding is enabled by a caller, `pixel_mask` marks valid pixels as 1 and padding as 0; D-FINE downsamples it to backbone feature map shapes.
- Detection postprocess needs original `target_sizes` `[B,2]` as `(height,width)` to scale boxes.

Postprocess details:

- Model outputs `pred_boxes` in normalized center format `[cx,cy,w,h]`.
- Convert to corner `[xmin,ymin,xmax,ymax]`, multiply by `[width,height,width,height]`.
- Focal-loss path: sigmoid logits, flatten `[queries * classes]`, top-k with `k=num_queries`, label=`index % num_classes`, query=`index // num_classes`, gather boxes.
- Non-focal path: softmax logits excluding final no-object class.
- Apply threshold per image.
- No NMS, box clipping, or class-wise suppression is performed by source postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d projection -> per-pixel Linear

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `groups=1`, static channel count.
- Consumer immediately flattens/permutes to `[B,H*W,C]` or remains local to an NCHW block.

Replacement:

```text
NCHW feature -> optional NHWC local view -> Linear(Cin -> Cout) -> restore expected layout
```

Weight transform: conv weight `[Cout,Cin,1,1]` to linear `[Cout,Cin]`. Preserve BatchNorm folding if eval.

Failure cases: layout escapes to backbone/FPN/PAN consumers, dynamic non-contiguous views, grouped conv.

### Rewrite: FrozenBatchNorm/BatchNorm eval fold

Preconditions: inference mode, frozen or eval BatchNorm buffers available.

Replacement: fold scale/bias into preceding Conv2d or Linear-equivalent projection.

Failure cases: training mode, missing running stats, user explicitly requires mutable BN.

### Rewrite: Deformable attention custom op

Source pattern:

```text
Linear offsets + Linear weights -> per-level grid_sample/gather -> weighted sum
```

Replacement: one provider op with explicit `spatial_shapes`, per-level points, `align_corners=false`, and method enum.

Preconditions: `decoder_method` admitted; reference shape last dim is 4 for current inference path; value layout and level order match source flatten order.

Failure cases: method `"discrete"` without gather kernel, reference last dim 2 path untested for D-FINE inference, NHWC translation across grid coordinates.

### Rewrite: postprocess top-k fusion

Preconditions: focal-loss path, fixed `num_queries`, no NMS, threshold applied after top-k.

Replacement:

```text
sigmoid logits -> flattened topk -> label/query decode -> gather boxes -> threshold
```

Failure cases: `use_focal_loss=false`, caller expects all query/class scores, or class-specific postprocessing is added externally.

## 10. Kernel fusion candidates

Highest priority:

- Multi-scale deformable attention kernel. The eager loop over levels plus `grid_sample` is the distinctive runtime cost and parity risk.
- Conv+BN(+SiLU/ReLU) blocks for HGNetV2 and D-FINE FPN/PAN projections.
- Top-k proposal selection and gather. This is in the inference path and shapes the decoder input.
- Distribution refinement/LQE fusion: bbox MLP output -> softmax bins -> weighted integral -> distance2bbox, plus top-k probability statistics.

Medium priority:

- Noncausal MHA for AIFI/decoder self-attention via existing dense attention provider.
- 2D sine embedding/anchor precompute for fixed `eval_size` or static image shapes.
- FPN/PAN concat + conv scheduling, including nearest upsample plus concat.

Lower priority:

- Training denoising query construction and loss/matcher paths.
- `decoder_method="discrete"` unless configs requiring it appear.
- Auxiliary outputs and intermediate-layer losses.

## 11. Runtime staging plan

1. Parse `DFineConfig` and fetched processor config; reject unsupported backbones except audited `hgnet_v2`.
2. Compose HGNetV2 backbone as an external audited stage; validate selected feature-map shapes/channels.
3. Implement D-FINE projections + hybrid encoder for one reachable config, starting with `dfine-small-coco`.
4. Implement proposal anchor generation, `topk`, query gather, and decoder without training denoising.
5. Implement multi-scale deformable cross-attention eager/reference first, then provider kernel.
6. Implement distribution refinement, LQE, class/bbox heads, and no-NMS postprocess.
7. Add variant coverage for nano two-level path and xlarge 384-channel encoder path.

Stub initially: losses, matcher, denoising, auxiliary loss outputs, Obj365 label metadata beyond class-count/label-map handling.

## 12. Parity and validation plan

- Unit parity for `build_2d_sinusoidal_position_embedding`, `generate_anchors`, `weighting_function`, `DFineIntegral`, `distance2bbox`, and LQE.
- Random tensor parity for `multi_scale_deformable_attention_v2` default mode against PyTorch `grid_sample`; include 2-level and 3-level point lists.
- Single-block parity for `DFineDecoderLayer` with fixed `spatial_shapes` and no denoising mask.
- Hybrid encoder parity after AIFI only, then after FPN/PAN for fixed random feature maps.
- End-to-end model parity for `dfine-small-coco` on one 640x640 image through logits/boxes before postprocess.
- Postprocess parity: sigmoid top-k, label/query decode, target-size scaling, threshold, and explicit no-NMS behavior.
- Suggested tolerances: fp32 `1e-4` to `5e-4` for most ops; deformable attention/provider fp16 likely needs `1e-2` class/box tolerance after decoder due sampling and softmax sensitivity.

## 13. Performance probes

- Processor throughput: resize/rescale/pad to 640x640, batch-size sweep.
- Backbone-only throughput for HGNetV2 selected variants.
- Hybrid encoder throughput separated into AIFI, top-down FPN, bottom-up PAN.
- Decoder-only throughput with precomputed `source_flatten`, sweep `num_queries`, feature levels, and `decoder_n_points`.
- Multi-scale deformable attention backend comparison: PyTorch `grid_sample` fallback vs DinoML provider.
- Top-k/gather latency and memory bandwidth for encoder proposals.
- Postprocess latency with batch-size and class-count sweeps, especially Obj365.
- Layout probe: NCHW baseline vs guarded channel-last conv/projection regions without crossing grid-sample or flatten order boundaries.

## 14. Skip/defer list

- Training losses, Hungarian matcher, denoising query construction, auxiliary loss returns.
- Gated/unavailable PekingU checkpoint specifics until access is available.
- Remote-code/non-library configs with `model_type` not equal to current in-library `d_fine`.
- `decoder_method="discrete"` optimized kernel unless a representative official config requires it.
- NMS, because source postprocess intentionally does not apply it.
- General AutoBackbone admission beyond audited HGNetV2.
- Quantization/packed weights; current source has no D-FINE-specific quantized weight path.

## 15. Final implementation checklist

- [ ] Parse `DFineConfig`, including backbone sub-config and per-level `decoder_n_points`.
- [ ] Admit audited HGNetV2 backbone configs and reject unsupported AutoBackbone bodies.
- [ ] Load D-FINE weights with class/bbox embed alias preservation.
- [ ] Implement NCHW Conv2d/BatchNorm/activation projections and FPN/PAN blocks.
- [ ] Implement AIFI noncausal self-attention with 2D sine embeddings.
- [ ] Implement anchor generation, valid masks, encoder proposal top-k, and gathers.
- [ ] Implement decoder self-attention and gated deformable cross-attention block.
- [ ] Implement multi-scale deformable attention provider with default bilinear grid sampling semantics.
- [ ] Implement D-FINE distribution bbox refinement and LQE score correction.
- [ ] Implement RT-DETR/D-FINE postprocess with no NMS.
- [ ] Add parity tests for small, nano, and xlarge config shape variants.
- [ ] Benchmark backbone, hybrid encoder, decoder, deformable attention, and postprocess separately.
