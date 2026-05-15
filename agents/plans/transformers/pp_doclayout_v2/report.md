# Transformers audit: pp_doclayout_v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-DocLayoutV2_safetensors
Config source: HF config.json at repo SHA 880e8971b88938518611c54fc0f59ad57849c9d4, plus source defaults
Source files inspected:
  transformers/src/transformers/models/pp_doclayout_v2/configuration_pp_doclayout_v2.py
  transformers/src/transformers/models/pp_doclayout_v2/image_processing_pp_doclayout_v2.py
  transformers/src/transformers/models/pp_doclayout_v2/modeling_pp_doclayout_v2.py
  transformers/src/transformers/models/pp_doclayout_v2/modular_pp_doclayout_v2.py
  transformers/src/transformers/models/hgnet_v2/configuration_hgnet_v2.py
  transformers/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py
  transformers/src/transformers/integrations/hub_kernels.py
Any missing files or assumptions: only one native pp_doclayout_v2 checkpoint/config was found; no gated source. Generated pp_doclayout_v2 files are exact runtime source, while modular_pp_doclayout_v2.py is the authoritative future edit source.
```

Primary DinoML target for this report: inference-only document layout object detection plus reading-order output, CUDA GPU first, faithful NCHW source semantics first. NHWC/channel-last is only a guarded optimization for local conv/image regions.

## 2. High-level architecture

PP-DocLayoutV2 is a structured document-layout detector, not an autoregressive language model.

```text
image preprocessing -> HGNetV2 NCHW conv backbone -> 1x1 feature projections
  -> RT-DETR-style hybrid encoder (AIFI + FPN + PAN)
  -> top-k encoder proposals / anchor boxes
  -> query decoder with dense self-attention + multiscale deformable cross-attention
  -> class logits + normalized boxes
  -> threshold/sort/remap boxes into layout elements
  -> LayoutLM-like reading-order encoder + GlobalPointer
  -> postprocess scores, labels, boxes, order_seq
```

Stage decomposition:

- CPU/data pipeline: image loading, RGB conversion if needed by generic image backend, resize to 800x800, rescale, normalize, optional padding/batching.
- Detection neural graph: HGNetV2 feature maps, hybrid encoder, proposal selection, decoder, class/box heads.
- Reading-order neural graph: uses detected boxes scaled/clamped to `[0,1000]`, remapped class ids, and valid mask; it is independently stageable once detection outputs are available.
- Postprocess: box format conversion, target-size scaling, class top-k, score thresholds, reading-order vote sort. No NMS is implemented in the inspected source.

## 3. Important config dimensions

| Field | Checkpoint value | Source / notes |
| --- | ---: | --- |
| task / architecture | `PPDocLayoutV2ForObjectDetection` | HF `config.json` |
| dtype | `float32` | HF `torch_dtype` |
| input processor size | `800x800` | HF `preprocessor_config.json` |
| backbone | `hgnet_v2`, `arch=L` | nested `backbone_config` |
| backbone outputs | `stage2, stage3, stage4` | `return_idx=[1,2,3]` |
| backbone feature channels consumed | `[512,1024,2048]` | detection config |
| feature strides | `[8,16,32]` | detection config |
| encoder hidden dim / heads | `256 / 8` | AIFI encoder |
| encoder layers | `1` on `encode_proj_layers=[2]` | only deepest projected level gets AIFI |
| encoder FFN dim / activation | `1024 / gelu` | detection config |
| hybrid FPN/PAN activation | `silu` | `activation_function` |
| decoder layers / heads | `6 / 8` | detection config |
| decoder head dim | `32` | inference: `d_model / decoder_attention_heads` |
| deformable levels / points | `3 / 4` | detection config |
| object queries | `300` | detection config |
| class labels | `25` ids in `id2label` | config has duplicated names but distinct ids |
| anchor image size | `null` | runtime dynamic anchors from feature shapes |
| reading-order hidden / layers / heads | `512 / 6 / 8` | nested config |
| reading-order sequence length | `num_queries + 2 = 302` max | start/end plus query slots |
| reading-order 2D coord embedding | max 1024, coord 171, shape 170 | boxes must index within range |
| reading-order relation bias | 16-dim sinusoidal, projected by 1x1 conv to 8 heads | source-derived |

Representative checkpoint sweep:

| Checkpoint | Availability | Operator-significant notes |
| --- | --- | --- |
| `PaddlePaddle/PP-DocLayoutV2_safetensors` | available, ungated | native `pp_doclayout_v2`; HGNetV2-L backbone; 300 queries; dynamic anchors; 25 class ids; reading-order head enabled |

No small/debug or alternate `pp_doclayout_v2` variants were found in official HF search during this audit.

## 3a. Family variation traps

- The neural body composes `hgnet_v2`; DinoML should either require a separate HGNetV2 audit or allowlist this exact `backbone_config`.
- Source tensors are NCHW for image/backbone/hybrid feature maps. Sequence regions flatten NCHW maps as `[B,C,H,W] -> [B,H*W,C]` in row-major spatial order.
- `anchor_image_size=null` means anchor generation depends on runtime feature-map shapes. Static-anchor lowering must reject this checkpoint unless it re-materializes per runtime shape.
- `disable_custom_kernels=true` is present, but the forward path still calls the decorated multiscale deformable-attention layer. Treat eager `grid_sample` as semantic reference and the hub CUDA kernel as an optional provider.
- Detection logits are multi-label sigmoid scores, not softmax with a background row. Postprocess flattens `[queries, classes]` and takes top `num_queries` pairs.
- No NMS: duplicate boxes/classes can pass thresholds and must remain for parity.
- Class ids are remapped through `class_order` before reading-order embeddings; raw detection labels are not the reading-order label ids.
- Reading-order boxes are expected as `[x_min,y_min,x_max,y_max]` in `[0,1000]`, then cast to long for 2D embedding lookup. Out-of-range coordinates raise or clamp only where source explicitly clamps width/height.
- `class_thresholds` are per raw detection class; changing label count requires matching thresholds/order tables.
- The reading-order model has its own LayoutLM-like dense attention with unscaled spatial relation bias and a GlobalPointer triangular mask; do not replace it with decoder KV-cache logic.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `Conv2d`, BatchNorm/FrozenBatchNorm, depthwise Conv2d, MaxPool2d with `ceil_mode=True`, explicit `F.pad`, NCHW concat on channel axis.
- `flatten(2)`, `transpose(1,2)`, `permute`, `reshape/view`, `contiguous`, `cat/concat`, `stack`, `split`, `unbind`, `gather`, `take_along_dim`, `topk`, `argsort`, `scatter_`, boolean masks.
- `F.interpolate` nearest for FPN upsample and mask resize; bilinear `grid_sample` with `align_corners=False` for deformable attention fallback.

Neural primitives:

- HGNetV2 conv blocks: Conv-BN-activation, 1x1/3x3/5x5, depthwise groups, aggregation concatenations.
- Hybrid encoder convs: `1x1` projections, RepVGG-style `3x3 + 1x1` residual branch, CSPRepLayer.
- Linear/GEMM heads: encoder score, encoder bbox MLP, decoder Q/K/V/O, decoder FFNs, class/bbox heads, reading-order embeddings/projections.
- LayerNorm, BatchNorm2d, FrozenBatchNorm affine, ReLU, GELU, SiLU, sigmoid, softmax.

Attention primitives:

- Dense noncausal MHA for AIFI and decoder query self-attention, with position embeddings added to Q/K only.
- Multiscale deformable cross-attention: sampled bilinear reads from multi-level feature maps plus per-query/head/level/point softmax weights.
- Reading-order dense self-attention with optional 2D spatial relation bias and CogView-style softmax stabilization.
- GlobalPointer: linear to query/key, matmul, scale by `sqrt(head_size)`, mask lower triangle to `-1e4`.

Pre/postprocessing-coupled ops:

- Resize/rescale/normalize processor.
- Box conversion center-width-height to corners and back.
- Anchor generation from spatial shapes.
- Score thresholding and class-order remap.
- Reading-order voting from pairwise logits.

## 5. Layer/block breakdown

HGNetV2 backbone, NCHW:

```text
pixel_values [B,3,H,W]
stem: conv3x3 s2 -> pad -> conv2x2 -> pad -> conv2x2
parallel maxpool branch + stem2 branch concat on C
conv3x3 s2 -> conv1x1
stages produce feature maps; PPDocLayout uses stage2/stage3/stage4
```

Detection projection and hybrid encoder:

```text
stage features [B,512,H/8,W/8], [B,1024,H/16,W/16], [B,2048,H/32,W/32]
each -> Conv1x1 + BN -> [B,256,h,w]
deepest selected level -> flatten [B,h*w,256] -> 1 encoder layer -> reshape NCHW
FPN top-down: nearest upsample, concat C, CSPRepLayer
PAN bottom-up: Conv3x3 stride2, concat C, CSPRepLayer
```

Decoder prep:

```text
encoder maps -> decoder Conv1x1+BN -> flatten each to [B,h*w,256]
source_flatten = concat levels on sequence
spatial_shapes = [[h0,w0], [h1,w1], [h2,w2]]
level_start_index = cumsum(h*w)
anchors = generated per level if anchor_image_size is None
memory = valid_mask * source_flatten
enc_outputs_class = Linear(256 -> num_labels)
enc_outputs_coord_logits = MLP(256 -> 256 -> 256 -> 4) + anchors
topk_ind = topk(max class logit, k=300)
target = gathered output_memory unless learn_initial_query
```

Decoder layer, repeated 6 times:

```text
object_query_pos = MLP(4 -> 512 -> 256)(sigmoid(reference_points))
x = dense self-attention(x + pos for Q/K, x for V)
x = LayerNorm(x + dropout(attn))
x = multiscale deformable cross-attention(query=x+pos, value=source_flatten)
x = LayerNorm(x + dropout(cross))
x = LayerNorm(x + FFN(256 -> 1024 -> 256, relu))
reference_points = sigmoid(bbox_head[layer](x) + inverse_sigmoid(reference_points)).detach()
class_logits[layer] = Linear(256 -> num_labels)(x)
```

Reading-order block:

```text
sorted valid boxes/classes -> input_ids [START, PRED..., END, PAD...]
word/token/position/spatial embeddings -> LayerNorm/dropout
6 x (QKV attention + unscaled 2D relation bias + FFN)
GlobalPointer Linear(512 -> 128) -> split Q/K [B,S,64]
order_logits = Q @ K^T / sqrt(64), lower triangle masked
```

## 6. Attention requirements

No autoregressive generation, KV cache, causal decode, RoPE text cache, or beam search is required.

Dense self-attention:

- Noncausal MHA.
- AIFI: sequence length is one selected feature level, typically `(H/32)*(W/32)` for the checkpoint's deepest level after resize.
- Decoder self-attention: query length `300`, or larger only in training denoising paths that first integration should reject.
- Reading-order attention: query/key length `num_queries + 2` max; attention mask is bidirectional over valid elements plus start/end.
- Backend compatibility: source declares attention backend support for the detection model, but reading-order disables SDPA/Flash/Flex flags. For first parity, lower dense matmul-softmax-matmul explicitly.

Multiscale deformable cross-attention:

- Query shape `[B,Q,256]`, heads `8`, head dim `32`.
- Value source `[B,sum_l H_l*W_l,256]`, reshaped to `[B,S,8,32]`.
- Sampling offsets: Linear `256 -> heads * levels * points * 2 = 192`.
- Attention weights: Linear `256 -> heads * levels * points = 96`, softmax over `levels*points`.
- Reference points are `[B,Q,1,4]` in this model; sampling location equation is `ref_xy + offsets / n_points * ref_wh * 0.5`.
- Eager fallback splits value by level, reshapes to `[B*heads, head_dim, H_l, W_l]`, samples with `grid_sample(2*locations-1, bilinear, zeros, align_corners=False)`, multiplies by attention weights, sums over `levels*points`.

## 7. Position encoding and custom math

Detection AIFI 2D sinusoidal embedding:

```python
def pos2d(height, width, dim=256, temperature=10000):
    omega = 1.0 / (temperature ** (arange(dim // 4) / (dim // 4)))
    grid_h, grid_w = meshgrid(arange(height), arange(width), indexing="ij")
    return cat([
        sin(flatten(grid_h)[:, None] * omega),
        cos(flatten(grid_h)[:, None] * omega),
        sin(flatten(grid_w)[:, None] * omega),
        cos(flatten(grid_w)[:, None] * omega),
    ], dim=1)
```

Dynamic dependency: height/width come from runtime feature maps when `eval_size is None`.

Reading-order relation bias:

```python
def relation_bias(boxes):
    src, tgt = boxes[:, :, None, :], boxes[:, None, :, :]
    rel_xy = log(abs(src[..., :2] - tgt[..., :2]) / (src[..., 2:] + eps) + 1)
    rel_wh = log((src[..., 2:] + eps) / (tgt[..., 2:] + eps))
    emb = sincos(cat([rel_xy, rel_wh], -1) * scale, inv_freq)
    return conv1x1(emb.permute(0, 3, 1, 2))  # [B, heads, S, S]
```

Box and anchor math:

- `inverse_sigmoid(x) = log(clamp(x)/(1-clamp(x)))`.
- Anchors are grid centers plus level-scaled width/height, transformed to logits; invalid anchors outside `(0.01,0.99)` become max-float sentinels.

## 8. Preprocessing and input packing

Processor:

- Groups images by shape for batched resize and processing.
- Resizes to `800x800` with bicubic and `antialias=False` to approximate OpenCV resize.
- Rescales by `0.00392156862745098`; mean/std are identity (`[0,0,0]`, `[1,1,1]`).
- Returns `pixel_values`; source model creates `pixel_mask` of ones when not supplied.

Detection postprocessing:

- Model outputs `pred_boxes` normalized center boxes and `logits` after internal sorting by valid threshold.
- Processor converts center boxes to corner boxes, optionally scales to `target_sizes` as `[height,width]`.
- Applies `sigmoid(logits)`, flattens `[queries,classes]`, keeps top `num_queries` query-class pairs, gathers boxes/order sequence, filters by global `threshold`, sorts by reading-order sequence.
- There is no NMS or class-wise suppression.

Reading-order input packing:

- Detection head computes raw bboxes in `[0,1000]` corner format and clamps.
- `max_probs >= class_thresholds[class_ids]` defines valid mask.
- Valid elements are sorted before padding using `argsort(mask.to(int8), descending=True)`.
- Class ids are remapped through `class_order` before label embeddings.
- Packed ids use start token at position 0, pred tokens for valid elements, end token after the last valid element, pad elsewhere.

## 9. Graph rewrite / lowering opportunities

### Rewrite: frozen BatchNorm2d fold into Conv2d

Preconditions:

- Inference only.
- Conv output feeds BatchNorm/FrozenBatchNorm directly.
- BN running stats and affine params are constants.
- Preserve NCHW semantics.

Replacement:

```text
Conv2d(weight,bias=None) -> FrozenBN(weight,bias,mean,var,eps)
=> Conv2d(weight_fused,bias_fused)
```

Failure cases: unfrozen training BN, non-direct consumers, dynamic BN state.

Parity test sketch: random NCHW input across HGNetV2 stem/stage convs, compare fp32 and fp16 tolerances.

### Rewrite: 1x1 Conv2d on NCHW feature map -> pointwise GEMM

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `groups=1`.
- Local region preserves source NCHW flatten order or uses a guarded NHWC transform with rewritten axes.
- Output consumer accepts the chosen layout.

Replacement:

```text
[B,C,H,W] -> flatten spatial -> GEMM(C_in -> C_out) -> reshape
```

Failure cases: BatchNorm not folded, layout consumer is still NCHW conv, dynamic shape code lacks stride guards.

### Rewrite: AIFI flatten-attention-reshape region

Preconditions:

- Source feature map is dense contiguous NCHW.
- Flatten order is exactly row-major `H,W`.
- Position embedding generated for the same runtime `(H,W)`.

Replacement:

```text
NCHW feature map -> sequence attention block -> NCHW feature map
```

Layout constraints: protect with `no_layout_translation()` unless a pass rewrites flatten, positional table construction, and reshape together.

### Rewrite: multiscale deformable attention provider boundary

Preconditions:

- `num_levels=3`, `num_points=4`, `heads=8`, `head_dim=32` for the checkpoint.
- `reference_points.shape[-1] == 4`.
- `spatial_shapes` and `level_start_index` exactly match source flatten order.
- Bilinear interpolation uses zero padding and `align_corners=False`.

Replacement:

```text
offset/weight linears + softmax + grid_sample loop -> provider-backed deformable attention
```

Failure cases: `reference_points` last dim 2 variant, custom kernel with different numeric behavior, malformed dynamic level sizes.

### Rewrite: detection postprocess as bounded top-k/gather pipeline

Preconditions:

- `num_queries=300`, class count equals threshold/order table length.
- End-to-end parity accepts no NMS.

Replacement:

```text
sigmoid -> flatten topk(k=Q) -> label/index decode -> gather boxes/order -> threshold -> order sort
```

Failure cases: caller expects class-wise NMS, output needs stable tie semantics not matched by backend top-k.

## 10. Kernel fusion candidates

Highest priority:

- Conv-BN-activation/frozen-BN folding in HGNetV2 and hybrid FPN/PAN. This dominates image feature extraction and is a clean inference rewrite.
- Multiscale deformable attention provider. The eager `grid_sample` loop is a high-cost custom attention family and the source already advertises a CUDA hub-kernel boundary.
- Dense MHA + FFN for decoder query blocks at fixed query length 300.

Medium priority:

- 1x1 conv/GEMM projections for encoder and decoder feature maps.
- AIFI sequence attention on deepest feature map with generated 2D sine position embeddings.
- Reading-order 2D relation-bias generation plus attention; pairwise `[S,S]` work is small but custom.
- Top-k/gather postprocess on GPU to avoid CPU transfers before reading-order.

Lower priority:

- Reading-order GlobalPointer matmul and triangular mask fusion.
- Anchor generation cache keyed by runtime feature shapes.
- Processor resize/rescale/normalize GPU path; useful only if DinoML owns image preprocessing.

## 11. Runtime staging plan

Stage 1: parse config, load weights, and admit only the exact HF checkpoint shape family: HGNetV2-L, 3 feature levels, 300 queries, dynamic anchors, inference only.

Stage 2: implement or compose HGNetV2 NCHW backbone parity through stage2/stage4 outputs. Keep it a separately auditable backbone component.

Stage 3: lower detection graph through hybrid encoder, proposal top-k, and decoder using eager dense/deformable reference ops. Stub optional hub custom kernel.

Stage 4: add detection heads and internal sorting/class-threshold logic up to `logits`, `pred_boxes`, and `order_logits` inputs.

Stage 5: lower reading-order model from sorted boxes/classes/mask to GlobalPointer logits.

Stage 6: implement exact processor postprocess: target-size scaling, flattened top-k, thresholding, reading-order sort, no NMS.

Stage 7: add optimized provider paths for Conv-BN, deformable attention, and selected dense attention/GEMM regions.

## 12. Parity and validation plan

- Processor parity: known PIL images through HF processor vs DinoML preprocessing, compare `pixel_values` after resize/rescale/normalize.
- Backbone parity: one random and one real preprocessed image, compare HGNetV2 selected feature maps stage2/stage4.
- Hybrid encoder parity: compare projected feature maps, AIFI output, FPN/PAN outputs for dynamic `800x800` and one non-default divisible size.
- Deformable attention parity: random tensors with 3 levels, reference dim 4, heads 8, points 4; compare eager fallback.
- Decoder parity: single decoder layer and full 6-layer stack, including iterative reference update.
- Detection parity: `enc_topk_logits`, `enc_topk_bboxes`, final logits/boxes after sorting.
- Reading-order parity: synthetic boxes/classes/masks, verify packed ids, relation bias, attention output, GlobalPointer logits.
- End-to-end parity: HF output dict after `post_process_object_detection`, including duplicate boxes and `order_seq`.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for most graph outputs; deformable attention may need `rtol=5e-4, atol=5e-4`; fp16 provider paths should be separately approved with looser box/postprocess tolerances.

No tests/imports were run for this audit.

## 13. Performance probes

- Image preprocessing throughput: decode/resize/rescale/normalize separately from model execution.
- HGNetV2 backbone throughput vs image size and batch size.
- Hybrid encoder FPN/PAN conv throughput; count time before decoder flatten.
- AIFI attention sequence length sweep from feature-map size.
- Deformable cross-attention provider comparison: eager grid_sample loop vs custom/provider kernel.
- Decoder-only throughput at Q=300, layers=6.
- Reading-order throughput as valid element count varies from sparse to 300.
- Postprocess latency: flattened top-k, gather, threshold, reading-order sort, CPU vs GPU.
- Dynamic image-size probe: 800x800, smaller document crop, and large aspect-ratio resized/padded cases if processor settings change.

## 14. Skip/defer list

- Training, denoising query construction, Hungarian matcher/loss, gradients, gradient checkpointing.
- General remote/custom hub kernel loading as a required runtime dependency; use eager math as source reference first.
- NMS, polygon points, mask/segmentation outputs; not source behavior for PPDocLayoutV2.
- Other HGNetV2 architectures beyond the exact nested config unless separately audited.
- NHWC global translation. Only local conv/provider regions should use channel-last with guards.
- `reference_points.shape[-1] == 2` deformable attention path until a config requires it.
- `learn_initial_query=true`, static `anchor_image_size`, alternate class counts, alternate threshold/order tables until represented by configs and tests.

## 15. Final implementation checklist

- [ ] Parse `PPDocLayoutV2Config`, nested `HGNetV2Config`, and `PPDocLayoutV2ReadingOrderConfig`.
- [ ] Load checkpoint weights with class/order/threshold tables preserved as runtime metadata.
- [ ] Compose or implement HGNetV2-L NCHW backbone for `stage2`, `stage3`, `stage4`.
- [ ] Implement Conv-BN/FrozenBN, depthwise Conv2d, MaxPool ceil, pad, concat, nearest interpolate.
- [ ] Implement hybrid encoder AIFI + FPN + PAN with guarded NCHW flatten/reshape.
- [ ] Implement dynamic anchor generation from runtime `spatial_shapes`.
- [ ] Implement dense noncausal attention and FFN blocks for encoder/decoder.
- [ ] Implement multiscale deformable attention eager reference and provider admission contract.
- [ ] Implement proposal top-k/gather and iterative decoder box refinement.
- [ ] Implement detection class/bbox heads and internal threshold/sort/remap pipeline.
- [ ] Implement reading-order embeddings, 2D relation bias, dense attention, and GlobalPointer.
- [ ] Implement processor postprocess with no NMS and exact flattened top-k behavior.
- [ ] Add parity tests for preprocessing, backbone, deformable attention, decoder, reading order, and end-to-end outputs.
- [ ] Benchmark backbone, deformable attention, decoder, reading order, and postprocess separately.
