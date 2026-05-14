# PP-DocLayoutV3 Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-DocLayoutV3_safetensors
Config source: HF config.json and preprocessor_config.json at main, repo sha 3ec586e86ed9245a567bb13395a3db64d5c077cc
Source files inspected: configuration_pp_doclayout_v3.py, image_processing_pp_doclayout_v3.py, modeling_pp_doclayout_v3.py, modular_pp_doclayout_v3.py, plus pp_doclayout_v2 and hgnet_v2 references
Any missing files or assumptions: only one official Transformers-native v3 checkpoint was found; no weights/safetensors metadata was opened; PaddleOCR-native base repo and ONNX exports were not treated as source of in-library behavior
```

`modeling_pp_doclayout_v3.py` is generated from `modular_pp_doclayout_v3.py`; future Transformers source edits should target the modular file. The report treats the generated modeling file as the exact runtime basis and the modular file as the authoritative edit source.

Primary runtime target: `PPDocLayoutV3ForObjectDetection` inference, including layout boxes, labels, reading order, and polygon points from masks. Training losses and denoising-label construction are deferred.

## 2. High-level architecture

PP-DocLayoutV3 is an image-only document layout detector. The neural body is not a text encoder and has no autoregressive generation path.

```text
image preprocessing -> HGNetV2 convolutional backbone -> 1x1 feature projections
-> hybrid encoder (AIFI on selected level + FPN/PAN + mask prototype head)
-> encoder top-k proposal selection and optional mask-enhanced reference boxes
-> deformable-attention decoder with iterative bbox refinement
-> class logits + normalized boxes + reading-order logits + query masks
-> postprocess: score/top-k/filter, order sort, box scaling, mask-to-polygon
```

Stage decomposition:

- CPU/data pipeline: image decode, resize to 800x800 by default, rescale by 1/255, no mean/std shift, and NCHW tensor creation.
- GPU/runtime stage 1: HGNetV2 NCHW convolutional backbone.
- GPU/runtime stage 2: hybrid encoder over three detector feature levels plus mask prototype map.
- GPU/runtime stage 3: proposal top-k, mask-to-box reference refinement, six-layer query decoder.
- CPU or mixed postprocess: score filtering, reading-order sort, mask thresholding, OpenCV contour/polygon extraction.

The backbone, hybrid encoder, decoder, and postprocess ABI are independently stageable. End-to-end parity needs the postprocess path because v3 returns polygon points, not just DETR-style boxes.

## 3. Important config dimensions

Source-derived defaults are from `PPDocLayoutV3Config`; checkpoint values are from `PaddlePaddle/PP-DocLayoutV3_safetensors/config.json`.

| Field | Source default | HF v3 checkpoint | Runtime impact |
| --- | ---: | ---: | --- |
| `model_type` | `pp_doclayout_v3` | `pp_doclayout_v3` | Selects native v3 classes. |
| `num_labels` | inherited/default | 25 via `id2label` | Detector class logits width. |
| `d_model` | 256 | 256 | Decoder hidden width and query width. |
| `encoder_hidden_dim` | 256 | 256 | Hybrid encoder channel width. |
| `encoder_in_channels` | `[512,1024,2048]` | `[512,1024,2048]` | Projected HGNetV2 stage2-4 channels. |
| `feature_strides` / `feat_strides` | `[8,16,32]` | `[8,16,32]` as `feature_strides` | Anchor and mask FPN stride assumptions. Source reads `feat_strides`; config alias handling should be verified. |
| `encode_proj_layers` | `[2]` | `[2]` | AIFI runs only on the coarsest projected feature. |
| `encoder_layers` | 1 | 1 | One intra-scale self-attention layer. |
| `encoder_attention_heads` | 8 | 8 | AIFI MHA, head dim 32. |
| `encoder_ffn_dim` | 1024 | 1024 | AIFI FFN. |
| `decoder_layers` | 6 | 6 | Number of query decoder layers. |
| `decoder_attention_heads` | 8 | 8 | Decoder self/cross heads, head dim 32. |
| `decoder_n_points` | 4 | 4 | Deformable attention samples per level. |
| `num_feature_levels` | 3 | 3 | Decoder attends over three levels. |
| `num_queries` | 300 | 300 | Fixed detector query count and top-k output budget. |
| `num_prototypes` | 32 | omitted, effective default 32 | Mask prototype channels. |
| `mask_feature_channels` | `[64,64]` | `[64,64]` | Mask FPN hidden/output channels. |
| `x4_feat_dim` | 128 | 128 | Stage1 lateral into mask feature map. |
| `mask_enhanced` | `True` | omitted, effective default `True` | Uses encoder masks to refine decoder reference boxes. |
| `global_pointer_head_size` | 64 | 64 | Reading-order pairwise head width. |
| `gp_dropout_value` | 0.1 | omitted, effective default 0.1 | Dropout disabled under eval. |
| `activation_function` | `silu` | `silu` | Conv/CSP activations. |
| `encoder_activation_function` | `gelu` | `gelu` | AIFI FFN activation. |
| `decoder_activation_function` | `relu` | `relu` | Decoder FFN activation. |
| `backbone_config` | HGNetV2 L, stage1-4 | HGNetV2 L, stage1-4 | v3 consumes stage1 for mask lateral and stage2-4 for detector. |
| `torch_dtype` | unspecified | `float32` | Config metadata; source does not force dtype. |
| `disable_custom_kernels` | `True` | `true` | Uses eager/grid-sample deformable attention unless a hub kernel is enabled. |

Representative checkpoint sweep:

| Repo | Basis | Model type | Key variation |
| --- | --- | --- | --- |
| `PaddlePaddle/PP-DocLayoutV3_safetensors` | official Transformers-native v3 | `pp_doclayout_v3` | HGNetV2-L stage1-4, mask prototypes, in-decoder order head, polygon postprocess. |
| `PaddlePaddle/PP-DocLayoutV3` | PaddleOCR-native base repo metadata | not Transformers config | Same family but delegated to PaddleOCR; route to separate importer/audit. |
| `Bei0001/PP-DocLayoutV3-ONNX` and other ONNX mirrors | export/quantized mirrors | ONNX | Useful for ABI hints only; not source of Transformers behavior. |
| `PaddlePaddle/PP-DocLayoutV2_safetensors` | adjacent family delta reference | `pp_doclayout_v2` | Separate reading-order transformer after detection; no polygon mask postprocess. |

## 3a. Family variation traps

- Only one official in-library v3 checkpoint was found. Do not assume a family sweep over small/base/large variants exists until more configs are available.
- V3 source reads `config.feat_strides`, but the HF v3 JSON uses `feature_strides`. DinoML config parsing should confirm Transformers alias/unknown-field behavior before trusting either spelling.
- V3 requests HGNetV2 `return_idx=[0,1,2,3]`, then pops stage1/x4 for mask lateral and projects stage2-4 for detector. V2 requested only stage2-4.
- V3 moves reading order into the decoder via per-layer `decoder_order_head` plus `GlobalPointer`; v2 ran a separate LayoutLMv3-like reading-order model after class thresholding and `class_order` remapping.
- V3 adds mask prototypes and query masks. Postprocess returns `polygon_points` and depends on OpenCV contours; v2 postprocess returns boxes/order only.
- V3 `PPDocLayoutV3ForObjectDetection.forward` rejects `labels`; training/denoising helper code exists but is not supported by the public inference head.
- Deformable attention can dispatch to a hub kernel via `use_kernel_forward_from_hub("MultiScaleDeformableAttention")`, but the inspected config disables custom kernels and source has an eager `grid_sample` fallback.
- The model is NCHW throughout source conv/backbone/feature-map code. NHWC is only a guarded optimization candidate, not the semantic graph.
- Axis-sensitive regions include `cat(..., dim=1)` for channel concatenation, `flatten(2).transpose(1,2)` for NCHW map-to-sequence, `grid_sample` NCHW input, `topk/gather` on query/class axes, and mask flattening for `bmm`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input `[B,3,H,W]`; default processor emits `[B,3,800,800]`.
- `flatten(2)`, `transpose(1,2)`, `permute`, `reshape`, `view`, `contiguous`.
- `cat/concat` on channel axis and sequence axis.
- `gather`, `take_along_dim`, `topk`, `argsort`, `sort`, `scatter_`, boolean indexing for postprocess.
- `where`, comparisons, masks, `masked_fill`, `tril`, `triu`.

Neural network primitives:

- HGNetV2 conv stem/stages: Conv2d, BatchNorm2d/FrozenBatchNorm2d, SiLU/ReLU, max-pool-like stem behavior from HGNetV2, depthwise/grouped Conv2d in light blocks.
- Projection Conv2d 1x1 and Conv2d 3x3 stride 2.
- FPN/PAN: nearest and bilinear interpolate, channel concat, residual adds, CSPRepLayer/RepVGG blocks.
- Linear layers with bias for attention projections, FFNs, class heads, bbox MLPs, order head, mask query head.
- LayerNorm, dropout as no-op in eval.
- `bmm` for query-mask coefficients times mask prototypes.

Attention primitives:

- Noncausal encoder self-attention over one flattened feature level, MHA with `d_model=256`, heads=8, head_dim=32.
- Noncausal decoder self-attention over 300 queries, MHA heads=8.
- Multiscale deformable cross-attention over three feature levels, heads=8, levels=3, points=4, value width 256.
- No KV cache and no causal decode.

Detection/postprocess ABI:

- Model raw outputs: `logits [B,300,25]`, `pred_boxes [B,300,4]` center-x/center-y/width/height normalized, `order_logits [B,300,300]`, `out_masks [B,300,Hm,Wm]`.
- Postprocess: sigmoid logits, flatten class/query scores, top-k `num_queries`, convert boxes to xyxy, optional scale by `target_sizes`, gather boxes/masks/order, threshold, sort by reading order, mask sigmoid/threshold, OpenCV contour to polygon.
- No source NMS. Multiple class/query hits are filtered by score and reading order, not suppressed by IoU.

Preprocessing-coupled ops:

- Resize with bicubic, `antialias=False` to approximate OpenCV resize.
- Rescale by `0.00392156862745098`, normalize with mean `[0,0,0]`, std `[1,1,1]`.
- Optional padding hooks exist in base processor flow but v3 config defaults to fixed resize.

Quantized/packed weight metadata ops:

- None in Transformers source. Safetensors is dense model storage. ONNX/quantized mirrors are out of scope for native v3.

## 5. Layer/block breakdown

Input:

```text
pixel_values: [B, 3, H, W], source NCHW
pixel_mask: optional [B, H, W], default all ones
```

Backbone:

```text
HGNetV2Backbone(pixel_values) -> feature maps stage1..stage4
stage1: x4 feature for mask lateral, checkpoint x4_feat_dim=128
stage2-4: projected by Conv1x1 + BN to 256 channels
```

Hybrid encoder:

```text
feature_maps = [P3, P4, P5], each [B,256,H_l,W_l]
P5 = AIFI(P5):
  x = flatten NCHW map to [B,H_l*W_l,256]
  pos = 2D sin/cos [1,H_l*W_l,256]
  x = MHA(q,k from x+pos, v from x) + residual + LayerNorm
  x = Linear(256->1024) + GELU + Linear(1024->256) + residual + LayerNorm
  reshape back to [B,256,H_l,W_l]
top-down FPN:
  lateral Conv1x1, nearest upsample x2, concat channel dim, CSPRepLayer
bottom-up PAN:
  stride-2 Conv3x3, concat channel dim, CSPRepLayer
mask_feat:
  MaskFeatFPN over PAN maps, bilinear upsample x2, add stage1 lateral Conv3x3, Conv3x3+Conv1x1 -> [B,32,Hm,Wm]
```

Proposal preparation:

```text
decoder sources = Conv1x1+BN over PAN maps -> [B,256,H_l,W_l]
source_flatten = cat_l flatten(source_l) -> [B, sum(H_l*W_l), 256]
anchors = logit grid boxes per level, width/height scale 0.05 * 2**level
memory = valid_mask * source_flatten
output_memory = Linear(256->256) + LayerNorm
enc_outputs_class = Linear(256->25)
enc_outputs_coord_logits = MLP(256->256->256->4) + anchors
topk_ind = topk(max_class_logit, k=300)
target = gathered output_memory at topk indices, detached unless learned queries are enabled
if mask_enhanced:
  mask_query_embed = MLP(norm(target)) -> [B,300,32]
  enc_out_masks = bmm(mask_query_embed, mask_feat.flatten(2)) -> [B,300,Hm,Wm]
  reference_points = mask_to_box_coordinate(enc_out_masks > 0)
  reference_points_unact = inverse_sigmoid(reference_points)
```

Decoder layer, repeated 6 times:

```text
reference_points = sigmoid(reference_points_unact)
query_pos = MLP(4->512->256)(reference_points)
x = MHA(q,k from x+query_pos, v from x) + residual + LayerNorm
x = MultiscaleDeformableAttention(query=x+query_pos, value=source_flatten, reference_points)
x = residual + LayerNorm
x = Linear(256->1024) + ReLU + Linear(1024->256) + residual + LayerNorm
predicted_corners = bbox MLP(x)
reference_points = sigmoid(predicted_corners + inverse_sigmoid(previous_reference_points)).detach()
out_query = decoder_norm(x)
class logits = Linear(256->25)(out_query)
mask coefficients = MLP(256->256->256->32)(out_query)
out_mask = bmm(mask coefficients, mask_feat.flatten(2)) -> [B,300,Hm,Wm]
order logits = GlobalPointer(Linear(256->256)(valid queries)) -> [B,300,300]
```

Object detection head returns only final-layer `logits`, `pred_boxes`, `order_logits`, and `out_masks`, while also carrying intermediate tensors in the output dataclass.

## 6. Attention requirements

Attention is required, but only for encoder/query detection, not text generation.

Dense self-attention:

- Noncausal self-attention in AIFI and decoder query layers.
- MHA, no GQA/MQA, `num_heads=8`, `head_dim=32`, q/k/v widths all 256.
- q/k receive positional embeddings; v uses unpositioned hidden states.
- Mask support exists via additive masks or bool mask-to-zero in deformable value path; inference default has no decoder self-attention mask.
- SDPA/Flash/Flex flags are advertised, but parity must preserve the source q/k positional-add order.

Multiscale deformable cross-attention:

- Query: `[B,300,256]`.
- Key/value source: concatenated multi-level image sequence `[B,S,256]`, where `S=sum(H_l*W_l)`.
- `spatial_shapes [L,2]`, `level_start_index [L]`, and Python `spatial_shapes_list` are part of the ABI.
- Sampling offsets: Linear 256 -> `heads * levels * points * 2 = 8*3*4*2`.
- Attention weights: Linear 256 -> `heads * levels * points = 8*3*4`, softmax over `levels*points`.
- Reference points may be 2D or 4D. V3 runtime path uses 4D boxes after proposal generation.
- Eager fallback normalizes sampling locations to `[-1,1]` and uses `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)` over per-level NCHW maps reshaped to `[B*heads, head_dim, H, W]`.

No KV cache, beam reorder, causal masks, or autoregressive generation controller is applicable.

## 7. Position encoding and custom math

2D sine/cosine positional embedding for AIFI:

```python
def pos2d(height, width, dim=256, temperature=10000):
    pos_dim = dim // 4
    omega = 1.0 / (temperature ** (arange(pos_dim) / pos_dim))
    grid_h, grid_w = meshgrid(arange(height), arange(width), indexing="ij")
    emb_h = flatten(grid_h).outer(omega)
    emb_w = flatten(grid_w).outer(omega)
    return cat([sin(emb_h), cos(emb_h), sin(emb_w), cos(emb_w)], dim=1)
```

Anchor generation:

```python
grid_xy = (meshgrid_xy + 0.5) / [width, height]
wh = full_like(grid_xy, 0.05 * (2.0 ** level))
anchors = log(concat([grid_xy, wh]) / (1 - concat([grid_xy, wh])))
anchors = where(valid_range_0p01_to_0p99, anchors, finfo_max)
```

Mask-to-box reference refinement:

```python
mask = enc_out_masks > 0
x_min, y_min, x_max, y_max = min/max occupied mask coordinates
xyxy = [x_min, y_min, x_max + 1, y_max + 1] / [width, height, width, height]
cxcywh = [(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0]
reference_points_unact = inverse_sigmoid(cxcywh)
```

GlobalPointer order logits:

```python
qk = Linear(256, 2 * head_size)(queries).reshape(B, Q, 2, head_size)
q, k = unbind(qk, dim=2)
logits = matmul(q, k.transpose(-2, -1)) / sqrt(head_size)
logits = masked_fill(lower_triangle_including_diagonal, -1e4)
```

Postprocess reading order:

```python
scores = sigmoid(order_logits)
votes = triu(scores, 1).sum(dim=1) + tril(1 - scores.transpose(1,2), -1).sum(dim=1)
pointers = argsort(votes)
order_seq.scatter_(1, pointers, arange(Q))
```

## 8. Preprocessing and input packing

Processor contract:

- Input images are resized to `800x800` by default.
- Resize uses bicubic and explicitly passes `antialias=False`.
- Pixel values are rescaled by `1/255`; mean/std are identity.
- The model input is `pixel_values`, source layout NCHW.
- No OCR, words, boxes, tokenizer, or text/layout token input is consumed by v3. It detects visual layout regions directly from pixels.

Postprocessing inputs and outputs:

- Inputs: `logits`, `pred_boxes`, `order_logits`, `out_masks`, and caller-supplied `target_sizes` as `(height,width)`.
- Boxes are normalized center format from the model and become absolute xyxy after scale.
- Scores are sigmoid multi-label scores; no background/no-object class handling is visible in postprocess.
- `torch.topk(scores.flatten(1), num_queries)` chooses class-query pairs.
- Per-image variable-length output is produced by `score >= threshold`.
- Masks are gathered by selected query, sigmoid-thresholded, cropped by each output box, resized to box dimensions with OpenCV nearest interpolation, contoured, approximated, and returned as `polygon_points`.
- No NMS is performed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: frozen BatchNorm2d into Conv2d affine

Source pattern:

```text
Conv2d -> FrozenBatchNorm2d -> optional activation
```

Replacement:

```text
Conv2d with folded weight/bias -> optional activation
```

Preconditions:

- Inference mode only.
- BN parameters and running stats are constants.
- Preserve epsilon. `PPDocLayoutV3FrozenBatchNorm2d` uses `epsilon = 1e-5` inside forward.

Failure cases:

- Do not fold live `BatchNorm2d` before `replace_batch_norm` has established frozen state.
- Do not fold training paths.

Parity test sketch:

- Compare stage outputs before/after fold on random NCHW tensors for fp32 and fp16 tolerance.

### Rewrite: Conv2d 1x1 NCHW projection to GEMM

Source pattern:

```text
Conv2d(Cin->256, kernel=1, stride=1, padding=0, bias=False) + BN
```

Replacement:

```text
flatten spatial [B*H*W,Cin] -> GEMM(weight.T) -> reshape [B,256,H,W]
```

Preconditions:

- Kernel 1x1, stride 1, padding 0, dilation 1, groups 1.
- Source tensor is contiguous NCHW or layout pass owns a correct NHWC rewrite.
- BN folded or lowered as channel affine.

Failure cases:

- Do not apply to stride-2 3x3 decoder projections or grouped/depthwise HGNetV2 layers.

### Rewrite: mask query bmm to batched GEMM

Source pattern:

```text
mask_query_embed [B,Q,32] @ mask_feat.flatten(2) [B,32,Hm*Wm]
```

Replacement:

```text
BMM_RRR -> reshape [B,Q,Hm,Wm]
```

Preconditions:

- Prototype channel count fixed to `num_prototypes`.
- `mask_feat` flatten order must stay NCHW row-major over H,W.

Failure cases:

- NHWC layout rewrite must also rewrite flatten order and any downstream crop/scale assumptions.

### Rewrite: source top-k proposal selection as detector primitive

Source pattern:

```text
topk(max(enc_outputs_class, dim=-1), k=num_queries)
gather coord logits, output memory, class logits
```

Replacement:

```text
TopKProposal(enc_logits, enc_boxes, memory, k=300)
```

Preconditions:

- `k == config.num_queries`.
- Tie behavior must match `torch.topk`.
- Class max is over logits before sigmoid.

Failure cases:

- Do not combine with postprocess top-k; these are separate top-k operations with different score tensors.

### Rewrite: no-layout-translation guard around deformable attention

Source pattern:

```text
value split by H*W -> reshape [B*heads, head_dim, H, W] -> grid_sample
```

Replacement:

```text
Either preserve NCHW eager path, or lower to a dedicated multiscale-deformable-attention kernel with explicit layout metadata
```

Preconditions for any NHWC optimization:

- Rewrite source feature maps, value projection storage interpretation, `spatial_shapes`, offset normalizer, `grid_sample` equivalent, and output reshape together.
- Preserve `align_corners=False`, zero padding, bilinear interpolation, and sampling coordinate convention.

Failure cases:

- A generic NCHW->NHWC pass that changes conv layout but leaves `grid_sample`/flatten semantics untouched will break parity.

### Rewrite: postprocess split into GPU bounded top-k and CPU polygon extraction

Source pattern:

```text
sigmoid -> topk -> gather boxes/masks/order -> threshold -> order sort -> cv2 polygons
```

Replacement:

```text
GPU detector selection ABI -> CPU/OpenCV polygon ABI
```

Preconditions:

- Return selected scores, labels, boxes, query indices, order ranks, and masks before contour extraction.
- Preserve absence of NMS.

Failure cases:

- Do not replace with common object-detection NMS unless an explicit different product contract is requested.

## 10. Kernel fusion candidates

Highest priority:

- NCHW Conv2d+FrozenBN+SiLU/ReLU fusion across HGNetV2 and FPN/PAN. This dominates early image compute and is easy to validate locally.
- Multiscale deformable attention kernel. The eager fallback uses many reshapes, splits, `grid_sample`, and reductions; a dedicated provider boundary is the key nonstandard detector op.
- Top-k proposal/gather primitive. It gates decoder input and is shape-sensitive but bounded at 300 queries.
- Mask prototype `BMM + reshape` and mask-to-box reduction. V3 depends on masks for reference refinement and polygon outputs.

Medium priority:

- Dense MHA/SDPA for AIFI and decoder self-attention. The shapes are moderate and noncausal; existing attention support can cover it after positional-add ordering is guarded.
- FFN/MLP Linear+activation fusion for encoder/decoder blocks and bbox/mask heads.
- Postprocess GPU selection path before CPU polygon extraction.

Lower priority:

- Full CPU OpenCV polygon parity inside DinoML runtime. It may remain a data-pipeline/postprocess helper initially.
- Training denoising helpers, loss/matcher paths, and dropout behavior.
- NHWC end-to-end conv layout optimization. Useful later, but source has multiple NCHW-specific boundary conditions.

## 11. Runtime staging plan

Stage 1: Config and preprocessing ABI.

- Parse `PPDocLayoutV3Config`, normalize `feature_strides`/`feat_strides`, and reject non-HGNetV2-L/unrecognized variants initially.
- Emit NCHW `pixel_values [B,3,800,800]`; leave image decode/resize in CPU pipeline.

Stage 2: Backbone and hybrid encoder parity.

- Compose audited HGNetV2 backbone or route to a separately audited HGNetV2 implementation.
- Lower feature projections, AIFI, FPN/PAN, and mask prototype head.

Stage 3: Proposal and decoder body.

- Implement anchor generation, top-k proposal selection, mask-enhanced reference-point construction, query decoder self-attention, deformable cross-attention, and iterative bbox refinement.

Stage 4: Detection outputs.

- Produce final `logits`, `pred_boxes`, `order_logits`, and `out_masks` with exact shapes.

Stage 5: Postprocess ABI.

- First return raw outputs plus a reference Python postprocess.
- Then split bounded GPU top-k/gather/threshold from CPU polygon extraction.

Stage 6: Optimizations.

- Add Conv+FrozenBN folding, conv activation fusion, deformable attention provider, BMM mask provider, and guarded layout rewrites.

## 12. Parity and validation plan

- Config parse test: HF v3 config round trip, including effective defaults for omitted `num_prototypes`, `mask_enhanced`, and `gp_dropout_value`.
- Processor parity: resize/rescale/normalize output against Transformers image processor for RGB test images.
- HGNetV2 stage parity: random/fixed image through backbone stage maps.
- Hybrid encoder parity: compare PAN maps and `mask_feat`.
- Position/anchor parity: exact anchors for 800x800 and a non-square guarded case.
- Deformable attention parity: random tensors for `value`, `spatial_shapes`, `reference_points`, offsets/weights; compare eager grid-sample output.
- Mask-to-box parity: masks with empty, single-pixel, rectangular, and ragged blobs.
- Decoder one-layer parity, then six-layer parity.
- End-to-end raw output parity: `logits`, `pred_boxes`, `order_logits`, `out_masks`.
- Postprocess parity: selected boxes/labels/order and polygon points against Transformers/OpenCV.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for neural blocks; stricter for pure index/box math.
- fp16/bf16: start with `rtol=5e-2`, `atol=5e-3` around attention/interpolate paths; tighten after provider decisions.

## 13. Performance probes

- Preprocessing throughput: image decode + resize + rescale for 800x800.
- Backbone-only throughput over batch sizes 1, 2, 4, 8.
- Hybrid encoder throughput, split into AIFI and FPN/PAN/mask prototype.
- Deformable attention backend comparison: eager grid_sample versus custom/provider kernel.
- Decoder-only latency for `Q=300`, `L=3`, `points=4`, 6 layers.
- Proposal top-k/gather latency and memory traffic.
- Mask head throughput: `BMM [B,300,32] x [B,32,Hm*Wm]`.
- Postprocess latency: GPU score selection versus CPU OpenCV polygon extraction.
- Resolution sweep if admitting non-800 sizes, because `S=sum(H_l*W_l)` and mask map size scale with input.
- NCHW baseline versus guarded NHWC conv-only islands, excluding deformable attention until a layout-aware kernel exists.

## 14. Skip/defer list

- Training, denoising query construction, matching, losses.
- `labels` path; public object detection head explicitly raises when labels are provided.
- PaddleOCR-native and ONNX mirror importers.
- NMS; source does not use it.
- General-purpose segmentation/mask APIs beyond the detector's query masks and polygon postprocess.
- General NHWC graph translation across deformable attention and postprocess boundaries.
- Custom hub kernel loading at first integration; start from eager semantics and add provider boundary later.
- Dynamic model families beyond the one official v3 checkpoint unless new configs appear.

## 15. Final implementation checklist

- [ ] Parse PP-DocLayoutV3 config and normalize/check `feat_strides` vs `feature_strides`.
- [ ] Add admission allowlist for `PaddlePaddle/PP-DocLayoutV3_safetensors`-style HGNetV2-L config.
- [ ] Own or compose HGNetV2 backbone audit/implementation.
- [ ] Implement NCHW Conv2d/BN/SiLU/ReLU/grouped-conv coverage needed by HGNetV2/FPN/PAN.
- [ ] Implement AIFI 2D sine/cos position embedding and noncausal MHA parity.
- [ ] Implement FPN/PAN CSPRepLayer path with nearest/bilinear interpolate guards.
- [ ] Implement mask prototype head and mask-query BMM.
- [ ] Implement anchor generation and top-k proposal/gather primitive.
- [ ] Implement mask-to-box reference refinement.
- [ ] Implement multiscale deformable attention ABI and eager/reference parity.
- [ ] Implement six-layer decoder with iterative bbox refinement.
- [ ] Emit raw detection ABI: logits, pred_boxes, order_logits, out_masks.
- [ ] Implement postprocess selection ABI with no NMS.
- [ ] Keep OpenCV polygon extraction as CPU postprocess first.
- [ ] Add Conv+FrozenBN folding rewrite with parity tests.
- [ ] Add no-layout-translation guards around flatten/grid_sample/postprocess boundaries.
- [ ] Benchmark backbone, hybrid encoder, deformable attention, decoder, mask head, and postprocess separately.
