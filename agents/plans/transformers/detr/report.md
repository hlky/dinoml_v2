# DETR Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary inference target: facebook/detr-resnet-50, DetrForObjectDetection.
  Additional sweep: facebook/detr-resnet-101, facebook/detr-resnet-50-dc5,
  facebook/detr-resnet-101-dc5, facebook/detr-resnet-50-panoptic.

Config source:
  https://huggingface.co/facebook/detr-resnet-50/raw/main/config.json
  https://huggingface.co/facebook/detr-resnet-50/raw/main/preprocessor_config.json
  Same raw paths for the additional sweep checkpoints above.

Source files inspected:
  transformers/src/transformers/models/detr/configuration_detr.py
  transformers/src/transformers/models/detr/modeling_detr.py
  transformers/src/transformers/models/detr/image_processing_detr.py
  transformers/src/transformers/models/detr/image_processing_pil_detr.py
  transformers/src/transformers/loss/loss_for_object_detection.py
  Conversion scripts were spot-checked for backbone naming/config history.

Any missing files or assumptions:
  No remote code is required for standard DETR. This report targets CUDA
  inference for object detection from already-preprocessed image tensors first.
  CPU/PIL/torchvision preprocessing and postprocessing can remain outside the
  compiled graph initially. Training loss, Hungarian matching, panoptic
  segmentation, and mask postprocessing are documented but deferred unless the
  product target explicitly expands beyond detection inference.
```

## 2. High-level architecture

DETR is an image object detector with a CNN backbone, a bidirectional transformer encoder over flattened image features, and a transformer decoder over a fixed learned set of object queries.

```text
CPU image preprocessing -> pixel_values/pixel_mask
  -> ResNet backbone feature maps
  -> 1x1 projection to d_model
  -> flatten Hf*Wf tokens + 2D position embedding
  -> transformer encoder
  -> learned object queries + transformer decoder cross-attention
  -> class logits + normalized cxcywh boxes
  -> CPU/GPU postprocess to scores/labels/xyxy boxes
```

Stage decomposition:

- CPU/data pipeline: decode, resize with aspect ratio, rescale/normalize, pad batch to common `H,W`, produce `pixel_values [B,3,H,W]` and `pixel_mask [B,H,W]`.
- CNN backbone: source uses NCHW ResNet feature maps with frozen batch norm; output stride is normally 32, or 16 for DC5/dilation configs.
- Transformer encoder: flattened spatial sequence `[B,S,256]`, where `S=Hf*Wf`, plus flattened mask and sine/learned 2D position embeddings.
- Transformer decoder: fixed `num_queries=100`, zero query content plus learned query position embeddings, repeated self-attention/cross-attention/MLP blocks.
- Detection heads: per-query class logits `[B,100,num_labels+1]` and normalized boxes `[B,100,4]`.
- Postprocessing: softmax without the final no-object class, threshold, center-to-corners conversion, target-size scaling. This output contract is required for end-to-end detection parity.

Independently stageable units: image processor handoff, ResNet backbone, 1x1 projection plus flatten/mask, encoder, decoder with object queries, detection heads, and postprocess.

## 3. Important config dimensions

Worked example: `facebook/detr-resnet-50`.

| Field | Value | Source |
| --- | ---: | --- |
| primary task | object detection | config architecture / source |
| `model_type` | `detr` | config.json |
| `d_model` | 256 | config.json |
| `encoder_layers` / `decoder_layers` | 6 / 6 | config.json |
| encoder heads / decoder heads | 8 / 8 | config.json |
| head dim | 32 | inferred `d_model / heads` |
| encoder FFN / decoder FFN | 2048 / 2048 | config.json |
| activation | ReLU | config.json |
| `num_queries` | 100 | config.json |
| position embedding | sine | config.json |
| backbone | resnet50 | config.json |
| dilation / DC5 | false | config.json |
| input channels | 3 | config/source default |
| detection class logits | `num_labels + 1` | source; extra class is no-object |
| COCO detection labels | 91 ids in inspected configs | config `id2label` |
| image processor resize | shortest edge 800, longest edge 1333 | preprocessor_config |
| image mean/std | ImageNet `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | preprocessor_config |
| padding | pad batch to max resized H/W and emit `pixel_mask` | image processor source |

Representative checkpoint sweep:

| Checkpoint | Architecture | Backbone | Dilation | Labels | Head target | Processor format |
| --- | --- | --- | --- | ---: | --- | --- |
| `facebook/detr-resnet-50` | `DetrForObjectDetection` | ResNet-50 | false | 91 | class + boxes | `coco_detection` |
| `facebook/detr-resnet-101` | `DetrForObjectDetection` | ResNet-101 | false | 91 | class + boxes | `coco_detection` |
| `facebook/detr-resnet-50-dc5` | `DetrForObjectDetection` | ResNet-50 | true | 91 | class + boxes | old feature extractor config |
| `facebook/detr-resnet-101-dc5` | `DetrForObjectDetection` | ResNet-101 | true | 91 | class + boxes | old feature extractor config |
| `facebook/detr-resnet-50-panoptic` | `DetrForSegmentation` | ResNet-50 | false | 250 | class + boxes + masks | `coco_panoptic` |

Source defaults from `DetrConfig` that may be omitted by older configs: `num_channels=3`, `backbone_config` consolidated to a ResNet AutoBackbone config, `auxiliary_loss=False`, `position_embedding_type="sine"`, `dropout=0.1`, `attention_dropout=0.0`, `activation_dropout=0.0`, and `is_encoder_decoder=True`.

## 3a. Family variation traps

- DETR source tensors are NCHW through the image processor, backbone, mask downsampling, input projection, and segmentation head. Treat NHWC/channel-last as an optimized internal region only after source-faithful parity is stable.
- DC5 configs set `dilation=true`, which requests `output_stride=16` for timm-style backbones. This roughly doubles each final feature-map spatial dimension versus the stride-32 path, and therefore increases encoder/cross-attention sequence length by about 4x.
- The current `DetrConfig` can use AutoBackbone or timm. Official older configs carry `backbone="resnet50"` / `"resnet101"` but not a full `backbone_config`; loaders consolidate this into a ResNet backbone config.
- Frozen batch norm is source behavior: all `BatchNorm2d` in the backbone are replaced by `DetrFrozenBatchNorm2d`, equivalent at inference to per-channel affine `x * scale + bias`.
- Sine position embeddings are mask-derived and depend on padded/valid pixels after mask interpolation to feature-map size. They are not a fixed table unless image/mask bucket is fixed.
- Learned position embeddings exist but only cover 50 rows and 50 columns in source. Do not assume all DETR checkpoints use sine, although inspected official configs do.
- Object queries are learned position embeddings; decoder content queries default to zeros. The learned query table is added to Q/K in decoder self-attention and to Q in decoder cross-attention.
- The decoder is not autoregressive. There is no KV cache, causal mask, prefill/decode loop, or token sampling.
- Detection and panoptic checkpoints share the base model and box/class heads, but panoptic adds a multi-scale FPN-like mask head and different label space.
- `auxiliary_loss=True` changes decoder outputs by stacking intermediate layernormed decoder states and applying heads to them for training loss. Inspected configs set it false.
- Hungarian matching, L1/GIoU losses, focal/dice mask losses, and SciPy `linear_sum_assignment` are training-only when labels are provided. They are deferred for inference.
- Axis-sensitive layout traps:
  - Source `pixel_values` is `[B,C,H,W]`; NHWC input requires an explicit boundary contract or transpose.
  - Source feature flatten is `feature_map.flatten(2).permute(0,2,1)`: NCHW spatial order is row-major `(h,w)`.
  - `pixel_mask` is `[B,H,W]`; mask interpolation target axes are feature-map `[-2:]`.
  - Position embedding cumsum axes are height axis 1 and width axis 2 on `[B,Hf,Wf]` masks.
  - Postprocess target sizes are `(height,width)`, but box scale order is `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor validation for `pixel_values [B,3,H,W]`.
- Batch padding and int64/bool `pixel_mask [B,H,W]`; runtime can initially accept the processor-produced mask.
- `interpolate` for mask downsampling to feature-map sizes, nearest/bilinear-equivalent bool threshold path from source `float -> interpolate -> bool`.
- NCHW Conv2d backbone feature maps and 1x1 projection.
- Flatten spatial axes and `permute` to `[B,S,C]`; reverse reshape for segmentation memory.
- Repeat/expand object query embeddings from `[100,256]` to `[B,100,256]`.
- Attention masks expanded to additive `[B,1,T,S]` masks through Transformers masking utilities.
- Postprocess boolean thresholding and variable-size per-image outputs.

Neural network primitives:

- ResNet backbone with Conv2d, frozen BatchNorm affine, ReLU, pooling, residual adds, bottleneck blocks. ResNet-50 vs ResNet-101 changes only backbone depth.
- 1x1 Conv2d `C_backbone_last -> 256`; ResNet final channels are typically 2048.
- Linear with bias for all Q/K/V/O projections, FFNs, class head, and box MLP.
- LayerNorm over hidden size 256 after each encoder/decoder sublayer and final decoder output.
- ReLU activation in FFN and box MLP.
- Dropout appears in source but is disabled in inference.

Attention primitives:

- Encoder noncausal self-attention over image tokens `[B,S,256]`, MHA heads=8, head_dim=32.
- Decoder noncausal self-attention over object queries `[B,100,256]`.
- Decoder cross-attention: queries length 100, key/value length S from encoder.
- Additive padding masks, no causal masks, no RoPE, no ALiBi, no cache.
- Source can dispatch through eager, SDPA, FlashAttention, or flex attention interfaces. Eager math is `matmul(q,k^T)*scale + mask -> softmax -> dropout -> matmul(weights,v)`.

Position/custom math ops:

- Sine 2D position embedding from feature mask cumsums.
- Optional learned row/column embedding table path.

Detection/postprocessing ops:

- Class logits: `Linear(256 -> num_labels+1)` per query.
- Box MLP: `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 4) -> sigmoid`.
- Object detection postprocess: softmax over classes, ignore final no-object class for scores/labels, max over classes, threshold, `cxcywh -> xyxy`, scale to target image sizes.
- No NMS is used by DETR postprocess. Do not insert NMS for parity.

Segmentation optional ops:

- Multi-head attention-map softmax over spatial grid.
- Query-expanded feature maps with concat.
- Conv3x3 + GroupNorm + ReLU mask head.
- Nearest upsample and FPN feature fusion.
- Mask sigmoid, bilinear resize, argmax/threshold, segment merge logic in postprocessing.

Training/loss deferred ops:

- Hungarian assignment through SciPy, classification/L1/GIoU costs, generalized box IoU, DICE/focal mask losses, auxiliary decoder losses.

## 5. Layer/block breakdown

Image processor output:

```text
pixel_values: [B,3,Hpad,Wpad] float, NCHW, normalized
pixel_mask:   [B,Hpad,Wpad] int64, 1 for valid pixels and 0 for padding
```

Backbone and projection:

```text
features = ResNet(pixel_values)                         # list of NCHW maps
for each map: mask_i = interpolate(pixel_mask, map.HW).bool()
feature_map, mask = features[-1]                        # [B,Cb,Hf,Wf], [B,Hf,Wf]
projected = Conv2d(Cb -> 256, kernel=1)(feature_map)    # [B,256,Hf,Wf]
tokens = projected.flatten(2).permute(0,2,1)            # [B,S,256], S=Hf*Wf
pos = position_embedding(feature_map.shape, mask)       # [B,S,256]
flat_mask = mask.flatten(1)                             # [B,S]
```

Encoder layer, repeated 6 times:

```text
residual = x
q,k = Linear(x + pos), Linear(x + pos)
v = Linear(x)
y = noncausal_mha(q,k,v, flat_mask)
y = Linear(y)
x = LayerNorm(residual + Dropout(y))

residual = x
y = Linear(256 -> 2048)(x)
y = ReLU(y)
y = Linear(2048 -> 256)(y)
x = LayerNorm(residual + Dropout(y))
```

Decoder setup:

```text
query_pos = Embedding(num_queries=100, 256).repeat(B,1,1)
queries = zeros_like(query_pos) unless decoder_inputs_embeds is supplied
```

Decoder layer, repeated 6 times:

```text
residual = q_content
self_q,self_k = Linear(q_content + query_pos), Linear(q_content + query_pos)
self_v = Linear(q_content)
y = noncausal_mha(self_q,self_k,self_v, decoder_attention_mask)
q_content = LayerNorm(residual + Dropout(Linear(y)))

residual = q_content
cross_q = Linear(q_content + query_pos)
cross_k = Linear(encoder_tokens + pos)
cross_v = Linear(encoder_tokens)
y = cross_attention(cross_q,cross_k,cross_v, encoder_flat_mask)
q_content = LayerNorm(residual + Dropout(Linear(y)))

residual = q_content
y = Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)
q_content = LayerNorm(residual + Dropout(y))
```

Decoder final/head:

```text
hidden = final_decoder_layernorm(q_content)             # [B,100,256]
logits = Linear(256 -> num_labels+1)(hidden)            # [B,100,C+1]
pred_boxes = sigmoid(MLP_3layer(256 -> 256 -> 256 -> 4))# [B,100,4], cxcywh normalized
```

Segmentation optional head:

```text
memory = encoder_tokens.permute(0,2,1).view(B,256,Hf,Wf)
bbox_mask = attention_map(decoder_hidden, memory, flat_mask)  # [B,100,8,Hf,Wf]
mask_input = concat(query-expanded projected map, bbox_mask)
mask_head = Conv/GN/ReLU + FPN nearest-upsample fusions
pred_masks = [B,100,Hmask,Wmask]
```

## 6. Attention requirements

- Encoder attention: noncausal self-attention, MHA, 8 heads, head_dim 32, Q/K receive 2D spatial position embeddings, V does not.
- Decoder self-attention: noncausal self-attention over 100 object queries, MHA, 8 heads, Q/K receive learned object query position embeddings, V does not.
- Decoder cross-attention: noncausal cross-attention, Q receives object query position embedding, K receives encoder spatial position embedding, V is plain encoder hidden states.
- Masks: bidirectional additive masks. Encoder mask blocks padded image tokens. Decoder query mask is optional. Cross-attention mask blocks padded encoder tokens.
- Cache/generation: not applicable. DETR decoder is a fixed-depth set decoder, not an autoregressive text decoder.
- Sliding/local attention: none.
- RoPE/ALiBi/relative bias: none.
- FlashAttention/SDPA compatibility: source advertises support. Fused attention must preserve position-add-before-projection behavior and additive mask semantics. Eager path scales QK scores by `head_dim ** -0.5`, adds mask before softmax, applies dropout only in training, then multiplies by V.

For common resized images around 800x1333, ResNet stride-32 final maps can produce on the order of `25x42=1050` encoder tokens after padding; DC5 stride-16 can produce roughly `50x84=4200` tokens. These are shape inferences from the processor/backbone stride contract and should be bucketed/confirmed per actual resized input.

## 7. Position encoding and custom math

Sine position embeddings are required for inspected official checkpoints:

```python
def detr_sine_pos(mask, num_feats=128, temperature=10000, scale=2*pi):
    # mask is [B,Hf,Wf] after feature-mask interpolation; source uses 1 for valid.
    y = mask.cumsum(1, dtype=float)
    x = mask.cumsum(2, dtype=float)
    y = y / (y[:, -1:, :] + 1e-6) * scale
    x = x / (x[:, :, -1:] + 1e-6) * scale
    dim = temperature ** (2 * floor(arange(num_feats) / 2) / num_feats)
    px = stack([sin((x[..., None] / dim)[..., 0::2]),
                cos((x[..., None] / dim)[..., 1::2])]).flatten_last()
    py = stack([sin((y[..., None] / dim)[..., 0::2]),
                cos((y[..., None] / dim)[..., 1::2])]).flatten_last()
    return concat([py, px], dim=-1).flatten_spatial_to_BS()
```

Note: source passes a mask where valid positions are true after interpolation. This differs from the original DETR convention that often cumsums `not_mask`; Dinoml parity should follow the inspected HF source.

Learned position embeddings, optional:

```text
x_emb = column_embedding[0:Wf]
y_emb = row_embedding[0:Hf]
pos[h,w] = concat(x_emb[w], y_emb[h])
pos -> [B,Hf*Wf,256]
```

Precompute opportunities:

- Sine position embeddings can be cached per `(batch mask pattern, Hf, Wf, dtype)`; for fully valid single-size batches, cache per feature size.
- Learned embeddings can be materialized per `(Hf,Wf)`.
- Object query position embeddings are constant weights expanded across batch.

## 8. Preprocessing and input packing

Processor contract from `DetrImageProcessor`:

- Input images are converted to channel-first tensors before internal processing.
- Resize keeps aspect ratio with `size={"shortest_edge":800,"longest_edge":1333}` in current official preprocessors.
- Rescale and normalize use ImageNet mean/std.
- Batch images are padded to the maximum resized height/width in the batch unless `pad_size` is specified.
- `pixel_mask` is int64 `[B,Hpad,Wpad]`, 1 for original/resized valid pixels and 0 for right/bottom padding.
- Model input names are `pixel_values` and `pixel_mask`.

Recommended first Dinoml boundary:

```text
CPU processor owns decode/resize/rescale/normalize/pad.
Dinoml runtime accepts:
  pixel_values: [B,3,Hpad,Wpad] float32/float16 NCHW, contiguous
  pixel_mask:   [B,Hpad,Wpad] int64 or bool-compatible, same padded H/W
  target_sizes: [B,2] optional postprocess input, CPU-side initially
```

End-to-end output contract for object detection:

```text
raw runtime outputs:
  logits:     [B,100,num_labels+1]
  pred_boxes: [B,100,4] normalized center_x, center_y, width, height in [0,1]

post_process_object_detection(threshold, target_sizes) returns list length B:
  scores: [Ni] max softmax probability excluding final no-object class
  labels: [Ni] argmax class id excluding final no-object class
  boxes:  [Ni,4] absolute xyxy in target image pixel coordinates
```

Postprocessing details:

- `prob = softmax(logits, -1)`.
- `scores, labels = prob[..., :-1].max(-1)`.
- `boxes = center_to_corners_format(pred_boxes)`.
- If target sizes are present, scale with `[img_w,img_h,img_w,img_h]`.
- Filter each image independently with `score > threshold`.
- No NMS or top-k selection is part of the HF postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: FrozenBatchNorm2d -> affine

Source pattern:

```text
y = x * (weight / sqrt(running_var + 1e-5)) + (bias - running_mean * scale)
```

Replacement:

```text
PerChannelAffine_NCHW(x, scale[C], bias[C])
```

Preconditions:

- Module is `DetrFrozenBatchNorm2d` or an inference-only BatchNorm2d with frozen running stats.
- Channel axis is source NCHW axis 1.
- No training updates required.

Failure cases: training mode, dynamic batch-norm statistics, or wrong channel axis after layout translation.

Parity test sketch: compare each ResNet block before/after folding BN affine into adjacent Conv2d where legal, or direct affine otherwise.

### Rewrite: Conv2d 1x1 projection -> GEMM

Source pattern:

```text
Conv2d(Cb -> 256, kernel=1) -> flatten(2).permute(0,2,1)
```

Replacement:

```text
NCHW/NHWC spatial rows [B*Hf*Wf,Cb] -> GEMM_RCR_Bias(Cb -> 256) -> reshape [B,S,256]
```

Preconditions:

- Kernel 1, stride 1, padding 0, dilation 1, groups 1.
- Flatten order preserves source row-major spatial order.
- If NHWC is used internally, transform activation layout and weight access consistently.

Weight transform:

```python
w = conv.weight.reshape(256, Cb)  # source OIHW with 1x1
b = conv.bias
```

Failure cases: arbitrary Conv2d projection, non-contiguous layout without accessor support, or hidden NHWC/NCHW mismatch before position/mask flattening.

### Rewrite: QKV packing within each attention module

Source pattern:

```text
q = Linear(x_q)
k = Linear(x_k)
v = Linear(x_v)
```

Replacement:

- Encoder self-attention can pack Q/K together only after `x + pos`; V uses plain `x`, so full QKV packing is unsafe unless the input add is represented separately for Q/K.
- Decoder self-attention has the same Q/K vs V distinction with query position.
- Cross-attention has different query and key/value sources; only K/V can be packed if K position add is split from V source, or if the packed op supports source-specific pre-adds.

Exact preconditions:

- Same source tensor and same pre-add for packed projections.
- Bias settings match.
- Output split order is Q,K,V or K,V as declared.

Failure cases: packing Q/K/V naively from one tensor would incorrectly add position embeddings to V.

Parity test sketch: compare packed and unpacked attention inputs for encoder self-attention and decoder self-attention with nonzero position embeddings.

### Rewrite: position embedding cache per feature bucket

Source pattern:

```text
interpolate pixel_mask -> cumsum/sin/cos -> flatten
```

Replacement:

```text
CachedPositionEmbedding[Hf,Wf,mask_pattern,dtype]
```

Preconditions:

- Feature size and valid-pixel mask pattern are known or bucketed.
- Fully valid masks are common and cacheable by shape.
- Padded batches with different valid extents need mask-aware cache keys or source computation fallback.

Failure cases: mixed image sizes in a padded batch where each sample has different valid extent.

### Rewrite: NCHW/NHWC islands for backbone

Source pattern: ResNet backbone and projection are NCHW. Dinoml may prefer NHWC/channel-last for Conv2d.

Replacement:

```text
NCHW public ABI -> guarded NHWC backbone region -> NCHW-compatible flatten/postprocess boundary
```

Required axis rewrites:

- Conv weights must transform from PyTorch OIHW to provider layout.
- Frozen BN channel axis `1` becomes last channel axis if folded/unfolded in NHWC.
- Mask remains `[B,H,W]` and should not be channel-translated.
- Feature flatten order must still produce `[h major, w minor]` sequence tokens.

No-layout-translation guards:

- Public `pixel_values` ABI until a new processor/runtime contract is declared.
- Mask interpolation and sine position embedding cumsum axes.
- Detection postprocess and target-size scaling.
- Segmentation head outputs unless the API explicitly advertises NHWC masks/features.

## 10. Kernel fusion candidates

Highest priority:

- ResNet inference backbone primitives: Conv2d + frozen-BN affine + ReLU/residual. This dominates early runtime and is required before transformer work is meaningful.
- Noncausal encoder/cross-attention with additive padding masks for long image-token sequences, especially DC5 stride-16 configs.
- 1x1 projection/Linear/GEMM paths using existing CUTLASS GEMM where possible.
- LayerNorm + residual patterns in encoder/decoder blocks.
- Detection postprocess kernel or vectorized CPU helper for `softmax -> max excluding no-object -> box convert/scale -> threshold`, because runtime output parity depends on exact filtering.

Medium priority:

- Position embedding generation/cache for padded feature masks.
- ReLU FFN fusion `Linear -> ReLU -> Linear` for fixed 256/2048 dimensions.
- Attention projection packing with strict Q/K position-add guards.
- Mask interpolation/downsampling specialized for bool/int masks and known backbone stride.

Lower priority:

- Segmentation attention-map and mask head kernels.
- Panoptic/instance segment merging and RLE conversion.
- Hungarian matcher and training losses.
- GPU image preprocessing; useful later but CPU processor handoff is enough for first parity.

## 11. Runtime staging plan

Stage 1: Processor and config handoff.

- Parse `DetrConfig` and preprocessor config.
- Accept preprocessed `pixel_values` and `pixel_mask`.
- Load official ResNet-50 detection weights and verify tensor names/shapes.

Stage 2: Backbone/projection parity.

- Implement or import ResNet backbone path with frozen BN semantics.
- Lower final 1x1 projection and flatten/mask/position outputs.
- Validate stride-32 and DC5 stride-16 shape equations.

Stage 3: Encoder-only parity.

- Implement sine position embedding and encoder blocks.
- Compare encoder last hidden state for a fixed preprocessed image and mask.

Stage 4: Decoder/object-query parity.

- Implement learned query expansion, zero query content, decoder self/cross attention, and final decoder layernorm.
- Validate final `last_hidden_state [B,100,256]`.

Stage 5: Detection heads and raw outputs.

- Implement class head and box MLP with sigmoid.
- Validate `logits` and `pred_boxes`.

Stage 6: Postprocessing parity.

- Implement HF-compatible `post_process_object_detection`, initially as CPU helper.
- Verify scores/labels/boxes against HF for target sizes and thresholds.

Stage 7: Optimizations.

- Add layout-region rewrite for backbone if profitable.
- Add attention/GEMM fusions, position cache, and vectorized postprocess.

Stage 8: Optional segmentation path.

- Add bbox attention map, mask head, and semantic/instance/panoptic postprocess only if panoptic target is requested.

## 12. Parity and validation plan

Random/operator tests:

- Frozen batch norm affine for NCHW tensors.
- Mask pad and mask interpolation to representative feature sizes.
- Sine position embedding for full-valid and padded masks.
- Attention Q/K position-add behavior: verify V does not receive position embeddings.
- Center-to-corners box conversion and target-size scaling.

Model slice tests:

- Processor contract fixture: save HF `pixel_values` and `pixel_mask` for one or more images; use these as runtime inputs.
- Backbone final feature map parity for ResNet-50 and ResNet-101.
- Projection/flatten/position parity for stride-32 and DC5 stride-16.
- Single encoder layer parity, then 6-layer encoder parity.
- Single decoder layer parity with object queries, then 6-layer decoder parity.
- Raw `DetrForObjectDetection` logits and `pred_boxes` parity.
- End-to-end postprocessed detection parity: compare number of kept boxes, labels, scores, and xyxy boxes for fixed threshold/target sizes.

Segmentation optional tests:

- Bbox attention-map parity.
- Mask-head parity using captured multi-scale backbone features.
- Semantic/panoptic postprocess parity for `facebook/detr-resnet-50-panoptic`.

Suggested tolerances:

- fp32 source-faithful: `rtol=1e-4`, `atol=1e-5` per block; full model boxes/logits may need `rtol=2e-4` after backbone+transformer accumulation.
- fp16/bf16 optimized: compare against a reduced-precision PyTorch run with tolerances chosen per Conv/GEMM/attention backend; keep postprocess thresholds fixed and inspect near-threshold detections separately.

## 13. Performance probes

- CPU image processor throughput: resize/normalize/pad images/sec at common COCO sizes.
- Backbone-only latency/throughput for ResNet-50 vs ResNet-101, stride-32 vs DC5 stride-16.
- Feature projection + flatten + position embedding time.
- Encoder attention/MLP time as a function of `S=Hf*Wf`.
- Decoder self-attention and cross-attention time for `Q=100` and varying encoder S.
- Detection head and postprocess time, including variable number of retained boxes.
- Batch-size sweep with mixed image sizes to measure padding waste.
- NCHW baseline vs guarded NHWC/channel-last backbone region.
- Attention backend comparison: eager/SDPA/Flash/flex equivalent for noncausal masks.
- End-to-end images/sec with processor included and excluded.

No benchmark measurements are included; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training mode, dropout behavior, gradients, and gradient checkpointing.
- Hungarian matching, L1/GIoU/classification loss, DICE/focal mask losses, auxiliary decoder loss heads.
- Panoptic/instance/semantic segmentation mask head and postprocessing for the first object-detection target.
- Learned position embedding path unless a target checkpoint uses it.
- GPU resize/normalize/pad preprocessing.
- NMS, because standard DETR postprocessing does not use it.
- Autoregressive generation, KV cache, beam search, speculative decoding, and text-token machinery; not applicable.
- Multi-GPU tensor parallelism and quantization.

## 15. Final implementation checklist

- [ ] Parse `DetrConfig`, including `d_model`, encoder/decoder layer counts, heads, `num_queries`, backbone, dilation, position embedding type, and label count.
- [ ] Parse `DetrImageProcessor` config: resize policy, mean/std, format, and padding/mask behavior.
- [ ] Load ResNet backbone, frozen-BN, 1x1 projection, transformer, query embedding, class head, and box MLP weights.
- [ ] Accept preprocessed `pixel_values [B,3,H,W]` and `pixel_mask [B,H,W]`.
- [ ] Implement frozen BatchNorm2d inference affine or fold it into Conv2d where legal.
- [ ] Implement ResNet-50/101 backbone parity, including DC5 stride/dilation shape handling.
- [ ] Implement mask downsampling and spatial flattening with source NCHW row-major order.
- [ ] Implement DETR sine position embedding from feature masks.
- [ ] Implement encoder self-attention where Q/K receive spatial position embeddings and V does not.
- [ ] Implement decoder learned object queries with zero content queries.
- [ ] Implement decoder self-attention and cross-attention position-add rules.
- [ ] Implement FFN ReLU MLPs and LayerNorm/residual ordering.
- [ ] Implement class logits head and 3-layer sigmoid box head.
- [ ] Implement HF-compatible object-detection postprocess with no NMS.
- [ ] Add parity tests for backbone, position embeddings, encoder, decoder, raw outputs, and postprocessed detections.
- [ ] Add DC5 shape/performance coverage.
- [ ] Add guarded layout-region and 1x1 Conv2d-to-GEMM rewrites after source parity.
- [ ] Defer Hungarian/loss and segmentation mask paths until explicitly targeted.
