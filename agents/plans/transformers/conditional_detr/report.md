# Conditional DETR Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/conditional-detr-resnet-50 for the primary object-detection target
Config source: Hugging Face config.json/preprocessor_config.json plus local ConditionalDetrConfig defaults
Source files inspected:
- transformers/src/transformers/models/conditional_detr/configuration_conditional_detr.py
- transformers/src/transformers/models/conditional_detr/modeling_conditional_detr.py
- transformers/src/transformers/models/conditional_detr/modular_conditional_detr.py
- transformers/src/transformers/models/conditional_detr/image_processing_conditional_detr.py
- transformers/src/transformers/models/conditional_detr/image_processing_pil_conditional_detr.py
Any missing files or assumptions:
- modeling_conditional_detr.py and image_processing_conditional_detr.py are generated from modular_conditional_detr.py; future upstream edits should target the modular file, but DinoML parity should follow the generated runtime files.
- No feature_extraction_conditional_detr.py or processing_conditional_detr.py exists in this checkout.
- Several historical facebook/microsoft DC5 or panoptic repo IDs referenced in examples were unavailable or returned 401 during config fetch. Non-official open mirrors/fine-tunes are labeled below.
```

Primary scope: inference for `ConditionalDetrForObjectDetection`. `ConditionalDetrModel` is required as the base encoder-decoder. `ConditionalDetrForSegmentation` is optional/deferred unless DinoML chooses panoptic/instance/semantic segmentation parity.

## 2. High-level architecture

Conditional DETR is a vision backbone plus transformer encoder-decoder detector. It is not an autoregressive generation model and has no KV cache. The runtime contract is structured detection: image tensors and pixel masks in, fixed `num_queries` query outputs, class logits and normalized boxes out, then processor postprocessing returns variable-length per-image records.

```text
CPU/image processor -> NCHW pixel_values + pixel_mask
-> ResNet/AutoBackbone feature maps
-> 1x1 channel projection + 2D position embedding
-> encoder self-attention over flattened Hf*Wf pixels
-> learned object queries initialized as zeros
-> conditional decoder self-attention + spatially conditioned cross-attention
-> class linear + bbox MLP with reference-point offset
-> postprocess top-k/sigmoid/filter + cxcywh->xyxy + scale to target sizes
```

Stage decomposition:

- CPU/data pipeline: resize with aspect ratio, rescale by `1/255`, ImageNet normalize, batch pad, produce `pixel_values` `[B,3,Hpad,Wpad]` and `pixel_mask` `[B,Hpad,Wpad]`.
- Backbone stage: ResNet/timm/AutoBackbone in NCHW, frozen batch norm, multi-scale feature maps. Detection uses only the last feature map; segmentation additionally uses earlier FPN maps.
- Transformer stage: flatten projected final feature map to `[B,S,256]`, where `S=Hf*Wf`; encoder and decoder are independently testable if flattened inputs, masks, and positional embeddings are supplied.
- Head/postprocess stage: class logits `[B,Q,C]`, boxes `[B,Q,4]` in normalized center format; postprocess is required for end-to-end detection parity.

## 3. Important config dimensions

Source defaults from `ConditionalDetrConfig`:

| Field | Default / typical value | Operator significance |
|---|---:|---|
| `num_channels` | 3 | NCHW image input channels |
| `num_queries` | 300 | Fixed decoder query slots, affects decoder self-attention and heads |
| `d_model` | 256 | Transformer hidden size and projected feature channels |
| `encoder_layers` / `decoder_layers` | 6 / 6 | Repeated transformer blocks |
| `encoder_attention_heads` / `decoder_attention_heads` | 8 / 8 | MHA heads; head_dim 32 for normal attention |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 2048 / 2048 | FFN inner dimension |
| `activation_function` | `relu` | FFN and segmentation conv block activation |
| `position_embedding_type` | `sine` | `sine` or `learned`; learned uses 50 row + 50 col embeddings |
| `dilation` | false | DC5/output stride path only when timm backbone is used |
| `backbone_config` | default ResNet stage4 | Backbone determines channel count and feature stride |
| `attention_dropout` / `dropout` | 0.0 / 0.1 | Dropout disabled in eval; still appears in source |
| `auxiliary_loss` | false | Training/loss path; only relevant if labels are supplied |
| `is_encoder_decoder` | true | Structured encoder-decoder output class, not text generation |

Representative checkpoint sweep:

| Model/config | Source label | Architecture | Backbone | Layers | FFN | Queries | Labels | Processor size |
|---|---|---|---|---:|---:|---:|---:|---|
| `microsoft/conditional-detr-resnet-50` | official config | object detection | resnet50 | 6 enc / 6 dec | 2048 | 300 | 91 COCO ids | shortest 800, longest 1333 |
| `hf-tiny-model-private/tiny-random-ConditionalDetrForObjectDetection` | tiny/debug config | object detection | resnet50 | 2 enc / 2 dec | 4 | 12 | 91 dummy ids | shortest 800, longest 1333 |
| `Omnifact/conditional-detr-resnet-101-dc5` | open mirror/fine-tune config | object detection | resnet101 | 6 enc / 6 dec | 2048 | 300 | 91 COCO ids | legacy `size=800`, `max_size=1333` |
| `qubvel-hf/microsoft-conditional-detr-resnet-50-finetuned-10k-cppe5` | fine-tune config | object detection | resnet50/timm | 6 enc / 6 dec | 2048 | 300 | 5 | shortest 600, longest 600 |
| `davanstrien/conditional-detr-resnet-50_fine_tuned_beyond_words` | fine-tune config | object detection | resnet50/timm | 6 enc / 6 dec | 2048 | 300 | 7 | shortest 800, longest 1333 |

Config notes: many checkpoint configs omit `backbone_config`; the current config class consolidates that to an AutoBackbone/ResNet config with default `out_features=["stage4"]`. Historical configs may include `backbone`, `use_timm_backbone`, and `use_pretrained_backbone`; the current source handles this through `consolidate_backbone_kwargs_to_config` and `load_backbone`.

## 3a. Family variation traps

- Conditional DETR differs from DETR mainly in decoder conditioning: learned object query positions seed reference points, sine encodings of those spatial anchors are multiplied by per-layer query scales, and decoder cross-attention concatenates content and positional channels for Q/K.
- Cross-attention Q/K effective head dimension is doubled from 32 to 64 for default `d_model=256, heads=8`; V remains 32. Fused attention kernels must accept different QK head width vs V head width or lower through explicit matmul/softmax.
- Detection postprocess uses per-class sigmoid and global top-k over `Q*C`, not softmax over classes and not NMS.
- `num_labels` changes the classifier output width. COCO configs have 91 entries including `N/A`; fine-tunes may have 5 or 7 labels.
- `position_embedding_type="learned"` is source-supported but has fixed 50x50 row/column embedding tables. That can fail on feature maps exceeding 50 in either axis.
- `dilation=True` only affects timm backbone kwargs (`output_stride=16`) and changes feature map spatial size and attention sequence length.
- Segmentation is a materially different head: it requires multi-scale backbone features, a per-query attention map, FPN-style upsampling, mask postprocessing, and optional COCO RLE output.
- NCHW is semantic source layout for image/backbone/conv/FPN regions. NHWC/channel-last is only a guarded optimization candidate. Flatten order from `projected_feature_map.flatten(2).permute(0,2,1)` is row-major `H,W` sequence order and must match positional embeddings and mask flattening.
- Axis-sensitive no-layout guards: `Conv2d`, `BatchNorm2d` replacement, `GroupNorm` over channels, `flatten(2)`, `permute(0,2,1)`, mask interpolation over `[-2:]`, `view(B,C,H,W)` reconstruction, segmentation `einsum("bqc,bqhw->bchw")`, and postprocess box scaling `[w,h,w,h]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors; `flatten(start_dim=2)`, `permute`, `transpose`, `view/reshape`, `contiguous`, `repeat`, `expand`, `unsqueeze`, `stack`, `cat`, `gather`, `topk`, modulo/div floor for top-k index decode.
- Dynamic shape handling for padded images and feature map `Hf,Wf`.
- Boolean/int pixel masks, interpolation of masks to feature map size, flatten mask to `[B,S]`.

Neural network primitives:

- ResNet/AutoBackbone inference with frozen batch norm replacement.
- `Conv2d`: backbone, `1x1` input projection, segmentation 1x1/3x3 convs, and `conv2d` implementation of `k_proj` in `ConditionalDetrMHAttentionMap`.
- `LayerNorm(256)`, `GroupNorm(min(8,out_channels), out_channels)`, ReLU, dropout as no-op in eval.
- Linear projections: default self-attn `256->256`; FFN `256->2048->256`; bbox MLP `256->256->256->4`; class head `256->num_labels`; query scale MLP `256->256->256`; reference point MLP `256->256->2`.

Attention primitives:

- Noncausal encoder self-attention over `S=Hf*Wf`.
- Noncausal decoder self-attention over `Q=300` object queries.
- Noncausal decoder cross-attention `Q x S`, with conditional spatial query Q/K construction and padding mask.
- Attention map for segmentation: query/key matmul over spatial map returning softmax weights `[B,Q,heads,H,W]` without value matmul.

Position/custom math:

- 2D sine positional embedding from pixel mask cumsum, normalized by last row/column, interleaved sin/cos, concat `[pos_y,pos_x]`.
- Learned 2D position embedding alternative.
- Conditional query anchor sine embedding from reference points.
- `inverse_sigmoid` with clamp for bbox offset.

Pre/postprocessing-coupled ops:

- Image resize, rescale, normalize, pad and pixel-mask production.
- Detection postprocess: sigmoid, flatten, top-k, class/query index decode, cxcywh-to-xyxy, gather boxes, scale to target sizes, threshold filter.
- Segmentation postprocess if enabled: softmax/sigmoid, `einsum`, bilinear resize, argmax, mask thresholding, overlap filtering, optional RLE.

## 5. Layer/block breakdown

Backbone and projection:

```text
pixel_values [B,3,H,W], pixel_mask [B,H,W]
features = backbone(pixel_values)                         # NCHW feature maps
mask = interpolate(pixel_mask[None].float(), feature[-2:]).bool()[0]
feature_map = features[-1]                                # usually [B,2048,H/32,W/32] for R50 stage4
x = Conv2d(C_backbone -> 256, kernel=1)(feature_map)
x_seq = x.flatten(2).permute(0,2,1)                       # [B,S,256]
pos = position_embedding(feature_map.shape, mask)          # [B,S,256]
mask_seq = mask.flatten(1)                                # [B,S]
```

Encoder layer, repeated `encoder_layers`:

```text
q,k = Linear(x + pos)       # bias true
v = Linear(x)
attn = noncausal_attention(q,k,v, mask=[B,1,S,S])
x = LayerNorm(x + Dropout(OutProj(attn)))
x = LayerNorm(x + FFN(Linear(256->2048), ReLU, Linear(2048->256)))
```

Decoder initialization:

```text
query_pos = Embedding(num_queries,256).repeat(B,1,1)
queries = zeros_like(query_pos)
reference_logits = MLP(query_pos, 256->256->2)
reference_points = sigmoid(reference_logits).transpose(0,1)   # source produces reference output
query_sine_base = encode_sinusoidal_position_embedding(reference_points[..., :2].transpose(0,1), 128)
```

Decoder layer, repeated `decoder_layers`:

```text
query_sine = query_sine_base if layer0 else query_sine_base * query_scale(hidden)
self_q = q_content(hidden) + q_pos(query_pos)
self_k = k_content(hidden) + k_pos(query_pos)
self_v = v(hidden)
hidden = LayerNorm(hidden + self_attn(self_q,self_k,self_v))

cross_q_content = q_content(hidden)
cross_k_content = k_content(encoder_hidden)
cross_v = v(encoder_hidden)
if layer0:
    cross_q_content += q_pos(query_pos)
    cross_k_content += k_pos(spatial_pos)
cross_q = concat_per_head(cross_q_content, q_pos_sine(query_sine))   # head_dim 64
cross_k = concat_per_head(cross_k_content, k_pos(spatial_pos))       # head_dim 64
hidden = LayerNorm(hidden + cross_attn(cross_q,cross_k,cross_v, encoder_mask))

hidden = LayerNorm(hidden + FFN(256->2048->256))
```

Detection head:

```text
logits = Linear(256 -> num_labels)(hidden)                 # [B,Q,C]
box_delta = MLP(256->256->256->4)(hidden)
box_delta[..., :2] += inverse_sigmoid(reference_points).transpose(0,1)
pred_boxes = sigmoid(box_delta)                            # normalized cx,cy,w,h
```

## 6. Attention requirements

All attention is noncausal, encoder-style, and inference has no cache. Default heads are MHA: 8 Q heads and 8 K/V heads, no GQA/MQA. Masks are additive expanded bidirectional masks produced from pixel masks; padded pixels receive a very negative value before softmax.

Attention variants:

- Encoder self-attention: `[B,heads,S,32] x [B,heads,S,32] -> [B,S,256]`, with position added before Q/K projections.
- Decoder self-attention: `[B,heads,Q,32]`, separate content and position projections added before attention.
- Decoder cross-attention: Q length `Q`, K/V length `S`; Q/K head width is `2*head_dim=64` after concatenating projected content and sine/spatial position per head, while V head width remains 32. Scaling uses `(2*d_model/heads)^-0.5`.
- Segmentation attention map: returns only softmax weights `[B,Q,heads,H,W]`, uses query linear and key `1x1` conv over `[B,256,H,W]`.

The source dispatches through `ALL_ATTENTION_FUNCTIONS` with eager fallback. DinoML can initially lower the exact matmul/softmax path. SDPA/FlashAttention compatibility needs the cross-attention QK/V dimension mismatch checked carefully; if backend APIs require equal Q/K/V head dimension, use a custom lowering for conditional cross-attention.

## 7. Position encoding and custom math

Image sine position embedding:

```python
not_pad = mask
y = cumsum(not_pad, dim=1)
x = cumsum(not_pad, dim=2)
y = y / (y[:, -1:, :] + 1e-6) * (2*pi)
x = x / (x[:, :, -1:] + 1e-6) * (2*pi)
dim_t = 10000 ** (2 * floor(arange(F) / 2) / F)
pos = concat(interleave_sin_cos(y/dim_t), interleave_sin_cos(x/dim_t))
pos = pos.permute(0,3,1,2).flatten(2).permute(0,2,1)
```

Conditional query anchor sine embedding:

```python
def encode_anchor(pos, num_pos_feats=128):
    dim_t = 10000 ** (2 * floor(arange(num_pos_feats) / 2) / num_pos_feats)
    parts = [interleave_sin_cos(coord[..., None] * 2*pi / dim_t) for coord in pos.unbind(-1)]
    parts[0], parts[1] = parts[1], parts[0]
    return cat(parts, dim=-1).to(pos.dtype)
```

BBox offset math:

```python
def inverse_sigmoid(x, eps=1e-5):
    x = clamp(x, 0, 1)
    return log(clamp(x, min=eps) / clamp(1 - x, min=eps))
```

Image sine embeddings depend on dynamic `pixel_mask` and feature shape. For fixed padded image buckets they can be cached per shape/mask pattern, but arbitrary padding keeps them runtime-dependent. Learned position embeddings are static tables but depend on feature height/width.

## 8. Preprocessing and input packing

Processor output for inference:

- `pixel_values`: torch tensor `[B,3,Hpad,Wpad]`, channel-first, resized/rescaled/normalized image.
- `pixel_mask`: torch int64 tensor `[B,Hpad,Wpad]`, 1 for valid pixels and 0 for padding.

Default processor behavior:

- Resize keeps aspect ratio so shortest edge is 800 and longest edge at most 1333, unless config overrides.
- Rescale by `0.00392156862745098`.
- Normalize with ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Pad batch to max height/width unless disabled or explicit `pad_size` is provided.

Detection postprocess contract:

- Inputs: `logits [B,Q,C]`, `pred_boxes [B,Q,4]` normalized cxcywh, optional `target_sizes [B,2]` in `(height,width)`.
- Applies `sigmoid` to logits, flattens to `[B,Q*C]`, keeps `top_k` entries, decodes query index by floor-divide by `C`, class label by modulo `C`.
- Converts boxes to xyxy, gathers boxes by selected query index, scales by `[width,height,width,height]`, filters by score threshold.
- Returns list length `B`; each item has variable-length `scores`, `labels`, `boxes`. No NMS is performed.

Segmentation postprocess, deferred unless segmentation target is enabled:

- Semantic: `softmax(logits)` and `sigmoid(pred_masks)`, `einsum("bqc,bqhw->bchw")`, optional bilinear resize, argmax.
- Instance/panoptic: softmax class max, no-object removal via `label != num_labels`, mask thresholding, overlap validity filtering, optional target resize and RLE.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d projection -> per-pixel Linear

Source pattern: `Conv2d(Cin, 256, kernel=1)` followed by `flatten(2).permute(0,2,1)`.

Replacement:

```text
NCHW feature -> optional NHWC local view -> Linear(Cin -> 256) per spatial point -> [B,Hf*Wf,256]
```

Preconditions: kernel 1, stride 1, padding 0, dilation 1, groups 1, no consumer needs original NCHW projected map. Segmentation cannot use this replacement globally because it needs projected NCHW features for the mask head.

Weight transform: `linear.weight = conv.weight[:, :, 0, 0]`, same bias.

Failure cases: segmentation path, non-contiguous/layout-sensitive consumers, non-1x1 conv.

Parity test: compare projected feature map and flattened sequence for random `[B,C,H,W]`.

### Rewrite: conditional cross-attention explicit QK packing

Source pattern: content projections plus sine/spatial projections, per-head concat, attention over doubled Q/K width.

Replacement:

```text
Q_content, Q_pos_sine, K_content, K_pos -> packed QK heads with head_dim=64
V -> heads with head_dim=32
Attention(QK_width=64, V_width=32)
```

Preconditions: noncausal, no KV cache, same number of Q/K/V heads, scale equals `64^-0.5` for default dims, first-layer `q_pos`/`k_pos` addition respected.

Failure cases: fused attention primitive assumes Q/K/V same head dim; layer index handling for first-layer-only `q_pos_proj`.

Parity test: one decoder cross-attention layer against Transformers eager output and attention weights.

### Rewrite: detection postprocess as structured runtime helper

Source pattern: sigmoid, flatten, top-k, index decode, gather, box conversion, scaling, threshold.

Replacement: a structured postprocess operator or CPU helper returning per-image records.

Preconditions: class logits are independent sigmoid scores; no NMS; target sizes are original image sizes in `(height,width)`.

Failure cases: using DETR softmax/no-object semantics or NMS; label count mismatch; wrong target size axis order.

Parity test: random logits/boxes against `ConditionalDetrImageProcessor.post_process_object_detection`.

### Layout pass guard: backbone and feature flatten

Candidate: local NHWC/channel-last optimization inside conv-heavy backbone and mask head.

Required rewrites: channel axis for `GroupNorm`, `Conv2d` weights/layout, `flatten(2)` sequence order, `permute(0,2,1)` elimination only if row-major spatial order remains identical, `view(B,C,H,W)` reconstruction in segmentation.

Guard: transformer sequence tensors are `[B,S,C]`; post-backbone transition should be an explicit no-layout-translation boundary unless the pass proves identical `H,W` flatten order.

## 10. Kernel fusion candidates

Highest priority:

- ResNet backbone conv/bn/relu/maxpool blocks: dominates pixel compute before transformer.
- LayerNorm + residual around attention/FFN for `[B,S,256]` and `[B,Q,256]`.
- FFN `Linear -> ReLU -> Linear` for hidden size 256 and inner 2048.
- Conditional decoder cross-attention packing plus attention, because Q/K doubled width and first-layer behavior are easy to get subtly wrong.
- Detection postprocess helper, because end-to-end parity depends on structured variable-length outputs.

Medium priority:

- 2D sine position embedding with mask cumsum and sin/cos.
- Encoder/decoder MHA for small head dim 32, including attention mask handling.
- Bbox MLP plus inverse-sigmoid reference offset plus sigmoid.
- 1x1 projection-to-linear rewrite for detection-only path.

Lower priority:

- Learned position embedding path.
- Segmentation mask head FPN conv/upsample/groupnorm stack.
- Panoptic/instance segmentation postprocess and RLE conversion.
- Training losses, Hungarian matching, auxiliary loss outputs.

## 11. Runtime staging plan

Stage 1: parse config and processor metadata; load official R50 detection weights and run processor-produced `pixel_values`/`pixel_mask` through Transformers as baseline.

Stage 2: implement/validate backbone + frozen batch norm + final feature projection, or initially accept precomputed final feature maps to isolate transformer work.

Stage 3: one encoder layer parity with 2D sine positions and expanded pixel mask.

Stage 4: one decoder layer parity, especially conditional self-attention and cross-attention.

Stage 5: full `ConditionalDetrModel` parity for `last_hidden_state` and `reference_points`.

Stage 6: detection heads and postprocess parity for `logits`, `pred_boxes`, and final variable-length detections.

Stage 7: optimize with conv/layout passes, attention kernels, FFN fusions, and structured postprocess helper.

Stage 8: optional segmentation target with multi-scale backbone outputs, bbox attention map, mask head, and segmentation postprocessors.

Initial stubs: training labels/losses, auxiliary loss outputs, segmentation, learned position embeddings, and uncommon backbone variants can be deferred for first detection parity.

## 12. Parity and validation plan

- Unit-test `inverse_sigmoid` and both sine embedding functions against Transformers over varied masks, dtypes, and feature sizes.
- Random tensor parity for encoder self-attention, decoder self-attention, and decoder cross-attention with masks; use fp32 tight tolerance first (`rtol=1e-4`, `atol=1e-5`), then fp16 relaxed tolerance.
- Single encoder layer and single decoder layer parity with loaded weights.
- Full base model parity for official R50 on one resized/padded image: compare `last_hidden_state`, `encoder_last_hidden_state`, and `reference_points`.
- Detection head parity: compare raw `logits` and `pred_boxes`.
- Postprocess parity: compare scores/labels/boxes for `threshold=0.5`, `top_k=100`, with explicit target sizes.
- Batch padding parity: two different image sizes in one batch, verify pixel mask, downsampled mask, final detections.
- Optional segmentation parity: raw `pred_masks`, semantic segmentation map, panoptic records.

Recommended tolerances: fp32 graph parity `rtol=1e-4`, `atol=1e-5`; fp16/bf16 attention and conv paths may need `rtol=1e-2`, `atol=1e-2`, with postprocess comparisons allowing small box-coordinate drift near threshold boundaries.

## 13. Performance probes

- Processor throughput: resize/normalize/pad images/sec separately from GPU runtime.
- Backbone-only latency/throughput for common padded sizes, including 600x600 and 800/1333 aspect-ratio buckets.
- Encoder sequence-length sweep over `S=Hf*Wf`; DC5/dilation paths should be measured separately.
- Decoder-only throughput for `Q=300`, including cross-attention over varied `S`.
- End-to-end object detection throughput with batch-size sweep.
- Postprocess CPU time for `Q*C` top-k and box scaling.
- Attention backend comparison: eager matmul vs SDPA/custom for encoder, decoder self-attention, and conditional cross-attention separately.
- Memory probes: activation memory for encoder attention at large padded image sizes; segmentation mask head memory if enabled.

## 14. Skip/defer list

- Training losses, Hungarian matcher, auxiliary decoder losses, gradient checkpointing, layerdrop.
- Segmentation and panoptic/instance/semantic postprocess for first object-detection integration.
- Learned 2D position embeddings until a checkpoint requiring them is targeted.
- Remote-code or unavailable/gated checkpoint behavior.
- General NMS support; source object detection does not use NMS.
- Quantization and multi-GPU/tensor-parallel execution.
- Broad NHWC conversion across the full model without explicit axis/layout guards.

## 15. Final implementation checklist

- [ ] Parse `ConditionalDetrConfig`, including backbone consolidation and label count.
- [ ] Parse `ConditionalDetrImageProcessor` resize/rescale/normalize/pad metadata.
- [ ] Load NCHW backbone weights with frozen batch norm behavior.
- [ ] Implement 2D sine position embedding from feature masks.
- [ ] Implement flattened feature projection and mask flattening with source row-major order.
- [ ] Implement encoder self-attention with position-added Q/K.
- [ ] Implement decoder self-attention with content and query-position projections.
- [ ] Implement conditional decoder cross-attention with doubled Q/K head width.
- [ ] Implement reference-point MLP, query-scale MLP, and anchor sine embedding.
- [ ] Implement detection class head and bbox MLP with inverse-sigmoid reference offset.
- [ ] Implement structured detection postprocess with sigmoid/top-k/no-NMS semantics.
- [ ] Add one-layer encoder/decoder parity tests.
- [ ] Add full official R50 raw-output parity test.
- [ ] Add end-to-end postprocess parity test with target sizes.
- [ ] Benchmark processor, backbone, encoder, decoder, and postprocess separately.
