# DAB-DETR DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: IDEA-Research/dab-detr-resnet-50 primary; IDEA-Research/dab-detr-resnet-50-dc5 variant
Config source: HF config/preprocessor JSON snapshots under _sources/hf_configs/
Source files inspected:
  transformers/src/transformers/models/dab_detr/configuration_dab_detr.py
  transformers/src/transformers/models/dab_detr/modeling_dab_detr.py
  transformers/src/transformers/models/dab_detr/__init__.py
  transformers/src/transformers/models/conditional_detr/image_processing_conditional_detr.py
  transformers/src/transformers/models/conditional_detr/modular_conditional_detr.py
Any missing files or assumptions:
  The dab_detr package has no dedicated image_processing_dab_detr.py. Official configs use
  ConditionalDetrImageProcessor, so postprocess/preprocess facts below come from that file.
  IDEA-Research/dab-detr-resnet-101 and IDEA-Research/dab-detr-resnet-101-dc5 returned 401 for raw config files.
```

Gated/401 checkpoint links:

- [IDEA-Research/dab-detr-resnet-101](https://huggingface.co/IDEA-Research/dab-detr-resnet-101)
- [IDEA-Research/dab-detr-resnet-101-dc5](https://huggingface.co/IDEA-Research/dab-detr-resnet-101-dc5)

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dab_detr/modeling_dab_detr.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dab_detr/configuration_dab_detr.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/conditional_detr/image_processing_conditional_detr.py

## 2. High-level architecture

DAB-DETR is an image object detector with a CNN backbone, flattened transformer encoder, query-driven non-autoregressive transformer decoder, and detection heads. It has no text generation, no KV cache, and no autoregressive decode loop.

```text
CPU image resize/rescale/normalize/pad -> NCHW pixel_values + pixel_mask
  -> ResNet/timm backbone -> final feature map + downsampled mask + sine 2D feature positions
  -> 1x1 projection + flatten Hf*Wf tokens -> encoder self-attention
  -> learned 4D anchor/refpoint queries -> decoder self-attention + encoder cross-attention
  -> class logits + iterative bbox refinement -> ConditionalDetr postprocess -> per-image boxes/scores/labels
```

Stage decomposition:

- CPU/data pipeline: resize shortest edge 800, cap longest edge 1333, rescale by 1/255, ImageNet normalize, pad to batch max, emit `pixel_mask`.
- Backbone stage: NCHW ResNet-style feature extraction with frozen BatchNorm replacement. This can compose a separately audited ResNet/timm backbone if DinoML routes it through an existing backend.
- Transformer stage: single flattened image token sequence, encoder layers, then fixed query decoder layers. Encoder output can be cached only within one image request; this is not a generation KV cache.
- Postprocess stage: sigmoid multi-label top-k, threshold, center-to-corner box conversion, optional target-size scaling. No source NMS.

## 3. Important config dimensions

| Field | resnet-50 | resnet-50-dc5 | Source/default notes |
|---|---:|---:|---|
| `hidden_size` | 256 | 256 | transformer width |
| `encoder_layers` / `decoder_layers` | 6 / 6 | 6 / 6 | symmetric |
| `encoder_attention_heads` / `decoder_attention_heads` | 8 / 8 | 8 / 8 | MHA |
| encoder head dim | 32 | 32 | `256 / 8` |
| decoder self-attn Q/K/V head dim | 32 | 32 | self-attn width 256 |
| decoder cross-attn Q/K head dim | 64 | 64 | Q/K width is `2 * hidden_size = 512` |
| decoder cross-attn V head dim | 32 | 32 | V/output width is 256 |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 2048 / 2048 | 2048 / 2048 | PReLU activation |
| `num_queries` | 300 | 300 | learned anchor/refpoint slots |
| `query_dim` | 4 | 4 | strict config validation rejects other values |
| `num_patterns` | 0 | 0 | source supports pattern expansion if > 0 |
| `random_refpoints_xy` | false | false | source supports frozen random x/y init |
| `keep_query_pos` | false | false | affects cross-attn query-pos projection after layer 0 |
| `auxiliary_loss` | false | false | intermediate states still produced by decoder for bbox refinement |
| backbone | `resnet50` | `resnet50` | config field |
| backbone output stride | inferred 32 | 16 | DC5 config has `backbone_kwargs.output_stride=16` |
| final backbone channels | inferred 2048 | inferred 2048 | from ResNet stage4 convention/backbone `.channels` |
| input projection | Conv2d `2048 -> 256`, 1x1 | same | channel count depends on backbone |
| labels/classes | 91 `id2label` entries | 91 `id2label` entries | from config JSON; includes COCO `N/A` holes |
| cache support | none | none | no autoregressive path |

Representative checkpoint sweep:

| Checkpoint | Access | Operator-significant variation |
|---|---|---|
| `IDEA-Research/dab-detr-resnet-50` | OK | standard ResNet-50 stride-32 final feature map |
| `IDEA-Research/dab-detr-resnet-50-dc5` | OK | DC5/output-stride-16 backbone, longer flattened encoder sequence and higher attention cost |
| [IDEA-Research/dab-detr-resnet-101](https://huggingface.co/IDEA-Research/dab-detr-resnet-101) | 401 | would confirm deeper backbone only; transformer config likely same, but access is needed |
| [IDEA-Research/dab-detr-resnet-101-dc5](https://huggingface.co/IDEA-Research/dab-detr-resnet-101-dc5) | 401 | would confirm deeper DC5 backbone; access is needed |

Preprocessor config for accessible checkpoints is identical: `ConditionalDetrImageProcessor`, RGB ImageNet mean/std, resize shortest edge 800/longest edge 1333, padding enabled, COCO detection format.

## 3a. Family variation traps

- The decoder cross-attention uses rectangular width: Q/K are 512-wide after content+sine/key-position concatenation, while V/output remain 256-wide.
- DC5 changes feature-map stride from about 32 to 16, which multiplies encoder token count and cross-attention K/V length.
- `query_dim` is source-validated to exactly 4, so DinoML should reject configs that request 2D reference points for this native class.
- `num_patterns > 0` repeats queries and reference anchors to `num_queries * num_patterns`; accessible checkpoints do not use it.
- `keep_query_pos=False` removes the cross-attention query-position projection for decoder layers after the first; fusing weights across layers must respect this structural difference.
- `random_refpoints_xy=True` freezes only the first two anchor coordinates after inverse-sigmoid random initialization; accessible checkpoints do not use it.
- Public preprocessing/model tensor layout is NCHW. NHWC should be a guarded local optimization around backbone/projection regions, with no-layout-translation guards around `flatten(2).permute(0,2,1)`, channel-axis concat/gather assumptions, and mask interpolation.
- Source output docs mention no-object in copied text, but the native head is `Linear(hidden_size -> config.num_labels)` and official postprocess uses sigmoid multi-label top-k, not DETR softmax over `num_classes + 1`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors `[B,3,H,W]`, pixel masks `[B,H,W]`.
- `interpolate` mask downsample to feature map size, bool cast.
- Conv feature flatten: `[B,256,Hf,Wf] -> [B,Hf*Wf,256]`.
- Position flatten: `[B,256,Hf,Wf] -> [B,Hf*Wf,256]`.
- Query repeat/tiling for batch and optional patterns.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `cat`, `stack`, `gather`, `topk`, boolean threshold indexing.

Neural network primitives:

- Backbone Conv2d/BatchNorm2d/ReLU/ResNet ops, or an external backbone contract. Frozen BN runtime math is `x * weight / sqrt(var+1e-5) + bias - mean * scale`.
- Input projection `Conv2d(C_backbone -> 256, kernel=1, bias=True)`, normally `2048 -> 256`.
- Linear projections with bias:
  - encoder Q/K/V/O `Linear(256 -> 256)`.
  - encoder FFN `Linear(256 -> 2048)`, PReLU, `Linear(2048 -> 256)`.
  - decoder self Q content/Q pos/K content/K pos/V each `Linear(256 -> 256)`, output `Linear(256 -> 256)`.
  - decoder cross Q content/Q pos/K content/K pos/V/sine each `Linear(256 -> 256)`, attention output `Linear(256 -> 256)`.
  - query-scale MLPs `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256)`.
  - ref-point head `Linear(512 -> 256) -> ReLU -> Linear(256 -> 256)`.
  - ref-anchor head `Linear(256 -> 256) -> ReLU -> Linear(256 -> 2)`.
  - bbox MLP `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 4)`.
  - class head `Linear(256 -> num_labels)`.
- LayerNorm over last dim 256.
- Dropout is present in source but disabled for inference.

Attention primitives:

- Dense noncausal encoder self-attention over `S=Hf*Wf`.
- Dense noncausal decoder self-attention over `Q=num_queries` or `num_queries*num_patterns`.
- Dense noncausal decoder cross-attention with query length Q and key/value length S; Q/K width 512, V width 256.
- Additive bidirectional masks from `create_bidirectional_mask`.
- Softmax in fp32 then cast back to query dtype.

Position/custom math ops:

- 2D normalized sine image position embedding from `pixel_mask.cumsum` with separate width/height temperatures.
- Anchor/reference sine embedding for 4 coordinates with `scale=2*pi`, coordinate order swap for x/y.
- `inverse_sigmoid` clamp/logit, sigmoid, per-layer bbox refinement.
- Height/width attention modulation by predicted anchor size divided by current reference `w/h`.

Preprocessing/postprocessing-coupled ops:

- Conditional DETR image resize/rescale/normalize/pad and pixel mask.
- Postprocess sigmoid, flatten class/query grid, top-k, box gather, center-to-corner conversion, scale to `(height,width)`, score threshold. No NMS.

Parameter sharing:

- `model.decoder.bbox_embed` is the same logical module as top-level `bbox_predictor` (`_tied_weights_keys`), and lowering must preserve this alias rather than clone independent weights.

## 5. Layer/block breakdown

Backbone and flatten path:

```text
pixel_values: [B,3,H,W], pixel_mask: [B,H,W]
feature_maps = backbone(pixel_values).feature_maps
for each map [B,Ci,Hi,Wi]:
  mask_i = interpolate(pixel_mask[None].float(), size=(Hi,Wi)).bool()[0]
  pos_i = sine_2d_position(feature_map_i, mask_i) -> [B,256,Hi,Wi]
feature_map, mask = final map, normally [B,2048,Hf,Wf]
x = Conv2d(2048 -> 256, 1x1)(feature_map)
x = flatten spatial -> [B,S,256], S=Hf*Wf
object_queries = flatten final sine pos -> [B,S,256]
```

Encoder layer, repeated 6 times:

```text
pos_scales = MLP(256 -> 256 -> 256, ReLU)(x)
scaled_pos = object_queries * pos_scales
q = Linear(256 -> 256)(x + scaled_pos) * (32 ** -0.5)
k = Linear(256 -> 256)(x + scaled_pos)
v = Linear(256 -> 256)(x)
x = x + Linear(256 -> 256)(softmax(q @ k.T + mask) @ v)
x = LayerNorm(x)
y = Linear(256 -> 2048)(x)
y = PReLU(y)
y = Linear(2048 -> 256)(y)
x = LayerNorm(x + y)
```

Decoder setup:

```text
ref_logits = Embedding(num_queries, 4) repeated across batch
reference_points = sigmoid(ref_logits)       # [B,Q,4]
queries = zeros([B,Q,256]) unless num_patterns > 0
```

Decoder layer, repeated 6 times:

```text
query_sine_full = encode_anchor_sine(reference_points, num_pos_feats=128) # [B,Q,512]
query_pos = MLP(512 -> 256 -> 256, ReLU)(query_sine_full)
scale = 1 for layer 0 else MLP(256 -> 256 -> 256, ReLU)(hidden)
query_sine = query_sine_full[..., :256] * scale
anchor_hw = sigmoid(MLP(256 -> 256 -> 2, ReLU)(hidden))
query_sine[..., 128:256] *= anchor_hw[...,0] / reference_points[...,2]
query_sine[...,   0:128] *= anchor_hw[...,1] / reference_points[...,3]

# decoder self-attention over Q
q = Linear(256 -> 256)(hidden) + Linear(256 -> 256)(query_pos)
k = Linear(256 -> 256)(hidden) + Linear(256 -> 256)(query_pos)
v = Linear(256 -> 256)(hidden)
hidden = LayerNorm(hidden + MHA_256(q,k,v))

# cross-attention over encoder sequence S
q_content = Linear(256 -> 256)(hidden)
if first layer or keep_query_pos:
  q_content += Linear(256 -> 256)(query_pos)
k_content = Linear(256 -> 256)(encoder_hidden)
if first layer or keep_query_pos:
  k_content += Linear(256 -> 256)(object_queries)
q = concat_per_head(q_content, Linear(256 -> 256)(query_sine))  # [B,Q,512]
k = concat_per_head(k_content, Linear(256 -> 256)(object_queries)) # [B,S,512]
v = Linear(256 -> 256)(encoder_hidden)
hidden = LayerNorm(hidden + CrossAttention(q_width=512, v_width=256))

hidden = LayerNorm(hidden + Linear(2048 -> 256)(PReLU(Linear(256 -> 2048)(hidden))))
delta = bbox_mlp(hidden)                 # [B,Q,4]
reference_points = sigmoid(delta + inverse_sigmoid(reference_points)).detach()
intermediate += final_decoder_layernorm(hidden)
```

Detection head:

```text
logits = Linear(256 -> num_labels)(intermediate[-1])  # [B,Q,C]
bbox_delta = bbox_mlp(intermediate)                   # [L,B,Q,4]
pred_boxes = sigmoid(bbox_delta + inverse_sigmoid(reference_points))[-1]
```

All listed source Linear/Conv2d modules use bias unless otherwise noted by the loaded backbone.

## 6. Attention requirements

Encoder self-attention:

- Noncausal MHA, Q=K=V length `S=Hf*Wf`.
- 8 heads, Q/K/V head dim 32, output width 256.
- Object/query position is added before Q/K projection; V uses original hidden states.
- Mask is additive over key positions from flattened pixel mask.

Decoder self-attention:

- Noncausal MHA over object queries.
- Query length `Q=300` by default; `Q=300*num_patterns` if patterns enabled.
- 8 heads, Q/K/V head dim 32.
- No decoder attention mask is passed in the source default path.

Decoder cross-attention:

- Noncausal rectangular cross-attention from query slots to encoder image tokens.
- Query length Q, key/value length S.
- Q/K projection width 512, split as 8 heads of 64, built by concatenating 32-dim content and 32-dim sine/key-pos chunks per head.
- V width 256, split as 8 heads of 32. Output projection is 256 -> 256.
- Mask applies only to encoder key/value positions.
- No KV cache, no causal mask, no sliding-window/local attention, no RoPE/ALiBi.
- FlashAttention/SDPA compatibility requires a backend that supports `qk_head_dim != v_head_dim`; otherwise lower cross-attn to explicit matmul-softmax-matmul or pad/specialize with guards.

Eager fallback risk: encoder self-attention on DC5 stride-16 feature maps can be expensive because S roughly quadruples relative to stride-32 for the same input size.

## 7. Position encoding and custom math

Image 2D sine position embedding depends on the runtime pixel mask and feature map size:

```python
def dab_image_sine(mask, hidden=256, temp_h=20, temp_w=20, scale=2*pi):
    y = mask.cumsum(1, dtype=float32)
    x = mask.cumsum(2, dtype=float32)
    y = y / (y[:, -1:, :] + 1e-6) * scale
    x = x / (x[:, :, -1:] + 1e-6) * scale
    # sin/cos interleave for 128 dims each, concat [pos_y, pos_x], return NCHW
```

Anchor/reference sine embedding is per decoder layer and depends on dynamic reference points:

```python
def dab_anchor_sine(ref_xywh, num_pos_feats=128):
    dim = temperature ** (2 * floor(arange(num_pos_feats) / 2) / num_pos_feats)
    parts = []
    for coord in unbind(ref_xywh, -1):
        e = coord[..., None] * (2*pi) / dim
        parts.append(stack([sin(e[..., 0::2]), cos(e[..., 1::2])], -1).flatten(-2))
    parts[0], parts[1] = parts[1], parts[0]
    return cat(parts, -1)  # [B,Q,512] for xywh
```

Iterative reference update:

```python
def inverse_sigmoid(x, eps=1e-5):
    x = clamp(x, 0, 1)
    return log(clamp(x, eps) / clamp(1 - x, eps))

new_ref = sigmoid(bbox_mlp(hidden)[..., :4] + inverse_sigmoid(old_ref))
```

Precomputable: sine frequency denominators and static query embedding weights. Dynamic: image positions from padded masks, anchor sine embeddings, query scale, reference anchor size modulation, and iterative box references.

## 8. Preprocessing and input packing

CPU/data pipeline from `ConditionalDetrImageProcessor`:

- Converts images to RGB-like channel-first tensors, resizes with PIL resample id 2, rescales by `0.00392156862745098`, normalizes by mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Resize policy is shortest edge 800 and longest edge 1333.
- Pads a batch to the maximum resized `(height,width)` unless `pad_size` is supplied.
- Emits `pixel_values: [B,3,Hpad,Wpad]` and `pixel_mask: [B,Hpad,Wpad]`, with 1 for real pixels and 0 for padding.
- For detection labels, COCO annotations can be converted/resized/normalized, but labels and Hungarian loss are training-only for first inference integration.

GPU/runtime graph inputs:

- `pixel_values` and `pixel_mask` only for detection inference.
- Pixel masks are downsampled with float interpolation to each backbone feature map and cast to bool. DinoML should preserve this behavior because it affects position cumsum and attention masks at padded boundaries.

Postprocessing:

- Inputs: `outputs.logits [B,Q,C]`, `outputs.pred_boxes [B,Q,4]` in normalized center format.
- `sigmoid(logits)`, flatten `[Q*C]`, top-k up to 100, derive `topk_boxes=floor(index/C)` and `labels=index % C`.
- Convert selected boxes from center `(cx,cy,w,h)` to corners `(x0,y0,x1,y1)`.
- If `target_sizes` is supplied as `(height,width)`, multiply by `[width,height,width,height]`.
- Filter by `score > threshold`; return variable-length per-image dictionaries. No NMS is applied.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d input projection -> channel matmul

Source pattern: final backbone map `[B,C,Hf,Wf] -> Conv2d(C -> 256, kernel=1) -> flatten(2).permute(0,2,1)`.

Replacement:

```text
NCHWToTokens([B,C,H,W] -> [B,H*W,C]) -> MatMul(W.T) -> BiasAdd
```

Preconditions:

- Kernel size 1, stride 1, padding 0, dilation 1, groups 1.
- Consumer is exactly flattened image tokens; no other consumer requires the projected NCHW tensor.
- Source NCHW flatten order must match token order `h-major, w-minor`.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels).T
```

Failure cases: shared projected feature-map consumers, non-1x1 projection, changed backbone layout, or post-training quantization metadata that expects Conv2d storage.

Parity sketch: compare projected tokens before encoder for random `[B,C,H,W]` and official checkpoint weights.

### Rewrite: Frozen BatchNorm2d -> affine scale/bias

Source pattern:

```text
x * (weight / sqrt(running_var + 1e-5)) + (bias - running_mean * scale)
```

Replacement: precompute per-channel `scale` and `bias`, fuse into preceding Conv2d when Conv2d has a single consumer and source layout is stable.

Layout constraints: source axis is channel dim 1 in NCHW. In NHWC fusion, scale/bias axis becomes last dim and conv weight layout must be transformed together.

Failure cases: unfrozen training mode, reused BatchNorm outputs, or dynamic BN stats.

### Rewrite: guarded NHWC backbone/projection island

Source pattern: NCHW Conv/BN/activation backbone through final 1x1 projection.

Replacement: translate only a fully-contained backbone island to NHWC/channel-last for provider kernels, then materialize canonical `[B,S,256]` tokens at the boundary.

Preconditions:

- All internal consumers are controlled by the layout pass.
- Axis-sensitive ops are rewritten: channel dim `1 -> -1`, spatial dims preserved, mask remains `[B,H,W]`.
- Boundary output token order exactly matches NCHW `flatten(2).permute(0,2,1)`.

Failure cases: external backbone feature consumers, unsupported dilated DC5 conv in NHWC provider, or any op with ambiguous `view` semantics. Wrap transformer flatten/mask/postprocess in `no_layout_translation()`.

### Rewrite: decoder Linear projections -> packed QKV-like groups

Source pattern: separate content/pos/sine linear projections for self and cross attention.

Replacement: pack only projections with identical input tensor and identical shape, for example self-attn content Q/K/V from `hidden`. Do not pack query-pos, key-pos, and sine projections with content projections unless the replacement preserves their separate inputs and addition/concat order.

Preconditions:

- Same source tensor, same dtype, same batch/query shape, same bias behavior.
- Cross-attn packed kernel supports 512-wide Q/K and 256-wide V or keeps them separate.

Failure cases: `keep_query_pos` structural branch, layer 0 versus later layer differences, or cross-attn backends assuming Q/K/V equal head dim.

## 10. Kernel fusion candidates

Highest priority:

- Dense attention kernels for encoder self-attn and decoder self-attn. Encoder S can be large, especially DC5.
- Custom rectangular cross-attention with Q/K head dim 64 and V head dim 32. This is the most DAB-specific attention requirement.
- MLP + activation fusion for encoder/decoder FFNs `256 -> 2048 -> 256` with PReLU.
- Iterative bbox refinement fusion: bbox MLP + inverse-sigmoid + add + sigmoid over `[L,B,Q,4]`.

Medium priority:

- Backbone Conv+FrozenBN(+activation) fusion, especially for ResNet-50/101.
- 1x1 projection Conv2d-to-token-GEMM.
- Sine position embedding kernels or precomputed frequency tables with efficient cumsum/mask handling.
- Query sine modulation and per-head concat preparation fused into cross-attn input packing.

Lower priority:

- Postprocess top-k/threshold/box-scale on GPU. Useful for high-throughput serving, but CPU postprocess is acceptable for first parity.
- Optional `num_patterns` query expansion specialization.
- Output attentions materialization; not needed for production detection.

## 11. Runtime staging plan

1. Parse `DabDetrConfig`, processor JSON, and checkpoint weights; reject `query_dim != 4` for native source parity.
2. Compose or stub the ResNet/timm backbone behind a feature contract: final NCHW map, final mask, final sine positions.
3. Implement input projection, flatten, image sine positions, and one encoder layer parity.
4. Implement full encoder with bidirectional additive masks.
5. Implement decoder self-attn, DAB anchor sine/refpoint logic, rectangular cross-attn, and iterative bbox update.
6. Add class/bbox heads and Conditional-DETR object-detection postprocess.
7. Add DC5/output-stride-16 admission and performance probes.
8. Add guarded NHWC backbone/projection fusions and specialized cross-attn kernels.

Initially stub: labels/loss, gradient checkpointing, output attentions, ResNet-101 gated checkpoints, optional patterns if not needed by chosen deployment checkpoint.

## 12. Parity and validation plan

- Unit-test `inverse_sigmoid` around 0, 1, and mid-range values with fp32 tolerance `1e-6`.
- Unit-test image sine position embedding on hand-built masks, including padded rows/columns.
- Unit-test anchor sine encoding shape/order for `[B,Q,4] -> [B,Q,512]`.
- Compare FrozenBN affine rewrite and 1x1 projection-to-GEMM rewrite on random tensors.
- Single encoder layer parity with random `[B,S,256]`, object positions, and masks.
- Single decoder layer parity, specifically cross-attn with Q/K width 512 and V width 256.
- Full model parity at transformer boundary using saved backbone features, then end-to-end image parity.
- Postprocess parity for known logits/boxes: top-k labels, box gather, center-to-corner, scaling, threshold.
- Suggested tolerances: fp32 `1e-5` absolute for hidden states and boxes; fp16/bf16 `2e-2` hidden-state tolerance, verify postprocess labels/scores with a threshold margin to avoid borderline flips.

## 13. Performance probes

- Preprocessor throughput: resize/normalize/pad and pixel-mask creation.
- Backbone throughput for ResNet-50 stride-32 versus DC5 stride-16.
- Encoder-only throughput and memory as a function of image resolution and S.
- Decoder-only throughput with Q=300 and optional `num_patterns` sweep.
- Cross-attention backend comparison for unequal Q/K and V head dimensions.
- End-to-end images/sec by batch size and image aspect ratio.
- Postprocess top-k/threshold time and per-image result counts.
- NHWC backbone/projection island benchmark versus faithful NCHW translation.
- Attention memory usage for stride-32 versus stride-16 feature maps.

## 14. Skip/defer list

- Training losses, Hungarian matching, auxiliary-loss dictionaries, and annotation conversion.
- Gradient checkpointing and dropout behavior.
- Output attention tensors unless needed for debugging.
- Segmentation/panoptic heads; native DAB-DETR source here exposes object detection only.
- Multi-GPU/tensor parallelism.
- Quantization/packed weights; no source-coupled packed format was observed.
- Gated ResNet-101 checkpoint-specific confirmation until HF access is available.

## 15. Final implementation checklist

- [ ] Parse `DabDetrConfig` and reject unsupported `query_dim` values.
- [ ] Load DAB-DETR weights with `bbox_predictor` / `model.decoder.bbox_embed` alias preserved.
- [ ] Implement Conditional DETR image preprocess contract or accept equivalent `pixel_values`/`pixel_mask`.
- [ ] Implement/compose ResNet backbone with FrozenBN parity.
- [ ] Implement 2D image sine position embedding from masks.
- [ ] Implement 1x1 projection and NCHW-to-token flatten.
- [ ] Implement encoder self-attention and FFN blocks.
- [ ] Implement anchor sine embedding, query-scale, ref-anchor modulation, and iterative refpoint update.
- [ ] Implement decoder self-attention and rectangular decoder cross-attention.
- [ ] Implement class head, bbox head, and sigmoid top-k postprocess with no NMS.
- [ ] Add guarded Conv1x1-to-GEMM and FrozenBN fusion rewrites.
- [ ] Add guarded NHWC backbone/projection island with `no_layout_translation()` around token flatten/mask/postprocess.
- [ ] Add parity tests for custom math, one layer, full transformer, end-to-end image output, and postprocess.
- [ ] Benchmark stride-32 versus DC5 stride-16 throughput and attention memory.
