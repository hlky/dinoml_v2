# Table Transformer DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/table-transformer-detection; microsoft/table-transformer-structure-recognition; microsoft/table-transformer-structure-recognition-v1.1-{pub,fin,all}
Config source: official Hugging Face config.json and preprocessor_config.json snapshots under _sources/hf_configs/
Source files inspected:
- X:/H/transformers/src/transformers/models/table_transformer/configuration_table_transformer.py
- X:/H/transformers/src/transformers/models/table_transformer/modeling_table_transformer.py
- X:/H/transformers/src/transformers/models/detr/image_processing_detr.py for AutoImageProcessor preprocessing/postprocess
- X:/H/transformers/src/transformers/backbone_utils.py for backbone_config/load_backbone behavior
- X:/H/transformers/src/transformers/masking_utils.py for create_bidirectional_mask behavior
Any missing files or assumptions: no gated/401/403 gaps found. Table Transformer has no local image_processing_table_transformer.py; official checkpoints route to legacy DetrFeatureExtractor or DetrImageProcessor. Training losses/matching were not audited as first-target runtime features.
```

Primary runtime target: `TableTransformerForObjectDetection` inference for table detection and table-structure object detection.

Source URLs:
- [modeling_table_transformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/table_transformer/modeling_table_transformer.py)
- [configuration_table_transformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/table_transformer/configuration_table_transformer.py)
- [image_processing_detr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/detr/image_processing_detr.py)

## 2. High-level architecture

Table Transformer is a DETR-like image object detector: convolutional backbone plus encoder-decoder Transformer over flattened image features. It is not autoregressive generation and has no KV cache.

```text
CPU/PIL image preprocessing -> NCHW pixel_values + pixel_mask
  -> ResNet/AutoBackbone feature maps + downsampled masks
  -> 1x1 Conv projection to d_model
  -> flatten Hf*Wf sequence + 2D position encodings
  -> noncausal Transformer encoder
  -> learned object query decoder with self-attention + encoder cross-attention
  -> class logits + normalized bbox MLP
  -> softmax/threshold + cxcywh->xyxy + target-size scaling
```

Stageable pieces:
- Processor resize/rescale/normalize/pad and pixel-mask creation can stay in CPU/data pipeline first.
- Backbone can compose a separately audited ResNet/timm/AutoBackbone path; this report owns the feature contract and Transformer neck/heads.
- Encoder and decoder are independently parity-testable from projected feature maps and masks.
- Postprocess is required for end-to-end detection parity but is not part of the core GPU module graph.

## 3. Important config dimensions

| Field | Source/default | Detection | Structure v1 | Structure v1.1 |
|---|---:|---:|---:|---:|
| `d_model` | config/default | 256 | 256 | 256 |
| encoder / decoder layers | config/default | 6 / 6 | 6 / 6 | 6 / 6 |
| encoder / decoder heads | config/default | 8 / 8 | 8 / 8 | 8 / 8 |
| `head_dim` | inferred from source guard | 32 | 32 | 32 |
| encoder / decoder FFN | config/default | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 |
| activation | config/default | ReLU | ReLU | ReLU |
| dropout / attention dropout | config/default | 0.1 / 0.0 | 0.1 / 0.0 | 0.1 / 0.0 |
| `num_queries` | config | 15 | 125 | 125 |
| classes plus no-object | config + source | 2 + 1 | 6 + 1 | 6 + 1 |
| backbone | config | legacy `backbone=resnet18` | legacy `backbone=resnet18` | native ResNet basic, `[64,128,256,512]` |
| backbone features consumed | source | final returned map | final returned map | final returned map, usually `stage4` after all returned maps |
| position embedding | config | sine | sine | sine |
| learned position table option | source | 50 rows + 50 cols, optional | same | same |
| bbox format | source | normalized cx,cy,w,h after sigmoid | same | same |
| dtype | config metadata | float32 | float32 | float32 |
| cache support | source | none | none | none |

Representative checkpoint sweep:

| Checkpoint | Task labels | Queries | Processor size | Processor type | Config/backbone shape note |
|---|---:|---:|---|---|---|
| `microsoft/table-transformer-detection` | table, table rotated | 15 | `size=800`, `max_size=800` | `DetrFeatureExtractor` | legacy `backbone=resnet18`, no embedded `backbone_config` |
| `microsoft/table-transformer-structure-recognition` | 6 table-structure labels | 125 | `size=800`, `max_size=1000` | `DetrFeatureExtractor` | legacy `backbone=resnet18`, no embedded `backbone_config` |
| `microsoft/table-transformer-structure-recognition-v1.1-pub` | 6 labels | 125 | `{"longest_edge":800}` | `DetrImageProcessor` | native `backbone_config.model_type=resnet`, `out_features=stage1..stage4` |
| `microsoft/table-transformer-structure-recognition-v1.1-fin` | 6 labels | 125 | `{"longest_edge":800}` | `DetrImageProcessor` | same operator structure as v1.1 pub |
| `microsoft/table-transformer-structure-recognition-v1.1-all` | 6 labels | 125 | `{"longest_edge":800}` | `DetrImageProcessor` | same operator structure as v1.1 pub |

## 3a. Family variation traps

- Official checkpoints vary `num_queries` substantially: 15 for table detection, 125 for structure recognition. Decoder and head tensor sizes depend on this.
- Legacy configs carry `backbone`, `use_pretrained_backbone`, `max_position_embeddings`, and `scale_embedding`. Current Table Transformer source does not read `max_position_embeddings` or `scale_embedding`; backbone kwargs are consolidated through `backbone_utils`.
- Newer v1.1 configs embed a native ResNet `backbone_config` and set `use_timm_backbone=false`. Older configs rely on legacy `backbone=resnet18`; DinoML loading should either reproduce Transformers consolidation or reject unsupported legacy/timm variants explicitly.
- `position_embedding_type="learned"` changes custom math to fixed learned row/column tables of length 50 each; sine is used by sampled official checkpoints.
- Modeling code is NCHW at the backbone and Conv2d boundary. NHWC should be introduced only in guarded convolution/fusion regions, with `dim=1` channel ops, `flatten(2)`, `permute(0,2,1)`, mask interpolation, and public processor contracts protected by no-layout-translation guards.
- This is a DETR fork. The inspected source is mostly copied from DETR, but the official table checkpoints differ in classes, query counts, ResNet-18/table-specific weights, and processor resize limits.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW image tensors `[B,3,H,W]`, pixel masks `[B,H,W]`.
- `interpolate` for mask downsampling to feature-map size, then bool cast.
- Conv feature map flatten: `[B,256,Hf,Wf] -> [B,Hf*Wf,256]`.
- Position map flatten: `[B,256,Hf,Wf] -> [B,Hf*Wf,256]`.
- Query embedding broadcast/repeat: `[Q,256] -> [B,Q,256]`; zero query tensor allocation.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `stack`, boolean/float mask conversions.

Neural network primitives:
- Composed backbone, commonly ResNet-18/basic: Conv2d, FrozenBatchNorm2d, ReLU, maxpool/downsample/residual adds.
- FrozenBatchNorm2d inference math: `x * weight * rsqrt(running_var + 1e-5) + (bias - running_mean * scale)`.
- 1x1 Conv2d projection `C_backbone -> 256`; for ResNet-18 stage4 `512 -> 256`.
- LayerNorm(256) before attention/FFN and final encoder/decoder norms.
- Linear projections with bias: Q/K/V/O `Linear(256 -> 256)`.
- FFNs: encoder and decoder `Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)`.
- Detection classifier `Linear(256 -> num_labels+1)`, e.g. `256 -> 3` or `256 -> 7`.
- BBox MLP: `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 4) -> sigmoid`.

Attention primitives:
- Manual eager MHA via batched matmul, softmax over keys, dropout in training only, second batched matmul.
- Encoder bidirectional self-attention over image tokens `[B,S,S]`, with spatial position embeddings added before Q/K projection.
- Decoder bidirectional self-attention over object queries `[B,Q,Q]`, with learned query position embeddings added before Q/K projection.
- Decoder cross-attention query length `Q`, key/value length `S`, spatial position embeddings added to K path only and learned query positions added to Q path.
- Attention masks are 4D additive masks `[B,1,T,S]` or skipped when fully valid.

Position/custom ops:
- 2D sine position embeddings from cumulative pixel masks, normalized to `2*pi`.
- Optional learned row/column embeddings for feature-map dimensions up to 50.

Preprocessing/postprocessing ops:
- Resize preserving aspect ratio or longest edge, rescale by `1/255`, normalize by ImageNet mean/std, batch pad, `pixel_mask`.
- Postprocess softmax excluding no-object, threshold, center-to-corners box conversion, scale by target `(height,width)`.
- No NMS in source postprocess.

Distributed/tensor parallel:
- None in source.

## 5. Layer/block breakdown

Backbone and projection:
```text
pixel_values: [B,3,H,W], pixel_mask: [B,H,W]
features = AutoBackbone(pixel_values) -> list NCHW maps
for each map: mask = interpolate(pixel_mask[None].float(), feature_hw).bool()[0]
feature_map, mask = final feature
projected = Conv2d(C_backbone -> 256, kernel=1)(feature_map)
src = projected.flatten(2).permute(0,2,1)       # [B,S,256]
pos = position_embedding(feature_map, mask).flatten(2).permute(0,2,1)
flat_mask = mask.flatten(1)                     # [B,S], 1=valid
```

Encoder layer, repeated 6 times:
```text
residual = x
x = LayerNorm(256)(x)
q,k input = x + pos
v input = x
q,k,v = Linear(256 -> 256, bias=True)
x = MHA(q,k,v, bidirectional pixel mask)
x = residual + Linear(256 -> 256)(x)
residual = x
x = LayerNorm(256)(x)
x = Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)
x = residual + x
```
The encoder applies a final LayerNorm(256).

Decoder layer, repeated 6 times:
```text
queries = zeros_like(query_pos)                  # [B,Q,256]
query_pos = Embedding(num_queries,256).repeat(B)

residual = x
x = LayerNorm(256)(x)
self q,k input = x + query_pos
self v input = x
x = bidirectional MHA over Q query slots
x = residual + out

residual = x
x = LayerNorm(256)(x)
cross q input = x + query_pos
cross k input = encoder_hidden + spatial_pos
cross v input = encoder_hidden
x = cross MHA, query length Q, key/value length S, encoder pixel mask
x = residual + out

residual = x
x = LayerNorm(256)(x)
x = Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)
x = residual + x
```
The decoder applies a final LayerNorm(256). If `auxiliary_loss=True`, intermediate normalized decoder states are stacked, but sampled inference configs set it false.

Detection head:
```text
sequence_output: [B,Q,256]
logits = Linear(256 -> num_labels+1)(sequence_output)
pred_boxes = sigmoid(MLP_3layer_256_to_4(sequence_output))  # normalized cxcywh
```

## 6. Attention requirements

- Encoder attention: noncausal self-attention, MHA, 8 heads, head dim 32, Q/K/V width 256. Query length and KV length are both `S=Hf*Wf`.
- Decoder self-attention: noncausal self-attention, MHA, 8 heads, head dim 32, Q/K/V width 256, length `Q=num_queries`.
- Decoder cross-attention: noncausal cross-attention, MHA, 8 heads, head dim 32. Query source is learned query slots plus current decoder state, shape `[B,Q,256]`; K/V source is encoder hidden states, shape `[B,S,256]`. Rectangular attention shape is `[B,8,Q,S]`.
- No MQA/GQA, RoPE, ALiBi, sliding window, sparse attention, packed varlen attention, or autoregressive KV cache.
- Masking: 2D mask `[B,S]` or `[B,Q]` is converted to a bidirectional 4D additive mask `[B,1,T,S]`; bool masks in the attention module are converted to `-inf`, otherwise additive masks are consumed as-is.
- Backend: the Table Transformer attention implementation is manual eager `torch.bmm`/softmax, not Transformers SDPA/FlashAttention dispatch. A fused attention backend is safe only if it preserves the source order: add positions before Q/K projection, scale Q before scores, add mask before softmax, and use unpositioned tensors for V.

## 7. Position encoding and custom math

Sine 2D position embeddings depend on the downsampled pixel mask and feature-map size; they are batch/dynamic because padding changes cumulative coordinates.

```python
def table_sine_pos(pixel_mask, dim=128, temperature=10000, scale=2 * math.pi):
    y = pixel_mask.cumsum(1, dtype=float)
    x = pixel_mask.cumsum(2, dtype=float)
    y = y / (y[:, -1:, :] + 1e-6) * scale
    x = x / (x[:, :, -1:] + 1e-6) * scale
    dim_t = temperature ** (2 * floor(arange(dim) / 2) / dim)
    px = stack([sin(x[..., 0::2] / dim_t[0::2]), cos(x[..., 1::2] / dim_t[1::2])]).flatten()
    py = stack([sin(y[..., 0::2] / dim_t[0::2]), cos(y[..., 1::2] / dim_t[1::2])]).flatten()
    return concat([py, px], channel=-1).to_nchw()
```

Attention position placement:
```python
q = Wq(hidden + query_or_spatial_pos) * (head_dim ** -0.5)
k = Wk(key_value + spatial_pos)
v = Wv(original_key_value_without_pos)
```

Optional learned position embeddings use `row_embeddings[0:Hf]` and `column_embeddings[0:Wf]`, concatenate to 256 channels, then broadcast over batch. Guard `Hf <= 50` and `Wf <= 50`.

## 8. Preprocessing and input packing

Processor contract:
- Inputs are RGB images. The processor resizes, rescales, normalizes, pads, and returns `pixel_values` in NCHW plus `pixel_mask`.
- Legacy checkpoint preprocessors use `feature_extractor_type="DetrFeatureExtractor"`, `size=800`, and `max_size` 800 or 1000. Current `DetrImageProcessor` defaults convert this style to shortest/longest-edge resize semantics.
- v1.1 preprocessors use `image_processor_type="DetrImageProcessor"`, `do_resize=true`, `do_rescale=true`, `rescale_factor=1/255`, `do_normalize=true`, `do_pad=true`, `size={"longest_edge":800}`.
- Normalization uses ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Padding is to the batch max height/width unless a pad size is supplied; `pixel_mask[:original_h,:original_w]=1`, padding is 0.

Postprocess contract:
- Inputs: `logits [B,Q,C+1]`, `pred_boxes [B,Q,4]` normalized center coordinates, optional `target_sizes [B,2]` as `(height,width)`.
- `softmax(logits, -1)`, ignore last no-object class, take max class per query.
- Convert boxes `cx,cy,w,h -> x0,y0,x1,y1`; if target sizes provided, multiply by `[width,height,width,height]`.
- Filter with `score > threshold` per image and return variable-length `{scores, labels, boxes}`.
- No source NMS; duplicate detections are possible by design.

CPU/data-pipeline first: image decode, resize, rescale, normalization, padding, and target-size metadata. GPU/runtime first: backbone, Transformer, heads. Postprocess can initially be host-side.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d projection -> per-pixel Linear

Source pattern: `Conv2d(C_backbone -> 256, kernel=1)` on final feature map.

Replacement:
```text
NCHW feature -> optional NHWC/local channel-last view -> MatMul(C_backbone,256) + bias -> restore required layout
```

Preconditions:
- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Only the projection and immediate flatten consumer are in the rewritten region.

Weight transform:
```python
w_linear = conv.weight.reshape(256, C_backbone).T
b_linear = conv.bias
```

Failure cases: non-1x1 projection, grouped conv, consumers requiring original NCHW feature map before projection. Parity sketch: compare projected NCHW and flattened `[B,S,256]` for random feature maps.

### Rewrite: static FrozenBatchNorm2d fold into Conv2d

Source pattern: ResNet Conv2d followed by `TableTransformerFrozenBatchNorm2d`.

Replacement: fold scale/bias into preceding Conv2d weights and bias.

Preconditions:
- Conv output is consumed only by that frozen BN.
- Frozen BN buffers are constants at inference.
- Preserve NCHW channel axis or explicitly rewrite axis for NHWC.

Weight transform:
```python
scale = weight * rsqrt(running_var + 1e-5)
w2 = conv.weight * scale.reshape(-1, 1, 1, 1)
b2 = (conv.bias or 0) * scale + bias - running_mean * scale
```

Failure cases: shared conv output before BN, training mode, mutable BN buffers.

### Rewrite: DETR flatten/permute layout elimination

Source pattern: `projected.flatten(2).permute(0,2,1)` and same for position maps.

Replacement: if projection/backbone tail is already in NHWC inside a guarded fused region, emit `[B,Hf,Wf,C] -> [B,Hf*Wf,C]` without an explicit transpose.

Preconditions:
- All consumers in the region consume sequence-last-channel `[B,S,C]`.
- Pixel mask remains `[B,Hf,Wf]` and flatten order is row-major equivalent to PyTorch NCHW flatten(2).
- Source-facing outputs and backbone contracts remain NCHW unless explicitly materialized.

Failure cases: exposing intermediate NCHW feature maps, learned position embedding code that assumes channel-first before flatten, or any `dim=1` channel op not rewritten. Use a conceptual `no_layout_translation()` guard around processor, pixel mask, and public backbone-output ABI.

### Rewrite: class + bbox heads as batched GEMMs

Source pattern: per-query Linear/MLP over `[B,Q,256]`.

Replacement: flatten to `[B*Q,256]`, run GEMMs, reshape back.

Preconditions: no query-dependent control flow; activation exactly ReLU for first two bbox layers, final sigmoid preserved.

Parity sketch: random decoder output, compare logits and boxes.

## 10. Kernel fusion candidates

Highest priority:
- ResNet backbone Conv2d + FrozenBatchNorm2d + ReLU folding/fusion. Most FLOPs are in the backbone and official checkpoints use ResNet-style feature extraction.
- Encoder attention over `S=Hf*Wf` image tokens. Sequence length grows with resolution; fused noncausal MHA can replace slow eager `bmm`/softmax if mask and position-add semantics are preserved.
- Decoder cross-attention `[B,Q,S]`, especially structure models with `Q=125`; this is the main table-specific query-to-image operation.

Medium priority:
- 1x1 projection plus flatten to `[B,S,256]`; good place for guarded NHWC local optimization.
- LayerNorm + Q/K/V projections in encoder/decoder blocks.
- FFN `Linear -> ReLU -> Linear` for both encoder/decoder.

Lower priority:
- Bbox MLP and class head batched GEMM; small but easy.
- Postprocess softmax/threshold/box conversion on GPU; useful for high-throughput batches but host-side is acceptable initially.
- Learned position embedding path; sampled official checkpoints use sine.

## 11. Runtime staging plan

1. Parse configs, including legacy `backbone` fields and native `backbone_config`; reject unsupported timm/legacy variants until loader parity is explicit.
2. Load weights and run backbone-output contract parity using a separately audited ResNet path or a stubbed captured feature map.
3. Implement Table Transformer neck: sine position embedding, 1x1 projection, flatten, mask flatten.
4. Implement one encoder layer and one decoder layer parity from random tensors and masks.
5. Implement full `TableTransformerModel` encoder-decoder parity.
6. Add object-detection heads and postprocess parity for detection and structure checkpoints.
7. Add optimized attention and guarded layout/fusion passes after functional parity.

Initially stubbable: training losses, auxiliary-loss outputs, backbone weight conversion variants, GPU postprocess, learned positional embeddings if no target checkpoint uses them.

## 12. Parity and validation plan

- Custom op tests: FrozenBatchNorm2d, sine position embeddings from masks, learned position table guard, center-to-corners box conversion, postprocess thresholding.
- Shape tests: detection `Q=15`, structure `Q=125`, varying padded image sizes and masks.
- Single-layer parity: encoder layer with `[B,S,256]`, decoder self-attention with `[B,Q,256]`, cross-attention with `[B,Q,S]`.
- Neck parity: final ResNet feature map `[B,512,Hf,Wf]` through 1x1 projection, position embedding, flatten, mask flatten.
- Full model parity: logits `[B,Q,C+1]` and boxes `[B,Q,4]` for official checkpoints.
- End-to-end parity: processor output, model output, postprocessed boxes for representative table images.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=1e-2, atol=1e-2`, with postprocess label/threshold tests avoiding scores too close to threshold.

No DinoML tests were run for this audit.

## 13. Performance probes

- Processor throughput: resize/rescale/normalize/pad for document image batches.
- Backbone-only throughput by input resolution and batch size.
- Encoder-only sweep by feature sequence length `S`.
- Decoder-only sweep by `Q=15` vs `Q=125` and `S`.
- Cross-attention backend comparison: eager vs fused noncausal attention.
- End-to-end images/sec for detection and structure recognition checkpoints.
- Memory probes: feature maps, encoder activations, attention score tensors `[B,8,S,S]` and `[B,8,Q,S]`.
- Layout probe: NCHW baseline vs guarded NHWC conv/projection tail with explicit materialization points.
- Postprocess throughput and variable-length output allocation overhead.

## 14. Skip/defer list

- Training, Hungarian matching, bbox/GIoU loss, auxiliary decoding losses.
- Segmentation/panoptic DETR processor paths; `TableTransformerForObjectDetection` does not emit masks.
- Autoregressive generation, beam search, KV cache, speculative decoding.
- Multi-GPU tensor parallelism.
- Timm-backbone runtime parity for old configs until admission rules decide whether to support it directly or convert to native backbone configs.
- Learned position embeddings unless a target checkpoint requires `position_embedding_type="learned"`.

## 15. Final implementation checklist

- [ ] Parse `TableTransformerConfig`, including legacy backbone fields and native `backbone_config`
- [ ] Load Table Transformer and backbone weights
- [ ] Compose or import ResNet/AutoBackbone feature contract
- [ ] Implement FrozenBatchNorm2d inference/fold
- [ ] Implement image mask downsampling and flattening
- [ ] Implement 2D sine position embedding from pixel masks
- [ ] Implement 1x1 projection and `[B,C,H,W] -> [B,H*W,C]` sequence conversion
- [ ] Implement bidirectional MHA with position-add-before-QK semantics
- [ ] Implement encoder and decoder blocks
- [ ] Implement object query embedding initialization
- [ ] Implement class head and bbox MLP with sigmoid
- [ ] Implement object-detection postprocess without NMS
- [ ] Add Conv+FrozenBN fold rewrite
- [ ] Add guarded 1x1 Conv/flatten NHWC rewrite
- [ ] Add encoder/decoder/cross-attention parity tests
- [ ] Benchmark processor, backbone, encoder, decoder, and postprocess separately
