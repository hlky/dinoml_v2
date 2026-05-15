# OmDet-Turbo Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: primary in-library checkpoint [`omlab/omdet-turbo-swin-tiny-hf`](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf). Also checked [`Blueway/inference-endpoint-for-omdet-turbo-swin-tiny-hf`](https://huggingface.co/Blueway/inference-endpoint-for-omdet-turbo-swin-tiny-hf) as a derivative config copy, and [`omlab/OmDet-Turbo_tiny_SWIN_T`](https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T), which is public but has only `.pth`/CLIP files and no HF `config.json`.

Config source: saved snapshots under `_sources/` for the official HF `config.json`, `preprocessor_config.json`, and `tokenizer_config.json`.

Source files inspected:

- `src/transformers/models/omdet_turbo/configuration_omdet_turbo.py`
- `src/transformers/models/omdet_turbo/modeling_omdet_turbo.py`
- `src/transformers/models/omdet_turbo/processing_omdet_turbo.py`

Any missing files or assumptions: no tokenizer implementation is local to this family; the processor couples to `CLIPTokenizer`. The vision body is delegated through `AutoBackbone` and, for the representative checkpoint, `use_timm_backbone=true` with `swin_tiny_patch4_window7_224`; this report owns the OmDet-Turbo wrapper, neck, fusion, decoder, and postprocess, but Swin/timm backbone operator coverage should compose separate Swin/timm audits. No gated configs were found; the older official `omlab/OmDet-Turbo_tiny_SWIN_T` repo returned 404 for `config.json`.

## 2. High-level architecture

Primary runtime target: zero-shot/open-vocabulary object detection inference with `OmDetTurboForObjectDetection`.

Dataflow:

```text
image resize/normalize + class/task tokenization
-> vision backbone feature maps + CLIP text features
-> 1x1 channel projection + transformer-on-one-level + FPN/PAN neck
-> prompt/class fusion + top-k query proposal selection
-> deformable transformer decoder
-> class similarity logits + normalized center boxes
-> processor threshold/topk/NMS/scale/clip
```

Stage decomposition:

- CPU/data pipeline: image resize/normalize to `pixel_values`, CLIP tokenization for class labels and task prompt, `classes_structure` packing.
- Cacheable text stage: class embeddings and task embeddings are cached by token id/mask keys in model-owned LRU caches. For DinoML, these can become separately compiled/cacheable text-encoder calls or precomputed inputs.
- Vision stage: AutoBackbone emits image-like NCHW feature maps for three scales; representative checkpoint uses Swin tiny via timm.
- OmDet-owned neck: channel projections, one encoder self-attention pass on selected level, FPN/PAN convolutional aggregation.
- Decoder: proposal top-k, query/task dense self-attention fusion, multi-scale deformable cross-attention over flattened vision levels, FFN, iterative box refinement.
- Postprocess: sigmoid class scores, class filtering, center-to-corners box conversion, scale to target sizes, per-class batched NMS, clipping.

## 3. Important config dimensions

Representative checkpoint: `omlab/omdet-turbo-swin-tiny-hf` config-derived unless noted.

| Field | Value | Notes |
|---|---:|---|
| `model_type` | `omdet-turbo` | In-library source. |
| `image_size` | 640 | Processor resizes to 640 x 640. |
| `backbone` | `swin_tiny_patch4_window7_224` | `use_timm_backbone=true`, `out_indices=[1,2,3]`, `always_partition=true`. |
| `encoder_in_channels` | `[192, 384, 768]` | Backbone feature channels consumed by first projection. |
| `vision_features_channels` | `[256, 256, 256]` | FPN/PAN outputs consumed by decoder projection. |
| `encoder_hidden_dim` / `d_model` | 256 | Neck/encoder/deformable attention width. |
| `encoder_attention_heads` | 8 | Head dim 32. |
| `encoder_layers` | 1 | Applied only to `encoder_projection_indices=[2]`. |
| `decoder_hidden_dim` | 256 | Decoder query/task width after projection. |
| `decoder_num_heads` | 8 | Dense self-attn and deformable cross-attn heads. |
| `decoder_num_layers` | 6 | In inference source breaks after final layer logits. |
| `decoder_num_points` | 4 | Per-level deformable samples. |
| `num_feature_levels` | 3 | FPN/PAN/deformable levels. |
| `num_queries` | 900 | Top-k selected proposals. |
| `class_embed_dim` | 512 | CLIP text feature/class space. |
| `task_encoder_hidden_dim` | 1024 | Task prompt MLP hidden. |
| `class_distance_type` | `cosine` | Alternative source path supports `dot`. |
| `text_config` | `clip_text_model` | Defaults supplied by config if omitted. |
| `torch_dtype` | `float32` | Checkpoint config metadata. |

Checkpoint sweep:

| Repo | Accessible config | Operator-significant result |
|---|---|---|
| [`omlab/omdet-turbo-swin-tiny-hf`](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) | yes | Main HF config; timm Swin tiny, CLIP text, 640 image, 900 queries. |
| [`Blueway/inference-endpoint-for-omdet-turbo-swin-tiny-hf`](https://huggingface.co/Blueway/inference-endpoint-for-omdet-turbo-swin-tiny-hf) | yes | Same architecture/config as official checkpoint; derivative endpoint copy. |
| [`omlab/OmDet-Turbo_tiny_SWIN_T`](https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T) | no HF config | Public original-weight repo with `.pth` and `ViT-B-16.pt`; use only as out-of-scope raw-weight source without HF config ABI. |

Effective defaults to remember when configs omit fields: `disable_custom_kernels=false`, `apply_layernorm_after_vision_backbone=true`, `class_distance_type="cosine"`, `learn_initial_query=false`, `cache_size=100`, default `text_config=clip_text_model`, and default backbone consolidation to Swin tiny with timm kwargs or native Swin config.

## 3a. Family variation traps

- `backbone_config` is delegated. DinoML should allowlist known Swin/timm bodies and reject unreviewed `AutoBackbone` variants until separately audited.
- Source assumes image-like NCHW features after the vision backbone. The layernorm wrapper normalizes a channel-last view and then permutes back to NCHW; this is a guarded layout opportunity, not a default ABI change.
- Open-vocabulary class count is runtime data: `classes_structure` controls per-image valid labels after batched padding to the max class count.
- Text embeddings are not ordinary classifier weights. Class labels and task prompts are CLIP token sequences, cached by token ids and projected/pooled at runtime.
- Class embedding ABI is unusual: class embedding pooling uses `hidden_states.argmax(dim=-1)` over token ids, which effectively selects the max token id position in each class prompt, then applies `text_projection`.
- Query selection depends on `topk(max_class_similarity, num_queries)` across flattened multi-level anchors, so detection order and top-k tie behavior are parity-sensitive.
- Postprocess includes `torchvision.ops.batched_nms`; raw model outputs alone are not end-to-end detection parity.
- Multi-scale deformable attention has a custom kernel hook (`use_kernel_forward_from_hub("MultiScaleDeformableAttention")`) and an eager `grid_sample` fallback. DinoML should model it as a custom provider/fallback family.
- Processor docs/examples in the source appear historically drifted in argument names (`classes`/`score_threshold` in example versus `text_labels`/`threshold` in function). Use function signature as authoritative.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW feature maps, `flatten(2)`, `permute(0,2,1)`, `reshape`, `contiguous`, `torch.cat` along channel and sequence axes.
- `topk`, advanced indexing/gather for top-k proposals, `stack`, `repeat`, `unsqueeze`, `transpose`, `view`.
- Dynamic-ish sequence lengths from class count, task token length, and multi-scale `sum(H_l * W_l)`.

Neural network primitives:

- Conv2d 1x1 and 3x3, BatchNorm2d, LayerNorm, Linear, Embedding if `learn_initial_query=true`, dropout as identity in inference.
- Activations: GELU, SiLU, ReLU.
- Dense MLPs: box heads `Linear(256->256)->Linear(256->256)->Linear(256->4)`, task MLP `512->1024->512`, FFNs `256->2048->256`.

Attention primitives:

- Dense noncausal MHA for encoder and decoder query/task self-attention, MHA 8 heads x 32 dim.
- Multi-scale deformable cross-attention with value projection 256->256, offsets `256 -> heads * levels * points * 2`, weights `256 -> heads * levels * points`, softmax over `levels*points`, bilinear `grid_sample`.

Position/custom math:

- 2D sin/cos absolute position embedding for selected encoder feature map.
- Anchor generation over each level, inverse sigmoid/logit transform, valid mask, iterative sigmoid box refinement.
- Cosine class similarity: L2 normalize query/class features, multiply by `exp(log(1/0.07))`.

Preprocessing-coupled ops:

- DetrImageProcessor resize/normalize, CLIPTokenizer tokenization, task prompt construction, class flattening and `classes_structure`.

Postprocessing ops:

- Sigmoid class logits, flatten class scores, top-k, boolean filtering, center-to-corners conversion, scale by target `(width,height,width,height)`, per-class `batched_nms`, clamp boxes.

## 5. Layer/block breakdown

Text branch:

```text
CLIP text model(input_ids)
if class:
  pooled = last_hidden_state[batch, argmax(input_ids)]
  class_feature = pooled @ text_projection       # 512 -> 512
if task:
  task_tokens = last_hidden_state[:, :max_nonpad]
  return task_tokens.T and truncated mask
```

Vision branch:

```text
pixel_values [B,3,640,640]
-> AutoBackbone feature maps, representative channels [192,384,768]
-> optional LayerNorm per feature map and permute back to NCHW
```

Hybrid encoder/neck:

```text
for each backbone level:
  Conv1x1(C_l -> 256, bias=False) + BatchNorm2d
selected level 2:
  flatten NCHW -> [B, H*W, 256]
  add 2D sin/cos position
  1 x encoder layer:
    dense MHA + residual + LayerNorm
    Linear(256->2048) + ReLU + Linear(2048->256) + residual + LayerNorm
  reshape back to NCHW
FPN:
  lateral ConvNorm + nearest upsample x2 + concat channel + CSPRepLayer
PAN:
  stride-2 ConvNorm + concat channel + CSPRepLayer
```

Decoder proposal and fusion:

```text
project 3 FPN/PAN maps with Conv1x1+BN
flatten and concat levels -> vision_features [B, sum(HW), 256]
generate normalized anchors per level -> logit anchors + valid_mask
class logits = cosine/dot(predicted_vision_features, projected_class_features)
topk over max class score -> 900 query indices
query embeddings = selected vision features unless learn_initial_query
reference_points = selected bbox logits -> sigmoid boxes
```

Decoder layer, repeated 6 times:

```text
query_pos = MLP(reference_points)
self-attn over concat([queries + query_pos, task_features])
split updated task_features and query embeddings
deformable cross-attn(query + query_pos, flattened multi-level vision features, reference_points)
FFN 256->2048->256
box = sigmoid(bbox_mlp(query) + inverse_sigmoid(reference_points))
class = cosine/dot(query, projected_class_features) on final inference layer
```

## 6. Attention requirements

Dense attention:

- Noncausal self-attention only; no autoregressive KV cache.
- Encoder selected feature level uses sequence length `H*W`.
- Decoder self-attention fuses `num_queries + task_token_length`; attention mask is bidirectional from `create_bidirectional_mask` with task padding masked.
- MHA, not GQA/MQA: query/key/value all project to 8 heads x 32 dim.

Deformable cross-attention:

- Query source: learned or top-k selected object query embeddings, shape `[B, 900, 256]`.
- Key/value source: flattened multi-level FPN/PAN vision features `[B, sum(H_l*W_l), 256]`.
- Reference points: source passes `[B, num_queries, 1, 4]` for box-form points after sigmoid.
- Sampling offsets shape: `[B, Q, 8, 3, 4, 2]`.
- Attention weights shape: `[B, Q, 8, 3, 4]`, softmax over 12 samples per head.
- Eager fallback: split value by levels, reshape to `[B*heads, head_dim, H, W]`, convert locations to `2*loc-1`, call `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)`, weighted sum over levels/points.
- No packed/varlen attention or decode cache. The cacheable state is text embeddings, not KV cache.

## 7. Position encoding and custom math

2D sin/cos embedding for the encoder-selected feature map:

```python
grid_w, grid_h = meshgrid(arange(width), arange(height), indexing="ij")
omega = 1.0 / (temperature ** (arange(embed_dim // 4) / (embed_dim // 4)))
pos = concat([(grid_w @ omega).sin(), (grid_w @ omega).cos(),
              (grid_h @ omega).sin(), (grid_h @ omega).cos()], dim=1)
```

Anchor and box refinement:

```python
xy = (grid_xy + 0.5) / [width, height]
wh = grid_size * (2.0 ** level)
anchors = log([xy, wh] / (1 - [xy, wh]))
valid = all((anchor_sigmoid > 1e-2) & (anchor_sigmoid < 1 - 1e-2))
box = sigmoid(delta + inverse_sigmoid(reference_points))
```

Class similarity:

```python
if cosine:
    logits = (1 / 0.07) * bmm(l2norm(query, dim=2), l2norm(class_proj, dim=1))
else:
    logits = bmm(query, class_proj)
```

Precompute candidates: static sin/cos and anchors can be precomputed for fixed image size and level shapes. Dynamic target sizes affect only postprocess scaling, not model logits.

## 8. Preprocessing and input packing

Processor input:

- `images`: passed to `DetrImageProcessor`.
- `text`: either comma-separated string or list/list-of-lists of class names.
- optional `task`: defaults to `Detect {class1, class2, ...}.` per image.

Processor output tensors:

- `pixel_values`: DetrImageProcessor output, representative `[B,3,640,640]`, normalized with mean `[123.675,116.28,103.53]`, std `[58.395,57.12,57.375]`, `do_rescale=false`.
- `classes_input_ids`, `classes_attention_mask`: flattened class prompts, shape `[sum(classes_structure), token_length]`.
- `tasks_input_ids`, `tasks_attention_mask`: per-image task prompts, shape `[B, token_length]`.
- `classes_structure`: int64 `[B]` counts for regrouping class embeddings and filtering padded class logits.

GPU/runtime work includes CLIP text model, prompt/class MLPs, vision/neck/decoder. CPU/data pipeline can own image resizing and tokenization. A first DinoML integration can accept pre-tokenized tensors and optionally precomputed class/task embeddings to avoid owning CLIP initially.

Postprocess ABI:

- Raw `decoder_coord_logits`: `[B, 900, 4]`, normalized center boxes.
- Raw `decoder_class_logits`: `[B, 900, max_classes_in_batch]`.
- `target_sizes`: list/tensor `[B,2]` in `(height,width)`.
- Output is per-image variable-length records: `boxes [N,4]` in xyxy pixels, `scores [N]`, `labels [N]`, optional `text_labels`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed 1x1 Conv2d + BatchNorm -> per-pixel Linear

Source pattern: channel projections and decoder feature projections use `Conv2d(Cin, 256, kernel=1, bias=False) + BatchNorm2d`.

Replacement:

```text
NCHW -> flatten spatial -> GEMM(Cin -> 256) -> affine BN -> reshape
```

Preconditions: inference mode, frozen BN statistics, contiguous dense NCHW or a fully controlled NHWC region, stride/padding/dilation all trivial. Weight transform folds BN scale/bias into conv weights. Failure cases: training mode, unfrozen BN, arbitrary non-contiguous feature layout.

### Rewrite: fixed-size anchors and encoder positions as constants

Source pattern: anchors and 2D sin/cos embeddings generated from feature map shapes for fixed 640 image.

Replacement: materialize per-level anchor logits, valid masks, and selected-level position table as constants.

Preconditions: fixed processor image size/backbone level shapes and no dynamic backbone resolution. Failure cases: variable input resolution or altered `out_indices`.

### Rewrite: CLIP text branch as cacheable side graph

Source pattern: `get_cached_class_embeddings` and `get_cached_task_embeddings` cache by token ids/masks.

Replacement: expose class/task embeddings as optional runtime inputs, or compile CLIP text encoder separately.

Preconditions: class/task strings stable across many images or request batches. Failure cases: per-request fully dynamic class sets without caching benefit.

### Rewrite: deformable attention provider

Source pattern: eager `grid_sample` fallback over 3 levels x 4 points x 8 heads.

Replacement: dedicated multi-scale deformable attention kernel/provider with same coordinate normalization and `align_corners=False`.

Preconditions: levels, heads, points fixed or profiled; value layout and spatial shape metadata valid. Failure cases: source custom kernel unavailable and no grid-sample parity fallback.

### Layout opportunity: local NHWC/channel-last neck

Source pattern: NCHW conv/BN/activation blocks with frequent flatten-to-sequence boundaries.

Candidate: keep convolutional neck in NHWC/channel-last only inside a guarded region.

Required axis rewrites: concat channel `dim=1 -> dim=-1`, feature shape reads `[2:] -> [1:3]`, flatten spatial order must preserve row-major H,W token order, BatchNorm channel axis changes, `grid_sample` fallback expects NCHW and should be a no-layout-translation boundary unless rewritten entirely.

## 10. Kernel fusion candidates

Highest priority:

- Multi-scale deformable attention kernel: central decoder cost and includes `grid_sample`-like sampling not covered by dense attention.
- Conv1x1/3x3 + BatchNorm + activation for FPN/PAN and projections: repeated vision-neck hot path.
- Dense MHA + FFN blocks at width 256: useful existing GEMM/softmax/layernorm coverage can cover encoder and decoder self-attention.

Medium priority:

- Top-k proposal selection and gather: 900 queries from all anchors, shape-sensitive but isolated.
- Cosine class similarity: L2 norm + batched GEMM + fixed scale over `[B,Q,512] x [B,512,C]`; important when many classes are queried.
- Box refinement: bbox MLP + inverse sigmoid + sigmoid, can fuse elementwise around small MLP outputs.

Lower priority:

- Processor postprocess NMS on GPU: useful for throughput, but can be CPU/torchvision-owned initially.
- CLIP text encoder fusion: likely reusable from separate CLIP audit; caching may make it less urgent for image throughput.

## 11. Runtime staging plan

1. Parse OmDet-Turbo config and reject unallowlisted backbones; load official Swin-tiny HF checkpoint only.
2. First parity target with precomputed backbone feature maps and precomputed class/task embeddings: run OmDet-owned hybrid encoder, proposal selection, decoder, raw logits/boxes.
3. Add processor ABI and postprocess parity using PyTorch/torchvision NMS as a reference helper.
4. Compose the audited Swin/timm backbone path and CLIP text encoder path, still allowing cached text embeddings.
5. Replace eager deformable attention with a DinoML provider while retaining source-compatible fallback.
6. Add guarded layout/fusion passes for neck convs and fixed-size anchor/position constants.
7. Add throughput scheduling: cache class/task embeddings across requests and batch images with compatible class padding.

Initially stub or externalize: text tokenization, image resize/normalize, CLIP text encoder, torchvision NMS, and unallowlisted `AutoBackbone` variants.

## 12. Parity and validation plan

- Unit parity for custom math: 2D sin/cos positions, anchor generation, inverse sigmoid, cosine/dot class similarity, center-to-corners scaling/clipping.
- Deformable attention random tensor parity against HF eager fallback for fixed `(B,Q,heads,levels,points,head_dim)` with fp32 tolerance around `1e-5`.
- Single decoder-layer parity with synthetic multi-level features and class/task embeddings.
- OmDet-owned neck+decoder parity using saved backbone/text features from the HF model.
- End-to-end official checkpoint parity on small image/class sets: compare raw `decoder_coord_logits`, `decoder_class_logits`, then postprocessed detections with same `threshold`, `nms_threshold`, `target_sizes`.
- Suggested tolerances: fp32 raw logits/boxes `rtol=1e-4, atol=1e-5`; fp16 after optimized kernels `rtol=5e-3, atol=5e-3`, with postprocess comparisons using score/box tolerances and label set equality after NMS.

## 13. Performance probes

- Processor throughput: resize/normalize plus CLIP tokenization by number of class labels.
- Text branch cache hit/miss latency: classes only, task only, full language embeddings.
- Vision backbone-only throughput at 640 x 640.
- OmDet neck-only throughput by feature map shapes.
- Decoder-only throughput by `num_queries`, class count, and level shapes.
- Deformable attention backend comparison: eager grid-sample fallback versus custom provider.
- Postprocess cost by class count, `max_num_det`, threshold, and NMS threshold.
- End-to-end images/sec for fixed class set versus changing class set.
- Memory probes for flattened multi-level vision features, deformable attention temporaries, and class padding to max batch class count.

## 14. Skip/defer list

- Training and losses: source raises `NotImplementedError` when `labels` are passed.
- Denoising queries: source has inference placeholders only.
- Unreviewed `AutoBackbone` or remote/timm variants beyond the official Swin-tiny config.
- GPU-native tokenizer/image processor for first integration.
- GPU NMS and variable-length output allocation beyond a helper ABI.
- Custom kernel download path from Hub as an execution dependency; DinoML should own or explicitly fallback for deformable attention.
- `learn_initial_query=true` and `class_distance_type="dot"` variants can be admitted later after source-default parity.

## 15. Final implementation checklist

- [ ] Parse `OmDetTurboConfig` and representative processor/tokenizer configs.
- [ ] Add admission allowlist for `omlab/omdet-turbo-swin-tiny-hf` Swin-tiny/timm backbone contract.
- [ ] Load OmDet-owned weights and preserve text projection/class head shapes.
- [ ] Define runtime ABI for `pixel_values`, class/task token tensors, `classes_structure`, and optional precomputed text embeddings.
- [ ] Implement/cache CLIP class/task embedding stage or route to separate CLIP audit.
- [ ] Implement OmDet hybrid encoder neck ops: Conv2d, BatchNorm2d, LayerNorm, activations, nearest upsample, concat.
- [ ] Implement dense MHA/FFN blocks for encoder and decoder self-attention.
- [ ] Implement anchor generation, top-k proposal selection, gather, inverse-sigmoid box refinement.
- [ ] Implement multi-scale deformable attention provider with eager parity fallback.
- [ ] Implement class similarity logits and dynamic class padding/filtering.
- [ ] Implement postprocess helper: sigmoid, score threshold, top-k, xywh-to-xyxy, scale, class-filter, batched NMS, clip.
- [ ] Add fixed-size anchor/position constant rewrite.
- [ ] Add guarded 1x1 Conv+BN-to-GEMM rewrite.
- [ ] Add single-layer, neck+decoder, and official end-to-end parity tests.
- [ ] Benchmark processor, text cache, backbone, neck, decoder, deformable attention, and postprocess separately.
