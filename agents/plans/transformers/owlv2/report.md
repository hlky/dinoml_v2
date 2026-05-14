# OWLv2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/owlv2-base-patch16, google/owlv2-base-patch16-ensemble, google/owlv2-base-patch16-finetuned, google/owlv2-large-patch14, google/owlv2-large-patch14-ensemble, google/owlv2-large-patch14-finetuned
Config source: official Hugging Face config/preprocessor/tokenizer JSON files downloaded from resolve/main into this folder.
Primary runtime target: Owlv2ForObjectDetection text-conditioned zero-shot object detection.
Source files inspected:
- X:/H/transformers/src/transformers/models/owlv2/modeling_owlv2.py
- X:/H/transformers/src/transformers/models/owlv2/configuration_owlv2.py
- X:/H/transformers/src/transformers/models/owlv2/processing_owlv2.py
- X:/H/transformers/src/transformers/models/owlv2/image_processing_owlv2.py
- X:/H/transformers/src/transformers/models/owlv2/image_processing_pil_owlv2.py
- X:/H/transformers/src/transformers/models/owlv2/modular_owlv2.py
- X:/H/transformers/src/transformers/activations.py
Any missing files or assumptions: no gated official configs were encountered. Modeling is copied/generated-style from OWL-ViT; `modeling_owlv2.py` is the import-time inference authority, while `modular_owlv2.py` is useful for future source edits to image processing.
```

Source URLs at the pinned commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlv2/modeling_owlv2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlv2/configuration_owlv2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlv2/processing_owlv2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlv2/image_processing_owlv2.py

Local config snapshots:

- `google__owlv2-base-patch16__config.json`, `google__owlv2-base-patch16__preprocessor_config.json`, `google__owlv2-base-patch16__tokenizer_config.json`
- `google__owlv2-base-patch16-ensemble__config.json`, `google__owlv2-base-patch16-ensemble__preprocessor_config.json`, `google__owlv2-base-patch16-ensemble__tokenizer_config.json`
- `google__owlv2-base-patch16-finetuned__config.json`, `google__owlv2-base-patch16-finetuned__preprocessor_config.json`, `google__owlv2-base-patch16-finetuned__tokenizer_config.json`
- `google__owlv2-large-patch14__config.json`, `google__owlv2-large-patch14__preprocessor_config.json`, `google__owlv2-large-patch14__tokenizer_config.json`
- `google__owlv2-large-patch14-ensemble__config.json`, `google__owlv2-large-patch14-ensemble__preprocessor_config.json`, `google__owlv2-large-patch14-ensemble__tokenizer_config.json`
- `google__owlv2-large-patch14-finetuned__config.json`, `google__owlv2-large-patch14-finetuned__preprocessor_config.json`, `google__owlv2-large-patch14-finetuned__tokenizer_config.json`

## 2. High-level architecture

OWLv2 is a CLIP-like dual encoder adapted for open-vocabulary detection. The text branch is a short causal self-attention encoder over CLIP tokenizer prompts. The image branch is a ViT encoder over square-padded/resized images. Detection removes ordinary CLS-only vision use: patch tokens are combined with the broadcast CLS token, normalized, then fed to per-patch class, box, and objectness heads.

```text
CPU image/text preprocessing
  -> pixel_values [B,3,H,W] and flattened prompt input_ids [B*Q,T]
  -> ViT image encoder + CLIP text encoder
  -> per-patch image features [B,Nh,Nw,V]
  -> text/image query embeddings [B,Q,Tdim]
  -> class logits [B,Nh*Nw,Q], boxes [B,Nh*Nw,4], objectness [B,Nh*Nw]
  -> CPU/GPU postprocess: sigmoid, threshold, cxcywh->xyxy, square-scale boxes, optional NMS for image-guided path
```

Stage decomposition:

- CPU/data pipeline: CLIP tokenization; image RGB conversion, rescale, CLIP normalization, square padding, antialiased resize.
- Independently cacheable branches: text query embeddings can be cached for repeated image batches; image encoder outputs can be cached for repeated query sets. Image-guided detection has a separate query-image encoder pass.
- Runtime graph target: encoder-only inference plus detection heads. There is no autoregressive decode or KV cache.
- Postprocessing: required for end-to-end detection parity, but separable from neural graph parity.

Implemented heads:

- `Owlv2ForObjectDetection.forward`: required for first DinoML target.
- `Owlv2Model`: optional contrastive image/text similarity target.
- `Owlv2TextModel` and `Owlv2VisionModel`: useful independently for branch parity.
- `Owlv2ForObjectDetection.image_guided_detection`: optional/deferred initially because it includes dynamic per-image box selection, IoU/GIoU logic, and an NMS-like postprocess.

## 3. Important config dimensions

Effective defaults come from `configuration_owlv2.py` when checkpoint configs omit subfields.

| Field | Base effective value | Large effective value | Source |
| --- | ---: | ---: | --- |
| `projection_dim` | 512 | 768 | config.json |
| text `vocab_size` | 49408 | 49408 | source default, omitted in configs |
| text `hidden_size` | 512 | 768 | source default for base, config for large |
| text `intermediate_size` | 2048 | 3072 | source default for base, config for large |
| text `num_hidden_layers` | 12 | 12 | source default |
| text `num_attention_heads` | 8 | 12 | source default for base, config for large |
| text `head_dim` | 64 | 64 | inferred from source `hidden_size // heads` |
| text `max_position_embeddings` | 16 | 16 | source default |
| text activation | `quick_gelu` | `quick_gelu` | source default |
| vision `hidden_size` | 768 | 1024 | source default for base, config for large |
| vision `intermediate_size` | 3072 | 4096 | source default for base, config for large |
| vision `num_hidden_layers` | 12 | 24 | source default for base, config for large |
| vision `num_attention_heads` | 12 | 16 | source default for base, config for large |
| vision `head_dim` | 64 | 64 | inferred |
| vision `image_size` | 960 | 1008 | config.json |
| vision `patch_size` | 16 | 14 | config.json |
| vision patches | 60 x 60 = 3600 | 72 x 72 = 5184 | inferred |
| `torch_dtype` | float32 | float32 | config.json |
| attention backend flags | SDPA/Flash/Flex supported by HF class | same | source class flags |

Representative checkpoint sweep:

| Checkpoint | Variant role | Text dim/layers/heads | Vision dim/layers/heads | Image/patch | Patches | Operator structure change |
| --- | --- | --- | --- | --- | ---: | --- |
| `google/owlv2-base-patch16` | base | 512/12/8 | 768/12/12 | 960/16 | 3600 | baseline |
| `google/owlv2-base-patch16-ensemble` | common example | 512/12/8 | 768/12/12 | 960/16 | 3600 | no config-visible structural change |
| `google/owlv2-base-patch16-finetuned` | finetuned | 512/12/8 | 768/12/12 | 960/16 | 3600 | no config-visible structural change |
| `google/owlv2-large-patch14` | large | 768/12/12 | 1024/24/16 | 1008/14 | 5184 | larger GEMMs and longer vision sequence |
| `google/owlv2-large-patch14-ensemble` | large ensemble | 768/12/12 | 1024/24/16 | 1008/14 | 5184 | no config-visible structural change |
| `google/owlv2-large-patch14-finetuned` | large finetuned | 768/12/12 | 1024/24/16 | 1008/14 | 5184 | no config-visible structural change |

Preprocessor sweep:

| Checkpoint group | Size | Pad | Rescale/normalize | Mean/std | Layout emitted |
| --- | --- | --- | --- | --- | --- |
| base variants | 960 x 960 | square zero pad before resize | yes, rescale `1/255`, CLIP normalize | OpenAI CLIP mean/std | `pixel_values` NCHW |
| large variants | 1008 x 1008 | square zero pad before resize | yes, rescale `1/255`, CLIP normalize | OpenAI CLIP mean/std | `pixel_values` NCHW |

Tokenizer snapshot: `CLIPTokenizer`, `model_max_length=16`, `pad_token="!"` id 0, BOS id 49406, EOS id 49407, lowercase enabled.

## 3a. Family variation traps

- Base configs omit most text/vision fields; DinoML must apply `Owlv2TextConfig` and `Owlv2VisionConfig` defaults before lowering.
- Large variants change both projection width and vision sequence length; the vision branch has 5185 tokens including CLS, not just wider layers.
- Text pooling is not "last non-pad token"; source uses `argmax(input_ids)` because EOS has the highest token id. This depends on CLIP token ids.
- Text prompts are flattened from `[B,Q,T]` to `[B*Q,T]`, then reshaped back using `max_text_queries = input_ids.shape[0] // batch_size`.
- Padded text query detection is `input_ids[..., 0] > 0`. The processor pads missing queries with `" "`, and tokenizer padding uses id 0.
- The text encoder passes a causal mask. It is encoder-style for fixed prompts, but attention math is causal self-attention.
- Vision source is NCHW through `Conv2d`; candidate NHWC/channel-last optimizations need guarded local layout rewrites.
- `interpolate_pos_encoding=True` changes positional embedding and box-bias grid size dynamically. Static checkpoint parity can initially reject it.
- Box scaling uses `max(height,width)` for all coordinates because preprocessing square-pads images. It does not independently scale x by width and y by height.
- Text-conditioned postprocess has no NMS in `post_process_object_detection`; image-guided postprocess does have a greedy IoU suppression loop.
- `objectness_predictor` detaches image features in source. In inference this is numerically irrelevant but should not be mistaken for an extra op.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `Conv2d` patch embedding with `kernel_size=stride=patch_size`, no bias.
- `flatten(2)`, `transpose(1,2)`, `reshape/view`, `permute`, `contiguous`, `cat`, `broadcast_to`, indexing/gather by token position, boolean mask construction.
- Static/dynamic shape arithmetic for patch counts when `interpolate_pos_encoding` is admitted.

Neural network primitives:

- Embedding lookup for text token and position ids.
- Learned CLS/class parameter expansion.
- Linear projections with bias for Q/K/V/out, MLPs, detection heads, logit shift/scale.
- Biasless linear projections for `visual_projection` and `text_projection`.
- LayerNorm with eps `1e-5`.
- `quick_gelu(x) = x * sigmoid(1.702*x)` for transformer MLPs.
- Standard GELU for box/objectness heads.
- ELU + 1 for class-head logit scale.
- Sigmoid for boxes and postprocess scores.
- L2 norm/divide for contrastive/class embeddings.
- Matmul/einsum for image-text similarity and patch-query logits.

Attention primitives:

- Dense self-attention only.
- Text: causal MHA over `T<=16`.
- Vision: noncausal MHA over 3601 or 5185 tokens.
- No GQA/MQA, cross-attention, KV cache, local/sliding attention, RoPE, ALiBi, or relative bias.
- HF class advertises SDPA/Flash/Flex support; eager equivalent is `matmul -> mask add -> softmax -> dropout(training only) -> matmul`.

Position/custom math:

- Text learned absolute position embeddings length 16.
- Vision learned absolute position embeddings over square patch grid plus CLS.
- Optional bicubic interpolation of vision patch position table.
- Box bias uses logit-like transform of patch center and patch size.

Preprocessing-coupled ops:

- RGB conversion, rescale, CLIP normalization, square pad, antialiased bilinear resize.
- Processor packs nested text queries and query images; query images override text prompts.

Postprocessing/detection ops:

- `center_to_corners_format`, sigmoid score conversion, max over query/classes, threshold filter.
- Square-scale boxes by `max(target_h,target_w)`.
- Image-guided path: pairwise IoU/GIoU, argsort, greedy suppression, score alpha scaling.

Not required:

- Autoregressive generation ops, KV cache, tokenizer-controlled generation, MoE, quantized/packed weights, distributed/tensor-parallel ops.

## 5. Layer/block breakdown

Text branch, repeated 12 layers:

```text
input_ids [B*Q,T] -> token_embedding + position_embedding
causal attention mask [B*Q,1,T,T]
for each layer:
  residual = x
  x = LayerNorm(x)
  q,k,v = Linear(text_hidden -> text_hidden), bias=True
  q,k,v -> [B*Q, heads, T, 64]
  x = MHA(q,k,v, causal mask)
  x = residual + Linear(text_hidden -> text_hidden, bias=True)
  residual = x
  x = LayerNorm(x)
  x = Linear(text_hidden -> intermediate, bias=True)
  x = quick_gelu(x)
  x = Linear(intermediate -> text_hidden, bias=True)
  x = residual + x
x = final LayerNorm(x)
pooled = x[batch_index, argmax(input_ids)]
text_embeds = Linear(text_hidden -> projection_dim, bias=False)(pooled)
```

Vision branch:

```text
pixel_values [B,3,H,W] NCHW
patches = Conv2d(3 -> vision_hidden, kernel=patch, stride=patch, bias=False)
patches [B,V,Nh,Nw] -> flatten/transpose [B,Nh*Nw,V]
x = concat(CLS, patches) + learned/interpolated absolute positions
x = pre LayerNorm(x)
repeat vision_layers:
  same pre-norm MHA + MLP block as text, noncausal mask None
pooled_cls = post LayerNorm(x[:,0,:])
base model image_embeds = Linear(vision_hidden -> projection_dim, bias=False)(pooled_cls)
```

Detection feature construction:

```text
vision_last_hidden [B,1+P,V]
image_embeds = vision_post_layernorm(vision_last_hidden)
class_token_out = broadcast(image_embeds[:, :1, :], image_embeds[:, :-1, :].shape)
patch_features = LayerNorm(image_embeds[:, 1:, :] * class_token_out)
feature_map = reshape [B,Nh,Nw,V]
image_feats = reshape [B,P,V]
```

Class head:

```text
image_class_embeds = Linear(V -> text_hidden)(image_feats)
image_class_embeds = l2_normalize(image_class_embeds, eps=1e-6)
query_embeds = l2_normalize(text_embeds.reshape[B,Q,text_hidden], eps=1e-6)
pred_logits = einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
pred_logits = (pred_logits + Linear(V -> 1)(image_feats)) * (ELU(Linear(V -> 1)(image_feats)) + 1)
pred_logits = where(query_mask == 0, finfo_min, pred_logits).to(float32)
```

Box/objectness heads:

```text
BoxPredictionHead(out_dim):
  Linear(V -> V) -> GELU -> Linear(V -> V) -> GELU -> Linear(V -> out_dim)

pred_boxes = sigmoid(box_head(image_feats) + box_bias[P,4])
objectness_logits = objectness_head(detached_image_feats)[...,0]
```

## 6. Attention requirements

Text attention:

- Causal self-attention, MHA.
- Base: 8 heads, head dim 64, Q/K/V width 512.
- Large: 12 heads, head dim 64, Q/K/V width 768.
- Query length and key/value length both `T`, normally 16.
- Mask is produced by `create_causal_mask` using tokenizer attention mask and no past key values.
- No cache: all prompt tokens are recomputed as one short sequence.

Vision attention:

- Noncausal self-attention, MHA.
- Base: 12 heads, head dim 64, Q/K/V width 768, sequence length 3601 including CLS.
- Large: 16 heads, head dim 64, Q/K/V width 1024, sequence length 5185 including CLS.
- Mask is normally `None`; no padding mask enters vision after processor emits fixed square tensors.

FlashAttention/SDPA compatibility:

- Source routes through `ALL_ATTENTION_FUNCTIONS` with eager fallback. Dropout is zero in inference.
- Dense attention is enough semantically. For performance, vision attention length is the dominant attention bottleneck; text attention is tiny.
- Cached keys/values are not required. Cached text embeddings or image features are branch-level output caches, not KV caches.

## 7. Position encoding and custom math

Text positions are learned absolute embeddings indexed from `0..T-1`.

Vision positions are learned absolute embeddings over `[CLS] + square_grid`. Optional interpolation:

```python
def interpolate_vision_pos(pos_table, embeddings, height, width, patch):
    cls = pos_table[:, :1]
    patch_pos = pos_table[:, 1:]
    old = int((patch_pos.shape[1]) ** 0.5)
    new_h, new_w = height // patch, width // patch
    patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patch_pos = bicubic_interpolate(patch_pos, size=(new_h, new_w), align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    return concat([cls, patch_pos], dim=1)
```

Box bias:

```python
def box_bias(num_h, num_w):
    x = arange(1, num_w + 1) / num_w
    y = arange(1, num_h + 1) / num_h
    centers = meshgrid_xy(x, y).reshape(num_h * num_w, 2).clip(0, 1)
    center_bias = log(centers + 1e-4) - log1p(-centers + 1e-4)
    size = full_like(centers, 1.0)
    size[:, 0] /= num_w
    size[:, 1] /= num_h
    size_bias = log(size + 1e-4) - log1p(-size + 1e-4)
    return concat([center_bias, size_bias], dim=-1)
```

QuickGELU:

```python
def quick_gelu(x):
    return x * sigmoid(1.702 * x)
```

Precomputable: static text position ids, static vision position ids, static box bias for configured image size, and static square-grid coordinate tensors. Dynamic: interpolated position embeddings and box bias when `interpolate_pos_encoding=True`.

## 8. Preprocessing and input packing

Image processor:

- Accepts images or query images.
- Converts to RGB by default, emits `pixel_values`.
- Tensor layout for model input is NCHW `[B,3,H,W]`.
- Rescale by `1/255`, then later normalize by OpenAI CLIP mean `[0.48145466,0.4578275,0.40821073]` and std `[0.26862954,0.26130258,0.27577711]`.
- Pads each image to a square by adding zeros to bottom and right, then resizes to checkpoint size. Resize may apply Gaussian blur before downsampling and calls backend resize with antialias disabled afterward.
- Base emits `[B,3,960,960]`; large emits `[B,3,1008,1008]`.

Text processor:

- Uses `CLIPTokenizer`, lowercase enabled, max length 16, padding to max length by default.
- Flat string/list becomes one encoding batch. Nested list `[B][Q]` is padded to the maximum number of queries using `" "`, tokenized per image, then concatenated to `[B*Q,T]`.
- `attention_mask` is concatenated with the same leading dimension.
- `query_images` override text prompts and produce only `query_pixel_values` plus target `pixel_values` if images are supplied.

Detection postprocessing:

- Text-conditioned path: `scores = sigmoid(max(logits, dim=-1).values)`, `labels = argmax(logits, dim=-1)`, threshold filter, boxes from cxcywh to xyxy, optional square-scale by max target side. No NMS.
- Image-guided path: max over logits, sigmoid, cxcywh to xyxy, greedy IoU suppression if `nms_threshold < 1`, optional square scaling, threshold and alpha normalization. This is a separate admission target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> GEMM

Source pattern:

```text
Conv2d(C -> V, kernel=patch, stride=patch, padding=0, dilation=1, groups=1, bias=False)
-> flatten spatial -> transpose to [B,P,V]
```

Replacement:

```text
WindowFlatten_NCHW([B,C,H,W], patch, patch) -> GEMM([B*P, C*patch*patch] x [C*patch*patch,V]) -> reshape [B,P,V]
```

Preconditions:

- `H` and `W` divisible by `patch_size`.
- `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`, no bias.
- NCHW flatten order must match PyTorch Conv2d weight layout `[V,C,Kh,Kw]`.
- For NHWC optimization, either control the whole image-preprocess-to-patch region or insert explicit layout guards and weight transforms.

Failure cases: dynamic non-divisible input when interpolation is admitted, grouped convolution, nonzero padding/dilation, or changed processor layout.

Parity sketch: compare patch embedding outputs on random NCHW tensors for base and large shapes in fp32/fp16 tolerances.

### Rewrite: QKV projection packing

Source pattern: separate `q_proj`, `k_proj`, `v_proj`, all `Linear(D -> D)` with bias.

Replacement: one packed GEMM producing `[Q,K,V]` then split in q/k/v order.

Preconditions:

- Same input tensor, same output width, all three bias settings present.
- Preserve split order exactly as source: q, k, v.
- Weight layout transform from PyTorch linear weights `[out,in]` to provider GEMM layout.

Failure cases: future configs with differing projection widths or missing biases.

### Rewrite: normalized similarity

Source pattern:

```text
x / (norm(x, dim=-1, keepdim=True) + eps) ; einsum("...pd,...qd->...pq")
```

Replacement: fused row L2 normalize plus batched GEMM.

Preconditions: dense row-major last dimension, fixed eps (`1e-6` for detection head, none added in base contrastive model), no zero-vector special behavior beyond source arithmetic.

### Rewrite: detection class head to batched GEMM

Source pattern: `einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)`.

Replacement: per-batch GEMM `[P,D] x [D,Q] -> [P,Q]`.

Preconditions: `query_embeds` already reshaped to `[B,Q,D]`, query count fixed for batch. Apply query mask after shift/scale as source does.

### Layout rewrite: controlled vision encoder channel-last

Source region is NCHW patch embedding followed by sequence tensors. A channel-last optimization is local only around preprocessing/patch projection and perhaps elementwise normalization. Axis-sensitive source operations include `flatten(2)`, `transpose(1,2)`, image processor padding/resizing axes, and box-feature reshape `[B,Nh,Nw,V]`. Any NHWC pass must rewrite those axes or guard the region as source-layout.

## 10. Kernel fusion candidates

Highest priority:

- Vision dense attention for 3601/5185 tokens. This dominates runtime; fused SDPA/Flash-style attention is the most important neural kernel.
- Linear + QuickGELU + Linear MLP, especially vision MLP widths 768->3072 and 1024->4096.
- LayerNorm + QKV projection packing for both branches.
- Detection class-head normalize + batched patch-query GEMM, because output shape `[B,P,Q]` can be large when users provide many prompts.

Medium priority:

- Conv patch embedding to GEMM or direct optimized patch projection.
- Patch feature construction: post LayerNorm, CLS broadcast multiply, final LayerNorm.
- Box/objectness MLP heads across all patches.
- Box bias add + sigmoid.

Lower priority:

- Text attention kernels; sequence length 16 makes it small.
- Contrastive `Owlv2Model` global logits; useful but not central to detection.
- Image-guided query selection and NMS; likely CPU/postprocess first.

## 11. Runtime staging plan

Stage 1: config/processor ingestion.

- Parse nested `text_config` and `vision_config` with defaults.
- Load tokenizer/preprocessor metadata enough to validate input tensor contracts.
- Reject `interpolate_pos_encoding=True` initially.

Stage 2: branch parity.

- Implement text encoder parity for `[B*Q,16]`.
- Implement vision encoder parity for static base and large image sizes.
- Validate pooled/projection outputs independently.

Stage 3: detection head parity.

- Implement image feature construction, class head, objectness head, box head, static box bias.
- Return raw `logits`, `pred_boxes`, `objectness_logits`, `class_embeds`, `image_embeds`, `text_embeds`.

Stage 4: end-to-end text-conditioned object detection.

- Reproduce processor packing assumptions.
- Add postprocess parity for thresholding and square-scaled boxes. Keep no-NMS behavior for text-conditioned path.

Stage 5: performance lowering.

- Add patch Conv2d->GEMM, QKV packing, fused attention, MLP fusion, normalize+GEMM.
- Add cached text embeddings for repeated prompt sets.

Stage 6: optional surfaces.

- Admit `interpolate_pos_encoding=True` with dynamic position and box-bias generation.
- Add `image_guided_detection` including query image branch, IoU/GIoU query selection, and NMS-like postprocess.

## 12. Parity and validation plan

- Unit tests for `quick_gelu`, box bias, `center_to_corners_format`, `_scale_boxes`, and class-head logit scale/shift.
- Patch embedding rewrite parity for base `[1,3,960,960]` and large `[1,3,1008,1008]`, plus smaller synthetic divisible shapes if interpolation is tested.
- Single encoder layer parity for text and vision with random weights.
- Full text branch parity on random token ids with CLIP EOS as highest id; include padded query rows with first token id 0.
- Full vision branch parity on random pixel tensors, static position path first.
- Detection head parity from saved/random hidden states to isolate head math.
- End-to-end raw output parity versus Transformers for one image and nested prompts.
- Postprocess parity for target sizes where height != width to verify max-side scaling.
- Recommended tolerances: fp32 `atol=1e-5, rtol=1e-4` for branch/head unit tests; fp16/bf16 `atol=2e-2, rtol=2e-2` for fused attention/GEMM paths, with stricter fp32 accumulation checks where available.

## 13. Performance probes

- Image preprocessing throughput: RGB/rescale/normalize/pad/resize separately from GPU model.
- Vision encoder latency and throughput for base and large at batch sizes 1, 2, 4.
- Attention backend comparison for sequence lengths 3601 and 5185.
- MLP GEMM throughput for vision and text dimensions.
- Prompt count sweep for class head: `Q=1,4,16,64`.
- Text embedding cache hit probe: repeated prompts over many images.
- End-to-end detection throughput split into preprocessing, encoder, detection head, and postprocess.
- Memory probe for vision attention activations and temporary buffers at base/large resolutions.
- Postprocess probe for threshold density and image-guided NMS cost.

## 14. Skip/defer list

- Training losses and Hungarian matching-style training helpers; first target is inference.
- Gradient checkpointing and dropout behavior.
- `return_loss` contrastive training path.
- `interpolate_pos_encoding=True` dynamic-resolution inference, until static base/large parity is stable.
- Image-guided detection, IoU/GIoU query selection, and NMS-like postprocess.
- NHWC/channel-last global layout translation beyond guarded patch/vision regions.
- Quantization, packed weights, tensor parallelism, multi-GPU scheduling.
- Beam search, decoding, and KV cache: not applicable to OWLv2.

## 15. Final implementation checklist

- [ ] Parse `Owlv2Config` with nested text/vision defaults.
- [ ] Load base and large checkpoint weights with PyTorch linear/conv layouts preserved.
- [ ] Implement CLIP tokenizer ABI or accept pretokenized `[B*Q,16]` tensors for first graph target.
- [ ] Implement image processor ABI or accept preprocessed NCHW `pixel_values` for first graph target.
- [ ] Implement text embeddings, causal MHA, LayerNorm, QuickGELU MLP, EOS-argmax pooling.
- [ ] Implement vision patch embedding, learned CLS/position embeddings, noncausal MHA, LayerNorm, QuickGELU MLP.
- [ ] Implement detection feature construction with CLS broadcast multiply.
- [ ] Implement class prediction head including L2 normalization, batched patch-query GEMM, shift/ELU-scale, query mask.
- [ ] Implement box/objectness heads and static box bias.
- [ ] Implement text-conditioned postprocess: sigmoid scores, threshold, cxcywh->xyxy, max-side scaling, no NMS.
- [ ] Add parity tests for base and large raw outputs.
- [ ] Add postprocess parity tests for non-square target sizes.
- [ ] Add guarded patch Conv2d->GEMM rewrite.
- [ ] Add QKV packing rewrite for self-attention.
- [ ] Benchmark vision attention, MLP, class-head prompt sweep, and end-to-end throughput.
