# OWL-ViT Transformers Family Audit

Primary target: open-vocabulary object detection with `OwlViTForObjectDetection`, including text-query detection and the optional image-guided detection path. Base `OwlViTModel` contrastive image-text scoring is useful as a staged CLIP-like subtarget, but detection parity requires patch-level class embeddings, box heads, and postprocessing.

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary: google/owlvit-base-patch32.
  Sweep: hf-internal-testing/tiny-random-owlvit, google/owlvit-base-patch16,
  google/owlvit-large-patch14.

Config source:
  https://huggingface.co/hf-internal-testing/tiny-random-owlvit/resolve/main/config.json
  https://huggingface.co/google/owlvit-base-patch32/resolve/main/config.json
  https://huggingface.co/google/owlvit-base-patch32/resolve/main/preprocessor_config.json
  https://huggingface.co/google/owlvit-base-patch16/resolve/main/config.json
  https://huggingface.co/google/owlvit-base-patch16/resolve/main/preprocessor_config.json
  https://huggingface.co/google/owlvit-large-patch14/resolve/main/config.json
  https://huggingface.co/google/owlvit-large-patch14/resolve/main/preprocessor_config.json

Source files inspected:
  transformers/src/transformers/models/owlvit/modeling_owlvit.py
  transformers/src/transformers/models/owlvit/configuration_owlvit.py
  transformers/src/transformers/models/owlvit/processing_owlvit.py
  transformers/src/transformers/models/owlvit/image_processing_owlvit.py
  transformers/src/transformers/models/owlvit/image_processing_pil_owlvit.py
  transformers/tests/models/owlvit/test_modeling_owlvit.py
  transformers/tests/models/owlvit/test_processing_owlvit.py
  transformers/tests/models/owlvit/test_image_processing_owlvit.py

Any missing files or assumptions:
  No remote code is required for the official open Google checkpoints. The
  google/*-ensemble repos returned 401 during this audit, so they are not
  treated as required representative configs. This report targets inference;
  contrastive loss and detection losses are deferred.
```

Pinned source URLs:

- `modeling_owlvit.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlvit/modeling_owlvit.py
- `configuration_owlvit.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlvit/configuration_owlvit.py
- `processing_owlvit.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlvit/processing_owlvit.py
- `image_processing_owlvit.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/owlvit/image_processing_owlvit.py

## 2. High-level architecture

OWL-ViT is a CLIP-style dual encoder reused for open-vocabulary detection. The base `OwlViTModel` runs a causal-masked text Transformer and a ViT image Transformer, projects pooled outputs, L2-normalizes them, and forms image-text similarity logits. `OwlViTForObjectDetection` goes further: it uses the vision patch tokens as dense detection locations, combines each patch token with the CLS token, predicts class embeddings and boxes per patch, and scores each patch against text-query or image-query embeddings.

```text
CPU image processor -> pixel_values[B,3,H,W]
  -> ViT patch conv + CLS + learned/interpolated positions -> vision encoder
  -> post LayerNorm over all tokens -> patch tokens * broadcast CLS
  -> detection LayerNorm -> feature_map[B,Hp,Wp,V]
  -> flatten patches[B,Hp*Wp,V] -> class head + box MLP

text prompts -> CLIP tokenizer -> input_ids[B*Q,S], attention_mask[B*Q,S]
  -> causal text encoder -> argmax-token pool -> text projection
  -> reshape to query_embeds[B,Q,D]

patch class embeddings x query embeddings -> logits[B,Hp*Wp,Q]
box MLP + grid box bias -> sigmoid cxcywh boxes[B,Hp*Wp,4]
postprocess -> thresholded variable-length xyxy boxes / labels / scores
```

Stage decomposition:

- CPU/data pipeline: CLIP tokenizer, nested prompt padding to `max_text_queries`, image resize/rescale/normalize/RGB conversion to NCHW `pixel_values`.
- Independently cacheable text branch: text encoder plus `text_projection`; text query embeddings can be reused across images.
- Independently cacheable image branch: ViT feature map, patch class embeddings, and box predictions can be reused for multiple query sets.
- Detection head: patch-feature flatten, normalized class embedding dot products, learned logit shift/scale, box bias and sigmoid.
- Postprocessing: object detection thresholding and box conversion is required for end-to-end parity. Image-guided detection additionally has source NMS-like suppression in postprocess.

Contrasts with prior CLIP and DETR audits:

- Like CLIP, OWL-ViT has dual encoders, causal text attention, projection heads, L2 normalization, and `exp(logit_scale)` for base contrastive logits.
- Unlike CLIP, the primary product output is not only an image-text similarity matrix. Detection uses all vision patch tokens after a patch*CLS interaction and separate class/box heads.
- Unlike DETR, there is no transformer decoder, no learned object queries, no bipartite matching in inference, and no no-object softmax class in the text-query postprocess. Each image patch is a candidate detection location and each text prompt is an open-vocabulary class.

Other heads:

- `OwlViTModel`: required staged subtarget for contrastive branch validation.
- `OwlViTTextModel` and `OwlViTVisionModel`: required for independent branch parity.
- `OwlViTForObjectDetection.image_guided_detection`: optional but important; it reuses the vision branch twice and replaces text queries with selected query-image patch embeddings.
- Training losses and contrastive loss: deferred for inference.

## 3. Important config dimensions

Source defaults from `configuration_owlvit.py`:

| Field | Text default | Vision default | Notes |
| --- | ---: | ---: | --- |
| `vocab_size` | 49408 | n/a | CLIP tokenizer vocabulary. |
| `hidden_size` | 512 | 768 | Text and vision can differ. |
| `intermediate_size` | 2048 | 3072 | Ungated MLP. |
| `num_hidden_layers` | 12 | 12 | Separate encoders. |
| `num_attention_heads` | 8 | 12 | MHA only, no GQA/MQA. |
| `head_dim` | 64 | 64 | Derived as hidden/heads. |
| `max_position_embeddings` | 16 | n/a | Short text prompt cap in official checkpoints. |
| `image_size` | n/a | 768 | Square source default. |
| `patch_size` | n/a | 32 | Non-overlap Conv2d kernel/stride. |
| `num_channels` | n/a | 3 | Processor converts to RGB by default. |
| `hidden_act` | quick_gelu | quick_gelu | Encoder MLP activation. |
| `layer_norm_eps` | 1e-5 | 1e-5 | PyTorch LayerNorm. |
| `attention_dropout` | 0.0 | 0.0 | Dropout disabled in inference. |
| `projection_dim` | 512 | 512 | Top-level default; base contrastive projection dimension. |
| `logit_scale_init_value` | 2.6592 | 2.6592 | Top-level scalar parameter; runtime uses exp. |
| KV cache | none | none | Encoder-style full pass only. |

Representative checkpoint sweep:

| Checkpoint | Text encoder | Vision encoder | Projection | Image/patch | Patch tokens | Processor | Source |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| `hf-internal-testing/tiny-random-owlvit` | 5 layers, H=8, heads=2, MLP=16, max text=16 | 12 layers, H=8, heads=2, MLP=3072 | 16 | 768 / 32 | 576 | not fetched | config.json; stress fixture |
| `google/owlvit-base-patch32` | 12 layers, H=512, heads=8, MLP=2048, max text=16 | 12 layers, H=768, heads=12, MLP=3072 | 512 | 768 / 32 | 576 | resize 768x768, no center crop | config/preprocessor |
| `google/owlvit-base-patch16` | 12 layers, H=512, heads=8, MLP=2048, max text=16 | 12 layers, H=768, heads=12, MLP=3072 | 512 | 768 / 16 | 2304 | resize 768x768, no center crop | config/preprocessor |
| `google/owlvit-large-patch14` | 12 layers, H=768, heads=16, MLP=3072, max text=16 | 24 layers, H=1024, heads=16, MLP=4096 | 768 | 840 / 14 | 3600 | resize 840x840, no center crop | config/preprocessor |

Effective defaults and omitted or historical fields:

- Current text config defaults `bos_token_id=49406`, `eos_token_id=49407`, `pad_token_id=0`; inspected Google configs carry older CLIP-style `bos_token_id=0`, `eos_token_id=2`, `pad_token_id=1`. Unlike current CLIP source, OWL-ViT pooling always uses `input_ids.argmax(dim=-1)`, so this historical compatibility behavior is part of the model path.
- `projection_intermediate_dim` appears in the tiny config but is not read by the current native source.
- The current image processor class defaults to dict sizes, but older Google preprocessor configs use integer/list forms such as `size: [768, 768]` and `crop_size: 768`; the backend normalizes these.

## 3a. Family variation traps

- Text and vision hidden sizes can differ. Detection class head maps vision hidden size to text hidden size before dotting against query embeddings.
- The text branch is CLIP-like causal self-attention, not bidirectional BERT attention, despite being used as a text encoder.
- Text pooling is `last_hidden_state[batch, input_ids.to(int).argmax(-1)]`. This relies on CLIP EOT being the highest token id in normal prompts and should be preserved exactly.
- The processor flattens nested prompt batches from `[B][Q]` into `input_ids[B*maxQ,S]`, padding missing prompts with a single-space prompt. The model reconstructs `max_text_queries = input_ids.shape[0] // image_batch`.
- Padded query detection mask is `input_ids.reshape(B,Q,S)[...,0] > 0`, not `attention_mask.any`. This interacts with historical pad/BOS ids and should be treated as a source contract.
- Vision source layout is NCHW through image processor and patch Conv2d. The feature map returned by detection is NHWC-like `[B,Hp,Wp,V]` only after Transformer tokens are reshaped.
- Position interpolation is optional. When enabled, bicubic interpolation resizes the learned square patch-position table to `(H // patch, W // patch)` and box bias is recomputed for the runtime grid.
- `OwlViTForObjectDetection` registers `box_bias` for the config grid. Dynamic/interpolated sizes recompute box bias in Python/PyTorch source; static artifacts may precompute only fixed grids.
- Object detection postprocess has no NMS. Image-guided postprocess does apply iterative IoU suppression when `nms_threshold < 1.0`.
- Source advertises SDPA/Flash/Flex attention support through shared Transformers attention dispatch. DinoML can use a fused noncausal/causal encoder attention path, but exact mask semantics must match the shared `create_causal_mask` contract.
- Axis-sensitive layout traps:
  - Source input `pixel_values` is `[B,C,H,W]`; NHWC is an optimization boundary only.
  - Patch embedding flatten is `Conv2d -> flatten(2).transpose(1,2)`, preserving NCHW row-major patch order.
  - Detection feature map is reshaped to `[B,Hp,Wp,V]`, then flattened back to `[B,Hp*Wp,V]` for heads.
  - `einsum("...pd,...qd->...pq")` outputs patch-major logits `[B,P,Q]`; do not transpose into query-major DETR-like logits.
  - Postprocess target sizes are `(height,width)`, while scaling factor order is `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `pixel_values[B,3,H,W]` input, dtype cast to patch Conv2d weight dtype.
- Non-overlap Conv2d patch embedding: `Conv2d(3 -> V, kernel=patch, stride=patch, bias=False)`.
- Flatten spatial patches and transpose to `[B,P,V]`; concat CLS token; learned position embedding add.
- Optional bicubic interpolate of patch positional embedding table from `[1,G,G,V]` to `[1,Hp,Wp,V]`.
- Token embedding and learned text position embedding add.
- Reshape text queries `[B*Q,S] <-> [B,Q,S]`; reshape feature maps `[B,P,V] <-> [B,Hp,Wp,V]`.
- Broadcast CLS token to patch-token shape and elementwise multiply.
- L2 norm reductions over last dim with epsilon behavior in detection class head.
- `einsum`/batched matmul for `[B,P,T] x [B,Q,T] -> [B,P,Q]` query logits.
- `where` mask fill with dtype min for invalid query prompts.
- Meshgrid/arange/stack/clip/log/log1p/concat for box bias if computed in graph.
- Postprocess boolean thresholding and variable-length per-image records.
- Image-guided selection uses loops, box IoU, generalized IoU, nonzero, argmin, mean, and stack; first integration can keep this outside the compiled graph.

Neural network primitives:

- Linear with bias for Q/K/V/O attention projections in both encoders.
- Linear with bias for encoder MLP `hidden -> intermediate -> hidden`.
- Linear without bias for base `visual_projection` and `text_projection`.
- Detection `class_head.dense0`: `vision_hidden -> text_hidden`, plus `logit_shift: vision_hidden -> 1` and `logit_scale: vision_hidden -> 1`.
- Detection `box_head`: three-layer MLP `vision_hidden -> vision_hidden -> vision_hidden -> 4` with GELU between layers.
- LayerNorm before each attention/MLP sublayer, text final LayerNorm, vision pre/post LayerNorm, and detection LayerNorm.
- QuickGELU for encoder MLPs; exact GELU for detection box MLP; ELU+1 for learned detection logit scale.
- Sigmoid for normalized boxes and postprocess scores.

Attention primitives:

- Text self-attention: causal MHA, no cache, shape `[B*Q,S,H]`.
- Vision self-attention: noncausal MHA, no mask in normal path, shape `[B,1+P,V]`.
- Shared attention backend can be eager/SDPA/Flash/Flex; no cross-attention.

Preprocessing-coupled ops:

- CLIPTokenizer with lower-casing and `model_max_length=16` for inspected Google checkpoints.
- Processor defaults `padding="max_length"` and `return_tensors="pt"`.
- Nested text prompt padding to equal query count per image.
- Image resize to square model size, bicubic, no center crop for inspected Google preprocessors, rescale, normalize with OpenAI CLIP mean/std, RGB conversion.

Detection postprocess ops:

- Text-query detection: `max(logits, dim=-1)`, sigmoid score, label index, `center_to_corners_format(cxcywh)`, optional scaling to target sizes, threshold filter. No NMS.
- Image-guided detection: same box conversion/scaling plus iterative IoU suppression and alpha-like score scaling.

Parameter aliasing:

- No ALBERT-style cross-layer sharing. Base `OwlViTForObjectDetection` owns one nested `OwlViTModel`; detection heads are separate. The registered `box_bias` is nonpersistent and derived from config or runtime grid, not a learned weight.

## 5. Layer/block breakdown

Text branch:

```text
input_ids[B*Q,S]
  -> token_embedding + position_embedding
  -> causal mask from attention_mask
  -> repeat text layer N times:
       residual = x
       x = LayerNorm(x)
       x = MHA(q_proj,k_proj,v_proj,out_proj, causal additive mask)
       x = residual + x
       residual = x
       x = LayerNorm(x)
       x = Linear(H -> I) -> quick_gelu -> Linear(I -> H)
       x = residual + x
  -> final LayerNorm
  -> pool at argmax(input_ids, dim=-1)
  -> text_projection(H_text -> projection_dim, bias=False)
```

Vision branch:

```text
pixel_values[B,3,H,W]
  -> Conv2d(3 -> V, kernel=patch, stride=patch, bias=False)
  -> flatten patches to [B,Hp*Wp,V]
  -> prepend learned class embedding
  -> add learned/interpolated position embedding
  -> pre LayerNorm
  -> repeat vision layer N times:
       LayerNorm -> noncausal MHA -> residual add
       LayerNorm -> Linear(V -> I) -> quick_gelu -> Linear(I -> V) -> residual add
  -> pooled CLS = post LayerNorm(last_hidden_state[:,0])
  -> visual_projection(V -> projection_dim, bias=False) for base contrastive path
```

Detection feature path:

```text
vision_last_hidden_state[B,1+P,V]
  -> post LayerNorm over all tokens
  -> class_token_out = broadcast token 0 to patch shape [B,P,V]
  -> patch_tokens = tokens[:,1:,:] * class_token_out
  -> detection LayerNorm
  -> feature_map[B,Hp,Wp,V]
  -> image_feats[B,P,V]
```

Detection class head:

```text
image_class_embeds = Linear(V -> T)(image_feats)
image_class_embeds = image_class_embeds / (norm + 1e-6)
query_embeds = query_embeds / (norm + 1e-6)
pred_logits = einsum("...pd,...qd->...pq")
pred_logits = (pred_logits + Linear(V -> 1)(image_feats)) * (ELU(Linear(V -> 1)(image_feats)) + 1)
pred_logits = where(query_mask == 0, finfo.min, pred_logits).to(float32)
```

Detection box head:

```text
raw_boxes = Linear(V -> V) -> GELU -> Linear(V -> V) -> GELU -> Linear(V -> 4)
box_bias = [logit(grid_center_x), logit(grid_center_y), logit(1/Wp), logit(1/Hp)]
pred_boxes = sigmoid(raw_boxes + box_bias)
```

## 6. Attention requirements

Text attention:

- Causal self-attention, full-sequence encoder pass, no KV cache.
- MHA only: `num_key_value_heads == num_attention_heads`.
- Head counts: base text 8 heads with H=512/head_dim=64; large text 16 heads with H=768/head_dim=48.
- Masking: `create_causal_mask` combines causal and padding mask semantics. Source passes `is_causal=True` into the encoder and removes any incoming `is_causal` kwarg.
- Dropout is zero in inference.
- Flash/SDPA compatibility should be good for fixed prompt length 16, but a first implementation can use eager attention for parity.

Vision attention:

- Noncausal self-attention over `[CLS + patches]`; no attention mask in normal inference.
- Base patch32 sequence length is 577, base patch16 is 2305, large patch14 is 3601 at official processor sizes.
- Large-patch14 attention is a major bottleneck: 24 layers, 3601 tokens, 16 heads.
- No relative bias, RoPE, ALiBi, sliding window, varlen packing, or cache.

Source-specific math order:

- Q/K/V are projected, reshaped to `[B,heads,S,head_dim]`, attention scores are scaled by `head_dim**-0.5`, additive mask is applied, then softmax/dropout, value matmul, transpose/contiguous, output projection.
- Detection class similarity normalizes projected patch embeddings and query embeddings before `einsum`, then applies learned shift and positive scale.

## 7. Position encoding and custom math

Text positions are learned absolute embeddings sliced to sequence length. Vision positions are learned absolute patch embeddings plus a learned CLS position. Optional interpolation is copied from CLIP/DINO-style ViT interpolation.

Position interpolation sketch:

```python
def interpolate_vision_pos(weight, height, width, patch):
    cls = weight[:1]
    patch_weight = weight[1:]              # [G*G, V]
    G = int((patch_weight.shape[0]) ** 0.5)
    patch_weight = patch_weight.reshape(1, G, G, V).permute(0, 3, 1, 2)
    patch_weight = bicubic_interpolate(
        patch_weight, size=(height // patch, width // patch), align_corners=False
    )
    patch_weight = patch_weight.permute(0, 2, 3, 1).reshape(1, -1, V)
    return concat(cls.reshape(1, 1, V), patch_weight, dim=1)
```

Box bias math:

```python
def box_bias(Hp, Wp):
    x = arange(1, Wp + 1) / Wp
    y = arange(1, Hp + 1) / Hp
    xx, yy = meshgrid(x, y, indexing="xy")
    center = clip(stack([xx, yy], -1).reshape(-1, 2), 0.0, 1.0)
    center_bias = log(center + 1e-4) - log1p(-center + 1e-4)
    size = full_like(center, 1.0)
    size[:, 0] /= Wp
    size[:, 1] /= Hp
    size_bias = log(size + 1e-4) - log1p(-size + 1e-4)
    return concat([center_bias, size_bias], dim=-1)
```

Precompute opportunities:

- Fixed-size artifacts can precompute `position_ids`, text position embeddings lookup pattern, vision position embeddings, and `box_bias`.
- Dynamic/interpolated image sizes need runtime position interpolation and runtime box-bias recomputation, or guarded buckets with precomputed tables.

## 8. Preprocessing and input packing

Image processor contract from inspected Google preprocessors:

- Input images are converted to RGB.
- Resize to square `768x768` for base checkpoints or `840x840` for large-patch14; bicubic resampling.
- `do_center_crop=false` in official Google configs, despite source class having crop settings.
- Rescale and normalize with OpenAI CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Output `pixel_values[B,3,H,W]`.

Text processor contract:

- Uses `CLIPTokenizer`; tokenizer configs report lower-casing, BOS `<|startoftext|>`, EOS/UNK `<|endoftext|>`, pad token `"!"`, `model_max_length=16`.
- Processor default text kwargs set `padding="max_length"` and default common kwargs set `return_tensors="pt"`.
- Flat text list produces `input_ids[Q,S]`; nested list `[B][Q_i]` pads each image to `max(Q_i)` by appending `" "` prompts, then concatenates to `[B*maxQ,S]`.
- `attention_mask[B*Q,S]` goes into text causal mask; padded-query validity later uses first token id.

Detection postprocess:

- Raw text-query outputs: `logits[B,P,Q]`, `pred_boxes[B,P,4]` in normalized `cxcywh`.
- Object detection postprocess takes max over query dimension, applies sigmoid to the max logit, converts boxes to `xyxy`, optionally scales by target sizes `(height,width)`, then filters `scores > threshold`. There is no NMS in this path.
- `post_process_grounded_object_detection` adds `text_labels` by indexing each image's prompt list with output labels.
- Image-guided postprocess starts from `logits[B,P,Q]` and `target_pred_boxes[B,P,4]`, applies source IoU suppression when `nms_threshold < 1.0`, then threshold/alpha scoring and optional scaling.

Image-guided detection runtime contract:

- Processor with `query_images` returns `query_pixel_values` and clears text data; query images override text prompts.
- Model runs the vision encoder for target and query images.
- Query embedding selection computes query-image class embeddings and boxes, picks patches overlapping the full image box, and stacks one selected embedding per query image. This path has dynamic selection control flow and can be CPU/PyTorch-side initially while the two vision encoders and target scoring are compiled.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> GEMM

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- Input is dense NCHW or an explicitly supported NHWC internal layout.
- `H` and `W` are divisible by `patch_size`.

Replacement:

```text
WindowFlatten[B,Hp*Wp,3*patch*patch] -> MatMul(weight_flat.T) -> [B,Hp*Wp,V]
```

Weight transform:

```python
w = conv.weight.reshape(V, 3 * patch * patch)
```

Layout constraints and failure cases:

- Flatten order must match PyTorch NCHW Conv2d patch order. NHWC lowering needs an explicit weight permutation or a layout-aware im2col.
- Do not apply if dynamic image sizes are not divisible by patch or if interpolation/bucket handling cannot preserve patch order.

Parity test sketch:

- Compare Conv2d patch output after `flatten(2).transpose(1,2)` against lowered WindowFlatten+GEMM for patch16 and patch32.

### Rewrite: detection class einsum -> batched GEMM

Preconditions:

- Patch class embeddings `[B,P,T]` and query embeddings `[B,Q,T]` are contiguous or have supported strides.
- Query embeddings have already been normalized and reshaped to `[B,Q,T]`.

Replacement:

```text
BatchMatMul(PxT, QxT^T) -> [B,P,Q]
```

Failure cases:

- Flat contrastive `OwlViTModel` logits use text-by-image orientation; do not reuse that orientation for detection logits.
- Query mask fill must happen after shift/scale in source order.

Parity test sketch:

- Random normalized tensors plus learned shift/scale; compare source `einsum` and GEMM path before and after query mask fill.

### Rewrite: fixed-grid box bias as constant

Preconditions:

- `interpolate_pos_encoding=False`.
- Runtime image size matches `vision_config.image_size`.
- Patch size and grid dimensions are static.

Replacement:

```text
box_head(image_feats) + precomputed_box_bias[P,4] -> sigmoid
```

Failure cases:

- Interpolated/dynamic image sizes require box bias for the runtime `Hp,Wp`. Guard fixed-grid artifacts and fall back to runtime computation or precomputed bucket tables.

Parity test sketch:

- Compare `compute_box_bias(config_grid)` with serialized constant and verify first boxes for base-patch32.

### Rewrite: patch*CLS feature fusion as fused elementwise + LayerNorm

Preconditions:

- Vision last hidden state is `[B,1+P,V]`.
- CLS token is token 0 and patch tokens are tokens 1:.

Replacement:

```text
post_layernorm(all_tokens)
cls = broadcast(tokens[:,0:1,:], [B,P,V])
patch = tokens[:,1:,:] * cls
LayerNorm(patch)
```

Optimization:

- Fuse broadcast multiply with the following LayerNorm read where practical.

Failure cases:

- Do not use the base CLIP pooled projection output for detection features. Detection needs all patch tokens after post LayerNorm and CLS multiplication.

### Layout pass candidate: local NHWC vision region

Preconditions:

- Boundary transpose from source NCHW is explicit.
- Patch embedding, token flatten, and downstream reshape preserve row-major `(h,w)` patch order.
- Axis rewrites are applied for spatial operations and interpolation.

Protected regions:

- Processor output contract, source-level Conv2d semantics, and postprocess box scaling should be treated as no-layout-translation boundaries.

Failure cases:

- Silent NHWC conversion that changes patch flatten order will scramble box-location bias and detection boxes.

## 10. Kernel fusion candidates

Highest priority:

- ViT patch embedding as GEMM/conv: base-patch16 and large-patch14 produce thousands of tokens; efficient patch projection matters.
- Encoder LayerNorm + QKV projection + attention + output projection: shared with CLIP/ViT, but large-patch14 has 3601 vision tokens and needs efficient attention.
- Detection class head normalization + batched GEMM + shift/scale: this is the open-vocabulary scoring core and easy to validate independently.
- Box MLP + box bias + sigmoid: small but latency-sensitive and required for detection parity.

Medium priority:

- QuickGELU MLP fusion for text/vision encoders.
- Patch*CLS multiply + detection LayerNorm fusion.
- Fixed-grid position and box-bias constant folding for non-interpolated official sizes.
- Postprocess threshold/box conversion kernel for large `P` values if CPU postprocess becomes visible.

Lower priority:

- Dynamic bicubic position interpolation inside the compiled graph; guarded fixed-size/bucket support is simpler first.
- Image-guided query embedding selection and postprocess NMS in compiled runtime; dynamic loops make this a later structured-output target.
- Contrastive loss and detection losses; training-only.

## 11. Runtime staging plan

Stage 1: parse configs/processors and load weights for `OwlViTTextModel`, `OwlViTVisionModel`, and `OwlViTModel`. Validate text and image branch outputs independently on fixed official sizes.

Stage 2: compile base CLIP-like `OwlViTModel` contrastive inference: text/image embeddings, L2 normalization, `exp(logit_scale)`, and logits orientations.

Stage 3: add detection feature map extraction from vision last hidden state: post LayerNorm all tokens, patch*CLS multiply, detection LayerNorm, reshape to `[B,Hp,Wp,V]`.

Stage 4: add text-query detection heads: class head, query reshape/mask, batched patch-query logits, box MLP, fixed-grid box bias, sigmoid.

Stage 5: implement CPU or small GPU postprocess for text-query detection: max over queries, sigmoid, `cxcywh -> xyxy`, target-size scaling, threshold, text label mapping.

Stage 6: add guarded `interpolate_pos_encoding=True` support through bucketed image sizes or runtime interpolation plus dynamic box bias.

Stage 7: add image-guided detection: compile both vision passes, keep query embedding selection and NMS outside compiled runtime first, then lower stable pieces.

Initial stubs:

- Keep tokenizer/image processor on CPU.
- Keep variable-length result assembly on CPU.
- Require fixed official image sizes and `interpolate_pos_encoding=False` for first model artifacts.
- Defer training losses and image-guided NMS kernels.

## 12. Parity and validation plan

Concrete tests:

- Config tests for base-patch32, base-patch16, large-patch14, and tiny-random, including derived patch token counts.
- Patch embedding rewrite parity against PyTorch Conv2d for fixed NCHW inputs.
- One text layer and one vision layer parity with fp32 random tensors and fixed masks.
- Full `OwlViTModel` parity for `get_text_features`, `get_image_features`, `logits_per_image`, and `logits_per_text`.
- Detection feature parity: compare `feature_map[B,Hp,Wp,V]` after patch*CLS and detection LayerNorm.
- Detection class head parity with multiple images and ragged nested prompt lists padded by processor.
- Box head parity including `box_bias` and sigmoid for base-patch32, base-patch16, and large-patch14 grids.
- End-to-end text-query object detection parity using the Transformers slow-test cat image and prompts `["a photo of a cat", "a photo of a dog"]`.
- Postprocess parity: thresholded labels/scores/boxes with and without `target_sizes`.
- Image-guided smoke parity: query and target image produce matching raw output shapes and selected postprocess boxes.

Tolerances:

- fp32 encoder/head parity: start with `rtol=1e-4`, `atol=1e-4`; tighten sub-op tests where reductions are deterministic.
- fp16 CUDA: use looser end-to-end tolerances, for example `rtol=1e-2`, `atol=1e-2`, especially around attention, L2 norms, and sigmoid boxes.
- Postprocess labels/keep masks should be exact for fp32; near-threshold score tests should avoid ambiguous thresholds.

## 13. Performance probes

- CPU processor throughput: image resize/normalize and CLIP tokenization, separated from model runtime.
- Text encoder throughput by `B*Q` and sequence length 16; cache reusable query embeddings.
- Vision encoder throughput for patch32 577 tokens, patch16 2305 tokens, and large patch14 3601 tokens.
- Attention backend comparison for vision sequence lengths: eager/SDPA/Flash-equivalent.
- Detection head throughput as `P x Q` varies, especially many prompts per image.
- Postprocess throughput for thresholding/scaling variable-length outputs; compare CPU vs GPU threshold pipeline.
- Batch-size sweep for `B=1,2,4,8` at official image sizes.
- Interpolated image-size sweep if dynamic image support is enabled.
- Image-guided path probe: target vision pass, query vision pass, query selection, target scoring, and NMS/postprocess separately.

## 14. Skip/defer list

- Training losses: contrastive loss, L1/GIoU detection losses, generalized IoU except image-guided query selection fallback.
- Gradient checkpointing and dropout training behavior.
- Gated/private ensemble checkpoints; not accessible during this audit.
- Dynamic arbitrary image sizes in the first artifact; start fixed official sizes.
- Compiled CLIP tokenizer and image resizing; CPU/data pipeline first.
- Compiled image-guided query selection and NMS; keep as framework-side first.
- Quantization, multi-GPU tensor parallelism, distributed inference.
- Generation/prefill/decode/KV cache; not applicable.

## 15. Final implementation checklist

- [ ] Parse `OwlViTConfig`, nested text/vision configs, and processor configs.
- [ ] Load text, vision, projection, detection class-head, detection box-head weights.
- [ ] Implement CLIP-style text encoder with argmax-token pooling.
- [ ] Implement ViT image encoder with NCHW patch Conv2d and fixed learned positions.
- [ ] Add optional guarded position interpolation for vision positions.
- [ ] Implement base `OwlViTModel` contrastive embeddings and logits orientations.
- [ ] Implement detection feature extraction from all vision tokens, including patch*CLS and detection LayerNorm.
- [ ] Implement detection class head: dense0, L2 normalize, batched patch-query GEMM, shift/ELU-scale, query mask fill.
- [ ] Implement detection box head and fixed-grid box-bias constant.
- [ ] Implement text-query postprocess: max query, sigmoid score, `cxcywh -> xyxy`, target-size scale, threshold, label/text-label mapping.
- [ ] Add fixed-size parity tests for base-patch32, base-patch16, and large-patch14.
- [ ] Add nested-prompt packing/query-mask parity tests.
- [ ] Add patch embedding Conv2d-to-GEMM rewrite test.
- [ ] Add detection head microbenchmarks over patch-token and prompt-count sweeps.
- [ ] Add image-guided detection as a second-stage target with framework-side query selection and NMS first.
