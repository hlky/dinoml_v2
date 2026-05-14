# EdgeTAM Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/EdgeTAM at refs/pr/1; yonigozlan/EdgeTAM-hf mirror/config
Config source: HF config.json plus Transformers config defaults
Source files inspected: edgetam modeling/config/modular/convert, SAM2 processor/image processor, boundary-only edgetam_video modeling/config
Any missing files or assumptions: RepViT/TimmWrapper backbone not audited here; full video tracker state machine belongs to edgetam_video
```

Primary DinoML target for this report: single-image promptable segmentation using the EdgeTAM image encoder/FPN, prompt encoder, and mask decoder. The public HF checkpoint config advertises `EdgeTamVideoModel` / `model_type="edgetam_video"`, so end-to-end public checkpoint parity is gated on an adjacent video-state audit. This report treats video memory as a boundary and focuses the runnable first slice on the stateless image SAM-style call.

Generated-file note: `configuration_edgetam.py` is generated from `modular_edgetam.py`; future Transformers source edits should use the modular file.

## 2. High-level architecture

EdgeTAM is a prompt-conditioned segmentation model. The plain `edgetam` model is:

```text
CPU image/point/box/mask preprocessing
-> RepViT TimmWrapper image backbone
-> EdgeTAM FPN neck + 2D sine position encodings
-> prompt encoder for points/boxes/masks
-> two-way mask decoder
-> low-res mask logits + IoU scores + object-score logits
-> CPU/GPU postprocess resize/threshold/non-overlap
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize to `1024x1024`, rescale/normalize, channels-first packing, original-size capture, point/box normalization to processor target size, padding point labels with `-10`.
- Independently cacheable image encoder: `pixel_values [B,3,1024,1024] -> image_embeddings` as three FPN levels. This is explicitly exposed by `get_image_embeddings` and `image_embeddings` input to `forward`.
- Prompt encoder: points/boxes/masks can be recomputed per click without recomputing the image backbone.
- Mask decoder: prompt-conditioned two-way attention and hypernetwork mask projection. This is the first high-value DinoML runtime target after image features are available.
- Postprocess: resize low-res logits to original image sizes and optionally binarize or enforce non-overlap. This is required for end-to-end segmentation parity but can be staged outside the compiled graph first.

Video boundary:

```text
video processor/frame sampling -> per-frame image features -> session memory lookup
-> memory attention / object pointers -> single-frame SAM head
-> memory encoder + spatial perceiver update -> propagation state
```

The video model persists per-frame/object outputs, object pointers, encoded mask memories, temporal positions, and reverse/forward propagation state. Treat that as a session/state ABI, not a KV cache.

## 3. Important config dimensions

Accessible public configs (`facebook/EdgeTAM` refs/pr/1 and `yonigozlan/EdgeTAM-hf`) share the same operator-significant dimensions.

| Field | Value | Source |
| --- | --- | --- |
| top-level architecture | `EdgeTamVideoModel` | HF `config.json` |
| top-level dtype | `float32` | HF `config.json` |
| image size | `1024` | HF `config.json` / processor |
| processor output layout | channels-first | `preprocessor_config.json` |
| backbone | `timm_wrapper`, `repvit_m1`, `features_only=True`, `out_indices=[0,1,2,3]` | HF `config.json` |
| backbone channels consumed by FPN | `[384,192,96,48]` | config |
| FPN hidden size | `256` | config |
| FPN feature sizes consumed by mask head | `[[256,256],[128,128],[64,64]]` | config |
| prompt hidden size | `256` | config |
| prompt patch size | `16` | config |
| prompt image embedding size | `64x64` | inferred from `1024/16` |
| dense mask prompt input size | `256x256` | inferred from `4*1024/16` |
| prompt mask channels | `16` | config |
| point embeddings | `4` point/box semantic embeddings plus not-a-point | source/config |
| mask decoder layers | `2` two-way blocks | config |
| mask decoder attention heads | `8` | config |
| attention downsample rate | `2` for cross-attn, `1` for sparse self-attn | source/config |
| cross-attn internal dim/head dim | `128 / 16` | inferred from `256 / 2 / 8` |
| sparse self-attn internal dim/head dim | `256 / 32` | inferred from `256 / 1 / 8` |
| mask tokens | `4` total: single + 3 multimask | source/config |
| multimask outputs | `3` when `multimask_output=True` | config |
| IoU head | MLP depth `3`, hidden `256`, sigmoid output | config/source |
| postprocess mask resize | bilinear to original `(H,W)`, threshold `0.0` by default | processor |

Representative checkpoint sweep:

| Model/config | Availability | Operator-significant notes |
| --- | --- | --- |
| `facebook/EdgeTAM` `refs/pr/1` | accessible | Public config is video model, RepViT M1 wrapper, same dimensions above. |
| `yonigozlan/EdgeTAM-hf` | accessible mirror/config | Same public operator shape observed; model card tags `edgetam_video`. |
| `danelcsb/edgetam.1_hiera_tiny` | inaccessible/API returned auth error | Mentioned in source docstring; could resolve historical image-only doc mismatch if accessible. |

## 3a. Family variation traps

- Plain `edgetam` source is image promptable segmentation, but accessible public checkpoints use `edgetam_video`. DinoML should reject or route `model_type="edgetam_video"` to a separate video-state path until that audit is complete.
- The vision backbone is delegated to `TimmWrapperConfig`; EdgeTAM wrapper only defines the consumed feature contract. Do not infer RepViT op coverage from this audit.
- The backbone/FPN boundary flips layouts: processor and backbone use NCHW, backbone feature maps are permuted to NHWC, FPN immediately permutes back to NCHW for Conv2d.
- Mask decoder uses sequence layout `HW x B x C` in the cached image feature path, then reconstructs NCHW. This needs strict shape/layout guards.
- `attention_similarity` is an additive float attention bias for target-guided prompting. FlashAttention is explicitly bypassed for this call because additive bias masks are not compatible with the requested FA path.
- `multimask_output=False` is not simply slice token 0 in inference: dynamic stability can pick the best multimask output by IoU if token-0 stability is below threshold.
- Boxes cannot be ragged-padded by the processor; unequal box counts across images are rejected.
- Point padding uses label `-10`, distinct from source labels `1`, `0`, and `-1`.
- Video config fields such as memory attention, memory encoder, object pointers, spatial perceiver, and propagation are ignored by the plain `edgetam` image model but required for public `EdgeTamVideoModel`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors, NHWC feature-map islands, `permute`, `flatten(2)`, `view`, `unsqueeze`, `repeat`, `repeat_interleave`, `cat`, `stack`, `gather`, `argmax`, `where`, boolean comparisons, slicing.
- Layout-sensitive path: FPN hidden states are emitted as `HW x B x C`; mask decoder wants NCHW `[B,256,64,64]` plus high-res `[B,32,256,256]` and `[B,64,128,128]`.

Neural primitives:

- External RepViT/TimmWrapper backbone: separately audited allowlist required.
- Conv2d `1x1` FPN projections from `[384,192,96,48] -> 256`.
- Nearest upsample by factor 2 in FPN.
- LayerNorm over NHWC and channels-first LayerNorm implemented as NCHW->NHWC->NCHW.
- Prompt mask embedding: Conv2d `1->4` k2/s2, Conv2d `4->16` k2/s2, Conv2d `16->256` k1/s1 with GELU and channels-first LayerNorm.
- Mask decoder upscaling: ConvTranspose2d `256->64` k2/s2, ConvTranspose2d `64->32` k2/s2, GELU, high-res skip additions.
- Linear/MLP: attention q/k/v/o projections, feed-forward blocks, hypernetwork MLPs, IoU head, object score head.

Attention primitives:

- Noncausal dense MHA for sparse self-attention and bidirectional sparse/image cross-attention.
- Additive float bias mask support for target-guided attention.
- Softmax computed in fp32 in eager path, cast back to query dtype.
- No autoregressive cache in plain image model.

Position/custom math:

- Random Fourier prompt positional embedding: normalize coords to `[0,1]`, map to `[-1,1]`, multiply by learned/buffer Gaussian matrix, apply `sin/cos`.
- FPN sine/cosine 2D embedding from cumulative valid-pixel coordinates, normalized to `2*pi`.

Preprocessing/postprocessing:

- Resize/rescale/normalize RGB images to channels-first `1024x1024`.
- Normalize points/boxes by scale from original image size to target size.
- Resize input segmentation maps to mask size `256x256`.
- Resize predicted masks to original sizes with bilinear interpolation and optional threshold/non-overlap.

State/video ops, deferred:

- Session-owned frame/object dictionaries, memory frame selection, object pointer concatenation, memory attention, mask-memory encoder, spatial perceiver, bidirectional propagation ordering.

## 5. Layer/block breakdown

Image encoder wrapper:

```text
pixel_values [B,3,1024,1024]  # NCHW
backbone(pixel_values) -> 4 NCHW feature maps from RepViT/TimmWrapper
for each feature: NCHW -> NHWC
FPN neck:
  NHWC -> NCHW
  Conv2d(C_i -> 256, k=1,s=1,p=0)
  selected top-down levels add nearest_upsample(prev, scale=2)
  position_encoding(NCHW shape) -> [B,256,H,W]
select last 3 levels and reverse to high->low resolution
get_image_features:
  conv_s0: [B,256,256,256] -> [B,32,256,256]
  conv_s1: [B,256,128,128] -> [B,64,128,128]
  flatten each NCHW to [HW,B,C]
```

Prompt encoder:

```text
points [B,P,N,2], labels [B,P,N]
  points += 0.5
  optional pad one point with label -1 when no boxes
  normalize by input_image_size=(1024,1024)
  random_fourier(coords) -> [B,P,N,256]
  label -1 -> not_a_point embedding
  label -10 -> zero embedding
  labels >= 0 -> add point_embed[label]

boxes [B,P,4]
  boxes += 0.5
  reshape to two corners + padded third point
  random_fourier -> [B,P,3,256]
  corner 0 adds point_embed[2], corner 1 adds point_embed[3]

masks [B,1,H,W] or resized to [B,1,256,256]
  Conv2d/GELU/LayerNorm -> dense [B,256,64,64]
else
  no_mask_embed expanded to [B,256,64,64]

sparse_embeddings = concat(points, boxes) along prompt-token axis
```

Two-way decoder block, repeated 2 times:

```text
queries: [B,P,T,256]        # output tokens + sparse prompts
keys:    [B,P,4096,256]     # image tokens from [B,256,64,64]

if not first block:
  queries = LN(queries + SelfAttention(queries + query_pe, queries + query_pe, queries))
else:
  queries = LN(SelfAttention(queries, queries, queries))

queries = LN(queries + CrossAttention(queries + query_pe, keys + image_pe, keys, additive_bias?))
queries = LN(queries + MLP(256 -> 2048 -> 256))
keys    = LN(keys + CrossAttention(keys + image_pe, queries + query_pe, queries))
```

Mask decoder head:

```text
tokens = [object_score_token, iou_token, 4 mask_tokens, sparse_prompts]
image_embeddings += dense_prompt_embeddings
repeat image embeddings by point_batch_size
two_way_transformer(...)
iou_token_out = tokens[:, :, 1, :]
mask_tokens_out = tokens[:, :, 2:6, :]

image_tokens -> [B*P,256,64,64]
upscale:
  deconv 256->64 to 128x128 + high_res feat_s1 [B*P,64,128,128]
  LayerNorm/GELU
  deconv 64->32 to 256x256 + high_res feat_s0 [B*P,32,256,256]
  GELU

for each mask token:
  hyper_i = MLP(256 -> 256 -> 256 -> 32)
masks = hyper @ flattened_upscaled_embedding
      -> [B,P,4,256,256]
iou_scores = MLP(256 -> 256 -> 256 -> 4) with sigmoid
object_score_logits = MLP(256 -> 256 -> 256 -> 1)
```

## 6. Attention requirements

Plain image EdgeTAM attention is noncausal and has no KV cache.

| Attention site | Query | Key/value | Heads | Head dim | Masking |
| --- | --- | --- | --- | --- | --- |
| sparse self-attn | output/prompt tokens | same sparse tokens | 8 | 32 | none |
| token-to-image cross-attn | sparse tokens | 64x64 image tokens | 8 | 16 | optional additive `attention_similarity` |
| image-to-token cross-attn | 64x64 image tokens | sparse tokens | 8 | 16 | none |
| final token-to-image cross-attn | sparse tokens | 64x64 image tokens | 8 | 16 | none in source call |

The q/k/v projection width is `hidden_size/downsample_rate`. Cross-attention uses `128` internal width; sparse self-attention uses `256`. All projections have PyTorch `nn.Linear` default bias.

Flash/SDPA compatibility:

- Dense SDPA can cover the standard noncausal paths.
- Target-guided `attention_similarity` is an additive float mask; source forces an SDPA fallback if FlashAttention was requested.
- A safe first DinoML lowering should support eager/SDPA-style additive bias before enabling FlashAttention for these calls.

Video state attention, deferred:

- `edgetam_video` adds memory attention over current frame tokens and concatenated memory/object-pointer tokens. This is cross-attention over session state, not generation cache. It must be audited with the exact `EdgeTamVideoMemoryAttention` and spatial perceiver contracts before public video parity.

## 7. Position encoding and custom math

Prompt coordinate embedding:

```python
def prompt_pe(coords_xy, input_shape=(1024, 1024), gaussian):
    coords = coords_xy.clone()
    coords[..., 0] = coords[..., 0] / input_shape[1]
    coords[..., 1] = coords[..., 1] / input_shape[0]
    coords = 2 * coords - 1
    phase = 2 * pi * (coords @ gaussian)  # gaussian [2, hidden/2]
    return concat(sin(phase), cos(phase), dim=-1)
```

This depends on prompt coordinates but the Gaussian matrix is a model buffer and can be constant-loaded.

FPN sine position embedding:

```python
not_mask = (~mask).to(dtype)  # default mask is all valid
y = cumsum(not_mask, dim=1)
x = cumsum(not_mask, dim=2)
y = y / (y[:, -1:, :] + 1e-6) * (2*pi)
x = x / (x[:, :, -1:] + 1e-6) * (2*pi)
dim_t = temperature ** (2 * floor(arange(C/2) / 2) / (C/2))
pos = concat([sin/cos(y/dim_t), sin/cos(x/dim_t)], dim=channel).permute(NHWC_to_NCHW)
```

For fixed `1024x1024` input and all-valid masks, FPN positional encodings for `256x256`, `128x128`, and `64x64` are precomputable per dtype/device. Prompt PE is dynamic per user prompt.

## 8. Preprocessing and input packing

Image processor:

- Converts to RGB, resizes to `1024x1024`, rescales by `1/255`, normalizes with ImageNet mean/std, emits `pixel_values` in channels-first layout.
- `original_sizes` is captured from input image shape before resize.
- No SAM-style pad is active in the inspected processor defaults (`do_pad=null`, `pad_size=null`).
- Segmentation maps, if supplied to processor, resize to `mask_size={"height":256,"width":256}` with nearest behavior.

Prompt processor:

- Points are nested `[image, object/point_batch, point, xy]`; padded to rectangular shape with point pad value `-10`.
- Labels are nested `[image, object/point_batch, point]`; labels must match point nesting.
- Boxes are `[image, box, xyxy]`; ragged per-image box counts requiring padding are rejected.
- Coordinates are normalized from original `(H,W)` to processor target size. Padding points retain `-10`.

Model prompt ABI:

- `input_points`: `[B,P,N,2]`, `float`, x/y order from docstring.
- `input_labels`: `[B,P,N]`, int labels `1`, `0`, `-1`, `-10`.
- `input_boxes`: `[B,P,4]` when provided with points; `P` must match point batch size.
- `input_masks`: source docs say `[B,image_size,image_size]`, but model and conv path require a channel dimension for Conv2d. First DinoML admission should require `[B,1,H,W]` or insert a guarded unsqueeze only if source processor output confirms that shape.
- If no points/boxes are provided, the model creates one dummy point with label `-1`.

Postprocess:

- `processor.post_process_masks(outputs.pred_masks, original_sizes)` resizes each image's low-res masks to original image size with bilinear interpolation, optionally applies non-overlap constraints, and thresholds with `mask_threshold=0.0` by default.
- Automatic mask-generation helper paths include stability filtering, mask-to-box, RLE, and crop NMS. These are not required for first click/box segmentation parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: cached image encoder boundary

Source pattern:

```text
pixel_values -> get_image_features -> image_embeddings -> repeated prompt/mask decoder calls
```

Replacement:

```text
compile/cache image encoder artifact separately from prompt+mask decoder artifact
```

Preconditions:

- Same `pixel_values`, same model weights, same dtype/device.
- `image_embeddings` list has exactly three NCHW tensors matching `[B,32,256,256]`, `[B,64,128,128]`, `[B,256,64,64]` after `get_image_features`/projection.

Failure cases:

- Public video tracker needs frame/session ownership around the feature cache.
- Shape mismatch or un-audited backbone variant.

Parity test sketch: compare full `forward(pixel_values, prompts)` against `get_image_embeddings(pixel_values)` then `forward(image_embeddings=..., prompts)`.

### Rewrite: fixed all-valid FPN positional encodings -> constants

Source pattern:

```text
EdgeTamSinePositionEmbedding(shape, device, dtype, mask=None)
```

Replacement:

```text
precomputed constant per FPN level/dtype/device
```

Preconditions:

- `mask is None`.
- Static input size and FPN sizes match config.
- No processor padding that would make invalid pixels.

Failure cases:

- Dynamic image size, explicit mask, or video path with different feature sizes.

Parity test sketch: source function output vs loaded constant for all three FPN shapes.

### Rewrite: channels-first LayerNorm as NHWC LayerNorm island

Source pattern:

```text
NCHW -> permute NHWC -> LayerNorm(C) -> permute NCHW
```

Replacement:

```text
layout-aware channels-first LayerNorm kernel over channel dimension
```

Preconditions:

- Normalized shape equals channel count.
- Input is dense contiguous or the kernel handles strides explicitly.
- No broader layout pass changes consumer expectations.

Failure cases:

- Treating it as last-dim LayerNorm on NCHW without an axis rewrite.

Parity test sketch: random NCHW tensors for mask embedding and upscaling LN, fp32/fp16 tolerances.

### Rewrite: FPN 1x1 Conv2d -> per-pixel Linear

Source pattern:

```text
Conv2d(Cin -> 256, kernel=1, stride=1, padding=0)
```

Replacement:

```text
NHWC pixel matrix [B*H*W,Cin] @ W.T + b -> [B,H,W,256]
```

Preconditions:

- Kernel `1x1`, stride `1`, padding `0`, dilation `1`, groups `1`.
- Weight flatten keeps PyTorch Conv2d `[out,in,1,1]` order.
- Local region layout is fully controlled; downstream FPN addition/upsample axes are rewritten if staying NHWC.

Failure cases:

- RepVIT backbone outputs or FPN consumers still require NCHW and the layout pass is incomplete.

Parity test sketch: per-level conv outputs before top-down additions.

### Rewrite: hypernetwork mask projection -> batched GEMM

Source pattern:

```text
hyper_in [B,P,mask_tokens,32] @ upscaled [B,P,32,H*W]
```

Replacement:

```text
Batched GEMM over B*P: [M=mask_tokens,K=32] x [K=32,N=H*W]
```

Preconditions:

- Upscaled embedding is `[B,P,32,256*256]`.
- Mask token count is `4` before slicing.
- Output reshaped to `[B,P,4,256,256]`.

Failure cases:

- Dynamic multimask selection must happen after the full 4-token projection when `multimask_output=False` with stability enabled.

Parity test sketch: compare raw all-mask logits before any slicing/stability fallback.

### Layout guard: no broad NHWC translation across prompt/mask ABI

Protected regions:

- Processor/model input `pixel_values` is NCHW.
- `input_masks` for Conv2d mask embedding is NCHW.
- `pred_masks`/postprocess masks use `[B,P,num_masks,H,W]` with spatial axes last.
- `attention_similarity`, if used, must align with `[B*P, heads, query_len, key_len]` semantics.

Safe layout opportunity:

- Local Conv2d/LayerNorm/GELU islands can be channels-last internally only when all axis-sensitive ops and consumers are rewritten together.

## 10. Kernel fusion candidates

Highest priority:

- Prompt+mask decoder dense attention with additive-bias support: the two-way transformer is small but called repeatedly per prompt/click.
- Channels-first LayerNorm + GELU around mask/downscale/upscale paths: current source permutes for LayerNorm.
- Hypernetwork mask projection as batched GEMM: large `4 x 65536` projection per object/prompt.
- Image feature cache boundary: avoids recomputing RepViT/FPN for iterative prompting.

Medium priority:

- FPN 1x1 conv + top-down nearest upsample + add for static feature levels.
- Prompt coordinate PE and label embedding stitch as a bounded indexed/where kernel.
- Dynamic multimask stability: flatten, threshold counts, argmax/gather/where in one small selection kernel.

Lower priority:

- Postprocess resize/threshold/non-overlap on GPU. Useful for throughput, but CPU postprocess can validate first.
- Full video memory attention/spatial perceiver fusion. Important for tracker parity, but gated on the separate video-state ABI.

## 11. Runtime staging plan

Stage 1: config and wrapper admission.

- Parse `edgetam` image config.
- Reject `model_type="edgetam_video"` or route to a video follow-up until state support lands.
- Require an allowlisted RepViT/TimmWrapper audit before compiling the backbone.

Stage 2: prompt+mask decoder with supplied image embeddings.

- Accept precomputed `[B,32,256,256]`, `[B,64,128,128]`, `[B,256,64,64]`.
- Implement prompt encoder, two-way transformer, mask decoder, and low-res outputs.
- Stub image backbone and postprocess.

Stage 3: image encoder/FPN parity.

- Compose the audited RepViT backbone.
- Add FPN neck, sine positional constants, image feature cache.

Stage 4: postprocess parity.

- Resize logits to original sizes, threshold, optional non-overlap.
- Keep automatic mask generation/crop NMS deferred.

Stage 5: optimized lowering.

- Add layout-guarded LayerNorm/GEMM rewrites, attention provider selection, hypernetwork GEMM, feature-cache APIs.

Stage 6: video tracker follow-up.

- Audit and implement `edgetam_video` inference session, memory bank, object pointers, memory encoder, spatial perceiver, forward/reverse propagation.

## 12. Parity and validation plan

- Prompt PE unit parity: fixed Gaussian buffer, random points/boxes, labels including `1`, `0`, `-1`, `-10`.
- Mask embedding parity: random `[B,1,256,256]` masks through three convs and two channels-first LayerNorms.
- Two-way attention parity: one block with small query/key lengths; include additive `attention_similarity`.
- Mask decoder parity with supplied random image embeddings and high-res features; compare all four raw masks before slicing.
- Dynamic multimask parity: construct stable and unstable token-0 logits and verify argmax/gather fallback.
- Single-image end-to-end parity after backbone admission: `pixel_values + point prompt -> low_res_masks/iou/object_score`.
- Postprocess parity: original sizes with non-square images, threshold on/off, non-overlap on/off.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 relaxed around attention/upsample `rtol=1e-2, atol=1e-2`.

No validation was run during this audit, per request.

## 13. Performance probes

- Processor throughput: image resize/normalize and point/box packing separately.
- Image encoder/FPN throughput for `B=1,2,4` at `1024x1024`.
- Prompt+mask decoder latency with cached image embeddings, sweeping point batch `P` and prompt count `N`.
- Attention backend comparison for sparse self-attn and token-to-image cross-attn, with/without additive bias.
- Hypernetwork mask projection throughput as batched GEMM vs naive matmul.
- Postprocess resize/threshold throughput for original image sizes.
- Video follow-up probes: memory-bank token count, object count, frame count, forward/reverse propagation, memory encoder update cost.

## 14. Skip/defer list

- Training, gradients, checkpoint conversion, and sanity-check scripts.
- Full RepViT/TimmWrapper implementation until separately audited.
- Public `EdgeTamVideoModel` session/state parity.
- Automatic mask-generation crop grid, RLE, stability filtering, and crop NMS for first click/box segmentation target.
- FlashAttention for target-guided additive-mask attention until additive-bias support is available or guarded fallback is kept.
- Quantization and packed weight formats; none are source-coupled in the inspected plain EdgeTAM wrapper.

## 15. Final implementation checklist

- [ ] Parse plain `EdgeTamConfig`, `EdgeTamVisionConfig`, `EdgeTamPromptEncoderConfig`, and `EdgeTamMaskDecoderConfig`.
- [ ] Add admission gate for `model_type="edgetam_video"` pending video-state audit.
- [ ] Add allowlist/delegation plan for RepViT `TimmWrapperConfig`.
- [ ] Load prompt encoder and mask decoder weights with tied/shared positional buffer handling.
- [ ] Implement random Fourier prompt positional embedding.
- [ ] Implement point/box sparse prompt embedding with labels `1`, `0`, `-1`, `-10`.
- [ ] Implement dense mask prompt embedding conv stack.
- [ ] Implement noncausal MHA/SDPA with additive float attention bias.
- [ ] Implement two-way transformer block and final token-to-image attention.
- [ ] Implement mask decoder deconv upscaling, high-res skip additions, hypernetwork batched GEMM, IoU/object heads.
- [ ] Implement dynamic multimask stability selection.
- [ ] Implement cached image-embedding decoder path first.
- [ ] Audit/compose RepViT backbone and FPN neck.
- [ ] Add layout guards for NCHW/NHWC boundaries.
- [ ] Add postprocess mask resize/threshold/non-overlap parity.
- [ ] Add parity tests for prompt encoder, mask decoder, cached image path, and postprocess.
- [ ] Benchmark image encoder, cached prompt decoder, attention backend, and mask projection.
