# Transformers Audit: `sam3_tracker`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: primary in-library family `sam3_tracker`; open config mirror `onnx-community/sam3-tracker-ONNX`; gated upstream `facebook/sam3`
Config source: local source defaults plus open ONNX mirror config; direct `facebook/sam3` config is gated
Source files inspected:
  X:/H/transformers/src/transformers/models/sam3_tracker/configuration_sam3_tracker.py
  X:/H/transformers/src/transformers/models/sam3_tracker/modeling_sam3_tracker.py
  X:/H/transformers/src/transformers/models/sam3_tracker/modular_sam3_tracker.py
  X:/H/transformers/src/transformers/models/sam3_tracker/processing_sam3_tracker.py
  X:/H/transformers/src/transformers/models/sam3/configuration_sam3.py
  X:/H/transformers/src/transformers/models/sam3/modeling_sam3.py
  X:/H/transformers/src/transformers/models/sam2/image_processing_sam2.py
  X:/H/transformers/src/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py
  X:/H/transformers/src/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py
Any missing files or assumptions:
  `facebook/sam3` and `danelcsb/sam3_tracker.1_hiera_tiny` configs were gated/401.
  `modeling_sam3_tracker.py`, `configuration_sam3_tracker.py`, and `processing_sam3_tracker.py` are generated; `modular_sam3_tracker.py` is authoritative for upstream source edits.
```

Primary DinoML target: image/prompt segmentation for the tracker head: NCHW image input or cached image embeddings, point/box/mask prompts, low-resolution masks, IoU scores, and object-presence logits. Persistent video tracking state is implemented in `sam3_tracker_video`; this report documents that ABI as a boundary because SAM3 tracker weights/configs may be exported with video fields, but the first `sam3_tracker` integration should reject or route those fields to a separate video audit.

## 2. High-level architecture

Dataflow:

```text
CPU image processor + prompt normalization
-> SAM3 ViT vision encoder + FPN neck
-> prompt encoder for points/boxes/masks
-> two-way prompt/image transformer mask decoder
-> low-res masks + IoU scores + object-score logits
-> processor mask resize / threshold / optional non-overlap postprocess
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize to NCHW `pixel_values`, `original_sizes`, coordinate normalization to target image size, padding for point prompts, and validation that box counts are uniform when batched.
- Cacheable vision stage: `get_image_features()` and `get_image_embeddings()` allow image features to be computed once and reused across prompt clicks. The cached ABI is three FPN levels flattened as `(H*W, B, C)` internally, then reshaped back to `(B, C, H, W)` for mask decoding.
- Prompt stage: sparse point/box embeddings and dense mask/no-mask embeddings can be recomputed per prompt without rerunning the image encoder.
- Decoder stage: small two-way noncausal attention over prompt tokens and image tokens, transposed-conv upsampling, hypernetwork mask projection, IoU head, and object-score head.
- Video tracking boundary: `sam3_tracker_video` adds session objects, per-object prompt histories, memory features, object pointers, and memory attention. This is required for end-to-end video tracking, but not for the tracker-only image segmentation graph.

## 3. Important config dimensions

Representative concrete values below come from the open ONNX mirror unless labeled as source default.

| Field | Value | Source / runtime impact |
|---|---:|---|
| `model_type` | `sam3_tracker` | Open mirror config |
| image processor size | 1008x1008 | Open mirror preprocessor; NCHW output |
| prompt `hidden_size` | 256 | Source default and mirror |
| prompt `patch_size` | 14 | Prompt grid 72x72 for 1008 input |
| prompt `mask_input_size` | 288x288 | Computed as `4 * image_size / patch_size` |
| mask decoder hidden | 256 | Two-way transformer width |
| mask decoder layers | 2 | Source default and mirror |
| mask decoder heads | 8 | Internal attention heads |
| attention downsample | 2 | Cross-attn internal dim 128, head dim 16 |
| mask tokens | 4 | 1 single-mask token + 3 multimask tokens |
| IoU head | depth 3, hidden 256, sigmoid output | Source default and mirror |
| dynamic multimask | enabled, delta 0.05, threshold 0.98 | Single-mask inference can gather best multimask |
| vision `model_type` | `sam3_vision_model` | Composed AutoModel backbone |
| ViT hidden/layers/heads | 1024 / 32 / 16 | Open mirror |
| ViT patch/window | patch 14, window 24 | Windowed attention except global layers |
| ViT global layers | `[7, 15, 23, 31]` | Full-image attention layers |
| FPN hidden/features | 256; 288x288, 144x144, 72x72 | Mask decoder consumes 3 levels |
| video memory fields | `num_maskmem=7`, memory dim 64, max pointers 16 | Mirror has these, but local `Sam3TrackerConfig` does not own them |

Checkpoint/config sweep:

| Checkpoint/config | Access | Operator-significant notes |
|---|---|---|
| `onnx-community/sam3-tracker-ONNX` | Open | `sam3_tracker`; concrete 1008 ViT-L-like backbone, ONNX split into `vision_encoder` and `prompt_encoder_mask_decoder`; includes video memory fields that are not part of strict local `Sam3TrackerConfig`. |
| `facebook/sam3` | Gated raw config; API metadata readable | API reports `model_type: sam3_video`, `architectures: ["Sam3VideoModel"]`; should route to `sam3_tracker_video` for true video/session state. |
| `danelcsb/sam3_tracker.1_hiera_tiny` | 401 during audit | Mentioned in in-source examples, but config could not be verified. Treat as source-doc example only until access is available. |

## 3a. Family variation traps

- `sam3_tracker` composes `sam3_vision_model`; DinoML should not treat the tracker directory as self-contained operator coverage.
- The open ONNX mirror has tracker-video memory fields under a `sam3_tracker` config. The local strict config class does not define those fields, while `sam3_tracker_video` does. Admission should reject unknown video-state fields for tracker-only compile or route to video.
- Source accepts either `pixel_values` or `image_embeddings`, exactly one. Cached image embeddings are a first-class ABI, not an optimization-only detail.
- Point prompts can be padded with label `-10`; label `-1` is not-a-point, labels `0/1` receive learned negative/positive point embeddings, box corners use learned embeddings 2 and 3.
- Boxes are not padded by the processor; batched images must have the same number of boxes.
- The mask decoder has optional `attention_similarity` additive bias and `target_embedding`; Flash Attention is explicitly avoided for target-guided additive masks.
- `multimask_output=False` is dynamic in eval mode: it may choose token 0 or gather the best of tokens 1..3 based on stability. This creates `sum`, comparisons, `argmax`, `gather`, and `where` in the output selection path.
- Vision code alternates NCHW maps, NHWC/token views, and `(HW, B, C)` FPN cache tensors. Layout passes need explicit guards around each boundary.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW image tensor input `[B, 3, 1008, 1008]`.
- Conv patch embedding `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- Reshape/view/flatten/permute between `[B, C, H, W]`, `[B, H, W, C]`, `[B, H*W, C]`, and `[H*W, B, C]`.
- Repeat/repeat_interleave for batch and prompt expansion.
- Cat/stack over token, prompt, and mask-token dimensions.
- Gather/argmax/where for dynamic multimask selection.
- Interpolate bilinear with `align_corners=False`, sometimes `antialias=True`, for mask prompt resizing, high-res mask generation in video boundary, and postprocess mask resizing.

Neural primitives:
- Conv2d, ConvTranspose2d, depthwise Conv2d in the video memory fuser boundary.
- Linear/GEMM with bias throughout attention, MLPs, hypernetworks, IoU/object heads.
- LayerNorm over channels-last token tensors and explicit channels-first wrapper that permutes NCHW -> NHWC -> NCHW.
- GELU and ReLU source activations; sigmoid for IoU head and memory-mask probabilities in video boundary.
- GroupNorm appears in SAM3 detector/mask decoder source, not in the tracker-only decoder path.

Attention primitives:
- SAM3 ViT self-attention: noncausal MHA with 2D RoPE; mostly windowed 24x24 patch windows, global at configured layers.
- Tracker mask decoder attention: noncausal two-way attention over prompt tokens and dense image tokens; MHA with internal downsampled dims for cross-attn.
- Optional additive `attention_similarity` bias for target-guided prompt-to-image attention.
- Video boundary memory attention: cross-attention from current frame features to concatenated spatial memories and object-pointer tokens, with 2D RoPE excluding the object-pointer prefix.

Position/custom math:
- Random Fourier positional embeddings for prompt points/boxes and image-wide decoder embeddings.
- SAM3 ViT absolute position interpolation plus 2D RoPE/window partitioning.
- Video boundary uses 1D sine temporal encodings for object pointers and learned temporal memory embeddings.

Pre/postprocessing-coupled ops:
- Coordinate normalization from original `(H, W)` to target 1008, preserving point pad value `-10`.
- Mask postprocess: per-image bilinear upsample to original size, optional non-overlap `argmax(dim=0)`, optional threshold to bool.
- Box postprocess is not a tracker-head model output; boxes are prompt inputs. Video memory uses object-presence logits, not NMS.

## 5. Layer/block breakdown

Vision encoder, composed `sam3_vision_model`:

```text
pixel_values [B,3,H,W] NCHW
-> Conv2d patch projection k=s=14 -> [B,1024,H/14,W/14]
-> flatten/add position embedding -> [B,H/14*W/14,1024]
-> 32 ViT layers:
     x = LayerNorm(x)
     x = window_partition(x, 24) unless layer in [7,15,23,31]
     q,k,v = Linear(1024 -> 1024) each, q/k get 2D RoPE
     x = attention + residual
     x = LayerNorm + MLP(1024 -> 4736 -> 1024) + residual
-> FPN neck -> three NCHW maps [B,256,288,288], [B,256,144,144], [B,256,72,72]
```

Tracker prompt encoder:

```text
points [B,P,N,2], labels [B,P,N]
-> +0.5 center shift -> random Fourier PE(2 -> 256)
-> where(label == -1, not_a_point_embed, pe)
-> where(label == -10, 0, previous)
-> + learned point_embed[label] for labels >= 0

boxes [B,P,4]
-> +0.5 -> view as two corners plus padded third corner
-> random Fourier PE -> add corner embeddings 2/3 and not_a_point on pad

masks [B,1,H,W] or no mask
-> if provided: Conv2d(1->4,k=2,s=2), channels-first LN, act,
                Conv2d(4->16,k=2,s=2), channels-first LN, act,
                Conv2d(16->256,k=1)
-> if absent: learned no_mask_embed expanded to [B,256,72,72]
```

Mask decoder:

```text
tokens = [obj_score_token, iou_token, 4 mask_tokens] + sparse prompts
image = last FPN map [B,256,72,72] + dense prompt
repeat image per point/object prompt
-> TwoWayTransformer, 2 layers:
     prompt self-attn
     prompt-to-image cross-attn, optional additive similarity
     prompt MLP(256 -> 2048 -> 256)
     image-to-prompt cross-attn
   final prompt-to-image attention + LN
-> image tokens -> [B*P,256,72,72]
-> ConvTranspose2d(256->64,k=2,s=2) + high-res 144x144 feature
-> channels-first LN + GELU
-> ConvTranspose2d(64->32,k=2,s=2) + high-res 288x288 feature + GELU
-> per-mask-token hypernetwork MLP(256 -> 256 -> 32)
-> batched hyper_in @ upscaled_embedding -> masks [B,P,4,288,288]
-> IoU MLP(256 -> 256 -> 4) with sigmoid
-> object_score MLP(256 -> 256 -> 1)
-> output 3 masks if multimask, else token 0 or dynamic stable/best fallback
```

## 6. Attention requirements

Tracker-only attention is noncausal and has no autoregressive KV cache.

Mask decoder:
- Self-attention over sparse prompt/output tokens.
- Cross-attention token-to-image and image-to-token.
- Head count 8, hidden 256. Cross-attn downsample rate 2 gives internal dim 128 and head dim 16; self-attn uses downsample 1 and head dim 32.
- Query shape is effectively `[B, point_batch, Q, 256]`; source flattens `B * point_batch` for attention and returns the point-batch dimension.
- Attention masks are optional additive bias tensors via `attention_similarity`; source falls back from Flash Attention to SDPA when this bias is present.
- No causal masks, no KV cache, no packed varlen ABI.

Vision attention:
- Noncausal ViT MHA with windowed attention for most layers and full attention for configured global layers.
- 2D RoPE is applied to q/k before attention.
- Window partition/unpartition pads spatial patch grids to window multiples. For the 1008/14=72 grid and window 24, no padding is needed, but dynamic sizes need pad guards.

Video tracking state ABI boundary:
- `sam3_tracker_video` owns persistent state, not tracker-only `Sam3TrackerModel`.
- Session state includes object id/index maps, point and mask prompt dictionaries per object/frame, conditioning and non-conditioning frame output maps, tracked-frame metadata, a small vision-feature cache, and processed frame storage.
- Stored per-frame outputs include `pred_masks`, `high_res_masks`, `object_pointer`, `object_score_logits`, and, after memory encoding, `maskmem_features` and `maskmem_pos_enc`.
- Memory selection uses up to `max_cond_frame_num` nearest conditioning frames plus up to `num_maskmem - 1` prior or future non-conditioning frames depending on reverse tracking.
- Object pointers are stacked as tokens, optionally receive temporal sine positional encoding, and are split from hidden dim 256 into memory dim 64 chunks when `mem_dim < hidden_dim`.
- Mask memory features are cast to `bfloat16`, flattened to `(H*W, B, Cmem)`, and persisted on `inference_state_device`.

## 7. Position encoding and custom math

Prompt Fourier PE:

```python
def prompt_pe(coords, table, input_shape=None):
    # coords [..., 2], x/y in target image pixels or normalized input
    if input_shape is not None:
        coords[..., 0] = coords[..., 0] / input_shape[1]
        coords[..., 1] = coords[..., 1] / input_shape[0]
    coords = 2 * coords - 1
    angles = 2 * pi * (coords @ table)  # table [2, hidden/2]
    return cat([sin(angles), cos(angles)], dim=-1)
```

Image-wide prompt PE is a static 72x72 grid for the default 1008/14 setup and can be precomputed per dtype/device unless image size changes.

Dynamic single-mask selection:

```python
def choose_single_mask(all_logits, all_iou, delta=0.05, thresh=0.98):
    single = all_logits[:, :, 0:1]
    multi = all_logits[:, :, 1:]
    best = argmax(all_iou[:, :, 1:], dim=-1)
    best_logits = gather(multi, dim=2, index=best)
    stable = (sum(single > delta) / sum(single > -delta).clamp_min(1)) >= thresh
    return where(stable[..., None, None], single, best_logits)
```

Video boundary memory math:

```python
mask_for_mem = sigmoid_or_binary(pred_masks_high_res)
mask_for_mem = mask_for_mem * sigmoid_scale_for_mem_enc + sigmoid_bias_for_mem_enc
maskmem = memory_encoder(pix_feat, mask_for_mem).to(bfloat16).flatten(2).permute(2, 0, 1)
```

## 8. Preprocessing and input packing

Processor:
- Images are processed by SAM3/SAM2 image processor to NCHW `pixel_values`; open mirror config uses resize to 1008x1008, RGB conversion, rescale `1/255`, normalize mean/std `[0.5,0.5,0.5]`, and `data_format: channels_first`.
- `original_sizes` are retained for postprocess and may be provided without images when using cached embeddings.
- `input_points` expected format is `[image, object/point_batch, point, xy]`; padded to max dims with `point_pad_value=-10`.
- `input_labels` expected format is `[image, object/point_batch, point]`; must match point dims when both are provided.
- `input_boxes` expected format is `[image, box, xyxy]`; boxes are normalized but not padded.
- Coordinate normalization scales x by `target_width / original_width` and y by `target_height / original_height`.
- `input_masks` are graph inputs, not image-processor segmentation labels; model resizes them to 288x288 if needed before prompt mask embedding.

Postprocess:
- `post_process_masks(masks, original_sizes, mask_threshold=0.0, binarize=True, apply_non_overlapping_constraints=False)` loops per image, bilinear-upsamples low-res NCHW masks to original `(H,W)`, optionally applies non-overlap suppression, then optionally thresholds to bool.
- Non-overlap suppression computes `argmax` over object/channel dimension 0 at each spatial location and clamps losing logits to `<= -10`.
- There is no NMS in tracker-head prompt segmentation postprocess. NMS appears in SAM2 automatic mask-generation utilities and SAM3 detector/instance segmentation, not in this first tracker target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear

Source pattern:

```text
Conv2d(3 -> 1024, kernel_size=14, stride=14, padding=0, bias=False)
```

Replacement:

```text
WindowFlatten[NCHW, 14x14 non-overlap] -> GEMM(588 -> 1024) -> reshape [B, Hp, Wp, 1024]
```

Preconditions:
- Kernel equals stride, dilation 1, groups 1, padding 0.
- Input H/W divisible by 14 or source fallback with exact PyTorch Conv2d semantics.
- Flatten order must match PyTorch NCHW convolution weight order: `out, in, kh, kw`.
- Layout pass must either keep NCHW flatten order or explicitly transform NHWC patch flatten and weights.

Failure cases:
- Non-divisible input shape, changed patch size, nonzero padding, or grouped conv.

Parity test sketch:
- Compare Conv2d output and rewrite output for random fp32/fp16 images at 1008 and one rejected non-divisible size.

### Rewrite: channels-first LayerNorm wrapper

Source pattern:

```text
NCHW -> permute NHWC -> LayerNorm(C) -> permute NCHW
```

Replacement:

```text
ChannelsFirstLayerNorm(NCHW, normalized_dim=C)
```

Preconditions:
- Only normalize over channel axis.
- Weight/bias shape equals C.
- Consumers remain NCHW.

Failure cases:
- Any intervening op between permutes and LayerNorm, or downstream consumer expects NHWC.

### Rewrite: cached image encoder split

Source pattern:

```text
pixel_values -> get_image_features -> projected FPN cache
cached image_embeddings + prompts -> prompt_encoder + mask_decoder
```

Replacement:

```text
Artifact A: vision_encoder(pixel_values) -> image_embedding cache
Artifact B: prompt_encoder_mask_decoder(image_embeddings, prompts) -> masks/scores
```

Preconditions:
- Same weights/dtype/layout for both artifacts.
- Cache includes all three FPN levels with exact expected orientation and feature sizes.
- `no_memory_embedding` addition to last map is preserved for tracker-only image mode.

Failure cases:
- Video memory-conditioned features, target-specific layout translations, or missing high-resolution FPN levels.

### Rewrite: dynamic single-mask selection as small indexed select

Source pattern:

```text
argmax(iou[:, :, 1:]) -> gather(multimask logits) -> stability where
```

Replacement:

```text
Small per-object output-select kernel
```

Preconditions:
- `num_multimask_outputs == 3`.
- Mask logits shape `[B, P, 4, H, W]`.
- Eval mode only; training path can be rejected.

Failure cases:
- Different mask token count or required attention outputs.

## 10. Kernel fusion candidates

Highest priority:
- SAM3 ViT window attention with 2D RoPE: dominates vision encoder cost at 72x72 tokens; window partition/unpartition should be fused or layout-aware.
- LayerNorm + QKV linear + RoPE + attention for ViT blocks: avoids repeated reshapes and improves memory locality.
- Prompt decoder two-way attention: small but repeated for interactive clicks; important once image embeddings are cached.
- ConvTranspose upsample + high-res FPN add + GELU: direct mask quality path, stable shape 72->144->288.

Medium priority:
- Hypernetwork mask projection: `hyper_in @ upscaled_embedding` is a batched small GEMM over `[B,P,T,32] x [B,P,32,H*W]`.
- Channels-first LayerNorm kernel for prompt mask embedding and upscaler.
- Postprocess mask resize + threshold + non-overlap suppression for end-to-end latency.
- Video boundary memory encoder and memory attention, only if `sam3_tracker_video` is admitted.

Lower priority:
- Prompt Fourier PE for points/boxes; small and often CPU/host-side.
- Dynamic multimask stability selection; small output-side kernel.
- ONNX/export compatibility branches such as explicit `torch.where` expansion.

## 11. Runtime staging plan

Stage 1: parse strict `Sam3TrackerConfig`, load weights, and reject/route configs with video-only state fields unless targeting `sam3_tracker_video`.

Stage 2: implement vision-only parity for `sam3_vision_model` and FPN outputs at `[B,3,1008,1008]`.

Stage 3: implement prompt encoder parity for points, labels, boxes, and no-mask dense embeddings. Defer mask prompts initially if needed.

Stage 4: implement mask decoder parity with cached image embeddings, default `multimask_output=True`, no `attention_similarity`, no `target_embedding`.

Stage 5: add `multimask_output=False` dynamic stability/gather path and mask prompt embedding/resizing.

Stage 6: add processor postprocess parity for mask resize/threshold and optional non-overlap.

Stage 7: split artifacts into vision encoder and prompt+decoder cache workflow.

Stage 8: separately audit/implement `sam3_tracker_video` session ABI for object tracking memory, reverse propagation, and streamed frames.

## 12. Parity and validation plan

- Config parse tests: source-default config and open mirror config; verify tracker-only admission rejects video fields or routes them.
- Prompt PE tests: random points/boxes with pad labels `-10`, `-1`, `0`, `1`; compare embeddings to PyTorch.
- Mask embedding tests: random masks at native 288x288 and non-native sizes requiring bilinear antialias resize.
- One two-way attention block parity: fp32 tolerance `1e-5`, fp16/bf16 tolerance around `2e-2` for attention paths.
- Full prompt+mask decoder parity from cached image embeddings with fixed random FPN maps.
- Vision encoder/FPN parity at 1008x1008 and one smaller/static debug shape if a representative config is accessible.
- End-to-end image segmentation parity with one point and one box prompt, comparing low-res logits before postprocess.
- Postprocess parity for original sizes, thresholding, and non-overlap suppression.
- Video boundary later: session update/order tests for point input, mask input, object pointer persistence, memory feature dtype/layout, forward/reverse propagation.

No DinoML tests or imports were run for this audit.

## 13. Performance probes

- Image processor throughput: resize/rescale/normalize to NCHW 1008.
- Vision encoder latency and memory by batch size: 1, 2, 4; separate windowed and global attention layers.
- Cached prompt latency: point-only, box-only, mask prompt, and combined point+box.
- Mask decoder point-batch sweep: P = 1, 4, 16 objects/prompts.
- Postprocess sweep over original output sizes and number of masks/objects.
- Layout pass probe: NCHW baseline versus guarded NHWC/channel-last for local Conv/LayerNorm regions.
- Video boundary later: memory bank size sweep (`num_maskmem`), object pointer count sweep, CPU/GPU state-device transfers, and streaming cache hit rate.

## 14. Skip/defer list

- Training, losses, gradient checkpointing behavior.
- SAM3 detector/text prompt branch and instance/object detection postprocess.
- `sam3_tracker_video` persistent memory implementation in the first tracker-only integration.
- `attention_similarity`/PerSAM target-guided additive attention until default prompt segmentation works.
- `target_embedding` semantic prompting.
- Automatic mask generation crop/NMS utilities.
- Quantized ONNX variants from the open mirror; treat as deployment artifacts, not source behavior.
- Full dynamic input resolutions beyond source-configured static 1008 unless explicit shape guards and FPN feature sizes are supplied.

## 15. Final implementation checklist

- [ ] Parse strict `Sam3TrackerConfig` and composed `Sam3VisionConfig`.
- [ ] Add admission rule for video-only memory fields in `sam3_tracker` configs.
- [ ] Load tracker, vision, prompt encoder, and mask decoder weights with alias checks.
- [ ] Implement SAM3 ViT patch embedding, window/global attention, 2D RoPE, and FPN neck.
- [ ] Implement prompt Fourier PE for points/boxes with label semantics.
- [ ] Implement prompt mask embedding and no-mask embedding expansion.
- [ ] Implement two-way mask decoder attention and MLP blocks.
- [ ] Implement ConvTranspose upscaler plus high-resolution FPN additions.
- [ ] Implement hypernetwork mask projection and IoU/object-score heads.
- [ ] Implement dynamic multimask stability selection.
- [ ] Implement mask postprocess resize/threshold/non-overlap.
- [ ] Add cached image embedding artifact split.
- [ ] Add layout guards around NCHW/NHWC and `(HW,B,C)` cache boundaries.
- [ ] Add parity tests for prompt-only, decoder-only, vision-only, and end-to-end image prompt segmentation.
- [ ] Open a separate `sam3_tracker_video` task for session/state ABI and memory attention.
