# Transformers Audit: sam3_video

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/sam3 for official docs/checkpoint references; official checkpoint is gated.
Config source: source defaults plus open mirror config saved at _sources/bodhicitta_sam3_config_mirror.json.
Source files inspected:
- transformers/src/transformers/models/sam3_video/configuration_sam3_video.py
- transformers/src/transformers/models/sam3_video/modeling_sam3_video.py
- transformers/src/transformers/models/sam3_video/processing_sam3_video.py
- transformers/src/transformers/models/sam3/configuration_sam3.py
- transformers/src/transformers/models/sam3/modeling_sam3.py
- transformers/src/transformers/models/sam3/image_processing_sam3.py
- transformers/src/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py
- transformers/src/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py
Any missing files or assumptions: official facebook/sam3 config/weights are gated; mirror config is not treated as authoritative for licensing or release metadata. No tests/imports were run.
```

Primary runtime target: text-prompted video segmentation/tracking with persistent object identities. This is not an autoregressive generation model; cache/state means video/frame memory, not KV cache.

## 2. High-level architecture

`sam3_video` is a stateful orchestration wrapper around two neural bodies:

```text
video decode/sampling + image/video processor
  -> per-frame NCHW pixel tensor [3,H,W], text prompt ids/mask
  -> SAM3 detector: shared vision encoder + CLIP text encoder + DETR detector + mask decoder
  -> detection/tracker association heuristics
  -> SAM3 tracker: memory-conditioned mask decoder + memory encoder
  -> per-object low-res masks and object ids
  -> processor postprocess: upsample, threshold, boxes, prompt grouping
```

Stage decomposition:

- CPU/data pipeline: video decode, frame sampling, image resize/rescale/normalize, CLIP tokenization, original-size bookkeeping.
- Independently cacheable encoders: text prompt embeddings per prompt id; per-frame detector vision embeddings; tracker FPN feature maps and position embeddings.
- Stateful tracker runtime: per-object conditioning/non-conditioning outputs, memory features, object pointers, occlusion/hotstart bookkeeping.
- Postprocess: low-res mask upsample to original video size, threshold, zero-area/suppressed-id filtering, mask-to-box conversion, prompt-group non-overlap.

## 3. Important config dimensions

Representative dimensions from source defaults and the open mirror config:

| Component | Field | Value |
|---|---:|---:|
| Video wrapper | `model_type` | `sam3_video` |
| Video wrapper | `low_res_mask_size` | 288 |
| Video wrapper | `det_nms_thresh` / `score_threshold_detection` | 0.1 / 0.5 |
| Video wrapper | `hotstart_delay` / `hotstart_unmatch_thresh` / `hotstart_dup_thresh` | 15 / 8 / 8 |
| Detector ViT | `image_size`, `patch_size` | 1008, 14 |
| Detector ViT | patch grid | 72 x 72 |
| Detector ViT | `hidden_size`, layers, heads | 1024, 32, 16 |
| Detector ViT | local/global attention | window 24; global layers 7, 15, 23, 31 |
| Detector text | CLIP hidden/layers/heads | 1024 / 24 / 16 |
| Detector text | max text length / vocab | 32 / 49408 |
| Detector DETR | hidden/layers/heads | 256 / 6 encoder + 6 decoder / 8 |
| Detector DETR | queries | 200 plus presence token |
| Tracker | `num_maskmem` | 7 |
| Tracker | `max_cond_frame_num` | 4 |
| Tracker | `max_object_pointers_in_encoder` | 16 |
| Tracker memory attention | hidden/layers/heads | 256 / 4 / 1 |
| Tracker memory encoder | output channels | 64 |
| Tracker memory RoPE grid | 72 x 72 |

Representative checkpoint sweep:

| Source | Availability | Operator-significant notes |
|---|---|---|
| `facebook/sam3` | gated official repo | Official docs use this for `Sam3VideoModel`; config access requires approval. |
| `bodhicitta/sam3` | open mirror snapshot | `Sam3VideoModel`, 1008 input, ViT-H-like 32-layer detector, tracker memory enabled. |
| source default `Sam3VideoConfig()` | local source | Same composition defaults; `recondition_on_trk_masks=True` in source defaults, mirror config sets it `false`. |
| custom-resolution example | official docs/source behavior | `config.image_size=560` propagates into detector/tracker image size and feature grids; accuracy warning in docs. |
| `facebook/sam3.1` | repo metadata only | Search result states no Transformers integration; out of scope for this `sam3_video` source basis. |

## 3a. Family variation traps

- `sam3_video` composes `sam3` and `sam3_tracker_video`; DinoML should not audit or admit it as one flat transformer block.
- The official checkpoint is gated. Mirror configs can show dimensions but must not be the only admission source for production.
- `recondition_on_trk_masks` differs between source default and mirror config; it changes whether tracker or detector masks feed reconditioning memory.
- `image_size` changes feature sizes and memory RoPE sizes. The setter rewrites detector and tracker image size; layout/shape guards must bind all derived grids together.
- The detector vision body alternates local window attention and global attention. Window partition uses NHWC token maps internally even though public pixel ABI is NCHW.
- Text prompt caching is session state. Prompt additions affect future detections without changing model weights.
- Tracker memories are not KV caches. They are per-object, per-frame mask memory tensors plus object pointer tokens with temporal positional encodings.
- Streaming mode disables future-frame hotstart heuristics; preloaded-video parity and streaming parity have different object removal behavior.
- Postprocess depends on `torchvision.ops.masks_to_boxes`; model outputs alone are not end-to-end parity.
- Optional `kernels` package accelerates connected components for hole filling/sprinkle removal. Source falls back by skipping those quality steps when unavailable.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image resize/rescale/normalize in processor; model consumes `[B,3,H,W]`.
- Conv2d patch embedding `3 -> 1024`, kernel=stride=14, no bias.
- Token reshape/flatten/transpose: NCHW -> `[B,HW,C]`, `[B,H,W,C]`, FPN `[B,C,H,W]`, tracker `[HW,B,C]`.
- Window partition/unpartition with padding for local attention.
- Interpolate bilinear/nearest, ConvTranspose2d upsampling, padding, cat, stack, clone, nonzero, index_select, masked boolean indexing.
- `torch.where`, clamp, sigmoid, threshold comparisons, reductions over H/W and object axes.

Neural primitives:

- ViT MHA with 2D axial RoPE, MLP GELU, LayerNorm.
- CLIP text encoder, 24 layers, hidden 1024, max length 32; projected to 256 per token.
- DETR encoder/decoder MHA and cross-attention, hidden 256, FFN 2048 ReLU.
- Dot-product text/query scoring: query projection x pooled/projected text.
- Pixel decoder/FPN convolutions, transposed convolutions, mask embeddings, `einsum("bqc,bchw->bqhw")`.
- Tracker two-way transformer attention, memory attention, memory fuser depthwise Conv2d plus pointwise Linear in NHWC.
- Memory encoder mask downsampler Conv2d stack and 1x1 projection.

State/cache ops:

- Frame store: `processed_frames[frame_idx] -> [3,H,W]`.
- Vision cache: `frame_idx -> {"vision_feats": list[HW,B,256], "vision_pos_embeds": list[HW,B,256]}` with oldest-frame eviction.
- Per-object outputs: `cond_frame_outputs` and `non_cond_frame_outputs` keyed by frame.
- Memory ABI: `maskmem_features` flattened `[Hmem*Wmem, 1, 64]` in bfloat16; `maskmem_pos_enc` same leading shape.
- Object pointer ABI: pointer tokens originally `[1,256]`, split/repacked to 64-wide memory tokens when appended to memory sequence.

Postprocessing-coupled ops:

- Mask IoU flattening and pairwise reductions.
- NMS over masks, not boxes.
- Connected components with optional external kernel for holes/sprinkles.
- Bilinear upsample from low-res logits to original frame size, threshold `> 0`, mask-to-box.

## 5. Layer/block breakdown

Detector frame path:

```text
pixel_values [B,3,1008,1008]
  -> Conv2d patch embed -> tokens [B,72*72,1024]
  -> ViT x32: LayerNorm -> local/global RoPE attention -> residual -> LayerNorm -> GELU MLP -> residual
  -> reshape to [B,1024,72,72]
  -> vision neck/FPN -> multi-level [B,256,288,288], [B,256,144,144], [B,256,72,72], ...
```

Text prompt path:

```text
CLIP input_ids/attention_mask [1,32]
  -> CLIP text encoder hidden [1,32,1024]
  -> Linear 1024 -> 256 for every token
  -> cached in inference_session.prompt_embeddings[prompt_id]
```

Detection head:

```text
FPN lowest level + prompt text/geometry
  -> DETR encoder cross-fuses vision and prompt features
  -> DETR decoder learned queries + presence token
  -> box offsets + inverse_sigmoid(reference_boxes) -> sigmoid -> cxcywh -> xyxy
  -> dot-product query/text logits and presence logits
  -> mask decoder -> pred_masks [B,200,288,288]
```

Tracker per-object frame path:

```text
current tracker vision features [HW,B,256]
  -> if initial conditioning: add no_memory_embedding and reshape [B,256,72,72]
  -> else gather cond/non-cond mask memories + object pointer tokens
  -> memory attention with 2D RoPE, excluding pointer tokens from key RoPE
  -> SAM mask decoder -> low/high-res masks, object_score_logits, object_pointer
  -> memory encoder writes maskmem_features/maskmem_pos_enc for future frames
```

## 6. Attention requirements

Required attention variants:

- Detector ViT self-attention: noncausal, MHA, 16 heads, head dim 64, 2D axial RoPE. Most layers use local 24x24 windows; layers 7/15/23/31 use full grid attention.
- CLIP text self-attention: noncausal text encoder attention; max length 32.
- Detector DETR encoder: vision self-attention plus prompt/text cross-attention, hidden 256, 8 heads.
- Detector DETR decoder: learned query self-attention, text cross-attention, vision cross-attention with relative-position bias matrix for query boxes.
- Tracker mask decoder/two-way transformer: sparse prompt/output tokens attend to dense image embeddings and reverse cross-attention from image tokens to sparse tokens.
- Tracker memory attention: current frame tokens attend over concatenated mask memories and object pointer tokens. It uses 2D RoPE on image/memory tokens and explicitly excludes appended object pointer tokens from key RoPE.

No autoregressive KV cache is required. The state ABI is:

```text
session
  processed_frames: frame_idx -> [3,H,W]
  prompt_input_ids / prompt_attention_masks / prompt_embeddings: prompt_id -> tensors
  output_dict_per_obj[obj_idx]["cond_frame_outputs"][frame_idx]
  output_dict_per_obj[obj_idx]["non_cond_frame_outputs"][frame_idx]
    pred_masks, high_res_masks, object_pointer, object_score_logits,
    maskmem_features [Hmem*Wmem,1,64] bf16,
    maskmem_pos_enc [Hmem*Wmem,1,64]
  object lifecycle metadata: scores, prompt ownership, occlusion, keep-alive, removed/suppressed ids
```

Frame/memory cache guard: memory reads are temporal and object-local. A graph rewrite may not reorder, batch, or evict frame memories unless it preserves conditioning/non-conditioning priority, reverse tracking direction, `num_maskmem`, `max_cond_frame_num`, and object pointer temporal offsets.

## 7. Position encoding and custom math

Detector and tracker use axial 2D RoPE. Shape depends on patch grid or memory grid. A compact equivalent:

```python
def axial_rope_indices(end_x, end_y, theta, dim, scale=1.0):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 4).float() / dim))
    idx = torch.arange(end_x * end_y)
    x = (idx % end_x) * scale
    y = torch.div(idx, end_x, rounding_mode="floor") * scale
    freqs = torch.cat([x[:, None] * inv, y[:, None] * inv], dim=-1)
    return freqs.cos(), freqs.sin()
```

Tracker object pointers also use 1D sine temporal encodings normalized by `max_object_pointers_in_encoder - 1`, projected from 256 to memory dim 64 when pointer tokens are split.

The memory encoder mask transform is source-specific:

```python
mask_for_mem = sigmoid(pred_masks_high_res)   # unless point-derived eval masks are binarized
mask_for_mem = mask_for_mem * 20.0 - 10.0
```

## 8. Preprocessing and input packing

Video decode/frame sampling is outside the model. `Sam3VideoProcessor.init_video_session(video=...)` calls the configured video processor and stores `processed_video.pixel_values_videos[0]`. The model then processes one frame at a time. Streaming callers instead preprocess each frame through `processor(images=frame, return_tensors="pt")` and pass `inputs.pixel_values[0]`.

Frame ABI:

- Processor/model source uses channel-first frame tensors: `[3,H,W]` in session, `[1,3,H,W]` for detector.
- Original frame size is kept as `[height,width]` for postprocess; streaming requires explicit `original_sizes` if the session lacks video dimensions.
- Text prompts are tokenized with CLIP tokenizer, `padding="max_length"`, `max_length=32`, and cached per prompt id.

Postprocess ABI:

- Input: `obj_id_to_mask` maps object id to low-res mask logits shaped `[1,288,288]`.
- Upsample: bilinear to original video `(H_video,W_video)`, `align_corners=False`.
- Binarize: `mask > 0`.
- Filter: remove zero-area masks, suppressed ids, and hotstart-removed ids.
- Boxes: `torchvision.ops.masks_to_boxes` on binary masks, XYXY absolute coordinates.
- Non-overlap: prompt-group object-wise suppression based on tracker probabilities.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> Linear

Source pattern: ViT patch embedding `Conv2d(3,1024,kernel=14,stride=14,padding=0,bias=False)` followed by flatten/transpose.

Replacement:

```text
NCHW fixed-size image -> non-overlap 14x14 patch flatten -> GEMM(weight_flat.T) -> [B,72*72,1024]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == dilation == 0/1` as source.
- `groups == 1`, no bias.
- `H` and `W` divisible by patch size.
- Flatten order must match PyTorch Conv2d NCHW memory order.

Failure cases: custom image sizes not divisible by patch size; channel-last input unless a full layout rewrite owns patch extraction.

Parity test: compare Conv2d patch output to unfolded+GEMM output for fp32/fp16 on 1008 and custom 560.

### Rewrite: prompt text embedding cache

Source pattern: repeated `get_text_features(input_ids, attention_mask).pooler_output` per prompt id.

Replacement: session-owned prompt embedding constant/input cache.

Preconditions:

- Prompt ids and attention masks unchanged.
- Cache invalidated on `reset_state()` or prompt table clear.
- Dtype/device placement matches inference session.

Failure cases: tokenizer config changes, dynamic prompt edits under same prompt id.

### Rewrite: mask postprocess as bounded output kernel

Source pattern: low-res mask cat -> bilinear interpolate -> threshold -> area filter -> optional prompt-group non-overlap -> boxes.

Replacement: keep low-res mask graph compileable, run postprocess as explicit bounded runtime stage first.

Preconditions:

- Output resolution known from session/original sizes.
- `masks_to_boxes` available or replaced by a tested reduction kernel.
- Variable number of objects remains a host/runtime record, not fixed tensor ABI.

Failure cases: zero objects, empty masks, hidden object ids, prompt group filtering.

### Rewrite: NCHW/NHWC local layout region

Source pattern: NCHW image -> patch tokens -> ViT layers use `[B,H,W,C]` for window partition and LayerNorm-like channels-last operations, then return to sequence/NCHW FPN.

Replacement: a guarded local layout pass may keep ViT token maps in NHWC/channel-last inside the ViT/FPN boundary.

Preconditions:

- Entire region from patch embedding output through window partition/unpartition and per-token MLP is controlled.
- Axis-sensitive ops rewritten together: LayerNorm over `C`, window pad/view/permute, flatten `[H,W]`, FPN reshape back to NCHW.
- Consumers expecting NCHW FPN outputs are protected by a layout boundary.

Failure cases: exposing NHWC to tracker memory encoder, processor, postprocess, Conv2d/FPN layers without transformed weights.

## 10. Kernel fusion candidates

Highest priority:

- ViT patch embedding + initial layout materialization: large image path, simple guarded conv-to-GEMM.
- Window attention with 2D RoPE: dominant detector cost at 72x72 tokens with 24x24 windows.
- Tracker memory attention: source has fixed small hidden size but stateful memory sequence; this is the core video-tracking cost.
- Mask decoder `einsum("bqc,bchw->bqhw")`: direct batched GEMM/conv-like lowering for query masks.

Medium priority:

- FPN/pixel decoder conv + upsample regions.
- Memory encoder mask transform + downsample + fuser.
- Postprocess bilinear upsample + threshold + area reduction.
- Prompt text encoder cache and batched multi-prompt detector execution.

Lower priority:

- Mask NMS and connected components acceleration. Important for quality, but can be runtime/postprocess first.
- Hotstart/object suppression heuristics. Keep host-controlled until state ABI is stable.

## 11. Runtime staging plan

Stage 1: config and session ABI only. Parse `Sam3VideoConfig`, reject unsupported ungated/unknown subconfig combinations, model prompt/frame/object state records explicitly.

Stage 2: detector-only parity for one frame and one text prompt. Reuse the separately audited `sam3` detector body; allow dense PyTorch/postprocess fallback for tracker.

Stage 3: tracker single-object propagation with precomputed detector/tracker features. Validate memory tensor shapes, object pointer shapes, and cond/non-cond output storage.

Stage 4: memory encoder/update parity. Add mask memory write/read ABI, dtype conversion to bf16, temporal memory selection, and object pointer appending.

Stage 5: full per-frame `Sam3VideoModel.forward` parity with host-side association/hotstart heuristics.

Stage 6: postprocess parity and variable-object output records.

Stage 7: optimized layout/attention/fusion passes behind strict guards.

Initially stub: optional connected-components kernel, mask NMS acceleration, custom resolution beyond fixed admitted grids, reverse propagation, multi-prompt batching optimizations.

## 12. Parity and validation plan

- Source-shape tests for config setters: `image_size=1008` and `560` propagate patch/FPN/memory RoPE grids.
- Processor ABI tests: preloaded video session and streaming frame path produce `[3,H,W]` frames plus original sizes.
- Detector parity: one frame, one prompt, compare pred logits/boxes/masks/presence logits at fp32.
- Text cache parity: repeated prompt id returns same embeddings and detector outputs.
- Tracker memory ABI tests: initial conditioning frame writes `object_pointer`, `maskmem_features`, `maskmem_pos_enc`; subsequent frame reads exactly selected memories.
- Postprocess tests: zero objects, one object, multiple prompt groups, suppressed ids, hotstart-removed ids, empty mask removal.
- Tolerances: fp32 `1e-4` to `1e-5` for local ops; fp16/bf16 `1e-2` for attention/mask logits and exact boolean parity only after threshold margins are separated from zero.

## 13. Performance probes

- Video processor throughput: frames/sec decode + resize on CPU vs GPU.
- Detector vision encoder throughput by image size: 1008, 560, any admitted custom buckets.
- Text prompt cache hit/miss cost and multi-prompt sweep.
- Detector DETR/mask decoder time per prompt and per query count.
- Tracker per-object propagation time vs object count.
- Memory attention time vs selected memory frames and pointer tokens.
- Memory storage footprint: per object per frame for bf16 `[72*72,1,64]` plus position encodings and pointers.
- Postprocess time vs object count and output resolution, split into interpolate/threshold/boxes/NMS/connected components.
- Streaming vs preloaded video latency, including hotstart buffering.

## 14. Skip/defer list

- Training, losses, gradient checkpointing.
- Official checkpoint weight loading until gated access is resolved.
- `facebook/sam3.1` and any non-Transformers remote-code variant.
- General-purpose dynamic arbitrary video frame sampling inside DinoML; first integration should consume preprocessed frames.
- Optional connected-components quality kernel as compiled graph op; use host/runtime fallback or reject quality-enhanced mode first.
- Full host heuristic lowering for hotstart, association, removed/suppressed ids. Keep as runtime/controller logic.
- Broad NHWC translation outside guarded ViT-local regions.

## 15. Final implementation checklist

- [ ] Parse `Sam3VideoConfig` and nested `sam3` / `sam3_tracker_video` configs.
- [ ] Add gated-source policy for official `facebook/sam3` weights/config.
- [ ] Model `Sam3VideoInferenceSession` as explicit runtime state, not hidden graph state.
- [ ] Implement/admit NCHW frame ABI and original-size metadata.
- [ ] Compose or reuse `sam3` detector audit implementation.
- [ ] Compose or reuse `sam3_tracker_video` tracker audit implementation.
- [ ] Implement prompt embedding cache ABI.
- [ ] Implement frame vision feature cache ABI with bounded eviction.
- [ ] Implement per-object cond/non-cond output records.
- [ ] Implement mask memory tensor ABI and object pointer token ABI.
- [ ] Add detector-only one-frame parity.
- [ ] Add tracker single-object memory parity.
- [ ] Add full per-frame orchestration parity with host-side heuristics.
- [ ] Add postprocess parity for masks, boxes, filtering, prompt grouping.
- [ ] Add guarded Conv2d patch-embed to GEMM rewrite.
- [ ] Add guarded ViT local NHWC/channel-last layout rewrite.
- [ ] Benchmark detector, tracker memory attention, postprocess, and state memory footprint.
