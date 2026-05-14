# Transformers Audit: `sam3_tracker_video`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/sam3, tracker sub-config/model family sam3_tracker_video
Config source: pinned source defaults; raw facebook/sam3 config is gated and returned HTTP 401
Source files inspected:
  X:/H/transformers/src/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py
  X:/H/transformers/src/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py
  X:/H/transformers/src/transformers/models/sam3_tracker_video/processing_sam3_tracker_video.py
  X:/H/transformers/src/transformers/models/sam3_tracker_video/modular_sam3_tracker_video.py
  X:/H/transformers/src/transformers/models/sam3/configuration_sam3.py
  X:/H/transformers/src/transformers/models/sam3/modeling_sam3.py
Any missing files or assumptions:
  facebook/sam3 raw config and preprocessor configs are manually gated. This report uses source defaults plus public HF API metadata.
```

The generated `configuration_*`, `processing_*`, and `modeling_*` files are generated from `modular_sam3_tracker_video.py`. The modular file is the upstream edit source; the generated modeling file is the clearest audit source because it expands the inherited SAM2-video implementation.

Primary target for DinoML: inference-only interactive/streaming video object segmentation and propagation using point, box, or mask prompts. Full `sam3_video` detector association is a separate composite audit; this report owns the tracker body and session ABI.

## 2. High-level architecture

```text
CPU/video processor
  -> processed frames [T,C,H,W] stored in Sam3TrackerVideoInferenceSession
  -> per-frame vision encoder/FPN cache
  -> per-object prompt inputs and history
  -> memory-conditioned SAM mask decode per object/frame
  -> mask memory encoder writes per-object frame memories
  -> propagation iterator reuses memory and object pointers forward or reverse
```

Stage decomposition:

- CPU/data pipeline: video decode/sampling is owned by `video_processor`, not this model. The processor normalizes points/boxes to target size, pads points/labels, rejects box padding, resizes mask prompts, and creates/updates the Python session.
- Cacheable encoder: `get_image_features` runs a SAM3 vision encoder and FPN once per frame, projects high-resolution levels through mask-decoder `1x1` convs, flattens `NCHW -> HW,N,C`, and stores cached frame features.
- Stateful tracker core: per object, current frame features are fused with selected conditioning/non-conditioning memory frames plus object pointer tokens through memory attention.
- Prompt/mask decode: sparse point/box embeddings and dense mask/no-mask embeddings feed a two-way transformer mask decoder with multimask and stability selection.
- Memory writeback: predicted high-res masks are downsampled and fused with current visual features, then stored as per-object `maskmem_features`, `maskmem_pos_enc`, `object_pointer`, masks, and object score logits.

Independently validatable stages: processor packing, vision/FPN feature ABI, prompt encoder, mask decoder on precomputed image features, memory attention on synthetic memories, memory encoder writeback, and end-to-end session propagation.

## 3. Important config dimensions

Source-default dimensions:

| Field | Default | Source/provenance |
| --- | ---: | --- |
| `image_size` | 1008 | tracker config default |
| `vision_config.model_type` | `sam3_vision_model` | tracker config default |
| `backbone_config.model_type` | `sam3_vit_model` | composed SAM3 vision config |
| ViT hidden/layers/heads | 1024 / 32 / 16 | `sam3` vision source defaults |
| ViT patch/window/global layers | patch 14, window 24, global `[7,15,23,31]` | `sam3` vision source defaults |
| FPN hidden size | 256 | `Sam3VisionConfig` default |
| tracker feature sizes | `[[288,288],[144,144],[72,72]]` | `4x,2x,1x image_size/patch_size` |
| prompt hidden / point embeddings | 256 / 4 | tracker prompt config |
| prompt mask input size | `288x288` | `4 * image_size // patch_size` |
| mask decoder layers/heads | 2 / 8 | tracker mask decoder config |
| mask decoder MLP dim | 2048 | tracker mask decoder config |
| multimask outputs | 3 plus single-mask token | tracker mask decoder config |
| memory slots | 7 | tracker config |
| max conditioning frames | 4 | tracker config |
| max object pointers | 16 | tracker config |
| memory attention | 4 layers, hidden 256, 1 head, FFN 2048 | tracker config |
| memory RoPE feature size | `[72,72]` | derived from image/patch |
| memory encoder output channels | 64 | tracker config |
| memory fuser | 2 ConvNeXt-like blocks, depthwise `7x7`, FFN 1024 | tracker config |
| mask downsampler | total stride 16, repeated stride-2 convs | tracker config |

Representative checkpoint sweep:

| Checkpoint/config | Accessibility | Operator-significant facts |
| --- | --- | --- |
| `facebook/sam3` | Gated raw config, public API visible | API reports `Sam3VideoModel`, `model_type=sam3_video`, F32 safetensors metadata. Tracker sub-config contents could not be inspected. |
| Source default `Sam3TrackerVideoConfig()` | Local pinned source | Best available tracker basis: 1008 image size, 72x72 memory grid, 7 memory slots. |
| Custom source-derived image size | Local pinned source | `image_size` setter rewrites prompt/vision image size, feature sizes, and memory RoPE grid together. DinoML must reject inconsistent serialized configs. |

## 3a. Family variation traps

- `sam3_tracker_video` may be loaded from a parent `sam3_video` config with `tracker_config`; the tracker class unwraps that config. DinoML should normalize to the tracker config before graph planning.
- The vision encoder is delegated through `AutoModel.from_config(config.vision_config)`. Treat SAM3 vision as a composed family, not tracker-owned primitive coverage, unless the first target explicitly includes the vision backbone.
- `image_size`, ViT patch size, FPN feature sizes, prompt mask size, and memory RoPE feature size are coupled. Rewriting one without the others corrupts flatten/view and RoPE shapes.
- Source semantics are mostly `NCHW` for image maps, with explicit local `NCHW <-> NHWC` permutes around LayerNorm and pointwise Linear blocks. NHWC is an optimization only under tight region guards.
- Memory cache is not a KV cache. It is per-object, per-frame state with conditioning/non-conditioning histories, object pointer tokens, and mask memory features.
- Box prompts are converted into two point prompts with labels 2 and 3 and must precede later clicks; box padding across images is rejected.
- Mask prompts bypass the normal prompt+decoder output path for initial frame output and are converted directly to logits, but still call the SAM decoder to produce object pointers.
- `maskmem_features` are intentionally cast to bfloat16 for storage even when the session dtype differs.
- Flash Attention is allowed generally, but target-guided additive attention masks force SDPA fallback in prompt decoder attention.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `flatten`, `permute`, `transpose`, `contiguous`, `repeat`, `repeat_interleave`, `expand`, `cat`, `stack`, `gather`, advanced indexing for best-IoU mask selection.
- NCHW image maps: convs, transposed convs, bilinear interpolate, `flatten(2).permute(2,0,1)` sequence packing.
- NHWC local regions: LayerNorm on channel-last after permuting from NCHW, pointwise Linear over channel dimension, then permute back.

Neural primitives:

- SAM3 vision encoder/FPN from `sam3` family: ViT patch embedding, window/global attention, FPN `1x1`/upsample paths.
- Prompt coordinate Fourier embedding: coordinate normalization, matmul with learned random table, sin/cos concat.
- Prompt mask embedding: Conv2d `1->4`, Conv2d `4->16`, Conv2d `16->256`, LayerNorm channels-first, GELU.
- Mask decoder: two-way transformer, ConvTranspose2d `256->64`, ConvTranspose2d `64->32`, hypernetwork MLPs, `hyper_in @ upscaled_embedding`, IoU/object-score MLPs.
- Memory encoder: mask downsampler repeated stride-2 convs to stride 16, feature projection `1x1`, depthwise `7x7` memory fuser, pointwise Linear `256->1024->256`, projection `1x1` to 64.

Attention primitives:

- Noncausal dense MHA for prompt decoder self/cross attention: 8 heads, hidden 256, internal dim 256 or 128 when downsample rate 2.
- Noncausal memory attention: self-attention over current image tokens and cross-attention from current tokens to memory tokens, 1 head, head dim 256.
- 2D RoPE on memory attention q/k, with object pointer tail tokens excluded from key RoPE.

Generation/cache ops:

- No autoregressive generation and no KV cache.
- Required session state: frame storage, frame feature LRU cache, per-object prompt/mask inputs, output history, conditioning/non-conditioning frame maps, object ID maps, new-input flags.

Pre/postprocessing-coupled ops:

- Point/box coordinate normalization to target size.
- Pad points/labels with `point_pad_value=-10`; preserve padding through normalization.
- Reject padded boxes; boxes become point coordinates plus labels `[2,3]`.
- Resize mask prompts to target size and threshold at 0.5 in processor.
- Model postprocess: object-score masking to `NO_OBJ_SCORE=-1024`, bilinear upsample low-res masks to `image_size`, select best IoU mask, optionally apply dynamic stability fallback.

## 5. Layer/block breakdown

Vision feature path:

```text
pixel_values [B,3,H,W]
  -> Sam3VisionModel -> last_hidden_state [B,(H/14)*(W/14),1024]
  -> view [B,H/14,W/14,1024] -> permute [B,1024,H/14,W/14]
  -> FPN levels [B,256,4H/14,4W/14], [B,256,2H/14,2W/14], [B,256,H/14,W/14]
  -> conv_s0/conv_s1 project high-res levels to 32 and 64 channels
  -> flatten levels to [HW,B,C] for tracker cache
```

Prompt encoder:

```text
points [B,O,P,2], labels [B,O,P]
  -> +0.5 pixel-center shift
  -> coordinate positional embedding [B,O,P,256]
  -> where labels == -1 / -10
  -> add point label embedding for labels >= 0
boxes [B,O,4]
  -> reshape [B,O,2,2] + padding corner
  -> point labels 2/3/not-a-point
masks [B,O,H,W]
  -> conv/ln/gelu downsample to dense embedding [B,256,72,72]
```

Memory-conditioned frame block:

```text
current lowest feature [5184,B,256]
if initial conditioning frame:
  add no_memory_embedding -> view [B,256,72,72]
else:
  gather conditioning and recent non-conditioning memory features
  gather object pointers and temporal pointer embeddings
  cat memories [M,B,64] and memory pos [M,B,64]
  memory attention:
    LayerNorm -> RoPE self-attn over current tokens
    LayerNorm -> RoPE cross-attn to memory tokens
    LayerNorm -> MLP 256->2048->256
  view [B,256,72,72]
```

Mask decoder:

```text
image embedding [B,256,72,72] + dense prompt embedding
tokens = obj_score token + iou token + 4 mask tokens + sparse prompt tokens
repeat image per object/prompt batch
2 x two-way attention block:
  token self-attn
  token->image cross-attn
  token MLP
  image->token cross-attn
final token->image attention + LayerNorm
upscale image embeddings with ConvTranspose2d and FPN skip features
mask logits = hypernetwork(mask_tokens) @ flattened upscaled embedding
iou scores and object score logits from MLP heads
```

Memory write:

```text
high_res_mask logits
  -> resize to memory mask size if needed
  -> hard threshold if prompt-derived else sigmoid
  -> scale * 20 + bias -10
  -> memory encoder with current visual feature
  -> cast maskmem_features to bf16
  -> flatten/permute to [Hmem*Wmem,B,64]
  -> store per object/frame
```

## 6. Attention requirements

The tracker uses noncausal dense attention only. There is no decode-time KV cache.

Prompt/mask decoder attention:

- Self-attention over sparse prompt/output tokens and bidirectional cross-attention between prompt tokens and flattened image tokens.
- MHA with `num_attention_heads=8`; downsampled cross-attention uses internal dim `hidden_size // 2 = 128`, head dim 16.
- Optional `attention_similarity` is an additive float mask/bias for token-to-image attention. If Flash Attention is requested, source falls back to SDPA for that call.

Memory attention:

- Noncausal self-attention over current image tokens plus noncausal cross-attention from current image tokens to memory tokens.
- `memory_attention_num_attention_heads=1`, hidden/internal dim 256, head dim 256.
- Keys/values for memory cross-attention have value width 64 before projection; q projection is `256->256`, k/v are `64->256`.
- RoPE is applied to q and k. Object pointer tokens are appended at the tail of memory keys and excluded from key RoPE using `num_k_exclude_rope`.

State ABI, separate from KV cache:

- State owner: `Sam3TrackerVideoInferenceSession`.
- Frame storage: `processed_frames: dict[int, Tensor[C,H,W]]`.
- Vision feature cache: LRU-ish dict keyed by frame index; stores `vision_feats` and `vision_pos_embeds`, moved between `inference_state_device` and `inference_device`.
- Per-object IDs: external `obj_id` maps to compact `obj_idx`.
- Prompt histories: `point_inputs_per_obj[obj_idx][frame_idx]`, `mask_inputs_per_obj[obj_idx][frame_idx]`.
- Output histories: `output_dict_per_obj[obj_idx]["cond_frame_outputs"|"non_cond_frame_outputs"][frame_idx]`.
- Stored output keys: `pred_masks`, `high_res_masks`, `object_pointer`, `object_score_logits`, later `maskmem_features`, `maskmem_pos_enc`.
- Eviction/window: vision feature cache size is configurable; memory retrieval uses up to `num_maskmem - 1` recent non-conditioning frames plus up to `max_cond_frame_num` closest conditioning frames.
- Direction: propagation can run forward or reverse; memory frame lookup and object pointer eligibility reverse temporal offsets accordingly.

## 7. Position encoding and custom math

2D sine position embeddings for image/memory maps are generated from cumulative unmasked x/y coordinates, normalized to `2*pi`, then interleaved sin/cos and returned as NCHW.

Memory RoPE is fixed for the configured memory grid:

```python
freqs = 1.0 / (theta ** (arange(0, head_dim, 4) / head_dim))
flat = arange(grid_w * grid_h)
x = flat % grid_w
y = flat // grid_w
cos_sin_arg = cat([outer(x, freqs), outer(y, freqs)], -1).repeat_interleave(2, -1)
```

Application uses pairwise rotation, not Llama half-rotation:

```python
def rotate_pairwise(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x0, x1 = x.unbind(-1)
    return stack((-x1, x0), -1).flatten(-2)

q = q.float() * cos + rotate_pairwise(q.float()) * sin
k_rot = k[..., :-num_pointer_tokens, :]
k_pass = k[..., -num_pointer_tokens:, :]
k = cat([k_rot.float() * cos_k + rotate_pairwise(k_rot.float()) * sin_k, k_pass], -2)
```

Precompute candidates:

- Memory RoPE cos/sin are fixed by `memory_attention_rope_feat_sizes`, `head_dim`, and `theta`.
- Image-wide prompt positional embeddings depend on configured prompt image embedding size and learned random positional table; they can be cached per model/device/dtype.
- Temporal object-pointer sine embeddings depend on selected frame offsets and `max_object_pointers_to_use`, so they are runtime state.

## 8. Preprocessing and input packing

Video/frame ABI:

- `init_video_session(video=...)` calls `video_processor(videos=video, return_tensors="pt")`, takes `pixel_values_videos[0]`, and stores each frame independently. This model does not define decode/fps/frame sampling; that belongs to the video processor.
- Stored frame layout is per-frame `[C,H,W]`; model calls add batch to `[1,C,H,W]`.
- Streaming mode allows `frame` to be provided directly to `forward`; the session stores it with optional `frame_idx`.

Prompt ABI:

- Points: processor expects `[image, object, point, 2]`; labels `[image, object, point]`.
- Labels: `1` positive, `0` negative, `-1` background/not-a-point, `-10` padding.
- Boxes: `[image, object, 4]` in `x1,y1,x2,y2`, normalized to target size, then reshaped to two point coordinates with labels `[2,3]`.
- Masks: processor accepts `[H,W]` up to `[B,O,H,W]`, resizes to target size, thresholds when resized, and stores float masks in session.

Postprocessing:

- Processor `post_process_masks` delegates to the image processor to remove padding, resize to original size, threshold, optionally fill holes/sprinkles and apply non-overlap constraints.
- Model outputs remain logits at model resolution unless caller invokes processor postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: channels-first LayerNorm island

Source pattern:

```text
NCHW -> permute NHWC -> LayerNorm(C) -> permute NCHW
```

Replacement: native channels-first LayerNorm or fused Conv2d/LayerNorm/activation kernel.

Preconditions: normalized dimension is channel count; input is contiguous or stride-aware lowering exactly preserves NCHW indexing; no consumer observes NHWC intermediate.

Failure cases: generic LayerNorm over non-channel axes, dynamic non-contiguous tensors without stride support.

Parity test sketch: random NCHW feature maps for mask downsampler and memory fuser, compare fp32 and bf16-tolerant outputs.

### Rewrite: local pointwise Linear as 1x1 Conv

Source pattern:

```text
depthwise_conv NCHW -> LayerNorm -> permute NHWC -> Linear C->4C -> GELU -> Linear 4C->C -> scale -> permute NCHW
```

Replacement: keep as NHWC GEMM island or transform Linear weights to `1x1` conv weights for an all-NCHW fused block.

Preconditions: Linear is applied only over channel-last dimension; no broadcasting beyond batch/spatial; layer scale is per-channel.

Failure cases: if a layout pass cannot rewrite both Linears and the scale multiply together.

### Rewrite: non-overlap ConvTranspose2d upsample

Source pattern: mask decoder uses `ConvTranspose2d(kernel_size=2, stride=2)` twice.

Replacement: specialized 2x learned upsample kernel or im2col/GEMM.

Preconditions: kernel 2, stride 2, padding 0, dilation 1, groups 1, static channel counts.

Failure cases: do not replace bilinear `F.interpolate`; those have different math.

### Rewrite: hypernetwork mask matmul

Source pattern:

```text
hyper_in [B,O,T,32] @ upscaled_embedding [B,O,32,H*W] -> masks [B,O,T,H,W]
```

Replacement: batched GEMM/BMM with small K=32 and variable `B*O*T`.

Preconditions: upscaled embedding contiguous after view; mask token count known; output token slicing preserved.

### Rewrite: memory frame gather as explicit state read plan

Source pattern: Python dict lookup, temporal sorting, `torch.cat` memories and positional embeddings.

Replacement: DinoML session-side state manifest with explicit memory slots, frame indices, object ids, direction, and selected conditioning/non-conditioning entries.

Preconditions: admission fixes max objects, max memory slots, max pointer tokens, and frame window policy; missing frames are skipped like source.

Failure cases: arbitrary Python dict mutation during graph run; changing object ids without session reset.

## 10. Kernel fusion candidates

Highest priority:

- Memory attention q/k/v projections + 2D RoPE + dense attention. This is on every propagated frame and has unusual pointer-token RoPE exclusion.
- Mask decoder two-way attention. It mixes small prompt-token attention with larger image-token cross-attention; a good lowering avoids generic scatter/gather overhead.
- Memory encoder mask downsample + visual fusion + depthwise ConvNeXt block. This runs after each tracked object/frame writeback.
- Hypernetwork mask BMM and upsample path. It directly gates mask throughput and has stable small channel dimensions.

Medium priority:

- Channels-first LayerNorm fusion for conv blocks.
- Bilinear resize with antialias for mask prompt and memory-mask preparation.
- Best-IoU/multimask selection gather and stability fallback.
- Object pointer packing/splitting and temporal sine embedding.

Lower priority:

- Processor-side coordinate normalization and padding on GPU. Keep CPU first unless batching becomes a bottleneck.
- Vision backbone optimizations should be handled by the separate `sam3` vision audit.

## 11. Runtime staging plan

1. Parse tracker config and reject inconsistent `image_size`/feature-size/RoPE-grid combinations.
2. Compose or stub SAM3 vision features. First useful tracker parity can accept precomputed `fpn_hidden_states` and `fpn_position_encoding`.
3. Implement prompt encoder for points/boxes/no-mask on one frame.
4. Implement mask decoder on precomputed image embeddings; validate low-res and high-res masks.
5. Add session manifest: frame ids, object ids, prompt histories, output histories, memory slots, device residency.
6. Implement memory encoder writeback and bfloat16 memory storage.
7. Implement memory attention and pointer-token packing for propagation.
8. Add processor-compatible mask prompt path and direct mask-as-output path.
9. Add guarded performance rewrites and layout islands.

Stub initially:

- Full `sam3_video` detector association and NMS.
- Processor-owned video decode/fps sampling.
- Hole/sprinkle/non-overlap postprocess unless end-to-end mask postprocess parity is the target.

## 12. Parity and validation plan

- Processor tests: point padding/normalization preserves `-10`; boxes reject uneven counts; boxes prepend labels `[2,3]`; mask prompt resize and threshold match source.
- Prompt encoder parity: random points/labels/boxes/masks against PyTorch source at fp32.
- Mask decoder parity: synthetic image embeddings, prompt embeddings, high-res features; cover multimask true/false and stability fallback.
- Memory encoder parity: cover prompt-derived hard mask and sigmoid mask paths; assert stored dtype/shape `[H*W,B,64]`.
- Memory attention parity: synthetic current tokens, memory tokens, pointer tail tokens; verify RoPE exclusion.
- Session state tests: add object ids, update same/different frames, reset tracking data vs reset full session, forward and reverse propagation selection.
- End-to-end tracker: one short video with one point prompt, then propagate; compare frame masks/object scores to Transformers.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; bf16/fp16 stateful paths need looser mask-logit tolerance and exact shape/state checks.

No DinoML tests/imports were run for this audit, per scope.

## 13. Performance probes

- Processor/video decode throughput vs model throughput.
- Vision encoder cache hit/miss cost per frame.
- Single-object propagation frames/sec at different `num_maskmem`, `max_cond_frame_num`, and pointer counts.
- Multi-object scaling where source loops per object for inference but batches memory encoding.
- Memory attention token sweep: current grid 72x72, memory slots, object pointer tokens.
- Mask decoder prompt count sweep: points/boxes per object and multimask mode.
- Memory state residency: CPU vs GPU `inference_state_device`, transfer volume for `maskmem_features` and frame cache.
- Layout rewrite A/B: source NCHW with permute islands vs fused channel-last islands.

## 14. Skip/defer list

- Training, dropout behavior, gradient checkpointing.
- Full `Sam3VideoModel` detection/association hotstart policy.
- Video decode and frame sampling ownership.
- Processor image postprocess morphology/non-overlap for first graph parity.
- General NHWC conversion across the whole model.
- General Python dict mutation inside compiled graphs; require explicit session operations.
- Quantization and packed weights.

## 15. Final implementation checklist

- [ ] Add gated config admission for `sam3_tracker_video` source-default-compatible configs.
- [ ] Define explicit DinoML tracker session/state ABI.
- [ ] Compose separately audited `sam3_vision_model` feature provider or accept precomputed FPN tensors.
- [ ] Implement point/box/no-mask prompt encoder.
- [ ] Implement mask prompt embedding and direct mask-as-output path.
- [ ] Implement two-way mask decoder attention and hypernetwork mask BMM.
- [ ] Implement multimask/stability/best-IoU selection.
- [ ] Implement memory encoder writeback with bf16 `maskmem_features`.
- [ ] Implement memory frame selection and object pointer packing.
- [ ] Implement 2D memory RoPE with pointer-token exclusion.
- [ ] Add layout guards for NCHW/NHWC islands.
- [ ] Add parity tests for one-frame, memory write, and forward/reverse propagation.
- [ ] Benchmark cache hit/miss, object count, memory slots, and state device placement.
