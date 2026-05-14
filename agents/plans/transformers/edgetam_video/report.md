# Transformers Audit: `edgetam_video`

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: yonigozlan/EdgeTAM-hf
Config source: https://huggingface.co/yonigozlan/EdgeTAM-hf/raw/main/config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/edgetam_video/{configuration,modeling,modular}_edgetam_video.py
  X:/H/transformers/src/transformers/models/edgetam/{configuration,modeling}_edgetam.py
  X:/H/transformers/src/transformers/models/sam2_video/{processing,video_processing}_sam2_video.py
  X:/H/transformers/src/transformers/models/sam2/image_processing_sam2.py
  X:/H/transformers/src/transformers/models/timm_wrapper/{configuration,modeling}_timm_wrapper.py
Any missing files or assumptions:
  modeling/configuration files are generated from modular source; generated modeling is the runtime basis.
  The vision backbone is delegated through AutoModel to edgetam_vision_model and then timm_wrapper/RepViT.
  No DinoML tests/imports were run.
```

HF configs inspected:

| Source | Status | Notes |
| --- | --- | --- |
| `yonigozlan/EdgeTAM-hf` | open Transformers checkpoint | `model_type=edgetam_video`, `architectures=["EdgeTamVideoModel"]`, `dtype=float32`, Apache-2.0 metadata. |
| `facebook/EdgeTAM` | open reference repo | HF API exposes `edgetam.pt`, no Transformers `config.json`; useful as provenance, not in-library config basis. |
| Source defaults | available | `EdgeTamVideoConfig` defaults mostly match the HF config except HF disables occlusion spatial embedding and temporal pointer PE. |

## 2. High-Level Architecture

EdgeTAM Video is an interactive, stateful video object segmentation/tracking model, not a text generation model. The first useful DinoML target should be a single-frame/streaming tracking runtime that consumes already-decoded, processor-normalized frames plus prompt state, then returns per-object mask logits and object-presence scores.

Dataflow:

```text
video decode/frame sampling outside model
-> Sam2VideoProcessor resize/rescale/normalize to NCHW 1024x1024
-> EdgeTAM vision encoder/FPN feature cache
-> per-object prompt or prior-memory-conditioned mask decoder
-> memory encoder + spatial perceiver update
-> session output/history store
-> processor post_process_masks to original frame size
```

Stage decomposition:

| Stage | Owner | Cacheability |
| --- | --- | --- |
| Video decode and frame sampling | CPU/data pipeline | Outside neural graph; source video preprocessor has optional `num_frames`/`fps`, but checkpoint leaves them `null`. |
| Frame preprocessing | `Sam2VideoVideoProcessor` | Deterministic resize/rescale/normalize; produces channels-first frames. |
| Vision encoder/FPN | `EdgeTamVideoModel.get_image_features` | Cached per `frame_idx` in `EdgeTamVideoInferenceCache`, moved between inference and state devices. |
| Prompt encoding | `EdgeTamVideoPromptEncoder` | Depends on points/labels/boxes/masks; cache per user interaction if inputs unchanged. |
| Memory-conditioned frame inference | `EdgeTamVideoMemoryAttention` + mask decoder | Depends on per-object history, direction, and selected memory window. |
| Memory update | `_encode_new_memory` | Persists `maskmem_features`, `maskmem_pos_enc`, `object_pointer`, and object logits per object/frame. |
| Postprocess | processor | Upsample/crop/binarize masks to original frame size; not NMS for tracking path. |

## 3. Important Config Dimensions

Primary checkpoint values from `yonigozlan/EdgeTAM-hf`:

| Field | Value | Runtime impact |
| --- | ---: | --- |
| `image_size` | 1024 | Model and processor target frame size. |
| `dtype` | `float32` | Config metadata; runtime can receive other tensor dtypes but parity target should start fp32. |
| `num_maskmem` | 7 | Maximum spatial memory slots: conditioning memories plus up to 6 recent non-conditioning frames. |
| `max_object_pointers_in_encoder` | 16 | Object pointer token history cap. |
| `max_cond_frame_num` | source default `-1` | Use all conditioning frames unless overridden. |
| `memory_attention_hidden_size` | 256 | Query channel width for memory attention. |
| `memory_attention_num_layers` | 2 | Memory attention block count. |
| `memory_attention_num_attention_heads` | 1 | MHA with one head; head dim 256. |
| `memory_attention_mlp_hidden_size` | 2048 | Memory attention FFN width. |
| `memory_attention_mlp_hidden_act` | `relu` | Non-GELU FFN in memory attention. |
| `memory_attention_rope_feat_sizes` | `[64, 64]` | Fixed query RoPE table for lowest FPN grid. |
| `memory_attention_rope_k_sizes` | `[16, 16]` | Fixed key RoPE table for resampled memory tokens. |
| `memory_encoder_hidden_size` | 256 | Feature/mask fuser input width. |
| `memory_encoder_output_channels` | 64 | Spatial memory token width before pointer split handling. |
| `perceiver_resampler_num_latents` | 256 | 1D memory latent tokens. |
| `perceiver_resampler_num_latents_2d` | 256 | 2D memory latent tokens arranged as 16x16. |
| `perceiver_resampler_hidden_size` | 64 | Perceiver hidden width. |
| `mask_decoder.hidden_size` | 256 | SAM-style decoder width. |
| `mask_decoder.num_hidden_layers` | 2 | Two-way transformer depth. |
| `mask_decoder.num_attention_heads` | 8 | Mask decoder attention heads; head dim 32 after downsample. |
| `mask_decoder.num_multimask_outputs` | 3 | Three alternate masks plus one single-mask token. |
| `prompt_encoder.hidden_size` | 256 | Prompt embedding width. |
| `prompt_encoder.patch_size` | 16 | Image embedding grid is 64x64 at 1024 input. |
| `vision_config.model_type` | `edgetam_vision_model` | Delegates to EdgeTAM vision wrapper. |
| `vision_config.backbone_config.model_type` | `timm_wrapper` | Delegated RepViT body; needs separate allowlist. |
| `vision_config.backbone_config.architecture` | `repvit_m1` | Concrete delegated backbone. |
| `vision_config.backbone_feature_sizes` | `[[256,256],[128,128],[64,64]]` | FPN feature maps consumed by decoder and memory attention. |
| `vision_config.fpn_hidden_size` | 256 | FPN channel width. |

Representative checkpoint sweep:

| Config basis | Architecture | Key differences |
| --- | --- | --- |
| `yonigozlan/EdgeTAM-hf` | Transformers `EdgeTamVideoModel` | Full config; disables `enable_occlusion_spatial_embedding` and `enable_temporal_pos_encoding_for_object_pointers`; RepViT M1 delegated backbone. |
| `EdgeTamVideoConfig()` source defaults | same class | Defaults enable occlusion spatial embedding and temporal object-pointer positional encoding; same image size/memory widths. |
| `facebook/EdgeTAM` | upstream `edgetam` library artifact | No in-library config found; route through converter/provenance only, not a separate DinoML admission target. |

## 3a. Family Variation Traps

- The vision backbone is not owned by `edgetam_video`; it is delegated through `AutoModel.from_config(config.vision_config)`. First integration should allowlist `edgetam_vision_model` + `timm_wrapper` + `repvit_m1` or reject.
- HF config disables temporal pointer PE and occlusion spatial embedding, while source defaults enable both. DinoML must read the checkpoint config, not assume defaults.
- Source uses both sequence-first memory tensors (`HW, B, C`) and NCHW feature maps. Layout conversion points are ABI boundaries.
- Memory attention has object-pointer tokens appended after spatial memory tokens, with `num_k_exclude_rope` and `rope_k_repeat` controlling key RoPE application. A generic dense attention call loses this ABI unless these counts are explicit.
- `mask_inputs` and `point_inputs` cannot be present simultaneously for the same object/frame in `_run_single_frame_inference`.
- Prompt boxes cannot be padded by the processor; batched boxes must have consistent count.
- Multimask behavior is input/count dependent: init frames and tracking can request multiple masks only when point count lies within configured min/max.
- Dynamic multimask fallback uses stability scores only in eval and only when `multimask_output=False`.
- `propagate_in_video_iterator` mutates session output histories; runtime parity requires state order, not just tensor output.
- `NCHW` is semantic at processor/model entry. NHWC/channel-last is only a guarded local optimization inside the delegated vision/FPN/pointwise linear regions.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- NCHW resize/normalize input frames; NCHW frame storage.
- `flatten(2).permute(2,0,1)` for FPN maps: `[B,C,H,W] -> [H*W,B,C]`.
- Sequence-to-map reshapes: `[H*W,B,C] -> [B,C,H,W]`.
- `repeat`, `repeat_interleave`, `expand`, `stack`, `cat`, `gather`, `where`, `argmax`, boolean comparisons.
- Device/state copies for session caches; DinoML should model these as runtime state transfers, not graph ops at first.

Neural primitives:

- Delegated RepViT/timm feature extractor: conv, depthwise conv, batch/layer norms, activations, pooling per `repvit_m1` audit.
- FPN neck: 1x1 conv lateral projections, top-down nearest interpolation/adds, sine positional embeddings.
- Prompt encoder: point/box positional embedding, label embeddings, mask embedding conv stack.
- Memory encoder: mask downsampler conv stack, memory fuser ConvNeXt-style depthwise conv + channels-first LayerNorm + linear MLP, feature/mask add.
- Perceiver resampler: learned latent tables, cross attention, self attention, FFN, 1D flatten path, 2D window partition path.
- Mask decoder: two-way transformer, ConvTranspose2d upscaling, 1x1 high-res feature projections, hypernetwork MLPs, batched token-to-mask matmul.

Attention primitives:

- Noncausal dense MHA through Transformers attention interface for mask decoder and perceiver.
- Memory self-attention with 2D RoPE over 64x64 current feature tokens.
- Memory cross-attention to memory tokens with partial key RoPE exclusion for object pointers.
- Source can request FlashAttention/SDPA, but target-guided additive masks force SDPA fallback.

Position/custom math:

- Learned random Fourier prompt position embedding.
- Fixed 2D axial RoPE tables for memory attention.
- 1D sine temporal PE for object pointers when enabled.
- Sine FPN positional encodings inherited from EdgeTAM/SAM2 vision.

State/cache ops:

- Session object-id mapping and per-object dictionaries.
- Frame store: `processed_frames[frame_idx]`.
- Vision feature cache with max size and eviction of smallest frame index.
- Output histories split into `cond_frame_outputs` and `non_cond_frame_outputs`.
- Memory tensors: `maskmem_features`, `maskmem_pos_enc`, `object_pointer`, `object_score_logits`, `pred_masks`.

Pre/postprocessing-coupled ops:

- Resize frames to 1024x1024, rescale by `1/255`, ImageNet mean/std normalize, output channels-first.
- Point/box coordinate scaling from original `(H,W)` into model target coordinate space.
- Mask prompt resize to 256x256 when needed.
- Output mask upsampling/cropping/binarization back to original sizes.

## 5. Layer/Block Breakdown

Frame feature path:

```text
pixel_values [B,3,1024,1024]
-> delegated EdgeTAM vision encoder / RepViT feature maps
-> FPN maps [B,256,256,256], [B,256,128,128], [B,256,64,64]
-> conv_s0: 256 -> 32 on 256x256 map
-> conv_s1: 256 -> 64 on 128x128 map
-> flatten maps to [HW,B,C] for cache and memory attention
```

Memory attention layer, repeated 2 times:

```text
q = LayerNorm(current tokens)
q = RoPE self-attention(q,q,q) + residual
q = LayerNorm(q)
q = RoPE cross-attention(q, memory + memory_pos, memory) + residual
q = LayerNorm(q)
q = Linear(256 -> 2048) -> ReLU -> Dropout -> Linear(2048 -> 256) + residual
```

Mask decoder:

```text
tokens = [obj_score, iou, mask_tokens, sparse_prompt_tokens]
image = low_res_image_embedding + dense_prompt_embedding
two-way transformer(tokens, image tokens, image PE)
upscale image tokens with ConvTranspose2d(256->64) then ConvTranspose2d(64->32)
add high-res FPN projections at 128x128 and 256x256
per-mask hypernetwork Linear stack maps mask token 256 -> 32
mask logits = hyper_in @ upscaled_embedding_flat
iou/object score heads produce quality and presence logits
```

Memory update:

```text
current [4096,B,256] -> [B,256,64,64]
mask logits -> hard binary if from prompts in eval, else sigmoid
mask = mask * 20 - 10
memory_encoder(image_features, mask) -> [B,64,Hm,Wm], pos_enc
optional no-object spatial embedding
spatial_perceiver -> memory tokens and positional encodings
persist in session output history
```

## 6. Attention Requirements

No autoregressive KV cache is used. This family has stateful video tracking memory, not text decode cache.

Memory attention:

- Noncausal self-attention over current frame lowest-resolution tokens.
- Noncausal cross-attention from current tokens to concatenated memory tokens.
- Query hidden width 256, 1 head, head dim 256.
- Spatial memory token width is 64; cross-attention has `kv_in_dim=64` and projects to the attention internal dim.
- Object pointers are stored at hidden width 256 and split/repeated into 64-wide tokens when `mem_dim < hidden_dim`.
- Key sequence is spatial memory tokens first, object pointer tokens last.
- `num_spatial_memory_tokens` and `num_object_pointer_tokens` are required ABI fields for RoPE behavior.

Mask decoder attention:

- Two-way noncausal attention with point/token self-attention, token-to-image cross-attention, MLP, image-to-token cross-attention, final token-to-image attention.
- Hidden width 256, 8 heads, attention downsample rate 2 for internal width 128, head dim 16 for downsampled projections.
- Optional `attention_similarity` additive masks are incompatible with FlashAttention in source and route to SDPA.

Perceiver memory compression:

- Cross-attention from learned latents to flattened image/memory features.
- Self-attention over learned latents.
- 1D latents: 256 tokens.
- 2D latents: 256 tokens, arranged as 16x16 windows over spatial memory.

Stateful video ABI:

| State | Shape basis | Lifetime |
| --- | --- | --- |
| `processed_frames[frame_idx]` | `[3,1024,1024]` after processor; source accepts `[1,3,H,W]` and squeezes in streaming add | Session lifetime until reset. |
| cached `vision_feats` | list of `[HW,B,C]` tensors for FPN levels | Evicted by `max_vision_features_cache_size`; moved to `inference_state_device`. |
| cached `vision_pos_embeds` | list of `[HW,B,C]` tensors | Same as features. |
| `point_inputs_per_obj[obj_idx][frame_idx]` | dict with `point_coords [B,P,N,2]`, `point_labels [B,P,N]` | Persistent user prompts. |
| `mask_inputs_per_obj[obj_idx][frame_idx]` | mask tensor, source expects object dimension for video path | Persistent user prompts. |
| `cond_frame_outputs` | dict per object/frame | Conditioning frames with user inputs. |
| `non_cond_frame_outputs` | dict per object/frame | Propagated frames. |
| `maskmem_features` | source comments and code use `[B,mem_tokens,64]` after perceiver; consumed as `[mem_tokens,B,64]` | Reused by future memory attention. |
| `maskmem_pos_enc` | same token basis as `maskmem_features` | Reused by future memory attention. |
| `object_pointer` | `[B,256]` per object/frame | Reused as memory tokens; may split into four 64-wide tokens. |
| `object_score_logits` | `[B,1]` | Output and occlusion/no-object gating. |

## 7. Position Encoding and Custom Math

2D axial RoPE for memory attention:

```python
freqs = 1.0 / (theta ** (arange(0, dim, 4) / dim))
x = arange(end_x * end_y) % end_x
y = floor_divide(arange(end_x * end_y), end_x)
angles = cat([outer(x, freqs), outer(y, freqs)], dim=-1)
rope = repeat_interleave(angles, 2, dim=-1)
cos, sin = cos(rope), sin(rope)
```

Pointer temporal PE when enabled:

```python
normalized = temporal_offsets / float(max_object_pointers_to_use - 1)
sine_pe = get_1d_sine_pe(normalized, dim=hidden_dim)
pointer_pos = Linear(hidden_dim -> mem_dim)(sine_pe)
```

Mask-memory logit scaling:

```python
if is_mask_from_points and eval:
    mask_for_mem = (pred_masks_high_res > 0).to(dtype)
else:
    mask_for_mem = sigmoid(pred_masks_high_res)
mask_for_mem = mask_for_mem * 20.0 - 10.0
```

Dynamic single-mask fallback:

```python
stability = sum(mask > delta) / sum(mask > -delta)
if stability < 0.98:
    choose argmax IoU among multimask tokens 1..3
else:
    choose single-mask token 0
```

## 8. Preprocessing and Input Packing

Video/frame preprocessing:

- The checkpoint `video_preprocessor_config.json` uses `Sam2VideoVideoProcessor`.
- Output `pixel_values` are channels-first (`data_format="channels_first"`).
- Resize target is 1024x1024, bilinear resample, RGB conversion enabled, rescale `1/255`, ImageNet mean/std normalize.
- `num_frames`, `fps`, and `do_sample_frames` are `null` in the checkpoint. Video decode and sampling policy is not fixed by this config; DinoML should require caller-provided decoded frames or an explicit data-pipeline policy.
- The video processor records `original_sizes` and `reshaped_input_sizes` for postprocess.

Prompt packing:

- Points: nested `[image, object, point, xy]`, tensor rank 4, last dim 2.
- Labels: nested `[image, object, point]`, tensor rank 3. Padding label is `-10`.
- Boxes: nested `[image, box, xyxy]`, tensor rank 3, last dim 4. Processor rejects box padding across images.
- Coordinates are scaled from original `(H,W)` to target 1024 coordinate space.
- If no points or boxes reach `_single_frame_forward`, source inserts a dummy point `[0,0]` with label `-1`.

Postprocess:

- `Sam2VideoVideoProcessor.post_process_masks` expects masks as `[batch/object, channels, H, W]` style tensors.
- It interpolates to processor pad/target size, crops to `reshaped_input_sizes`, then interpolates to `original_sizes`.
- It optionally binarizes with `mask_threshold=0.0`.
- Tracking path does not apply NMS; SAM image automatic mask generation has separate crop/NMS utilities outside the first video target.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: NCHW FPN flatten to layout-aware sequence view

Source pattern:

```text
feature_map.flatten(2).permute(2,0,1)
```

Replacement:

```text
NCHW map -> sequence view [H*W,B,C]
```

Preconditions:

- Input map is contiguous NCHW.
- Consumers are memory attention or decoder reshape paths expecting row-major `h*w` order.
- No NHWC translation crosses the persisted cache ABI unless the cache manifest records layout.

Failure cases:

- Strided/non-contiguous maps.
- Any downstream operation consuming `[HW,B,C]` as materialized memory with PyTorch-compatible strides.

Parity test sketch: compare cached feature tensors and a full `_prepare_memory_conditioned_features` call for one frame.

### Rewrite: mask decoder hypernetwork matmul to grouped GEMM

Source pattern:

```text
hyper_in [B,P,M,32] @ upscaled_embedding [B,P,32,H*W]
```

Replacement:

```text
batched GEMM over (B*P*M) x 32 by 32 x (H*W)
```

Preconditions:

- `num_mask_tokens=4`, upscaled channel width 32 for this config.
- Dense contiguous flatten of `[B,P,32,H*W]`.
- Preserve mask token order: object score token, IoU token, mask tokens, with mask tokens sliced according to multimask mode.

Failure cases:

- Dynamic decoder configs changing hidden size or transposed-conv output width.
- Non-contiguous output embedding after fusion.

Parity test sketch: single decoder forward with fixed random embeddings, compare masks before IoU selection.

### Rewrite: ConvTranspose upscaling plus high-res add fusion

Source pattern:

```text
ConvTranspose2d(256->64,k=2,s=2) + feat_s1
LayerNorm channels_first + GELU
ConvTranspose2d(64->32,k=2,s=2) + feat_s0
GELU
```

Replacement:

```text
specialized upsample-deconv-add-norm-activation kernels or explicit primitive chain
```

Preconditions:

- Kernel 2, stride 2, no padding/output padding.
- `feat_s1` and `feat_s0` are already projected by 1x1 convs to 64 and 32 channels.
- NCHW source layout preserved.

Failure cases:

- Channel-last rewrite without changing LayerNorm axis.
- Dynamic feature resolutions not matching 64->128->256 output progression.

### Rewrite: memory fuser ConvNeXt block local fusion

Source pattern:

```text
depthwise Conv2d -> channels_first LayerNorm -> NHWC Linear -> GELU -> Linear -> scale -> NCHW -> residual
```

Replacement:

```text
NCHW depthwise conv + per-pixel MLP fusion, or channel-last local region
```

Preconditions:

- Groups equal channels.
- Pointwise linear is semantically 1x1 over channel dimension after NCHW->NHWC.
- Layout pass rewrites LayerNorm axis exactly.

Failure cases:

- Leaking NHWC layout into memory encoder/session cache.
- Dynamic channel count not equal `memory_fuser_embed_dim`.

### Rewrite: hard mask-input output fast path

Source pattern:

```text
mask_inputs * 20 - 10
bilinear downsample by 4
dummy IoU ones
SAM decoder call only to produce object_pointer
```

Replacement:

```text
direct mask logits + low-res resize + pointer-only decoder subgraph
```

Preconditions:

- `mask_inputs` path selected and no point inputs present.
- Object pointer parity is still required; cannot skip pointer decoder if future frames need memory.

Failure cases:

- Treating this as final-only output and failing to update `object_pointer`/memory state.

## 10. Kernel Fusion Candidates

Highest priority:

- NCHW FPN/cache reshape and memory attention input assembly. This is hot for every tracked frame and heavily layout-sensitive.
- Memory attention RoPE + dense attention. Preserve key-token exclusion/repeat metadata.
- Mask decoder ConvTranspose/add/LayerNorm/GELU and hypernetwork GEMM. This dominates per-object mask generation after vision features are cached.
- Memory encoder mask downsample + fuser + perceiver path. This is required to make tracking state persistent across frames.

Medium priority:

- Prompt encoder point/box embedding and mask embedding conv stack. Important for interactive latency but smaller than frame/memory paths.
- Processor postprocess upsample/crop/binarize on GPU. Avoid CPU roundtrip for high-resolution masks.
- Vision FPN 1x1 projections and top-down adds if the delegated backbone is admitted.

Lower priority:

- Automatic mask generation crop/NMS utilities from SAM2 image processor; not required for first video tracking target.
- FlashAttention path for mask decoder; SDPA/eager is simpler and additive target-guided masks already force fallback.
- Multi-object batching; source loops per object because prompts/history differ.

## 11. Runtime Staging Plan

Stage 1: Parse config and admit only `yonigozlan/EdgeTAM-hf`-style configs: `edgetam_video`, 1024 image size, RepViT M1 delegated backbone, fp32, fixed feature sizes.

Stage 2: Implement/compose preprocessing outside compiled graph: decoded frames -> NCHW 1024 tensors; points/boxes normalized; masks resized.

Stage 3: Bring up stateless single-frame image segmentation path with precomputed image embeddings, prompt encoder, mask decoder, and postprocess.

Stage 4: Add session ABI without optimization: object ids, frame store, point/mask input stores, feature cache, output dictionaries.

Stage 5: Add vision feature cache and `_prepare_vision_features` parity for one frame.

Stage 6: Add memory-conditioned tracking for one object forward direction: gather conditioning memories, run memory attention, update memory encoder.

Stage 7: Add multi-object source loop, reverse propagation, streaming frame addition, and cache eviction policies.

Stage 8: Add optimized kernels/fusions behind exact layout and shape guards.

Stub initially:

- Automatic mask generation crop/NMS.
- Dynamic HF video frame sampling.
- Non-allowlisted delegated backbones.
- Target-guided `attention_similarity`/PerSAM path unless needed by a user.

## 12. Parity and Validation Plan

- Processor parity: one image/video frame, compare `pixel_values`, `original_sizes`, `reshaped_input_sizes`, normalized points/boxes, and padded labels.
- Prompt encoder parity: random/fixture points, boxes, masks; compare sparse/dense embeddings.
- Mask decoder parity: random fixed image embeddings and prompt embeddings; compare low-res masks, IoU scores, object logits, and selected mask token behavior.
- Single-frame end-to-end: one frame plus one point prompt, compare `pred_masks`, `object_score_logits`, and postprocessed masks.
- Session ABI parity: add object id, add points, run frame 0, verify `cond_frame_outputs` keys and stored tensor shapes/devices.
- Memory update parity: run `_encode_new_memory` with prompt-derived and non-prompt-derived masks; verify hard threshold vs sigmoid branch.
- Forward tracking parity: frame 0 conditioning then frame 1 non-conditioning; compare memory attention outputs and session histories.
- Reverse propagation parity: start from later conditioning frame, verify selected previous/next frames and `frames_tracked_per_obj[frame_idx]["reverse"]`.
- Tolerances: start fp32 `rtol=1e-4`, `atol=1e-4` for most tensors; mask logits after multiple attention/interpolate paths may need `atol=2e-4`. fp16/bf16 should be a later acceptance target with mask-threshold sensitivity checks.

## 13. Performance Probes

- CPU/video preprocessing throughput: decode/sample/resize/normalize frames per second.
- Vision encoder/FPN latency per frame at batch 1 and small frame batches.
- Vision feature cache hit vs miss latency.
- Single-object prompt-to-mask latency on cached features.
- Memory-conditioned tracking latency per object per frame.
- Memory encoder update latency and memory tensor write bandwidth.
- Multi-object loop scaling: objects 1, 2, 4, 8, 16.
- `num_maskmem` sweep: 1, 3, 7 with object-pointer tokens enabled/disabled.
- State memory footprint by frame count and object count, split by frame store, feature cache, cond/non-cond outputs.
- Postprocess mask upsample/crop/binarize GPU vs CPU.
- Attention backend comparison for memory attention and mask decoder: eager, SDPA, FlashAttention where masks allow.

## 14. Skip/Defer List

- Training and gradient checkpointing.
- Automatic mask generation crop grids, RLE encoding, and NMS.
- General video decode/fps sampling inside DinoML runtime.
- Non-RepViT/unknown `timm_wrapper` backbones.
- Target-guided PerSAM `attention_similarity` and `target_embedding`.
- Broad NHWC/channel-last translation across persisted session caches.
- Quantization/packed weights.
- Multi-GPU/distributed state sharding.
- General dynamic image sizes; first target should require 1024x1024 model input and source feature sizes.

## 15. Final Implementation Checklist

- [ ] Parse `EdgeTamVideoConfig` and nested prompt/mask/vision configs.
- [ ] Allowlist or separately audit `edgetam_vision_model` + `timm_wrapper` `repvit_m1` backbone.
- [ ] Load weights with generated-file/module-name mapping awareness.
- [ ] Implement processor ABI for NCHW 1024 frames, prompt coordinate normalization, and output mask postprocess.
- [ ] Implement session ABI: frames, object id maps, point/mask stores, feature cache, cond/non-cond output histories.
- [ ] Implement `get_image_features` and FPN cache layout `[HW,B,C]`.
- [ ] Implement prompt encoder parity.
- [ ] Implement two-way mask decoder and dynamic multimask selection.
- [ ] Implement memory attention with 2D RoPE and object-pointer key exclusion/repeat metadata.
- [ ] Implement memory encoder plus spatial perceiver memory token update.
- [ ] Add guarded NCHW/NHWC rewrite tests around FPN flatten, memory fuser, and mask decoder upscaling.
- [ ] Add single-frame segmentation parity test.
- [ ] Add two-frame tracking parity test with state inspection.
- [ ] Add streaming frame and reverse propagation parity tests.
- [ ] Benchmark cached-frame prompt latency, tracking latency, and state memory footprint.
