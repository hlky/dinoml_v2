# SAM2 Video DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/sam2_video.1-hiera-tiny / small / base-plus / large family
Config source: source defaults and conversion script; official raw HF configs returned 401
Source files inspected:
- transformers/src/transformers/models/sam2_video/configuration_sam2_video.py
- transformers/src/transformers/models/sam2_video/modeling_sam2_video.py
- transformers/src/transformers/models/sam2_video/modular_sam2_video.py
- transformers/src/transformers/models/sam2_video/processing_sam2_video.py
- transformers/src/transformers/models/sam2_video/video_processing_sam2_video.py
- transformers/src/transformers/models/sam2_video/convert_sam2_video_to_hf.py
- transformers/src/transformers/models/sam2/configuration_sam2.py
- transformers/src/transformers/models/sam2/modeling_sam2.py
Any missing files or assumptions:
- Official raw configs/processors for [facebook/sam2_video.1-hiera-tiny](https://huggingface.co/facebook/sam2_video.1-hiera-tiny), [small](https://huggingface.co/facebook/sam2_video.1-hiera-small), [base-plus](https://huggingface.co/facebook/sam2_video.1-hiera-base-plus), and [large](https://huggingface.co/facebook/sam2_video.1-hiera-large) returned 401. Access would confirm serialized JSON and any non-default processor fields.
- Generated files identify modular_sam2_video.py as the future-edit source.
- Scope is stateful video object segmentation. Shared SAM2/Hiera image encoder behavior is summarized because sam2_video composes AutoModel(config.vision_config).
```

Snapshots are under `agents/plans/transformers/sam2_video/_sources/`.

## 2. High-level architecture

SAM2 Video is a stateful prompt-conditioned video object segmentation model:

```text
video/image preprocessing -> Hiera vision encoder + FPN neck -> prompt/memory conditioning -> SAM two-way mask decoder -> masks/object pointer -> memory encoder -> session state for later frames
```

CPU/data-pipeline stages decode/resize/rescale/normalize frames, validate prompts, normalize point/box coordinates to the 1024 target square, and manage session dictionaries. GPU/runtime stages compute per-frame vision features, memory-conditioned current-frame features, mask logits, object pointers, and new mask-memory tensors. The frame vision features are independently cacheable by `frame_idx`; mask-memory and object-pointer histories are cacheable per object.

## 3. Important config dimensions

| Field | Value / source | Runtime effect |
|---|---:|---|
| `image_size` | 1024 | processor target and high-res mask size |
| `vision_config.fpn_hidden_size` | 256 | SAM decoder image channel width and object pointer width |
| `vision_config.backbone_feature_sizes` | `[[256,256],[128,128],[64,64]]` | three high-to-low FPN feature maps |
| `prompt_encoder.hidden_size` | 256 | point/box/mask prompt embedding width |
| `prompt_encoder.patch_size` | 16 | 64x64 prompt image grid |
| `prompt_encoder.mask_input_size` | derived 256x256 | dense mask prompt resolution |
| `mask_decoder.num_hidden_layers` | 2 | two-way transformer depth |
| `mask_decoder.num_attention_heads` | 8 | SAM mask-decoder MHA |
| `mask_decoder.attention_downsample_rate` | 2 | token-image attention width 128, head_dim 16 |
| `mask_decoder.mlp_dim` | 2048 | two-way block MLP width |
| `num_maskmem` | 7 | memory slots used by tracking |
| `max_object_pointers_in_encoder` | 16 | maximum pointer history frames |
| `memory_attention_hidden_size` | 256 | memory attention query width |
| `memory_attention_num_layers` | 4 | memory attention depth |
| `memory_attention_num_attention_heads` | 1 | memory attention head_dim 256 |
| `memory_attention_feed_forward_hidden_size` | 2048 | memory attention MLP width |
| `memory_encoder_output_channels` | 64 | stored mask-memory width |
| `memory_attention_rope_feat_sizes` | default `[64,64]` | fixed 4096-token 2D RoPE table |
| `mask_downsampler_total_stride` | 16 | memory mask downsample to 64x64 |
| `memory_fuser_num_layers` | 2 | ConvNeXT-like memory fuser blocks |

Representative variant sweep from source/converter, not raw config JSON:

| Variant | HF link | Raw config | Hiera blocks | Hiera dims | Hiera heads | Global blocks | Window sizes |
|---|---|---|---|---|---|---|---|
| Tiny | [facebook/sam2_video.1-hiera-tiny](https://huggingface.co/facebook/sam2_video.1-hiera-tiny) | 401 | `[1,2,7,2]` | `[96,192,384,768]` | `[1,2,4,8]` | `[5,7,9]` | `[8,4,14,7]` |
| Small | [facebook/sam2_video.1-hiera-small](https://huggingface.co/facebook/sam2_video.1-hiera-small) | 401 | `[1,2,11,2]` | `[96,192,384,768]` | `[1,2,4,8]` | `[7,10,13]` | `[8,4,14,7]` |
| Base-plus | [facebook/sam2_video.1-hiera-base-plus](https://huggingface.co/facebook/sam2_video.1-hiera-base-plus) | 401 | `[2,3,16,3]` | `[112,224,448,896]` | `[2,4,8,16]` | `[12,16,20]` | `[8,4,14,7]` |
| Large | [facebook/sam2_video.1-hiera-large](https://huggingface.co/facebook/sam2_video.1-hiera-large) | 401 | `[2,6,36,4]` | `[144,288,576,1152]` | `[2,4,8,16]` | `[23,33,43]` | `[8,4,16,8]` |

For `sam2.1` conversions the source enables temporal pointer positional encoding and occlusion spatial embedding; older non-2.1 conversions disable both.

## 3a. Family variation traps

- This is not a stateless video batch. `Sam2VideoInferenceSession` owns mutable per-object/per-frame histories.
- Hiera uses NHWC hidden states after the initial NCHW Conv2d, but FPN, masks, memory encoder, and postprocess are NCHW. Treat NHWC as guarded local optimization, not default translation.
- Spatial memory K/V width is 64, while object pointers are 256 and split into four 64-wide tokens when `mem_dim < hidden_dim`.
- Memory cross-attention applies 2D RoPE to spatial memory tokens but excludes appended pointer tokens.
- Mask inputs bypass prompt+mask decoding for the mask logits, but still invoke the SAM decoder to derive an object pointer for future tracking.
- Memory features are deliberately stored as `bfloat16` `[4096,B,64]`; this is a state ABI detail.
- Boxes are transformed into point prompts with labels 2 and 3 and must be inserted before ordinary points.
- `num_maskmem == 0` disables video memory conditioning and should be routed as a special/image-like mode.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image/video tensors, NHWC Hiera activations, NCHW FPN/masks, flattened sequence-first memory tensors `[HW,B,C]`.
- `permute`, `view/reshape`, `flatten`, `transpose`, `unsqueeze/squeeze`, `repeat`, `repeat_interleave`, `expand`, `cat`, `stack`, `gather`, `where`, boolean comparisons, `argmax`.
- Window partition/unpartition with right/bottom padding for local Hiera attention.

Neural network primitives:

- Conv2d patch embed `3 -> C0`, kernel 7, stride 4, padding 3.
- FPN Conv2d 1x1 to 256 plus nearest upsample/add.
- LayerNorm on NHWC and channels-first LayerNorm through NCHW/NHWC permutation.
- Linear QKV/O projections and MLPs with ReLU/GELU/sigmoid.
- ConvTranspose2d upscalers `256 -> 64 -> 32`, kernel/stride 2.
- Mask prompt convs `1 -> 4 -> 16 -> 256`; memory mask downsampler `1 -> 4 -> 16 -> 64 -> 256`.
- Memory fuser depthwise Conv2d 7x7 groups 256, Linear `256 -> 1024 -> 256`, layer-scale, residual.
- Bilinear/nearest/bicubic interpolation, max_pool2d, fp32 softmax, sigmoid, threshold.

Attention primitives:

- Hiera local/global noncausal MHA with packed QKV split `[q,k,v]`.
- SAM two-way dense self-attention plus rectangular token-image cross-attention.
- Memory attention self-attention over current 4096 tokens and cross-attention to prior spatial memory plus pointer tokens with 2D RoPE.
- Backend dispatch via `ALL_ATTENTION_FUNCTIONS`; eager dense fallback is likely too slow for global Hiera and memory attention.

Position/cache/state ops:

- Learned Hiera background/window position, FPN sine 2D position, prompt random Fourier coordinates, memory 2D RoPE, object-pointer 1D temporal sine PE.
- No autoregressive KV cache. Required state is processed frames, feature cache, prompt inputs, cond/non-cond outputs, mask memories, object pointers, tracked direction, and object-id maps.

Preprocessing/postprocess ops:

- Video resize/rescale/normalize/RGB conversion to `pixel_values_videos`; point/box coordinate scaling; point padding label `-10`; box-to-corner-point packing.
- Mask postprocess upsample to pad size, crop to reshaped size, upsample to original size, optional threshold.

## 5. Layer/block breakdown

Hiera vision encoder:

```text
pixel_values [B,3,1024,1024]
  -> Conv2d(3 -> C0, k7 s4 p3) -> NHWC [B,256,256,C0]
  -> learned bicubic/tiled position add
  -> repeated Hiera blocks:
       x_norm = LayerNorm(x)
       if local: x_win = window_partition(x_norm)
       q,k,v = Linear(dim -> 3*dim_out, bias=True)
       if stage transition: q = max_pool2d(q, stride=2)
       y = noncausal attention(q,k,v)
       if local: y = window_unpartition(y)
       x = residual/proj/pool + Linear(dim_out -> dim_out)
       x = x + MLP(dim_out -> 4*dim_out -> dim_out)
  -> FPN Conv2d/nearest/add -> three [B,256,H,W] maps
```

Prompt encoder:

```text
points/boxes -> +0.5 pixel-center shift
  -> random Fourier coordinate embedding 256
  -> label embeddings for pos/neg/box corners; -1 not-a-point; -10 zero
masks -> Conv2d(1->4,k2s2) -> LN/GELU -> Conv2d(4->16,k2s2) -> LN/GELU -> Conv2d(16->256,k1)
no mask -> learned [1,256,1,1] expanded to [B,256,64,64]
```

Memory conditioning:

```text
current lowest feature [4096,B,256]
initial conditioning frame:
  add no_memory_embedding -> reshape [B,256,64,64]
tracked frame:
  gather cond/non-cond maskmem_features [4096,B,64]
  add temporal memory embedding by relative offset
  gather object pointers [T,B,256]
  temporal PE -> Linear(256 -> 64)
  split pointers to [4T,B,64]
  cat spatial memory + pointer tokens
  memory attention x4: LN -> RoPE self-attn -> LN -> RoPE cross-attn -> LN -> MLP(256->2048->256)
  reshape [B,256,64,64]
```

Mask decoder:

```text
image [B,256,64,64] + dense prompt
tokens = obj_score + iou + 4 mask tokens + sparse prompts
two-way transformer x2: token self-attn, token->image cross-attn, MLP, image->token cross-attn
final token->image cross-attn
ConvTranspose2d upsample with high-res skips -> [B*P,32,256,256]
per-mask hypernet MLP(256->256->256->32)
masks = hypernet @ upscaled_embedding -> [B,P,M,256,256]
iou head sigmoid; object-score head; object_pointer = MLP(mask_token)
```

Memory encoder:

```text
high_res_mask logits -> resize to 1024x1024 if needed
if point/mask prompt in eval: threshold > 0 else sigmoid
mask = mask * 20 - 10
mask downsample total stride 16 -> [B,256,64,64]
pix_feat [B,256,64,64] -> Conv2d(256->256,k1)
add mask, memory fuser x2, Conv2d(256->64,k1)
store bf16 features [4096,B,64] and position [4096,B,64]
```

## 6. Attention requirements

Hiera attention is noncausal MHA. Most blocks are local-window attention with padding and unpartition; configured blocks are global. Stage-transition blocks pool only Q, creating rectangular attention. There is no cache.

SAM mask-decoder attention is noncausal dense attention over prompt/output tokens plus rectangular cross-attention to image tokens. `hidden_size=256`; downsampled token-image projections use internal width 128 with 8 heads. Optional additive `attention_similarity` is incompatible with FlashAttention in source.

Memory attention is noncausal. Queries are current 64x64 image tokens with width 256. Spatial memory K/V input width is 64. Pointer tokens are appended after spatial memories, have width 64 after split/projection, and are excluded from key RoPE. Default per-object K length can be several `4096` spatial-memory frames plus up to `16*4` pointer tokens. No autoregressive KV cache is present.

## 7. Position encoding and custom math

Memory RoPE:

```python
def sam2_video_rope(q, k, cos, sin, num_k_exclude_rope):
    k_rot = k[..., : k.shape[-2] - num_k_exclude_rope, :]
    k_pass = k[..., k.shape[-2] - num_k_exclude_rope :, :]
    q = q.float() * cos + rotate_pairwise(q.float()) * sin
    k_rot = k_rot.float() * cos + rotate_pairwise(k_rot.float()) * sin
    return q.to(k.dtype), torch.cat([k_rot.to(k.dtype), k_pass], dim=-2)
```

The cos/sin table is precomputed for `[64,64]`; pointer exclusion is dynamic. Prompt coordinates use a learned/random `2 x 128` matrix, map normalized coordinates to `[-1,1]`, multiply by `2*pi`, then concat sin/cos. Dynamic multimask selection computes stability from areas above `+delta` and `-delta`, then switches to the best IoU multimask token when unstable.

## 8. Preprocessing and input packing

- Video defaults: resize 1024x1024, rescale, normalize ImageNet mean/std, convert RGB. `_preprocess` records `original_sizes` and `(1024,1024)` `reshaped_input_sizes`.
- `init_video_session(video=...)` stores processed frames in a dictionary; streaming can add a processed `frame` directly.
- Points are `[image, object, point, xy]`, labels `[image, object, point]`, boxes `[image, box, xyxy]`.
- Coordinates are scaled from original H/W to the target 1024 square; the prompt encoder later divides by image size for Fourier features.
- Points are padded with `-10`; boxes cannot be padded across images.
- In the session helper, boxes become two point prompts with labels `[2,3]` and are concatenated before existing points.
- Mask prompts are exclusive with points/boxes; resized masks are binarized at 0.5 before session storage.
- Model `forward` returns model-resolution masks; processor postprocess handles original-size resizing and optional thresholding.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Hiera patch Conv2d -> im2col/GEMM

Preconditions: kernel 7, stride 4, padding 3, dilation 1, groups 1, NCHW input, output immediately consumed as NHWC. Replacement: `PadNCHW -> WindowFlatten -> MatMul(weight_flat.T) -> BiasAdd -> NHWC view`. Failure cases are non-default patch config or consumers requiring NCHW materialization.

### Rewrite: local Hiera window attention fusion

Source pattern: `NHWC -> pad -> window_partition -> packed QKV -> attention -> window_unpartition`. Replacement can fuse partition/QKV/attention/unpartition for static H/W and window size when output attentions are not requested. Guard stage-transition Q pooling separately.

### Rewrite: FPN 1x1 Conv2d -> channel GEMM

Preconditions: kernel 1, stride 1, padding 0, groups 1. This can stay NCHW or become an NHWC island only if top-down upsample/add and position encoding axes are also rewritten. Otherwise protect with `no_layout_translation()`.

### Rewrite: memory mask downsampler/fuser

Source pattern: four stride-2 Conv2d+channels-first-LN+GELU layers, final 1x1 Conv, then depthwise 7x7 ConvNeXT-like fuser. Candidate: fused NHWC conv/LN/GELU region, guarded by fixed mask resolution and channel-axis rewrites.

### Rewrite: object pointer split as view

Source pattern: `[T,B,256] -> [T,B,4,64] -> [4T,B,64]`. Use metadata-only reshape/permute when contiguous and `hidden_dim % mem_dim == 0`.

## 10. Kernel fusion candidates

Highest priority:

- Memory attention with RoPE and pointer exclusion; K length grows with history and eager dense fallback is expensive.
- Hiera local/global attention with Q pooling.
- Mask decoder two-way attention and hypernetwork matmul.
- Memory encoder conv/LN/GELU/fuser after each tracked frame/object.

Medium priority:

- NHWC Hiera blocks and window partition elimination.
- FPN 1x1 conv + nearest upsample/add.
- ConvTranspose2d + high-res skip add + GELU in mask decoder.
- GPU postprocess resize/crop/threshold.

Lower priority:

- CPU nested prompt validation/padding.
- Dynamic multimask stability reductions/gather.
- Session dictionary management.

## 11. Runtime staging plan

1. Parse nested config and load weights, rejecting unaudited non-default memory widths.
2. Reuse or implement SAM2 Hiera+FPN one-frame parity.
3. Implement `_single_frame_forward` with prompt encoder and mask decoder, no memory history.
4. Implement `Sam2VideoInferenceSession` and frame feature cache.
5. Implement memory encoder and bf16 stored memory tensors.
6. Implement memory-frame selection, object pointer history, temporal PE, pointer split, and memory attention.
7. Add streaming and forward/reverse propagation.
8. Add guarded NHWC/layout and attention fusions.

Initial stubs: training, output attentions, progress bar, CPU video decode, hole/sprinkle cleanup.

## 12. Parity and validation plan

- Random op tests for window partition/unpartition, Q pooling, prompt coordinate embedding, 2D RoPE, pointer temporal PE/split, dynamic multimask selection.
- Single Hiera local/global/stage-transition block parity.
- FPN parity and prompt encoder parity for points, boxes, padding, masks, and no-mask embedding.
- Mask decoder parity for multimask, single-mask, and mask-input direct path.
- Memory encoder parity for thresholded versus sigmoid masks and bf16 memory storage.
- Stateful parity: one object one conditioning frame; one object multi-frame; multi-object; reverse propagation; streaming frame add.
- End-to-end video mask parity after postprocess.
- Suggested tolerances: fp32 `rtol=1e-4/atol=1e-5`; fp16/bf16 memory/attention `rtol=5e-3/atol=5e-3`.

## 13. Performance probes

- Video preprocessing frames/sec and transfer cost.
- Vision encoder frame throughput and feature-cache hit/miss cost.
- Prompt-only interaction latency on cached frames.
- Memory attention latency versus number of memory frames, object pointers, and objects.
- Memory encoder throughput for batched object memories.
- Propagation frames/sec for 1/4/16 objects and short/long videos.
- Peak state memory for processed frames, cached features, bf16 mask memories, pointers, and masks.
- Eager/SDPA/Flash comparison for Hiera, mask decoder, and memory attention.
- Source-layout versus guarded NHWC islands.
- Postprocess resize/threshold throughput at original video resolutions.

## 14. Skip/defer list

- Training, gradients, dropout, gradient checkpointing.
- Output attentions and hidden-state recording.
- Hole filling, sprinkle removal, non-overlap constraints unless product parity requires them.
- Older non-2.1 converted checkpoints until configs are accessible.
- Distributed execution, quantization, packed weights.
- Runtime video decoding; keep decode in data pipeline first.

## 15. Final implementation checklist

- [ ] Parse `Sam2VideoConfig` and nested vision/backbone configs.
- [ ] Load weights and preserve conversion aliases.
- [ ] Compose/implement Hiera vision encoder + FPN.
- [ ] Implement prompt encoder and mask decoder.
- [ ] Implement session state ABI.
- [ ] Implement per-frame vision feature cache.
- [ ] Implement memory encoder and bf16 memory storage.
- [ ] Implement memory frame selection and object pointer temporal PE/splitting.
- [ ] Implement memory attention with 2D RoPE and pointer exclusion.
- [ ] Implement video processor/postprocess contracts.
- [ ] Add guarded NHWC/layout fusion candidates.
- [ ] Add single-frame, memory-update, and propagation parity tests.
- [ ] Benchmark encoder, memory attention, memory encoder, and postprocess separately.
