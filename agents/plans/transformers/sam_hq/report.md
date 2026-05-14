# SAM-HQ Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: syscv-community/sam-hq-vit-base, syscv-community/sam-hq-vit-large, syscv-community/sam-hq-vit-huge
Config source: HF config.json and preprocessor_config.json snapshots under _sources/
Primary inference task: promptable image segmentation / mask generation
DinoML target assumption: inference-only CUDA first, faithful NCHW/NHWC axes first, channel-last as guarded optimization only.
```

Source files inspected:

- Transformers generated modeling: [`modeling_sam_hq.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/sam_hq/modeling_sam_hq.py)
- Transformers modular source: [`modular_sam_hq.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/sam_hq/modular_sam_hq.py)
- Transformers config: [`configuration_sam_hq.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/sam_hq/configuration_sam_hq.py)
- Transformers processor: [`processing_sam_hq.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/sam_hq/processing_sam_hq.py)
- Shared SAM image processor/postprocess: [`image_processing_sam.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/sam/image_processing_sam.py)

Checkpoint configs inspected:

- [`syscv-community/sam-hq-vit-base`](https://huggingface.co/syscv-community/sam-hq-vit-base)
- [`syscv-community/sam-hq-vit-large`](https://huggingface.co/syscv-community/sam-hq-vit-large)
- [`syscv-community/sam-hq-vit-huge`](https://huggingface.co/syscv-community/sam-hq-vit-huge)

Missing files or assumptions: no gated config was encountered. The modeling/config files are generated from `modular_sam_hq.py`; upstream edits should treat modular source as authoritative, but this report verifies behavior against generated source. No model execution, imports, or DinoML tests were run.

## 2. High-level architecture

SAM-HQ is a prompt-conditioned segmentation model, not a language generation model.

```text
CPU image/prompt preprocessing
  -> ViT image encoder
  -> cacheable image embeddings + intermediate global-layer embeddings
  -> prompt encoder for points/boxes/masks
  -> two-way mask decoder with HQ output token
  -> low-resolution mask logits + IoU scores
  -> mask postprocess: upsample, crop padding, upsample to original image, optional threshold
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize longest edge to 1024, rescale, normalize, pad to 1024x1024, normalize point/box coordinates, track `original_sizes` and `reshaped_input_sizes`.
- Cacheable image stage: `SamHQModel.get_image_embeddings(pixel_values)` returns `(image_embeddings, intermediate_embeddings)`. For full HQ parity, both are cacheable and both must be supplied when bypassing `pixel_values`.
- Prompt stage: point, label, box, and optional mask prompts produce sparse prompt tokens and dense `[B,256,64,64]` prompt embeddings.
- Decoder stage: prompt tokens attend to image tokens and image tokens attend back to prompt tokens; HQ token path fuses upscaled decoder features with global/local ViT features.
- Postprocess stage: processor-owned ABI converts low-res logits to original image masks.

## 3. Important config dimensions

Common source defaults and official config values:

| Field | Base | Large | Huge | Source / impact |
|---|---:|---:|---:|---|
| `vision_config.image_size` | 1024 | 1024 | 1024 | Config; patch input guard requires exact H/W. |
| `vision_config.patch_size` | 16 | 16 | 16 | Config; patch grid is 64x64. |
| `vision_config.hidden_size` | 768 | 1024 | 1280 | Config; ViT token width and HQ feature compression input. |
| `vision_config.output_channels` | 256 | 256 | 256 | Config; final image embedding channels. |
| `vision_config.num_hidden_layers` | 12 | 24 | 32 | Config; ViT depth. |
| `vision_config.num_attention_heads` | 12 | 16 | 16 | Config; ViT head dim is 64/64/80. |
| `vision_config.mlp_dim` | 3072 | 4096 | 5120 | Config; source default is `hidden_size * mlp_ratio`. |
| `vision_config.window_size` | 14 | 14 | 14 | Config; non-global layers use padded local windows. |
| `vision_config.global_attn_indexes` | `[2,5,8,11]` | `[5,11,17,23]` | `[7,15,23,31]` | Config; also selects collected intermediate embeddings. |
| `vision_config.qkv_bias` | true | true | true | Config; packed QKV Linear has bias. |
| `vision_config.use_abs_pos` | true | true | true | Config; absolute `[1,64,64,H]` pos table. |
| `vision_config.use_rel_pos` | true | true | true | Config; decomposed relative pos in vision attention. |
| `mask_decoder_config.hidden_size` | 256 | 256 | 256 | Config; prompt/mask decoder width. |
| `mask_decoder_config.num_hidden_layers` | 2 | 2 | 2 | Config; two-way transformer depth. |
| `mask_decoder_config.num_attention_heads` | 8 | 8 | 8 | Config; decoder head dim is 32 or 16 after downsample. |
| `mask_decoder_config.attention_downsample_rate` | 2 | 2 | 2 | Config; cross attention internal dim is 128. |
| `mask_decoder_config.num_multimask_outputs` | 3 | 3 | 3 | Config; SAM mask tokens before HQ token are 4. |
| `mask_decoder_config.vit_dim` | 768 | 1024 | 1280 | Config; must match selected ViT hidden width. |
| `prompt_encoder_config.mask_input_channels` | 16 | 16 | 16 | Config; mask prompt downsampling channels. |
| `torch_dtype` | float32 | float32 | float32 | HF config metadata. |

Preprocessor snapshots are identical across the three official configs: `SamHQProcessor` with `SamImageProcessor`, resize longest edge 1024, pad to 1024x1024, image mean/std ImageNet values, `mask_size.longest_edge=256`, and `mask_pad_size=256x256`.

## 3a. Family variation traps

- `mask_decoder_config.vit_dim` must match `vision_config.hidden_size`; the HQ compression path has ConvTranspose2d input channels equal to `vit_dim`.
- `vision_config.global_attn_indexes` changes by backbone and controls both global attention placement and which intermediate embeddings are collected for HQ fusion.
- The source uses NCHW `pixel_values`, then NHWC ViT hidden states, then NCHW neck/image embeddings. A naive all-NHWC or all-NCHW lowering will break axis-sensitive `LayerNorm`, `permute`, window partition, and Conv2d/ConvTranspose2d boundaries.
- Vision attention source supports eager and SDPA. `output_attentions=True` forces eager-like behavior for attention weights; first DinoML target can reject attention weight outputs.
- `multimask_output=True` sorts IoU scores and gathers SAM masks before adding the HQ mask. `multimask_output=False` uses only the first SAM mask plus HQ mask.
- `hq_token_only=True` returns only the HQ token mask, bypassing SAM+HQ addition.
- `attention_similarity` and `target_embedding` support PerSAM-style personalization; they are optional but source-visible and should be gated for a first integration.
- If users pass precomputed `image_embeddings`, full SAM-HQ quality requires `intermediate_embeddings`; otherwise the decoder falls back to final-image-embedding HQ features only.
- Official checkpoints are open. No gated/remote-code-only config gap was found in the `syscv-community` sweep.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input validation, exact static image H/W guard.
- Conv2d non-overlap patch embedding: `[B,3,1024,1024] -> [B,Hv,64,64] -> permute [B,64,64,Hv]`.
- `permute`, `reshape`, `flatten`, `unsqueeze`, `repeat`, `repeat_interleave`, `expand`, `cat`, `stack`, `slice`, `gather`, `sort`.
- Local window partition/unpartition on NHWC hidden states with `F.pad`, reshape, permute, contiguous, and crop.
- Channels-first LayerNorm implemented as NCHW -> NHWC LayerNorm -> NCHW.

Neural primitives:

- Linear packed QKV: `Linear(Hv -> 3*Hv, bias=True)` in vision encoder.
- Linear projection: `Linear(Hv -> Hv)`.
- ViT MLP: `Linear(Hv -> 4*Hv) -> GELU -> Linear(4*Hv -> Hv)`.
- Neck: Conv2d `Hv -> 256` kernel 1, channels-first LayerNorm, Conv2d `256 -> 256` kernel 3 padding 1, channels-first LayerNorm.
- Prompt mask embedding: Conv2d `1 -> 4` stride 2, Conv2d `4 -> 16` stride 2, Conv2d `16 -> 256` kernel 1, LayerNorm/GELU after first two convs.
- Mask decoder transposed convs: `256 -> 64 -> 32` upscaling final embeddings to 256x256.
- HQ compression: ConvTranspose2d `vit_dim -> 256 -> 32`.
- HQ encoder path: ConvTranspose2d `256 -> 64 -> 32`.
- HQ mask refinement: Conv2d `32 -> 64 -> 32` with kernel 3 padding 1.
- Hypernetwork MLPs: per-mask `FeedForward(256 -> 256 -> 32)` with ReLU; HQ token has its own MLP.
- IoU head: `FeedForward(256 -> 256 -> 5)` for 4 SAM mask tokens plus 1 HQ token internally, then slices to selected SAM scores.

Attention primitives:

- Vision encoder self-attention, noncausal MHA over either 14x14 local windows or full 64x64 image grid.
- Decoder two-way attention, noncausal, query-driven:
  - sparse prompt self-attention,
  - sparse prompt tokens attending to dense image tokens,
  - image tokens attending back to sparse tokens,
  - final token-to-image attention.
- Optional `attention_similarity` additive attention mask/bias in token-to-image cross attention.

Position/custom math:

- Absolute ViT positional table add on NHWC tokens.
- Decomposed 2D relative position bias for vision attention.
- Random Fourier-style prompt positional embedding using learned `[2,num_pos_feats]` matrix, sine/cosine concat.
- Image-wide positional embedding generated from a 64x64 normalized coordinate grid.

Preprocessing-coupled ops:

- Resize longest edge, normalize and pad image.
- Normalize point/box coordinates from original image space to resized image space.
- Pad point prompts and labels with `point_pad_value`; source uses `-10` to zero out padded points in model.
- Postprocess masks with bilinear interpolate, crop to resized shape, bilinear interpolate to original shape, and optional threshold.

## 5. Layer/block breakdown

Vision encoder:

```text
pixel_values: [B,3,1024,1024] NCHW
patch_embed = Conv2d(3 -> Hv, kernel=16, stride=16)
x = patch_embed.permute(0,2,3,1)              # [B,64,64,Hv] NHWC
x = x + abs_pos[1,64,64,Hv]
repeat ViT layer N times:
  residual = x
  x = LayerNorm(Hv)(x)                         # channels-last LN
  if local layer:
    x_windows = window_partition(x, 14)        # pads 64x64 to 70x70; windows [B*25,14,14,Hv]
    y = MHA_with_rel_pos(x_windows)
    y = window_unpartition(y)[:, :64, :64, :]
  else:
    y = MHA_with_rel_pos(x)                    # full 4096-token attention
    collect y as intermediate embedding
  x = residual + y
  x = x + MLP(LayerNorm(x))
neck:
  x = x.permute(0,3,1,2)                       # [B,Hv,64,64]
  x = Conv2d(Hv -> 256, 1x1, no bias)
  x = channels-first LayerNorm(256)
  x = Conv2d(256 -> 256, 3x3, pad=1, no bias)
  image_embeddings = channels-first LayerNorm(256)
```

Prompt encoder:

```text
points: [B,P,N,2], labels: [B,P,N]
boxes: [B,NB,4]
input_masks: [B,1?,256,256] expected by Conv2d path

point_embedding = FourierPos(points + 0.5, input_shape=(1024,1024))
point_embedding = where(label == -1, not_a_point_embed, point_embedding)
point_embedding = where(label == -10, 0, point_embedding)
point_embedding += foreground/background point embedding for labels 1/0

box_embedding = FourierPos(reshape(boxes + 0.5, [B,NB,2,2]))
box corners add point_embed[2] and point_embed[3]

dense_embedding = mask_embed(input_masks) or no_mask_embed.expand([B,256,64,64])
sparse_embeddings = cat(point_embedding, box_embedding, dim=2) when both exist
```

Mask decoder/HQ path:

```text
image_embeddings: [B,256,64,64]
intermediate_embeddings[0]: [B,64,64,vit_dim] when available
sparse_prompt_embeddings: [B,P,S,256]
dense_prompt_embeddings: [B,256,64,64]

hq_features = encoder_conv2(GELU(LN(encoder_conv1(image_embeddings))))
if intermediate_embeddings:
  vit_features = intermediate_embeddings[0].permute(0,3,1,2)
  hq_features += compress_vit_conv2(GELU(LN(compress_vit_conv1(vit_features))))

tokens = cat([iou_token, mask_tokens(4), hq_token, sparse_prompts], dim=2)
image_embeddings = repeat_interleave(image_embeddings + dense_prompt_embeddings, P, dim=0)
point_embedding, image_tokens = two_way_transformer(tokens, image_embeddings, image_pos)
mask_tokens_out = point_embedding[:, :, 1:6, :]

upscaled = ConvT(256->64,stride=2) -> LN -> GELU -> ConvT(64->32,stride=2) -> GELU
upscaled_hq = Conv2d(32->64) -> LN -> GELU -> Conv2d(64->32)
upscaled_hq += repeated hq_features
hyper = stack(mask_mlp_i(mask_tokens_out_i), dim=2)      # [B,P,5,32]
masks_sam = hyper[:,:,:4] @ upscaled.reshape([B,P,32,256*256])
masks_hq = hyper[:,:,4:] @ upscaled_hq.reshape([B,P,32,256*256])
select/sort SAM masks by multimask flag
output = masks_hq if hq_token_only else selected_sam_masks + masks_hq
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- MHA only; no GQA/MQA.
- Base: 12 heads, head dim 64. Large: 16 heads, head dim 64. Huge: 16 heads, head dim 80.
- QKV is one packed `Linear(Hv -> 3*Hv)` with split order Q, K, V after reshape/permutation.
- Local layers attend over 14x14 windows with padded window partition; global layers attend over 4096 image tokens.
- Relative position bias is added before softmax. Eager path upcasts softmax to fp32 then casts back to query dtype.
- SDPA path can consume the decomposed relative position tensor as `attn_mask`, but cannot return attention weights.
- No autoregressive KV cache. The cacheable state is image embeddings/intermediate embeddings for repeated prompts on one image.

Mask decoder attention:

- Noncausal two-way transformer over sparse prompt/output tokens and dense image tokens.
- Decoder hidden size 256, 8 heads.
- Self-attention downsample rate 1 gives internal dim 256, head dim 32.
- Cross-attention downsample rate 2 gives internal dim 128, head dim 16.
- Query/key/value projections are separate Linear layers with bias by PyTorch default.
- No packed/varlen sequence metadata in source; prompt count and image token count are ordinary dense dimensions.
- Optional `attention_similarity` is an additive mask/bias for token-to-image attention only.

## 7. Position encoding and custom math

Prompt/image-wide positional embedding:

```python
def sam_hq_fourier_pos(coords, input_shape, pos_weight):
    # coords final dim is [x, y]
    coords = coords.clone()
    coords[..., 0] = coords[..., 0] / input_shape[1]
    coords[..., 1] = coords[..., 1] / input_shape[0]
    coords = 2 * coords - 1
    phase = 2 * pi * (coords @ pos_weight)      # pos_weight: [2, num_pos_feats]
    return cat([sin(phase), cos(phase)], dim=-1)
```

Vision relative position bias:

```python
def rel_pos_1d(q_size, k_size, table):
    max_rel = 2 * max(q_size, k_size) - 1
    table = linear_interpolate(table, length=max_rel)
    q = arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k = arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    idx = (q - k) + (k_size - 1) * max(q_size / k_size, 1.0)
    return table[idx.long()]
```

The 2D bias is decomposed by einsum over NHWC query reshaped to `[B,H,W,C]` and added to attention logits. For fixed official input sizes, relative tables and index maps can be precomputed per attention shape: 14x14 local and 64x64 global for each backbone.

## 8. Preprocessing and input packing

Processor contract:

- Input images are converted to RGB, resized so the longest edge is 1024, rescaled by `1/255`, normalized by ImageNet mean/std, then padded to `[3,1024,1024]`.
- `original_sizes` is the source image `(height,width)`.
- `reshaped_input_sizes` is the resized pre-pad `(height,width)`.
- Points use user-space `[x,y]` coordinates and processor scales x by `new_w/old_w`, y by `new_h/old_h`.
- Boxes use `[x1,y1,x2,y2]`, internally reshaped to two points, scaled the same way, then restored to `[4]`.
- Points become `[B, point_batch_size, num_points, 2]`; labels become `[B, point_batch_size, num_points]`; boxes become `[B, num_boxes, 4]`.
- Padding points use label `-10`; model zeros the positional embedding for those entries.

Postprocess ABI:

```text
inputs:
  pred_masks: list/tensor with per-image masks shaped [num_prompt_or_batch, num_masks, 256, 256]
  original_sizes: list[(H_orig,W_orig)]
  reshaped_input_sizes: list[(H_resized,W_resized)]
  pad_size: default 1024x1024

steps:
  bilinear interpolate logits to 1024x1024, align_corners=False
  crop [..., :H_resized, :W_resized]
  bilinear interpolate to original_size, align_corners=False
  optionally binarize with mask > mask_threshold
```

Automatic mask generation crop helpers live in the shared SAM image processor. They generate crop boxes, point grids, mask boxes, stability scores, crop-edge filtering, padding back to full image, and NMS across crops. This is postprocessing/control-flow surface, not required for first prompted-mask parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d to Linear

Source pattern: `Conv2d(3, Hv, kernel_size=16, stride=16, padding=0)` on exact `[B,3,1024,1024]`.

Replacement:

```text
NCHW image -> patch flatten [B,64,64,3*16*16] -> Linear(768 -> Hv) -> NHWC [B,64,64,Hv]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == dilation == 0/1`, `groups == 1`.
- input H/W exactly config image size and divisible by patch size.
- Weight transform: `conv.weight.reshape(Hv, 3*16*16)` with PyTorch Conv2d flatten order.

Failure cases: non-official patch sizes, non-NCHW input, dynamic image sizes, or nonzero padding.

Parity sketch: compare Conv2d output after NHWC permute against flattened GEMM for random fp32/fp16 tensors.

### Rewrite: channels-first LayerNorm

Source pattern: NCHW -> permute NHWC -> LayerNorm(C) -> permute NCHW.

Replacement: native channels-first LayerNorm over channel axis.

Preconditions:

- Normalized shape equals channel count.
- No consumer observes the intermediate NHWC view.
- Epsilon and affine weights/bias preserved.

Failure cases: layout pass crossing Conv2d/ConvTranspose2d boundaries without rewriting axes.

### Rewrite: fixed 14x14 window attention

Source pattern: NHWC pad to 70x70, partition into 25 windows of 14x14, dense attention per window, unpartition and crop.

Replacement: windowed attention kernel or batched dense attention over `[B*25, heads, 196, head_dim]`.

Preconditions:

- official 64x64 patch grid and `window_size=14`.
- relative position tables correspond to 14x14 windows.
- pad/crop behavior preserved.

Failure cases: non-1024 image size, changed patch size, output attentions required, or custom relative-position settings.

### Rewrite: low-res mask hypernetwork matmul

Source pattern: per-mask `hyper_in [B,P,M,32] @ upscaled [B,P,32,65536]`.

Replacement: batched GEMM/BMM with `M` small and `N=65536`, or fused 1x1 dynamic-filter application.

Preconditions:

- upscaled feature channel count is 32.
- selected mask count is small and fixed by `num_multimask_outputs`.
- preserve separate SAM and HQ paths before final mask addition.

Failure cases: exposing arbitrary prompt batches without bounded temporary planning.

## 10. Kernel fusion candidates

Highest priority:

- Vision LayerNorm + packed QKV + relative-position attention for both local and global blocks. This dominates encoder cost and has fixed official grids.
- Window partition + local attention + unpartition as one layout-aware region. Avoid materializing padded NHWC windows when possible.
- ConvTranspose/LN/GELU upscaling chains in the mask decoder and HQ feature path. These are small but sit on every prompt decode.
- Postprocess bilinear upsample + crop + threshold. End-to-end segmentation parity depends on exact `align_corners=False` and crop ABI.

Medium priority:

- Prompt Fourier positional embedding with label-conditioned `where` operations. Useful for low-latency interactive prompting.
- Hypernetwork mask matmul as batched GEMM or dynamic 1x1 convolution.
- Channels-first LayerNorm primitive for NCHW Conv/ConvTranspose regions.

Lower priority:

- Automatic mask generation crop/NMS pipeline. Valuable for full pipeline parity but mostly processor/control-flow work.
- Attention weight outputs. They are not needed for normal mask inference.
- PerSAM `attention_similarity`/`target_embedding` path.

## 11. Runtime staging plan

Stage 1: parse config and load official base weights, with strict admission for 1024 image size, patch size 16, `use_abs_pos=True`, `use_rel_pos=True`, and known backbone widths.

Stage 2: implement and validate the ViT image encoder only, returning both final `[B,256,64,64]` image embeddings and global-layer intermediate NHWC embeddings.

Stage 3: implement prompt encoder for points and boxes first; defer input mask prompts if necessary.

Stage 4: implement mask decoder with standard SAM masks plus HQ token path, including `multimask_output` and `hq_token_only`.

Stage 5: wire processor-compatible postprocess for prompted segmentation.

Stage 6: add cache API for `(image_embeddings, intermediate_embeddings)` so repeated prompts avoid rerunning the ViT encoder.

Stage 7: add optimized local/global attention kernels and mask upsample fusions.

Stub candidates for first pass: automatic crop mask generation, attention outputs, PerSAM optional tensors, training labels/losses.

## 12. Parity and validation plan

- Processor parity: fixed PIL/numpy image -> `pixel_values`, `original_sizes`, `reshaped_input_sizes`; point and box coordinate scaling tests.
- Patch embedding rewrite parity: Conv2d vs flattened Linear for each backbone hidden width.
- Vision one-layer parity: local-window and global-attention layers separately, fp32 tolerance around `1e-4`.
- Full image encoder parity: compare final NCHW image embeddings and collected intermediate NHWC embeddings.
- Prompt encoder parity: points with labels `1`, `0`, `-1`, `-10`; boxes; no-mask dense embedding.
- Mask decoder parity: cached image embeddings plus prompts, test both `multimask_output=True/False` and `hq_token_only=True/False`.
- Postprocess parity: low-res masks through upsample/crop/upsample/threshold for multiple aspect ratios.
- End-to-end prompted mask parity against Transformers for base first, then large/huge. Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16 after fusion `rtol=5e-3, atol=5e-3` around mask logits, plus binary mask equality after threshold where stable.

## 13. Performance probes

- Preprocessing throughput by image size and aspect ratio.
- Image encoder throughput for base/large/huge at batch sizes 1, 2, 4.
- Local-window attention vs global attention timing, separated by layer type.
- Prompt decoder throughput with cached image embeddings for varying point batch size `P` and prompt token count.
- HQ path cost: decoder with and without intermediate embeddings, and `hq_token_only` vs SAM+HQ.
- Postprocess throughput for bilinear upsample/crop/threshold at common original image sizes.
- Cache memory: final image embeddings `[B,256,64,64]` plus four intermediate `[B,64,64,vit_dim]` tensors for official checkpoints.
- Automatic mask generation control-flow benchmark only after prompted path is stable.

## 14. Skip/defer list

- Training, labels, losses, and gradient checkpointing.
- Automatic crop-based mask generation and crop NMS for the first prompted-mask target.
- Attention weight output parity.
- PerSAM `attention_similarity` and `target_embedding` unless a downstream integration requires personalization.
- Non-official image sizes, patch sizes, disabled relative/absolute position configs, and remote-code variants.
- General NHWC conversion across the full model; only guarded local layout rewrites should be admitted.
- Quantized/packed weights; official configs inspected are float32 safetensors metadata.

## 15. Final implementation checklist

- [ ] Parse `SamHQConfig` with nested vision, prompt encoder, and mask decoder configs.
- [ ] Reject unsupported config combinations: non-1024 image size, non-16 patch size, mismatched `vit_dim`, unknown attention implementation, and missing HQ fields.
- [ ] Load tied positional embedding alias correctly for `prompt_encoder.shared_embedding` and `shared_image_embedding`.
- [ ] Implement NCHW image preprocessing ABI or require processor-produced `pixel_values`.
- [ ] Implement ViT patch embedding, absolute position add, local/global relative-position attention, MLP, and neck.
- [ ] Return cacheable final image embeddings and intermediate global-layer embeddings.
- [ ] Implement prompt positional embedding, point/label/box embedding, no-mask dense embedding, and optional mask prompt embedding.
- [ ] Implement two-way mask decoder attention and HQ token path.
- [ ] Implement mask selection/sort/gather and `hq_token_only` behavior.
- [ ] Implement postprocess mask upsample/crop/upsample/threshold ABI.
- [ ] Add parity tests for base encoder, prompt encoder, decoder from cached embeddings, postprocess, and end-to-end prompted masks.
- [ ] Add performance probes for encoder, cached prompt decoding, HQ path, and postprocess.
