# Transformers SAM Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary target: facebook/sam-vit-base.
  Representative sizing references: facebook/sam-vit-large,
  facebook/sam-vit-huge, hf-internal-testing/tiny-random-SamModel.

Config source:
  https://huggingface.co/facebook/sam-vit-base/raw/main/config.json
  https://huggingface.co/facebook/sam-vit-base/raw/main/preprocessor_config.json
  https://huggingface.co/facebook/sam-vit-large/raw/main/config.json
  https://huggingface.co/facebook/sam-vit-large/raw/main/preprocessor_config.json
  https://huggingface.co/facebook/sam-vit-huge/raw/main/config.json
  https://huggingface.co/facebook/sam-vit-huge/raw/main/preprocessor_config.json
  https://huggingface.co/hf-internal-testing/tiny-random-SamModel/raw/main/config.json
  https://huggingface.co/hf-internal-testing/tiny-random-SamModel/raw/main/preprocessor_config.json

Source files inspected:
  transformers/src/transformers/models/sam/configuration_sam.py
  transformers/src/transformers/models/sam/modeling_sam.py
  transformers/src/transformers/models/sam/processing_sam.py
  transformers/src/transformers/models/sam/image_processing_sam.py
  transformers/tests/models/sam/test_modeling_sam.py

Any missing files or assumptions:
  No custom remote-code files are required for standard SAM checkpoints.
  This report targets inference for promptable image segmentation. Training,
  automatic mask generation postprocessing, NMS/RLE export, and PerSAM target
  prompting are optional/deferred unless called out. Dinoml assumption:
  inference-only first, CUDA target, preserve PyTorch semantic axes initially,
  and treat NHWC/channel-last as a guarded layout optimization.
```

## 2. High-level architecture

SAM is a multi-stage vision segmentation model:

```text
image preprocessing -> ViT image encoder -> prompt encoder -> two-way mask decoder -> low-res masks/iou -> postprocess masks
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, longest-edge resize, rescale, ImageNet normalization, zero padding to square, coordinate normalization for points/boxes, optional segmentation-map preprocessing.
- Image encoder: NCHW `pixel_values` enter a non-overlap Conv2d patch projection; the main ViT blocks operate on NHWC `[B,64,64,H]`; the neck returns NCHW image embeddings `[B,256,64,64]`.
- Prompt encoder: point/box prompts become sparse prompt embeddings `[B,point_batch,T,256]`; mask prompts become dense embeddings `[B,256,64,64]`; no-mask uses an expanded learned embedding.
- Mask decoder: cacheable `image_embeddings` plus prompt embeddings feed a two-way transformer, transposed-conv upscaler, hypernetwork mask heads, and IoU MLP.
- Postprocess: processor upsamples low-resolution masks to padded image size, crops away padding using `reshaped_input_sizes`, upsamples to `original_sizes`, and optionally thresholds.

Independently stageable pieces: preprocessing handoff, image encoder cache, prompt encoder, mask decoder from cached image embeddings, and postprocessing.

## 3. Important config dimensions

Worked example: `facebook/sam-vit-base`.

| Field | Value | Source |
|---|---:|---|
| primary runtime target | promptable image segmentation | source/model metadata |
| input image size | 1024 x 1024 | config + preprocessor |
| patch size | 16 | config.json |
| patch grid | 64 x 64 | inferred |
| vision hidden size | 768 | config.json |
| vision layers | 12 | config.json |
| vision heads / head dim | 12 / 64 | config + inferred |
| vision MLP dim | 3072 | config.json |
| vision output channels | 256 | config.json |
| vision qkv bias | true | config.json |
| abs/relative position | true / true | config.json |
| window size | 14 | config.json |
| global attention layers | [2, 5, 8, 11] | config.json |
| prompt hidden size | 256 | config.json |
| prompt image embedding size | 64 | config.json |
| prompt mask input channels | 16 | config.json |
| mask decoder hidden size | 256 | config.json |
| mask decoder layers | 2 | config.json |
| mask decoder heads / head dim | 8 / 32 | config + inferred |
| mask decoder MLP dim | 2048 | config.json |
| attention downsample rate | 2 | config.json |
| multimask outputs | 3 plus single-mask token | config/source |
| dtype | float32 | config.json |
| cache support | image embeddings can be precomputed | source API |

Representative checkpoint sweep:

| Checkpoint | Vision H | Layers | Heads | MLP | Global layers | Output C | Prompt/decoder H | Image/patch/grid |
|---|---:|---:|---:|---:|---|---:|---:|---|
| hf-internal-testing/tiny-random-SamModel | 36 | 2 | 4 | 144 | [2,5,8,11] | 32 | 32 | 24 / 2 / 12 |
| facebook/sam-vit-base | 768 | 12 | 12 | 3072 | [2,5,8,11] | 256 | 256 | 1024 / 16 / 64 |
| facebook/sam-vit-large | 1024 | 24 | 16 | 4096 | [5,11,17,23] | 256 | 256 | 1024 / 16 / 64 |
| facebook/sam-vit-huge | 1280 | 32 | 16 | 5120 | [7,15,23,31] | 256 | 256 | 1024 / 16 / 64 |

Processor defaults for official production checkpoints: resize longest edge to 1024, rescale by `1/255`, normalize with ImageNet mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`, convert RGB, and pad to `[1024,1024]`. Source defaults for mask preprocessing are `mask_size={"longest_edge":256}` and `mask_pad_size={"height":256,"width":256}`, but the inspected production preprocessor JSONs omit those fields and rely on source defaults when mask labels are processed.

## 3a. Family variation traps

- Source `pixel_values` contract is NCHW `[B,3,1024,1024]`; patch embeddings immediately return NHWC `[B,64,64,H]`.
- The vision encoder intentionally mixes layouts: NHWC transformer blocks, then `SamVisionNeck` permutes to NCHW for `Conv2d + channels_first LayerNorm + Conv2d`.
- Prompt and mask tensors are NCHW: `input_masks` must be `[B,1,256,256]` in practice for the mask embedding conv stack to produce `[B,256,64,64]`; docstrings omit the channel dimension in places.
- Window attention uses window size 14 on a 64x64 grid, so padding to 70x70 happens inside local attention layers and is cropped after unpartition.
- Global attention layer indexes vary with model size. Naively assuming every fourth layer or always `[2,5,8,11]` breaks large/huge.
- Vision attention uses decomposed relative position bias and optional SDPA. The rel-pos bias is additive attention bias, not RoPE.
- Mask decoder attention is noncausal but has a separate downsampled internal dimension: cross-attention projects 256 -> 128 for q/k/v, self-attention uses 256 -> 256.
- `point_batch_size` is a real dynamic axis. The decoder repeats image embeddings along batch by this factor and returns `[B,point_batch,num_masks,256,256]` low-res masks.
- If no boxes are supplied, `_embed_points` appends a padding point with label `-1`; processor-added padding uses label `-10` and is zeroed.
- Boxes and points must agree on point/box batch size when both are present.
- The tiny random checkpoint is useful for tests but has unusual preprocessor metadata: config image size is 24 while fetched `pad_size` remains 1024. Treat it as a synthetic modeling test, not production processor guidance.
- Layout pass guards: do not globally translate NCHW Conv/ConvTranspose mask and neck regions to NHWC unless all Conv/LayerNorm axis rewrites and consumers are covered.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW image input `[B,3,1024,1024]`.
- Non-overlap patch Conv2d output permute to NHWC `[B,64,64,H]`.
- NHWC LayerNorm over channel dim.
- Window partition/unpartition with pad, reshape, permute, contiguous, crop.
- Flatten spatial image embeddings `[B,256,64,64] -> [B,4096,256]`.
- `repeat`, `repeat_interleave`, `expand`, `cat`, `stack`, `slice`, `reshape`, `transpose`, `permute`.
- Dynamic prompt axes: `point_batch_size`, sparse token count, mask count.
- `torch.where` label-conditioned point embedding updates.
- Batched hypernetwork matmul `hyper_in @ upscaled_embedding`.

### Neural network primitives

- Patch projection: `Conv2d(3 -> H, kernel=16, stride=16, bias=True)`.
- Vision QKV: `Linear(H -> 3H, bias=True)`; base is `768 -> 2304`, large `1024 -> 3072`, huge `1280 -> 3840`.
- Vision output projection `Linear(H -> H)`.
- Vision MLP `Linear(H -> 4H) -> GELU -> Linear(4H -> H)`.
- Neck: `Conv2d(H -> 256, kernel=1, bias=False)`, channels-first LayerNorm, `Conv2d(256 -> 256, kernel=3, padding=1, bias=False)`, channels-first LayerNorm.
- Prompt mask embedding: `Conv2d(1 -> 4, k=2,s=2)`, LayerNorm, GELU, `Conv2d(4 -> 16,k=2,s=2)`, LayerNorm, GELU, `Conv2d(16 -> 256,k=1)`.
- Decoder attention projections: self `Linear(256 -> 256)`, cross `Linear(256 -> 128)`, output `Linear(128 -> 256)`.
- Decoder MLP blocks `Linear(256 -> 2048) -> ReLU -> Linear(2048 -> 256)`.
- Upscaler: `ConvTranspose2d(256 -> 64,k=2,s=2)`, channels-first LayerNorm, GELU, `ConvTranspose2d(64 -> 32,k=2,s=2)`, GELU.
- Hypernetwork MLPs: four copies of `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 32)`.
- IoU head: `Linear(256 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 4)`.

### Attention primitives

- Vision noncausal MHA on NHWC spatial tokens, local window or global.
- Decoder noncausal self-attention over sparse tokens.
- Decoder bidirectional cross-attention token-to-image and image-to-token.
- Additive attention bias for decomposed relative position; optional `attention_similarity` for decoder token-to-image.
- No causal mask, KV cache, GQA/MQA, RoPE, or packed varlen metadata.

### Position/custom math ops

- Learned absolute 2D vision position table `[1,64,64,H]`.
- Decomposed relative position tables per attention layer/window: local `27 x head_dim`, global `127 x head_dim` for 64 grid.
- Random Fourier positional embedding for points, boxes, and image-wide decoder positions.
- Coordinate scaling and `sin/cos` concat.

### Preprocessing-coupled ops

- Resize longest edge, zero pad, normalization, coordinate normalization from original image space to resized image space.
- Postprocess masks with bilinear interpolate, crop, bilinear interpolate, threshold.
- Optional automatic mask generation utilities use crop generation, stability score reductions, mask-to-box, NMS, padding, and RLE; defer for first core model runtime.

## 5. Layer/block breakdown

Image encoder:

```text
pixel_values [B,3,1024,1024] NCHW
x = Conv2d(3 -> H, k=16,s=16)(pixel_values).permute(0,2,3,1)  # [B,64,64,H]
x = x + abs_pos[1,64,64,H]
repeat L vision layers:
  residual = x
  y = LayerNorm(x) over C
  if local layer: y = window_partition(y, 14), pad 64x64 -> 70x70
  y = MHA(qkv=Linear(H -> 3H), relative_bias, noncausal)
  if local layer: y = window_unpartition(y), crop to 64x64
  x = residual + Linear(H -> H)(y)
  x = x + MLP(LayerNorm(x))
x = permute NHWC -> NCHW
x = Conv2d(H -> 256,k=1,bias=False); LayerNorm channels_first
x = Conv2d(256 -> 256,k=3,pad=1,bias=False); LayerNorm channels_first
```

Prompt encoder:

```text
points [B,P,N,2], labels [B,P,N]
point_pe = random_fourier(points + 0.5, input_shape=(1024,1024))
point_pe = label-conditioned where/add for negative, positive, background, padding

boxes [B,P,4] -> corners [B,P,2,2]
box_pe = random_fourier(corners + 0.5)
box_pe[:, :, 0] += corner0_embedding
box_pe[:, :, 1] += corner1_embedding

sparse = concat(point_pe, box_pe, dim=2)
dense = mask_embed(input_masks) or no_mask_embed expanded to [B,256,64,64]
```

Mask decoder:

```text
output_tokens = concat(iou_token[1,256], mask_tokens[4,256])
tokens = concat(output_tokens repeated [B,P,5,256], sparse_prompts, dim=2)
image = (image_embeddings + dense_prompt_embeddings).repeat_interleave(P, dim=0)
pos = image_positional_embeddings.repeat_interleave(P, dim=0)

point_embedding, image_tokens = two_way_transformer(tokens, image, pos)
iou_token_out = point_embedding[:, :, 0, :]
mask_tokens_out = point_embedding[:, :, 1:5, :]

image_tokens -> [B*P,256,64,64]
upscaled = ConvTranspose2d(256->64,2,2) -> LN -> GELU -> ConvTranspose2d(64->32,2,2) -> GELU
hyper_in = stack(4 hypernetwork MLP(mask_tokens_out[..., i, :]), dim=2)  # [B,P,4,32]
masks = hyper_in @ upscaled.reshape(B,P,32,256*256) -> [B,P,4,256,256]
iou = iou_head(iou_token_out) -> [B,P,4]
if multimask_output: select masks/iou indices 1:4 else 0:1
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention, MHA.
- Head dim is `vision_hidden_size / num_heads`: base 64, large 64, huge 80.
- Local layers attend over padded 14x14 windows, sequence length 196 per window.
- Global layers attend over 64x64 = 4096 tokens.
- Source eager math: `(query * scale) @ key.T`, add decomposed rel-pos bias, fp32 softmax cast back to query dtype, dropout, matmul with value.
- Source SDPA path passes rel-pos as `attn_mask`; it does not return attention weights.

Mask decoder attention:

- Noncausal self- and cross-attention over tensors shaped `[B,point_batch,T,C]`.
- Self-attention internal dim 256 with 8 heads, head dim 32.
- Cross-attention internal dim 128 with 8 heads, head dim 16.
- No KV cache; staged inference should cache image embeddings, not attention keys.
- `attention_similarity` can be passed as an additive mask/bias to token-to-image cross attention for PerSAM; defer unless target prompting is required.

Flash/SDPA compatibility: vision global attention with rel-pos additive bias needs an attention backend that accepts per-head additive bias `[B,heads,Q,K]`; local window attention is SDPA-compatible after window partition. Decoder attention is small but dynamic in prompt-token count, and eager matmul may be acceptable for first parity.

## 7. Position encoding and custom math

Random Fourier point/image positional embedding:

```python
def sam_positional_embedding(coords, gaussian, input_shape=None):
    # coords final dim is [x, y]
    if input_shape is not None:
        coords[..., 0] = coords[..., 0] / input_shape[1]
        coords[..., 1] = coords[..., 1] / input_shape[0]
    coords = 2 * coords - 1
    phase = (coords @ gaussian) * (2 * pi)
    return concat([sin(phase), cos(phase)], dim=-1)
```

Image-wide decoder position grid:

```python
grid = ones([64, 64])
y = cumsum(grid, dim=0) - 0.5
x = cumsum(grid, dim=1) - 0.5
pos = sam_positional_embedding(stack([x / 64, y / 64], dim=-1), gaussian)
pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1,256,64,64]
```

Decomposed relative position bias:

```python
rel_h = einsum("bhwc,hkc->bhwk", query.reshape(Bh, qh, qw, D), rel_pos_h_indexed)
rel_w = einsum("bhwc,wkc->bhwk", query.reshape(Bh, qh, qw, D), rel_pos_w_indexed)
bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
```

Precompute candidates:

- Static absolute position table and image-wide decoder positional embedding.
- Local/global relative coordinate index matrices for fixed 14 and 64 grids.
- If supporting non-default image grids later, `get_rel_pos` includes linear interpolation and coordinate scaling, so bucket and cache per `(q_size,k_size)`.

## 8. Preprocessing and input packing

Processor runtime tensors:

- `pixel_values`: source model expects `[B,3,1024,1024]` NCHW float tensor after resize/rescale/normalize/pad.
- `original_sizes`: list/tensor `[B,2]`, original `(height,width)` before resize.
- `reshaped_input_sizes`: list/tensor `[B,2]`, resized `(height,width)` before square padding.
- `input_points`: optional `[B,point_batch,num_points,2]` in resized image coordinates.
- `input_labels`: optional `[B,point_batch,num_points]`; default labels are ones if points are supplied without labels.
- `input_boxes`: optional `[B,num_boxes,4]`; processor unsqueezes to a box batch when needed.
- `input_masks`: optional runtime input to model, expected as NCHW mask map for conv embedding, normally `[B,1,256,256]`.

Coordinate normalization uses original image size and the same longest-edge resize rule:

```text
new_h,new_w = round(old_h * 1024 / max(old_h,old_w)), round(old_w * 1024 / max(old_h,old_w))
x *= new_w / old_w
y *= new_h / old_h
```

Staged inference path:

- Run `get_image_embeddings(pixel_values)` once per image: output `[B,256,64,64]`.
- For each prompt batch, run prompt encoder + mask decoder using cached image embeddings.
- Postprocess each low-res mask batch using `original_sizes` and `reshaped_input_sizes`.

CPU/data-pipeline vs GPU/runtime: image resize/normalize/pad and coordinate normalization can remain CPU initially. GPU graph should start at `pixel_values` or cached `image_embeddings`; postprocess can be CPU first, then GPU later if end-to-end segmentation throughput requires it.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> NHWC Linear

Preconditions:

- `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input image height/width exactly match config and are divisible by patch size.
- Source pattern is `Conv2d(...).permute(0,2,3,1)`.

Replacement:

```text
NCHW or NHWC WindowFlatten(16,16) -> GEMM_RCR_Bias -> Reshape[B,64,64,H]
```

Weight transform for preferred NHWC activation flatten:

```python
w = conv.weight.permute(0, 2, 3, 1).reshape(out_channels, 16 * 16 * 3)
```

Failure cases: non-square/non-1024 runtime images after preprocessing, different patch sizes, grouped conv, or preserving NCHW flatten order without matching weight transform.

Parity test sketch: compare source Conv2d+permute with rewritten GEMM on random fp32/fp16 inputs for base and tiny shapes.

### Rewrite: vision packed QKV stays packed

Source pattern: `Linear(H -> 3H)` followed by reshape/permute/split.

Replacement: keep one packed GEMM and split logical Q/K/V views.

Preconditions: output width exactly `3 * hidden_size`, bias setting from config, no consumer requires separate Q/K/V materialization.

### Rewrite: local window attention lowering

Source pattern:

```text
NHWC pad -> reshape -> permute -> reshape windows -> attention -> inverse reshape/permute -> crop
```

Replacement: use a windowed-attention view/copy planner or fused partition-attention-unpartition kernel.

Preconditions: fixed grid 64, fixed window 14, padding to 70, local layer only, no output attentions needed.

Failure cases: arbitrary dynamic grid, output attentions requested, or backend cannot represent padded/cropped windows.

### Rewrite: channels-first LayerNorm in NCHW conv regions

Source pattern: `permute NCHW->NHWC -> LayerNorm(C) -> permute NHWC->NCHW`.

Replacement: native channels-first LayerNorm over C for NCHW, or NHWC conv region with transformed Conv weights.

Preconditions: only normalize channel axis, no intervening consumer observes NHWC temporary.

Axis rewrites: `normalized_shape=C`, reduce axis `dim=1` in NCHW equivalent; if translating to NHWC, reduce axis becomes `dim=-1`.

### Rewrite: cached image embeddings

Source API already exposes `get_image_embeddings` and accepts `image_embeddings` instead of `pixel_values`.

Replacement:

```text
image_encoder(pixel_values) -> persisted image_embeddings
prompt batches reuse image_embeddings -> mask_decoder
```

Preconditions: same image, same image encoder weights/config, prompt-only changes.

Parity test sketch: compare full `model(pixel_values,prompt)` against `model(image_embeddings=get_image_embeddings(pixel_values),prompt)`.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d -> GEMM and NHWC vision-block layout preservation.
- Vision global and local attention with decomposed relative bias.
- LayerNorm + packed QKV for vision blocks.
- MLP `Linear + GELU/ReLU + Linear` epilogue fusion where profitable.
- Mask decoder hypernetwork batched matmul for `[B,P,4,32] @ [B,P,32,65536]`.

Medium priority:

- Window partition/unpartition fused with local attention.
- Neck Conv1x1/Conv3x3 with channels-first LayerNorm, or a guarded NHWC conv rewrite.
- ConvTranspose upscaler kernels.
- Prompt random Fourier sin/cos embedding fused with label-conditioned adds.
- Postprocess bilinear resize/crop/threshold on GPU.

Lower priority:

- Automatic mask generation NMS/RLE/stability-score kernels.
- PerSAM `target_embedding` and `attention_similarity` specializations.
- Dynamic arbitrary image-size rel-pos interpolation.

## 11. Runtime staging plan

Stage 1: Parse `SamConfig` and preprocessor metadata; load weights; support production base/large/huge dimensions.

Stage 2: Image encoder parity from NCHW `pixel_values` to `[B,256,64,64]`; start with faithful PyTorch layout regions.

Stage 3: Add cached image embedding API and validate `get_image_embeddings` equivalence.

Stage 4: Prompt encoder parity for no prompt, one point, multiple points, boxes, and mask prompts.

Stage 5: Mask decoder parity from cached image embeddings and prompt embeddings to low-res masks/IoU.

Stage 6: Processor-compatible postprocess for mask upsample/crop/original-size resize; CPU first is acceptable.

Stage 7: Add optimized rewrites: patch GEMM, packed QKV, attention backends with rel-pos bias, window fusion, hypernetwork matmul tuning.

Stage 8: Optional automatic mask generation pipeline: crop boxes, batched prompt scheduling, stability score, NMS, RLE.

## 12. Parity and validation plan

- Processor handoff: verify `pixel_values`, `original_sizes`, `reshaped_input_sizes`, normalized points, boxes, and labels against HF processor for a fixed image.
- Patch projection parity: source Conv2d+permute vs any GEMM rewrite.
- Vision one-block parity for local and global attention layers separately.
- Full image encoder parity for base and tiny configs.
- Random Fourier positional embedding parity for points, boxes, and image-wide grid.
- Prompt encoder parity for no prompt, positive/negative points, padded points (`-10`), boxes, and mask prompts.
- Mask decoder parity from fixed random cached embeddings and prompts.
- Staged path parity: full model with `pixel_values` vs cached `image_embeddings`.
- End-to-end low-res mask/IoU parity using HF integration scenarios: no point, one point, point+box, batched points/images, boxes.
- Postprocess parity for `post_process_masks` including crop-to-reshaped-size and resize-to-original.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for full model initially due to attention/resize sensitivity; isolated fp32 ops `1e-5/1e-6`; fp16/bf16 `rtol=2e-2, atol=2e-2` until fused attention/conv policies are tuned.

## 13. Performance probes

- CPU processor throughput: resize/normalize/pad and coordinate normalization.
- Image encoder throughput by model size: base/large/huge, batch size sweep.
- Local-window vs global-attention layer timings; include rel-pos bias construction time.
- Patch projection Conv2d vs GEMM rewrite.
- Neck Conv/LN timing and layout conversion cost.
- Prompt encoder latency by prompt type and prompt count.
- Mask decoder latency vs `point_batch_size` and number of sparse tokens.
- Cached-image path: masks/sec for many prompt batches per image.
- Postprocess bilinear resize/crop/threshold throughput.
- Memory probes: image embeddings `[B,256,64,64]`, global attention temporary `[B,heads,4096,4096]`, repeated decoder image tokens by `point_batch_size`.

## 14. Skip/defer list

- Training and segmentation label loss.
- Gradient checkpointing and output attentions.
- Automatic mask generation crop scheduler, stability score filtering, NMS, mask-to-RLE.
- PerSAM `attention_similarity` and `target_embedding`.
- Arbitrary image sizes without fixed preprocessing to 1024 and rel-pos interpolation buckets.
- Full GPU preprocessing/postprocessing.
- Multi-GPU sharding and quantization.
- Tiny random checkpoint processor quirks beyond modeling smoke tests.

## 15. Final implementation checklist

- [ ] Parse `SamConfig` subconfigs and processor defaults.
- [ ] Define NCHW `pixel_values` and cached `[B,256,64,64]` image embedding ABI.
- [ ] Load image encoder, prompt encoder, mask decoder weights.
- [ ] Implement faithful patch Conv2d, NHWC vision blocks, and NCHW neck.
- [ ] Implement decomposed relative position bias for local/global vision attention.
- [ ] Implement point/box random Fourier positional embeddings and label-conditioned prompt logic.
- [ ] Implement mask prompt embedding and no-mask dense embedding expansion.
- [ ] Implement two-way transformer self/cross attention with prompt batch axis.
- [ ] Implement ConvTranspose upscaler, hypernetwork mask matmul, IoU head, and multimask slicing.
- [ ] Add cached image embedding staged inference path.
- [ ] Add postprocess mask resize/crop/original-size resize parity.
- [ ] Add patch Conv2d -> GEMM rewrite with layout/weight guards.
- [ ] Add attention backend path for additive rel-pos bias.
- [ ] Add parity tests for no prompt, point, box, point+box, batched prompts, mask prompt, and cached image embeddings.
- [ ] Benchmark processor, image encoder, prompt encoder, mask decoder, postprocess, and many-prompts-per-image throughput.
