# VitMatte DinoML Operator Assessment

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: hustvl/vitmatte-small-composition-1k, hustvl/vitmatte-base-composition-1k,
  hustvl/vitmatte-small-distinctions-646, hustvl/vitmatte-base-distinctions-646
Config source: official Hugging Face config.json and preprocessor_config.json snapshots
Source files inspected:
  transformers/src/transformers/models/vitmatte/configuration_vitmatte.py
  transformers/src/transformers/models/vitmatte/modeling_vitmatte.py
  transformers/src/transformers/models/vitmatte/image_processing_vitmatte.py
  transformers/src/transformers/models/vitmatte/image_processing_pil_vitmatte.py
  transformers/src/transformers/models/vitdet/configuration_vitdet.py
  transformers/src/transformers/models/vitdet/modeling_vitdet.py
Any missing files or assumptions:
  VitMatte composes a nested backbone through load_backbone(config). The sampled public checkpoints use native
  VitDetBackbone; this report records the VitDet contract VitMatte consumes, but broader VitDet optimization can be
  staged as a reusable backbone audit. No remote-code files are required for the sampled checkpoints.
```

Primary source links:

- [VitMatte modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitmatte/modeling_vitmatte.py)
- [VitMatte configuration](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitmatte/configuration_vitmatte.py)
- [VitMatte image processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitmatte/image_processing_vitmatte.py)
- [VitMatte PIL image processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitmatte/image_processing_pil_vitmatte.py)
- [VitDet modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitdet/modeling_vitdet.py)
- [VitDet configuration](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitdet/configuration_vitdet.py)

Snapshots and notes are stored under `agents/plans/transformers/vitmatte/_sources/`.

## 2. High-level architecture

VitMatte is an image matting model: a 4-channel RGB+trimap tensor is encoded by a ViTDet image backbone, then decoded by a lightweight NCHW convolutional detail-capture module into a single alpha matte.

```text
CPU image + trimap preprocessing -> 4-channel NCHW pixel_values
  -> ViTDet patch/backbone feature map at stage12
  -> ConvStream detail pyramid from original pixel_values
  -> top-down fusion + bilinear upsampling decoder
  -> matting head -> sigmoid alpha
```

Runtime stages:

- CPU/data pipeline: RGB conversion, rescale, image normalization, trimap rescale, concatenate trimap as channel 4, bottom/right pad to a multiple of 32.
- Backbone: ViTDet consumes `[B,4,H,W]`, patchifies with stride/kernel 16, alternates global/windowed noncausal attention, and returns `stage12` feature map.
- Decoder: NCHW Conv-BN-ReLU stream extracts high-resolution detail features; fusion blocks upsample backbone features and concatenate channel-wise with detail maps.
- Output: `alphas` is `[B,1,H_pad,W_pad]` after sigmoid. Source has no crop-to-original-size or trimap-aware alpha compositing postprocess.

The first useful DinoML target is inference-only `VitMatteForImageMatting.alphas`. Training loss is explicitly not implemented in the source.

## 3. Important config dimensions

Representative checkpoint sweep:

| Field | small composition | base composition | small distinctions | base distinctions | Source/default note |
|---|---:|---:|---:|---:|---|
| HF model | `hustvl/vitmatte-small-composition-1k` | `hustvl/vitmatte-base-composition-1k` | `hustvl/vitmatte-small-distinctions-646` | `hustvl/vitmatte-base-distinctions-646` | official HF configs |
| `model_type` | `vitmatte` | `vitmatte` | `vitmatte` | `vitmatte` | config |
| backbone type | `vitdet` | `vitdet` | `vitdet` | `vitdet` | config |
| input channels | 4 | 4 | 4 | 4 | image + trimap |
| input image size | 512 | 512 | 512 | 512 | config; processor can pad arbitrary sizes |
| patch size | 16 | 16 | 16 | 16 | omitted; VitDet default |
| patch grid at 512 | 32 x 32 | 32 x 32 | 32 x 32 | 32 x 32 | inference from config/default |
| VitMatte `hidden_size` | 384 | 768 | 384 | 768 | config |
| backbone hidden size | 384 | 768 | 384 | 768 | small explicit; base from VitDet default |
| layers | 12 | 12 | 12 | 12 | omitted; VitDet default |
| attention heads | 6 | 12 | 6 | 12 | small explicit; base from VitDet default |
| head dim | 64 | 64 | 64 | 64 | inferred as hidden/heads |
| MLP ratio | 4 | 4 | 4 | 4 | omitted; VitDet default |
| QKV bias | true | true | true | true | omitted; VitDet default |
| relative position | true | true | true | true | config |
| window size | 14 | 14 | 14 | 14 | config |
| window blocks | 0,1,3,4,6,7,9,10 | same | same | same | config |
| global/residual blocks | 2,5,8,11 | same | same | same | config |
| output feature | `stage12` | `stage12` | `stage12` | `stage12` | config |
| ConvStream channels | 48,96,192 | same | same | same | config |
| Fusion channels | 256,128,64,32 | same | same | same | config |
| processor mean/std | 0.5/0.5/0.5 | same | same | same | preprocessor config overrides source default |
| pad divisor | 32 | 32 | 32 | 32 | `size_divisibility`; mapped to `size_divisor` |
| dtype | float32 | float32 | float32 | float32 | `torch_dtype` in config |
| cache support | none | none | none | none | image encoder, no generation cache |

## 3a. Family variation traps

- VitMatte is backbone-composed. A future config can pass another `backbone_config`; first admission should allowlist native `vitdet` only, or route other backbones to separate audits.
- `config.hidden_size` must match the selected backbone output width. The decoder constructs its first fusion input as `hidden_size + convstream_hidden_sizes[-1]`; a mismatch is a hard shape error.
- Public base configs omit backbone `hidden_size` and `num_attention_heads`; effective values come from `VitDetConfig` defaults.
- The processor packs trimap into channel 4 and the backbone `num_channels` is 4. Plain RGB ViTDet weights are not compatible without a weight/input adaptation policy.
- Source image processor defaults are ImageNet mean/std, but public HUST-VL preprocessor configs override to mean/std 0.5. DinoML should bind preprocessor config, not source defaults.
- Trimap values are only rescaled; source does not validate three discrete trimap classes or enforce unknown-region semantics.
- Input and decoder are NCHW. VitDet transformer blocks switch to NHWC internally for LayerNorm, attention, MLP, window partitioning, and relative-position math, then return to NCHW. Layout optimization must respect this boundary.
- Window attention pads token maps to multiples of `window_size`; with 512 input and patch size 16, a 32x32 token map is padded to 42x42 inside window blocks.
- Relative-position attention uses dynamic interpolation and integer index gather when current query/key size differs from stored table length.
- Alpha postprocess is absent in native source. End-to-end applications often need crop-to-original padded size, optional resize, and possibly trimap-guided known foreground/background restoration, but those are not implemented here.

## 4. Operator coverage checklist

Preprocessing-coupled ops:

- RGB image conversion and channel-first tensor preparation outside the GPU graph.
- Rescale image and trimap by `1/255`.
- Normalize image channels with preprocessor mean/std; do not normalize trimap.
- Concatenate image `[B,3,H,W]` and trimap `[B,1,H,W]` into `[B,4,H,W]` along channel axis.
- Constant bottom/right pad to `ceil(H/32)*32`, `ceil(W/32)*32`.

Backbone-owned neural primitives:

- Conv2d patch embedding: `Conv2d(4 -> C, kernel=16, stride=16, bias=True)`.
- Absolute position path: reshape/permute, optional bicubic interpolate from square pretrain table, add to NHWC embeddings.
- Repeated 12 ViTDet blocks:
  - NCHW to NHWC permute.
  - LayerNorm over channel-last hidden dimension.
  - QKV Linear `C -> 3C` with bias, split order `q,k,v`.
  - Noncausal MHA, global or windowed, heads 6/12 and head_dim 64.
  - Optional decomposed relative-position score add via interpolate, arange, integer gather, einsum, broadcast add.
  - Softmax over key axis and attention matmul.
  - Output Linear `C -> C`.
  - Residual adds, MLP Linear `C -> 4C -> C` with GELU, dropout identity at inference.
  - Optional residual bottleneck after blocks 2,5,8,11: Conv1x1 `C -> C/2`, custom NCHW channel LayerNorm, GELU, Conv3x3 `C/2 -> C/2`, LayerNorm, GELU, Conv1x1 `C/2 -> C`, LayerNorm, residual add.
- Window partition/unpartition: pad NHWC tokens, view, permute, contiguous, crop.

VitMatte decoder operators:

- ConvStream detail pyramid:
  - detail0: `[B,4,H,W]`.
  - Conv3x3-BN-ReLU stride 2: `4 -> 48`, output `[B,48,H/2,W/2]`.
  - Conv3x3-BN-ReLU stride 2: `48 -> 96`, output `[B,96,H/4,W/4]`.
  - Conv3x3-BN-ReLU stride 2: `96 -> 192`, output `[B,192,H/8,W/8]`.
- Fusion block 0: bilinear upsample backbone `[B,C,H/16,W/16]` by 2, concat with detail3 on dim 1, Conv3x3-BN-ReLU `(C+192) -> 256`.
- Fusion block 1: upsample, concat detail2, Conv3x3-BN-ReLU `352 -> 128`.
- Fusion block 2: upsample, concat detail1, Conv3x3-BN-ReLU `176 -> 64`.
- Fusion block 3: upsample, concat detail0, Conv3x3-BN-ReLU `68 -> 32`.
- Matting head: Conv3x3 `32 -> 16`, BatchNorm2d, ReLU, Conv1x1 `16 -> 1`, sigmoid.

Structured-output/postprocessing ops:

- Required native output: alpha tensor `[B,1,H_pad,W_pad]`.
- Deferred/wrapper-owned parity gap: crop alpha to original unpadded size and any application-level alpha compositing with trimap.

## 5. Layer/block breakdown

Processor packing:

```text
image = rescale(image, 1/255)
image = normalize(image, mean, std)
trimap = rescale(trimap, 1/255)
pixel_values = concat([image, trimap], dim=1)
pixel_values = pad_bottom_right(pixel_values, multiple=32)
```

ViTDet backbone:

```text
x_nchw = Conv2d(4 -> C, kernel=16, stride=16)(pixel_values)
if absolute_position_embeddings:
  x_nhwc = permute(x_nchw, NCHW -> NHWC)
  pos = resize_or_reshape_abs_pos(pos_table, grid_h, grid_w)
  x_nchw = permute(x_nhwc + pos, NHWC -> NCHW)

for layer i in 0..11:
  x = permute(x, NCHW -> NHWC)
  shortcut = x
  y = LayerNorm(x)
  if i in window_block_indices:
    y, pad_hw = window_partition(y, window_size=14)
  y = MHA_with_optional_relative_position(y)
  if windowed:
    y = window_unpartition(y, pad_hw, original_hw)
  x = shortcut + y
  x = x + MLP(LayerNorm(x))
  x = permute(x, NHWC -> NCHW)
  if i in residual_block_indices:
    x = x + BottleneckConvNormGeluConvNormGeluConvNorm(x)
```

VitMatte decoder:

```text
details = ConvStream(pixel_values)
features = stage12_feature_map
for detail in [details[3], details[2], details[1], details[0]]:
  features = bilinear_upsample(features, scale_factor=2, align_corners=False)
  features = concat([detail, features], dim=1)
  features = Conv3x3 + BatchNorm2d + ReLU(features)
alpha = sigmoid(Conv1x1(BN(ReLU(Conv3x3(features)))))
```

## 6. Attention requirements

VitMatte has no decoder/generation attention. Attention is encoder-only, noncausal self-attention inside the nested ViTDet backbone.

Required attention variants:

- Global noncausal MHA on NHWC spatial tokens for layers 2,5,8,11.
- Windowed noncausal MHA on padded 14x14 windows for layers 0,1,3,4,6,7,9,10.
- MHA only: `num_key_value_heads == num_attention_heads`; no GQA/MQA.
- Head dimensions: small `6 x 64`, base `12 x 64`.
- Masking: no user attention mask in source. Window padding is handled by cropping after unpartition, not an explicit attention mask.
- Relative bias: decomposed height/width learned relative-position tables are enabled in sampled configs.
- Packed/varlen: none.
- KV cache: none.
- FlashAttention/SDPA: source uses explicit matmul/softmax/matmul, with relative-position score injection before softmax. A fused path must either support score-bias injection or fall back for relative-position configs.

## 7. Position encoding and custom math

Absolute position embeddings, if enabled, are stored as a square table with an optional leading CLS slot. Source removes the CLS slot, reshapes to `[1,S,S,C]`, bicubic-interpolates in NCHW form if runtime grid differs, then returns NHWC positions.

Relative-position score addition:

```python
def add_decomposed_relative_positions(attn, q, rel_h, rel_w, q_hw, k_hw):
    rh = resize_1d_rel_pos_if_needed(rel_h, 2 * max(q_hw[0], k_hw[0]) - 1)
    rw = resize_1d_rel_pos_if_needed(rel_w, 2 * max(q_hw[1], k_hw[1]) - 1)
    h_index = scaled_relative_coord_indices(q_hw[0], k_hw[0])
    w_index = scaled_relative_coord_indices(q_hw[1], k_hw[1])
    Rh = rh[h_index]  # [QH, KH, head_dim]
    Rw = rw[w_index]  # [QW, KW, head_dim]
    q_grid = q.reshape(BH, QH, QW, head_dim)
    add_h = einsum("bhwc,hkc->bhwk", q_grid, Rh)
    add_w = einsum("bhwc,wkc->bhwk", q_grid, Rw)
    return attn.view(BH, QH, QW, KH, KW) + add_h[..., None] + add_w[..., None, :]
```

Precompute candidates:

- For fixed padded input size, absolute-position interpolation and relative-position index tensors can be precomputed per grid/window size.
- Relative-position value interpolation depends on stored weights and runtime query/key sizes; for fixed 512 checkpoints it is static per global/window resolution.

## 8. Preprocessing and input packing

Input contract:

- `images`: RGB-like image inputs prepared as channel-first tensors.
- `trimaps`: 2D image-like inputs prepared as `[1,H,W]`.
- `pixel_values`: only model input, shape `[B,4,H_pad,W_pad]`.
- Source processor groups same-shape images for batched torchvision processing, then reorders to original input order.

Packing details:

- Image is rescaled and normalized.
- Trimap is rescaled by `1/255` with `do_normalize=False`.
- Concatenation is channel axis `dim=1`/`axis=0` per sample.
- Padding is constant zero on bottom and right only; this pads trimap and image channels together after packing.

Alpha postprocess:

- Native model returns padded alpha directly.
- There is no `post_process_image_matting` method in the inspected VitMatte processor.
- DinoML wrapper should record original `(H,W)` before padding and provide an optional crop-to-original helper for application parity. Any rule that forces known foreground/background from trimap should be a separate non-native postprocess with an explicit opt-in.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> Linear/GEMM

Source pattern:

```text
Conv2d(4 -> C, kernel=16, stride=16, padding=0)
```

Replacement:

```text
WindowFlatten_NCHW_16x16 -> GEMM([B*GH*GW, 4*16*16] x [4*16*16, C]) -> reshape to [B,C,GH,GW]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == dilation == 0/1` as source Conv2d default.
- `groups == 1`.
- Runtime padded H/W divisible by 16.
- NCHW flatten order exactly matches PyTorch Conv2d weight layout `[out,in,kh,kw]`.

Failure cases:

- Non-ViTDet backbone or nonstandard patch conv.
- Dynamic layouts that already converted to NHWC without a matching weight transform.

Parity test sketch: compare patch embedding output before position add for random `[B,4,512,512]` and odd original sizes padded to multiple of 32.

### Rewrite: Conv-BN-ReLU inference folding

Source pattern:

```text
Conv2d(bias=False) -> BatchNorm2d -> ReLU
Conv2d -> BatchNorm2d -> ReLU in matting head first layer
```

Replacement:

```text
Conv2d(folded_weight, folded_bias) -> ReLU
```

Preconditions:

- Eval mode with frozen BN running stats.
- BN epsilon from `config.batch_norm_eps` for BasicConv3x3; PyTorch default for matting head BN.
- Preserve NCHW axis semantics.

Failure cases:

- Training mode or missing BN stats.
- Quantized/fused weight loading that cannot represent folded constants.

### Rewrite: fixed bilinear scale-2 fusion upsample

Source pattern:

```text
interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
concat(detail, upsampled, dim=1)
Conv3x3-BN-ReLU
```

Replacement options:

- Keep as explicit bilinear resize + concat + conv for first parity.
- Later fuse resize/concat/conv into one tiled CUDA kernel for the decoder.

Preconditions:

- Exact scale factor 2 at all four fusion blocks.
- `align_corners=False`.
- Detail feature spatial size equals doubled feature size.
- NCHW layout retained through fusion.

Failure cases:

- Runtime size mismatch from non-32-divisible input without processor padding.
- Layout pass converts only one concat operand.

### Rewrite: NHWC transformer island

Source pattern:

```text
NCHW -> NHWC -> LayerNorm/Attention/MLP/window ops -> NCHW
```

Replacement:

- Treat each VitDet layer as a channel-last island and remove redundant internal permutes when all consumers stay NHWC.

Preconditions:

- Entire layer island is controlled.
- Axis-sensitive operations are rewritten: channel concat remains decoder-only NCHW `dim=1`; transformer LayerNorm is last axis; attention flatten order is row-major H,W; window partition view/permute order is preserved.

Failure cases:

- Crossing into ConvStream/fusion decoder without converting back to NCHW.
- Residual bottleneck blocks, which are NCHW Conv2d + channel LayerNorm and interrupt the NHWC island after layers 2,5,8,11.

## 10. Kernel fusion candidates

Highest priority:

- Processor-to-input packing guard: 4-channel RGB+trimap NCHW ABI, pad-to-32 shape recording, and original-size metadata for output cropping.
- Decoder Conv-BN-ReLU folding plus bilinear-upsample/concat/conv kernels; this is VitMatte-specific and dominates after reusable ViTDet work exists.
- ViTDet QKV projection + attention + relative-position score-bias support. Existing dense attention without relative-bias injection will not cover sampled configs.

Medium priority:

- Window partition/unpartition kernels for NHWC token maps with pad/crop.
- Relative-position index precompute and fused score add for fixed 32x32 global and 14x14 window cases.
- Patch Conv2d-to-GEMM lowering for 4-channel inputs.

Lower priority:

- Absolute-position bicubic interpolation for arbitrary runtime sizes; public configs target fixed 512 but processor accepts variable padded sizes.
- Residual bottleneck ConvNeXt-style NCHW channel LayerNorm fusion.
- Optional alpha crop/composite helper outside the native model graph.

## 11. Runtime staging plan

Stage 1: config and preprocessing admission.

- Parse VitMatte config plus nested VitDet config.
- Allowlist `backbone_config.model_type == "vitdet"` for first integration.
- Produce `[B,4,H_pad,W_pad]` and original-size metadata from image+trimap inputs.

Stage 2: decoder-only parity.

- Feed synthetic backbone features and `pixel_values` into ConvStream + fusion + matting head.
- Validate bilinear upsample, channel concat, Conv-BN-ReLU, sigmoid, and padded alpha shape.

Stage 3: nested ViTDet backbone parity.

- Reuse or implement VitDet patch embedding, NHWC transformer blocks, relative-position attention, window attention, and residual bottleneck blocks.
- Validate stage12 feature map for small and base configs.

Stage 4: full model parity.

- Run full `[B,4,512,512] -> [B,1,512,512]` alpha parity.
- Add variable original-size inputs padded to 32 and crop wrapper validation.

Stage 5: optimized lowering.

- Fold BN, lower patch conv to GEMM, fuse decoder resize/concat/conv, and specialize relative-position/window attention for fixed shapes.

## 12. Parity and validation plan

- Processor packing tests: RGB tensor plus synthetic trimap with known values; verify image normalization, trimap rescale-only behavior, channel order, bottom/right padding, and original order after grouped batching.
- Decoder random tensor tests: compare each ConvStream output, each fusion block, matting head logits, and sigmoid alpha for small/base channel widths.
- Backbone one-block tests: global block and window block separately, including relative position enabled and residual bottleneck enabled.
- Window partition tests: 32x32 token map with `window_size=14` pads to 42x42, partitions, unpartitions, and crops exactly.
- End-to-end checkpoint tests: one fixture image+trimap for small composition and base composition; compare alpha tensor at padded size.
- Alpha wrapper tests: odd-size input padded to 32, output crop to original size, no native trimap compositing unless explicitly enabled.
- Tolerances: fp32 `atol=1e-5, rtol=1e-4` for isolated ops; fp16/bf16 should use looser alpha tolerances after sigmoid, e.g. `atol=2e-3, rtol=2e-3`, after calibrating against PyTorch.

## 13. Performance probes

- Processor throughput: image+trimap packing and pad throughput by resolution.
- Backbone throughput: ViTDet stage12 only, split global vs window block timing.
- Decoder throughput: ConvStream/fusion/head only for 512 and larger padded sizes.
- Resolution sweep: 512, 640x960 example shape, and odd sizes requiring padding.
- Batch-size sweep: B=1,2,4,8 for memory pressure, since alpha matting is image-heavy.
- Attention backend comparison: eager matmul/softmax vs fused attention with relative-position fallback.
- Decoder fusion comparison: explicit bilinear+concat+conv vs fused resize/concat/conv.
- Layout probe: cost of NCHW/NHWC permutes around VitDet islands and NCHW decoder boundaries.

## 14. Skip/defer list

- Training loss and label handling; source raises `NotImplementedError`.
- Non-ViTDet backbones until separately audited.
- General arbitrary trimap validation or foreground/background compositing as native behavior.
- General NHWC translation across the whole model; use guarded transformer islands only.
- Output attentions and hidden states for production alpha-only runtime, except as debug/parity aids.
- Quantization and packed weights; sampled configs are dense float32.
- Multi-GPU or tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse VitMatte config and nested VitDet config with allowlist admission.
- [ ] Load preprocessor config and implement RGB+trimap packing to `[B,4,H_pad,W_pad]`.
- [ ] Preserve original image size metadata for optional alpha crop.
- [ ] Implement/fold Conv2d-BatchNorm-ReLU blocks for ConvStream and fusion.
- [ ] Implement bilinear upsample `scale_factor=2`, `align_corners=False`.
- [ ] Implement matting head and sigmoid alpha output.
- [ ] Compose or import VitDet backbone coverage for 4-channel patch embedding.
- [ ] Implement ViTDet window partition/unpartition with pad/crop.
- [ ] Implement decomposed relative-position score add or guarded fallback.
- [ ] Add decoder-only parity tests for small/base channel widths.
- [ ] Add processor packing parity tests for image/trimap/pad behavior.
- [ ] Add full checkpoint alpha parity tests at padded and cropped sizes.
- [ ] Benchmark backbone, decoder, preprocessing, and layout-boundary costs separately.
