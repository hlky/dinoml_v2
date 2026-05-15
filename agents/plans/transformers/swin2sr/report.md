# Swin2SR Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: caidas/swin2SR-* family; primary target caidas/swin2SR-classical-sr-x2-64
Config source: official HF raw config/preprocessor files, fetched 2026-05-13
Source files inspected:
- transformers/src/transformers/models/swin2sr/modeling_swin2sr.py
- transformers/src/transformers/models/swin2sr/configuration_swin2sr.py
- transformers/src/transformers/models/swin2sr/image_processing_swin2sr.py
- transformers/src/transformers/models/swin2sr/image_processing_pil_swin2sr.py
- transformers/src/transformers/models/swin2sr/convert_swin2sr_original_to_pytorch.py
- transformers/tests/models/swin2sr/test_modeling_swin2sr.py
Any missing files or assumptions: no gated/401 configs encountered for the sampled caidas checkpoints. This report targets in-library Transformers source, not remote code or the original BasicSR implementation.
```

Representative config snapshot: [config_sweep.md](config_sweep.md).

Primary runtime target: `Swin2SRForImageSuperResolution` inference for image super-resolution/restoration. `Swin2SRModel` is an independently useful encoder/body target that returns restored feature maps, but end-to-end parity requires the reconstruction head and post-crop/renormalization path.

## 2. High-level architecture

Swin2SR is a vision restoration model, not an autoregressive text model. The neural body is an NCHW convolutional stem, an internal sequence/window-attention SwinV2-style encoder, and an image reconstruction head.

```text
image preprocessing -> NCHW pixel_values
  -> model reflect pad + mean/range normalize
  -> 3x3 conv stem
  -> patch embed as [B,H*W,C]
  -> repeated residual Swin transformer stages
  -> layer norm + unembed to NCHW
  -> 3x3 body residual conv
  -> upsampler/restoration head
  -> divide by img_range + add mean
  -> crop to height*upscale,width*upscale
```

Stage decomposition:

- CPU/data pipeline: image decode, channel-first conversion, rescale by `1/255`, optional symmetric pad to a size divisor.
- GPU/runtime body: NCHW conv stem, NHWC-like window partitioning inside attention blocks, MLP, residual conv per stage, body residual conv.
- Head: selected by `config.upsampler`; may use PixelShuffle, nearest interpolation, bicubic auxiliary input, or direct residual restoration.
- Postprocess ABI: model returns NCHW float reconstruction. Example source clamps to `[0,1]`, moves channel to last axis, multiplies by 255, and rounds in user code, not inside model.

Independently stageable pieces: processor parity, `Swin2SRModel` body parity on already padded tensors, each upsampler head variant, and final reconstruction crop/renormalization.

## 3. Important config dimensions

| Field | Source default | Sampled production values | Runtime impact |
| --- | ---: | --- | --- |
| `image_size` | 64 | 48 or 64 | build-time patch-grid default; runtime accepts image-shaped tensors but shape assumptions depend on window divisibility |
| `patch_size` | 1 | 1 | patch embedding is effectively a 1x1 conv over stem features |
| `num_channels` | 3 | 3 | input NCHW channels; JPEG dynamic converter path uses 1 but was not sampled as an HF repo |
| `num_channels_out` | defaults to `num_channels` | omitted in sampled configs, effective 3 | reconstruction channels |
| `embed_dim` | 180 | 60 or 180 | hidden width and conv body width |
| `depths` | `[6,6,6,6,6,6]` | four or six stages | total Swin layer count |
| `num_heads` | `[6,6,6,6,6,6]` | all 6 in sampled configs | per-stage MHA heads |
| `head_dim` | inferred `embed_dim / heads` | 10 for lightweight, 30 for full | source requires divisibility |
| `window_size` | 8 | 8 | local attention window and padding divisor |
| `mlp_ratio` | 2.0 | 2.0 | FFN width is `2*embed_dim` |
| `qkv_bias` | true | true | query and value have bias; key is bias-free in source |
| `hidden_act` | `gelu` | `gelu` | MLP activation |
| `resi_connection` | `1conv` | `1conv` | stage residual conv; `3conv` source path exists |
| `upsampler` | `pixelshuffle` | `pixelshuffle`, `pixelshuffle_aux`, `pixelshuffledirect`, `nearest+conv` | major head/operator variation |
| `upscale` | 2 | 2 or 4 | output crop shape and upsampler expansion |
| `img_range` | 1.0 | 1.0 | normalization scale |
| `use_absolute_embeddings` | false | false | optional learned position table exists but sampled configs disable it |

Representative checkpoint sweep:

| Model id | Task flavor | Width/depth | Head | Output scale |
| --- | --- | --- | --- | --- |
| `caidas/swin2SR-classical-sr-x2-64` | classical SR | `C=180`, 6x6 layers | multi-step PixelShuffle | x2 |
| `caidas/swin2SR-classical-sr-x4-64` | classical SR | `C=180`, 6x6 layers | multi-step PixelShuffle | x4 |
| `caidas/swin2SR-compressed-sr-x4-48` | compressed SR | `C=180`, 6x6 layers | PixelShuffle aux plus bicubic branch | x4 |
| `caidas/swin2SR-lightweight-x2-64` | lightweight SR | `C=60`, 4x6 layers | one-step PixelShuffle | x2 |
| `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | real-world SR | `C=180`, 6x6 layers | nearest+conv | x4 |

## 3a. Family variation traps

- `upsampler` changes the head graph. Do not compile a single "Swin2SR head" without routing on `pixelshuffle`, `pixelshuffle_aux`, `pixelshuffledirect`, `nearest+conv`, and fallback restoration.
- `pixelshuffle_aux` consumes an extra bicubic upsample of the original input and returns an auxiliary map internally. End-to-end output is still `reconstruction`; training labels are not supported.
- `nearest+conv` rejects `upscale != 4` in source.
- `Upsample` supports power-of-two scales and scale 3, but sampled configs use x2/x4. Non-power-of-two except 3 should be rejected for that head.
- The converter has a JPEG dynamic branch with `num_channels=1`, `upscale=1`, `image_size=126`, `window_size=7`, `img_range=255.0`, and empty `upsampler`. Treat this as a follow-up unless an accessible HF config is selected.
- Source uses NCHW for convs and output, but transforms to `[B,H,W,C]` inside window partition/roll/reverse. Layout translation must be region-scoped and guarded.
- Processor pad and model pad are different: processor uses symmetric pad to the next multiple of `size_divisor`; model uses reflect pad modulo `window_size`. Shape/crop parity depends on where original image size is preserved.
- Attention mask generation and window partition assume padded height/width divisible by `window_size`.
- Source attention adds the shifted-window attention mask twice in the inspected commit. For source parity, reproduce or explicitly test any intentional correction.
- `path_norm` appears in sampled old configs but the inspected `Swin2SRConfig` does not define or read it; treat it as ignored legacy config data for this source basis.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `Conv2d` with 3x3 padding 1, 1x1 padding 0, and patch `kernel=stride=patch_size`.
- `flatten(2)`, `transpose(1,2)`, sequence-to-NCHW `view`, NCHW-to-sequence reshape.
- NHWC-style `view`, `transpose(2,3)`, `contiguous`, and reverse for window partition/reverse.
- `torch.roll` over spatial axes for shifted windows.
- `pad`: processor symmetric pad; model reflect pad for pixels; hidden-state zero pad through `nn.functional.pad`.
- Strided slicing for patch merging source path and crop `[:, :, :H*scale, :W*scale]`.
- `cat` along channel/last axis for patch merging if admitted.
- `PixelShuffle(scale)` for x2, x3, x4 through repeated x2 or direct scale.
- `interpolate`: nearest x2 twice for real-world head; bicubic resize to exact output size for aux head with `align_corners=False`.

Neural primitives:

- `LayerNorm(C)` over sequence last dimension.
- Linear projections: Q `C->C` with bias, K `C->C` without bias, V `C->C` with bias, attention output `C->C`, MLP `C->2C->C`.
- GELU activation in MLP; ReLU in continuous position bias MLP; LeakyReLU in upsamplers and optional `3conv` residual path.
- Residual adds.
- Dropout/DropPath are inference identity for `.eval()`; first integration can require eval mode.

Attention primitives:

- Local noncausal window self-attention over `W^2` tokens.
- Cosine attention: normalize Q and K along head dim, matmul, multiply by clamped learned logit scale.
- Continuous relative position bias: small MLP over precomputed coordinate table, gather by relative index, sigmoid, scale by 16.
- Shifted-window attention mask with values `0` and `-100`.
- Softmax over key/window axis, then attention-probability matmul with V.

Preprocessing-coupled ops:

- Rescale image to float by `1/255`.
- Optional symmetric pad to divisor 8 in processor.
- Model mean subtract/add with mean `[0.4488,0.4371,0.4040]` only for 3 input and 3 output channels; otherwise zero mean.
- `img_range` multiply before body and divide after head.

Postprocess ABI:

- Model output is `ImageSuperResolutionOutput.reconstruction`, NCHW float, cropped to `height*upscale,width*upscale` where `height,width` are the tensor dimensions received by `Swin2SRForImageSuperResolution.forward`.
- No NMS, boxes, masks, generation, tokenizer, KV cache, or recurrent state.

## 5. Layer/block breakdown

Stem and body:

```text
pixel_values: [B,3,H,W]
pixel_values = reflect_pad_to_window(pixel_values)
x0 = (pixel_values - mean) * img_range
embeddings = Conv2d(3 -> C, 3x3, pad=1)(x0)             # [B,C,Hp,Wp]
tokens = Conv2d(C -> C, patch_size, stride=patch_size)(embeddings)
tokens = flatten_hw(tokens).transpose(1,2)              # [B,Hp*Wp,C]
tokens = LayerNorm(C)(tokens) if normalize_patches
```

Swin2SR layer, repeated per stage:

```text
shortcut = tokens
y = tokens.view(B,H,W,C)
y = pad_to_window_multiple(y)
y = roll(y, -shift, spatial) if shifted
windows = window_partition(y, window_size).view(B*nW, W*W, C)
mask = shifted_window_mask(Hpad,Wpad) if shifted
attn = cosine_window_attention(windows, mask)
y = window_reverse(attn).roll(+shift) if shifted
y = crop_pad(y).view(B,H*W,C)
y = LayerNorm(C)(y)
y = shortcut + DropPath(y)
z = Linear(C -> 2C)(y)
z = GELU(z)
z = Linear(2C -> C)(z)
tokens = y + DropPath(LayerNorm(C)(z))
```

Residual Swin Transformer Block stage:

```text
residual = tokens
for layer in depth:
    tokens = Swin2SRLayer(tokens)
image = tokens.transpose(1,2).view(B,C,H,W)
image = stage_conv(image)       # 1conv: Conv2d(C->C,3,pad=1); 3conv optional
tokens = Conv2d(C->C,patch,stride)(image).flatten(2).transpose(1,2)
tokens = tokens + residual
```

After all stages:

```text
tokens = LayerNorm(C)(tokens)
body = tokens.transpose(1,2).view(B,C,H,W)
body = Conv2d(C -> C, 3x3, pad=1)(body) + stem_embeddings
```

Heads:

- `pixelshuffle`: `Conv2d(C->64) -> LeakyReLU -> repeated [Conv2d(64->4*64), PixelShuffle(2)] for x2/x4 or `[Conv2d(64->9*64), PixelShuffle(3)] for x3 -> Conv2d(64->out)`.
- `pixelshuffledirect`: `Conv2d(C -> scale^2*out, 3x3,pad=1) -> PixelShuffle(scale)`.
- `nearest+conv`: `Conv2d(C->64) -> LeakyReLU -> nearest x2 -> Conv2d -> LeakyReLU -> nearest x2 -> Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> final Conv2d(64->out)`.
- `pixelshuffle_aux`: bicubic original input to target size; body branch `Conv2d(C->64) -> LeakyReLU -> Conv2d(64->3) -> Conv2d(3->64)+LeakyReLU -> upsample -> crop/add bicubic conv branch -> final Conv2d`.
- fallback restoration: `pixel_values + Conv2d(C->out)(body)`, then renormalize/crop.

## 6. Attention requirements

Swin2SR uses encoder-style noncausal local self-attention, not generation attention.

| Property | Requirement |
| --- | --- |
| Causal | No |
| Attention type | Window self-attention with alternating shifted windows |
| MHA/GQA/MQA | MHA only; Q/K/V heads equal |
| Heads/head dim | sampled full models: 6 heads x 30; lightweight: 6 heads x 10 |
| Query/key/value widths | all `C`, split to `[B*nW, heads, window_size^2, head_dim]` |
| Masking | shifted-window mask, additive `-100` for cross-region pairs |
| Local/window | yes, fixed square `window_size`; source pads spatial dims to multiples |
| Packed/varlen | none |
| Relative bias | continuous SwinV2 relative position bias generated from coordinate table |
| KV cache | none |
| FlashAttention/SDPA | not directly compatible without a custom local-window packing and relative-bias/mask path |

The dense attention problem per window is small (`64x64` for default window 8), so a specialized window-attention kernel may matter more than generic full-sequence FlashAttention.

## 7. Position encoding and custom math

No RoPE or ALiBi. The model uses SwinV2 continuous relative position bias per attention module. The coordinate table and relative index are buffers, not persistent weights; the MLP parameters are weights.

Concise source-equivalent math:

```python
def swin2sr_relative_bias(coords_table, relative_index, cpb_mlp, heads, window):
    # coords_table: [1, 2*Wh-1, 2*Ww-1, 2]
    table = cpb_mlp(coords_table).view(-1, heads)
    bias = table[relative_index.reshape(-1)]
    bias = bias.view(window * window, window * window, heads)
    bias = bias.permute(2, 0, 1).contiguous()
    return 16.0 * sigmoid(bias)  # [heads, W^2, W^2]
```

Coordinate normalization at init:

```python
coords = sign(coords) * log2(abs(coords * 8 / denom) + 1) / log2(8)
```

`denom` uses `pretrained_window_size - 1` when positive, otherwise `window_size - 1`. For sampled configs, `pretrained_window_size=0`, so the current window size controls the table. The table/index can be precomputed per config/window/head dtype; the CPB MLP output depends on weights but not on image content.

## 8. Preprocessing and input packing

Processor contract:

- Input images are converted to channel-first tensors by the image backend, then rescaled by `1/255` when `do_rescale=true`.
- Official sampled preprocessors set `do_pad=true` and legacy `pad_size=8`; current processor maps `pad_size` to `size_divisor`.
- Torchvision backend groups images by shape before processing and returns `pixel_values`; tests note batched processing requires equal-resolution images.
- Padding mode differs by backend implementation name but intended behavior is symmetric pad in processor.

Model-coupled preprocessing:

- `Swin2SRModel.forward` records `height,width` from the received tensor before internal padding.
- It reflect-pads to multiples of `config.window_size`, subtracts mean, and multiplies by `img_range`.
- For RGB-to-RGB configs, mean is `[0.4488, 0.4371, 0.4040]` in NCHW broadcast shape; otherwise zero.

Input/output shapes:

```text
pixel_values: [B, C_in, H, W] float
body output: [B, embed_dim, H, W] when H/W already window-divisible
reconstruction: [B, C_out, H*upscale, W*upscale] after crop
```

Important ABI caveat: if the processor pads the image, model output shape is scaled from the padded processor shape, not necessarily the original unpadded image. The source example only shows user-side clamp/convert, not an automatic crop back to original size.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch embedding with `patch_size=1` to 1x1 Conv or Linear

Source pattern: `Conv2d(C -> C, kernel=1, stride=1)` followed by `flatten(2).transpose(1,2)`.

Replacement: keep as 1x1 conv in NCHW or lower to per-pixel `Linear(C->C)` after controlled NHWC transform.

Preconditions:

- `patch_size == 1`.
- NCHW tensor is contiguous or layout pass owns the full conversion.
- Consumer accepts `[B,H*W,C]`.

Failure cases: non-1 patch size, dynamic layout with unknown stride, or absolute embeddings whose sequence length does not match runtime shape.

Parity test: random NCHW tensor through Conv2d+flatten path versus rewrite for fp32/fp16.

### Rewrite: window partition/reverse as layout views plus batched window attention

Source pattern: `[B,H,W,C] -> view(B,H/Ws,Ws,W/Ws,Ws,C) -> transpose(2,3) -> [B*nW,Ws*Ws,C]`, then reverse.

Replacement: explicit `window_partition`/`window_reverse` layout op or fused local-window attention kernel.

Preconditions:

- `Hpad % window_size == 0` and `Wpad % window_size == 0`.
- Window size static.
- Padding/crop boundaries preserved.

Layout constraints: this region is NHWC-like. A NCHW-optimized pass must rewrite axes consistently and may need a `no_layout_translation` guard around source view/transpose semantics.

Failure cases: arbitrary strided input, non-square windows not represented in config, missing crop after pad.

Parity test: round-trip partition/reverse for odd and divisible H/W, then attention-block parity.

### Rewrite: Q/K/V separate Linear to grouped GEMM

Source pattern: query, key, value are separate `Linear(C,C)` modules; Q and V have bias, K is bias-free.

Replacement: one packed GEMM producing `[Q,K,V]` followed by splits.

Preconditions:

- Preserve split order Q, K, V.
- K bias must be represented as zeros or omitted in epilogue.
- Weight layout converted from PyTorch linear `[out,in]` to DinoML GEMM layout.

Failure cases: checkpoint conversion already split original `qkv`; do not assume a packed weight exists in HF state dict.

Parity test: compare packed projection output against three linears for sampled full and lightweight widths.

### Rewrite: PixelShuffle to reshape-permute-copy

Source pattern: `Conv2d(C -> r^2*Cout) -> PixelShuffle(r)`.

Replacement:

```text
[B, r^2*Cout, H, W]
  -> view [B, Cout, r, r, H, W]
  -> permute [B, Cout, H, r, W, r]
  -> reshape [B, Cout, H*r, W*r]
```

Preconditions:

- Channel dimension divisible by `r^2`.
- NCHW semantic axes preserved.
- Crop after upsample remains in graph.

Failure cases: channel-last layout pass without axis rewrite, unsupported scale, non-contiguous tensors.

Parity test: random tensors for r=2,3,4 and full head parity.

### Rewrite: nearest+conv upsampling as fused resize-conv tile

Source pattern: nearest x2 followed by 3x3 Conv2d, twice.

Replacement: either explicit nearest upsample plus conv or a fused kernel that reads replicated source pixels directly.

Preconditions:

- `mode="nearest"`, `scale_factor=2`.
- Conv stride 1, padding 1, dilation 1.
- `upscale == 4`.

Failure cases: any alternate interpolation mode, dynamic scale, or non-NCHW layout.

Parity test: compare real-world head block at random sizes divisible by window size.

### Rewrite: `pixelshuffle_aux` bicubic branch isolation

Source pattern: bicubic original input to target size, `conv_bicubic`, add to cropped upsampled aux branch, then final conv.

Replacement: stage bicubic as preprocessing/runtime resize op and compile the branch separately.

Preconditions:

- `align_corners=False`.
- Target size is exactly `(H*upscale, W*upscale)` from forward input shape.
- Addition crop extents match both branches.

Failure cases: processor-padded shape versus original shape mismatch; unsupported bicubic kernel parity.

Parity test: compressed x4 checkpoint with a non-square image and known output slice.

## 10. Kernel fusion candidates

Highest priority:

- NCHW Conv2d 3x3/1x1 coverage and autotuned CUDA path: convs dominate stem, stage residuals, and heads.
- Window partition + cosine attention + relative bias: default attention is many small `64x64` attentions; reducing layout traffic is central.
- PixelShuffle head lowering: required for classical and lightweight checkpoints.
- LayerNorm over `[B,H*W,C]` plus residual add: repeated in every Swin layer.

Medium priority:

- Packed QKV GEMM with Q/V bias and K no-bias handling.
- MLP `Linear -> GELU -> Linear` fusion around `2C` hidden width.
- Shifted-window mask generation as cached static metadata per padded H/W/window.
- Nearest upsample plus conv fusion for real-world x4.

Lower priority:

- DropPath and Dropout kernels; inference can treat as identity.
- `PatchMerging`; source class exists but is not used by the current Swin2SR encoder construction.
- Absolute position embeddings; sampled configs disable them.
- `3conv` residual stage path; source supports it but sampled configs use `1conv`.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse `Swin2SRConfig`, reject unsupported `upsampler`/scale combinations, preserve ignored legacy fields as metadata only.
- Load NCHW conv and linear weights; maintain split Q/K/V names from HF checkpoints.

Stage 2: processor and shape ABI.

- Implement or compose rescale and pad preprocessing.
- Define whether DinoML runtime receives already padded tensors or owns model pad. First integration should require H/W divisible by `window_size` and document processor-owned padding.

Stage 3: body-only parity.

- Compile `Swin2SRModel` for `pixel_values -> last_hidden_state` using NCHW convs, sequence LayerNorm, window attention, and residual stages.

Stage 4: first full checkpoint.

- Target `caidas/swin2SR-lightweight-x2-64` or classical x2. Lightweight has smaller width/depth but requires direct PixelShuffle; classical x2 exercises the common full-width body and simple PixelShuffle head.

Stage 5: add remaining heads.

- Add x4 PixelShuffle, `nearest+conv`, and `pixelshuffle_aux` with bicubic branch. Keep each behind config guards.

Stage 6: optimize.

- Fuse window attention/layout paths, QKV projection, LayerNorm/residual, PixelShuffle reshape, and resize-conv.

Stage 7: production packaging.

- Add fp16 path, shape buckets by H/W/window count, cache static relative-position bias outputs, and processor-output crop metadata.

Initially stubbable: hidden-state/attention optional outputs, training loss, gradient checkpointing, DropPath randomness, and user-side uint8 conversion.

## 12. Parity and validation plan

- Processor parity: PIL and torchvision processor output for odd/even image sizes, rescale and pad enabled, compare shapes and border values.
- Custom layout tests: `window_partition`/`window_reverse` round trip for `window_size=7,8` and non-square H/W after pad.
- Attention unit tests: one `Swin2SRSelfAttention` module with random weights, fp32 tolerance `rtol=1e-4, atol=1e-5`; include shifted mask and no-mask cases.
- Relative bias tests: compare coordinate table/index and CPB MLP gather output.
- Layer parity: one `Swin2SRLayer` for shift 0 and shift `window//2`.
- Stage parity: one RSTB stage including residual conv and patch embed/unembed.
- Body parity: `Swin2SRModel` on small padded images.
- Head parity: one test per `upsampler` using random body feature maps.
- End-to-end checkpoint parity: use HF slow-test reference for `caidas/swin2SR-classical-sr-x2-64`; expected reconstruction shape `[1,3,976,1296]` and slice tolerance from upstream tests.
- fp16 parity: source test uses `rtol=2e-4, atol=2e-4` for expected slice; broader tensor comparisons may need looser tolerances around bicubic/attention softmax.

## 13. Performance probes

- Processor throughput: decode/rescale/pad time versus runtime time for common image sizes.
- Body-only throughput by image size: H/W sweep at multiples of 8.
- Window-attention microbench: number of windows, window size 8, `C=60/180`, heads 6.
- Layout traffic probe: NCHW conv to NHWC window path and back, before/after fusion.
- Conv head probe: PixelShuffle x2/x4, nearest+conv x4, and pixelshuffle_aux x4.
- Bicubic branch probe for compressed SR.
- Batch-size sweep with equal-size batches; processor grouping behavior for mixed sizes.
- fp32 versus fp16 reconstruction parity and throughput.
- Memory probe: temporary windows `[B*nW,64,C]`, attention scores `[B*nW,heads,64,64]`, and PixelShuffle intermediates.

## 14. Skip/defer list

- Training and labels: source raises `NotImplementedError` when labels are supplied.
- Gradient checkpointing and stochastic DropPath behavior.
- Multi-GPU/data parallel issues.
- Hidden-state and attention output materialization for first end-to-end runtime, unless needed for debugging.
- JPEG dynamic grayscale checkpoint until an accessible target config/checkpoint is selected.
- `3conv` residual connection and absolute embeddings unless a checkpoint requires them.
- General NHWC translation of the whole model; start with faithful NCHW semantics and guarded local layout rewrites.
- General dense/global attention, KV cache, generation, tokenizer, detection/segmentation postprocess.

## 15. Final implementation checklist

- [ ] Parse `Swin2SRConfig` and checkpoint configs.
- [ ] Reject unsupported `upsampler`, `upscale`, `resi_connection`, and legacy remote-only fields with clear errors.
- [ ] Load Conv2d/Linear/LayerNorm weights, including split Q/K/V HF names.
- [ ] Implement processor-compatible rescale and pad or require preprocessed `pixel_values`.
- [ ] Implement NCHW Conv2d coverage for 3x3/1x1 and static padding.
- [ ] Implement sequence LayerNorm and MLP GELU block.
- [ ] Implement `window_partition`, `window_reverse`, shifted roll, crop, and mask.
- [ ] Implement Swin2SR cosine window attention with clamped logit scale and continuous relative position bias.
- [ ] Implement PixelShuffle lowering for scale 2/3/4.
- [ ] Implement nearest interpolation x2 and bicubic resize for head variants.
- [ ] Add body-only parity tests against `Swin2SRModel`.
- [ ] Add one end-to-end checkpoint parity test for classical x2.
- [ ] Add config-specific parity for lightweight, compressed aux, and real-world heads.
- [ ] Add performance probes for window attention, conv heads, and layout traffic.
