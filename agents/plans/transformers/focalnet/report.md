# FocalNet DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/focalnet-tiny, microsoft/focalnet-tiny-lrf, microsoft/focalnet-small, microsoft/focalnet-small-lrf, microsoft/focalnet-base, microsoft/focalnet-base-lrf
Config source: HF config.json and preprocessor_config.json snapshots under _sources/hf_configs/
Source files inspected:
  X:/H/transformers/src/transformers/models/focalnet/configuration_focalnet.py
  X:/H/transformers/src/transformers/models/focalnet/modeling_focalnet.py
  X:/H/transformers/src/transformers/models/focalnet/convert_focalnet_to_hf_format.py
  X:/H/transformers/src/transformers/models/bit/image_processing_bit.py
  X:/H/transformers/src/transformers/models/bit/image_processing_pil_bit.py
  X:/H/transformers/src/transformers/image_processing_backends.py
Any missing files or assumptions:
  No family-specific image processor exists; AutoImageProcessor maps focalnet to BitImageProcessor.
  microsoft/focalnet-base-simmim-window6-192 returned 401 for config/preprocessor snapshots, so SimMIM checkpoint-specific values are not confirmed.
  microsoft/focalnet-large-lrf-fl3 and microsoft/focalnet-large-lrf-fl4 repos were visible through model search but config/preprocessor resolve returned 404.
```

Primary runtime target: image classification and backbone feature extraction. Masked image modeling is implemented and documented as optional/deferred for first inference integration.

## 2. High-level architecture

FocalNet is a vision encoder with no self-attention. It replaces attention with focal modulation: channel-last token projections produce query/context/gates, context is aggregated by repeated depthwise NCHW convolutions plus global average context, then a pointwise convolution produces a modulator multiplied with the query.

```text
image preprocessing -> NCHW pixel_values -> patch Conv2d stem -> token LayerNorm
  -> 4-stage focal modulation encoder with optional downsample patch Conv2d
  -> final LayerNorm -> AdaptiveAvgPool1d -> classifier logits
```

Backbone mode returns selected NCHW feature maps from reshaped hidden states. Masked image modeling adds a mask token before the encoder and a `1x1 Conv2d -> PixelShuffle(encoder_stride)` reconstruction head.

## 3. Important config dimensions

| Field | Default / behavior |
|---|---:|
| `image_size` | 224 |
| `patch_size` | 4 |
| `num_channels` | 3 |
| `embed_dim` | 96 |
| executed stage widths | `embed_dim * [1,2,4,8]` for four default stages |
| `hidden_sizes` | `(192,384,768,768)`, used by `FocalNetBackbone.num_features`, not by encoder stage construction |
| `depths` | `(2,2,6,2)` |
| `focal_levels` | `(2,2,2,2)` |
| `focal_windows` | `(3,3,3,3)` |
| focal kernel sizes | levels 2 -> `[3,5]`, levels 3 -> `[3,5,7]` |
| `mlp_ratio` | 4.0 |
| `hidden_act` | GELU in sampled configs |
| `layer_norm_eps` | `1e-5` |
| `drop_path_rate` | 0.1 in sampled configs, identity at inference |
| `use_conv_embed` | false in sampled configs; source supports true |
| `use_layerscale` | false in sampled configs; source supports true |
| cache support | none; non-generation encoder |

Representative checkpoint sweep:

| Checkpoint | Task/head | Preprocess | `embed_dim` | executed widths | `depths` | focal levels | focal windows | notable variation |
|---|---|---|---:|---|---|---|---|---|
| `microsoft/focalnet-tiny` | classification | resize shortest 256, center crop 224 | 96 | 96/192/384/768 | 2/2/6/2 | 2/2/2/2 | 3/3/3/3 | standard focal levels |
| `microsoft/focalnet-tiny-lrf` | classification | same | 96 | 96/192/384/768 | 2/2/6/2 | 3/3/3/3 | 3/3/3/3 | adds 7x7 depthwise level |
| `microsoft/focalnet-small` | classification | same | 96 | 96/192/384/768 | 2/2/18/2 | 2/2/2/2 | 3/3/3/3 | deeper stage 3 |
| `microsoft/focalnet-small-lrf` | classification | same | 96 | 96/192/384/768 | 2/2/18/2 | 3/3/3/3 | 3/3/3/3 | deeper plus LRF |
| `microsoft/focalnet-base` | classification | same | 128 | 128/256/512/1024 | 2/2/18/2 | 2/2/2/2 | 3/3/3/3 | config `hidden_sizes` does not match executed widths |
| `microsoft/focalnet-base-lrf` | classification | same | 128 | 128/256/512/1024 | 2/2/18/2 | 3/3/3/3 | 3/3/3/3 | base plus LRF |

## 3a. Family variation traps

- There is no attention path. Do not allocate KV cache or FlashAttention work for this family.
- `hidden_sizes` is not authoritative for encoder widths. `FocalNetStage` computes `embed_dim * 2**stage`; `FocalNetBackbone.num_features` uses `[config.embed_dim] + config.hidden_sizes`, which can be metadata-inconsistent for base checkpoints.
- `use_conv_embed=True` changes patch/downsample convolutions from non-overlap `kernel=stride` to overlapping `7x7 stride 4 padding 2` stem and `3x3 stride 2 padding 1` downsample. Sampled public configs set it false.
- `focal_levels` changes depthwise convolution count and `projection_in` output width: `2*C + focal_level + 1`.
- `use_post_layernorm`, `use_post_layernorm_in_modulation`, `normalize_modulator`, and `use_layerscale` are source-supported even though sampled configs set them false.
- Modeling source alternates token NHWC-ish tensors `[B,H*W,C]` / `[B,H,W,C]` with NCHW convolution tensors. Treat layout changes as local guarded optimizations, not whole-graph NHWC translation.
- MIM source assumes square final token grids via `height = width = floor(sqrt(sequence_length))`; non-square dynamic inputs should be rejected or separately validated for MIM.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW input validation, right/bottom zero padding to patch divisibility, Conv2d output shape tracking.
- Flatten spatial: `[B,C,H,W] -> [B,H*W,C]` via flatten+transpose.
- Token reshape: `[B,H*W,C] -> [B,H,W,C]`.
- Channel-last to NCHW and back around modulation: `permute(0,3,1,2).contiguous()` and `permute(0,2,3,1).contiguous()`.
- Stage downsample reshape: `[B,H*W,C] -> [B,C,H,W]`.
- Backbone feature reshape: `[B,H*W,C] -> [B,C,H,W]`.
- Split along channel axis after projected NCHW tensor: q, ctx, gates.
- Global spatial reductions over NCHW `H` then `W`, keepdim.

### Neural network primitives

- Standard patch stem: `Conv2d(3 -> E, kernel=4, stride=4, padding=0, bias=True)`.
- Optional conv embedding stem: `Conv2d(3 -> E, kernel=7, stride=4, padding=2, bias=True)`.
- Standard downsample: `Conv2d(C -> 2C, kernel=2, stride=2, padding=0, bias=True)`, then LayerNorm over `2C`.
- Optional conv downsample: `Conv2d(C -> 2C, kernel=3, stride=2, padding=1, bias=True)`.
- LayerNorm over last channel for token/channel-last tensors.
- Modulation input: `Linear(C -> 2*C + focal_level + 1, bias=True)`.
- Depthwise Conv2d focal layers: `Conv2d(C -> C, groups=C, kernel=3/5[/7], stride=1, padding=kernel//2, bias=False)`, then GELU.
- Pointwise context projection: `Conv2d(C -> C, kernel=1, stride=1, bias=True)`.
- Elementwise gate multiply, context accumulation, optional divide by `focal_level + 1`, query-modulator multiply.
- Output projection: `Linear(C -> C, bias=True)`.
- MLP: `Linear(C -> int(C*mlp_ratio)) -> GELU -> Linear(int(C*mlp_ratio) -> C)`.
- Residual adds, optional layer-scale vectors `gamma_1/gamma_2` of shape `[C]`.
- Final `LayerNorm(C_last)`, `AdaptiveAvgPool1d(1)` over sequence length, flatten, classifier `Linear(C_last -> num_labels)`.
- Optional MIM head: `Conv2d(C_last -> encoder_stride^2 * num_channels, kernel=1) -> PixelShuffle(encoder_stride)`.

### Attention / position / cache ops

None. No attention masks, RoPE, ALiBi, relative bias, or KV cache are required.

### Preprocessing-coupled ops

- `BitImageProcessor`: RGB conversion, resize shortest edge to 256, center crop 224x224, rescale by `1/255`, normalize by ImageNet mean/std, output channels-first `pixel_values`.
- Dynamic model padding for non-divisible input height/width occurs inside `FocalNetPatchEmbeddings`; processor crop normally avoids it for sampled classification checkpoints.

## 5. Layer/block breakdown

For `[B,3,224,224]`, the standard stem produces `[B,56*56,E]`.

```text
Patch embeddings:
  x_nchw = pad_right_bottom(pixel_values, multiple=patch_size)
  x = Conv2d(3 -> E, kernel=4, stride=4)(x_nchw)
  x = flatten_hw_transpose_to_tokens(x)        # [B, 56*56, E]
  x = LayerNorm(E)(x)
```

FocalNet block, repeated `depths[stage]` times at width `C = embed_dim * 2**stage`:

```text
shortcut = x_tokens                              # [B, H*W, C]
u = LayerNorm(C)(x_tokens)                       # pre-LN unless use_post_layernorm
u = view(u, [B, H, W, C])
p = Linear(C -> 2*C + focal_level + 1)(u)
p = permute_to_nchw(p)
q, ctx, gates = split_channel(p, [C, C, focal_level + 1])
ctx_all = 0
for level, kernel in enumerate([3,5] or [3,5,7]):
    ctx = GELU(DepthwiseConv2d(C, kernel, padding=kernel//2)(ctx))
    ctx_all += ctx * gates[:, level:level+1]
ctx_global = GELU(mean_hw(ctx, keepdim=True))
ctx_all += ctx_global * gates[:, focal_level:]
if normalize_modulator:
    ctx_all = ctx_all / (focal_level + 1)
modulator = Conv2d(C -> C, kernel=1)(ctx_all)
u = q * modulator
u = permute_to_bhwc(u)
if use_post_layernorm_in_modulation:
    u = LayerNorm(C)(u)
u = Linear(C -> C)(u).view(B, H*W, C)
if use_post_layernorm:
    u = LayerNorm(C)(u)
x = shortcut + gamma_1 * u
mlp_in = MLP(LayerNorm(C)(x))                    # or LayerNorm(MLP(x)) in post-LN mode
x = x + gamma_2 * mlp_in
```

Stage boundary, for stages 0-2:

```text
tokens_before_downsample = x                     # [B, H*W, C]
x = transpose_reshape_to_nchw(x, [B,C,H,W])
x = Conv2d(C -> 2C, kernel=2, stride=2)(x)       # standard sampled configs
x = flatten_hw_transpose_to_tokens(x)
x = LayerNorm(2C)(x)
```

Classification:

```text
x = final LayerNorm(C_last)(x)
pooled = AdaptiveAvgPool1d(1)(transpose(x, [B,C_last,L]))
logits = Linear(C_last -> num_labels)(flatten(pooled))
```

## 6. Attention requirements

No attention is required. There is no causal or noncausal attention, no self-attention or cross-attention, no MHA/MQA/GQA, no attention mask, no packed/varlen attention, no relative bias/RoPE, and no KV cache. Focal modulation is convolutional and gate-based; optimizing it belongs under convolution/layout/fusion work rather than attention backend work.

## 7. Position encoding and custom math

There are no learned absolute position embeddings, RoPE, ALiBi, or relative attention biases. The custom math to reproduce is focal modulation:

```python
def focal_modulation(x_bhwc, proj_in, depthwise_layers, proj_context, proj_out, levels, normalize):
    c = x_bhwc.shape[-1]
    p = proj_in(x_bhwc).permute(0, 3, 1, 2).contiguous()
    q, ctx, gates = split_channels(p, [c, c, levels + 1])
    ctx_all = 0
    for level, dwconv_gelu in enumerate(depthwise_layers):
        ctx = dwconv_gelu(ctx)
        ctx_all = ctx_all + ctx * gates[:, level : level + 1]
    ctx_global = gelu(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
    ctx_all = ctx_all + ctx_global * gates[:, levels:]
    if normalize:
        ctx_all = ctx_all / (levels + 1)
    out = q * proj_context(ctx_all)
    return proj_out(out.permute(0, 2, 3, 1).contiguous())
```

Depthwise kernels, projection weights, and layer-scale vectors are static. Gates and global context depend on the current batch/image and cannot be precomputed across inputs.

## 8. Preprocessing and input packing

FocalNet uses `pixel_values` only. Sampled preprocessors are `BitImageProcessor` with RGB conversion, resize shortest edge to 256, center crop to `224x224`, rescale by `0.00392156862745098`, ImageNet normalization, and channels-first output `[B,3,224,224]`.

There is no text, token packing, placeholder token, attention mask, or generation controller. Backbone output selection is controlled by `BackboneConfigMixin` `out_features/out_indices`; feature maps are returned as NCHW tensors from `outputs.reshaped_hidden_states`.

For MIM, `bool_masked_pos` has shape `[B, num_patches]`, where `num_patches=(image_size//patch_size)^2` in examples. The model replaces corresponding patch embeddings with a learned mask token before the encoder. MIM loss and L1 masking are training/eval-parity concerns, not required for first classification inference.

## 9. Graph rewrite / lowering opportunities

### Rewrite: standard patch/downsample Conv2d -> WindowFlatten + GEMM

Source pattern: `Conv2d(Cin -> Cout, kernel=K, stride=K, padding=0, dilation=1, groups=1)` in patch stem and standard downsample.

Replacement:

```text
RightBottomPadIfNeeded -> WindowFlatten(NCHW, KxK, stride K) -> MatMul(weight_flat.T) -> BiasAdd -> ReshapeTokens
```

Preconditions:

- `use_conv_embed == False`.
- `kernel_size == stride == patch_size` for stem or `2` for downsample.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Padding semantics match source: right then bottom only.
- Flatten order matches PyTorch Conv2d channel/kernel order.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases: `use_conv_embed=True`, nonzero symmetric padding, grouped conv, dynamic shape without right/bottom pad parity. Parity test: random NCHW image with odd H/W against source patch embedding output and `output_dimensions`.

### Rewrite: focal modulation channel-last island

Source pattern: `Linear(B,H,W,C) -> permute NCHW -> split -> depthwise conv/gates/global mean -> 1x1 conv -> multiply -> permute BHWC -> Linear`.

Replacement: keep a local optimized NHWC/channel-last island, or fuse the whole modulation into a custom kernel family.

Preconditions:

- All consumers inside the island are controlled through `FocalNetModulation`.
- LayerNorm axis remains last channel before and after the island.
- Depthwise convolution, gates, global reductions, and 1x1 projection preserve source padding and axes.
- Output materializes back to `[B,H*W,C]` tokens before residual/MLP unless the surrounding block is also admitted to the same layout region.

Layout constraints: source NCHW operations use `dim=1` for q/ctx/gates split and gate slicing, spatial means over dims 2 and 3. A layout pass must rewrite these axes if stored NHWC. Use a conceptual `no_layout_translation()` guard around backbone outputs and public NCHW feature maps.

Failure cases: optional `use_post_layernorm_in_modulation`, `normalize_modulator`, or non-default focal levels are not implemented in the fused kernel; mismatch in gate broadcast axis; public hidden-state consumers request NCHW reshaped states.

Parity test: compare per-block output for levels 2 and 3, with non-square H/W and both `normalize_modulator` settings.

### Rewrite: final AdaptiveAvgPool1d -> ReduceMean over tokens

Source pattern: `AdaptiveAvgPool1d(1)(sequence_output.transpose(1,2)) -> flatten`.

Replacement: `ReduceMean(sequence_output, axis=1)`.

Preconditions: pool output size is exactly 1, sequence tensor is `[B,L,C]`, and no masks are applied.

Failure cases: future masked pooling semantics or hidden-state selection changes. Parity test: random `[B,L,C]` tensors, compare to source pooler.

### Rewrite: MIM PixelShuffle head decomposition

Source pattern: `Conv2d(C_last -> encoder_stride^2 * 3, kernel=1) -> PixelShuffle(encoder_stride)`.

Replacement: `1x1 MatMul/Conv -> depth-to-space`.

Preconditions: MIM target is enabled, `encoder_stride` is positive, output channels are divisible by `encoder_stride^2`, and final token grid is square as source assumes.

Failure cases: non-square token grid, classification-only integration, or training loss parity excluded.

## 10. Kernel fusion candidates

Highest priority:

- Focal modulation fused island: `Linear projection_in + split + depthwise conv levels + gate multiply + global context + 1x1 projection + q multiply + output Linear` dominates the non-MLP work and currently has multiple layout transitions.
- Depthwise Conv2d + GELU + gated accumulation for kernel sizes 3/5/7. LRF checkpoints add the 7x7 level.
- Patch/downsample Conv2d to GEMM for `use_conv_embed=False` checkpoints, especially because sampled public configs all use non-overlap kernels.

Medium priority:

- LayerNorm + MLP Linear/GELU/Linear residual fusion over `[B,L,C]`.
- Final LayerNorm + token ReduceMean + classifier for classification throughput.
- Layout-pass elimination of `transpose/reshape/permute/contiguous` between token and convolution views under strict axis guards.

Lower priority:

- Optional layer-scale multiply folded into residual kernels.
- MIM `1x1 Conv2d + PixelShuffle` head.
- Training-only DropPath and loss kernels can be ignored for inference.

## 11. Runtime staging plan

1. Parse config and construct widths from `embed_dim * 2**stage`; record `hidden_sizes` only as backbone metadata.
2. Load weights for `FocalNetModel` and run patch embedding plus one focal block parity with random NCHW tensors.
3. Implement full encoder/classification parity for tiny standard levels 2.
4. Add LRF levels 3 coverage and base `embed_dim=128` coverage.
5. Add backbone feature map output contract, including NCHW materialization and `out_features/out_indices` filtering.
6. Add guarded Conv2d-to-GEMM and focal modulation layout/fusion rewrites.
7. Optionally add MIM mask-token and PixelShuffle reconstruction parity once classification/backbone are stable.

Stub initially: training losses, DropPath stochastic behavior, gradient checkpointing, MIM loss, and large/404 repo variants.

## 12. Parity and validation plan

- Random tensor tests for patch embedding `maybe_pad`, including H/W divisible and non-divisible by patch size.
- Focal modulation unit parity for widths 96/192/384/768/1024, levels 2 and 3, kernel sequences `[3,5]` and `[3,5,7]`.
- Single FocalNet block parity with `use_post_layernorm` false/true and `normalize_modulator` false/true.
- Stage parity through downsample for standard `use_conv_embed=False`; add optional conv-embed tests if admitted.
- Full `FocalNetForImageClassification` logits parity for tiny and base configs.
- Backbone parity for selected `out_features`, verifying NCHW feature map shapes and channel widths.
- MIM parity for mask-token replacement and reconstruction head if/when included.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=2e-2, atol=2e-2` around GELU/convolution-heavy paths, tightened after kernel validation.

## 13. Performance probes

- Preprocessing throughput: resize/crop/normalize to `[B,3,224,224]`.
- Patch embedding throughput with Conv2d backend vs GEMM lowering.
- Focal modulation microbench by width, H/W, focal levels, and kernel size sequence.
- Stage-by-stage encoder throughput for tiny/small/base.
- Layout transition overhead: baseline permute/contiguous versus fused channel-last island.
- Classification head throughput and end-to-end images/sec across batch sizes.
- Backbone output materialization cost for different `out_features`.
- Memory bandwidth/temp allocation probe for depthwise conv/gate accumulation.
- Optional MIM PixelShuffle reconstruction throughput.

## 14. Skip/defer list

- Training losses and DropPath stochastic training behavior.
- Gradient checkpointing.
- Masked image modeling loss and SimMIM checkpoint-specific config until the gated checkpoint is accessible: [microsoft/focalnet-base-simmim-window6-192](https://huggingface.co/microsoft/focalnet-base-simmim-window6-192).
- Repos without accessible config files: [microsoft/focalnet-large-lrf-fl3](https://huggingface.co/microsoft/focalnet-large-lrf-fl3), [microsoft/focalnet-large-lrf-fl4](https://huggingface.co/microsoft/focalnet-large-lrf-fl4).
- Multi-GPU/tensor parallelism.
- Quantization; no native quantized/packed path is implemented in the inspected source.
- Attention/KV-cache/generation support, not applicable.

## 15. Final implementation checklist

- [ ] Parse `FocalNetConfig` and derive executed stage widths from `embed_dim`.
- [ ] Load Conv2d, Linear, LayerNorm, layer-scale, classifier, and optional MIM weights.
- [ ] Implement NCHW patch embedding with right/bottom padding.
- [ ] Implement token/channel-last LayerNorm paths.
- [ ] Implement focal modulation with levels 2 and 3, depthwise Conv2d kernels 3/5/7, gates, global context, optional normalization, and optional post modulation LayerNorm.
- [ ] Implement MLP and residual block with optional layer-scale.
- [ ] Implement stage downsample and output dimension propagation.
- [ ] Implement classification pooler as token ReduceMean or AdaptiveAvgPool1d-equivalent.
- [ ] Implement backbone NCHW feature map materialization and `out_features` filtering.
- [ ] Add guarded Conv2d-to-GEMM rewrite for standard non-overlap stem/downsample.
- [ ] Add guarded focal modulation layout/fusion pass with axis rewrite tests.
- [ ] Add parity tests for patch embedding, modulation, one block, full classifier, and backbone outputs.
- [ ] Benchmark focal modulation, patch/downsample lowering, layout transition overhead, and end-to-end images/sec.
