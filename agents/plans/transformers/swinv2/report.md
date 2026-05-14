# Transformers Audit: swinv2

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers` (`2026-05-11`).

Model id: primary checkpoint `microsoft/swinv2-tiny-patch4-window8-256`; representative sweep also covers small, base 22k, base 22k-to-1k fine-tune, and large 22k-to-1k fine-tune.

Config source: local `Swinv2Config` plus downloaded Hugging Face `config.json` and `preprocessor_config.json` snapshots in `_sources/`.

Source files inspected:

- `src/transformers/models/swinv2/configuration_swinv2.py`
- `src/transformers/models/swinv2/modeling_swinv2.py`
- `src/transformers/models/swinv2/convert_swinv2_timm_to_pytorch.py`
- `src/transformers/models/auto/image_processing_auto.py`
- `src/transformers/models/vit/image_processing_vit.py`
- `tests/models/swinv2/test_modeling_swinv2.py`

Source URLs:

- [configuration_swinv2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/swinv2/configuration_swinv2.py)
- [modeling_swinv2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/swinv2/modeling_swinv2.py)
- [image_processing_auto.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/auto/image_processing_auto.py)
- [image_processing_vit.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vit/image_processing_vit.py)

Any missing files or assumptions: there is no SwinV2-specific image processor file; AutoImageProcessor maps `swinv2` to the ViT image processor. This report targets inference for image classification and reusable backbone feature extraction. Masked image modeling is documented as optional/deferred because it adds mask-token replacement, 1x1 Conv2d, PixelShuffle, and L1 loss/postprocessing not needed for classification/backbone parity.

## 2. High-level architecture

SwinV2 is a hierarchical vision encoder with local shifted-window self-attention. The neural body is not autoregressive and has no KV cache.

Dataflow:

```text
PIL/image preprocessing -> NCHW pixel_values -> patch Conv2d -> windowed SwinV2 stages
-> final LayerNorm -> average pool -> classifier logits
```

Backbone dataflow:

```text
pixel_values -> patch Conv2d -> stage1/stage2/stage3/stage4 feature maps
-> selected NCHW feature_maps tuple
```

Main stages:

- CPU/data pipeline: resize to configured square size, rescale by `1/255`, normalize with ImageNet mean/std, emit `pixel_values` in NCHW.
- Stem: non-overlapping patch Conv2d with kernel=stride=`patch_size`, then flatten to `[B, H/patch * W/patch, embed_dim]` and LayerNorm.
- Encoder: 4 stages by default. Each stage repeats SwinV2 blocks with alternating non-shifted and shifted windows. Stages 1-3 end with patch merging.
- Heads: `Swinv2Model` returns sequence and pooled output; `Swinv2ForImageClassification` applies a Linear classifier; `Swinv2Backbone` returns selected pre-downsample NCHW feature maps.

## 3. Important config dimensions

Source defaults from `Swinv2Config`:

| Field | Default | Operator significance |
| --- | ---: | --- |
| `image_size` | 224 | Static grid seed; runtime source pads patch/window regions if needed. |
| `patch_size` | 4 | Patch Conv2d kernel and stride. |
| `num_channels` | 3 | Patch Conv2d input channels. |
| `embed_dim` | 96 | Stem width; stage widths double after patch merging. |
| `depths` | `[2,2,6,2]` | Blocks per stage. |
| `num_heads` | `[3,6,12,24]` | Per-stage MHA heads. |
| `window_size` | 7 | Local attention window side. |
| `pretrained_window_sizes` | `[0,0,0,0]` | Enables log-spaced continuous bias normalization against pretraining window sizes. |
| `mlp_ratio` | 4.0 | FFN hidden width is `mlp_ratio * stage_dim`. |
| `qkv_bias` | true | Query/value bias only in native source; key projection has `bias=False`. |
| `hidden_act` | `gelu` | FFN activation. |
| `use_absolute_embeddings` | false | If true, adds absolute position table and optional bicubic interpolation. |
| `encoder_stride` | 32 | Masked image modeling decoder upsampling factor. |

Representative checkpoint sweep, from downloaded `config.json` snapshots:

| Model id | Task head | image | patch | embed | depths | heads | window | pretrained windows | labels |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: |
| `microsoft/swinv2-tiny-patch4-window8-256` | classification | 256 | 4 | 96 | 2,2,6,2 | 3,6,12,24 | 8 | 0,0,0,0 | 1000 |
| `microsoft/swinv2-small-patch4-window8-256` | classification | 256 | 4 | 96 | 2,2,18,2 | 3,6,12,24 | 8 | 0,0,0,0 | 1000 |
| `microsoft/swinv2-base-patch4-window12-192-22k` | classification | 192 | 4 | 128 | 2,2,18,2 | 4,8,16,32 | 12 | 0,0,0,0 | 21841 |
| `microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft` | classification | 256 | 4 | 128 | 2,2,18,2 | 4,8,16,32 | 16 | 12,12,12,6 | 1000 |
| `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` | classification | 256 | 4 | 192 | 2,2,18,2 | 6,12,24,48 | 16 | 12,12,12,6 | 1000 |

Preprocessor snapshots: legacy `feature_extractor_type=ViTFeatureExtractor`, `do_resize=true`, `size` equal to model image size, `do_normalize=true`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, bilinear-like `resample=3`. Auto mapping in current source routes `swinv2` to `ViTImageProcessor`/`ViTImageProcessorPil`.

## 3a. Family variation traps

- Window size changes attention matrix shape: tiny/small use `8x8` windows, base pretrain uses `12x12`, fine-tuned base/large use `16x16`.
- Fine-tuned `window12to16` configs set `pretrained_window_sizes=[12,12,12,6]`; this changes continuous relative position bias coordinate normalization and must not be ignored.
- `hidden_size` in config is the last stage channel width, not the per-block width for all stages.
- `qkv_bias=true` does not mean all three projections have bias in the native source: query/value can have bias, key is bias-free.
- The conversion helper still writes a split `key.bias` from timm packed qkv bias; DinoML weight loading should key off native module structure and reject or ignore unexpected key-bias tensors deliberately.
- Shifted-window attention is noncausal local self-attention with cyclic roll and an additive mask. It is not sliding-window causal decode attention.
- Source pads at multiple points: input pixels to patch-size divisibility, stage hidden grids to window-size divisibility, and patch merging grids to even H/W.
- Backbone outputs are NCHW feature maps even though attention works in `[B,H,W,C]` windows internally.
- Layout translation is tempting but axis-sensitive: patch Conv2d consumes NCHW; window partition assumes `[B,H,W,C]`; returned feature maps are NCHW; `LayerNorm` is over channel-last token dimension.
- `use_absolute_embeddings` is false in inspected checkpoints. If enabled, first integration needs absolute table add and optional bicubic interpolation.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input validation.
- Constant right/bottom padding for patch divisibility.
- Conv2d patch embedding: `Conv2d(Cin -> embed_dim, kernel=patch, stride=patch, padding=0)`.
- Flatten spatial to sequence and transpose `[B,C,Hp,Wp] -> [B,Hp*Wp,C]`.
- Reshape sequence to grid `[B,H*W,C] -> [B,H,W,C]`.
- Constant right/bottom padding to window multiples.
- `torch.roll` over H/W for shifted windows.
- Window partition/reverse using reshape, transpose, contiguous/view.
- Slice/crop padded regions.
- Patch merging strided slices `0::2/1::2` over H/W, concatenate 4 channel groups, Linear `4C -> 2C`, LayerNorm.
- Backbone reshape/permute `[B,H*W,C] -> [B,C,H,W]`.
- AdaptiveAvgPool1d over sequence length for classification pooling.

Neural network primitives:

- Linear projections for Q, K, V and attention output.
- Linear FFN `C -> mlp_ratio*C -> C`.
- GELU by default through `ACT2FN`.
- LayerNorm over last dimension with `eps=1e-5`.
- Residual adds.
- Dropout/DropPath are identity in inference.
- Classifier Linear `final_dim -> num_labels`.

Attention primitives:

- Local noncausal MHA per window.
- Q/K L2 normalization along head dim.
- Batched attention score matmul `[B*nW, heads, M2, head_dim] @ [B*nW, heads, head_dim, M2]`.
- Learnable `logit_scale`, clamped to `log(100)`, then exponentiated.
- Continuous relative position bias MLP and gather by relative position index.
- Add shifted-window mask with `0` or `-100.0` values; current source adds it twice.
- Softmax over key/window-token axis.
- Attention-probability matmul with V.

Position/relative-bias ops:

- Static coordinate table generation per window/pretrained window size.
- Log-spaced transform `sign(x) * log2(abs(x)+1) / log2(8)`.
- Bias MLP `Linear(2 -> 512) + ReLU + Linear(512 -> heads, bias=False)`.
- Bias post-transform `16 * sigmoid(...)`.

Preprocessing-coupled ops:

- Resize to checkpoint size.
- Rescale and ImageNet normalization.
- Emit NCHW `pixel_values`.

Optional/deferred head ops:

- Masked image modeling: bool mask token mix, 1x1 Conv2d, PixelShuffle(`encoder_stride`), L1 masked reconstruction loss.
- Training losses for classification/regression.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B, 3, H, W]
pad right/bottom to multiples of patch_size
x = Conv2d(3 -> embed_dim, kernel=patch_size, stride=patch_size)
x = flatten spatial, transpose -> [B, Hp*Wp, embed_dim]
x = LayerNorm(embed_dim)
optional mask token blend for masked image modeling
optional absolute position add
```

SwinV2 block at stage width `C`, repeated per configured depth:

```text
shortcut = x                                      # [B, H*W, C]
grid = view(x, [B,H,W,C])
grid = pad to window multiples
grid = roll(grid, [-shift, -shift]) if shifted
windows = partition(grid)                         # [B*nW, window^2, C]
attn = cosine_window_attention(windows, mask)
grid = reverse_windows(attn)
grid = roll(grid, [shift, shift]) if shifted
grid = crop to original H,W
x_attn = view(grid, [B,H*W,C])
x_attn = LayerNorm(C)(x_attn)
x = shortcut + DropPath(x_attn)
mlp = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
x = x + DropPath(LayerNorm(C)(mlp))
```

Patch merging after stages 1-3:

```text
grid = view(x, [B,H,W,C])
grid = pad to even H,W
x0 = grid[:, 0::2, 0::2, :]
x1 = grid[:, 1::2, 0::2, :]
x2 = grid[:, 0::2, 1::2, :]
x3 = grid[:, 1::2, 1::2, :]
merged = concat([x0,x1,x2,x3], dim=-1)             # [B, ceil(H/2), ceil(W/2), 4C]
x = Linear(4C -> 2C, bias=False)(merged)
x = LayerNorm(2C)(x)
```

For tiny at 256x256, stage feature shapes are approximately:

| Stage | Sequence/grid | Channels |
| --- | --- | ---: |
| stem | 64x64 | 96 |
| stage1 output before merge | 64x64 | 96 |
| stage2 output before merge | 32x32 | 192 |
| stage3 output before merge | 16x16 | 384 |
| stage4/final | 8x8 | 768 |

Classification:

```text
x = final LayerNorm([B, 64, 768])
pooled = AdaptiveAvgPool1d(1)(transpose to [B,768,64]) -> [B,768]
logits = Linear(768 -> num_labels)
```

## 6. Attention requirements

Attention type: encoder-only noncausal self-attention within local square windows. No cross-attention, no autoregressive cache, no GQA/MQA, no packed variable-length metadata.

Per stage:

- Tiny/small: heads `[3,6,12,24]`, widths `[96,192,384,768]`, head_dim `32`.
- Base: heads `[4,8,16,32]`, widths `[128,256,512,1024]`, head_dim `32`.
- Large: heads `[6,12,24,48]`, widths `[192,384,768,1536]`, head_dim `32`.

Window attention shapes:

```text
windows: [B * num_windows, Ww*Ww, C]
q/k/v: [B * num_windows, heads, Ww*Ww, head_dim]
scores: [B * num_windows, heads, Ww*Ww, Ww*Ww]
```

Masking:

- Even-numbered blocks in each stage use `shift_size=0`.
- Odd-numbered blocks use `shift_size=window_size//2`, unless source `_compute_window_shift` disables shift when resolution is smaller than window.
- Mask is built after padding to window multiples by assigning 3x3 region ids and comparing ids within each partitioned window.
- Mask values are `0` for same region and `-100.0` for blocked tokens. In this commit the source adds the mask twice in `Swinv2SelfAttention.forward`; preserve this behavior for exact parity or test that replacing it with one add is numerically acceptable.

FlashAttention/SDPA compatibility: vanilla full-sequence SDPA is not the right primitive because attention is window-local after roll/partition and includes continuous per-head relative bias. A useful optimized primitive would be "batched local window cosine attention with additive per-window mask and per-head relative bias".

## 7. Position encoding and custom math

SwinV2 uses continuous relative position bias. The bias depends on `window_size`, `pretrained_window_size`, and learned MLP weights. The coordinate table and index can be precomputed per layer/window shape; the MLP output can also be precomputed at load time for fixed weights/window size if DinoML records it as derived constant provenance.

Concise source-equivalent math:

```python
def swinv2_relative_coords(window_h, window_w, pretrained_h=0, pretrained_w=0):
    coords_h = arange(-(window_h - 1), window_h)
    coords_w = arange(-(window_w - 1), window_w)
    table = meshgrid(coords_h, coords_w)  # [2, 2H-1, 2W-1]
    table = permute(table, [1, 2, 0])[None, ...]
    if pretrained_h > 0:
        table[..., 0] /= pretrained_h - 1
        table[..., 1] /= pretrained_w - 1
    else:
        table[..., 0] /= window_h - 1
        table[..., 1] /= window_w - 1
    table = table * 8
    return sign(table) * log2(abs(table) + 1.0) / log2(8)

def swinv2_attention(q, k, v, coords_table, rel_index, logit_scale, cpb_mlp, mask=None):
    scores = normalize(q, dim=-1) @ transpose(normalize(k, dim=-1), -2, -1)
    scores = scores * exp(clamp(logit_scale, max=log(100.0)))
    bias_table = cpb_mlp(coords_table).reshape(-1, num_heads)
    bias = gather(bias_table, rel_index.flatten()).reshape(M2, M2, num_heads)
    scores = scores + (16 * sigmoid(permute(bias, [2, 0, 1])))[None, ...]
    if mask is not None:
        scores = apply_shifted_window_mask(scores, mask)
    return softmax(scores, dim=-1) @ v
```

Absolute position embeddings are optional and off in the inspected representative checkpoints. If enabled, the source can bicubic-interpolate the patch position table when `interpolate_pos_encoding=True`.

## 8. Preprocessing and input packing

The model consumes `pixel_values` only. There are no tokenizers, text placeholders, masks for classification, or generation controller inputs.

Processor contract:

- Input image is resized to the checkpoint square size (`192` or `256` in snapshots).
- Pixel values are rescaled and normalized by ImageNet mean/std.
- Runtime tensor is NCHW `[B, 3, image_size, image_size]`.
- First DinoML integration can treat preprocessing as CPU/data-pipeline work and start at `pixel_values`.

Backbone output contract:

- `Swinv2Backbone` forces `output_hidden_states=True` and `output_hidden_states_before_downsampling=True`.
- It returns selected `feature_maps` in NCHW layout.
- Default output features come from `BackboneConfigMixin`; representative source tests verify `out_features` and `out_indices`.

Masked image modeling contract:

- Optional `bool_masked_pos` shape `[B, num_patches]`.
- Masked patches are blended by `embeddings * (1-mask) + mask_token * mask`.
- Reconstruction output is `[B, num_channels, image_size, image_size]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d to Linear/GEMM

Source pattern:

```text
Conv2d(Cin -> C, kernel=patch, stride=patch, padding=0) -> flatten -> transpose
```

Replacement:

```text
PatchExtract/WindowFlatten -> Linear(Cin*patch_h*patch_w -> C)
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Pixel input is NCHW and padded exactly as source right/bottom padding.
- Flatten order matches PyTorch Conv2d over NCHW.

Failure cases: dynamic image sizes not covered by static patch-grid guards; nonstandard `num_channels`; channel-last input without a proven layout rewrite.

Parity sketch: compare patch embedding output after Conv2d+flatten+transpose for odd and divisible H/W.

### Rewrite: patch merging to strided gather plus Linear

Source pattern:

```text
view [B,H,W,C] -> pad even -> 4 strided slices -> concat last dim -> Linear(4C -> 2C) -> LayerNorm
```

Replacement:

```text
PatchMergeGather2x2(NHWC) -> GEMM/Linear -> LayerNorm
```

Preconditions:

- Input sequence length equals `H*W` from tracked grid state.
- Gather order is exactly `[top-left, bottom-left, top-right, bottom-right]` as source concatenates `x0,x1,x2,x3`.
- Padding is right/bottom only for odd H/W.

### Rewrite: window partition/attention/reverse to local-attention kernel

Source pattern:

```text
pad -> optional roll -> partition windows -> cosine MHA + CPB + mask -> reverse -> optional roll back -> crop
```

Replacement: fused shifted-window attention kernel over NHWC grid.

Preconditions:

- Square windows from config.
- Shift is either 0 or `window_size//2`.
- Window padding and mask construction match source.
- Relative bias table/index are precomputed for the actual window and pretrained window sizes.

Failure cases: output attentions requested, dynamic H/W without shape-specialized masks, or `use_absolute_embeddings` interpolation requiring separate dynamic table work.

### Layout opportunity: controlled NHWC internal region

Patch Conv2d starts NCHW and backbone returns NCHW, but most SwinV2 block math uses channel-last grid/sequence semantics. A local layout pass can keep stage internals NHWC-like and only convert at ingress/egress.

Guards:

- Preserve LayerNorm over last/channel dimension.
- Rewrite backbone `permute(0,3,1,2)` only at selected feature outputs.
- Keep classifier pooling over sequence length, not spatial channel axis.

## 10. Kernel fusion candidates

Highest priority:

- Patch merging gather + Linear + LayerNorm, because every stage transition uses it and it is layout-sensitive.
- Shifted-window cosine attention with continuous relative bias, because naive roll/partition/matmul/reverse creates many temporaries.
- LayerNorm + residual add around attention/MLP, because SwinV2 uses post-attention/post-MLP LayerNorm placements that should be fused carefully.

Medium priority:

- Patch Conv2d lowered to GEMM for non-overlapping patches.
- Relative bias MLP precompute into per-layer bias constants when window size is fixed.
- FFN Linear + GELU + Linear epilogue fusion.

Lower priority:

- Absolute position interpolation path, because inspected representative configs disable `use_absolute_embeddings`.
- Masked image modeling decoder, unless a SimMIM checkpoint becomes a target.

## 11. Runtime staging plan

Stage 1: parse SwinV2 config, processor metadata, and weights for one classification checkpoint. Reject `use_absolute_embeddings=true` and masked image modeling initially.

Stage 2: implement patch embedding, fixed-shape window partition/reverse, non-shifted window attention, FFN, LayerNorm, and classifier for tiny `window8` at 256.

Stage 3: add shifted-window roll and mask generation, including padded H/W cases and source mask-add parity tests.

Stage 4: add patch merging and multi-stage encoder parity, then classification logits.

Stage 5: add `Swinv2Backbone` selected feature maps in NCHW with `out_features`/`out_indices`.

Stage 6: add base/large and fine-tuned `pretrained_window_sizes` configs.

Stage 7: optimize with fused local window attention and patch-merge kernels.

Deferred: masked image modeling decoder, absolute embedding interpolation, training losses, output attentions as a high-performance path.

## 12. Parity and validation plan

- Unit-test coordinate table and relative index generation for `window=8,12,16` and pretrained windows `[0,0]`, `[12,12]`, `[6,6]`.
- Unit-test continuous position bias MLP gather and `16*sigmoid` transform against PyTorch for one layer.
- Random tensor test for window partition and reverse with and without padding.
- Random tensor test for shifted mask generation and attention mask values.
- Single-block parity for non-shifted and shifted blocks in fp32.
- Stage parity after patch merging for odd and even synthetic H/W grids.
- End-to-end tiny classification parity on a fixed image against HF logits; source tests use first three logits tolerance around `1e-4` for fp32/fp16.
- Backbone parity for selected `stage1` through `stage4` feature map shapes and values.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16 start with `rtol=5e-3, atol=5e-3` for full encoder unless fused softmax/cosine attention is proven tighter.

## 13. Performance probes

- Processor throughput: resize/normalize images per second.
- Patch embedding throughput by batch and resolution.
- Window attention microbench: window size 8/12/16, batch windows count sweep, heads sweep.
- Shifted versus non-shifted block latency to isolate roll/mask/partition overhead.
- Patch merging latency and memory bandwidth by stage.
- Encoder-only throughput for tiny/small/base/large at 192 and 256.
- Classification end-to-end latency and throughput by batch size.
- Backbone feature extraction throughput with all four stages returned.
- Derived relative-bias precompute versus per-run MLP cost.

## 14. Skip/defer list

- Training, stochastic DropPath behavior, dropout randomness.
- Classification/regression loss computation.
- Masked image modeling head and reconstruction loss for first classification/backbone target.
- Absolute position embeddings and bicubic interpolation unless a target checkpoint enables them.
- Output attentions as an optimized runtime output; allow debug fallback later.
- Multi-GPU, gradient checkpointing, feed-forward chunking.
- Remote-code behavior: not required for inspected official checkpoints.

## 15. Final implementation checklist

- [ ] Parse `Swinv2Config` including `depths`, `num_heads`, `window_size`, `pretrained_window_sizes`, `mlp_ratio`, `qkv_bias`, and `out_features`.
- [ ] Load `ViTImageProcessor`/legacy `ViTFeatureExtractor` preprocessing metadata.
- [ ] Load native separate Q/K/V weights and enforce key-bias policy.
- [ ] Implement patch Conv2d path and guarded Conv2d-to-Linear rewrite.
- [ ] Implement window partition and reverse with padding/crop.
- [ ] Implement shifted-window roll and mask generation.
- [ ] Implement cosine local MHA with clamped `logit_scale`.
- [ ] Implement continuous relative position bias table/index/MLP.
- [ ] Implement patch merging gather order and Linear `4C -> 2C`.
- [ ] Implement SwinV2 block post-attention/post-MLP LayerNorm residual ordering.
- [ ] Implement classification pooler and classifier head.
- [ ] Implement backbone NCHW feature map outputs.
- [ ] Add tiny `window8` end-to-end parity.
- [ ] Add base/large `window12`/`window16` and pretrained-window bias parity.
- [ ] Benchmark fused shifted-window attention and patch merging.
