# Transformers Swin Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary target: microsoft/swin-tiny-patch4-window7-224.
  Additional sizing/task references:
    hf-internal-testing/tiny-random-SwinModel
    microsoft/swin-base-patch4-window7-224
    microsoft/swin-base-patch4-window12-384
    microsoft/swin-large-patch4-window12-384
    microsoft/swin-base-simmim-window6-192

Config source:
  https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/raw/main/config.json
  https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/raw/main/preprocessor_config.json
  Additional configs and preprocessors fetched from the Hugging Face repos listed above.

Source files inspected:
  X:/H/transformers/src/transformers/models/swin/configuration_swin.py
  X:/H/transformers/src/transformers/models/swin/modeling_swin.py
  X:/H/transformers/src/transformers/models/swin/modular_swin.py
  X:/H/transformers/src/transformers/models/auto/image_processing_auto.py
  X:/H/transformers/src/transformers/models/vit/image_processing_vit.py
  X:/H/transformers/src/transformers/models/vit/image_processing_pil_vit.py

Any missing files or assumptions:
  modeling_swin.py is generated from modular_swin.py; future source edits should inspect
  modular_swin.py first, while generated modeling_swin.py is the installed runtime file.
  There is no Swin-specific image processor file. AutoImageProcessor maps `swin` to
  ViTImageProcessor/ViTImageProcessorPil. No remote code is required for the standard Swin
  checkpoints inspected. This report targets image classification and the base vision encoder;
  masked image modeling and backbone feature-map extraction are documented but optional.
```

## 2. High-level architecture

Swin is a hierarchical vision encoder with shifted-window self-attention. Source preprocessing produces NCHW `pixel_values`, patch embedding uses non-overlapping Conv2d, and the encoder alternates local window MSA and shifted-window MSA inside spatial stages. Stages except the last end with patch merging that halves spatial size and doubles channels. Classification uses final LayerNorm, adaptive average pooling over sequence tokens, and a linear classifier.

```text
image preprocessing -> NCHW patch Conv2d -> patch tokens -> Swin stages/window attention -> final norm/pool -> classifier logits
```

Stage decomposition:

- CPU/data pipeline: image resize, optional rescale, normalize, channel arrangement into `pixel_values`.
- Patch stage: source NCHW Conv2d with `kernel_size == stride == patch_size`; strong candidate for NHWC window-flatten + GEMM under layout guards.
- Encoder stage: repeated stage-local blocks over `[B, H*W, C]`, with temporary NHWC views for pad, roll, window partition/reverse, and patch merging.
- Head stage: final token LayerNorm, `AdaptiveAvgPool1d(1)`, flatten, classifier GEMM.
- Optional independent stages: `SwinBackbone` feature maps and `SwinForMaskedImageModeling` decoder can be validated separately from classification.

## 3. Important config dimensions

Worked example: `microsoft/swin-tiny-patch4-window7-224`.

| Field | Value | Source |
|---|---:|---|
| primary runtime target | image classification | HF repo metadata/config architecture |
| image_size | 224 | config/preprocessor |
| num_channels | 3 | config |
| patch_size | 4 | config |
| patch grid | 56 x 56 | inferred `image_size / patch_size` |
| embed_dim | 96 | config |
| hidden_size final | 768 | inferred `embed_dim * 2^3`; omitted in older config |
| depths | 2 / 2 / 6 / 2 | config |
| total blocks | 12 | inferred `sum(depths)` |
| stage channels | 96 / 192 / 384 / 768 | inferred |
| num_heads | 3 / 6 / 12 / 24 | config |
| head_dim | 32 each stage | inferred `stage_dim / heads` |
| window_size | 7 | config |
| attention tokens per window | 49 | inferred `window_size^2` |
| mlp_ratio | 4.0 | config |
| stage MLP widths | 384 / 768 / 1536 / 3072 | inferred |
| qkv_bias | true | config |
| hidden_act | gelu | config |
| layer_norm_eps | 1e-5 | config |
| use_absolute_embeddings | false | config |
| classifier labels | 1000 | config `id2label` |
| processor | ViTImageProcessor/legacy ViTFeatureExtractor fields | preprocessor/auto mapping |
| processor mean/std | ImageNet `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | preprocessor |
| dtype | not specified in inspected configs | config; runtime dtype should come from weights/deployment policy |

Representative checkpoint sweep:

| Checkpoint | Task/class | Image | Patch | Grid | Embed | Depths | Heads | Window | Window tokens | Final C | Notes |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| hf-internal-testing/tiny-random-SwinModel | `SwinModel` | 32 | 2 | 16x16 | 16 | 1/2/1 | 2/2/4 | 2 | 4 | 64 | Tiny/debug, 3 stages, `mlp_ratio=2`, processor mean/std 0.5 |
| microsoft/swin-tiny-patch4-window7-224 | classification | 224 | 4 | 56x56 | 96 | 2/2/6/2 | 3/6/12/24 | 7 | 49 | 768 | Common 224 classifier |
| microsoft/swin-base-patch4-window7-224 | classification | 224 | 4 | 56x56 | 128 | 2/2/18/2 | 4/8/16/32 | 7 | 49 | 1024 | Deeper stage 3 |
| microsoft/swin-base-patch4-window12-384 | classification | 384 | 4 | 96x96 | 128 | 2/2/18/2 | 4/8/16/32 | 12 | 144 | 1024 | Larger resolution/window |
| microsoft/swin-large-patch4-window12-384 | classification | 384 | 4 | 96x96 | 192 | 2/2/18/2 | 6/12/24/48 | 12 | 144 | 1536 | Wider large model |
| microsoft/swin-base-simmim-window6-192 | masked image modeling | 192 | 4 | 48x48 | 128 | 2/2/18/2 | 4/8/16/32 | 6 | 36 | 1024 | SimMIM decoder, processor mean/std 0.5 |

Older production classification configs omit `hidden_size`, `encoder_stride`, and explicit `do_rescale`; effective source defaults are `hidden_size=embed_dim*2^(len(depths)-1)`, `encoder_stride=32`, and ViTImageProcessor default `do_rescale=True` when the newer processor class is constructed.

## 3a. Family variation traps

- Source model input is NCHW `[B,C,H,W]`; preferred Dinoml NHWC/channel-last is an optimization pass, not the semantic graph.
- Inside Swin blocks, source repeatedly reshapes tokens to NHWC `[B,H,W,C]` for padding, `torch.roll`, window partition/reverse, and patch merging. These are the safest regions for guarded channel-last lowering.
- `window_size` is config-dependent: 7 for common 224 checkpoints, 12 for 384 checkpoints, 6 for SimMIM 192, 2 for tiny random. Relative position bias table shape depends on it.
- `set_shift_and_window_size` mutates per-layer `window_size`/`shift_size` when `min(input_resolution) <= window_size`. For 224/window7, the last 7x7 stage disables shift and uses one full window.
- Alternating blocks use W-MSA then SW-MSA: even block index shift 0, odd block index `window_size // 2` unless clamped.
- Patch embedding pads images whose H/W are not divisible by `patch_size`; block attention pads feature maps to multiples of `window_size`; patch merging pads odd H/W. Dynamic image support must preserve these right/bottom pad semantics.
- Patch merging concatenation order is source-specific: `[row::2, col::2]` for `col in 0..1` then `row in 0..1`, i.e. top-left, bottom-left, top-right, bottom-right along channels.
- `use_absolute_embeddings` defaults false in inspected checkpoints, but source supports learned absolute patch embeddings plus optional bicubic interpolation. A first classifier path may ignore it only with a config guard.
- `SwinBackbone` returns NCHW feature maps and applies extra per-stage LayerNorm by temporarily converting feature maps to `[B,H*W,C]`.
- SimMIM uses mask tokens plus Conv2d(1x1) and PixelShuffle decoder; do not mix it into the first classification target.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW image input `[B,3,H,W]` from processor.
- Right/bottom `pad` for patch-size divisibility on NCHW input.
- Conv2d patch embed, then `flatten(2).transpose(1,2)` to `[B,Hp*Wp,C]`.
- Token reshape/view `[B,L,C] <-> [B,H,W,C]`.
- NHWC right/bottom pad to window multiples.
- `torch.roll` on spatial axes `(1,2)` for cyclic shift/reverse shift.
- Window partition: NHWC view `[B,H/ws,ws,W/ws,ws,C]`, transpose middle window axes, contiguous, flatten windows.
- Window reverse: inverse view/transpose/contiguous to NHWC.
- Crop padded rows/cols after attention.
- Patch merging: NHWC strided gathers, channel concat, flatten spatial.
- Final `transpose(1,2)` for `AdaptiveAvgPool1d(1)`, flatten to `[B,C]`.
- Optional output hidden-state conversions to NCHW for backbone/diagnostics.

### Neural network primitives

- Patch Conv2d:
  - tiny: `Conv2d(3 -> 96, kernel=4, stride=4, bias=True)`.
  - base: `Conv2d(3 -> 128, kernel=4, stride=4, bias=True)`.
  - large: `Conv2d(3 -> 192, kernel=4, stride=4, bias=True)`.
- LayerNorm over channel dimension for patch embeddings, block pre/post norms, patch merging norm over `4*C`, final norm, and backbone feature norms.
- Linear projections with bias:
  - stage 1 tiny attention: Q/K/V/O `96 -> 96`; MLP `96 -> 384 -> 96`.
  - stage 2 tiny: Q/K/V/O `192 -> 192`; MLP `192 -> 768 -> 192`.
  - stage 3 tiny: Q/K/V/O `384 -> 384`; MLP `384 -> 1536 -> 384`.
  - stage 4 tiny: Q/K/V/O `768 -> 768`; MLP `768 -> 3072 -> 768`.
- Patch merging reduction `Linear(4*C -> 2*C, bias=False)` after `LayerNorm(4*C)`.
- GELU activation.
- Dropout and stochastic depth are no-ops for normal inference, but graph import should tolerate them.
- Classification head `Linear(final_C -> num_labels)` or Identity when `num_labels <= 0`.
- Optional SimMIM decoder: `Conv2d(final_C -> encoder_stride^2 * num_channels, kernel=1)` and PixelShuffle.

### Attention primitives

- Noncausal local self-attention over fixed windows.
- MHA only; no MQA/GQA, no KV cache.
- Per-stage head counts as config lists.
- Additive attention mask is `relative_position_bias + optional_shift_mask`.
- Eager math order: QK matmul, multiply by `head_dim^-0.5`, add mask, softmax with fp32 accumulation, cast to query dtype, dropout, AV matmul.
- Source declares SDPA support and no FlashAttention/FlexAttention support.

### Position/relative-bias ops

- Learned relative position bias table per layer: `[(2*ws-1)*(2*ws-1), num_heads]`.
- Non-persistent relative position index derived from `window_size`; can be precomputed per `(window_size, heads)`.
- Optional learned absolute patch position embeddings `[1,num_patches,embed_dim]` and bicubic interpolation.

### Preprocessing-coupled ops

- ViTImageProcessor resize, rescale, normalize, and channel-first output contract.
- Legacy preprocessors may use `feature_extractor_type=ViTFeatureExtractor`, scalar `size=224/384`, and `resample=3`; newer SimMIM/tiny configs include `image_processor_type=ViTImageProcessor`, dict size, `do_rescale`, and `rescale_factor`.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B,3,H,W] NCHW
pixel_values = right/bottom pad to multiples of patch_size
x = Conv2d(3 -> embed_dim, kernel=P, stride=P)(pixel_values)  # [B,C0,Hp,Wp]
input_dimensions = (Hp,Wp)
x = flatten spatial + transpose -> [B,Hp*Wp,C0]
x = LayerNorm(C0)(x)
optional mask token replacement for SimMIM
optional absolute position add
```

Swin block, repeated by stage depth:

```text
shortcut = x                                      # [B,H*W,C]
y = LayerNorm(C)(x)
y = view [B,H,W,C]
y = pad bottom/right to Hpad,Wpad multiples of ws
y = roll(y, shifts=(-shift,-shift), dims=(H,W))  # only shifted blocks
windows = window_partition(y, ws)                # [B*nW,ws*ws,C]
attn_mask = shifted_window_mask(Hpad,Wpad,ws,shift) or None
y = window_attention(windows, relative_position_bias, attn_mask)
y = window_reverse(y, ws, Hpad, Wpad)
y = roll(y, shifts=(shift,shift), dims=(H,W))
y = crop to original H,W
y = view [B,H*W,C]
x = shortcut + y                                 # drop_path identity in inference

residual = x
y = LayerNorm(C)(x)
y = Linear(C -> mlp_ratio*C)(y)
y = GELU(y)
y = Linear(mlp_ratio*C -> C)(y)
x = residual + y
```

Patch merging between stages:

```text
y = view [B,H,W,C]
y = pad bottom/right if H or W is odd
y = cat([y[:,0::2,0::2,:], y[:,1::2,0::2,:],
         y[:,0::2,1::2,:], y[:,1::2,1::2,:]], dim=-1)
y = view [B,ceil(H/2)*ceil(W/2),4*C]
y = LayerNorm(4*C)(y)
y = Linear(4*C -> 2*C, bias=False)(y)
```

Classification head:

```text
x = final LayerNorm(final_C)(x)                  # [B,L,final_C]
pooled = AdaptiveAvgPool1d(1)(x.transpose(1,2))  # [B,final_C,1]
pooled = flatten(pooled, 1)                      # [B,final_C]
logits = Linear(final_C -> num_labels)(pooled)
```

## 6. Attention requirements

- Type: noncausal local self-attention.
- Scope: each attention call sees one spatial window as a sequence of `ws*ws` tokens; batch dimension is folded with number of windows.
- Heads/head_dim:
  - tiny 224: `(3,6,12,24)` heads with `D=32`.
  - base 224/384: `(4,8,16,32)` heads with `D=32`.
  - large 384: `(6,12,24,48)` heads with `D=32`.
- Masking style:
  - W-MSA blocks use only relative position bias.
  - SW-MSA blocks add a generated mask shaped `[num_windows, ws*ws, ws*ws]`, expanded to `[B*num_windows,1,ws*ws,ws*ws]`, with `0.0` for same shifted region and `-100.0` otherwise.
- Relative bias:
  - Added as `[1,num_heads,ws*ws,ws*ws]` before backend attention.
  - Bias is learned per layer and read every forward.
- Backend compatibility:
  - Source uses `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`.
  - `_supports_sdpa=True`, `_supports_flash_attn=False`, `_supports_flex_attn=False`.
  - Fused attention must accept additive per-head bias plus optional shifted-window additive mask. A generic causal/prefix attention kernel is not enough.
- No packed/varlen sequence metadata, no sliding window in the language-model sense, no KV cache, no RoPE/ALiBi.

## 7. Position encoding and custom math

Relative position index can be precomputed from `window_size`:

```python
coords = meshgrid(arange(ws), arange(ws))          # 2, ws, ws
flat = flatten(coords, start_dim=1)                # 2, ws*ws
rel = flat[:, :, None] - flat[:, None, :]          # 2, N, N
rel = rel.permute(1, 2, 0)
rel[..., 0] += ws - 1
rel[..., 1] += ws - 1
rel[..., 0] *= 2 * ws - 1
index = rel.sum(-1)                                # N, N
bias = table[index.view(-1)].view(N, N, heads).permute(2, 0, 1)[None]
```

Shifted-window attention mask:

```python
img_mask = zeros([1, Hpad, Wpad, 1])
for h_slice in (slice(0,-ws), slice(-ws,-shift), slice(-shift,None)):
    for w_slice in (slice(0,-ws), slice(-ws,-shift), slice(-shift,None)):
        img_mask[:, h_slice, w_slice, :] = count
mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
attn_mask = where(attn_mask != 0, -100.0, 0.0)
```

Optional absolute position interpolation:

```python
pos = position_embeddings.reshape(1, sqrt_n, sqrt_n, C).permute(0, 3, 1, 2)
pos = bicubic_interpolate(pos, size=(H // patch_size, W // patch_size), align_corners=False)
pos = pos.permute(0, 2, 3, 1).view(1, -1, C)
```

Precompute/caching opportunities:

- Relative position index is static for each `window_size`.
- Shifted-window mask is static for a fixed padded feature-map size, window size, shift, dtype, and device.
- Absolute position interpolation can be cached per input resolution bucket when enabled.

## 8. Preprocessing and input packing

CPU/data-pipeline contract:

- AutoImageProcessor for `swin` maps to ViTImageProcessor/ViTImageProcessorPil.
- Common classification preprocessors resize to 224 or 384, normalize with ImageNet mean/std, and historically store `feature_extractor_type=ViTFeatureExtractor`.
- Tiny random and SimMIM preprocessors use mean/std `[0.5,0.5,0.5]`, `do_rescale=true`, and explicit `rescale_factor=1/255`.
- Processor output consumed by the model is `pixel_values` with source shape `[B,3,H,W]`.

GPU/runtime contract:

- Initial faithful import should accept source NCHW semantics.
- A Dinoml frontend may choose NHWC as the public image layout only if it owns the processor handoff and rewrites the patch embedding axes consistently.
- No text tokens, placeholder IDs, modality type IDs, grid metadata, packed sequence descriptors, or `cu_seqlens`.
- Optional `bool_masked_pos` for SimMIM has shape `[B,num_patches]` and controls patch-token replacement before encoder.

## 9. Graph rewrite / lowering opportunities

### Rewrite: source patch Conv2d -> NHWC patch GEMM

Preconditions:

- `Conv2d(in=C,out=E,kernel=(P,P),stride=(P,P),padding=0,dilation=1,groups=1)`.
- Source pattern is `maybe_pad -> Conv2d -> flatten(2) -> transpose(1,2)`.
- Pad semantics are right and bottom only.
- Input layout is known. For NHWC lowering, activation patch flatten order must be `[kh,kw,c]`.

Replacement:

```text
PadNHWCToMultiple(P) -> NonOverlapWindowFlattenNHWC(P,P) -> GEMM_RCR_Bias -> Reshape[B,Hp*Wp,E]
```

Weight transform:

```python
# source conv weight [E, C, P, P], NHWC patch flatten [kh, kw, c]
w = conv.weight.permute(0, 2, 3, 1).reshape(E, P * P * C)
```

Failure cases:

- Unknown/dynamic input layout.
- Nonzero Conv2d padding/dilation/groups.
- A caller requires bit-identical PyTorch Conv2d accumulation order beyond normal tolerance.

Parity test sketch:

- Compare source NCHW Conv2d path against NHWC flatten+GEMM for divisible and non-divisible H/W cases.

### Rewrite: window partition/reverse as layout metadata or fused attention tiling

Preconditions:

- Tensor is logically `[B,H,W,C]` contiguous or channel-last.
- `Hpad` and `Wpad` are multiples of `window_size`.
- Consumer is immediately local attention and producer after attention is immediate window reverse.

Replacement:

```text
NHWC Pad/Roll -> WindowedAttention(tile=ws, shift=s, relative_bias, mask) -> NHWC Crop
```

Axis rewrites:

- Source roll dims `(1,2)` remain spatial dims in NHWC.
- Source attention tensors become `[B*nW, heads, ws*ws, D]`; layout pass must not reinterpret `dim=-1` LayerNorm or softmax `dim=-1`.

Failure cases:

- `always_partition`/backbone mode with unusual resolutions must still match source clamping behavior.
- Any external consumer observes intermediate window tensors.

Parity test sketch:

- Compare `window_partition -> attention -> window_reverse` against fused windowed attention on W-MSA and SW-MSA blocks.

### Rewrite: shifted-window attention mask precompute

Preconditions:

- Static or bucketed `(Hpad,Wpad,window_size,shift_size)`.
- Dtype/device are known or mask can be materialized in target dtype at load/session init.

Replacement:

```text
Runtime mask-building loops -> CachedAdditiveMask[stage,block,resolution]
```

Failure cases:

- Dynamic arbitrary H/W without bucketization.
- Source clamping changes `window_size`/`shift_size` because resolution is smaller than window.

### Rewrite: patch merging -> NHWC gather + GEMM

Preconditions:

- Input is logical `[B,H,W,C]`.
- Source padding for odd H/W is preserved.
- Concatenation order is top-left, bottom-left, top-right, bottom-right.

Replacement:

```text
PadOddNHWC -> 2x2StridedGatherConcat -> LayerNorm(4C) -> GEMM_RCR(no bias, 4C -> 2C)
```

Weight transform:

- None for source concat order. If a backend chooses a different 2x2 flatten order, permute `reduction.weight` columns in matching 4 channel blocks.

Failure cases:

- Assuming the more common `[top-left, top-right, bottom-left, bottom-right]` order; that is not the source order.

### Rewrite: separate Q/K/V linears -> packed QKV

Preconditions:

- Same normalized window tokens feed q/k/v.
- q/k/v output widths are all `C` and share same bias setting.

Replacement:

```text
PackedLinear(C -> 3C, bias=qkv_bias) -> Split -> reshape heads
```

Weight transform:

```python
w_qkv = concat([q_proj.weight, k_proj.weight, v_proj.weight], axis=0)
b_qkv = concat([q_proj.bias, k_proj.bias, v_proj.bias], axis=0)
```

Failure cases:

- Any config or checkpoint with missing projection bias on only some projections; inspected standard configs have `qkv_bias=true`.

### Rewrite: final pool/classifier fusion

Preconditions:

- Image classification only.
- `pooler_output` is not separately requested.

Replacement:

```text
FinalLayerNorm -> MeanOverTokens -> ClassifierGEMM
```

Failure cases:

- `SwinModel` API returning pooler output or backbone/feature extraction paths.

## 10. Kernel fusion candidates

Highest priority:

- NHWC patch embedding GEMM: removes NCHW Conv2d plus flatten/transpose while matching the non-overlap patch contract.
- Window partition/roll/mask/attention fused region: Swin spends work reshaping and masking small windows; a fused windowed attention primitive avoids materializing window tensors and masks repeatedly.
- Relative position bias gather/precompute plus additive-mask attention: required for parity and source-specific enough to justify a dedicated path.
- Patch merging NHWC gather + LayerNorm + GEMM: common between every stage and sensitive to channel-last memory access.

Medium priority:

- Packed QKV projection for each window-attention block.
- LayerNorm + adjacent residual/add scheduling over `[B,L,C]`.
- MLP `Linear -> GELU -> Linear` fusion or at least activation fusion.
- Cached shifted-window masks per resolution bucket.

Lower priority:

- Backbone NCHW feature-map output normalization path.
- SimMIM decoder Conv2d + PixelShuffle.
- Absolute position interpolation cache, because inspected production Swin configs set `use_absolute_embeddings=false`.

## 11. Runtime staging plan

Stage 1: Parse SwinConfig and ViTImageProcessor/legacy preprocessor config; load classification weights.

Stage 2: Implement faithful source-layout patch embedding, one Swin block, relative position bias, and shifted-window mask parity on tiny random config.

Stage 3: Full encoder parity for `SwinModel` on tiny random and `swin-tiny-patch4-window7-224`.

Stage 4: Add final norm, adaptive mean pool, classifier logits parity for 224 classifiers.

Stage 5: Add 384/window12 config coverage and bucketed mask/precompute support.

Stage 6: Introduce guarded NHWC rewrites for patch embedding, window partition/reverse, and patch merging.

Stage 7: Optimize packed QKV, fused windowed attention with relative bias/mask, and MLP kernels.

Stage 8: Optional follow-ons: `SwinBackbone` feature maps and SimMIM masked image modeling decoder.

Initially stubbable: training losses, stochastic depth/dropout behavior, output attentions, hidden-state capture, SimMIM mask token path, and backbone feature-map API for a classifier-only target.

## 12. Parity and validation plan

- Processor handoff parity: confirm `pixel_values` shape/range for legacy `size=224/384` and dict-size preprocessors.
- Patch embedding parity for divisible and non-divisible input H/W.
- Relative position index/bias parity for `window_size` 2, 6, 7, and 12.
- Shifted-window mask parity for stage resolutions including 56/28/14/7 and 96/48/24/12.
- Window partition/reverse round-trip tests for NHWC tensors.
- Single W-MSA block and single SW-MSA block parity.
- Patch merging parity for even and odd H/W, explicitly checking source concat order.
- After-1-stage and full-encoder parity for tiny random, tiny 224, base 224, and base/large 384.
- Classification logits parity for `microsoft/swin-tiny-patch4-window7-224`.
- Optional SimMIM reconstruction-shape and mask-token parity.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-5`; fp16/bf16 optimized paths start at `rtol=2e-2, atol=2e-2`, then tighten after kernel accumulation choices are fixed.

## 13. Performance probes

- CPU preprocessing throughput separately from runtime.
- Patch embedding source Conv2d vs NHWC patch GEMM over batch and image size.
- Window partition/reverse materialization cost by stage.
- Shifted-window mask build vs cached mask lookup.
- Windowed attention throughput for token counts 36, 49, and 144 with head_dim 32.
- Patch merging bandwidth/GEMM cost at each stage.
- MLP GEMM throughput by stage, especially base/large stage 3 with depth 18.
- End-to-end images/sec for 224 and 384 classifiers across batch sizes.
- Memory bandwidth and allocation probes for hidden-state/backbone outputs disabled vs enabled.

## 14. Skip/defer list

- Training, gradients, stochastic depth randomness, and dropout.
- Loss functions for classification/regression and SimMIM L1 loss.
- Output attentions and hidden-state capture unless needed for tests.
- `SwinBackbone` feature maps for detection/segmentation consumers.
- SimMIM decoder and PixelShuffle.
- Absolute position embeddings/interpolation for first classifier pass unless a config requires it.
- Arbitrary dynamic image sizes without resolution buckets.
- Multi-GPU/tensor parallel sharding.
- Quantization and sparse/structured pruning.

## 15. Final implementation checklist

- [ ] Parse SwinConfig, including list-valued `depths` and `num_heads`.
- [ ] Parse ViTImageProcessor/legacy preprocessor contract for Swin.
- [ ] Load patch embedding, stage, norm, relative-bias, patch-merging, and classifier weights.
- [ ] Implement source-faithful NCHW patch embedding with right/bottom padding.
- [ ] Implement relative position index and bias gather.
- [ ] Implement window partition/reverse and shifted-window mask.
- [ ] Implement W-MSA and SW-MSA block parity.
- [ ] Implement patch merging with source concat order.
- [ ] Implement full encoder and classifier head parity.
- [ ] Add 224/window7 and 384/window12 checkpoint parity.
- [ ] Add guarded NHWC patch Conv2d -> GEMM rewrite.
- [ ] Add guarded NHWC window attention and patch-merging rewrites.
- [ ] Add packed QKV rewrite.
- [ ] Benchmark patch embed, window attention, patch merging, MLP, and end-to-end images/sec.
