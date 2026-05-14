# DinoML Transformers Audit: DINOv3 ViT

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  Primary attempted ids:
  - facebook/dinov3-vits16-pretrain-lvd1689m
  - facebook/dinov3-vits16plus-pretrain-lvd1689m
  - facebook/dinov3-vitb16-pretrain-lvd1689m
  - facebook/dinov3-vitl16-pretrain-lvd1689m
  - facebook/dinov3-vitl16-pretrain-sat493m
  - facebook/dinov3-vith16plus-pretrain-lvd1689m
  - facebook/dinov3-vit7b16-pretrain-lvd1689m
  - facebook/dinov3-vit7b16-pretrain-sat493m
Config source:
  Official raw config/preprocessor files returned 401 in this environment.
  Concrete variant dimensions below come from the pinned in-tree conversion script.
Source files inspected:
  - X:/H/transformers/src/transformers/models/dinov3_vit/configuration_dinov3_vit.py
  - X:/H/transformers/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py
  - X:/H/transformers/src/transformers/models/dinov3_vit/modular_dinov3_vit.py
  - X:/H/transformers/src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py
  - X:/H/transformers/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py
Any missing files or assumptions:
  modeling_dinov3_vit.py is generated from modular_dinov3_vit.py; future source edits should target the modular file.
  Official raw checkpoint config/preprocessor files were inaccessible via 401, so exact saved config JSON defaults were not confirmed.
  No remote code was required for the in-library implementation.
```

Snapshots are under `agents/plans/transformers/dinov3_vit/_sources/`. Gated access notes are under `_sources/hf_configs/gated_raw_access_notes.md`.

Gated/401 links that would resolve exact saved `config.json` and `preprocessor_config.json`:
[vits16-lvd](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
[vits16plus-lvd](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m),
[vitb16-lvd](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
[vitl16-lvd](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
[vitl16-sat](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m),
[vith16plus-lvd](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m),
[vit7b16-lvd](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m),
[vit7b16-sat](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m).

## 2. High-level architecture

DINOv3 ViT is a vision-only encoder/backbone. The primary runtime target is image feature extraction/backbone output, with optional pooled CLS output from `DINOv3ViTModel`. There is no autoregressive decode, text tokenizer, KV cache, or classification head in this family source.

```text
image preprocessing -> NCHW patch Conv2d -> [CLS, register, patch] token sequence
  -> dynamic 2D RoPE for patch tokens -> noncausal ViT encoder blocks
  -> final LayerNorm -> CLS pooled output and/or backbone feature maps
```

Stage decomposition:

- CPU/data pipeline: image decode, rescale, resize, optional center crop, normalize.
- GPU/runtime: patch embedding, token prefix concat, dynamic RoPE coordinate math, encoder blocks, final norm.
- Backbone projection: selected hidden states have prefix tokens stripped, then patch tokens optionally reshape to NCHW feature maps.

## 3. Important config dimensions

Source defaults from `DINOv3ViTConfig`:

| Field | Default / source behavior |
|---|---:|
| `image_size` | `224` |
| `num_channels` | `3` |
| `patch_size` | `16` |
| `hidden_size` | `384` |
| `num_hidden_layers` | `12` |
| `num_attention_heads` | `6` |
| `head_dim` | `hidden_size // num_attention_heads` |
| `intermediate_size` | `1536` |
| `hidden_act` | `gelu` |
| `layer_norm_eps` | `1e-5` |
| `rope_theta` | `100.0` |
| `query_bias` / `key_bias` / `value_bias` / `proj_bias` | `True` / `False` / `True` / `True` |
| `mlp_bias` | `True` |
| `layerscale_value` | `1.0` |
| `drop_path_rate` | `0.0` |
| `use_gated_mlp` | `False` |
| `num_register_tokens` | `0` default; official conversion variants use `4` |
| `cache support` | none; encoder-only noncausal attention |

Representative checkpoint sweep from `convert_dinov3_vit_to_hf.py`:

| Variant | Hidden | Layers | Heads | Head dim | MLP dim | Gated MLP | Act | Registers | Bias trap |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `vits16_lvd1689m` | 384 | 12 | 6 | 64 | 1536 | no | GELU | 4 | q/v/o bias, no k bias |
| `vits16plus_lvd1689m` | 384 | 12 | 6 | 64 | 1536 | yes | SiLU | 4 | q/v/o bias, no k bias |
| `vitb16_lvd1689m` | 768 | 12 | 12 | 64 | 3072 | no | GELU | 4 | q/v/o bias, no k bias |
| `vitl16_lvd1689m` / `vitl16_sat493m` | 1024 | 24 | 16 | 64 | 4096 | no | GELU | 4 | q/v/o bias, no k bias |
| `vith16plus_lvd1689m` | 1280 | 32 | 20 | 64 | 5120 | yes | SiLU | 4 | q/v/o bias, no k bias |
| `vit7b16_lvd1689m` / `vit7b16_sat493m` | 4096 | 40 | 32 | 128 | 8192 | yes | SiLU | 4 | output projection bias only; conversion sets q/k/v bias false |

For an input `B x 3 x H x W`, patch count is `P = floor(H/16) * floor(W/16)` for the official variants. Sequence length is `T = 1 + num_register_tokens + P`. For the default processor resize `224 x 224`, official variants have `P = 196` and `T = 201`.

## 3a. Family variation traps

- DINOv3 ViT uses dynamic 2D RoPE over patch-center coordinates. It does not use DINOv2-style learned absolute position tables or bicubic position interpolation.
- CLS and register tokens are prefix tokens. RoPE is applied only to patch tokens; prefix count is computed as `num_tokens - num_patches`.
- Official conversion variants use four register tokens even though the config class default is zero.
- `vit7b16` has `head_dim = 128` and `hidden_size == heads * head_dim`; smaller variants have `head_dim = 64`.
- MLP can be ordinary GELU FFN or gated SiLU FFN. The gated path uses `act(gate_proj(x)) * up_proj(x)` before `down_proj`.
- Biases are projection-specific. Do not assume one QKV bias flag; q, k, v, and output projection biases are independent.
- Patch embedding and backbone feature maps are NCHW at the public boundary. NHWC/channel-last should be a guarded optimization region, not a semantic rewrite.
- The backbone can return either patch token sequences or NCHW feature maps depending on `reshape_hidden_states`.
- `return_class_token` is read dynamically with `getattr`, but it is not declared in the config class fields in this source snapshot.
- Training-only stochastic pieces include DropPath, RoPE coordinate shift/jitter/rescale, and mask-token replacement for pretraining. Inference parity can initially reject or stub these.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `B x 3 x H x W`.
- Conv2d patch embedding with `kernel_size = stride = patch_size`, no explicit padding.
- Flatten spatial dims and transpose: `B x C x Hp x Wp -> B x (Hp*Wp) x C`.
- Prefix expansion and concat along token axis: `[CLS, registers, patches]`.
- Split/concat over token axis for RoPE prefix exclusion.
- Backbone token slice `hidden[:, 1 + R:, :]`, reshape to `B x Hp x Wp x C`, permute to NCHW, contiguous.
- Optional `torch.where(bool_masked_pos[..., None], mask_token, patch_embeddings)` for pretraining/masked-input parity.

Neural network primitives:

- `LayerNorm(hidden_size, eps=1e-5)` before attention, before MLP, and final output norm.
- Independent `Linear(C -> C)` q/k/v/o projections. Bias flags differ per projection.
- Ordinary MLP: `Linear(C -> I) -> GELU -> Linear(I -> C)`.
- Gated MLP: `Linear(C -> I)` gate, `Linear(C -> I)` up, activation, elementwise multiply, `Linear(I -> C)`.
- LayerScale: per-channel multiply by learned vector `[C]`.
- Residual add; DropPath is identity in inference.

Attention primitives:

- Noncausal encoder self-attention.
- MHA only; no MQA/GQA.
- SDPA/Flash/Flex backend dispatch through `ALL_ATTENTION_FUNCTIONS`, with eager fallback.
- Scaling by `head_dim ** -0.5`; dropout disabled in inference.

Position/custom math:

- Dynamic 2D patch-center coordinate generation in float32.
- `inv_freq = 1 / rope_theta ** arange(0, 1, 4 / head_dim)`.
- Cos/sin for `2*pi*coord*inv_freq`, flattened from `(y,x)` and tiled to full head dim.
- RoPE `rotate_half` on patch-token q/k only.

Preprocessing-coupled ops:

- Rescale before resize, then normalize after optional crop.
- Bilinear resize with antialiasing.
- ImageNet mean/std normalization.
- Shape grouping/reordering in processor is CPU/data-pipeline behavior.

Distributed/tensor-parallel ops:

- None in this source. Tensor parallel loading can be deferred.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: B x 3 x H x W, NCHW
patch = Conv2d(3 -> C, kernel=16, stride=16)(pixel_values)
patch = flatten_spatial(patch).transpose(1, 2)        # B x P x C
tokens = cat([cls: B x 1 x C, registers: B x R x C, patch], dim=1)
```

Encoder block, repeated `L` times:

```text
residual = x                                           # B x T x C
y = LayerNorm(C)(x)
q = Linear(C -> C, bias=query_bias)(y)
k = Linear(C -> C, bias=key_bias)(y)
v = Linear(C -> C, bias=value_bias)(y)
q,k,v = reshape to B x heads x T x head_dim
q_patch,k_patch = DINOv3 2D RoPE(q_patch,k_patch)
y = noncausal self-attention(q,k,v, scale=head_dim**-0.5)
y = Linear(C -> C, bias=proj_bias)(y)
x = residual + LayerScale(C)(y)
residual = x
y = LayerNorm(C)(x)
y = MLP or GatedMLP(y)
x = residual + LayerScale(C)(y)
```

Final model output:

```text
sequence_output = LayerNorm(C)(last_hidden_state)      # B x T x C
pooler_output = sequence_output[:, 0, :]               # CLS, B x C
```

Backbone output:

```text
for selected stage hidden_state:
  optionally LayerNorm(C)
  cls_token = hidden_state[:, 0, :] if return_class_token
  patch_tokens = hidden_state[:, 1 + num_register_tokens:, :]
  if reshape_hidden_states:
    fmap = patch_tokens.reshape(B, H/patch, W/patch, C).permute(0,3,1,2).contiguous()
  else:
    fmap = patch_tokens
```

## 6. Attention requirements

- Type: encoder self-attention, noncausal.
- Head layout: MHA, `q/k/v` all shaped `B x num_heads x T x head_dim`.
- Query/key/value widths: all `C`; `head_dim = C / num_heads`.
- Sequence length: `T = 1 + R + Hp*Wp`.
- Masking: model forward accepts kwargs routed to attention backends, but DINOv3 ViT source does not construct a causal mask or padding mask for normal image inference.
- Position interaction: q/k receive RoPE only for the final `Hp*Wp` patch tokens. CLS/register prefixes remain unrotated and still attend globally.
- Cache: no KV cache.
- Backend compatibility: source advertises SDPA, FlashAttention, FlexAttention, and generic attention backend support. Eager fallback materializes dense `T x T` attention weights and is likely too slow for large `vit7b16` or high-resolution images.

For `224 x 224`, `T = 201` with official four-register variants. For larger images, attention cost grows with `(floor(H/16) * floor(W/16) + 1 + R)^2`.

## 7. Position encoding and custom math

DINOv3 ViT dynamic RoPE can be reproduced as:

```python
def dinov3_patch_rope(pixel_h, pixel_w, patch, head_dim, theta, dtype):
    hp, wp = pixel_h // patch, pixel_w // patch
    y = (arange(0.5, hp) / hp) * 2.0 - 1.0
    x = (arange(0.5, wp) / wp) * 2.0 - 1.0
    coords = meshgrid(y, x, indexing="ij").reshape(hp * wp, 2)
    inv = 1.0 / (theta ** arange(0, 1, 4 / head_dim, dtype=float32))
    angles = (2 * pi * coords[:, :, None] * inv[None, None, :]).reshape(hp * wp, head_dim // 2)
    angles = tile(angles, 2)
    return cos(angles).astype(dtype), sin(angles).astype(dtype)
```

Application:

```python
def apply_dinov3_rope(q, k, cos, sin):
    # q/k: B x heads x T x head_dim; cos/sin: P x head_dim
    prefix = q.shape[-2] - sin.shape[-2]
    q_pre, q_patch = split(q, [prefix, sin.shape[-2]], dim=-2)
    k_pre, k_patch = split(k, [prefix, sin.shape[-2]], dim=-2)
    q_patch = q_patch * cos + rotate_half(q_patch) * sin
    k_patch = k_patch * cos + rotate_half(k_patch) * sin
    return concat([q_pre, q_patch], dim=-2), concat([k_pre, k_patch], dim=-2)
```

Inference can cache `cos/sin` per `(Hp, Wp, head_dim, theta, dtype, device)`. Training-only `pos_embed_shift`, `pos_embed_jitter`, and `pos_embed_rescale` add randomness and should be excluded from first inference parity.

## 8. Preprocessing and input packing

Processor defaults:

- `do_rescale=True`, then `do_resize=True`, then optional center crop, then `do_normalize=True`.
- Resize target defaults to `224 x 224`.
- Resize is bilinear with `antialias=True`.
- Mean/std defaults are ImageNet defaults.
- Output tensor key is `pixel_values`; model expects NCHW.

Runtime graph shape coupling:

- Input height/width should be divisible by `patch_size` for full image coverage. The model uses floor division through Conv2d and RoPE patch counts, so trailing pixels are ignored by patch embedding if dimensions are not divisible.
- `bool_masked_pos` has shape `B x P` and is only needed for masked/pretraining-style inference.
- No text tokens, placeholder tokens, token type IDs, masks, or generation metadata.
- Backbone consumers need to know whether features are NCHW maps or token sequences via `reshape_hidden_states`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> WindowFlatten + GEMM

Source pattern:

```text
Conv2d(3 -> C, kernel=patch_size, stride=patch_size, padding=0)
-> flatten(2)
-> transpose(1, 2)
```

Replacement:

```text
WindowFlatten_NCHW(B, 3, H, W, ph, pw, stride=ph,pw) -> MatMul(weight_flat.T) -> BiasAdd -> B x P x C
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `H` and `W` divisible by patch size, or fallback preserves Conv2d floor behavior.
- Flatten order matches PyTorch Conv2d weight layout `[out_channels, in_channels, kh, kw]`.

Weight transform:

```python
w = conv.weight.reshape(hidden_size, num_channels * patch_size * patch_size)
```

Layout constraints:

- Source boundary is NCHW. Direct NHWC translation is safe only for this local, fully contained patch extraction if the output remains `B x P x C`.
- A broader layout pass must protect later token-axis `dim=1` concats/slices from accidental channel-axis rewriting.

Failure cases:

- Non-divisible image sizes if the runtime chooses padding instead of Conv2d floor behavior.
- Any future grouped or overlapping patch embedding.

Parity sketch:

- Compare Conv2d+flatten+transpose vs flattened GEMM for random `B,H,W,C` and official patch size 16 in fp32/fp16/bf16.

### Rewrite: split QKV source checkpoint -> independent projections

Source/conversion pattern:

```text
original qkv weight/bias rows are chunked as [Q, K, V] along dim=0
```

Replacement:

```text
q_proj, k_proj, v_proj independent Linear(C -> C)
```

Preconditions:

- Only apply during original Meta checkpoint conversion or loader migration.
- `qkv.shape[0] == 3 * hidden_size`.
- Chunk order is exactly Q, K, V.

Failure cases:

- Already-converted HF weights; applying the split twice would corrupt names.

### Rewrite: backbone feature map materialization guard

Source pattern:

```text
patch_tokens.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
```

Replacement opportunity:

- If the only consumer is NHWC-capable DinoML code, keep feature maps as `B x Hp x Wp x C`.
- Otherwise materialize NCHW because `BackboneOutput.feature_maps` exposes NCHW source semantics.

Preconditions:

- Consumer layout is controlled and declared.
- Axis-sensitive downstream ops rewrite channel axis from `1` to `-1`.

Failure cases:

- External HF-compatible backbone consumers expecting NCHW maps.
- Mixed selected stages with untracked layout metadata.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d lowering to GEMM. This is the first NCHW-heavy op and can avoid a generic convolution path for all variants.
- LayerNorm + Linear projections around attention. Every block uses two LayerNorms and four `C -> C` projections.
- Dynamic RoPE + attention prepack. RoPE applies only to patch tokens, so fused attention needs prefix-aware rotary handling.
- Gated MLP fusion for plus/7B variants: `SiLU(gate) * up` before down projection.

Medium priority:

- LayerScale + residual add. Cheap but repeated twice per block.
- Backbone reshape/permute elision for controlled NHWC consumers.
- Attention backend selection for high-resolution feature extraction; eager dense attention becomes a bottleneck.

Lower priority:

- DropPath support, because inference uses identity.
- `bool_masked_pos` mask-token path for pretraining-style calls.
- Training-time random RoPE coordinate augmentations.

## 11. Runtime staging plan

1. Parse config and load converted HF weights for one accessible checkpoint once access is available; reject raw Meta QKV-packed checkpoints unless loader split is implemented.
2. Implement source-faithful NCHW patch embedding, token prefix assembly, dynamic 2D RoPE, one encoder block, and final norm.
3. Validate base `DINOv3ViTModel` pooled CLS and last hidden state at `224 x 224`.
4. Add backbone output selection, prefix stripping, optional NCHW feature-map reshape, and optional CLS-token return.
5. Add gated MLP variants and `vit7b16` bias/head-dim coverage.
6. Enable optimized attention and prefix-aware RoPE fusion.
7. Add guarded patch Conv2d-to-GEMM and optional NHWC feature-map pass for controlled consumers.

Initially stub or reject training-only DropPath, coordinate augmentation, and masked pretraining inputs.

## 12. Parity and validation plan

- Custom op tests:
  - patch-center coordinate generation for several `(H,W)` divisible by 16.
  - RoPE prefix exclusion with `R = 0` and `R = 4`.
  - patch Conv2d-to-GEMM lowering.
- Single-block parity:
  - ordinary GELU MLP variant: vits/vitb/vitl shape.
  - gated SiLU variant: vits16plus/vith16plus/vit7b shape.
- End-to-end encoder parity:
  - `last_hidden_state[:, 0, :]` CLS.
  - first patch token at index `1 + num_register_tokens`.
  - hidden-state shapes for non-224 dynamic resolutions.
- Backbone parity:
  - selected stages, `reshape_hidden_states=True`, NCHW map shape `B x C x Hp x Wp`.
  - `reshape_hidden_states=False`, token shape `B x P x C`.
  - `return_class_token=True` when enabled.
- Suggested tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5`.
  - fp16/bf16: `rtol=5e-2`, `atol=5e-2` for full-model parity, tighter for isolated ops where feasible.

## 13. Performance probes

- Processor throughput: decode/rescale/resize/normalize with and without shape grouping.
- Patch embedding: Conv2d vs WindowFlatten+GEMM at `224`, `518`, and larger resolutions.
- Encoder throughput by variant: vits, vitb, vitl, vithplus, vit7b.
- Attention backend comparison: eager vs SDPA vs Flash/Flex where available.
- Sequence-length sweep: `T = 1 + R + floor(H/16)*floor(W/16)`.
- Gated vs ungated MLP timing and memory.
- Backbone materialization cost: token output vs NCHW feature maps vs controlled NHWC maps.
- Activation memory by selected hidden-state outputs, especially backbone `output_hidden_states=True`.

## 14. Skip/defer list

- Training, gradient checkpointing, DropPath stochastic behavior.
- Masked image pretraining path using `bool_masked_pos`.
- Training-time RoPE coordinate shift/jitter/rescale.
- Raw Meta checkpoint conversion and QKV split in the runtime loader, unless required before HF weights are available.
- Multi-GPU/tensor parallel sharding for `vit7b16`.
- Classification/detection/segmentation heads; this family source only owns base encoder/backbone outputs.

## 15. Final implementation checklist

- [ ] Parse `DINOv3ViTConfig`, including independent q/k/v/o bias flags.
- [ ] Load HF-converted weights and preserve CLS/register/mask token tensors.
- [ ] Implement NCHW patch Conv2d and `B x P x C` token packing.
- [ ] Implement dynamic DINOv3 2D RoPE coordinate/cos/sin generation.
- [ ] Implement prefix-aware RoPE application over patch tokens only.
- [ ] Implement noncausal MHA encoder block with independent q/k/v projections.
- [ ] Implement ordinary GELU MLP and gated SiLU MLP variants.
- [ ] Implement LayerScale and inference residual path.
- [ ] Implement final LayerNorm and CLS pooled output.
- [ ] Implement backbone stage selection, prefix stripping, and optional NCHW feature-map output.
- [ ] Add guarded patch Conv2d-to-GEMM rewrite.
- [ ] Add guarded NHWC feature-map materialization optimization for controlled consumers.
- [ ] Add parity tests for RoPE, one block, full encoder, and backbone outputs.
- [ ] Benchmark processor, patch embedding, attention backend, MLP, and backbone materialization.
