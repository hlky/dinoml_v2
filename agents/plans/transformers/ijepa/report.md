# DinoML Transformers Audit: `ijepa`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/ijepa_vith14_1k, facebook/ijepa_vith14_22k, facebook/ijepa_vith16_1k, facebook/ijepa_vitg16_22k
Config source: official Hugging Face config.json and preprocessor_config.json snapshots
Source files inspected:
  - X:/H/transformers/src/transformers/models/ijepa/configuration_ijepa.py
  - X:/H/transformers/src/transformers/models/ijepa/modeling_ijepa.py
  - X:/H/transformers/src/transformers/models/ijepa/modular_ijepa.py
  - X:/H/transformers/src/transformers/models/ijepa/convert_ijepa_to_hf.py
  - X:/H/transformers/src/transformers/models/vit/image_processing_vit.py
  - X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
  - modeling_ijepa.py is generated from modular_ijepa.py; modular_ijepa.py is the future-edit source.
  - Official repos expose preprocessor_config.json, not image_processor_config.json.
  - No gated/401/403 gaps found for the sampled official Facebook checkpoints.
  - The public official checkpoints are base IJepaModel encoders, not image-classification heads.
```

Snapshots were written under `agents/plans/transformers/ijepa/_sources/`.

Primary DinoML target: encoder-only image feature extraction first; `IJepaForImageClassification` is a small optional head.

## 2. High-level architecture

I-JEPA in Transformers is a ViT-like image encoder without a CLS token in the embedding sequence. The source patchifies NCHW image tensors with a non-overlapping `Conv2d`, adds learned absolute patch-position embeddings, runs noncausal encoder self-attention blocks, applies a final `LayerNorm`, and optionally classifies from the mean over all patch tokens.

```text
CPU image preprocessing -> NCHW pixel_values
  -> Conv2d patch embedding -> patch tokens + absolute positions
  -> repeated encoder self-attention/MLP blocks
  -> final LayerNorm -> patch-token features
  -> optional mean-pool classifier logits
```

There is no autoregressive decode stage and no KV cache. `bool_masked_pos` and `mask_token` support are present in the base embedding module only when `IJepaModel(..., use_mask_token=True)` is constructed; the shipped public configs use `IJepaModel` and do not require a pretraining predictor/runtime decoder.

## 3. Important config dimensions

| Field | Source default | `vith14_1k` / `vith14_22k` | `vith16_1k` | `vitg16_22k` |
|---|---:|---:|---:|---:|
| architecture | config/source | `IJepaModel` | `IJepaModel` | `IJepaModel` |
| image_size | 224 | 224 | 448 | 224 |
| patch_size | 16 | 14 | 16 | 16 |
| num_patches | 196 | 256 | 784 | 196 |
| num_channels | 3 | 3 | 3 | 3 |
| hidden_size | 768 | 1280 | 1280 | 1408 |
| num_hidden_layers | 12 | 32 | 32 | 40 |
| num_attention_heads | 12 | 16 | 16 | 16 |
| head_dim | hidden / heads = 64 | 80 | 80 | 88 |
| q/k/v projection width | hidden_size | 1280 | 1280 | 1408 |
| intermediate_size | 3072 | 5120 | 5120 | 6144 |
| hidden_act | gelu | gelu | gelu | gelu |
| qkv_bias | true | true | true | true |
| attention dropout | 0.0 | 0.0 | 0.0 | 0.0 |
| hidden dropout | 0.0 | 0.0 | 0.0 | 0.0 |
| layer_norm_eps | 1e-12 | 1e-6 | 1e-6 | 1e-6 |
| pooler_output_size | defaults hidden_size | omitted, source default | omitted, source default | omitted, source default |
| torch_dtype | config metadata | float32 | float32 | float32 |
| cache support | source | none | none | none |

Preprocessor sweep: all sampled official repos use `ViTImageProcessor`, `do_resize=true`, `do_rescale=true`, `rescale_factor=1/255`, `do_normalize=true`, `image_mean=[0.5,0.5,0.5]`, `image_std=[0.5,0.5,0.5]`, `resample=2` (bilinear). The 448 model uses preprocessor size 448x448; the others use 224x224.

## 3a. Family variation traps

- I-JEPA has no CLS token in the patch embedding sequence. Do not apply ViT-style `[:, 0]` classification unless explicitly using `IJepaPooler`; `IJepaForImageClassification` uses `sequence_output.mean(dim=1)`.
- Absolute position embeddings are shaped `[1, num_patches, hidden]`, not `[1, 1 + num_patches, hidden]`.
- Public model configs omit `pooler_output_size` and `pooler_act`; effective source defaults are `hidden_size` and `"tanh"`.
- `head_dim` is inferred as `hidden_size // num_attention_heads`; sampled official models use 80 or 88, not a universal 64.
- `interpolate_pos_encoding=False` enforces exact configured image height/width. Higher-resolution or non-square use requires the explicit forward flag.
- The patch embedding source layout is NCHW and the flatten order is `Conv2d(N,C,H,W) -> [B,D,Hp,Wp] -> flatten spatial -> transpose -> [B,Hp*Wp,D]`.
- NHWC translation should be a guarded layout/fusion optimization around local conv/flatten consumers only. Axis-sensitive source ops include channel check `pixel_values.shape[1]`, `flatten(2)`, `transpose(1,2)`, attention transpose, and classifier `mean(dim=1)`.
- The `mlp_ratio` config field appears in official configs but current modeling source reads `intermediate_size` directly.
- The conversion script documents original huge/giant checkpoint variants and source-key transforms, but it is not part of inference runtime.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW input guard: `pixel_values.shape[1] == num_channels`.
- Optional dtype cast of `pixel_values` to patch Conv2d weight dtype.
- Conv output flatten/transpose: `[B,D,Hp,Wp] -> [B,Hp*Wp,D]`.
- Reshape/view/transpose for attention heads: `[B,S,D] -> [B,H,S,Hd]`.
- Final classifier mean over sequence axis: `[B,S,D].mean(dim=1)`.
- Optional hidden/attention output capture from source decorators.

### Neural network primitives

- Patch `Conv2d(3 -> hidden_size, kernel=patch_size, stride=patch_size, padding=0, groups=1, bias=True)`.
  - vith14: `Conv2d(3 -> 1280, 14x14, stride 14)`, output 16x16 at 224.
  - vith16_448: `Conv2d(3 -> 1280, 16x16, stride 16)`, output 28x28 at 448.
  - vitg16: `Conv2d(3 -> 1408, 16x16, stride 16)`, output 14x14 at 224.
- LayerNorm over hidden axis, eps from config.
- Linear Q/K/V with bias: `Linear(D -> num_heads * head_dim)`.
- Output projection with bias: `Linear(num_heads * head_dim -> D)`.
- MLP: `Linear(D -> intermediate_size) -> GELU -> Linear(intermediate_size -> D)`.
- Residual adds and dropout nodes; dropout is 0 in official inference configs.
- Optional pooler: first patch token `Linear(D -> pooler_output_size) -> tanh`.
- Optional classifier: `Linear(D -> num_labels)` or identity when `num_labels <= 0`.

### Attention primitives

- Encoder bidirectional dense self-attention.
- MHA only: `num_key_value_heads == num_attention_heads`.
- Attention score scaling `head_dim ** -0.5`.
- Additive attention mask from `create_bidirectional_mask`; may be `None` when the backend can skip a full bidirectional mask.
- Softmax in fp32 for eager path, cast back to query dtype.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager, SDPA, FlashAttention, or FlexAttention compatibility.

### Position/custom ops

- Learned absolute patch-position embedding add.
- Optional bicubic interpolation of patch-position table with `align_corners=False`.
- Optional mask-token replacement before position add when model is constructed with `use_mask_token=True`.

### Generation/cache ops

- No prefill/decode/generation cache required.
- Encoder outputs can be cached by an application for retrieval or downstream heads, but this is not a Transformers KV cache.

### Preprocessing-coupled ops

- Resize to configured square size, bilinear.
- Rescale by `1/255`.
- Normalize with mean/std `[0.5,0.5,0.5]`.
- Processor emits channel-first `pixel_values` for the PyTorch model contract.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B, 3, H, W]
patches = Conv2d(3 -> D, kernel=P, stride=P, bias=True)  # [B, D, H/P, W/P]
x = flatten_spatial(patches).transpose(1, 2)             # [B, S, D]
optional x = where(bool_masked_pos[...,None], mask_token, x)
x = x + position_embeddings_or_interpolated_positions    # [B, S, D]
```

Encoder block, repeated `L` times:

```text
residual = x
y = LayerNorm(D, eps)(x)
q = Linear(D -> H*Hd, bias=qkv_bias)(y).view(B,S,H,Hd).transpose(1,2)
k = Linear(D -> H*Hd, bias=qkv_bias)(y).view(B,S,H,Hd).transpose(1,2)
v = Linear(D -> H*Hd, bias=qkv_bias)(y).view(B,S,H,Hd).transpose(1,2)
y = Attention(q, k, v, bidirectional_mask, scale=Hd**-0.5)
y = Linear(H*Hd -> D, bias=True)(merge_heads(y))
x = residual + Dropout(y)

residual = x
y = LayerNorm(D, eps)(x)
y = Linear(D -> I, bias=True)(y)
y = GELU(y)
y = Linear(I -> D, bias=True)(y)
x = residual + Dropout(y)
```

Output/head:

```text
sequence_output = LayerNorm(D, eps)(x)
base output = sequence_output                         # [B, S, D]
optional pooler = tanh(Linear(D -> pooler_output)(sequence_output[:, 0]))
classification logits = Linear(D -> num_labels)(sequence_output.mean(dim=1))
```

Note: because there is no CLS token, `sequence_output[:, 0]` is the first patch token if the pooler is enabled. The classifier source intentionally uses mean pooling.

## 6. Attention requirements

- Type: noncausal bidirectional encoder self-attention.
- Heads: MHA, no MQA/GQA. Official variants: 16 heads with head dim 80 or 88.
- Query/key/value length: square patch grid length `S = floor(H/P) * floor(W/P)` after preprocessing or explicit interpolation. Public configs use `S=256`, `784`, or `196`.
- Q/K width: `num_heads * head_dim`, equal to hidden size in sampled official configs.
- Value width: same as Q/K width.
- Masking: `attention_mask` can be supplied as 2D `[B, S]` padding mask or prepared 4D `[B,1,S,S]`; otherwise full bidirectional masks can be skipped by some attention implementations.
- Packed/varlen: no model-specific packed metadata.
- Sliding/local attention: none.
- ALiBi/RoPE/relative bias: none.
- KV cache: none.
- Backend compatibility: source advertises SDPA, FlashAttention, FlexAttention, and generic attention backend support. Eager path is straightforward but too slow for large `S=784`, `D=1280`, `L=32` if used as the optimized path.

## 7. Position encoding and custom math

The model uses a learned absolute position table over patch tokens only. There is no CLS row. If `interpolate_pos_encoding=True`, the source reshapes the square training grid to image-like layout, bicubic-resizes it, and flattens back.

```python
def ijepa_interpolate_positions(pos, height, width, patch_size):
    # pos: [1, old_h * old_w, dim], old_h == old_w
    dim = pos.shape[-1]
    old = int(pos.shape[1] ** 0.5)
    new_h = height // patch_size
    new_w = width // patch_size
    x = pos.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    x = bicubic_interpolate(x, size=(new_h, new_w), align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
```

Precompute static position tables for the configured image size. Dynamic interpolation depends on runtime image height/width and must preserve the floor division by patch size. Inputs not divisible by patch size are not rejected by patch Conv2d; leftover border pixels are dropped by convolution and interpolation also uses `height // patch_size`, so parity tests should include non-divisible dimensions only when intentionally supporting that source behavior.

Mask-token replacement, only if `use_mask_token=True`:

```python
def replace_masked_patches(x, mask, mask_token):
    m = mask.unsqueeze(-1).to(x.dtype)
    return x * (1.0 - m) + mask_token.expand_as(x) * m
```

## 8. Preprocessing and input packing

CPU/data pipeline:

- Decode image to RGB-like 3-channel tensor.
- Resize with bilinear resampling to the processor `size`.
- Rescale by `1/255`.
- Normalize with `[0.5, 0.5, 0.5]` mean and std.
- Emit channel-first `pixel_values` `[B,3,H,W]`.

GPU/runtime graph:

- Patch embedding receives NCHW.
- No token type IDs, grid metadata, packed descriptors, or attention masks are produced by the normal image processor.
- Optional `bool_masked_pos` has shape `[B, num_patches]` and is caller-supplied, not generated by the public image processor.

Postprocessing:

- Base `IJepaModel` returns patch-token features. There is no source postprocessor.
- `IJepaForImageClassification` returns logits after mean pooling; standard label mapping is config metadata outside the model math.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> WindowFlatten + GEMM

Source pattern:

```text
Conv2d(C -> D, kernel=(P,P), stride=(P,P), padding=0, groups=1)
  -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW, window=P, stride=P, order=source Conv2d/im2col)
  -> MatMul(weight_flat.T) -> BiasAdd -> [B, S, D]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input channel count equals `config.num_channels`.
- Flatten order matches PyTorch Conv2d spatial order and source `flatten(2).transpose(1,2)`.
- Dynamic shapes must use `S = floor(H/P) * floor(W/P)`.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b = conv.bias
```

Layout constraints:

- Direct translation stays NCHW.
- NHWC is safe only inside a controlled patch-lowering region that also rewrites channel axis checks and weight layout. This is a good place for a conceptual `no_layout_translation()` boundary around the public NCHW model input/output unless the pass owns all consumers.

Failure cases:

- Non-square or tuple patch sizes must use the tuple values exactly.
- Non-divisible H/W drop border pixels in Conv2d; do not silently pad.
- If an upstream processor emits pre-flattened patches, use a separate input contract rather than this rewrite.

Parity test sketch:

- Compare Conv2d+flatten+transpose against WindowFlatten+GEMM for `P=14`, `P=16`, H/W divisible and non-divisible, fp32 and bf16.

### Rewrite: split Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(D -> H*Hd)
k = Linear(D -> H*Hd)
v = Linear(D -> H*Hd)
```

Replacement:

```text
PackedLinear(D -> 3*H*Hd) -> split [q, k, v] in all-Q/all-K/all-V row order
```

Preconditions:

- Same input tensor for Q/K/V.
- Same dtype/device.
- Preserve bias when `qkv_bias=True`.
- Split order is Q block, then K block, then V block.

Weight transform:

```python
w_qkv = concat([q.weight, k.weight, v.weight], axis=0)
b_qkv = concat([q.bias, k.bias, v.bias], axis=0)
```

Failure cases:

- Do not use original timm checkpoint packed order without conversion; the current HF source stores separate `q_proj`, `k_proj`, `v_proj`.

### Rewrite: skip full bidirectional mask

Source pattern:

```text
create_bidirectional_mask(..., attention_mask=None) -> maybe None
Attention(q,k,v,None)
```

Replacement:

```text
Dense noncausal attention without materialized mask
```

Preconditions:

- No padding mask, no custom mask functions.
- Attention backend supports unmasked noncausal attention.
- No output attention tensor requirement that forces eager materialization.

Failure cases:

- Supplied 2D/4D attention masks must be preserved.
- Capturing attentions may require dense attention weights.

### Rewrite: guarded layout cleanup around patch embed

Source pattern:

```text
NCHW Conv2d -> flatten(2) -> transpose(1,2) -> token-space ops
```

Replacement:

```text
NHWC local patch extraction/GEMM -> token-space [B,S,D]
```

Preconditions:

- The layout pass owns the Conv2d lowering and immediate flatten/transpose.
- No external consumer observes the intermediate `[B,D,Hp,Wp]`.
- Axis-sensitive source checks are rewritten under guard: channel `dim=1` becomes last channel only inside the optimized region.

Failure cases:

- Public model input remains NCHW.
- Position interpolation uses an image-like `[1,D,Hpos,Wpos]` temporary; keep its source axes or separately prove an NHWC interpolation variant.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d lowering to GEMM/window-flatten for P14/P16. It is the first large operator and gives a clean token input.
- Encoder LayerNorm + QKV packed projection. Large huge/giant models have 32-40 layers and high hidden sizes.
- FlashAttention/SDPA noncausal MHA for S up to 784, D up to 1408. Eager attention is not an acceptable production path for the 448 model.

Medium priority:

- MLP `Linear -> GELU -> Linear` scheduling/fusion, especially `1280 -> 5120 -> 1280` and `1408 -> 6144 -> 1408`.
- Residual add/dropout elimination for inference configs where dropout probability is 0.
- Position add fused with patch output or first LayerNorm when static image size is used.
- Mean-pool classifier head fused with final projection for classification variants.

Lower priority:

- Pooler first-token path, because official public configs are base encoders and classification uses mean pooling.
- Dynamic bicubic position interpolation cache by `(height,width,patch_size)`.
- Attention weight output materialization, mostly diagnostic.

## 11. Runtime staging plan

1. Parse `IJepaConfig`, including omitted defaults for `pooler_output_size` and `pooler_act`.
2. Load base encoder weights and verify patch embedding plus final LayerNorm shapes for `vith14`, `vith16_448`, and `vitg16`.
3. Implement patch Conv2d path faithfully in NCHW and add one-block encoder parity.
4. Implement full encoder parity with unmasked noncausal attention.
5. Add optional `interpolate_pos_encoding=True` path.
6. Add optional image-classification head with mean pooling.
7. Add optimized attention backend and packed QKV rewrite.
8. Add guarded patch Conv2d-to-GEMM and local NHWC fusion pass.

Initially stub or defer labels/losses, training masks, gradient checkpointing, and attention output tensors.

## 12. Parity and validation plan

- Random tensor test for patch Conv2d lowering with P14/P16, H/W divisible and non-divisible.
- Position interpolation parity for 224->448 and rectangular synthetic H/W, fp32 first.
- Mask-token replacement parity with synthetic `bool_masked_pos` when `use_mask_token=True`.
- Single attention block parity with supplied and absent attention masks.
- After-N-layer parity for `vith14_1k` and `vitg16_22k` shapes.
- Full encoder `last_hidden_state` parity on deterministic preprocessed image tensors.
- Optional classifier parity on a synthetic `num_labels` config, checking mean pooling over all patch tokens.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; bf16/fp16 `rtol=3e-2, atol=3e-2` after backend-specific calibration.

No DinoML tests were run for this report.

## 13. Performance probes

- Image preprocessing throughput for 224 and 448 processor sizes.
- Patch embedding throughput: Conv2d baseline vs WindowFlatten+GEMM.
- Encoder-only throughput by variant: `S=196/256/784`, `L=32/40`, `D=1280/1408`.
- Attention backend comparison: eager vs SDPA vs FlashAttention/FlexAttention.
- Batch-size sweep for feature extraction.
- Resolution sweep with `interpolate_pos_encoding=True`.
- Memory bandwidth/activation footprint for hidden-state capture disabled vs enabled.
- MLP GEMM utilization for huge and giant variants.
- Optional classifier head latency and mean-pool overhead.

## 14. Skip/defer list

- Training, loss computation, and gradient checkpointing.
- I-JEPA pretraining predictor/context-target masking pipeline; not implemented as a public inference head in this source.
- Autoregressive generation, beam search, speculative decoding, and KV cache.
- Multi-GPU tensor parallel.
- Quantization-specific loading/provider rules.
- Dense attention weight return unless needed for debugging.
- Layout translation outside the local patch-lowering/fusion region.

## 15. Final implementation checklist

- [ ] Parse `IJepaConfig` and source defaults.
- [ ] Load official encoder weights for `vith14`, `vith16_448`, and `vitg16`.
- [ ] Implement NCHW patch Conv2d embedding.
- [ ] Implement absolute patch-position add.
- [ ] Implement optional bicubic position interpolation.
- [ ] Implement encoder LayerNorm/MHA/MLP block.
- [ ] Implement unmasked and masked bidirectional attention paths.
- [ ] Implement final LayerNorm output.
- [ ] Implement optional mask-token replacement path.
- [ ] Implement optional mean-pool image-classification head.
- [ ] Add patch Conv2d-to-GEMM rewrite.
- [ ] Add packed QKV projection rewrite.
- [ ] Add guarded NHWC patch-embed fusion pass.
- [ ] Add parity tests for patch lowering, position interpolation, one block, full encoder, and classifier head.
- [ ] Benchmark preprocessing, patch embedding, attention backend, MLP throughput, and batch/resolution sweeps.
