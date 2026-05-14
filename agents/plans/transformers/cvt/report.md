# CvT DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/cvt-13; representative sweep also covers microsoft/cvt-21, microsoft/cvt-13-384, microsoft/cvt-21-384, microsoft/cvt-13-384-22k, microsoft/cvt-21-384-22k, microsoft/cvt-w24-384-22k
Config source: official Hugging Face config.json and preprocessor_config.json snapshots under _sources/hf_configs/
Source files inspected:
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cvt/configuration_cvt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cvt/modeling_cvt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cvt/convert_cvt_original_pytorch_checkpoint_to_pytorch.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/docs/source/en/model_doc/cvt.md
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/convnext/image_processing_convnext.py
Any missing files or assumptions: no CvT-specific image processor exists; official CvT preprocessor configs use legacy ConvNextFeatureExtractor metadata, so ConvNeXT image preprocessing source is included for end-to-end image pipeline parity. No gated/401/403 gaps were observed for the sampled official configs.
```

Local snapshots:

- `agents/plans/transformers/cvt/_sources/modeling_cvt.py`
- `agents/plans/transformers/cvt/_sources/configuration_cvt.py`
- `agents/plans/transformers/cvt/_sources/convert_cvt_original_pytorch_checkpoint_to_pytorch.py`
- `agents/plans/transformers/cvt/_sources/cvt.md`
- `agents/plans/transformers/cvt/_sources/image_processing_convnext.py`
- `agents/plans/transformers/cvt/_sources/hf_configs/*/{config.json,preprocessor_config.json}`

Primary DinoML target: image classification inference with `CvtForImageClassification`. `CvtModel` encoder-only feature extraction is required as the backbone. Training losses are deferred.

## 2. High-level architecture

CvT is a hierarchical vision encoder with convolutional token embeddings and convolutional Q/K/V projection before noncausal encoder self-attention. The classifier head consumes the final stage CLS token for official configs, applies LayerNorm, averages the one-token sequence, and applies a dense class head.

```text
CPU image resize/crop/normalize -> NCHW pixel_values
  -> stage 0 Conv2d patch embed + token LayerNorm -> CvT encoder blocks
  -> stage 1 Conv2d patch embed + token LayerNorm -> CvT encoder blocks
  -> stage 2 Conv2d patch embed + token LayerNorm + CLS token -> CvT encoder blocks
  -> final CLS LayerNorm/mean -> classifier logits
```

There is no generation, decoder, KV cache, cross-attention, tokenizer coupling, or position-id input. Each stage returns an NCHW feature map plus an optional CLS token.

## 3. Important config dimensions

Source defaults from `CvtConfig`:

| Field | Default / behavior |
| --- | --- |
| `num_channels` | `3` |
| `patch_sizes` | `[7, 3, 3]` |
| `patch_stride` | `[4, 2, 2]` |
| `patch_padding` | `[2, 1, 1]` |
| `embed_dim` | `[64, 192, 384]` |
| `num_heads` | `[1, 3, 6]` |
| `head_dim` | `embed_dim[stage] // num_heads[stage]`; default all stages `64` |
| `depth` | `[1, 2, 10]` |
| `mlp_ratio` | `[4.0, 4.0, 4.0]` |
| `qkv_bias` | `[true, true, true]` for Linear Q/K/V |
| `cls_token` | `[false, false, true]` |
| `qkv_projection_method` | `["dw_bn", "dw_bn", "dw_bn"]` |
| `kernel_qkv` / padding / strides | kernel `[3,3,3]`, `padding_q=[1,1,1]`, `padding_kv=[1,1,1]`, `stride_q=[1,1,1]`, `stride_kv=[2,2,2]` |
| `layer_norm_eps` | `1e-12` |
| Activation | GELU in MLP |
| Cache support | none |

Representative official checkpoint sweep:

| Model | Preprocessor size | Labels | `embed_dim` | `num_heads` | `depth` | Spatial stages for square input | Notes |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `microsoft/cvt-13` | 224 | 1000 | `64/192/384` | `1/3/6` | `1/2/10` | `56 -> 28 -> 14` | common small/base checkpoint |
| `microsoft/cvt-21` | 224 | 1000 | `64/192/384` | `1/3/6` | `1/4/16` | `56 -> 28 -> 14` | deeper stage 1/2 |
| `microsoft/cvt-13-384` | 384 | 1000 | `64/192/384` | `1/3/6` | `1/2/10` | `96 -> 48 -> 24` | larger input, same widths |
| `microsoft/cvt-21-384` | 384 | 1000 | `64/192/384` | `1/3/6` | `1/4/16` | `96 -> 48 -> 24` | deeper and larger input |
| `microsoft/cvt-w24-384-22k` | 384 | 1000 in inspected config | `192/768/1024` | `3/12/16` | `2/2/20` | `96 -> 48 -> 24` | wide variant; all head dims still `64` |

The 22k-named sampled configs still expose 1000 `id2label` entries in the inspected `config.json`; treat the label count as config-derived, not inferred from the repo name.

## 3a. Family variation traps

- CvT keeps public feature tensors as NCHW, but attention and LayerNorm operate on `[B, T, C]`. A layout pass must not blindly rewrite axis attributes around `view(...).permute(...)`.
- Official configs use `qkv_projection_method="dw_bn"`. The config doc says `"avg"` selects linear projection, but current `CvtSelfAttentionProjection.forward` unconditionally calls `self.convolution_projection`; for any non-`dw_bn` value that attribute is not created. DinoML should reject or separately audit non-`dw_bn` configs for this source basis.
- `cls_token` is only true for the final stage in official configs. Source initializes any stage CLS token with `config.embed_dim[-1]`, so earlier true stages with narrower embed dims are not covered by this report and should be rejected unless parity-checked separately.
- There are no absolute/relative position embeddings. Spatial information enters through padded/strided convolutions and convolutional Q/K/V projections.
- Attention score scale is `embed_dim ** -0.5`, not `head_dim ** -0.5`. Fused attention parity must preserve this source math.
- Q has stride 1 in official configs while K/V have stride 2, so attention is rectangular: query length differs from key/value length.
- Dropout and DropPath are present in source but inactive for inference. Training-time stochastic depth should be deferred.
- Preprocessing for size 224 uses resize shortest edge to `int(224 / 0.875)=256` then center crop to 224 in ConvNeXT preprocessing; size 384 warps/resizes to 384 square with no crop in the inspected current processor behavior.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors `[B, 3, H, W]`.
- Conv output shape math `floor((in + 2*pad - kernel) / stride) + 1`.
- `view`/reshape NCHW `[B,C,H,W] -> [B,C,H*W]`.
- `permute(0,2,1)` to sequence `[B,H*W,C]` and inverse.
- CLS split/concat on sequence axis `dim=1`.
- Mean reduction over token axis `dim=1` in classifier.

Neural network primitives:

- Patch embeddings:
  - stage 0 default: `Conv2d(3 -> 64, kernel=7, stride=4, padding=2, bias=True)`.
  - stage 1 default: `Conv2d(64 -> 192, kernel=3, stride=2, padding=1, bias=True)`.
  - stage 2 default: `Conv2d(192 -> 384, kernel=3, stride=2, padding=1, bias=True)`.
  - wide W24: `Conv2d(3 -> 192)`, `Conv2d(192 -> 768)`, `Conv2d(768 -> 1024)` with same kernels/strides/padding.
- Token-axis `LayerNorm(C, eps=1e-12)` after each patch Conv2d.
- Depthwise Conv2d Q/K/V preprojections with `groups=C`, `bias=False`, `kernel=3`, padding 1; Q stride 1, K/V stride 2.
- `BatchNorm2d(C)` after each depthwise Q/K/V convolution.
- Linear Q/K/V: `Linear(C -> C, bias=True)` separately for Q, K, V.
- Attention output Linear: `Linear(C -> C, bias=True)`.
- MLP: `Linear(C -> int(4*C)) -> GELU -> Linear(int(4*C) -> C)`.
- Residual adds and inference identity Dropout/DropPath.
- Classifier: final `LayerNorm(C)` plus `Linear(C -> num_labels)`.

Attention primitives:

- Noncausal self-attention with rectangular Q length vs K/V length.
- Explicit QK matmul/einsum, softmax over key axis, attention-probability times V.
- No causal mask, padding mask, relative bias, RoPE, ALiBi, sliding window, packed/varlen metadata, or cache.

Preprocessing-coupled ops:

- Image resize/crop/rescale/normalize into NCHW `pixel_values`.
- BICUBIC resampling from preprocessor config `resample=3`.

Distributed/tensor-parallel ops:

- None required by source.

## 5. Layer/block breakdown

For square input `S`, each stage patch Conv2d emits `H_s = W_s = floor((S_prev + 2p - k) / stride) + 1`; official 224 input gives `56, 28, 14`, official 384 input gives `96, 48, 24`.

Stage `s`, repeated for three stages:

```text
x_nchw = Conv2d(C_in -> C_s, patch_kernel, patch_stride, patch_padding, bias=True)(x_nchw)
x_seq = reshape_permute(x_nchw)             # [B, H_s*W_s, C_s]
x_seq = LayerNorm(C_s, eps=1e-12)(x_seq)
x_nchw = inverse_reshape_permute(x_seq)     # [B, C_s, H_s, W_s]
x_seq = reshape_permute(x_nchw)
if cls_token[s]:
  x_seq = concat(cls_token[B,1,C_s], x_seq, dim=1)
for layer in depth[s]:
  x_seq = CvTLayer(x_seq, H_s, W_s)
if cls_token[s]:
  cls, patch_seq = split(x_seq, [1, H_s*W_s], dim=1)
x_nchw = inverse_reshape_permute(patch_seq)
```

CvT layer:

```text
u = LayerNorm(C)(x_seq)
attn = CvTSelfAttention(u, H, W)
attn = Linear(C -> C, bias=True)(attn)
x = x_seq + attn                         # dropout/drop-path inactive in inference
m = LayerNorm(C)(x)
m = Linear(C -> 4C, bias=True)(m)
m = GELU(m)
m = Linear(4C -> C, bias=True)(m)
x = x + m
```

CvT self-attention with `dw_bn` projection:

```text
if with_cls:
  cls, patches = split(x_seq, [1, H*W], dim=1)
patches_nchw = patches.permute(0,2,1).view(B,C,H,W)

q_map = BatchNorm2d(C)(DepthwiseConv2d(C -> C, k=3, stride=1, pad=1, groups=C, bias=False)(patches_nchw))
k_map = BatchNorm2d(C)(DepthwiseConv2d(C -> C, k=3, stride=2, pad=1, groups=C, bias=False)(patches_nchw))
v_map = BatchNorm2d(C)(DepthwiseConv2d(C -> C, k=3, stride=2, pad=1, groups=C, bias=False)(patches_nchw))
q_seq = flatten_hw(q_map)                 # [B, H*W, C]
k_seq = flatten_hw(k_map)                 # [B, H_kv*W_kv, C]
v_seq = flatten_hw(v_map)
if with_cls:
  q_seq = concat(cls, q_seq, dim=1)
  k_seq = concat(cls, k_seq, dim=1)
  v_seq = concat(cls, v_seq, dim=1)

q = Linear(C -> C, bias=qkv_bias)(q_seq).view(B, Tq, heads, 64).permute(0,2,1,3)
k = Linear(C -> C, bias=qkv_bias)(k_seq).view(B, Tk, heads, 64).permute(0,2,1,3)
v = Linear(C -> C, bias=qkv_bias)(v_seq).view(B, Tk, heads, 64).permute(0,2,1,3)
scores = einsum(q, k) * (C ** -0.5)
probs = softmax(scores, dim=-1)
context = einsum(probs, v).permute(0,2,1,3).contiguous().view(B, Tq, C)
```

Classifier head for official configs:

```text
cls = outputs.cls_token_value              # [B, 1, C_final]
z = LayerNorm(C_final)(cls)
z = mean(z, dim=1)                         # [B, C_final], no-op for one token
logits = Linear(C_final -> num_labels)(z)
```

## 6. Attention requirements

- Variant: encoder self-attention only, noncausal, no cross-attention.
- Head structure: MHA, no MQA/GQA. Official head dim is 64 in all sampled variants.
- Q/K/V token counts:
  - Without CLS: `Tq = H*W`; `Tk = Tv = H_kv*W_kv`, where `H_kv = floor((H + 2*padding_kv - kernel_qkv) / stride_kv) + 1`.
  - With final-stage CLS: `Tq = 1 + H*W`; `Tk = Tv = 1 + H_kv*W_kv`.
- Masking: none in source. No padding mask is accepted by `forward`.
- Position interactions: no RoPE/ALiBi/relative bias. Convolutional projections provide local spatial mixing before dense attention.
- Backend compatibility: can lower to SDPA/Flash-like noncausal attention only if backend supports rectangular query/key lengths and accepts the custom scale `C ** -0.5`. Otherwise use explicit matmul-softmax-matmul.
- Cache: no autoregressive KV cache or encoder-decoder cache.
- Slow fallback risk: the source uses explicit `einsum` attention and separate depthwise Conv2d/BatchNorm projections; for 384 and W24, stage 2 has `Tq=577` and `Tk=145` with 16 heads/1024 width, so optimized rectangular attention and fused projection/layout handling matter.

## 7. Position encoding and custom math

CvT has no learned absolute position table and no rotary/relative positional bias. The custom math to preserve is convolutional tokenization/projection plus nonstandard attention scale.

```python
def cvt_conv_projection(x_seq, height, width, conv_dw, bn, with_cls=False):
    cls = None
    if with_cls:
        cls, x_seq = x_seq[:, :1], x_seq[:, 1:]
    b, t, c = x_seq.shape
    x = x_seq.transpose(1, 2).reshape(b, c, height, width)
    x = bn(conv_dw(x))                      # depthwise conv, NCHW BatchNorm2d
    qkv_seq = x.reshape(b, c, -1).transpose(1, 2)
    return qkv_seq if cls is None else concat([cls, qkv_seq], axis=1)
```

```python
def cvt_attention(q, k, v, embed_dim):
    # q/k/v are [B, heads, T, head_dim]
    scores = matmul(q, transpose(k, -1, -2)) * (embed_dim ** -0.5)
    probs = softmax(scores, axis=-1)
    return matmul(probs, v)
```

Static items that can be precomputed: convolution weights, BatchNorm inference affine/folded constants, classifier labels, and stage output sizes for fixed input sizes. Dynamic items: batch size and image resolution if DinoML admits non-square or non-official sizes.

## 8. Preprocessing and input packing

Official CvT preprocessor snapshots use:

- `feature_extractor_type`: `ConvNextFeatureExtractor`
- `do_resize=true`, `do_normalize=true`
- `resample=3` (BICUBIC)
- `image_mean=[0.485, 0.456, 0.406]`
- `image_std=[0.229, 0.224, 0.225]`
- `crop_pct=0.875`
- `size=224` or `size=384`

Current ConvNeXT image processor behavior:

- For `size < 384`: resize the shortest edge to `int(size / crop_pct)`, preserving aspect ratio, then center crop to `size x size`.
- For `size >= 384`: resize/warp to square `size x size` with no crop in the custom `resize` method.
- Rescale/normalize produce `pixel_values` in channel-first NCHW.

The image pipeline is CPU/data-pipeline work for first integration. The GPU/runtime graph begins at `pixel_values: [B, 3, size, size]`.

No input packing, masks, token type IDs, modality metadata, or position IDs are used.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> static Conv/GEMM tile

Source pattern:

```text
Conv2d(C_in -> C_out, kernel=k, stride=s, padding=p) -> flatten HW -> LayerNorm(C_out) -> restore NCHW
```

Replacement:

```text
Conv2d or im2col/implicit-GEMM -> NHWC local region -> LayerNorm(last dim) -> restore/export NCHW
```

Preconditions:

- Preserve source padding semantics exactly. Stage 0 has `kernel=7, stride=4, padding=2`; stages 1/2 have `kernel=3, stride=2, padding=1`.
- This is not a non-overlap patchify rewrite because `padding != 0` and stage 0 has overlapping effective windows at borders.
- Output spatial size must match PyTorch Conv2d formula.

Layout constraints:

- Safe NHWC candidate only inside a local patch-embed region if all following reshape/LayerNorm consumers are translated together.
- Public stage output remains NCHW unless every downstream consumer in the region is controlled.

Failure cases:

- Dynamic shapes whose padded Conv2d output differs from assumed fixed 224/384 sizes.
- Any consumer expecting physical NCHW memory immediately after patch Conv2d.

Parity sketch:

- Compare patch embedding output NCHW after inverse reshape for random fixed sizes 224 and 384, including border pixels.

### Rewrite: depthwise Conv2d + BatchNorm2d inference folding

Source pattern:

```text
DepthwiseConv2d(C -> C, groups=C, bias=False) -> BatchNorm2d(C)
```

Replacement:

```text
DepthwiseConv2d(C -> C, groups=C, bias=True folded)
```

Preconditions:

- Inference mode with frozen BatchNorm running mean/var.
- `eps`, `weight`, `bias`, `running_mean`, and `running_var` loaded from source weights.
- Groups equal channels and no later training update.

Weight transform:

```python
scale = bn.weight / sqrt(bn.running_var + bn.eps)
w_fold = conv.weight * scale.reshape(C, 1, 1, 1)
b_fold = bn.bias - bn.running_mean * scale
```

Layout constraints:

- NCHW source translation is direct.
- NHWC depthwise kernel is safe only inside a guarded local region and requires channel-axis rewrite from `dim=1` to `dim=-1` for any adjacent channel ops.

Failure cases:

- Training mode, unfrozen BN stats, or absent BN buffers.

Parity sketch:

- Random NCHW feature maps per stage; compare folded versus source Conv+BN in fp32.

### Rewrite: CvT sequence-layout roundtrip elimination

Source pattern:

```text
NCHW -> view/permute [B,T,C] -> LayerNorm(C) -> permute/view NCHW
```

Replacement:

```text
NHWC local tensor -> LayerNorm(channel) -> either continue NHWC or materialize NCHW at boundary
```

Preconditions:

- The region includes both reshapes and no external consumer observes intermediate stride/order.
- `T` corresponds to row-major `H*W` flattening.
- LayerNorm normalizes channels only.

Required axis rewrites:

- LayerNorm normalized axis from last dim in sequence remains channel-last in NHWC.
- Any `cat/split/mean` on sequence axis must remain on token axis; do not map it to channel axis.

Failure cases:

- Attention immediately requires `[B,T,C]` with CLS token insertion. Use a conceptual `no_layout_translation()` guard around CLS token sequence operations unless the pass explicitly models token axes.

Parity sketch:

- Compare stage embedding output and the sequence entering the first attention block for 224/384 inputs.

### Rewrite: rectangular encoder attention to fused attention

Source pattern:

```text
Q/K/V Linear -> reshape [B,heads,T,D] -> scores = QK^T * C^-0.5 -> softmax -> AV
```

Replacement:

```text
Fused noncausal MHA(q, k, v, scale=C^-0.5, causal=false)
```

Preconditions:

- Backend supports `Tq != Tk`.
- No mask and no dropout in inference.
- Custom scale is passed explicitly.
- Q/K/V have already had convolutional projection and optional CLS concat applied.

Failure cases:

- Backend assumes `scale=head_dim^-0.5`.
- Backend requires equal Q/K sequence lengths.

Parity sketch:

- Stage 2 with CLS at sizes 224 and 384; compare attention context before output projection.

## 10. Kernel fusion candidates

Highest priority:

- Depthwise Conv2d + BatchNorm2d folding/fusion for Q/K/V projections: appears three times per block and dominates CvT-specific operator surface.
- Rectangular noncausal attention with explicit scale: needed for every block, and K/V downsampling changes shapes from vanilla ViT.
- Layout roundtrip fusion around `NCHW <-> [B,T,C] <-> NCHW`: repeated in patch embeddings and attention projections.

Medium priority:

- Patch Conv2d + token LayerNorm local NHWC region: valuable for image throughput, but must preserve padding and NCHW public contracts.
- Linear + GELU + Linear MLP fusion: standard transformer FFN work across 13/21/24 layers.
- Classifier final LayerNorm + one-token mean elision: small but easy cleanup for official CLS-token configs.

Lower priority:

- Dropout/DropPath inference elision: required for graph cleanup, not a kernel.
- General support for non-`dw_bn` QKV projection modes: current source does not safely instantiate them.

## 11. Runtime staging plan

1. Parse `CvtConfig`, official preprocessor config, and reject unsupported non-`dw_bn`/early-CLS variants.
2. Load weights for `CvtModel` and `CvtForImageClassification`, including BN buffers and final classifier.
3. Implement patch embedding Conv2d -> sequence LayerNorm -> NCHW restore for one stage.
4. Implement one CvT layer with depthwise Q/K/V Conv+BN, rectangular attention, MLP, and residuals.
5. Run full encoder parity for CvT-13 at 224, then CvT-21 and 384 variants.
6. Add classifier head parity and end-to-end image preprocessing parity.
7. Enable BN folding and attention/layout fusions behind guards.
8. Add W24 width/depth performance validation.

Initial stubs: labels/id2label formatting, training losses, dropout/drop-path active behavior, and non-`dw_bn` configs.

## 12. Parity and validation plan

- Random tensor tests:
  - NCHW-to-sequence and inverse layout roundtrip for several `B,C,H,W`.
  - Patch Conv2d output shape/value parity for 224 and 384.
  - Depthwise Conv2d+BN folding parity in fp32.
  - Rectangular attention with `Tq != Tk` and scale `C ** -0.5`.
- Single-layer parity:
  - Stage 0 layer without CLS.
  - Stage 2 layer with CLS and K/V stride 2.
- Encoder parity:
  - CvT-13 after each stage at 224.
  - CvT-21 at 224 and 384.
  - W24 at 384 for wide channel stress.
- Classifier parity:
  - Final CLS `LayerNorm -> mean(dim=1) -> Linear` logits.
- End-to-end parity:
  - Image processor output `pixel_values` for 224 and 384 configs.
  - Top-1 logits for official sample image.
- Suggested tolerances:
  - fp32: `atol=1e-5`, `rtol=1e-4`.
  - fp16/bf16: `atol=2e-2`, `rtol=2e-2`, with attention softmax checked separately.

## 13. Performance probes

- Image preprocessing throughput for 224 resize/crop and 384 resize paths.
- Patch embedding throughput per stage, comparing NCHW Conv2d versus guarded NHWC/implicit-GEMM lowering.
- Depthwise Conv+BN Q/K/V projection throughput with and without BN folding.
- Attention backend comparison for rectangular shapes:
  - CvT-13 224 stage 2: `Tq=197`, `Tk=50`, heads 6, dim 64.
  - CvT-13 384 stage 2: `Tq=577`, `Tk=145`, heads 6, dim 64.
  - W24 384 stage 2: `Tq=577`, `Tk=145`, heads 16, dim 64.
- Batch-size sweep for 224 and 384.
- Full encoder-only images/sec and classifier logits/sec.
- Memory traffic probe around repeated permute/contiguous/view regions.

## 14. Skip/defer list

- Training losses and `problem_type` mutation.
- Active dropout and stochastic depth behavior.
- Non-`dw_bn` / documented `"avg"` projection mode until source parity is clarified.
- Early-stage CLS-token configs.
- Multi-GPU/tensor parallel.
- Quantization.
- Detection/segmentation heads; CvT source only implements base model and image classification.

## 15. Final implementation checklist

- [ ] Parse `CvtConfig` arrays and validate equal stage lengths.
- [ ] Load CvT weights, including Conv2d, Linear, LayerNorm, BatchNorm buffers, CLS token, classifier.
- [ ] Load/replicate ConvNeXT-style image preprocessor config for CvT checkpoints.
- [ ] Implement NCHW patch Conv2d embedding with token-axis LayerNorm.
- [ ] Implement guarded NCHW-to-sequence and sequence-to-NCHW layout ops.
- [ ] Implement depthwise Conv2d+BatchNorm Q/K/V projection.
- [ ] Implement final-stage CLS concat/split behavior.
- [ ] Implement rectangular noncausal MHA with `scale=embed_dim**-0.5`.
- [ ] Implement CvT MLP and residual block.
- [ ] Implement classifier head and CLS-token pooling path.
- [ ] Reject or separately route non-`dw_bn` and early-CLS configs.
- [ ] Add BN folding rewrite.
- [ ] Add guarded local NHWC/layout fusion rewrite for patch/projection regions.
- [ ] Add one-block, stage, encoder, classifier, and image-preprocessor parity tests.
- [ ] Benchmark 224/384 and W24 projection/attention bottlenecks.
