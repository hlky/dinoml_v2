# SwiftFormer full-audit report

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: MBZUAI/swiftformer-xs, MBZUAI/swiftformer-s, MBZUAI/swiftformer-l1, MBZUAI/swiftformer-l3
Config source: Hugging Face config.json and preprocessor_config.json for the four MBZUAI repos
Primary runtime target: vision image classification
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/swiftformer/modeling_swiftformer.py`
- `X:/H/transformers/src/transformers/models/swiftformer/configuration_swiftformer.py`
- `X:/H/transformers/src/transformers/models/swiftformer/convert_swiftformer_original_to_hf.py`
- `X:/H/transformers/src/transformers/models/auto/image_processing_auto.py`
- `X:/H/transformers/src/transformers/models/vit/image_processing_vit.py`
- `X:/H/transformers/src/transformers/image_processing_utils.py`
- `X:/H/transformers/src/transformers/image_transforms.py`

Primary URLs:

- [MBZUAI/swiftformer-xs config](https://huggingface.co/MBZUAI/swiftformer-xs/raw/main/config.json)
- [MBZUAI/swiftformer-s config](https://huggingface.co/MBZUAI/swiftformer-s/raw/main/config.json)
- [MBZUAI/swiftformer-l1 config](https://huggingface.co/MBZUAI/swiftformer-l1/raw/main/config.json)
- [MBZUAI/swiftformer-l3 config](https://huggingface.co/MBZUAI/swiftformer-l3/raw/main/config.json)

Any missing files or assumptions:

- No dedicated SwiftFormer image processor source exists. `AutoImageProcessor` maps `swiftformer` to `ViTImageProcessor`.
- No gated or 401 checkpoint/config links were encountered. `model.safetensors.index.json` returned 404 for the sampled repos, because `swiftformer-xs` lists a single `model.safetensors` and the larger sampled repos list `pytorch_model.bin`.
- Report scope is in-library Transformers source. No `trust_remote_code` path is required for the sampled checkpoints.

## 2. High-level architecture

SwiftFormer is a vision-only convolutional/additive-attention encoder with an image-classification head. It is not an autoregressive text model and does not use token generation, KV cache, SDPA, or FlashAttention.

```text
PIL/array image -> ViTImageProcessor -> pixel_values [B,3,224,224] NCHW
-> two-conv patch stem [B,C0,56,56]
-> 4 stages of conv encoders plus one additive-attention block per stage
-> final BN -> spatial mean pool -> two classifier heads -> averaged logits [B,num_labels]
```

Stage decomposition:

- CPU/data pipeline: resize to 224x224, rescale, ImageNet normalize, emit `pixel_values`.
- GPU/runtime encoder: NCHW conv stem, depthwise/pointwise conv blocks, stage downsample convs, additive spatial token mixer.
- GPU/runtime head: final BatchNorm2d, spatial flatten/mean, two dense heads, logits average.
- Independently validatable units: image processor, stem, one `SwiftFormerConvEncoder`, one final `SwiftFormerEncoderBlock`, one full stage, classification head.

## 3. Important config dimensions

Source defaults in `SwiftFormerConfig`:

| Field | Default | Runtime effect |
|---|---:|---|
| `image_size` | 224 | Expected processor resize target; modeling source itself does not validate it. |
| `num_channels` | 3 | Stem input channels. |
| `depths` | `[3,3,6,4]` | Blocks per stage; last block of each stage is additive-attention block. |
| `embed_dims` | `[48,56,112,220]` | Channel widths per stage. |
| `mlp_ratio` | 4 | Conv-MLP expansion width. |
| `downsamples` | `[true,true,true,true]` | Downsample between stages when not last stage. |
| `hidden_act` | `gelu` | Activation in `SwiftFormerMlp`; local/conv encoder hard-code GELU. |
| `down_patch_size`, `down_stride`, `down_pad` | `3,2,1` | Inter-stage downsample Conv2d geometry. |
| `drop_path_rate` | 0.0 | Stochastic depth, inference identity. |
| `drop_mlp_rate`, `drop_conv_encoder_rate` | 0.0 | Dropout, inference identity. |
| `use_layer_scale` | true | Enables per-channel learned layer-scale on attention and MLP residuals. |
| `layer_scale_init_value` | `1e-5` | Initial attention/MLP block residual scale. |
| `batch_norm_eps` | `1e-5` | BatchNorm2d epsilon. |

Representative checkpoint sweep:

| Model id | Depths | Channels | ConvEncoder count | Additive-attention blocks | Final spatial size at 224 | Head width |
|---|---:|---:|---:|---:|---:|---:|
| `MBZUAI/swiftformer-xs` | `3/3/6/4` | `48/56/112/220` | 12 | 4 | 7x7 | 220 |
| `MBZUAI/swiftformer-s` | `3/3/9/6` | `48/64/168/224` | 17 | 4 | 7x7 | 224 |
| `MBZUAI/swiftformer-l1` | `4/3/10/5` | `48/96/192/384` | 18 | 4 | 7x7 | 384 |
| `MBZUAI/swiftformer-l3` | `4/4/12/6` | `64/128/320/512` | 22 | 4 | 7x7 | 512 |

The sampled checkpoint configs omit `image_size`, `drop_mlp_rate`, and `drop_conv_encoder_rate`; current source supplies defaults `224`, `0.0`, and `0.0`.

## 3a. Family variation traps

- This is not standard Transformer attention. `SwiftFormerEfficientAdditiveAttention` uses Q and K linear projections, L2 normalization, a learned vector `w_g`, reduction across spatial tokens, and no QK score matrix.
- Source attention computes `query_weight` with shape `[B,N,1]` and applies `softmax(dim=-1)`. Since the last dimension is singleton, the softmax is always 1. DinoML should mirror this source behavior unless a separate compatibility decision is made.
- The source graph is semantically NCHW for convolutions and BatchNorm2d. The attention sub-block explicitly permutes to `[B,H,W,C]`, reshapes to `[B,H*W,C]`, then permutes back.
- BatchNorm2d uses channel axis 1. Any NHWC layout pass must rewrite or fuse BatchNorm and Conv axes, not merely reinterpret tensors.
- `flatten(2).mean(-1)` in the classifier head pools over spatial axes after NCHW final features. Under NHWC this becomes reduction over axes 1 and 2, not the last axis after a naive flatten.
- Inter-stage downsample is present after stages 0, 1, and 2 for sampled configs. `downsamples[3]` is ignored because there is no next stage.
- `hidden_act` only controls the MLP block. ConvEncoder and LocalRepresentation use `nn.GELU()` directly.
- Inference should treat `Dropout`, `DropPath`, and training loss as deferred or identity. Training mode includes stochastic operations.
- Two classifier heads are both required for checkpoint parity: `head`, `dist_head`, then average.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input tensor `[B,3,H,W]`, expected first target `[B,3,224,224]`.
- `permute(0,2,3,1)`, reshape `[B,H,W,C] -> [B,H*W,C]`, inverse reshape, `permute(0,3,1,2)`.
- `flatten(2)` and `mean(-1)` for spatial global average pooling.
- `unsqueeze`, `repeat`, and reduction `sum(dim=1)` in additive attention. Optimization can avoid materializing `repeat`.

Neural network primitives:

- Conv2d 3x3 stride2 padding1 for stem and downsample.
- Depthwise Conv2d 3x3 padding1 groups=`C`.
- Pointwise Conv2d 1x1, mainly `C -> 4C -> C` and `C -> C`.
- BatchNorm2d inference with affine scale/bias and running mean/variance.
- ReLU in stem.
- GELU in conv and local/MLP blocks.
- Linear `C -> C` for attention `to_query`, `to_key`, `proj`, `final`.
- Linear `C_last -> num_labels` for two classifier heads.
- Elementwise add, multiply, divide by scalar 2, learned per-channel layer-scale multiply.
- Dropout/DropPath as inference identity.

Attention primitives:

- L2 normalize over last dimension for query/key, equivalent to `x / max(norm2(x), eps)` with PyTorch default `eps=1e-12`.
- GEMV/matmul `query @ w_g`, where `w_g` is `[C,1]`.
- Softmax over singleton last dimension, source-derived parity behavior.
- Weighted token sum over spatial token axis.
- Per-token multiply `global_query * key`, linear projection, residual add with query, final linear.

Preprocessing-coupled ops:

- Resize to 224x224 with ViTImageProcessor defaults.
- Rescale by `1/255` via BaseImageProcessor default, because sampled preprocessor configs omit `do_rescale`.
- Normalize by ImageNet mean/std.
- Emit channel-first `pixel_values` for PyTorch model execution.

Not required:

- RoPE, ALiBi, relative bias, masks, causal attention, KV cache, tokenizers, generation, sparse/block attention, quantized or packed weight formats.

## 5. Layer/block breakdown

Stem:

```text
x: [B,3,224,224]
x = Conv2d(3 -> C0/2, kernel=3, stride=2, pad=1, bias=True)(x)
x = BatchNorm2d(C0/2)(x)
x = ReLU(x)
x = Conv2d(C0/2 -> C0, kernel=3, stride=2, pad=1, bias=True)(x)
x = BatchNorm2d(C0)(x)
x = ReLU(x)
out: [B,C0,56,56]
```

ConvEncoder, repeated `depth_i - 1` times per stage:

```text
residual = x
x = DepthwiseConv2d(C -> C, kernel=3, pad=1, groups=C)(x)
x = BatchNorm2d(C)(x)
x = Conv2d(C -> 4C, kernel=1)(x)
x = GELU(x)
x = Conv2d(4C -> C, kernel=1)(x)
x = residual + layer_scale[C,1,1] * x
```

EncoderBlock, final block of each stage:

```text
x = LocalRepresentation(C)(x)
tokens = permute_NCHW_to_NHWC(x).reshape(B, H*W, C)
attn = EfficientAdditiveAttention(C)(tokens)
attn = attn.reshape(B,H,W,C).permute(0,3,1,2)
x = x + layer_scale_1[C,1,1] * attn
x = x + layer_scale_2[C,1,1] * MLP(C -> 4C -> C)(x)
```

LocalRepresentation:

```text
residual = x
x = DepthwiseConv2d(C -> C, kernel=3, pad=1, groups=C)(x)
x = BatchNorm2d(C)(x)
x = Conv2d(C -> C, kernel=1)(x)
x = GELU(x)
x = Conv2d(C -> C, kernel=1)(x)
x = residual + layer_scale[C,1,1] * x
```

Downsample between stages:

```text
x = Conv2d(C_i -> C_{i+1}, kernel=3, stride=2, pad=1)(x)
x = BatchNorm2d(C_{i+1})(x)
```

Classification head:

```text
x = BatchNorm2d(C_last)(x)
x = x.flatten(2).mean(-1)       # [B,C_last]
logits = (Linear_head(x) + Linear_dist_head(x)) / 2
```

All Conv2d and Linear modules use bias unless PyTorch defaults are overridden; this source does not override them.

## 6. Attention requirements

No standard self-attention, cross-attention, masks, packed sequence metadata, local/sliding-window attention, or KV cache are required.

Required custom attention-like mixer:

```text
Input tokens: [B,N,C], where N = H*W
Q = Linear(C -> C)(tokens)
K = Linear(C -> C)(tokens)
Q = normalize(Q, dim=-1)
K = normalize(K, dim=-1)
w = softmax((Q @ w_g) * C^-0.5, dim=-1)   # source dim, shape [B,N,1]
g = sum(w * Q, dim=1)                     # [B,C]
G = repeat(g[:,None,:], N, axis=1)        # [B,N,C]
out = Linear(C -> C)(Linear(C -> C)(G * K) + Q)
```

Source parity note: because `w` has shape `[B,N,1]`, `softmax(dim=-1)` is over a singleton dimension. For current Transformers source, `w` is all ones. A DinoML rewrite may simplify this to `g = sum(Q, dim=1)` only if source parity tests confirm the same behavior for the target version.

SDPA/FlashAttention compatibility: not applicable. This is O(B*N*C) plus linear projections, not O(B*N*N).

## 7. Position encoding and custom math

SwiftFormer has no absolute position embedding table, RoPE, ALiBi, or relative position bias in the inspected source. Spatial information enters through convolutional stem, depthwise convs, and downsampling.

Custom additive mixer parity snippet:

```python
def swiftformer_additive_attention(x, to_query, to_key, w_g, proj, final):
    # x: [B, N, C]
    q = l2_normalize(to_query(x), axis=-1, eps=1e-12)
    k = l2_normalize(to_key(x), axis=-1, eps=1e-12)
    score = (q @ w_g) * (x.shape[-1] ** -0.5)  # [B, N, 1]
    weight = softmax(score, axis=-1)           # source axis, singleton
    g = sum(weight * q, axis=1)                # [B, C]
    y = proj(g[:, None, :] * k) + q
    return final(y)
```

Precomputable pieces: weights, BatchNorm inference fold constants, and static shape schedules. The global query reduction depends on input activations and cannot be precomputed.

## 8. Preprocessing and input packing

Processor contract:

- Processor class: `ViTImageProcessor` via AutoImageProcessor mapping for `swiftformer`.
- Input: PIL image, NumPy image, or compatible image input accepted by Transformers processors.
- Resize: enabled, `size: 224`; class default is square height/width 224.
- Rescale: sampled configs omit `do_rescale`; class/BaseImageProcessor default is enabled with `1/255`.
- Normalize: enabled, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- Runtime tensor: `pixel_values`, channel-first `[B,3,224,224]` for PyTorch model execution.

No token packing, masks, placeholder scatter, OCR metadata, video frames, or audio features are involved.

For first DinoML integration, keep preprocessing in CPU/data pipeline and require model runtime input to already be normalized NCHW `float32`. GPU preprocessing can be a later optimization.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference fold

Source pattern:

```text
Conv2d(bias=True) -> BatchNorm2d(affine=True, running stats)
```

Replacement:

```text
Conv2d with folded weight and bias
```

Preconditions:

- Inference/eval mode.
- BatchNorm running mean/variance available.
- BatchNorm not shared with another mutable training path.
- Preserve `eps=config.batch_norm_eps`.

Failure cases:

- Training mode or dynamic batch statistics.

Parity test sketch:

- Random input for each stem/downsample/depthwise/pointwise BN site, compare PyTorch eval output before and after fold.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern:

```text
[B,C,H,W] -> Conv2d(C -> D, kernel=1, stride=1, pad=0)
```

Replacement:

```text
NCHW_to_matrix [B*H*W,C] -> GEMM_RCR/GEMM_RRR -> matrix_to_NCHW [B,D,H,W]
```

Preconditions:

- Kernel 1x1, stride 1, dilation 1, padding 0, groups 1.
- Input dense contiguous or layout pass provides equivalent logical access.
- Bias preserved.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels)
```

Layout constraints:

- NCHW lowering may need im2col-like flattening or a channel-last fusion region.
- NHWC optimized path is attractive for 1x1 conv chains but must include BatchNorm/GELU/residual consumers or insert guarded transposes.

### Rewrite: additive attention without materialized repeat

Source pattern:

```text
g = sum(weight * q, dim=1)
G = g.unsqueeze(1).repeat(1, N, 1)
out = proj(G * k) + q
```

Replacement:

```text
Broadcast g over token axis inside fused multiply/projection input path.
```

Preconditions:

- `g` is `[B,C]`, `k` is `[B,N,C]`.
- No user-visible alias of repeated tensor.
- Source softmax-axis behavior is preserved.

Failure cases:

- Debug mode requiring exact intermediate materialization.

### Rewrite: singleton softmax simplification

Source pattern:

```text
softmax(score[B,N,1], dim=-1)
```

Replacement:

```text
ones_like(score)
```

Preconditions:

- Last dimension statically equals 1.
- Source commit uses `dim=-1`.

Failure cases:

- Future source changes softmax to token axis `dim=1`; guard on source/config signature.

### Rewrite: final spatial pool

Source pattern:

```text
BatchNorm2d -> flatten(2) -> mean(-1)
```

Replacement:

```text
BatchNorm2d -> mean over H,W
```

Preconditions:

- Input rank 4 NCHW.
- Mean is over all spatial positions only.

NHWC rewrite:

- Under NHWC, reduce axes `[1,2]`, then produce `[B,C]`.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BatchNorm2d + activation folding/fusion for stem and downsample, because the model is conv-heavy and inference BN is static.
- DepthwiseConv2d + BN + pointwise Conv2d + GELU + pointwise Conv2d + layer-scale residual for ConvEncoder and LocalRepresentation. This is the dominant repeated block.
- Additive attention fused token mixer: two linears, L2 normalize, singleton-softmax simplification, token sum, broadcast multiply, projection, residual, final projection.

Medium priority:

- 1x1 Conv2d chains lowered to GEMM or channel-last pointwise kernels.
- Final BatchNorm2d + global average pool + two heads. This is small but easy to validate.
- NCHW/NHWC guarded layout region covering pointwise-heavy blocks.

Lower priority:

- GPU image preprocessing, if input pipeline becomes a bottleneck.
- DropPath/Dropout training behavior; not needed for inference.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse `SwiftFormerConfig`, sampled config omissions, and two classifier heads.
- Load Conv2d, BatchNorm2d, Linear, layer-scale, and `w_g` tensors.

Stage 2: operator parity for small blocks.

- Implement/evaluate Conv2d, depthwise Conv2d, BatchNorm2d inference, ReLU, GELU, 1x1 Conv2d, Linear, L2 normalize, softmax/reduction, reshape/permute.

Stage 3: one-block parity.

- Match `SwiftFormerConvEncoder`, `SwiftFormerLocalRepresentation`, `SwiftFormerEfficientAdditiveAttention`, and `SwiftFormerEncoderBlock` independently.

Stage 4: full encoder parity.

- Run stem plus 4 stages for fixed `[B,3,224,224]`.
- Support optional `output_hidden_states` only after primary logits path is stable.

Stage 5: classification head parity.

- Final BN, spatial mean, two heads, average logits.

Stage 6: optimized lowering.

- Fold BN, simplify singleton softmax, avoid repeat materialization, add channel-last guarded fusion regions.

Stage 7: dynamic/resolution generalization.

- Only after 224 path is stable, admit other image sizes with shape guards for stem/downsample spatial equations.

Initially stub/defer: training loss, gradients, gradient checkpointing, DropPath randomness, `output_hidden_states`.

## 12. Parity and validation plan

Recommended tests:

- Processor smoke: one PIL image through `ViTImageProcessor`, assert `[1,3,224,224]`, mean/std normalization behavior.
- Random tensor tests for Conv2d + BN fold across regular, depthwise, and 1x1 convs.
- Random tensor tests for L2 normalize and singleton softmax simplification.
- Single `SwiftFormerConvEncoder` parity for each representative channel width.
- Single `SwiftFormerEncoderBlock` parity at spatial sizes 56, 28, 14, and 7.
- Full stage parity for each stage of `swiftformer-xs`.
- End-to-end logits parity for `MBZUAI/swiftformer-xs`, then sweep `s`, `l1`, `l3`.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for block outputs; logits may use `atol=1e-4`.
- fp16/bf16 optimized paths: begin with `rtol=1e-2`, `atol=1e-2`, then tighten per kernel.

No DinoML tests were run for this audit, per instruction.

## 13. Performance probes

- CPU preprocessing throughput: images/sec for resize/rescale/normalize.
- Stem throughput at `[B,3,224,224]`.
- Per-stage throughput and memory bandwidth for spatial sizes 56, 28, 14, 7.
- Depthwise+pointwise block throughput by channel width.
- Additive attention mixer throughput by `(B,N,C)`: `(B,3136,C0)`, `(B,784,C1)`, `(B,196,C2)`, `(B,49,C3)`.
- Impact of avoiding materialized `repeat` in additive attention.
- NCHW baseline versus guarded NHWC/channel-last region.
- Batch-size sweep for `B=1,4,8,16,32`.
- BN-folded versus unfused graph.

## 14. Skip/defer list

- Training loss and labels.
- Gradient checkpointing.
- Dropout and DropPath random training semantics.
- `output_hidden_states` ABI.
- Non-224 image sizes until fixed-size parity is stable.
- Quantization, sparsity, multi-GPU/tensor parallel.
- FlashAttention/SDPA, KV cache, generation controllers: not applicable.

## 15. Final implementation checklist

- [ ] Parse `SwiftFormerConfig` and fill omitted defaults used by current source.
- [ ] Load Conv2d, depthwise Conv2d, BatchNorm2d, Linear, layer-scale, and `w_g` weights.
- [ ] Implement NCHW Conv2d 3x3 stride2 pad1 and depthwise 3x3 pad1.
- [ ] Implement BatchNorm2d inference and BN-fold rewrite.
- [ ] Implement 1x1 Conv2d lowering or pointwise GEMM path.
- [ ] Implement GELU, ReLU, residual add, scalar divide, per-channel layer-scale multiply.
- [ ] Implement NCHW/NHWC permute and `[B,H,W,C] -> [B,H*W,C]` flatten around additive attention.
- [ ] Implement SwiftFormer additive attention with source singleton-softmax semantics.
- [ ] Add rewrite to avoid materialized `repeat`.
- [ ] Add guarded singleton-softmax-to-ones simplification.
- [ ] Implement final BN, spatial mean pool, two classifier heads, logits average.
- [ ] Add single-block parity tests for ConvEncoder, LocalRepresentation, additive attention, and EncoderBlock.
- [ ] Add end-to-end `MBZUAI/swiftformer-xs` logits parity.
- [ ] Add config sweep tests for `xs`, `s`, `l1`, and `l3`.
- [ ] Benchmark conv-heavy stages, additive mixer, BN folding, and NCHW versus NHWC guarded layout.
