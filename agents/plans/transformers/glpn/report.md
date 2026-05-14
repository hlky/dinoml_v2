# GLPN Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: vinvino02/glpn-kitti, vinvino02/glpn-nyu
Config source: Hugging Face config.json and preprocessor_config.json snapshots, plus GLPNConfig source defaults
Source files inspected:
- X:/H/transformers/src/transformers/models/glpn/modeling_glpn.py
- X:/H/transformers/src/transformers/models/glpn/configuration_glpn.py
- X:/H/transformers/src/transformers/models/glpn/image_processing_glpn.py
- X:/H/transformers/src/transformers/models/glpn/image_processing_pil_glpn.py
- comparison skim: dpt/modeling_dpt.py, depth_anything/modeling_depth_anything.py, zoedepth/modeling_zoedepth.py
Any missing files or assumptions: no gated or missing official GLPN files were observed for vinvino02/glpn-kitti or vinvino02/glpn-nyu. This report targets native in-library GLPN depth estimation, not remote-code variants.
```

Source URLs at the inspected commit:

- [modeling_glpn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glpn/modeling_glpn.py)
- [configuration_glpn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glpn/configuration_glpn.py)
- [image_processing_glpn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glpn/image_processing_glpn.py)
- [image_processing_pil_glpn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glpn/image_processing_pil_glpn.py)

Local snapshots are under `agents/plans/transformers/glpn/_sources/`.

## 2. High-level architecture

GLPN is an image-only dense depth model:

```text
image preprocessing -> NCHW pixel_values -> SegFormer/MixTransformer-style hierarchical encoder
  -> GLPN top-down decoder with selective feature fusion -> conv depth head
  -> sigmoid * max_depth -> optional bicubic postprocess resize
```

The primary runtime contract is `GLPNForDepthEstimation`: input `pixel_values` in NCHW `[B,3,H,W]`, output `predicted_depth` in `[B,H_out,W_out]`. There is no text path, no generation, no KV cache, no AutoBackbone routing, and no semantic segmentation head in the inspected GLPN source.

Stage decomposition:

| Stage | Runtime role | Independent validation |
|---|---|---|
| CPU/data pipeline | Convert images to channels-first tensors, resize height/width down to multiples of `size_divisor=32`, rescale by `1/255`. Official GLPN processor JSONs do not set normalization. | Processor-only shape/value parity. |
| Encoder | Four-stage overlapping Conv2d patch embedding plus noncausal efficient self-attention and Mix-FFN blocks. Emits four NCHW feature maps when `output_hidden_states=True`. | Patch embedding, one block, then each stage. |
| Decoder | Reverse-stage upsampling path; first stage projects deep features, later stages selectively fuse current global feature with previous upsampled feature. | Synthetic feature-map parity. |
| Depth head | Two 3x3 convs, ReLU, sigmoid scale by `max_depth`, squeeze channel. | Head-only parity. |
| Postprocess | Optional per-image bicubic resize from model output size to caller `target_size`. | CPU/Torch postprocess parity. |

Differences from adjacent depth families:

- Unlike DPT, GLPN does not use native ViT absolute position embeddings, CLS tokens, DPT reassemble factors, semantic segmentation heads, or AutoBackbone dispatch.
- Unlike Depth Anything, GLPN owns its MixTransformer encoder directly and does not compose DINOv2 through `load_backbone`; its depth head is always `sigmoid * max_depth`.
- Unlike ZoeDepth, GLPN has no metric bin centers, attractors, conditional log-binomial head, domain router, or relative-depth conditioning branch.

## 3. Important config dimensions

Source defaults from `GLPNConfig`:

| Field | Default | Notes |
|---|---:|---|
| `num_channels` | 3 | NCHW image input. |
| `num_encoder_blocks` | 4 | Four hierarchical stages. |
| `depths` | `[2,2,2,2]` | Source default only; official checkpoints use larger depths. |
| `hidden_sizes` | `[32,64,160,256]` | Source default channels per encoder stage. |
| `patch_sizes` | `[7,3,3,3]` | Overlapping Conv2d kernels with padding `patch_size//2`. |
| `strides` | `[4,2,2,2]` | Cumulative output strides are approximately 4, 8, 16, 32. |
| `num_attention_heads` | `[1,2,5,8]` | MHA per stage. |
| `head_dim` | `hidden_size / heads` | Source requires exact divisibility. |
| `sr_ratios` | `[8,4,2,1]` | Spatial reduction Conv2d for K/V when ratio > 1. |
| `mlp_ratios` | `[4,4,4,4]` | Mix-FFN expansion. |
| `hidden_act` | `gelu` | Via `ACT2FN`. |
| `layer_norm_eps` | `1e-6` | Token-last LayerNorm. |
| `decoder_hidden_size` | 64 | All decoder stages project to 64 channels. |
| `max_depth` | 10 | Depth scale after sigmoid. |
| `head_in_index` | -1 | Last decoder feature by default. |
| `dtype` | source default fp32 | Official configs record `torch_dtype: float32`. |
| `cache support` | none | Encoder-only dense prediction. |

Representative checkpoint sweep:

| Checkpoint | Source | Encoder dims | Depths | Heads/head_dim | SR ratios | Decoder/head | Processor |
|---|---|---:|---:|---|---|---|---|
| `GLPNConfig()` default | source defaults | `[32,64,160,256]` | `[2,2,2,2]` | `1/32, 2/32, 5/32, 8/32` | `[8,4,2,1]` | decoder 64, `max_depth=10` | class defaults: resize/rescale, divisor 32 |
| `vinvino02/glpn-kitti` | `config.json` | `[64,128,320,512]` | `[3,8,27,3]` | `1/64, 2/64, 5/64, 8/64` | `[8,4,2,1]` | decoder 64, `max_depth=10` | `do_resize=true`, `do_rescale=true`, `size_divisor=32`, bilinear |
| `vinvino02/glpn-nyu` | `config.json` | `[64,128,320,512]` | `[3,8,27,3]` | `1/64, 2/64, 5/64, 8/64` | `[8,4,2,1]` | decoder 64, `max_depth=10` | same as KITTI |

Official GLPN configs include historical fields `image_size`, `downsampling_rates`, and `classifier_dropout_prob`; the inspected modeling source does not read them for inference.

## 3a. Family variation traps

- Official checkpoints are much deeper/wider than source defaults. Do not size kernels from `GLPNConfig()` alone.
- `hidden_size == num_heads * head_dim`, but head dimension differs by checkpoint width. Official GLPN uses head dim 64 in every stage; source default uses 32.
- Patch embedding is overlapping Conv2d with padding, not non-overlap ViT patchification. The first patch embedding is not safely reducible to a simple `kernel_size == stride` patch GEMM.
- Sequence-reduction attention changes K/V length per stage: Q length is `H_i*W_i`, K/V length is roughly `floor((H_i-r_i)/r_i + 1) * floor((W_i-r_i)/r_i + 1)` for `sr_ratio=r_i` with no padding. This is rectangular attention for stages 1-3.
- GLPN reshapes token sequences back to NCHW after every stage. Axis-sensitive layout passes must guard `flatten(2)`, `transpose(1,2)`, `permute(0,3,1,2)`, `cat(dim=1)`, `squeeze(dim=1)`, and BatchNorm channel axis.
- `output_hidden_states=True` is forced inside `GLPNForDepthEstimation`; first integration must preserve hidden-state list production even if public output hidden states are disabled.
- Processor resize rounds height and width down to the closest multiple of 32. Small images below 32 in a dimension would resize to zero and should be rejected or handled before runtime.
- Official preprocessor configs omit `do_normalize`, `image_mean`, and `image_std`; do not silently apply ImageNet normalization for these checkpoints unless caller overrides it.
- Training-only `SiLogLoss` and DropPath random behavior are not required for inference; DropPath is identity in eval mode.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B,3,H,W]`.
- Conv output flatten `[B,C,H,W] -> [B,C,H*W] -> [B,H*W,C]`.
- Token-to-map reshape `[B,N,C] -> [B,H,W,C] -> [B,C,H,W]`.
- Token reduction path reshape `[B,N,C] -> [B,C,H,W] -> Conv2d -> [B,N_sr,C]`.
- Reverse hidden-state iteration and list indexing.
- `torch.cat(..., dim=1)` for selective fusion.
- Per-channel slices `attn[:,0,:,:]`, `attn[:,1,:,:]`, `unsqueeze(1)`.
- Final `squeeze(dim=1)` from `[B,1,H,W]` to `[B,H,W]`.

Neural network primitives:

- Overlapping Conv2d patch embeddings:
  - stage 1 official: `Conv2d(3 -> 64, kernel=7, stride=4, padding=3, bias=True)`.
  - stage 2: `Conv2d(64 -> 128, kernel=3, stride=2, padding=1)`.
  - stage 3: `Conv2d(128 -> 320, kernel=3, stride=2, padding=1)`.
  - stage 4: `Conv2d(320 -> 512, kernel=3, stride=2, padding=1)`.
- Token LayerNorm over last dim, eps `1e-6`.
- Linear Q/K/V/out projections with bias:
  - stage 1 official: `Linear(64 -> 64)` for Q/K/V/out.
  - stage 2: `Linear(128 -> 128)`.
  - stage 3: `Linear(320 -> 320)`.
  - stage 4: `Linear(512 -> 512)`.
- Spatial-reduction Conv2d for K/V:
  - `Conv2d(C -> C, kernel=stride=8/4/2)` for stages 1/2/3; none for stage 4.
- Mix-FFN:
  - stage 1 official: `Linear(64 -> 256)`, depthwise `Conv2d(256 -> 256, 3x3, groups=256)`, GELU, `Linear(256 -> 64)`.
  - stage 2: `128 -> 512 -> 128`, depthwise groups 512.
  - stage 3: `320 -> 1280 -> 320`, depthwise groups 1280.
  - stage 4: `512 -> 2048 -> 512`, depthwise groups 2048.
- Decoder:
  - first reverse stage: `Conv2d(512 -> 64, 1x1)`, bilinear upsample x2.
  - later reverse stages: `Conv2d(320/128 -> 64, 1x1)` or identity for `64 -> 64`; selective fusion convs `Conv2d(128 -> 64, 3x3) + BatchNorm2d + ReLU`, `Conv2d(64 -> 32, 3x3) + BatchNorm2d + ReLU`, `Conv2d(32 -> 2, 3x3) + Sigmoid`, weighted sum, bilinear upsample x2.
  - final decoder upsample x2 on last stage output.
- Depth head: `Conv2d(64 -> 64, 3x3)`, ReLU, `Conv2d(64 -> 1, 3x3)`, sigmoid, scalar multiply by `max_depth`.

Attention primitives:

- Noncausal encoder MHA.
- Rectangular attention for sequence-reduced K/V in stages with `sr_ratio > 1`.
- MatMul QK^T, scale by `1/sqrt(head_dim)`, softmax over key axis, dropout in source but disabled for official inference, MatMul P*V.
- No attention mask, no causal mask, no cross-attention, no RoPE/ALiBi/relative bias, no KV cache.

Preprocessing/postprocessing ops:

- Image convert to channels-first, optional grouping by shape.
- Resize down to floor multiple of `size_divisor=32`, bilinear.
- Rescale by `1/255`.
- Optional postprocess bicubic resize `[1,1,H,W] -> target_size`, `align_corners=False`.

Distributed/tensor-parallel ops:

- None required by source.

## 5. Layer/block breakdown

For official checkpoints, input after preprocessing is `[B,3,H0,W0]`, where `H0` and `W0` are multiples of 32.

Encoder stage `i`, repeated for four stages:

```text
features_NCHW = previous feature map
tokens, Hi, Wi = Conv2d(Cin -> Ci, kernel=patch_sizes[i], stride=strides[i], padding=patch//2)
tokens = flatten spatial -> [B, Hi*Wi, Ci]
tokens = LayerNorm(Ci)

for each layer in depths[i]:
  y = LayerNorm(tokens)
  q = Linear(Ci -> Ci, bias)(y).view(B, Nq, heads_i, head_dim).transpose(1, 2)
  if sr_ratio[i] > 1:
      y_kv = y.permute(0,2,1).reshape(B, Ci, Hi, Wi)
      y_kv = Conv2d(Ci -> Ci, kernel=stride=sr_ratio[i])(y_kv)
      y_kv = reshape/permute back to [B, Nkv, Ci]
      y_kv = LayerNorm(Ci)
  else:
      y_kv = y
  k,v = Linear(Ci -> Ci, bias)(y_kv)
  attn = softmax((q @ k.T) / sqrt(head_dim), dim=-1)
  context = attn @ v -> [B, Nq, Ci]
  tokens = tokens + DropPath(Linear(Ci -> Ci, bias)(context))

  z = LayerNorm(tokens)
  z = Linear(Ci -> 4*Ci, bias)(z)
  z = depthwise Conv2d(4*Ci -> 4*Ci, 3x3, groups=4*Ci) after token->NCHW reshape
  z = GELU(z)
  z = Linear(4*Ci -> Ci, bias)(z)
  tokens = tokens + DropPath(z)

tokens = final LayerNorm(Ci)
features_NCHW = tokens.reshape(B, Hi, Wi, Ci).permute(0,3,1,2).contiguous()
emit feature map if hidden states requested
```

Official stage shapes for input `[B,3,H0,W0]`:

| Stage | Output channels | Approx spatial size | Layers | SR ratio | K/V spatial size |
|---|---:|---|---:|---:|---|
| 1 | 64 | `ceil(H0/4) x ceil(W0/4)` by padded Conv2d formula | 3 | 8 | strided conv on stage-1 map |
| 2 | 128 | roughly previous / 2 | 8 | 4 | strided conv on stage-2 map |
| 3 | 320 | roughly previous / 2 | 27 | 2 | strided conv on stage-3 map |
| 4 | 512 | roughly previous / 2 | 3 | 1 | full stage-4 tokens |

GLPN decoder:

```text
hidden_states = [s1, s2, s3, s4]  # all NCHW
stage0 = Conv1x1(512 -> 64)(s4)
stage0 = bilinear_upsample_x2(stage0)

stage1 = Conv1x1(320 -> 64)(s3)
stage1 = selective_fusion(stage1, stage0)
stage1 = bilinear_upsample_x2(stage1)

stage2 = Conv1x1(128 -> 64)(s2)
stage2 = selective_fusion(stage2, stage1)
stage2 = bilinear_upsample_x2(stage2)

stage3 = Identity(64)(s1)
stage3 = selective_fusion(stage3, stage2)
stage3 = bilinear_upsample_x2(stage3)
stage3 = final_bilinear_upsample_x2(stage3)
```

Selective feature fusion:

```text
f = cat([local_features, global_features], dim=1)      # [B,128,H,W]
f = Conv3x3(128 -> 64) + BatchNorm + ReLU
f = Conv3x3(64 -> 32) + BatchNorm + ReLU
attn = sigmoid(Conv3x3(32 -> 2)(f))                    # [B,2,H,W]
out = local_features * attn[:,0,:,:].unsqueeze(1) + global_features * attn[:,1,:,:].unsqueeze(1)
```

Depth head:

```text
x = decoder_outputs[head_in_index]     # default last
x = Conv3x3(64 -> 64)(x)
x = ReLU(x)
x = Conv3x3(64 -> 1)(x)
predicted_depth = sigmoid(x) * max_depth
predicted_depth = squeeze channel dim -> [B,H,W]
```

## 6. Attention requirements

GLPN attention is encoder-only, noncausal self-attention. There is no generation cache.

| Field | Requirement |
|---|---|
| Causal | No. |
| Self/cross | Self-attention only. |
| MHA/MQA/GQA | Standard MHA; K/V heads equal Q heads. |
| Official heads/head dim | stage 1 `1 x 64`, stage 2 `2 x 64`, stage 3 `5 x 64`, stage 4 `8 x 64`. |
| Query length | `Nq = Hi * Wi`. |
| Key/value length | `Nkv=Nq` for `sr_ratio=1`; otherwise K/V come from Conv2d-reduced spatial map. |
| Mask | None in source. |
| Position bias | None. Position information comes from overlapping patch Conv2d and depthwise Mix-FFN Conv2d. |
| Packed/varlen | None. |
| Sliding/local | None; sequence reduction is convolutional downsampling before global attention, not local attention. |
| KV cache | Not implemented. |
| Flash/SDPA | Source uses eager matmul/softmax only. A DinoML fused attention path must preserve rectangular K/V length and no-mask semantics. |

Eager fallback risk: stage 3 has 27 layers at hidden 320. For large images, `Nq` at stride 16 and full attention to reduced K/V can still dominate; fused rectangular MHA or an efficient token GEMM path is useful.

## 7. Position encoding and custom math

GLPN has no learned absolute position embedding, RoPE, ALiBi, or relative bias. Spatial information is injected through overlapping Conv2d patch embeddings, sequence-reduction Conv2d for K/V, and depthwise 3x3 Conv2d inside the Mix-FFN.

Core source-specific math to reproduce:

```python
def glpn_sr_attention(x, h, w, q_proj, k_proj, v_proj, sr_conv=None, sr_norm=None, heads=1):
    # x: [B, H*W, C]
    q = split_heads(q_proj(x), heads)          # [B, heads, Nq, D]
    kv = x
    if sr_conv is not None:
        b, n, c = x.shape
        kv = x.permute(0, 2, 1).reshape(b, c, h, w)
        kv = sr_conv(kv)
        kv = kv.reshape(b, c, -1).permute(0, 2, 1)
        kv = sr_norm(kv)
    k = split_heads(k_proj(kv), heads)
    v = split_heads(v_proj(kv), heads)
    p = softmax((q @ k.transpose(-1, -2)) / sqrt(q.shape[-1]), dim=-1)
    return merge_heads(p @ v)                  # [B, Nq, C]
```

For fixed preprocessed image buckets, stage spatial sizes and K/V reduced lengths can be precomputed. Attention probabilities depend on runtime image features.

## 8. Preprocessing and input packing

Official `vinvino02/glpn-kitti` and `vinvino02/glpn-nyu` preprocessor configs:

| Field | Value |
|---|---|
| `do_resize` | `true` |
| `size_divisor` | `32` |
| `resample` | `2` / bilinear |
| `do_rescale` | `true` |
| `rescale_factor` | source class default `1/255` |
| `do_normalize` | omitted in official JSON and not a GLPN class attribute |
| Output | `pixel_values` as NCHW float tensor batch |

Resize behavior:

```python
new_h = height // size_divisor * size_divisor
new_w = width // size_divisor * size_divisor
pixel_values = resize(image, (new_h, new_w), bilinear)
```

There is no image patch packing metadata, no grid metadata, no attention mask, and no special tokens. The processor groups images by shape for efficient preprocessing but returns only `pixel_values`.

Postprocessing:

- Model output is raw `predicted_depth` in model/preprocessed resolution.
- `post_process_depth_estimation(outputs, target_sizes)` expects one `(height,width)` per batch item when provided.
- Each depth map is temporarily unsqueezed to `[1,1,H,W]`, bicubic-resized to target size with `align_corners=False`, then squeezed back to `[target_h,target_w]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: spatial-reduction Conv2d to window GEMM

Source pattern:

```text
[B,N,C] -> reshape NCHW [B,C,H,W] -> Conv2d(C -> C, kernel=stride=sr, padding=0)
  -> flatten/transpose -> LayerNorm -> K/V Linear
```

Replacement:

```text
WindowFlatten(non-overlap sr x sr) -> MatMul(weight_flat.T) -> BiasAdd
  -> token reshape -> LayerNorm -> K/V Linear
```

Preconditions:

- Applies only to `GLPNEfficientSelfAttention.sr`, not patch embeddings.
- `kernel_size == stride == sr_ratio`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Token sequence must correspond exactly to `[B,C,H,W]` with `N == H*W`.
- Use PyTorch Conv2d floor output semantics when `H/W` are not divisible by `sr_ratio`; do not pad unless the source does.

Shape equations:

- Input token map `[B,H*W,C]`.
- Output `Hsr = floor((H - sr) / sr) + 1`, `Wsr = floor((W - sr) / sr) + 1`.
- K/V token length is `Hsr * Wsr`.

Weight transform:

```python
w = sr.weight.reshape(C, C * sr * sr)
b = sr.bias
```

Layout constraints:

- NCHW flatten order must be preserved. If a channel-last pass is active, the window flatten order and weight layout must be rewritten together.

Failure cases:

- `sr_ratio == 1` has no Conv2d to lower.
- Dynamic shapes without guards can change floor output lengths.

Parity test sketch:

- Feed random `[B,H*W,C]` tokens for several `H/W`, including non-divisible by `sr`; compare Conv2d path before LayerNorm to window-GEMM result.

### Rewrite: 1x1 Conv2d decoder projection to GEMM

Source pattern:

```text
NCHW Conv2d(Cin -> 64, kernel=1, stride=1, padding=0)
```

Replacement:

```text
Flatten spatial to [B*H*W,Cin] -> MatMul(weight[:, :, 0, 0].T) -> BiasAdd -> reshape
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- Consumer layout remains NCHW or is inside a guarded NHWC decoder region.

Failure cases:

- Applying to selective-fusion 3x3 convs would be unsafe without im2col or native Conv2d.

### Rewrite: eval BatchNorm fold into preceding Conv2d

Source pattern:

```text
Conv2d -> BatchNorm2d -> ReLU
```

Replacement:

```text
Conv2d(folded_weight, folded_bias) -> ReLU
```

Preconditions:

- Inference/eval mode only.
- BatchNorm running mean/variance/weight/bias are available.
- Applies to selective feature fusion conv1/conv2 only.

Weight transform:

```python
scale = bn.weight / sqrt(bn.running_var + bn.eps)
w_fold = conv.weight * scale[:, None, None, None]
b_fold = (conv.bias - bn.running_mean) * scale + bn.bias
```

### Rewrite: guarded GLPN decoder channel-last region

Source pattern:

```text
NCHW Conv1x1/Conv3x3/BatchNorm/ReLU/Sigmoid/elementwise/upsample in decoder and head
```

Replacement:

```text
NCHW boundary -> NHWC/channel-last conv/interpolate/fusion kernels -> restore source output contract
```

Preconditions:

- Region boundaries are local and fully controlled: GLPN decoder plus head, not public processor input or postprocess.
- Axis rewrites are explicit: `cat(dim=1) -> cat(dim=-1)`, `attn[:,0,:,:] -> attn[...,0]`, `unsqueeze(1) -> unsqueeze(-1)`, `squeeze(dim=1) -> squeeze(dim=-1)`.
- BatchNorm channel axis is rewritten or folded into Conv2d first.
- Interpolate backend supports NHWC semantics or inserts explicit transposes.

Failure cases:

- Returning hidden states/attentions with source NCHW contracts.
- Postprocess expects `[B,H,W]` after source-style squeeze.
- Broadly translating encoder token/feature reshape code without updating `flatten`, `permute`, and attention paths.

Use a conceptual `no_layout_translation()` guard around processor semantics, encoder token reshapes, attention, and public hidden-state outputs until a full axis-aware pass proves the region.

## 10. Kernel fusion candidates

Highest priority:

- Overlapping Conv2d patch embeddings and spatial-reduction Conv2d. These dominate layout churn between image maps and token sequences and are central to GLPN's SegFormer-style encoder.
- Rectangular MHA for sequence-reduction attention. Stages 1-3 have `Nq != Nkv`; a fused no-mask attention path should preserve source scaling and softmax axis.
- LayerNorm + Linear around Q/K/V and Mix-FFN. GLPN uses token-last LayerNorm heavily across 41 layers in official checkpoints.
- Decoder Conv/BatchNorm/ReLU folding and channel-last local kernels. The decoder/head is dense image-map work and fits DinoML's NHWC preference if guarded.

Medium priority:

- Mix-FFN `Linear -> depthwise 3x3 Conv2d -> GELU -> Linear` fusion or scheduling. The depthwise Conv2d forces token/map conversion inside every block.
- Selective feature fusion kernel: concatenation, two conv-BN-ReLU blocks, sigmoid attention, channel slice, weighted sum.
- Bilinear upsample fusion with adjacent decoder convs where output sizes are bucketed.

Lower priority:

- Dropout/DropPath elimination in inference graphs.
- Optional postprocess bicubic resize on GPU. CPU postprocess is acceptable for model-only parity.
- Attention probability output support for debugging; not needed for first depth runtime.

## 11. Runtime staging plan

Stage 1: Parse `GLPNConfig` and processor JSON, load source-default and official checkpoint weights, and validate patch embedding plus one GLPN layer on fixed NCHW tensors.

Stage 2: Implement the full encoder with token/map reshapes, sequence-reduction attention, Mix-FFN depthwise Conv2d, and hidden-state list emission.

Stage 3: Implement decoder stages and selective feature fusion using direct NCHW Conv2d/BatchNorm/upsample.

Stage 4: Implement depth head and raw `predicted_depth` parity for `vinvino02/glpn-kitti` and `vinvino02/glpn-nyu`.

Stage 5: Add processor and postprocess parity: floor-to-divisor resize, rescale, and optional bicubic target resize.

Stage 6: Add graph rewrites and fusions: SR Conv2d lowering, BatchNorm folding, decoder channel-last local region, and rectangular attention backend.

Stage 7: Add production benchmarking and bucketed image-size scheduling.

Initial stubs: training loss, attention outputs, hidden-state public outputs beyond decoder needs, and GPU postprocess resize can be deferred.

## 12. Parity and validation plan

- Processor parity: random RGB images with dimensions divisible and not divisible by 32; verify output size floors to multiples of 32 and values are rescaled by `1/255` without unexpected normalization.
- Overlap patch embedding parity: per stage Conv2d -> flatten -> LayerNorm for fixed and bucketed image sizes.
- Sequence-reduction attention parity: compare one GLPN layer with `sr_ratio=8/4/2/1`, including rectangular K/V lengths.
- Mix-FFN parity: token-to-map depthwise Conv2d path with official hidden sizes and MLP ratios.
- Encoder parity: after each stage, compare NCHW feature maps in hidden-state list.
- Decoder parity: feed saved/synthetic NCHW feature maps `[64,128,320,512]` channels and compare selective fusion/upsampling outputs.
- Head parity: compare raw `predicted_depth` before postprocess, including sigmoid scale and `squeeze(dim=1)`.
- End-to-end parity: `vinvino02/glpn-kitti` and `vinvino02/glpn-nyu` raw depth at preprocessed resolution; then postprocessed resize to original image size.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-4`; fp16/bf16 direct Conv/MatMul `atol=2e-2, rtol=2e-2`, with looser postprocess resize checks if backend interpolation differs.

## 13. Performance probes

- Processor throughput: image resize-down-to-divisor and rescale images/sec, split by CPU/data pipeline versus GPU if implemented.
- Encoder stage throughput: stage-wise timing for `[64,128,320,512]` official dims and depths `[3,8,27,3]`.
- Attention backend comparison: eager rectangular attention versus fused DinoML backend, swept over image buckets and `sr_ratio`.
- Token/map layout churn: cost of `flatten/transpose/permute/contiguous` around patch embeddings, SR attention, and Mix-FFN depthwise conv.
- Decoder-only throughput: Conv/BN/ReLU/Sigmoid/upsample selective fusion with NCHW versus guarded channel-last kernels.
- Batch-size sweep and image-resolution sweep for multiples of 32.
- Memory probes: attention probability materialization by stage; hidden-state list memory required by decoder.
- Postprocess throughput: bicubic target resize per image.

## 14. Skip/defer list

- Training and `SiLogLoss`.
- DropPath stochastic behavior outside eval inference.
- Multi-GPU tensor parallelism.
- Quantization-specific loading or provider kernels; no source-coupled packed format was found.
- Returning attention probabilities unless a caller explicitly requests them.
- GPU postprocess resize for first model-only integration.
- Broad NCHW-to-NHWC graph translation across encoder token reshapes; keep it as guarded local fusion work.
- DPT, Depth Anything, and ZoeDepth-specific backbones/heads; they are separate families despite sharing depth-estimation task space.

## 15. Final implementation checklist

- [ ] Parse `GLPNConfig`, including official omitted/ignored historical fields.
- [ ] Parse `GLPNImageProcessor` settings: floor resize to `size_divisor`, bilinear resample, rescale, no implicit normalization for official configs.
- [ ] Load GLPN weights for `vinvino02/glpn-kitti` / `vinvino02/glpn-nyu`.
- [ ] Implement overlapping Conv2d patch embeddings with flatten/LayerNorm.
- [ ] Implement GLPN sequence-reduction MHA with rectangular K/V lengths.
- [ ] Implement Mix-FFN token/map depthwise Conv2d path.
- [ ] Emit encoder hidden-state NCHW feature maps for all four stages.
- [ ] Implement GLPN decoder stages, selective feature fusion, BatchNorm inference, and bilinear upsampling.
- [ ] Implement depth head sigmoid scale by `max_depth` and channel squeeze.
- [ ] Implement optional postprocess bicubic resize to target sizes.
- [ ] Add guarded SR Conv2d-to-GEMM rewrite.
- [ ] Add BatchNorm folding for decoder fusion convs.
- [ ] Add guarded channel-last decoder/head optimization with explicit axis rewrites.
- [ ] Add processor, one-block, stage-wise, decoder, raw-depth, and postprocess parity tests.
- [ ] Benchmark encoder stage timing, rectangular attention, layout churn, decoder channel-last path, and postprocess resize.
